"""Unit 5: dataset acquisition is guarded, and never silent.

Two properties, both of which a run has previously violated:

1. A run that requested ``--retrieval-setting none`` cannot benefit from the
   benchmark data, so it must not pay to acquire it. The guard reads the
   **requested** setting. It deliberately does *not* read the effective mode,
   which ``RetrieverAgent`` derives from whether ``ref.json`` exists: that only
   becomes true after acquisition, so gating on it is circular and would
   permanently prevent the bootstrap. One test below pins exactly that.
2. Acquisition reports how long it took. Run ``run_20260804_084953`` spent ~18
   minutes inside this call and then *succeeded*; nothing in the output said so,
   which is why it read as a hang and went undiagnosed.

Offline only. ``huggingface_hub`` is replaced with a recording fake; no test
here opens a socket.
"""

import contextlib
import zipfile
import json
import io
import sys
import tempfile
import types
import unittest
import unittest.mock
from argparse import Namespace
from pathlib import Path

from skill import run as skill_run

from tests.test_skill_run_manifest import args_namespace, full_result
from tests.test_skill_run_entry_point import DOWNLOAD_BANNER, stubbed_pipeline


class FakeHub:
    """Stands in for ``huggingface_hub``. Records calls; never touches a socket.

    Builds a real archive with the layout the live dataset actually has: an
    inner ``PaperBananaBench/`` root containing ``<task>/ref.json`` and
    ``<task>/images/``. The extraction path is the part most likely to break, so
    the fake exercises it rather than stubbing it out.
    """

    def __init__(self, *, side_effect=None, delay: float = 0.0, archive_root="PaperBananaBench",
                 tasks=("diagram", "plot")) -> None:
        self.calls: list[dict] = []
        self.side_effect = side_effect
        self.delay = delay
        self.archive_root = archive_root
        self.tasks = tasks

    def hf_hub_download(self, repo_id, filename, **kwargs):
        self.calls.append({"repo_id": repo_id, "filename": filename, **kwargs})
        if self.delay:
            import time

            time.sleep(self.delay)
        if self.side_effect is not None:
            raise self.side_effect

        local_dir = Path(kwargs.get("local_dir") or tempfile.mkdtemp())
        local_dir.mkdir(parents=True, exist_ok=True)
        archive = local_dir / filename
        with zipfile.ZipFile(archive, "w") as zf:
            for task in self.tasks:
                base = f"{self.archive_root}/{task}"
                zf.writestr(
                    f"{base}/ref.json",
                    json.dumps([{"id": "x", "path_to_gt_image": "images/x.jpg"}]),
                )
                zf.writestr(f"{base}/images/x.jpg", b"not-a-real-jpeg")
        return str(archive)


@contextlib.contextmanager
def fake_hub(**kwargs):
    hub = FakeHub(**kwargs)
    module = types.ModuleType("huggingface_hub")
    module.hf_hub_download = hub.hf_hub_download
    saved = sys.modules.get("huggingface_hub")
    sys.modules["huggingface_hub"] = module
    try:
        yield hub
    finally:
        if saved is None:
            sys.modules.pop("huggingface_hub", None)
        else:
            sys.modules["huggingface_hub"] = saved


@contextlib.contextmanager
def project_root(tmp: Path, *, dataset_present: bool, task: str = "diagram"):
    """Point ``ensure_dataset`` at a scratch tree instead of the real ``data/``.

    The repository's ``data/`` is a symlink into the checkout the live service
    runs from. No test may write through it.
    """
    if dataset_present:
        task_dir = tmp / "data" / "PaperBananaBench" / task
        (task_dir / "images").mkdir(parents=True)
        (task_dir / "ref.json").write_text("[]", encoding="utf-8")
    with unittest.mock.patch.object(skill_run, "PROJECT_ROOT", tmp):
        yield


class ScratchRootTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def capture(self, fn, *args, **kwargs):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            result = fn(*args, **kwargs)
        return result, out.getvalue(), err.getvalue()


class AcquisitionGuardTests(ScratchRootTestCase):
    def test_requested_none_does_not_attempt_acquisition(self) -> None:
        """The headline: --retrieval-setting none must reclaim the whole cost."""
        args = args_namespace(retrieval_setting="none")

        with unittest.mock.patch.object(skill_run, "ensure_dataset") as ensure:
            attempted, out, err = self.capture(skill_run.acquire_dataset, args)

        ensure.assert_not_called()
        self.assertFalse(attempted)
        self.assertIn("skipping dataset acquisition", err)
        self.assertEqual(out, "", "the skip note must not reach stdout")

    def test_every_other_requested_setting_acquires(self) -> None:
        for setting in ("auto", "manual", "random"):
            with self.subTest(setting=setting):
                args = args_namespace(retrieval_setting=setting, task="diagram")
                with unittest.mock.patch.object(skill_run, "ensure_dataset") as ensure:
                    attempted, _, _ = self.capture(skill_run.acquire_dataset, args)
                ensure.assert_called_once_with("diagram")
                self.assertTrue(attempted)

    def test_guard_reads_the_requested_flag_not_the_effective_mode(self) -> None:
        """Anti-circularity.

        With no ``ref.json`` on disk the *effective* mode is ``none`` -- that is
        what RetrieverAgent would downgrade to. A guard keyed on the effective
        mode would therefore skip acquisition here, and ``ref.json`` could never
        appear, so the effective mode could never be anything but ``none``. The
        requested setting is ``auto``, so acquisition must still happen.
        """
        args = args_namespace(retrieval_setting="auto")

        with project_root(self.tmp, dataset_present=False):
            with fake_hub() as hub:
                attempted, _, _ = self.capture(skill_run.acquire_dataset, args)

        self.assertTrue(attempted)
        self.assertEqual(len(hub.calls), 1, "bootstrap acquisition was skipped")

    def test_missing_retrieval_setting_attribute_still_acquires(self) -> None:
        """A caller without the flag is not a caller who opted out."""
        args = Namespace(task="diagram")

        with unittest.mock.patch.object(skill_run, "ensure_dataset") as ensure:
            attempted, _, _ = self.capture(skill_run.acquire_dataset, args)

        ensure.assert_called_once_with("diagram")
        self.assertTrue(attempted)


class AcquisitionShortCircuitTests(ScratchRootTestCase):
    def test_dataset_already_present_makes_no_network_call(self) -> None:
        with project_root(self.tmp, dataset_present=True):
            with fake_hub() as hub:
                _, _, err = self.capture(skill_run.ensure_dataset, "diagram")

        self.assertEqual(hub.calls, [])
        self.assertNotIn("Downloading", err)


class AcquisitionTimingTests(ScratchRootTestCase):
    def test_a_successful_acquisition_reports_its_elapsed_time(self) -> None:
        with project_root(self.tmp, dataset_present=False):
            with fake_hub(delay=0.25):
                _, out, err = self.capture(skill_run.ensure_dataset, "diagram")

        self.assertIn("[dataset] acquisition took ", err)
        self.assertNotIn("[dataset]", out, "timing is status, not output")
        reported = float(err.split("[dataset] acquisition took ")[1].split("s")[0])
        self.assertGreaterEqual(
            reported, 0.2, "the reported figure must track real elapsed time"
        )

    def test_a_failed_acquisition_reports_elapsed_time_and_still_raises(self) -> None:
        """The 18-minute call succeeded; a failing one must be no quieter."""
        boom = RuntimeError("connection reset by peer")

        with project_root(self.tmp, dataset_present=False):
            with fake_hub(side_effect=boom, delay=0.25):
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    with self.assertRaises(RuntimeError):
                        skill_run.ensure_dataset("diagram")

        self.assertIn("[dataset] acquisition took ", err.getvalue())
        reported = float(
            err.getvalue().split("[dataset] acquisition took ")[1].split("s")[0]
        )
        self.assertGreaterEqual(reported, 0.2)


class Ipv4WorkaroundTests(ScratchRootTestCase):
    """The IPv6 route to huggingface.co blackholes on this host.

    Measured: one un-timed connect to a single AAAA address costs 136.3s, and
    there are 8 of them. Forcing A records took the same cold-cache metadata call
    from 24.5s to 0.4s.
    """

    @contextlib.contextmanager
    def recording_resolver(self):
        import socket

        seen: list[int] = []

        def fake(host, port, family=0, type=0, proto=0, flags=0):
            seen.append(family)
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))]

        with unittest.mock.patch.object(socket, "getaddrinfo", fake):
            yield seen

    def test_resolution_is_pinned_to_ipv4_inside_the_block(self) -> None:
        import socket

        with self.recording_resolver() as seen:
            with skill_run.ipv4_only_dns() as forced:
                socket.getaddrinfo("huggingface.co", 443)

        self.assertTrue(forced)
        self.assertEqual(seen, [socket.AF_INET])

    def test_the_resolver_is_restored_even_when_the_block_raises(self) -> None:
        import socket

        with self.recording_resolver():
            before = socket.getaddrinfo
            with self.assertRaises(RuntimeError):
                with skill_run.ipv4_only_dns():
                    self.assertIsNot(socket.getaddrinfo, before)
                    raise RuntimeError("download exploded")
            self.assertIs(socket.getaddrinfo, before)

    def test_the_escape_hatch_leaves_resolution_untouched(self) -> None:
        import socket

        with self.recording_resolver():
            before = socket.getaddrinfo
            with unittest.mock.patch.dict(
                skill_run.os.environ, {"PAPERBANANA_ACQUIRE_IPV4": "0"}
            ):
                with skill_run.ipv4_only_dns() as forced:
                    self.assertIs(socket.getaddrinfo, before)
        self.assertFalse(forced)

    def test_acquisition_runs_inside_the_ipv4_block(self) -> None:
        """The workaround is wired to the call it exists for, not merely defined."""
        import socket

        observed: list[object] = []

        recorder = FakeHub()

        def recording_download(repo_id, filename, **kwargs):
            observed.append(socket.getaddrinfo)
            return recorder.hf_hub_download(repo_id, filename, **kwargs)

        module = types.ModuleType("huggingface_hub")
        module.hf_hub_download = recording_download
        with unittest.mock.patch.dict(sys.modules, {"huggingface_hub": module}):
            with project_root(self.tmp, dataset_present=False):
                outer = socket.getaddrinfo
                _, _, err = self.capture(skill_run.ensure_dataset, "diagram")

        self.assertEqual(len(observed), 1)
        self.assertIsNot(
            observed[0], outer, "the archive fetch ran with the default resolver"
        )
        self.assertIs(socket.getaddrinfo, outer, "resolver was not restored")
        self.assertIn("resolving IPv4-only", err)


class GuardAtTheProcessLevelTests(unittest.IsolatedAsyncioTestCase):
    """The guard is a property of a run, not of a helper called in isolation."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    async def execute(self, **overrides):
        args = args_namespace(
            num_candidates=1,
            output=str(self.tmp / "figure.png"),
            **overrides,
        )

        async def batch(data_list):
            for data in data_list:
                yield full_result(data["filename"])

        out, err = io.StringIO(), io.StringIO()
        with stubbed_pipeline(batch):
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                await skill_run.run(args)
        return out.getvalue(), err.getvalue()

    async def test_a_none_run_never_reaches_the_download(self) -> None:
        out, err = await self.execute(retrieval_setting="none")

        self.assertNotIn(DOWNLOAD_BANNER, err)
        self.assertIn("skipping dataset acquisition", err)
        for line in out.splitlines():
            if line.strip():
                self.assertTrue(Path(line).is_file(), f"stdout leaked: {line!r}")

    async def test_a_default_run_still_reaches_the_download(self) -> None:
        """The guard must be narrow: only ``none`` is exempted."""
        out, err = await self.execute(retrieval_setting="auto")

        self.assertIn(DOWNLOAD_BANNER, err)
        self.assertNotIn(DOWNLOAD_BANNER, out)


if __name__ == "__main__":
    unittest.main()


class ArchiveExtractionTests(ScratchRootTestCase):
    """Slice D: the archive is the only thing HuggingFace actually serves.

    ``allow_patterns=["<task>/*"]`` matched zero files because the dataset ships
    one ``PaperBananaBench.zip`` whose internal root is ``PaperBananaBench/``.
    Extracting it into ``data/`` yields the layout the Retriever and Planner
    already expect, so these tests pin the layout, not just the download.
    """

    def test_extraction_produces_the_layout_the_retriever_reads(self) -> None:
        with project_root(self.tmp, dataset_present=False):
            with fake_hub():
                self.capture(skill_run.ensure_dataset, "diagram")

            task_dir = self.tmp / "data" / "PaperBananaBench" / "diagram"
            self.assertTrue((task_dir / "ref.json").exists())
            self.assertTrue((task_dir / "images").is_dir())
            self.assertTrue(skill_run.dataset_present("diagram"))

    def test_the_image_paths_in_ref_json_resolve_where_the_planner_looks(self) -> None:
        """PlannerAgent opens <task>/<path_to_gt_image> with no guard."""
        with project_root(self.tmp, dataset_present=False):
            with fake_hub():
                self.capture(skill_run.ensure_dataset, "diagram")

            task_dir = self.tmp / "data" / "PaperBananaBench" / "diagram"
            entries = json.loads((task_dir / "ref.json").read_text(encoding="utf-8"))
            for entry in entries:
                self.assertTrue((task_dir / entry["path_to_gt_image"]).exists())

    def test_a_second_run_makes_no_network_call(self) -> None:
        with project_root(self.tmp, dataset_present=False):
            with fake_hub() as first:
                self.capture(skill_run.ensure_dataset, "diagram")
            self.assertEqual(len(first.calls), 1)

            with fake_hub() as second:
                self.capture(skill_run.ensure_dataset, "diagram")
            self.assertEqual(second.calls, [], "already-present short-circuit failed")

    def test_an_unexpected_archive_layout_raises_instead_of_half_landing(self) -> None:
        """A changed archive root must not leave a tree that looks acquired."""
        with project_root(self.tmp, dataset_present=False):
            with fake_hub(archive_root="SomethingElse"):
                with self.assertRaises(RuntimeError):
                    self.capture(skill_run.ensure_dataset, "diagram")

            self.assertFalse(
                skill_run.dataset_present("diagram"),
                "a failed acquisition satisfied the existence check",
            )

    def test_a_failed_download_leaves_nothing_that_satisfies_the_check(self) -> None:
        with project_root(self.tmp, dataset_present=False):
            with fake_hub(side_effect=RuntimeError("network died")):
                with self.assertRaises(RuntimeError):
                    self.capture(skill_run.ensure_dataset, "diagram")

            self.assertFalse(skill_run.dataset_present("diagram"))


class XetOverrideTests(ScratchRootTestCase):
    """Xet moves the transfer into a Rust stack the IPv4 patch cannot reach."""

    def test_xet_is_disabled_during_the_fetch_and_restored_after(self) -> None:
        seen = []

        recorder = FakeHub()

        def observing_download(repo_id, filename, **kwargs):
            seen.append(skill_run.os.environ.get("HF_HUB_DISABLE_XET"))
            return recorder.hf_hub_download(repo_id, filename, **kwargs)

        module = types.ModuleType("huggingface_hub")
        module.hf_hub_download = observing_download

        with unittest.mock.patch.dict(skill_run.os.environ, {}, clear=False):
            skill_run.os.environ.pop("HF_HUB_DISABLE_XET", None)
            with unittest.mock.patch.dict(sys.modules, {"huggingface_hub": module}):
                with project_root(self.tmp, dataset_present=False):
                    self.capture(skill_run.ensure_dataset, "diagram")

            self.assertEqual(seen, ["1"], "Xet was not disabled for the transfer")
            self.assertIsNone(
                skill_run.os.environ.get("HF_HUB_DISABLE_XET"),
                "the override leaked past the fetch into the provider calls",
            )


class DefaultRunCostTests(ScratchRootTestCase):
    """Acquisition working must not silently make every run more expensive."""

    def test_a_default_invocation_does_not_acquire_or_retrieve(self) -> None:
        args = skill_run.build_parser().parse_args(
            ["--content", "m", "--caption", "c"]
        )

        self.assertEqual(args.retrieval_setting, "none")
        with unittest.mock.patch.object(skill_run, "ensure_dataset") as ensure:
            attempted, _, _ = self.capture(skill_run.acquire_dataset, args)

        ensure.assert_not_called()
        self.assertFalse(attempted)


class IncompleteTaskDirectoryTests(ScratchRootTestCase):
    """A half-populated task directory must be repaired, not stranded.

    Skipping the move on mere existence left an incomplete directory in place
    forever: ``dataset_present`` stayed False, so every later run re-downloaded
    and re-extracted the 266MB archive and then discarded the good copy it had
    just produced. A loop that never converged.
    """

    def partial(self, *, ref: bool, images: bool) -> Path:
        task_dir = self.tmp / "data" / "PaperBananaBench" / "diagram"
        task_dir.mkdir(parents=True)
        if ref:
            (task_dir / "ref.json").write_text("[]", encoding="utf-8")
        if images:
            (task_dir / "images").mkdir()
        return task_dir

    def test_a_task_dir_missing_images_is_repaired(self) -> None:
        self.partial(ref=True, images=False)

        with project_root(self.tmp, dataset_present=False):
            with fake_hub():
                self.capture(skill_run.ensure_dataset, "diagram")

            self.assertTrue(
                skill_run.dataset_present("diagram"),
                "acquisition ran but left the dataset unusable",
            )

    def test_an_empty_task_dir_is_repaired(self) -> None:
        self.partial(ref=False, images=False)

        with project_root(self.tmp, dataset_present=False):
            with fake_hub():
                self.capture(skill_run.ensure_dataset, "diagram")

            self.assertTrue(skill_run.dataset_present("diagram"))

    def test_repairing_stops_the_endless_re_download(self) -> None:
        """The symptom that makes this expensive rather than merely wrong."""
        self.partial(ref=True, images=False)

        with project_root(self.tmp, dataset_present=False):
            with fake_hub() as first:
                self.capture(skill_run.ensure_dataset, "diagram")
            self.assertEqual(len(first.calls), 1)

            with fake_hub() as again:
                self.capture(skill_run.ensure_dataset, "diagram")
            self.assertEqual(
                again.calls, [], "a repaired dataset was downloaded a second time"
            )

    def test_an_already_complete_task_dir_is_left_alone(self) -> None:
        """Repair must not clobber good data that is already there."""
        task_dir = self.partial(ref=False, images=True)
        (task_dir / "ref.json").write_text('["LOCAL"]', encoding="utf-8")

        with project_root(self.tmp, dataset_present=False):
            with fake_hub() as hub:
                self.capture(skill_run.ensure_dataset, "diagram")

            self.assertEqual(hub.calls, [], "a complete dataset triggered a download")
            self.assertIn("LOCAL", (task_dir / "ref.json").read_text(encoding="utf-8"))

    def test_a_sibling_task_is_not_disturbed_while_repairing_another(self) -> None:
        base = self.tmp / "data" / "PaperBananaBench"
        (base / "plot" / "images").mkdir(parents=True)
        (base / "plot" / "ref.json").write_text('["PLOT-LOCAL"]', encoding="utf-8")
        self.partial(ref=True, images=False)

        with project_root(self.tmp, dataset_present=False):
            with fake_hub():
                self.capture(skill_run.ensure_dataset, "diagram")

            self.assertTrue(skill_run.dataset_present("diagram"))
            self.assertIn(
                "PLOT-LOCAL", (base / "plot" / "ref.json").read_text(encoding="utf-8")
            )
