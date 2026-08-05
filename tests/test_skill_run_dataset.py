"""Dataset acquisition for the headless skill CLI.

``ensure_dataset`` requested ``allow_patterns=["<task>/*"]`` from a repository
that publishes a single ``PaperBananaBench.zip``, so the pattern matched no
files, nothing was downloaded, and the ``ref.json`` + ``images/`` presence check
could never be satisfied. Every run therefore re-attempted the download, and
``RetrieverAgent`` silently downgraded ``auto``/``random`` to ``none`` because
the reference file was absent.

These tests pin the acquisition path. ``huggingface_hub`` is replaced with a
fake that builds a real archive with the published layout, so extraction is
exercised rather than stubbed. Offline; no test here opens a socket.
"""

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
import unittest.mock
import zipfile
from pathlib import Path

from skill import run as skill_run


class FakeHub:
    """Stands in for ``huggingface_hub``; records calls, never touches a socket."""

    def __init__(self, *, side_effect=None, archive_root="PaperBananaBench",
                 tasks=("diagram", "plot")) -> None:
        self.calls: list[dict] = []
        self.side_effect = side_effect
        self.archive_root = archive_root
        self.tasks = tasks

    def hf_hub_download(self, repo_id, filename, **kwargs):
        self.calls.append({"repo_id": repo_id, "filename": filename, **kwargs})
        if self.side_effect is not None:
            raise self.side_effect

        local_dir = Path(kwargs.get("local_dir") or tempfile.mkdtemp())
        local_dir.mkdir(parents=True, exist_ok=True)
        archive = local_dir / filename
        with zipfile.ZipFile(archive, "w") as archive_file:
            for task in self.tasks:
                base = f"{self.archive_root}/{task}"
                archive_file.writestr(
                    f"{base}/ref.json",
                    json.dumps([{"id": "x", "path_to_gt_image": "images/x.jpg"}]),
                )
                archive_file.writestr(f"{base}/images/x.jpg", b"not-a-real-jpeg")
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


class DatasetAcquisitionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        patcher = unittest.mock.patch.object(skill_run, "PROJECT_ROOT", self.tmp)
        patcher.start()
        self.addCleanup(patcher.stop)

    def acquire(self, task="diagram"):
        with contextlib.redirect_stdout(io.StringIO()):
            skill_run.ensure_dataset(task)

    def task_dir(self, task="diagram") -> Path:
        return self.tmp / "data" / "PaperBananaBench" / task


class ExtractionTests(DatasetAcquisitionTestCase):
    def test_extraction_produces_the_layout_the_retriever_reads(self) -> None:
        with fake_hub():
            self.acquire()

        self.assertTrue((self.task_dir() / "ref.json").exists())
        self.assertTrue((self.task_dir() / "images").is_dir())

    def test_the_image_paths_in_ref_json_resolve_where_the_planner_looks(self) -> None:
        """PlannerAgent opens <task>/<path_to_gt_image> with no guard."""
        with fake_hub():
            self.acquire()

        entries = json.loads((self.task_dir() / "ref.json").read_text(encoding="utf-8"))
        for entry in entries:
            self.assertTrue((self.task_dir() / entry["path_to_gt_image"]).exists())

    def test_a_second_run_makes_no_network_call(self) -> None:
        with fake_hub() as first:
            self.acquire()
        self.assertEqual(len(first.calls), 1)

        with fake_hub() as second:
            self.acquire()
        self.assertEqual(second.calls, [], "the presence check did not short-circuit")

    def test_an_unexpected_archive_layout_raises_instead_of_half_landing(self) -> None:
        with fake_hub(archive_root="SomethingElse"):
            with self.assertRaises(RuntimeError):
                self.acquire()

        self.assertFalse(
            task_dir_usable_via_module(self.task_dir()),
            "a failed acquisition left a tree that satisfies the presence check",
        )

    def test_a_failed_download_leaves_nothing_usable_behind(self) -> None:
        with fake_hub(side_effect=RuntimeError("network died")):
            with self.assertRaises(RuntimeError):
                self.acquire()

        self.assertFalse(task_dir_usable_via_module(self.task_dir()))


class IncompleteTaskDirectoryTests(DatasetAcquisitionTestCase):
    """A half-populated task directory must be repaired, not stranded.

    Skipping the move when the destination merely exists left such a directory
    in place permanently: it never satisfied the presence check, so every later
    run re-downloaded and re-extracted the archive and then discarded the
    complete copy it had just produced.
    """

    def partial(self, *, ref: bool, images: bool) -> Path:
        task_dir = self.task_dir()
        task_dir.mkdir(parents=True)
        if ref:
            (task_dir / "ref.json").write_text("[]", encoding="utf-8")
        if images:
            (task_dir / "images").mkdir()
        return task_dir

    def test_a_task_dir_missing_images_is_repaired(self) -> None:
        self.partial(ref=True, images=False)

        with fake_hub():
            self.acquire()

        self.assertTrue(task_dir_usable_via_module(self.task_dir()))

    def test_an_empty_task_dir_is_repaired(self) -> None:
        self.partial(ref=False, images=False)

        with fake_hub():
            self.acquire()

        self.assertTrue(task_dir_usable_via_module(self.task_dir()))

    def test_repairing_stops_the_endless_re_download(self) -> None:
        self.partial(ref=True, images=False)

        with fake_hub() as first:
            self.acquire()
        self.assertEqual(len(first.calls), 1)

        with fake_hub() as again:
            self.acquire()
        self.assertEqual(again.calls, [], "a repaired dataset downloaded a second time")

    def test_an_already_complete_task_dir_is_left_alone(self) -> None:
        task_dir = self.partial(ref=False, images=True)
        (task_dir / "ref.json").write_text('["LOCAL"]', encoding="utf-8")

        with fake_hub() as hub:
            self.acquire()

        self.assertEqual(hub.calls, [], "a complete dataset triggered a download")
        self.assertIn("LOCAL", (task_dir / "ref.json").read_text(encoding="utf-8"))

    def test_a_sibling_task_is_not_disturbed_while_repairing_another(self) -> None:
        base = self.tmp / "data" / "PaperBananaBench"
        (base / "plot" / "images").mkdir(parents=True)
        (base / "plot" / "ref.json").write_text('["PLOT-LOCAL"]', encoding="utf-8")
        self.partial(ref=True, images=False)

        with fake_hub():
            self.acquire()

        self.assertTrue(task_dir_usable_via_module(self.task_dir()))
        self.assertIn(
            "PLOT-LOCAL", (base / "plot" / "ref.json").read_text(encoding="utf-8")
        )


def task_dir_usable_via_module(task_dir: Path) -> bool:
    """Assert through the module's own predicate, not a copy of it."""
    return skill_run.task_dir_usable(task_dir)


if __name__ == "__main__":
    unittest.main()
