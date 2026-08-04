"""Unit 4b: the guarantees are asserted against ``run()``, not its helpers.

The published stdout contract is a property of the *process*, not of
``emit_results``. Testing the helpers in isolation is how a run whose first two
stdout lines were "Initialized Gemini Client with API Key" passed a green suite:
``utils.generation_utils`` prints one such line per configured key at import
time, and run.py's late imports sat outside the redirect.

Every module ``run()`` imports late is replaced here with a stub that prints to
stdout at the same moments the real one does: at import, at construction, and
while draining. No provider is contacted and no image model is called.
"""

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
import unittest.mock
from pathlib import Path

from skill import run as skill_run

from tests.test_skill_run_manifest import args_namespace, full_result


# The modules run() imports after argument parsing, in import order.
LATE_IMPORTS = {
    "agents.planner_agent": "PlannerAgent",
    "agents.visualizer_agent": "VisualizerAgent",
    "agents.stylist_agent": "StylistAgent",
    "agents.critic_agent": "CriticAgent",
    "agents.retriever_agent": "RetrieverAgent",
    "agents.vanilla_agent": "VanillaAgent",
    "agents.polish_agent": "PolishAgent",
}

IMPORT_BANNER = "Initialized Gemini Client with API Key"
DOWNLOAD_BANNER = "Downloading PaperBananaBench/diagram from HuggingFace..."
CONFIG_WARNING = "Warning: main_model_name not configured, falling back to '...'."
PIPELINE_CHATTER = "[Retriever] Running retrieval once for all candidates..."


class StubAgent:
    def __init__(self, *args, **kwargs) -> None:
        pass


class StubExpConfig:
    """Mirrors ExpConfig.__post_init__, which warns on stdout."""

    def __init__(self, **kwargs) -> None:
        print(CONFIG_WARNING)
        self.main_model_name = "gemini-3.1-pro-preview"
        self.image_gen_model_name = "gemini-3.1-flash-image-preview"
        for key, value in kwargs.items():
            setattr(self, key, value)


def chatty_module(name: str, exports: dict, banner: str | None = None):
    """A module that prints when an attribute is first read.

    A stub placed in ``sys.modules`` cannot print "at import time" the way
    ``utils.generation_utils`` does, so it prints at the ``from x import y``
    lookup instead: the same instant in run()'s execution.
    """
    module = types.ModuleType(name)
    state = {"announced": banner is None}

    def __getattr__(attr):
        if not state["announced"]:
            state["announced"] = True
            print(banner)
        if attr in exports:
            return exports[attr]
        raise AttributeError(attr)

    module.__getattr__ = __getattr__
    return module


@contextlib.contextmanager
def stubbed_pipeline(batch, *, processor_chatter: str = "[Pipeline] processor ready"):
    """Swap in chatty stubs for everything run() imports late."""

    class StubProcessor:
        def __init__(self, **kwargs) -> None:
            print(processor_chatter)

        def process_queries_batch(self, data_list, max_concurrent=10, do_eval=True):
            return batch(data_list)

    fakes = {
        name: chatty_module(name, {export: StubAgent})
        for name, export in LATE_IMPORTS.items()
    }
    fakes["utils.config"] = chatty_module("utils.config", {"ExpConfig": StubExpConfig})
    fakes["utils.generation_utils"] = chatty_module(
        "utils.generation_utils", {"openrouter_client": None}, banner=IMPORT_BANNER
    )
    fakes["utils.paperviz_processor"] = chatty_module(
        "utils.paperviz_processor", {"PaperVizProcessor": StubProcessor}
    )

    saved_modules = {}
    saved_attrs = {}
    for name, fake in fakes.items():
        saved_modules[name] = sys.modules.get(name)
        sys.modules[name] = fake
        package, _, tail = name.rpartition(".")
        parent = sys.modules.get(package)
        if parent is not None:
            # `from utils import config` reads the attribute off the already
            # imported package before it ever consults sys.modules.
            saved_attrs[(package, tail)] = getattr(parent, tail, None)
            setattr(parent, tail, fake)

    def stub_ensure_dataset(task_name):
        print(DOWNLOAD_BANNER)

    saved_ensure = skill_run.ensure_dataset
    skill_run.ensure_dataset = stub_ensure_dataset
    try:
        yield
    finally:
        skill_run.ensure_dataset = saved_ensure
        for name, original in saved_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
        for (package, tail), original in saved_attrs.items():
            parent = sys.modules.get(package)
            if parent is None:
                continue
            if original is None:
                delattr(parent, tail)
            else:
                setattr(parent, tail, original)


class RunEntryPointTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def args(self, **overrides):
        values = dict(
            num_candidates=2,
            output=str(self.tmp / "figure.png"),
            content="method text",
        )
        values.update(overrides)
        return args_namespace(**values)

    async def execute(self, batch, args):
        out, err = io.StringIO(), io.StringIO()
        with stubbed_pipeline(batch):
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                await skill_run.run(args)
        return out.getvalue(), err.getvalue()

    def manifest(self):
        path = self.tmp / "figure.manifest.json"
        self.assertTrue(path.exists(), "manifest was not written")
        return json.loads(path.read_text(encoding="utf-8"))

    async def test_every_stdout_line_of_a_real_run_is_an_existing_image_path(self) -> None:
        async def batch(data_list):
            print(PIPELINE_CHATTER)
            for data in data_list:
                yield full_result(data["filename"])

        out, err = await self.execute(batch, self.args())

        lines = [line for line in out.splitlines() if line.strip()]
        self.assertEqual(len(lines), 2)
        for line in lines:
            self.assertTrue(
                Path(line).is_file(), f"stdout line is not an image path: {line!r}"
            )
        # Everything that printed to stdout inside the run went to stderr instead.
        for chatter in (IMPORT_BANNER, DOWNLOAD_BANNER, CONFIG_WARNING, PIPELINE_CHATTER):
            self.assertIn(chatter, err)
            self.assertNotIn(chatter, out)
        self.assertIn("[manifest] ", err)
        self.assertNotIn("[manifest]", out)

    async def test_a_run_dying_mid_drain_still_emits_the_images_it_saved(self) -> None:
        async def batch(data_list):
            yield full_result(data_list[0]["filename"])
            raise RuntimeError("shared retriever exploded before the tasks were created")

        out, err = io.StringIO(), io.StringIO()
        with stubbed_pipeline(batch):
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                with self.assertRaises(RuntimeError):
                    await skill_run.run(self.args(num_candidates=3))

        lines = [line for line in out.getvalue().splitlines() if line.strip()]
        self.assertEqual(len(lines), 1)
        self.assertTrue(Path(lines[0]).is_file())
        # The caller is told where the record of the loss is, not left guessing.
        self.assertIn("[manifest] ", err.getvalue())

        manifest = self.manifest()
        self.assertEqual(manifest["run"]["status"], "partial")
        self.assertEqual(manifest["run"]["candidates_requested"], 3)
        self.assertEqual(manifest["run"]["candidates_succeeded"], 1)

    async def test_a_run_that_produced_nothing_exits_nonzero_with_empty_stdout(self) -> None:
        async def batch(data_list):
            for data in data_list:
                yield {"filename": data["filename"], "candidate_error": "RuntimeError: boom"}

        out, err = io.StringIO(), io.StringIO()
        with stubbed_pipeline(batch):
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                with self.assertRaises(SystemExit) as caught:
                    await skill_run.run(self.args())

        self.assertEqual(caught.exception.code, 1)
        self.assertEqual(out.getvalue().strip(), "")
        self.assertEqual(self.manifest()["run"]["status"], "failed")

    async def test_an_unattributed_failure_does_not_inflate_candidates_requested(self) -> None:
        """paperviz_processor yields filename=None when it cannot attribute a raise."""

        async def batch(data_list):
            yield full_result(data_list[0]["filename"])
            yield {"filename": None, "candidate_error": "RuntimeError: unattributable"}

        await self.execute(batch, self.args(num_candidates=3))

        manifest = self.manifest()
        self.assertEqual(manifest["run"]["candidates_requested"], 3)
        identities = [entry["identity"] for entry in manifest["candidates"]]
        self.assertEqual(len(identities), 4)
        self.assertIn("unattributed_failure_0", identities)
        # The candidate that actually vanished is still reported as lost.
        by_identity = {entry["identity"]: entry for entry in manifest["candidates"]}
        self.assertEqual(by_identity["skill_candidate_1"]["status"], "missing")

    async def test_the_manifest_records_the_retrieval_shared_across_candidates(self) -> None:
        """retriever_agent mutates data_list[0] in place; the record reads it back."""

        async def batch(data_list):
            data_list[0]["top10_references"] = [{"paper": "a"}, {"paper": "b"}]
            data_list[0]["retrieved_examples"] = [{"image": "x"}]
            for data in data_list:
                yield full_result(data["filename"])

        await self.execute(batch, self.args(num_candidates=1, retrieval_setting="auto"))

        retrieval = self.manifest()["retrieval"]
        self.assertEqual(retrieval["setting"], "auto")
        self.assertEqual(retrieval["top10_references_count"], 2)
        self.assertEqual(retrieval["retrieved_examples_count"], 1)


class DatasetBannerTests(unittest.TestCase):
    """The real ensure_dataset banner, not a stub of it."""

    def test_the_download_banner_goes_to_stderr(self) -> None:
        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.snapshot_download = lambda *args, **kwargs: None

        out, err = io.StringIO(), io.StringIO()
        with unittest.mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                skill_run.ensure_dataset("definitely_not_a_downloaded_task")

        self.assertEqual(out.getvalue(), "")
        self.assertIn("Downloading PaperBananaBench/", err.getvalue())


if __name__ == "__main__":
    unittest.main()
