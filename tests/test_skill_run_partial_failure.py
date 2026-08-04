"""Unit 5: one failing candidate must not discard the rest of a paid run.

process_queries_batch awaited each future unguarded inside an async generator.
A raising candidate closed the generator permanently and the remaining tasks
were cancelled when the loop closed, so no consumer-side try/except could
salvage the batch.

Offline only. Uses unittest.IsolatedAsyncioTestCase; pytest is not installed.
"""

import contextlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from skill import run as skill_run
from utils.paperviz_processor import PaperVizProcessor

from tests.test_skill_run_manifest import args_namespace, full_result


def make_processor(single_query):
    processor = PaperVizProcessor.__new__(PaperVizProcessor)
    # exp_mode 'vanilla' skips the run-once Retriever, which would otherwise
    # need a live agent.
    processor.exp_config = SimpleNamespace(exp_mode="vanilla", retrieval_setting="none")
    processor.process_single_query = single_query
    return processor


class ProcessQueriesBatchGuardTests(unittest.IsolatedAsyncioTestCase):
    async def drain(self, single_query, data_list):
        processor = make_processor(single_query)
        drained = []
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            async for result in processor.process_queries_batch(
                data_list, max_concurrent=len(data_list), do_eval=False
            ):
                drained.append(result)
        return drained

    async def test_all_successful_run_yields_every_candidate(self) -> None:
        data_list = [{"filename": f"skill_candidate_{i}"} for i in range(5)]

        async def ok(data, do_eval=True):
            return dict(data, ok=True)

        drained = await self.drain(ok, data_list)

        self.assertEqual(len(drained), 5)
        self.assertTrue(all(skill_run.candidate_identity(r) for r in drained))
        self.assertFalse(any(skill_run.is_error_record(r) for r in drained))

    async def test_one_of_ten_raising_still_yields_the_other_nine(self) -> None:
        data_list = [{"filename": f"skill_candidate_{i}"} for i in range(10)]

        async def sometimes(data, do_eval=True):
            if data["filename"] == "skill_candidate_4":
                raise RuntimeError("visualizer exhausted retries")
            return dict(data, ok=True)

        drained = await self.drain(sometimes, data_list)

        self.assertEqual(len(drained), 10)
        errors = [r for r in drained if skill_run.is_error_record(r)]
        self.assertEqual(len(errors), 1)
        self.assertEqual(skill_run.candidate_identity(errors[0]), "skill_candidate_4")
        self.assertIn("visualizer exhausted retries", errors[0]["candidate_error"])

        succeeded = {skill_run.candidate_identity(r) for r in drained if not skill_run.is_error_record(r)}
        self.assertEqual(len(succeeded), 9)
        self.assertNotIn("skill_candidate_4", succeeded)

    async def test_every_candidate_failing_still_drains_cleanly(self) -> None:
        data_list = [{"filename": f"skill_candidate_{i}"} for i in range(4)]

        async def always_raises(data, do_eval=True):
            raise ValueError(f"boom {data['filename']}")

        drained = await self.drain(always_raises, data_list)

        self.assertEqual(len(drained), 4)
        self.assertTrue(all(skill_run.is_error_record(r) for r in drained))
        self.assertEqual(
            {skill_run.candidate_identity(r) for r in drained},
            {f"skill_candidate_{i}" for i in range(4)},
        )

    async def test_each_failure_is_reported_against_its_own_identity(self) -> None:
        data_list = [{"filename": f"skill_candidate_{i}"} for i in range(6)]
        failing = {"skill_candidate_1", "skill_candidate_5"}

        async def sometimes(data, do_eval=True):
            if data["filename"] in failing:
                raise RuntimeError(f"failed {data['filename']}")
            return dict(data, ok=True)

        drained = await self.drain(sometimes, data_list)

        errors = {skill_run.candidate_identity(r) for r in drained if skill_run.is_error_record(r)}
        self.assertEqual(errors, failing)


class DrainBatchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.save_kwargs = {
            "exp_mode": "demo_full",
            "output_path": skill_run.default_output_path(self.tmp),
            "num_candidates": 3,
            "output_explicit": False,
            "figure_size": "14-17cm",
            "image_size": "4k",
            "aspect_ratio": "16:9",
        }

    async def _stream(self, items):
        for item in items:
            yield item

    async def test_failed_no_image_and_missing_are_distinct_statuses(self) -> None:
        data_list = [{"filename": f"skill_candidate_{i}"} for i in range(4)]
        entries = skill_run.seed_manifest_entries(data_list)
        stream = self._stream(
            [
                full_result("skill_candidate_0"),
                {"filename": "skill_candidate_1"},  # completed, produced nothing
                {"filename": "skill_candidate_2", "candidate_error": "RuntimeError: boom"},
                # skill_candidate_3 never yields at all
            ]
        )

        with contextlib.redirect_stderr(io.StringIO()):
            await skill_run.drain_batch(stream, entries, **self.save_kwargs)

        self.assertEqual(entries["skill_candidate_0"]["status"], "succeeded")
        self.assertEqual(entries["skill_candidate_1"]["status"], "no_image")
        self.assertEqual(entries["skill_candidate_2"]["status"], "failed")
        self.assertEqual(entries["skill_candidate_3"]["status"], "missing")
        self.assertEqual(skill_run.run_status(entries), "partial")

    async def test_failed_entry_names_the_candidate_and_its_error(self) -> None:
        entries = skill_run.seed_manifest_entries([{"filename": "skill_candidate_0"}])
        stream = self._stream(
            [{"filename": "skill_candidate_0", "candidate_error": "RuntimeError: exhausted retries"}]
        )

        with contextlib.redirect_stderr(io.StringIO()):
            await skill_run.drain_batch(stream, entries, **self.save_kwargs)

        entry = entries["skill_candidate_0"]
        self.assertEqual(entry["identity"], "skill_candidate_0")
        self.assertIn("exhausted retries", entry["error"])
        self.assertIsNone(entry["image_path"])

    async def test_surviving_images_are_written_when_a_sibling_fails(self) -> None:
        data_list = [{"filename": f"skill_candidate_{i}"} for i in range(3)]
        entries = skill_run.seed_manifest_entries(data_list)
        stream = self._stream(
            [
                full_result("skill_candidate_0"),
                {"filename": "skill_candidate_1", "candidate_error": "RuntimeError: boom"},
                full_result("skill_candidate_2"),
            ]
        )

        with contextlib.redirect_stderr(io.StringIO()):
            await skill_run.drain_batch(stream, entries, **self.save_kwargs)

        written = sorted(p.name for p in self.tmp.iterdir())
        self.assertEqual(written, ["skill_candidate_0.png", "skill_candidate_2.png"])

    async def test_run_dying_mid_drain_leaves_seeded_entries_partial(self) -> None:
        data_list = [{"filename": f"skill_candidate_{i}"} for i in range(10)]
        entries = skill_run.seed_manifest_entries(data_list)

        async def dying_stream():
            for i in range(4):
                yield full_result(f"skill_candidate_{i}")
            raise asyncio_cancel()

        def asyncio_cancel():
            return RuntimeError("event loop died")

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(RuntimeError):
                await skill_run.drain_batch(dying_stream(), entries, **self.save_kwargs)

        self.assertEqual(skill_run.run_status(entries), "partial")
        self.assertEqual(
            sum(1 for e in entries.values() if e["status"] == "succeeded"), 4
        )
        self.assertEqual(sum(1 for e in entries.values() if e["status"] == "missing"), 6)

    async def test_manifest_written_after_a_partial_drain_records_the_loss(self) -> None:
        data_list = [{"filename": f"skill_candidate_{i}"} for i in range(3)]
        entries = skill_run.seed_manifest_entries(data_list)
        stream = self._stream(
            [
                full_result("skill_candidate_0"),
                {"filename": "skill_candidate_1", "candidate_error": "RuntimeError: boom"},
            ]
        )

        with contextlib.redirect_stderr(io.StringIO()):
            await skill_run.drain_batch(stream, entries, **self.save_kwargs)

        manifest = skill_run.build_manifest(
            args=args_namespace(num_candidates=3),
            additional_info={"rounded_ratio": "16:9", "figure_size": "14-17cm", "image_size": "4k"},
            entries=entries,
            candidates_requested=len(data_list),
            content="method text",
            resolved_models={"main_model_name": "m", "image_gen_model_name": "i"},
            image_gen_backend="gemini",
            retrieval={"setting": "auto", "top10_references_count": 0, "retrieved_examples_count": 0},
            started_at="2026-08-04T10:15:00Z",
            finished_at="2026-08-04T10:38:00Z",
        )
        path = skill_run.write_manifest(
            skill_run.manifest_path_for(self.save_kwargs["output_path"]), manifest
        )

        written = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(written["run"]["status"], "partial")
        self.assertEqual(written["run"]["candidates_requested"], 3)
        self.assertEqual(written["run"]["candidates_succeeded"], 1)
        by_identity = {e["identity"]: e for e in written["candidates"]}
        self.assertEqual(by_identity["skill_candidate_1"]["status"], "failed")
        self.assertIn("boom", by_identity["skill_candidate_1"]["error"])
        self.assertEqual(by_identity["skill_candidate_2"]["status"], "missing")


class SingleEmitterTests(unittest.TestCase):
    def test_write_manifest_has_exactly_one_call_site_in_run_py(self) -> None:
        """Normal and failure paths must route through one emitter."""
        source = Path(skill_run.__file__).read_text(encoding="utf-8")
        calls = re.findall(r"^\s*(?:\w+\s*=\s*)?write_manifest\(", source, re.MULTILINE)

        self.assertEqual(len(calls), 1, "write_manifest must be invoked from one place")
        self.assertIn("finally:", source)


if __name__ == "__main__":
    unittest.main()
