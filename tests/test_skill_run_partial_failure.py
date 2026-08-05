"""Unit 5: one failing candidate must not discard the rest of a paid run.

process_queries_batch awaited each future unguarded inside an async generator.
A raising candidate closed the generator permanently and the remaining tasks
were cancelled when the loop closed, so no consumer-side try/except could
salvage the batch.

Offline only. Uses unittest.IsolatedAsyncioTestCase; pytest is not installed.
"""

import base64
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

from tests.test_skill_run_manifest import SENTINEL_KEY, args_namespace, full_result


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
        """The identity and the error text must come from the same candidate.

        as_completed hands back futures in completion order, which need not
        match the lowest-index scan that attributes the failure, so comparing
        only the *set* of failed identities cannot tell a correctly paired
        report from one that files candidate_1 under candidate_5's error.
        """
        data_list = [{"filename": f"skill_candidate_{i}"} for i in range(6)]
        failing = {"skill_candidate_1", "skill_candidate_5"}

        async def sometimes(data, do_eval=True):
            if data["filename"] in failing:
                raise RuntimeError(f"failed {data['filename']}")
            return dict(data, ok=True)

        drained = await self.drain(sometimes, data_list)

        errors = {
            skill_run.candidate_identity(record): record["candidate_error"]
            for record in drained
            if skill_run.is_error_record(record)
        }
        self.assertEqual(set(errors), failing)
        for identity, message in errors.items():
            self.assertIn(f"failed {identity}", message)

    async def test_the_raw_sdk_error_text_never_reaches_the_run_stream(self) -> None:
        """The processor's own announcement is upstream of every redaction.

        A provider SDK that echoes the key it was called with would put a live
        credential on the operator's log, which is exactly the stream a long
        headless run gets teed to a file.
        """
        sentinel = SENTINEL_KEY
        data_list = [{"filename": "skill_candidate_0"}]

        async def raises(data, do_eval=True):
            raise RuntimeError(f"400 INVALID_ARGUMENT: API key not valid: {sentinel}")

        processor = make_processor(raises)
        out, err = io.StringIO(), io.StringIO()
        drained = []
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            async for record in processor.process_queries_batch(
                data_list, max_concurrent=1, do_eval=False
            ):
                drained.append(record)

        stream = out.getvalue() + err.getvalue()
        self.assertNotIn(sentinel, stream)
        # The failure is still announced, by type.
        self.assertIn("skill_candidate_0 failed", stream)
        self.assertIn("RuntimeError", stream)
        # The record still carries the diagnosis for the caller to redact.
        self.assertIn("INVALID_ARGUMENT", drained[0]["candidate_error"])

    async def test_the_same_run_never_emits_the_key_both_masked_and_unmasked(self) -> None:
        """End to end: processor announcement plus the caller's redacted one."""
        sentinel = SENTINEL_KEY
        data_list = [{"filename": "skill_candidate_0"}]
        entries = skill_run.seed_manifest_entries(data_list)

        async def raises(data, do_eval=True):
            raise RuntimeError(f"401 Unauthorized (key={sentinel})")

        processor = make_processor(raises)
        out, err = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            save_kwargs = {
                "exp_mode": "demo_full",
                "output_path": skill_run.default_output_path(Path(tmp)),
                "num_candidates": 1,
                "output_explicit": False,
                "figure_size": "14-17cm",
                "image_size": "4k",
                "aspect_ratio": "16:9",
            }
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                await skill_run.drain_batch(
                    processor.process_queries_batch(
                        data_list, max_concurrent=1, do_eval=False
                    ),
                    entries,
                    **save_kwargs,
                )

        stream = out.getvalue() + err.getvalue()
        self.assertNotIn(sentinel, stream)
        self.assertIn(skill_run.REDACTED_CREDENTIAL, stream)
        self.assertNotIn(sentinel, entries["skill_candidate_0"]["error"])


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

    async def test_a_result_that_cannot_be_written_fails_alone(self) -> None:
        """The consumer side of "a failing candidate cannot destroy the batch".

        A raise inside the ``async for`` body escapes the loop, which closes the
        async generator and cancels every undrained candidate, so a single bad
        payload discarded pipeline work that had already been paid for.
        """
        data_list = [{"filename": f"skill_candidate_{i}"} for i in range(4)]
        entries = skill_run.seed_manifest_entries(data_list)

        undecodable = full_result("skill_candidate_1")
        # Decodes cleanly, is not an image: PIL raises on open.
        undecodable["target_diagram_critic_desc1_base64_jpg"] = base64.b64encode(
            b"definitely not a png"
        ).decode("ascii")

        stream = self._stream(
            [
                full_result("skill_candidate_0"),
                undecodable,
                full_result("skill_candidate_2"),
                full_result("skill_candidate_3"),
            ]
        )

        with contextlib.redirect_stderr(io.StringIO()):
            await skill_run.drain_batch(stream, entries, **self.save_kwargs)

        self.assertEqual(entries["skill_candidate_1"]["status"], "failed")
        self.assertIsNotNone(entries["skill_candidate_1"]["error"])
        for identity in ("skill_candidate_0", "skill_candidate_2", "skill_candidate_3"):
            self.assertEqual(entries[identity]["status"], "succeeded")
        self.assertEqual(
            sorted(p.name for p in self.tmp.iterdir()),
            ["skill_candidate_0.png", "skill_candidate_2.png", "skill_candidate_3.png"],
        )
        self.assertEqual(skill_run.run_status(entries), "partial")

    async def test_an_unwritable_destination_fails_candidates_not_the_drain(self) -> None:
        """Every candidate is accounted for even when nothing can be written."""
        readonly = self.tmp / "readonly"
        readonly.mkdir()
        readonly.chmod(0o500)
        self.addCleanup(readonly.chmod, 0o700)

        data_list = [{"filename": f"skill_candidate_{i}"} for i in range(4)]
        entries = skill_run.seed_manifest_entries(data_list)
        save_kwargs = dict(
            self.save_kwargs,
            output_path=skill_run.default_output_path(readonly / "figs"),
            num_candidates=4,
        )
        stream = self._stream([full_result(d["filename"]) for d in data_list])

        with contextlib.redirect_stderr(io.StringIO()):
            await skill_run.drain_batch(stream, entries, **save_kwargs)

        self.assertEqual(
            {entry["status"] for entry in entries.values()}, {"failed"}
        )
        self.assertEqual(skill_run.run_status(entries), "failed")
        for entry in entries.values():
            self.assertIn("Error", entry["error"])

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
