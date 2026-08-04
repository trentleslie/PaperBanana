"""Unit 4: every run persists a slim, credential-free record; stdout is clean.

Offline only. No model provider is contacted.
"""

import base64
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from skill import run as skill_run

from tests.test_skill_run_candidate_identity import png_b64


SENTINEL_KEY = "AIzaSy-THIS-IS-A-TEST-SENTINEL-NOT-A-REAL-KEY"


def long_base64_blob(nbytes: int = 2048) -> str:
    return base64.b64encode(os.urandom(nbytes)).decode("ascii")


def full_result(identity: str, critic_rounds: int = 2) -> dict:
    result = {
        "filename": identity,
        "caption": "Figure 1: overview",
        "target_diagram_desc0": "planner description",
        "target_diagram_desc0_base64_jpg": png_b64(),
        "target_diagram_stylist_desc0": "stylist description",
        "target_diagram_stylist_desc0_base64_jpg": png_b64(),
    }
    for round_idx in range(critic_rounds):
        result[f"target_diagram_critic_desc{round_idx}"] = f"critic {round_idx} description"
        result[f"target_diagram_critic_desc{round_idx}_base64_jpg"] = png_b64()
        result[f"target_diagram_critic_suggestions{round_idx}"] = f"critic {round_idx} suggestions"
    return result


def planner_stylist_only_result(identity: str) -> dict:
    """No critic stage at all, so no stage carries a 'suggestions_key'."""
    return {
        "filename": identity,
        "target_diagram_desc0": "planner description",
        "target_diagram_desc0_base64_jpg": png_b64(),
        "target_diagram_stylist_desc0": "stylist description",
        "target_diagram_stylist_desc0_base64_jpg": png_b64(),
    }


def args_namespace(**overrides) -> Namespace:
    values = dict(
        content="method text",
        content_file=None,
        caption="Figure 1: overview",
        task="diagram",
        output=None,
        aspect_ratio="16:9",
        figure_size="14-17cm",
        max_critic_rounds=3,
        num_candidates=2,
        retrieval_setting="auto",
        planner_metaphor=False,
        main_model_name="",
        image_gen_model_name="",
        exp_mode="demo_full",
    )
    values.update(overrides)
    return Namespace(**values)


class ManifestPlacementTests(unittest.TestCase):
    def test_manifest_shares_the_output_stem_and_sits_beside_the_images(self) -> None:
        path = skill_run.manifest_path_for(Path("/tmp/runs/run_1/output.png"))

        self.assertEqual(path.parent, Path("/tmp/runs/run_1"))
        self.assertEqual(path.name, "output.manifest.json")

        explicit = skill_run.manifest_path_for(Path("/tmp/figs/architecture.png"))
        self.assertEqual(explicit.name, "architecture.manifest.json")


class SeededEntryTests(unittest.TestCase):
    def test_entries_are_seeded_from_data_list_before_draining(self) -> None:
        data_list = [{"filename": f"skill_candidate_{i}"} for i in range(10)]

        entries = skill_run.seed_manifest_entries(data_list)

        self.assertEqual(len(entries), 10)
        self.assertEqual({e["status"] for e in entries.values()}, {"missing"})
        self.assertIn("skill_candidate_9", entries)
        self.assertIsNone(entries["skill_candidate_0"]["image_path"])

    def test_run_is_complete_only_when_no_entry_is_missing_or_failed(self) -> None:
        entries = skill_run.seed_manifest_entries(
            [{"filename": "a"}, {"filename": "b"}]
        )
        self.assertEqual(skill_run.run_status(entries), "failed")

        entries["a"]["status"] = "succeeded"
        self.assertEqual(skill_run.run_status(entries), "partial")

        entries["b"]["status"] = "succeeded"
        self.assertEqual(skill_run.run_status(entries), "complete")

        entries["b"]["status"] = "no_image"
        self.assertEqual(skill_run.run_status(entries), "partial")

    def test_run_that_dies_after_four_of_ten_yields_is_partial(self) -> None:
        data_list = [{"filename": f"skill_candidate_{i}"} for i in range(10)]
        entries = skill_run.seed_manifest_entries(data_list)
        for i in range(4):
            entries[f"skill_candidate_{i}"]["status"] = "succeeded"

        self.assertEqual(skill_run.run_status(entries), "partial")


class TraceTests(unittest.TestCase):
    def test_planner_stylist_and_every_critic_round_appear(self) -> None:
        trace = skill_run.build_candidate_trace(full_result("skill_candidate_0"), "demo_full")

        names = [stage["name"] for stage in trace]
        self.assertIn("Planner", names)
        self.assertIn("Stylist", names)
        self.assertIn("Critic Round 0", names)
        self.assertIn("Critic Round 1", names)

        critic0 = next(s for s in trace if s["name"] == "Critic Round 0")
        self.assertEqual(critic0["suggestions"], "critic 0 suggestions")
        self.assertEqual(critic0["description"], "critic 0 description")

    def test_planner_and_stylist_only_result_does_not_raise(self) -> None:
        """Only Critic stages carry 'suggestions_key'; a literal lookup would raise."""
        trace = skill_run.build_candidate_trace(
            planner_stylist_only_result("skill_candidate_0"), "demo_full"
        )

        names = [stage["name"] for stage in trace]
        self.assertEqual(names, ["Planner", "Stylist"])
        self.assertNotIn("suggestions", trace[0])

    def test_no_stage_carries_a_base64_image_payload(self) -> None:
        """image_key is retained as a pointer; the payload it names is not."""
        result = full_result("skill_candidate_0")
        trace = skill_run.build_candidate_trace(result, "demo_full")

        for stage in trace:
            for key, value in stage.items():
                self.assertFalse(
                    key.endswith(skill_run.BASE64_SUFFIX),
                    f"base64 key survived: {key}",
                )
                self.assertFalse(
                    skill_run.looks_like_base64_payload(value),
                    f"base64 payload survived under {key}",
                )

        serialized = json.dumps(trace)
        for key, value in result.items():
            if key.endswith(skill_run.BASE64_SUFFIX):
                self.assertNotIn(value, serialized)

    def test_payload_hidden_under_a_text_field_name_is_elided(self) -> None:
        """A key-name check alone would admit a blob stored under another name."""
        blob = long_base64_blob()
        self.assertTrue(skill_run.looks_like_base64_payload(blob))

        result = full_result("skill_candidate_0")
        result["target_diagram_critic_suggestions0"] = blob

        trace = skill_run.build_candidate_trace(result, "demo_full")

        self.assertNotIn(blob, json.dumps(trace))

    def test_short_prose_is_not_mistaken_for_a_payload(self) -> None:
        self.assertFalse(skill_run.looks_like_base64_payload("critic 0 suggestions"))
        self.assertFalse(skill_run.looks_like_base64_payload("Make the arrows thinner."))
        self.assertFalse(skill_run.looks_like_base64_payload(None))
        self.assertFalse(skill_run.looks_like_base64_payload(42))


class BackendDerivationTests(unittest.TestCase):
    def test_backend_is_derived_from_model_name_and_openrouter_client(self) -> None:
        self.assertEqual(
            skill_run.derive_image_gen_backend("gpt-image-1", None), "openai"
        )
        self.assertEqual(
            skill_run.derive_image_gen_backend("gpt-image-1", object()), "openai"
        )
        self.assertEqual(
            skill_run.derive_image_gen_backend("gemini-3.1-flash-image-preview", object()),
            "openrouter",
        )
        self.assertEqual(
            skill_run.derive_image_gen_backend("gemini-3.1-flash-image-preview", None),
            "gemini",
        )


class RepoCommitTests(unittest.TestCase):
    def test_commit_is_recorded_with_an_explicit_dirty_flag(self) -> None:
        record = skill_run.repo_commit_record()

        self.assertIn("commit", record)
        self.assertIn("dirty", record)
        self.assertIsInstance(record["dirty"], bool)
        if record["commit"] is not None:
            self.assertRegex(record["commit"], r"^[0-9a-f]{40}$")


class ManifestBuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def build(self, results, *, num_candidates=None, args=None, data_list=None):
        results = list(results)
        num_candidates = num_candidates if num_candidates is not None else len(results)
        data_list = data_list or [{"filename": r["filename"]} for r in results]
        args = args or args_namespace(num_candidates=num_candidates)
        entries = skill_run.seed_manifest_entries(data_list)

        output_path = skill_run.default_output_path(self.tmp)
        with contextlib.redirect_stderr(io.StringIO()):
            for result in results:
                skill_run.record_result(
                    result,
                    entries,
                    exp_mode="demo_full",
                    output_path=output_path,
                    num_candidates=num_candidates,
                    output_explicit=False,
                    figure_size="14-17cm",
                    image_size="4k",
                    aspect_ratio="16:9",
                )

        manifest = skill_run.build_manifest(
            args=args,
            additional_info={"rounded_ratio": "16:9", "figure_size": "14-17cm", "image_size": "4k"},
            entries=entries,
            content="method text",
            resolved_models={
                "main_model_name": "gemini-3.1-pro-preview",
                "image_gen_model_name": "gemini-3.1-flash-image-preview",
            },
            image_gen_backend="gemini",
            retrieval={"top10_references_count": 0, "retrieved_examples_count": 0},
            started_at="2026-08-04T10:15:00Z",
            finished_at="2026-08-04T10:38:00Z",
        )
        return entries, manifest, output_path

    def test_completed_run_entries_match_the_saved_images_one_for_one(self) -> None:
        entries, manifest, _ = self.build(
            [full_result("skill_candidate_0"), full_result("skill_candidate_1")]
        )

        self.assertEqual(manifest["run"]["status"], "complete")
        self.assertEqual(len(manifest["candidates"]), 2)
        for entry in manifest["candidates"]:
            self.assertEqual(entry["status"], "succeeded")
            self.assertTrue(Path(entry["image_path"]).exists())
            self.assertIn(entry["identity"], Path(entry["image_path"]).name)

    def test_identity_used_for_the_filename_is_byte_identical_to_the_manifest_key(self) -> None:
        entries, manifest, _ = self.build([full_result("skill_candidate_3")])

        entry = manifest["candidates"][0]
        self.assertIn(entry["identity"], entries)
        self.assertEqual(Path(entry["image_path"]).stem, entry["identity"])

    def test_candidate_producing_no_image_still_appears_with_a_null_path(self) -> None:
        _, manifest, _ = self.build(
            [full_result("skill_candidate_0"), {"filename": "skill_candidate_1"}]
        )

        by_identity = {e["identity"]: e for e in manifest["candidates"]}
        self.assertEqual(by_identity["skill_candidate_1"]["status"], "no_image")
        self.assertIsNone(by_identity["skill_candidate_1"]["image_path"])
        self.assertEqual(manifest["run"]["status"], "partial")

    def test_candidate_never_yielded_stays_missing(self) -> None:
        data_list = [{"filename": f"skill_candidate_{i}"} for i in range(3)]
        _, manifest, _ = self.build(
            [full_result("skill_candidate_0")], num_candidates=3, data_list=data_list
        )

        by_identity = {e["identity"]: e for e in manifest["candidates"]}
        self.assertEqual(by_identity["skill_candidate_1"]["status"], "missing")
        self.assertEqual(by_identity["skill_candidate_2"]["status"], "missing")
        self.assertEqual(manifest["run"]["status"], "partial")

    def test_manifest_pins_the_parameters_needed_to_reproduce_the_run(self) -> None:
        _, manifest, _ = self.build([full_result("skill_candidate_0")])

        run = manifest["run"]
        self.assertEqual(run["started_at"], "2026-08-04T10:15:00Z")
        self.assertEqual(run["finished_at"], "2026-08-04T10:38:00Z")
        self.assertEqual(run["image_gen_backend_derived"], "gemini")
        self.assertEqual(run["models"]["main_model_name"], "gemini-3.1-pro-preview")
        self.assertEqual(run["resolved_image_size"], "4k")
        self.assertIn("commit", run["repository"])

        params = manifest["parameters"]
        self.assertEqual(params["figure_size"], "14-17cm")
        self.assertEqual(params["aspect_ratio"], "16:9")
        self.assertEqual(params["exp_mode"], "demo_full")
        self.assertEqual(params["retrieval_setting"], "auto")

        self.assertEqual(manifest["input"]["content"], "method text")
        self.assertEqual(len(manifest["input"]["content_sha256"]), 64)

    def test_sentinel_credential_never_reaches_the_serialized_manifest(self) -> None:
        """Allowlist construction, not filtering. Sentinel is asserted non-empty
        so this fails loudly rather than no-opping."""
        self.assertTrue(SENTINEL_KEY)

        args = args_namespace(num_candidates=1)
        # Credential-shaped material hung off every surface the emitter sees.
        args.google_api_key = SENTINEL_KEY
        args.api_keys = {"google_api_key": SENTINEL_KEY}
        result = full_result("skill_candidate_0")
        result["api_keys"] = {"google_api_key": SENTINEL_KEY}

        _, manifest, _ = self.build([result], num_candidates=1, args=args)

        self.assertNotIn(SENTINEL_KEY, json.dumps(manifest))

    def test_manifest_is_orders_of_magnitude_smaller_than_the_ui_output(self) -> None:
        results = [full_result(f"skill_candidate_{i}", critic_rounds=3) for i in range(10)]
        _, manifest, output_path = self.build(results, num_candidates=10)

        path = skill_run.write_manifest(skill_run.manifest_path_for(output_path), manifest)

        self.assertTrue(path.exists())
        # The UI writes roughly 210-234MB per run.
        self.assertLess(path.stat().st_size, 1_000_000)

        def assert_no_payload(node, where="manifest"):
            if isinstance(node, dict):
                for key, value in node.items():
                    self.assertFalse(
                        isinstance(key, str) and key.endswith(skill_run.BASE64_SUFFIX),
                        f"base64 key survived at {where}.{key}",
                    )
                    assert_no_payload(value, f"{where}.{key}")
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    assert_no_payload(value, f"{where}[{index}]")
            else:
                self.assertFalse(
                    skill_run.looks_like_base64_payload(node),
                    f"base64 payload survived at {where}",
                )

        assert_no_payload(json.loads(path.read_text(encoding="utf-8")))

    def test_write_manifest_creates_the_parent_directory(self) -> None:
        _, manifest, _ = self.build([full_result("skill_candidate_0")])
        target = self.tmp / "nested" / "deeper" / "output.manifest.json"

        path = skill_run.write_manifest(target, manifest)

        self.assertTrue(path.exists())
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["run"]["status"], "complete")


class StdoutContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_pipeline_chatter_written_to_stdout_lands_on_stderr(self) -> None:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            with skill_run.quiet_pipeline_stdout():
                print("[Retriever] Running retrieval once for all candidates...")

        self.assertEqual(out.getvalue(), "")
        self.assertIn("[Retriever]", err.getvalue())

    def test_stdout_carries_only_image_paths_and_manifest_goes_to_stderr(self) -> None:
        image = self.tmp / "skill_candidate_0.png"
        image.write_bytes(b"not really a png")
        entries = {
            "skill_candidate_0": {
                "identity": "skill_candidate_0",
                "status": "succeeded",
                "image_path": str(image),
            },
            "skill_candidate_1": {
                "identity": "skill_candidate_1",
                "status": "no_image",
                "image_path": None,
            },
        }
        manifest_path = self.tmp / "output.manifest.json"

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            skill_run.emit_results(entries, manifest_path)

        lines = [line for line in out.getvalue().splitlines() if line]
        self.assertEqual(lines, [str(image)])
        for line in lines:
            self.assertTrue(Path(line).exists())
        self.assertNotIn(str(manifest_path), out.getvalue())
        self.assertIn(str(manifest_path), err.getvalue())


if __name__ == "__main__":
    unittest.main()
