"""Unit 4: every run persists a slim, credential-free record; stdout is clean.

Offline only. No model provider is contacted.
"""

import base64
import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from argparse import Namespace
from pathlib import Path

from skill import run as skill_run
from utils.legacy_ui_results import BASE64_SUFFIX

from tests.test_skill_run_candidate_identity import png_b64


SENTINEL_KEY = "AIzaSy-THIS-IS-A-TEST-SENTINEL-NOT-A-REAL-KEY"
# Deliberately NOT key-shaped, so only the model_config.yaml value-sourcing
# branch of redact_credentials can catch it. See the pair of tests below.
UNSHAPED_CONFIG_SENTINEL = "PBTEST-config-sourced-sentinel-0011223344"


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
                    skill_run.looks_like_base64_payload(value),
                    f"base64 payload survived under {key}",
                )

        # The pointer survives, the thing it points at does not. Asserted on the
        # payload values rather than on the key names: the trace is built from
        # this module's own field names, so a key-name assertion here could
        # never have gone red.
        serialized = json.dumps(trace)
        self.assertIn("image_key", trace[0])
        for key, value in result.items():
            if key.endswith(BASE64_SUFFIX):
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

    def test_the_length_floor_is_what_saves_short_single_words(self) -> None:
        """Pins BASE64_VALUE_MIN_LENGTH on its own.

        Every other short string in these tests contains a space or a period, so
        the character allowlist rejects it and the floor could be deleted with
        the suite still green. These are the manifest's own vocabulary: pure
        letters and digits, no separator, and short. Only the floor keeps them.
        """
        for word in ("succeeded", "complete", "Planner", "4k", "gemini"):
            self.assertIsNotNone(
                skill_run._BASE64_VALUE_RE.match(word),
                f"{word!r} must be allowlist-clean, or the floor goes untested",
            )
            self.assertFalse(skill_run.looks_like_base64_payload(word))

    def test_a_short_data_uri_is_still_a_payload(self) -> None:
        """Pins the data-URI prefix on its own.

        A data URI is punctuated and far under the length floor, so both other
        conditions reject it; without the prefix check an inlined image would
        reach the manifest verbatim.
        """
        data_uri = "data:image/png;base64," + png_b64()
        self.assertLess(len(data_uri), skill_run.BASE64_VALUE_MIN_LENGTH)

        self.assertTrue(skill_run.looks_like_base64_payload(data_uri))

    def test_long_punctuation_free_prose_survives_the_scrub(self) -> None:
        """The elision must not destroy the reasoning it sits next to.

        A character-class check alone matched any long string of letters, digits
        and whitespace, so a terse enumerated stage description with no
        punctuation was replaced by the elision marker and lost.
        """
        prose = "the model draws a box then an arrow then another box " * 12
        self.assertGreater(len(prose), skill_run.BASE64_VALUE_MIN_LENGTH)

        self.assertFalse(skill_run.looks_like_base64_payload(prose))

        result = full_result("skill_candidate_0")
        result["target_diagram_critic_suggestions0"] = prose
        trace = skill_run.build_candidate_trace(result, "demo_full")

        critic0 = next(stage for stage in trace if stage["name"] == "Critic Round 0")
        self.assertEqual(critic0["suggestions"], prose)
        self.assertNotIn(skill_run.ELIDED_PAYLOAD, json.dumps(trace))

    def test_a_line_wrapped_payload_is_still_elided(self) -> None:
        """Providers wrap base64 on newlines, so newlines cannot disqualify it."""
        blob = long_base64_blob()
        wrapped = "\n".join(blob[i:i + 76] for i in range(0, len(blob), 76))

        self.assertTrue(skill_run.looks_like_base64_payload(wrapped))


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
    """The manifest claims to pin the commit the run executed at.

    Asserting the shape of the commit only when it is not None, and asserting
    the dirty flag is a bool, passes in exactly the two cases worth catching: a
    commit lookup that silently returns nothing, and a dirty flag hardcoded
    clean. A throwaway repository makes both values knowable.
    """

    def make_repo(self) -> Path:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        self.git(tmp, "init", "-q")
        self.git(tmp, "config", "user.email", "test@example.invalid")
        self.git(tmp, "config", "user.name", "PaperBanana Tests")
        (tmp / "tracked.txt").write_text("original\n", encoding="utf-8")
        self.git(tmp, "add", "tracked.txt")
        self.git(tmp, "commit", "-q", "-m", "initial")
        return tmp

    @staticmethod
    def git(repo: Path, *argv) -> str:
        # hooksPath is neutralised because a developer's *global* hooks would
        # otherwise run against this throwaway repository; a secret-scanning
        # pre-commit hook on this machine blocks indefinitely when its stdout is
        # a pipe, which would wedge the whole suite rather than fail it.
        completed = subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", *argv],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return completed.stdout.strip()

    def record_for(self, repo: Path) -> dict:
        with unittest.mock.patch.object(skill_run, "PROJECT_ROOT", repo):
            return skill_run.repo_commit_record()

    def test_the_recorded_commit_is_the_commit_head_actually_points_at(self) -> None:
        repo = self.make_repo()
        expected = self.git(repo, "rev-parse", "HEAD")

        record = self.record_for(repo)

        self.assertEqual(record["commit"], expected)
        self.assertRegex(record["commit"], r"^[0-9a-f]{40}$")

    def test_the_dirty_flag_follows_the_state_of_the_checkout(self) -> None:
        repo = self.make_repo()

        self.assertFalse(self.record_for(repo)["dirty"])

        (repo / "tracked.txt").write_text("edited after the commit\n", encoding="utf-8")

        self.assertTrue(self.record_for(repo)["dirty"])

    def test_an_untracked_file_also_counts_as_dirty(self) -> None:
        repo = self.make_repo()
        (repo / "scratch.txt").write_text("not committed\n", encoding="utf-8")

        self.assertTrue(self.record_for(repo)["dirty"])

    def test_a_checkout_that_is_not_a_repository_records_no_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            record = self.record_for(Path(tmp))

        self.assertIsNone(record["commit"])
        self.assertFalse(record["dirty"])


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
            # What was asked for, fixed at seeding time and never re-read off the
            # entries dict, which record_failure can grow.
            candidates_requested=len(data_list),
            content="method text",
            resolved_models={
                "main_model_name": "gemini-3.1-pro-preview",
                "image_gen_model_name": "gemini-3.1-flash-image-preview",
            },
            image_gen_backend="gemini",
            retrieval={
                "setting": "auto",
                "top10_references_count": 0,
                "retrieved_examples_count": 0,
                "top10_references": [],
            },
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
        # Matched against the content it claims to pin, not merely well-formed:
        # hashing the wrong bytes still yields a plausible 64-char digest.
        self.assertEqual(
            manifest["input"]["content_sha256"],
            hashlib.sha256(b"method text").hexdigest(),
        )
        self.assertEqual(manifest["input"]["content_chars"], len("method text"))

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
        """Payload-sized, so the size assertion is not vacuous.

        Toy 8x8 PNGs total ~12KB across ten candidates, three orders of magnitude
        under the threshold: a regression that copied every payload verbatim would
        still have passed. The description fields are the channel that reaches the
        manifest verbatim, so each carries a realistic payload here.
        """
        results = []
        payload_bytes = 0
        for i in range(10):
            result = full_result(f"skill_candidate_{i}", critic_rounds=3)
            # A stage image's worth of base64, parked on the text fields the
            # manifest copies. Only the value-based scrub can catch these.
            for key in ("target_diagram_desc0", "target_diagram_stylist_desc0"):
                result[key] = long_base64_blob(150_000)
                payload_bytes += len(result[key])
            results.append(result)

        self.assertGreater(payload_bytes, 3_000_000)

        _, manifest, output_path = self.build(results, num_candidates=10)

        path = skill_run.write_manifest(skill_run.manifest_path_for(output_path), manifest)

        self.assertTrue(path.exists())
        # The UI writes roughly 210-234MB per run.
        self.assertLess(path.stat().st_size, 1_000_000)

        def assert_no_payload(node, where="manifest"):
            if isinstance(node, dict):
                for key, value in node.items():
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


class OutputPreflightTests(unittest.TestCase):
    """An unusable --output must cost seconds, not a paid run.

    Nothing touched the filesystem until the first image was saved, which on
    this CLI is after the whole 10-30 minute batch has finished computing.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def read_only_dir(self) -> Path:
        readonly = self.tmp / "readonly"
        readonly.mkdir()
        readonly.chmod(0o500)
        self.addCleanup(readonly.chmod, 0o700)
        return readonly

    def test_a_writable_destination_is_created_and_left_clean(self) -> None:
        target = self.tmp / "nested" / "deeper" / "figure.png"

        with contextlib.redirect_stderr(io.StringIO()):
            resolved = skill_run.preflight_output_path(target)

        self.assertEqual(resolved, target)
        self.assertTrue(target.parent.is_dir())
        # The write probe is not left behind next to the images.
        self.assertEqual(list(target.parent.iterdir()), [])

    def test_an_existing_directory_with_no_write_bit_is_caught(self) -> None:
        """mkdir alone is not proof: it succeeds on a directory it cannot write."""
        readonly = self.read_only_dir()

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as caught:
                skill_run.preflight_output_path(readonly / "figure.png")

        self.assertEqual(caught.exception.code, 2)
        self.assertIn("not writable", err.getvalue())

    def test_a_destination_whose_parent_cannot_be_created_is_caught(self) -> None:
        readonly = self.read_only_dir()

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as caught:
                skill_run.preflight_output_path(readonly / "figs" / "figure.png")

        self.assertEqual(caught.exception.code, 2)
        self.assertIn("not writable", err.getvalue())


class ManifestFallbackTests(unittest.TestCase):
    """The manifest must not be lost with the destination that lost the run.

    write_manifest targets the same directory whose unwritability is the most
    likely reason the run failed, so a bare write there and nowhere else means
    the operator gets a traceback and no record at all.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_an_unwritable_destination_falls_back_to_a_fresh_run_directory(self) -> None:
        readonly = self.tmp / "readonly"
        readonly.mkdir()
        readonly.chmod(0o500)
        self.addCleanup(readonly.chmod, 0o700)
        fallback_base = self.tmp / "skill_runs"
        manifest = {"run": {"status": "partial"}, "candidates": []}

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            with unittest.mock.patch.object(
                skill_run, "DEFAULT_RUN_BASE_DIR", fallback_base
            ):
                path = skill_run.write_manifest(
                    readonly / "figs" / "output.manifest.json", manifest
                )

        self.assertIsNotNone(path, "the run record was lost")
        self.assertTrue(path.exists())
        self.assertIn(fallback_base, path.parents)
        self.assertEqual(
            json.loads(path.read_text(encoding="utf-8"))["run"]["status"], "partial"
        )
        self.assertIn("falling back", err.getvalue())

    def test_a_writable_destination_is_still_used_verbatim(self) -> None:
        target = self.tmp / "figs" / "output.manifest.json"

        with contextlib.redirect_stderr(io.StringIO()):
            path = skill_run.write_manifest(target, {"run": {"status": "complete"}})

        self.assertEqual(path, target)


class UnattributedFailureTests(unittest.TestCase):
    """A failure the processor could not pin on a candidate is not a candidate.

    paperviz_processor yields ``filename: None`` when it cannot attribute a raised
    exception (the ``failed_index is None`` fallback), so run.py has to fold in a
    record that belongs to no requested candidate.
    """

    def test_it_does_not_inflate_the_count_of_what_was_requested(self) -> None:
        data_list = [{"filename": f"skill_candidate_{i}"} for i in range(10)]
        entries = skill_run.seed_manifest_entries(data_list)
        requested = len(entries)

        with contextlib.redirect_stderr(io.StringIO()):
            skill_run.record_failure(None, entries, RuntimeError("unattributable"))

        self.assertEqual(len(entries), 11)
        manifest = skill_run.build_manifest(
            args=args_namespace(num_candidates=10),
            additional_info={},
            entries=entries,
            candidates_requested=requested,
            content="method text",
            resolved_models={},
            image_gen_backend="gemini",
            retrieval={},
            started_at="2026-08-04T10:15:00Z",
            finished_at="2026-08-04T10:38:00Z",
        )

        self.assertEqual(manifest["run"]["candidates_requested"], 10)
        self.assertEqual(len(manifest["candidates"]), 11)

    def test_two_unattributed_failures_do_not_overwrite_each_other(self) -> None:
        entries = skill_run.seed_manifest_entries([{"filename": "skill_candidate_0"}])

        with contextlib.redirect_stderr(io.StringIO()):
            skill_run.record_failure(None, entries, RuntimeError("first boom"))
            skill_run.record_failure(None, entries, RuntimeError("second boom"))

        errors = [e["error"] for e in entries.values() if e["status"] == "failed"]
        self.assertEqual(len(errors), 2)
        self.assertTrue(any("first boom" in e for e in errors))
        self.assertTrue(any("second boom" in e for e in errors))

    def test_its_key_cannot_collide_with_a_seeded_identity_less_candidate(self) -> None:
        """seed_manifest_entries names those 'unidentified_candidate_<index>'."""
        entries = skill_run.seed_manifest_entries([{"filename": None}, {"filename": ""}])
        self.assertIn("unidentified_candidate_0", entries)

        with contextlib.redirect_stderr(io.StringIO()):
            skill_run.record_failure(None, entries, RuntimeError("boom"))

        self.assertEqual(len(entries), 3)
        self.assertEqual(entries["unidentified_candidate_0"]["status"], "missing")


class ErrorTextRedactionTests(unittest.TestCase):
    """candidates[].error is third-party text, not allowlisted values.

    Provider SDKs are known to echo the key they were called with, and the
    manifest is meant to be kept indefinitely.
    """

    def test_a_key_shaped_secret_in_an_exception_never_reaches_the_manifest(self) -> None:
        entries = skill_run.seed_manifest_entries([{"filename": "skill_candidate_0"}])
        error = RuntimeError(
            f"400 INVALID_ARGUMENT: API key not valid: {SENTINEL_KEY} (key=sk-or-v1-abcdefghijklmnop1234)"
        )

        with contextlib.redirect_stderr(io.StringIO()) as err:
            skill_run.record_failure("skill_candidate_0", entries, error)

        manifest = skill_run.build_manifest(
            args=args_namespace(num_candidates=1),
            additional_info={},
            entries=entries,
            candidates_requested=1,
            content="method text",
            resolved_models={},
            image_gen_backend="gemini",
            retrieval={},
            started_at="2026-08-04T10:15:00Z",
            finished_at="2026-08-04T10:38:00Z",
        )

        serialized = json.dumps(manifest)
        self.assertNotIn(SENTINEL_KEY, serialized)
        self.assertNotIn("sk-or-v1-abcdefghijklmnop1234", serialized)
        self.assertNotIn(SENTINEL_KEY, err.getvalue())
        # The diagnosis survives the redaction.
        self.assertIn("INVALID_ARGUMENT", entries["skill_candidate_0"]["error"])

    def test_the_configured_environment_key_is_redacted_by_value(self) -> None:
        """Catches a key whose shape no pattern anticipates."""
        secret = "totally-unpatterned-key-value-9182"
        with unittest.mock.patch.dict(os.environ, {"GOOGLE_API_KEY": secret}):
            redacted = skill_run.redact_credentials(f"401 Unauthorized for key {secret}")

        self.assertNotIn(secret, redacted)
        self.assertIn(skill_run.REDACTED_CREDENTIAL, redacted)

    def test_a_key_from_model_config_yaml_is_redacted_by_value(self) -> None:
        """R7a names configs/model_config.yaml explicitly as a surface.

        Uses a deliberately non-key-shaped sentinel. With the shaped SENTINEL_KEY
        this test could not fail: ``_KEY_SHAPED_RE`` caught it regardless, so the
        config-sourcing branch could be deleted outright and the test stayed green.
        """
        # Guard the guard. If the shape regex can match the sentinel, this test
        # is vacuous again and should fail loudly rather than pass for free.
        self.assertIsNone(
            skill_run._KEY_SHAPED_RE.search(UNSHAPED_CONFIG_SENTINEL),
            "sentinel must not be key-shaped, or the shape regex redacts it and "
            "the model_config.yaml branch goes untested",
        )

        with tempfile.TemporaryDirectory() as tmp:
            configs = Path(tmp) / "configs"
            configs.mkdir()
            (configs / "model_config.yaml").write_text(
                f'api_keys:\n  google_api_key: "{UNSHAPED_CONFIG_SENTINEL}"\n',
                encoding="utf-8",
            )
            with unittest.mock.patch.object(skill_run, "PROJECT_ROOT", Path(tmp)):
                redacted = skill_run.redact_credentials(
                    f"PermissionDenied: key {UNSHAPED_CONFIG_SENTINEL} lacks access"
                )

        self.assertNotIn(UNSHAPED_CONFIG_SENTINEL, redacted)
        self.assertIn(skill_run.REDACTED_CREDENTIAL, redacted)

    def test_an_unshaped_key_survives_when_it_is_not_in_the_config(self) -> None:
        """Pins that the previous test passes via config sourcing, not the regex.

        Same sentinel, same call, but no config declaring it: it must come back
        untouched. If this ever starts redacting, the sibling test has stopped
        proving what it claims.
        """
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "configs").mkdir()
            with unittest.mock.patch.object(skill_run, "PROJECT_ROOT", Path(tmp)):
                with unittest.mock.patch.dict(os.environ, {}, clear=True):
                    redacted = skill_run.redact_credentials(
                        f"PermissionDenied: key {UNSHAPED_CONFIG_SENTINEL} lacks access"
                    )

        self.assertIn(UNSHAPED_CONFIG_SENTINEL, redacted)

    def test_a_malformed_config_file_does_not_break_redaction(self) -> None:
        """A manifest must still be written when the config cannot be parsed."""
        with tempfile.TemporaryDirectory() as tmp:
            configs = Path(tmp) / "configs"
            configs.mkdir()
            (configs / "model_config.yaml").write_text("{[not: yaml", encoding="utf-8")
            with unittest.mock.patch.object(skill_run, "PROJECT_ROOT", Path(tmp)):
                self.assertEqual(skill_run.redact_credentials("plain error"), "plain error")

    def test_a_payload_echoed_back_in_an_error_is_elided_from_the_manifest(self) -> None:
        """The live call site for the scrub over the manifest entries.

        Everything else in an entry is written by this module: statuses, paths,
        dimensions, and a trace that build_candidate_trace already scrubbed.
        ``error`` is the one field whose content comes from a third-party SDK,
        and providers do echo the image they rejected. Without the scrub in
        build_manifest that payload is stored verbatim in a record meant to be
        kept indefinitely.
        """
        # Deterministic, not os.urandom: this blob is the one that goes through
        # redact_credentials, and 200KB of random base64 has a ~1.6% chance of
        # containing a literal "AIza". _KEY_SHAPED_RE would then punch a
        # "<redacted: credential>" into the middle of it, the result would no
        # longer look like a payload, and this test would fail one run in sixty
        # for a reason that has nothing to do with what it asserts.
        blob = ("PaperBananaRejectedImageBytes" * 9_000)[:200_000]
        self.assertIsNone(skill_run._KEY_SHAPED_RE.search(blob))
        self.assertTrue(skill_run.looks_like_base64_payload(blob))

        entries = skill_run.seed_manifest_entries([{"filename": "skill_candidate_0"}])

        with contextlib.redirect_stderr(io.StringIO()):
            skill_run.record_failure(
                "skill_candidate_0",
                entries,
                RuntimeError(blob),
            )

        # The entry itself still holds it; only the manifest view is scrubbed.
        self.assertIn(blob, entries["skill_candidate_0"]["error"])

        manifest = skill_run.build_manifest(
            args=args_namespace(num_candidates=1),
            additional_info={},
            entries=entries,
            candidates_requested=1,
            content="method text",
            resolved_models={},
            image_gen_backend="gemini",
            retrieval={},
            started_at="2026-08-04T10:15:00Z",
            finished_at="2026-08-04T10:38:00Z",
        )

        serialized = json.dumps(manifest)
        self.assertNotIn(blob, serialized)
        self.assertIn(skill_run.ELIDED_PAYLOAD, serialized)

    def test_ordinary_error_text_is_left_alone(self) -> None:
        self.assertEqual(
            skill_run.redact_credentials("RuntimeError: visualizer exhausted retries"),
            "RuntimeError: visualizer exhausted retries",
        )


class RetrievalRecordTests(unittest.TestCase):
    """Retrieval runs once per batch, and is recorded once per run.

    The record reads data_list[0] because retriever_agent mutates that dict in
    place; with --num-candidates 1 there is nothing else it could read.
    """

    def test_it_records_the_setting_and_the_shared_retrieval_result(self) -> None:
        data_list = [
            {"filename": "skill_candidate_0",
             "top10_references": [{"paper": "a"}, {"paper": "b"}],
             "retrieved_examples": [{"image": "x"}]},
            {"filename": "skill_candidate_1"},
        ]

        record = skill_run.build_retrieval_record(data_list, "auto")

        self.assertEqual(record["setting"], "auto")
        self.assertEqual(record["top10_references_count"], 2)
        self.assertEqual(record["retrieved_examples_count"], 1)
        self.assertEqual(record["top10_references"], [{"paper": "a"}, {"paper": "b"}])

    def test_retrieval_disabled_is_recorded_as_such_rather_than_omitted(self) -> None:
        record = skill_run.build_retrieval_record([{"filename": "skill_candidate_0"}], "none")

        self.assertEqual(record["setting"], "none")
        self.assertEqual(record["top10_references_count"], 0)
        self.assertEqual(record["retrieved_examples_count"], 0)

    def test_an_empty_data_list_does_not_raise(self) -> None:
        self.assertEqual(skill_run.build_retrieval_record([], "auto")["setting"], "auto")

    def test_a_payload_smuggled_into_a_reference_is_elided(self) -> None:
        blob = long_base64_blob()
        record = skill_run.build_retrieval_record(
            [{"top10_references": [{"caption": blob}]}], "auto"
        )

        self.assertNotIn(blob, json.dumps(record))


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
