"""Unit 2: candidate identity is the source of output filenames.

process_queries_batch yields via asyncio.as_completed, so completion order is
not submission order. Naming from enumerate() mislabels every multi-candidate
run. Offline only.
"""

import base64
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from skill import run as skill_run


def png_b64(width: int = 8, height: int = 8, color=(10, 20, 30)) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def make_result(name, **extra):
    result = {
        "filename": name,
        "target_diagram_critic_desc0_base64_jpg": png_b64(),
    }
    result.update(extra)
    return result


class CandidateIdentityTests(unittest.TestCase):
    def test_identity_comes_from_the_results_own_filename(self) -> None:
        self.assertEqual(
            skill_run.candidate_identity({"filename": "skill_candidate_7"}),
            "skill_candidate_7",
        )

    def test_identity_is_sanitized_for_filesystem_use(self) -> None:
        self.assertEqual(
            skill_run.candidate_identity({"filename": "skill candidate/7"}),
            "skill_candidate_7",
        )

    def test_identity_is_none_when_it_cannot_be_derived(self) -> None:
        self.assertIsNone(skill_run.candidate_identity({}))
        self.assertIsNone(skill_run.candidate_identity({"filename": ""}))
        self.assertIsNone(skill_run.candidate_identity({"filename": "   "}))
        self.assertIsNone(skill_run.candidate_identity({"filename": 3}))
        self.assertIsNone(skill_run.candidate_identity("not a dict"))

    def test_identity_is_stable_across_repeated_calls(self) -> None:
        data = {"filename": "skill_candidate_2"}
        self.assertEqual(
            skill_run.candidate_identity(data), skill_run.candidate_identity(data)
        )


class SaveResultImagesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def save(self, results, output_path, num_candidates):
        return skill_run.save_result_images(
            results,
            exp_mode="demo_full",
            output_path=output_path,
            num_candidates=num_candidates,
            figure_size="14-17cm",
            image_size="4k",
            aspect_ratio="16:9",
        )

    def test_out_of_order_results_are_named_from_their_own_identity(self) -> None:
        """Arrival position 0 holds candidate 3; the file must say candidate 3."""
        results = [
            make_result("skill_candidate_3"),
            make_result("skill_candidate_0"),
            make_result("skill_candidate_1"),
        ]

        saved = self.save(results, self.tmp / "output.png", num_candidates=3)

        by_identity = {entry["identity"]: entry["image_path"] for entry in saved}
        self.assertEqual(set(by_identity), {"skill_candidate_3", "skill_candidate_0", "skill_candidate_1"})
        for identity, path in by_identity.items():
            self.assertIn(identity, Path(path).name)
            self.assertTrue(Path(path).exists())

        # The arrival-order name that the old enumerate() would have produced
        # for candidate 3 must not exist.
        self.assertFalse((self.tmp / "output_0.png").exists())

    def test_single_candidate_with_explicit_output_writes_that_exact_path(self) -> None:
        target = self.tmp / "architecture.png"

        saved = self.save([make_result("skill_candidate_0")], target, num_candidates=1)

        self.assertEqual(len(saved), 1)
        self.assertEqual(Path(saved[0]["image_path"]), target)
        self.assertTrue(target.exists())

    def test_result_without_identity_is_skipped_not_fabricated(self) -> None:
        results = [make_result("skill_candidate_0"), make_result(None)]

        saved = self.save(results, self.tmp / "output.png", num_candidates=2)

        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["identity"], "skill_candidate_0")
        written = sorted(p.name for p in self.tmp.iterdir())
        self.assertEqual(written, ["output_skill_candidate_0.png"])

    def test_result_without_an_image_is_reported_with_no_path(self) -> None:
        results = [{"filename": "skill_candidate_4"}]

        saved = self.save(results, self.tmp / "output.png", num_candidates=1)

        self.assertEqual(len(saved), 1)
        self.assertIsNone(saved[0]["image_path"])
        self.assertEqual(saved[0]["identity"], "skill_candidate_4")

    def test_saved_entry_carries_the_dimension_record(self) -> None:
        saved = self.save(
            [make_result("skill_candidate_0")], self.tmp / "output.png", num_candidates=1
        )

        dimensions = saved[0]["dimensions"]
        self.assertEqual(dimensions["width"], 8)
        self.assertEqual(dimensions["height"], 8)
        self.assertEqual(dimensions["image_size"], "4k")
        self.assertTrue(dimensions["size_shortfall"])

    def test_data_url_prefixed_base64_is_decoded(self) -> None:
        result = make_result("skill_candidate_0")
        result["target_diagram_critic_desc0_base64_jpg"] = (
            "data:image/png;base64," + png_b64()
        )

        saved = self.save([result], self.tmp / "output.png", num_candidates=1)

        self.assertTrue(Path(saved[0]["image_path"]).exists())


if __name__ == "__main__":
    unittest.main()
