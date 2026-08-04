"""Unit 3: runs default to a timestamped directory.

A repeat invocation must not destroy a prior run's images or manifest.
Offline only.
"""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from skill import run as skill_run

from tests.test_skill_run_candidate_identity import make_result


BASE_ARGV = ["--content", "method text", "--caption", "Figure 1: overview"]


def parse(*extra):
    return skill_run.build_parser().parse_args(BASE_ARGV + list(extra))


class OutputArgumentTests(unittest.TestCase):
    def test_omitted_output_is_distinguishable_from_an_explicit_one(self) -> None:
        """argparse cannot tell 'output.png' from a literal default of the same name."""
        self.assertIsNone(parse().output)
        self.assertEqual(parse("--output", "output.png").output, "output.png")


class RunDirectoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name) / "skill_runs"
        self.addCleanup(self._tmp.cleanup)

    def test_timestamp_defaults_to_wall_clock_without_raising(self) -> None:
        """The omitted-timestamp path is the one production actually takes."""
        run_dir = skill_run.create_run_directory(self.base)

        self.assertTrue(run_dir.is_dir())
        self.assertTrue(run_dir.name.startswith(skill_run.RUN_DIR_PREFIX))
        self.assertGreater(len(run_dir.name), len(skill_run.RUN_DIR_PREFIX))

    def test_directory_is_created_when_absent(self) -> None:
        self.assertFalse(self.base.exists())

        run_dir = skill_run.create_run_directory(self.base, timestamp="20260804_101500")

        self.assertTrue(run_dir.is_dir())
        self.assertEqual(run_dir.parent, self.base)
        self.assertIn("20260804_101500", run_dir.name)

    def test_two_runs_in_the_same_second_do_not_merge(self) -> None:
        first = skill_run.create_run_directory(self.base, timestamp="20260804_101500")
        second = skill_run.create_run_directory(self.base, timestamp="20260804_101500")
        third = skill_run.create_run_directory(self.base, timestamp="20260804_101500")

        self.assertNotEqual(first, second)
        self.assertNotEqual(second, third)
        self.assertNotEqual(first, third)
        for path in (first, second, third):
            self.assertTrue(path.is_dir())

    def test_distinct_timestamps_produce_distinct_directories(self) -> None:
        first = skill_run.create_run_directory(self.base, timestamp="20260804_101500")
        second = skill_run.create_run_directory(self.base, timestamp="20260804_101501")

        self.assertNotEqual(first, second)

    def test_creation_is_idempotent_against_a_preexisting_base(self) -> None:
        self.base.mkdir(parents=True)
        (self.base / "unrelated.txt").write_text("keep me", encoding="utf-8")

        skill_run.create_run_directory(self.base, timestamp="20260804_101500")

        self.assertTrue((self.base / "unrelated.txt").exists())


class OutputPlacementTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name) / "skill_runs"
        self.addCleanup(self._tmp.cleanup)

    def save(self, results, output_path, num_candidates, output_explicit):
        with contextlib.redirect_stderr(io.StringIO()):
            return skill_run.save_result_images(
                results,
                exp_mode="demo_full",
                output_path=output_path,
                num_candidates=num_candidates,
                output_explicit=output_explicit,
                figure_size="14-17cm",
                image_size="4k",
                aspect_ratio="16:9",
            )

    def test_omitted_output_writes_identity_named_files_into_the_run_dir(self) -> None:
        run_dir = skill_run.create_run_directory(self.base, timestamp="20260804_101500")

        saved = self.save(
            [make_result("skill_candidate_0")],
            skill_run.default_output_path(run_dir),
            num_candidates=1,
            output_explicit=False,
        )

        self.assertEqual(
            Path(saved[0]["image_path"]), run_dir / "skill_candidate_0.png"
        )

    def test_explicit_output_is_honoured_verbatim(self) -> None:
        target = Path(self._tmp.name) / "figs" / "architecture.png"

        saved = self.save(
            [make_result("skill_candidate_0")],
            target,
            num_candidates=1,
            output_explicit=True,
        )

        self.assertEqual(Path(saved[0]["image_path"]), target)

    def test_two_consecutive_default_runs_both_survive(self) -> None:
        first_dir = skill_run.create_run_directory(self.base, timestamp="20260804_101500")
        self.save(
            [make_result("skill_candidate_0"), make_result("skill_candidate_1")],
            skill_run.default_output_path(first_dir),
            num_candidates=2,
            output_explicit=False,
        )
        second_dir = skill_run.create_run_directory(self.base, timestamp="20260804_101500")
        self.save(
            [make_result("skill_candidate_0"), make_result("skill_candidate_1")],
            skill_run.default_output_path(second_dir),
            num_candidates=2,
            output_explicit=False,
        )

        self.assertNotEqual(first_dir, second_dir)
        for directory in (first_dir, second_dir):
            names = sorted(p.name for p in directory.iterdir())
            self.assertEqual(names, ["skill_candidate_0.png", "skill_candidate_1.png"])


if __name__ == "__main__":
    unittest.main()
