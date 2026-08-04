"""Unit 1: figure-size passthrough, calibrated defaults, dimension recording.

Offline only. Nothing here touches a model provider.
"""

import contextlib
import io
import unittest

from skill import run as skill_run
from utils.legacy_generation_options import image_size_from_data


BASE_ARGV = ["--content", "method text", "--caption", "Figure 1: overview"]


def parse(*extra):
    return skill_run.build_parser().parse_args(BASE_ARGV + list(extra))


class FigureSizeArgumentTests(unittest.TestCase):
    def test_omitted_flags_default_to_calibrated_configuration(self) -> None:
        """The guard-trips case: a bare invocation must not land on 1k/21:9."""
        args = parse()

        self.assertEqual(args.figure_size, "14-17cm")
        self.assertEqual(args.aspect_ratio, "16:9")

        info = skill_run.build_additional_info(args)
        self.assertEqual(info["figure_size"], "14-17cm")
        self.assertEqual(info["image_size"], "4k")
        self.assertEqual(info["rounded_ratio"], "16:9")
        self.assertNotEqual(info["image_size"], "1k")
        self.assertNotEqual(info["rounded_ratio"], "21:9")

    def test_explicit_figure_size_flows_into_additional_info(self) -> None:
        args = parse("--figure-size", "14-17cm")
        info = skill_run.build_additional_info(args)

        self.assertEqual(info["figure_size"], "14-17cm")
        self.assertEqual(info["image_size"], "4k")

    def test_every_choice_maps_to_its_documented_tier(self) -> None:
        expected = {
            "1-3cm": "1k",
            "4-6cm": "1k",
            "7-9cm": "2k",
            "10-13cm": "2k",
            "14-17cm": "4k",
        }
        for figure_size, tier in expected.items():
            with self.subTest(figure_size=figure_size):
                args = parse("--figure-size", figure_size)
                self.assertEqual(skill_run.build_additional_info(args)["image_size"], tier)

    def test_unaccepted_figure_size_is_rejected_by_argparse(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse("--figure-size", "20cm")

    def test_aspect_ratio_choices_are_unchanged(self) -> None:
        for ratio in ("21:9", "16:9", "3:2"):
            with self.subTest(ratio=ratio):
                self.assertEqual(parse("--aspect-ratio", ratio).aspect_ratio, ratio)

    def test_additional_info_reaches_image_size_from_data_boundary(self) -> None:
        """R2 integration seam: assert at the consumer, not only at the helper."""
        args = parse()
        data = {"additional_info": skill_run.build_additional_info(args)}

        self.assertEqual(image_size_from_data(data), "4k")

        args_small = parse("--figure-size", "1-3cm")
        data_small = {"additional_info": skill_run.build_additional_info(args_small)}
        self.assertEqual(image_size_from_data(data_small), "1k")


class DimensionReportTests(unittest.TestCase):
    def test_calibrated_4k_measurement_is_not_flagged(self) -> None:
        """The one real calibration point: 4k at 16:9 measured 5504x3072."""
        desc = skill_run.describe_image_dimensions("14-17cm", "4k", "16:9", 5504, 3072)

        self.assertEqual(desc["width"], 5504)
        self.assertEqual(desc["height"], 3072)
        self.assertEqual(desc["image_size"], "4k")
        self.assertFalse(desc["size_shortfall"])
        self.assertFalse(desc["aspect_ratio_drift"])
        # 1.7917 vs the nominal 1.7778 must not raise a false alarm.
        self.assertAlmostEqual(desc["actual_ratio"], 1.7917, places=3)

    def test_material_shortfall_against_requested_tier_is_flagged(self) -> None:
        desc = skill_run.describe_image_dimensions("14-17cm", "4k", "16:9", 1024, 576)

        self.assertTrue(desc["size_shortfall"])

    def test_2k_and_1k_tiers_accept_their_own_output(self) -> None:
        self.assertFalse(
            skill_run.describe_image_dimensions("7-9cm", "2k", "16:9", 2752, 1536)["size_shortfall"]
        )
        self.assertFalse(
            skill_run.describe_image_dimensions("1-3cm", "1k", "16:9", 1376, 768)["size_shortfall"]
        )

    def test_aspect_ratio_drift_is_flagged_only_when_material(self) -> None:
        square = skill_run.describe_image_dimensions("14-17cm", "4k", "16:9", 4096, 4096)
        self.assertTrue(square["aspect_ratio_drift"])

        close = skill_run.describe_image_dimensions("14-17cm", "4k", "3:2", 5504, 3648)
        self.assertFalse(close["aspect_ratio_drift"])

    def test_report_names_requested_tier_actual_dimensions_and_shortfall(self) -> None:
        line = skill_run.format_dimension_report(
            skill_run.describe_image_dimensions("14-17cm", "4k", "16:9", 1024, 576)
        )

        self.assertIn("14-17cm", line)
        self.assertIn("4k", line)
        self.assertIn("1024x576", line)
        self.assertIn("SHORTFALL", line)

        ok_line = skill_run.format_dimension_report(
            skill_run.describe_image_dimensions("14-17cm", "4k", "16:9", 5504, 3072)
        )
        self.assertIn("5504x3072", ok_line)
        self.assertNotIn("SHORTFALL", ok_line)

    def test_degenerate_dimensions_do_not_raise(self) -> None:
        desc = skill_run.describe_image_dimensions("14-17cm", "4k", "16:9", 0, 0)
        self.assertTrue(desc["size_shortfall"])
        self.assertIsNone(desc["actual_ratio"])


if __name__ == "__main__":
    unittest.main()
