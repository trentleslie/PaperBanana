"""The CLI's figure-size wiring.

``generation_additional_info`` and ``image_size_for_figure_size`` are already
covered in ``test_legacy_generation_options``. What was untested is whether
``skill/run.py`` calls them: it built ``additional_info`` inline as
``{"rounded_ratio": ...}``, so ``figure_size`` never reached the pipeline and
``image_size_from_data`` fell back to its default tier no matter what the caller
asked for. These tests pin the call path rather than the helper.

Offline; no network and no model calls.
"""

import unittest

from skill.run import FIGURE_SIZE_CHOICES, build_additional_info, build_parser
from utils.legacy_generation_options import image_size_from_data


BASE_ARGV = ["--content", "method text", "--caption", "Figure 1: overview"]


def parse(*extra):
    return build_parser().parse_args(BASE_ARGV + list(extra))


def additional_info(*extra):
    """Exactly what run() puts on each data dict.

    Calls run.py's own builder rather than re-deriving it here. Re-deriving
    makes these tests pass even with the wiring reverted, which is the very
    defect they exist to catch.
    """
    return build_additional_info(parse(*extra))


class FigureSizeArgumentTests(unittest.TestCase):
    def test_the_flag_accepts_every_documented_tier(self) -> None:
        for choice in FIGURE_SIZE_CHOICES:
            with self.subTest(figure_size=choice):
                self.assertEqual(parse("--figure-size", choice).figure_size, choice)

    def test_an_unknown_size_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            parse("--figure-size", "20cm")

    def test_omitting_the_flag_keeps_the_previous_default(self) -> None:
        self.assertIsNone(parse().figure_size)


class AdditionalInfoWiringTests(unittest.TestCase):
    """The regression itself: a requested figure size must reach the pipeline."""

    def test_a_requested_size_reaches_additional_info(self) -> None:
        info = additional_info("--figure-size", "14-17cm")

        self.assertEqual(info["figure_size"], "14-17cm")
        self.assertEqual(info["image_size"], "4k")

    def test_the_agents_read_the_tier_back_out(self) -> None:
        """Asserted at the boundary the agents actually use."""
        info = additional_info("--figure-size", "7-9cm")

        self.assertEqual(image_size_from_data({"additional_info": info}), "2k")

    def test_omitting_the_flag_leaves_additional_info_unchanged(self) -> None:
        """Purely additive: an existing invocation behaves exactly as before."""
        info = additional_info()

        self.assertEqual(info, {"rounded_ratio": parse().aspect_ratio})


class CallSiteTests(unittest.TestCase):
    """The seam is only useful if the data dict actually goes through it.

    The unit tests above pin what ``build_additional_info`` returns. They cannot
    see a future edit that inlines the dict again at the call site, which is
    precisely how the original defect looked, so this checks the wiring itself.
    """

    def test_the_candidate_dict_is_built_through_the_helper(self) -> None:
        import inspect

        from skill import run as skill_run

        source = inspect.getsource(skill_run.run)

        self.assertIn("build_additional_info(args)", source)
        self.assertNotIn('"rounded_ratio": args.aspect_ratio', source)


if __name__ == "__main__":
    unittest.main()
