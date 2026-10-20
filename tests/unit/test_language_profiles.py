"""Unit tests for Vietnamese and English conversion profiles."""

import sys
from pathlib import Path

# Ensure src/ is on sys.path for test execution without package installation
src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import unittest

from voice_changer.conversion.base import ConversionConfig
from voice_changer.tuning.language import (
    DEFAULT_LANGUAGE,
    LANGUAGE_PROFILES,
    apply_language_profile,
    get_language_profile,
)


class TestLanguageProfiles(unittest.TestCase):
    def test_default_language_is_english(self):
        self.assertEqual(DEFAULT_LANGUAGE, "en")

    def test_english_profile_matches_milestone_baseline(self):
        cfg = apply_language_profile(ConversionConfig(), "en")
        self.assertEqual(cfg.filter_radius, 3)
        self.assertEqual(cfg.protect_rate, 0.33)
        self.assertEqual(cfg.index_rate, 0.75)

    def test_vietnamese_profile_preserves_tone_contour(self):
        cfg = apply_language_profile(ConversionConfig(), "vi")
        # RVC applies the F0 median filter only when radius > 2
        self.assertLessEqual(cfg.filter_radius, 2)

    def test_vietnamese_profile_protects_consonants_more_than_english(self):
        vi = apply_language_profile(ConversionConfig(), "vi")
        en = apply_language_profile(ConversionConfig(), "en")
        self.assertGreater(vi.protect_rate, en.protect_rate)

    def test_vietnamese_profile_reduces_english_index_blend(self):
        vi = apply_language_profile(ConversionConfig(), "vi")
        en = apply_language_profile(ConversionConfig(), "en")
        self.assertLess(vi.index_rate, en.index_rate)

    def test_pitch_shift_and_f0_method_are_preserved(self):
        base = ConversionConfig(pitch_shift=10.0, f0_method="rmvpe")
        for lang in LANGUAGE_PROFILES:
            cfg = apply_language_profile(base, lang)
            self.assertEqual(cfg.pitch_shift, 10.0)
            self.assertEqual(cfg.f0_method, "rmvpe")

    def test_input_config_is_not_mutated(self):
        base = ConversionConfig()
        apply_language_profile(base, "vi")
        self.assertEqual(base, ConversionConfig())

    def test_explicit_index_rate_overrides_profile(self):
        cfg = apply_language_profile(ConversionConfig(), "vi", index_rate=0.9)
        self.assertEqual(cfg.index_rate, 0.9)

    def test_language_code_is_case_and_whitespace_insensitive(self):
        self.assertEqual(get_language_profile("  VI ").code, "vi")

    def test_unsupported_language_raises_value_error(self):
        with self.assertRaises(ValueError):
            get_language_profile("fr")

    def test_profile_fields_are_frozen(self):
        profile = get_language_profile("vi")
        with self.assertRaises(Exception):
            profile.filter_radius = 5  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
