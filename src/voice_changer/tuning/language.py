"""Language-aware conversion profiles (Vietnamese and English).

Vietnamese is a tonal language: the six lexical tones are carried by the F0
contour, and final consonants (-p, -t, -c, -ch, -m, -n, -ng, -nh) mark syllable
endings. English is non-tonal. Each profile tunes the parameters that affect
those cues, leaving pitch transposition unchanged: a uniform semitone shift
preserves the shape of a tone contour.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from voice_changer.conversion.base import ConversionConfig


@dataclass(frozen=True)
class LanguageProfile:
    """Conversion parameters tuned for one spoken language."""

    code: str
    name: str
    filter_radius: int
    protect_rate: float
    index_rate: float
    rationale: str


LANGUAGE_PROFILES = {
    "en": LanguageProfile(
        code="en",
        name="English",
        filter_radius=3,
        protect_rate=0.33,
        index_rate=0.75,
        rationale=(
            "Baseline values from Milestone 3A.1. Non-tonal speech tolerates the "
            "median F0 smoothing and the index retrieval blend at these levels."
        ),
    ),
    "vi": LanguageProfile(
        code="vi",
        name="Vietnamese",
        # RVC applies the F0 median filter only when radius > 2. A radius of 2
        # keeps the rising and falling edges of the thanh sắc, nặng, and hỏi tones.
        filter_radius=2,
        # Higher protection keeps the unvoiced and nasal endings and the breathy
        # onsets that separate Vietnamese syllables.
        protect_rate=0.40,
        # The retrieval index is built from English speech (VCTK). A lower blend
        # reduces the English phonetic colouring applied to Vietnamese vowels.
        index_rate=0.50,
        rationale=(
            "Tone contours are carried by F0, so the median smoothing is reduced. "
            "The index is English-trained, so its blend is reduced. Consonant "
            "protection is raised for syllable endings."
        ),
    ),
}

DEFAULT_LANGUAGE = "en"


def get_language_profile(language: str) -> LanguageProfile:
    """Return the profile for a language code such as 'vi' or 'en'.

    Raises:
        ValueError: If the language is not supported.
    """
    code = language.strip().lower()
    if code not in LANGUAGE_PROFILES:
        supported = ", ".join(sorted(LANGUAGE_PROFILES))
        raise ValueError(f"Unsupported language '{language}'. Supported: {supported}.")
    return LANGUAGE_PROFILES[code]


def apply_language_profile(
    config: ConversionConfig,
    language: str,
    index_rate: float | None = None,
) -> ConversionConfig:
    """Return a copy of `config` with the language profile applied.

    The input config is not modified. Pitch shift and the F0 method are kept.
    An explicit `index_rate` (for example, from a CLI flag) overrides the profile.
    """
    profile = get_language_profile(language)
    return replace(
        config,
        filter_radius=profile.filter_radius,
        protect_rate=profile.protect_rate,
        index_rate=profile.index_rate if index_rate is None else index_rate,
    )
