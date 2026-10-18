"""Voice conversion presets and tuning parameters.

Defines parameter profiles for male-to-female voice conversion.
Actual optimal values will be calibrated during Milestone 5 (Feminine Quality Tuning).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from voice_changer.conversion.base import ConversionConfig


@dataclass(frozen=True)
class VoicePreset:
    """Named preset profile holding conversion parameters."""

    name: str
    description: str
    config: ConversionConfig


def get_default_presets() -> Dict[str, VoicePreset]:
    """Return initial baseline presets for male-to-female conversion testing.

    Note: These presets are baseline starting points and will be evaluated
    and refined against the quality test plan in Milestone 5.
    """
    return {
        "natural_feminine": VoicePreset(
            name="Natural Feminine (Baseline)",
            description="Balanced pitch shift and index retrieval for standard male vocal range.",
            config=ConversionConfig(
                pitch_shift=12.0,
                f0_method="rmvpe",
                index_rate=0.75,
                filter_radius=3,
                protect_rate=0.33,
            ),
        ),
        "high_feminine": VoicePreset(
            name="Higher Pitch Feminine (Baseline)",
            description="Higher transposition for deeper male voices.",
            config=ConversionConfig(
                pitch_shift=14.0,
                f0_method="rmvpe",
                index_rate=0.80,
                filter_radius=3,
                protect_rate=0.30,
            ),
        ),
        "soft_feminine": VoicePreset(
            name="Soft / Breathy Feminine (Baseline)",
            description="Higher protect rate to retain natural breathiness and soft phonemes.",
            config=ConversionConfig(
                pitch_shift=12.0,
                f0_method="rmvpe",
                index_rate=0.65,
                filter_radius=3,
                protect_rate=0.45,
            ),
        ),
    }
