"""Abstract conversion interface.

Provides a decoupled protocol separating the real-time audio pipeline
from any concrete voice-conversion backend (RVC or future models).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class ConversionConfig:
    """Parameters governing voice conversion."""

    pitch_shift: float = 12.0
    f0_method: str = "rmvpe"
    index_rate: float = 0.75
    filter_radius: int = 3
    protect_rate: float = 0.33


# F0 methods verified in this project. Applio offers others (crepe, fcpe, ...) that have not been
# tested here and may need extra assets, so they are rejected rather than assumed to work.
SUPPORTED_F0_METHODS = ("rmvpe",)
PITCH_SHIFT_RANGE = (-24.0, 24.0)
FILTER_RADIUS_RANGE = (0, 7)
# Applio disables consonant protection at protect >= 0.5.
PROTECT_RATE_RANGE = (0.0, 0.5)


def validate_conversion_config(config: ConversionConfig) -> list[str]:
    """Return human-readable problems with `config` (empty list if valid)."""
    problems = []

    def _num(name: str, value: object, lo: float, hi: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"{name} must be a number, got {value!r}")
        elif not lo <= value <= hi:
            problems.append(f"{name} must be between {lo} and {hi}, got {value}")

    _num("pitch_shift", config.pitch_shift, *PITCH_SHIFT_RANGE)
    _num("index_rate", config.index_rate, 0.0, 1.0)
    _num("protect_rate", config.protect_rate, *PROTECT_RATE_RANGE)
    if isinstance(config.filter_radius, bool) or not isinstance(config.filter_radius, int):
        problems.append(f"filter_radius must be an integer, got {config.filter_radius!r}")
    else:
        _num("filter_radius", config.filter_radius, *FILTER_RADIUS_RANGE)
    if config.f0_method not in SUPPORTED_F0_METHODS:
        problems.append(f"f0_method must be one of {SUPPORTED_F0_METHODS}, got {config.f0_method!r}")
    return problems


@runtime_checkable
class VoiceConverter(Protocol):
    """Protocol defining the voice conversion engine interface.

    Any conversion engine (RVC, ONNX runtime, etc.) must implement this protocol
    to be usable by the streaming audio pipeline.
    """

    def load_model(self, model_path: str, index_path: Optional[str] = None) -> None:
        """Load model weights and optional feature retrieval index."""
        ...

    def convert(self, audio_chunk: bytes, config: ConversionConfig) -> bytes:
        """Process an audio chunk and return the converted audio bytes."""
        ...

    def convert_chunk(
        self,
        audio_data: Any,
        config: Optional[ConversionConfig] = None,
        input_sr: int = 40000,
        target_sr: Optional[int] = None,
    ) -> Any:
        """Process an in-memory audio chunk (float32 array)."""
        ...

    def reset(self) -> None:
        """Reset internal buffer states (e.g. pitch history, crossfades)."""
        ...
