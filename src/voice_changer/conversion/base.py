"""Abstract conversion interface.

Provides a decoupled protocol separating the real-time audio pipeline
from any concrete voice-conversion backend (RVC or future models).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


@dataclass
class ConversionConfig:
    """Parameters governing voice conversion."""

    pitch_shift: float = 12.0  # Semitones (+12 for typical male-to-female shift)
    f0_method: str = "rmvpe"   # Pitch extraction algorithm
    index_rate: float = 0.75   # Feature retrieval blending ratio
    filter_radius: int = 3     # Median filter radius for pitch
    protect_rate: float = 0.33 # Consonant and breath protection factor


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

    def reset(self) -> None:
        """Reset internal buffer states (e.g. pitch history, crossfades)."""
        ...
