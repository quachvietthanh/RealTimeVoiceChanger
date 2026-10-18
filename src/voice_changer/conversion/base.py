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
