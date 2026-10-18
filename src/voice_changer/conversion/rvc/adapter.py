"""RVC (Retrieval-based Voice Conversion) engine adapter.

FUTURE COMPONENT: Adapts an existing compatible RVC inference implementation
to the VoiceConverter protocol. Concrete inference will be integrated in
Milestone 2 (offline) and Milestone 4 (real-time).
"""

from __future__ import annotations

from typing import Optional

from voice_changer.conversion.base import ConversionConfig, VoiceConverter


class RVCAdapter(VoiceConverter):
    """Adapter bridging the core application to a compatible RVC inference engine.

    Keeps third-party RVC repository details, torch tensors, and custom CUDA
    routines decoupled from the audio streaming pipeline.
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._model_loaded = False
        self._model_path: Optional[str] = None
        self._index_path: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded

    def load_model(self, model_path: str, index_path: Optional[str] = None) -> None:
        """Load RVC weights (.pth) and optional index (.index)."""
        # Concrete implementation deferred to Milestone 2
        raise NotImplementedError(
            "RVC model loading will be implemented in Milestone 2 "
            "after inference backend selection."
        )

    def convert(self, audio_chunk: bytes, config: ConversionConfig) -> bytes:
        """Run RVC inference on an audio chunk using the provided configuration."""
        # Concrete inference deferred to Milestone 2 & 4
        raise NotImplementedError(
            "RVC conversion inference will be implemented in Milestone 2/4."
        )

    def reset(self) -> None:
        """Reset internal inference and pitch states."""
        # Concrete buffer reset deferred to Milestone 4
        pass
