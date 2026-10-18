"""Real-time audio processing pipeline.

FUTURE COMPONENT: Orchestrates audio capture, chunking, conversion inference,
and playback. Scheduled for implementation in Milestone 3 (passthrough)
and Milestone 4 (RVC conversion).
"""

from __future__ import annotations

from typing import Optional

from voice_changer.audio.input_stream import AudioInputStream
from voice_changer.audio.output_stream import AudioOutputStream
from voice_changer.conversion.base import ConversionConfig, VoiceConverter


class RealtimePipeline:
    """Orchestrates end-to-end real-time audio conversion.

    Notice: This pipeline depends strictly on the VoiceConverter protocol,
    ensuring no direct coupling with any specific RVC repository implementation.
    """

    def __init__(
        self,
        input_stream: AudioInputStream,
        output_stream: AudioOutputStream,
        converter: Optional[VoiceConverter] = None,
        config: Optional[ConversionConfig] = None,
    ) -> None:
        self.input_stream = input_stream
        self.output_stream = output_stream
        self.converter = converter
        self.config = config or ConversionConfig()
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    def start(self, passthrough_only: bool = False) -> None:
        """Start the real-time processing loop.

        Args:
            passthrough_only: If True, routes input directly to output without
                              inference (used in Milestone 3 for baseline testing).
        """
        raise NotImplementedError(
            "Realtime pipeline execution will be implemented in Milestone 3 and 4."
        )

    def stop(self) -> None:
        """Stop the real-time pipeline and release audio streams."""
        raise NotImplementedError(
            "Realtime pipeline execution will be implemented in Milestone 3 and 4."
        )
