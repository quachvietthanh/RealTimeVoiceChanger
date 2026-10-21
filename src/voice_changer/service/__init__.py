"""GUI-facing backend service (in-process API). See docs/gui-backend-contract.md."""

from voice_changer.service.errors import (
    AudioDeviceError,
    BackendError,
    ConfigurationError,
    ErrorInfo,
    InvalidStateError,
    ProfileError,
)
from voice_changer.service.service import (
    STREAM_SAMPLE_RATE,
    AudioSettings,
    DeviceInventory,
    EngineState,
    EngineStatus,
    NoiseSuppressionStatus,
    ProfileSummary,
    VoiceChangerService,
)
from voice_changer.audio.noise_suppression import NOISE_SUPPRESSION_MODES, NoiseSuppressionSettings

__all__ = [
    "STREAM_SAMPLE_RATE",
    "AudioDeviceError",
    "AudioSettings",
    "BackendError",
    "ConfigurationError",
    "DeviceInventory",
    "EngineState",
    "EngineStatus",
    "ErrorInfo",
    "InvalidStateError",
    "NOISE_SUPPRESSION_MODES",
    "NoiseSuppressionSettings",
    "NoiseSuppressionStatus",
    "ProfileError",
    "ProfileSummary",
    "VoiceChangerService",
]
