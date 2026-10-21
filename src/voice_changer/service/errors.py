"""Backend error types for the GUI-facing service.

Every error carries a stable machine-readable `code`, a one-line `message` for the user, and an
actionable `hint`. Technical details (tracebacks, PortAudio messages) go to the log and to `details`,
never as the only text shown to the user.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


class BackendError(Exception):
    """Base class of all errors raised by VoiceChangerService."""

    category = "backend"

    def __init__(self, code: str, message: str, hint: str = "", details: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.details = details

    def to_info(self) -> "ErrorInfo":
        return ErrorInfo(category=self.category, code=self.code, message=self.message, hint=self.hint, details=self.details)


class ConfigurationError(BackendError):
    """Settings are invalid. Nothing was changed; fix the values and retry."""

    category = "configuration"


class ProfileError(BackendError):
    """A voice profile is unknown, has missing/mismatched assets, or failed to load."""

    category = "profile"


class AudioDeviceError(BackendError):
    """An audio device is missing, rejects the stream format, or failed to open."""

    category = "audio_device"


class InvalidStateError(BackendError):
    """The operation is not allowed in the engine's current state (see EngineState)."""

    category = "state"


@dataclass(frozen=True)
class ErrorInfo:
    """Snapshot of the most recent error, as reported in EngineStatus.last_error."""

    category: str
    code: str
    message: str
    hint: str = ""
    details: str = ""
    timestamp: float = field(default_factory=time.time)
