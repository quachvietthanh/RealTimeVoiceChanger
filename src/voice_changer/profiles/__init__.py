"""Voice profile package.

Declares selectable voices (checkpoint + matching index + default settings) and validates them.
"""

from voice_changer.profiles.registry import (
    DATA_RIGHTS,
    DEFAULT_REGISTRY_PATH,
    PROFILE_STATUSES,
    ProfileRegistry,
    ProfileRegistryError,
    ProfileValidation,
    VoiceProfile,
)

__all__ = [
    "DATA_RIGHTS",
    "DEFAULT_REGISTRY_PATH",
    "PROFILE_STATUSES",
    "ProfileRegistry",
    "ProfileRegistryError",
    "ProfileValidation",
    "VoiceProfile",
]
