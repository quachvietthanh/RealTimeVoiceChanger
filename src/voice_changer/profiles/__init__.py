"""Voice profile package.

Declares selectable voices (checkpoint + matching index + default settings) and validates them.
"""

from voice_changer.profiles.registry import (
    DEFAULT_REGISTRY_PATH,
    PROFILE_STATUSES,
    ProfileRegistry,
    ProfileRegistryError,
    ProfileValidation,
    VoiceProfile,
)

__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "PROFILE_STATUSES",
    "ProfileRegistry",
    "ProfileRegistryError",
    "ProfileValidation",
    "VoiceProfile",
]
