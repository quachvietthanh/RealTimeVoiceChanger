"""Voice conversion subsystem package.

Defines the core VoiceConverter protocol and concrete conversion adapters.
"""

from voice_changer.conversion.base import (
    SUPPORTED_F0_METHODS,
    ConversionConfig,
    VoiceConverter,
    validate_conversion_config,
)

__all__ = [
    "SUPPORTED_F0_METHODS",
    "ConversionConfig",
    "VoiceConverter",
    "validate_conversion_config",
]
