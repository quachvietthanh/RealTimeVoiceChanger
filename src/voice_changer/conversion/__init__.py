"""Voice conversion subsystem package.

Defines the core VoiceConverter protocol and concrete conversion adapters.
"""

from voice_changer.conversion.base import ConversionConfig, VoiceConverter

__all__ = [
    "ConversionConfig",
    "VoiceConverter",
]
