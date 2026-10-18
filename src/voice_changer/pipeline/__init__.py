"""Pipeline orchestration package.

Coordinates real-time audio streaming, chunk buffering, voice conversion,
and latency monitoring.
"""

from voice_changer.pipeline.realtime import RealtimePipeline

__all__ = ["RealtimePipeline"]
