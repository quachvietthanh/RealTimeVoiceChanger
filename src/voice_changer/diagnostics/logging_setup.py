"""Backend log file setup for the desktop application.

The backend logs through the standard `logging` module (errors with tracebacks, lifecycle transitions,
device and model failures). The application calls configure_logging() once at startup; the backend
never configures logging by itself. Real-time audio callbacks do not log, so file I/O never runs on an
audio thread. Logs contain file paths and device names, no audio and no user data.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

LOG_FORMAT = "%(asctime)s %(levelname)-7s [%(threadName)s] %(name)s: %(message)s"


def default_log_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "RealTimeVoiceChanger" / "logs"


def configure_logging(log_dir: Optional[Path] = None, level: int = logging.INFO,
                      max_bytes: int = 2_000_000, backups: int = 5) -> Path:
    """Attach a rotating file handler to the `voice_changer` logger and return the log file path.

    Safe to call more than once (the handler is added only once per file).
    """
    directory = Path(log_dir) if log_dir else default_log_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "backend.log"
    logger = logging.getLogger("voice_changer")
    logger.setLevel(level)
    for h in logger.handlers:
        if isinstance(h, RotatingFileHandler) and Path(h.baseFilename) == path.resolve():
            return path
    handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)
    return path
