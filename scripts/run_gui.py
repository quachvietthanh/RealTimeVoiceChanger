"""Launcher script for the RealTimeVoiceChanger desktop GUI."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to sys.path for direct repository runs
repo_root = Path(__file__).resolve().parent.parent
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from voice_changer.gui import launch_gui

if __name__ == "__main__":
    launch_gui()
