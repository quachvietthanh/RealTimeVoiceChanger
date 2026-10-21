"""Main entry point for RealTimeVoiceChanger CLI.

NOTE: This is a project scaffold. The full real-time audio pipeline and RVC
inference are scheduled for implementation in subsequent milestones.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow direct execution without editable install
if __package__ is None or __package__ == "":
    src_dir = str(Path(__file__).resolve().parent.parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

from voice_changer import __version__
from voice_changer.diagnostics.environment import run_environment_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="voice-changer",
        description="RealTimeVoiceChanger: Real-time voice conversion for Windows.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Run environment and system diagnostic checks.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the desktop graphical user interface (default).",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in command-line mode instead of launching the desktop GUI.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.check_env:
        run_environment_summary()
        return 0

    if args.cli:
        print("=" * 60)
        print(f"RealTimeVoiceChanger v{__version__}")
        print("=" * 60)
        print("Backend service is active. For programmatic use, see docs/gui-backend-contract.md.")
        print("To launch the graphical desktop interface, run: voice-changer --gui")
        return 0

    # Default action: Launch Desktop GUI
    from voice_changer.gui import launch_gui

    launch_gui()
    return 0


if __name__ == "__main__":
    sys.exit(main())

