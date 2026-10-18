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
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.check_env:
        run_environment_summary()
        return 0

    print("=" * 60)
    print(f"RealTimeVoiceChanger v{__version__} (Scaffold)")
    print("=" * 60)
    print("This project is currently in the initial scaffolding stage.")
    print("The real-time conversion engine has not yet been initialized.")
    print("\nPlanned roadmap:")
    print("  1. Environment & Hardware Assessment (Run: python scripts/check_environment.py)")
    print("  2. Offline WAV Conversion & Model Evaluation")
    print("  3. Audio Passthrough to Virtual Cable")
    print("  4. Real-Time RVC Inference Integration")
    print("  5. Quality Tuning & Latency Optimization")
    print("\nSee docs/roadmap.md and docs/architecture.md for implementation details.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
