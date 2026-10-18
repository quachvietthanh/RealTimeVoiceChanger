"""Reproducibility helper for Applio Core v3.2.8 low-latency streaming patch.

Allows checking, applying, and reverting the tracked streaming patch
(patches/applio_streaming.patch) cleanly against the pinned Applio revision.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
APPLIO_ROOT = REPO_ROOT / "third_party" / "applio"
PATCH_FILE = REPO_ROOT / "patches" / "applio_streaming.patch"
TARGET_FILE = APPLIO_ROOT / "rvc" / "infer" / "pipeline.py"

PADDING_CODE = """    def set_padding(self, pad_seconds: float = 0.1, tgt_sr: int = 40000) -> None:
        \"\"\"Set dynamic edge padding duration in seconds for low-latency streaming.\"\"\"
        self.t_pad = int(self.sample_rate * pad_seconds)
        self.t_pad_tgt = int(tgt_sr * pad_seconds)
        self.t_pad2 = self.t_pad * 2
"""

FAISS_TARGET_CLEAN = """        if file_index != "" and os.path.exists(file_index) and index_rate > 0:
            try:
                index = faiss.read_index(file_index)
                big_npy = index.reconstruct_n(0, index.ntotal)"""

FAISS_TARGET_PATCHED = """        if isinstance(file_index, tuple) and len(file_index) == 2:
            index, big_npy = file_index
        elif file_index != "" and isinstance(file_index, str) and os.path.exists(file_index) and index_rate > 0:
            try:
                index = faiss.read_index(file_index)
                big_npy = index.reconstruct_n(0, index.ntotal)"""


def check_patch_status() -> int:
    """Check if Applio streaming patch is applied.

    Returns:
        0: Patch is applied and active.
        1: Patch is NOT applied (clean upstream).
        2: Applio source tree is missing.
    """
    if not TARGET_FILE.exists():
        print(f"[ERROR] Applio pipeline file not found at: {TARGET_FILE}")
        print("Ensure Applio Core v3.2.8 is cloned into third_party/applio.")
        return 2

    content = TARGET_FILE.read_text(encoding="utf-8")
    has_padding = "def set_padding" in content
    has_faiss_tuple = "isinstance(file_index, tuple)" in content

    if has_padding and has_faiss_tuple:
        print("[STATUS] Applio streaming patch is APPLIED and VERIFIED.")
        return 0
    elif not has_padding and not has_faiss_tuple:
        print("[STATUS] Applio streaming patch is NOT APPLIED (clean upstream).")
        return 1
    else:
        print("[STATUS] Applio pipeline is partially patched or modified.")
        return 1


def apply_patch() -> bool:
    """Apply the streaming patch to third_party/applio."""
    status = check_patch_status()
    if status == 0:
        print("[INFO] Patch is already applied. Nothing to do.")
        return True
    if status == 2:
        return False

    # Try git apply first if .git exists in third_party/applio
    if (APPLIO_ROOT / ".git").exists() and PATCH_FILE.exists():
        try:
            res = subprocess.run(
                ["git", "-C", str(APPLIO_ROOT), "apply", str(PATCH_FILE.resolve())],
                capture_output=True,
                text=True,
            )
            if res.returncode == 0 and check_patch_status() == 0:
                print("[SUCCESS] Successfully applied patch using 'git apply'.")
                return True
        except Exception:
            pass

    # Fallback to direct Python insertion
    print("[INFO] Applying patch via Python file update...")
    content = TARGET_FILE.read_text(encoding="utf-8")

    # 1. Insert set_padding method after device assignment in __init__
    anchor = "            device=self.device,\n        )\n"
    if "def set_padding" not in content and anchor in content:
        content = content.replace(anchor, anchor + "\n" + PADDING_CODE)

    # 2. Update FAISS index handling
    if FAISS_TARGET_CLEAN in content:
        content = content.replace(FAISS_TARGET_CLEAN, FAISS_TARGET_PATCHED)

    # 3. Add @staticmethod to AudioProcessor.change_rms
    if "@staticmethod\n    def change_rms(" not in content and "    def change_rms(" in content:
        content = content.replace("    def change_rms(", "    @staticmethod\n    def change_rms(")

    TARGET_FILE.write_text(content, encoding="utf-8")

    if check_patch_status() == 0:
        print("[SUCCESS] Successfully applied patch to Applio pipeline.")
        return True
    else:
        print("[ERROR] Failed to verify applied patch.")
        return False


def revert_patch() -> bool:
    """Revert the streaming patch back to upstream state."""
    status = check_patch_status()
    if status == 1:
        print("[INFO] Upstream is already clean. Nothing to revert.")
        return True
    if status == 2:
        return False

    # Try git checkout first
    if (APPLIO_ROOT / ".git").exists():
        try:
            res = subprocess.run(
                ["git", "-C", str(APPLIO_ROOT), "checkout", "--", "rvc/infer/pipeline.py"],
                capture_output=True,
                text=True,
            )
            if res.returncode == 0 and check_patch_status() == 1:
                print("[SUCCESS] Successfully reverted Applio to clean upstream via 'git checkout'.")
                return True
        except Exception:
            pass

    # Fallback Python reversal
    print("[INFO] Reverting patch via Python file update...")
    content = TARGET_FILE.read_text(encoding="utf-8")

    if PADDING_CODE in content:
        content = content.replace("\n" + PADDING_CODE, "")

    if FAISS_TARGET_PATCHED in content:
        content = content.replace(FAISS_TARGET_PATCHED, FAISS_TARGET_CLEAN)

    if "@staticmethod\n    def change_rms(" in content:
        content = content.replace("    @staticmethod\n    def change_rms(", "    def change_rms(")

    TARGET_FILE.write_text(content, encoding="utf-8")

    if check_patch_status() == 1:
        print("[SUCCESS] Successfully reverted Applio to clean upstream.")
        return True
    else:
        print("[ERROR] Failed to cleanly revert Applio pipeline.")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Applio Streaming Optimization Patch Manager")
    parser.add_argument("--check", action="store_true", help="Check if patch is applied")
    parser.add_argument("--apply", action="store_true", help="Apply streaming optimization patch")
    parser.add_argument("--revert", action="store_true", help="Revert patch to clean upstream")
    args = parser.parse_args()

    if args.apply:
        success = apply_patch()
        sys.exit(0 if success else 1)
    elif args.revert:
        success = revert_patch()
        sys.exit(0 if success else 1)
    else:
        code = check_patch_status()
        sys.exit(code)


if __name__ == "__main__":
    main()
