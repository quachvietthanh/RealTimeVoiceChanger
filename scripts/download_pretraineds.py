"""Safely download and verify RVC v2 40kHz pretrained G/D base weights.

Downloads:
- f0G40k.pth (Generator base, ~73.1 MB)
- f0D40k.pth (Discriminator base, ~142.9 MB)
From official IAHispano/Applio Hugging Face repository at pinned immutable commit:
e908bdd5bf9f18cede26bbbec724b1a4bc7c2f14
Destination:
third_party/applio/rvc/models/pretraineds/pretrained_v2/
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
DEST_DIR = REPO_ROOT / "third_party" / "applio" / "rvc" / "models" / "pretraineds" / "pretrained_v2"

COMMIT_SHA = "e908bdd5bf9f18cede26bbbec724b1a4bc7c2f14"
BASE_URL = f"https://huggingface.co/IAHispano/Applio/resolve/{COMMIT_SHA}/Resources/pretrained_v2"

ASSETS = {
    "f0G40k.pth": {
        "url": f"{BASE_URL}/f0G40k.pth",
        "expected_size": 73106273,
        "expected_sha256": "3b2c44035e782c4b14ddc0bede9e2f4a724d025cd073f736d4f43708453adfcb",
        "type": "Generator base",
    },
    "f0D40k.pth": {
        "url": f"{BASE_URL}/f0D40k.pth",
        "expected_size": 142875703,
        "expected_sha256": "6b6ab091e70801b28e3f41f335f2fc5f3f35c75b39ae2628d419644ec2b0fa09",
        "type": "Discriminator base",
    },
}


def download_and_verify():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for fname, meta in ASSETS.items():
        final_path = DEST_DIR / fname
        tmp_path = DEST_DIR / f"{fname}.tmp"

        print("=" * 70)
        print(f"Asset: {fname} ({meta['type']})")
        print(f"Target Path: {final_path}")
        print("=" * 70)

        if final_path.exists():
            print(f"File already exists at final destination: {final_path}")
            actual_size = final_path.stat().st_size
            hasher = hashlib.sha256()
            with open(final_path, "rb") as f:
                while chunk := f.read(1024 * 1024):
                    hasher.update(chunk)
            actual_sha = hasher.hexdigest()
            print(f"Existing size: {actual_size} bytes (Expected: {meta['expected_size']})")
            print(f"Existing SHA-256: {actual_sha}")
            assert actual_size == meta["expected_size"], "Size mismatch on existing file"
            assert actual_sha == meta["expected_sha256"], "SHA-256 mismatch on existing file"
            results[fname] = {"status": "PASS", "path": str(final_path), "size": actual_size, "sha256": actual_sha}
            continue

        print(f"Downloading from: {meta['url']}")
        req = urllib.request.Request(meta["url"], headers={"User-Agent": "RealTimeVoiceChanger/1.0"})
        hasher = hashlib.sha256()
        downloaded_bytes = 0

        with urllib.request.urlopen(req) as resp, open(tmp_path, "wb") as out_f:
            status = resp.status
            if status != 200:
                raise RuntimeError(f"HTTP Error {status} when downloading {meta['url']}")
            while chunk := resp.read(1024 * 1024):
                out_f.write(chunk)
                hasher.update(chunk)
                downloaded_bytes += len(chunk)
                print(f"\rDownloaded {downloaded_bytes / (1024*1024):.2f} MB...", end="", flush=True)

        print("\nDownload complete. Validating integrity...")
        actual_size = tmp_path.stat().st_size
        actual_sha = hasher.hexdigest()

        print(f"Actual size  : {actual_size} bytes")
        print(f"Expected size: {meta['expected_size']} bytes")
        print(f"Actual SHA-256  : {actual_sha}")
        print(f"Expected SHA-256: {meta['expected_sha256']}")

        if actual_size != meta["expected_size"]:
            tmp_path.unlink()
            raise ValueError(f"Size mismatch: {actual_size} vs {meta['expected_size']}")

        if actual_sha != meta["expected_sha256"]:
            tmp_path.unlink()
            raise ValueError(f"SHA-256 mismatch: {actual_sha} vs {meta['expected_sha256']}")

        # Inspect checkpoint safely
        print("Inspecting PyTorch checkpoint contents safely...")
        ckpt = torch.load(tmp_path, map_location="cpu", weights_only=False)
        assert isinstance(ckpt, dict), "Checkpoint must be a dictionary"
        assert "model" in ckpt, "Checkpoint must contain 'model' key for Applio train.py loading"
        model_weights = ckpt["model"]
        print(f"Model keys count: {len(model_weights.keys())}")
        sample_keys = list(model_weights.keys())[:5]
        print(f"Sample keys: {sample_keys}")

        # Promote tmp to final
        if final_path.exists():
            final_path.unlink()
        tmp_path.rename(final_path)
        print(f"Successfully promoted to: {final_path}\n")

        results[fname] = {
            "status": "PASS",
            "path": str(final_path),
            "size": actual_size,
            "sha256": actual_sha,
            "weight_keys_count": len(model_weights.keys()),
        }

    print("=" * 70)
    print("ALL PRETRAINED ASSETS VERIFIED AND INSTALLED SUCCESSFULLY")
    print("=" * 70)
    return results


if __name__ == "__main__":
    download_and_verify()
