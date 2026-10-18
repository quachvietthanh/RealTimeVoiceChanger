"""Download and verify RVC auxiliary model assets (RMVPE and ContentVec)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
APPLIO_ROOT = REPO_ROOT / "third_party" / "applio"
CHECKPOINTS_DIR = REPO_ROOT / "models" / "checkpoints"
CONTENTVEC_DIR = CHECKPOINTS_DIR / "contentvec"

HF_REPO = "IAHispano/Applio"
HF_COMMIT = "e908bdd5bf9f18cede26bbbec724b1a4bc7c2f14"
BASE_URL = f"https://huggingface.co/{HF_REPO}/resolve/{HF_COMMIT}/Resources"


@dataclass(frozen=True)
class AssetSpec:
    name: str
    filename: str
    url: str
    dest_path: Path
    applio_sync_path: Path
    expected_size: int
    expected_sha256: str
    asset_type: str


ASSETS: list[AssetSpec] = [
    AssetSpec(
        name="RMVPE Pitch Predictor",
        filename="rmvpe.pt",
        url=f"{BASE_URL}/predictors/rmvpe.pt",
        dest_path=CHECKPOINTS_DIR / "rmvpe.pt",
        applio_sync_path=APPLIO_ROOT / "rvc" / "models" / "predictors" / "rmvpe.pt",
        expected_size=181184272,
        expected_sha256="6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193",
        asset_type="rmvpe",
    ),
    AssetSpec(
        name="ContentVec Embedder Config",
        filename="config.json",
        url=f"{BASE_URL}/embedders/contentvec/config.json",
        dest_path=CONTENTVEC_DIR / "config.json",
        applio_sync_path=APPLIO_ROOT / "rvc" / "models" / "embedders" / "contentvec" / "config.json",
        expected_size=1388,
        expected_sha256="2ddde063b795d38d9051a7215a092fecf4cfe148b54251e38de51d88d356898b",
        asset_type="json_config",
    ),
    AssetSpec(
        name="ContentVec Embedder Weights",
        filename="pytorch_model.bin",
        url=f"{BASE_URL}/embedders/contentvec/pytorch_model.bin",
        dest_path=CONTENTVEC_DIR / "pytorch_model.bin",
        applio_sync_path=APPLIO_ROOT / "rvc" / "models" / "embedders" / "contentvec" / "pytorch_model.bin",
        expected_size=378342945,
        expected_sha256="d8dd400e054ddf4e6be75dab5a2549db748cc99e756a097c496c099f65a4854e",
        asset_type="contentvec_weights",
    ),
]


def download_with_progress(url: str, dest_tmp: Path, expected_size: int) -> str:
    """Download file in chunks and compute SHA-256 on the fly."""
    dest_tmp.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "RealTimeVoiceChanger-AssetDownloader/1.0"},
    )

    sha256 = hashlib.sha256()
    downloaded = 0
    chunk_size = 1024 * 1024

    start_time = time.perf_counter()
    with urllib.request.urlopen(req) as resp, open(dest_tmp, "wb") as out_f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_f.write(chunk)
            sha256.update(chunk)
            downloaded += len(chunk)
            elapsed = time.perf_counter() - start_time
            speed_mb = (downloaded / (1024 * 1024)) / elapsed if elapsed > 0 else 0
            pct = (downloaded / expected_size * 100) if expected_size > 0 else 0
            print(
                f"\r  Progress: {downloaded / (1024*1024):.1f}/{expected_size / (1024*1024):.1f} MB "
                f"({pct:.1f}%) - {speed_mb:.2f} MB/s",
                end="",
                flush=True,
            )

    print()
    return sha256.hexdigest()


def verify_asset_compatibility(asset_type: str, file_path: Path) -> tuple[bool, str]:
    """Verify that the downloaded file is a valid PyTorch model and compatible with Applio."""
    import torch

    try:
        if asset_type == "json_config":
            with open(file_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            archs = cfg.get("architectures", [])
            if "HubertModelWithFinalProj" not in archs:
                return False, f"Unexpected architecture in config: {archs}"
            return True, "Valid HubertModelWithFinalProj config"

        elif asset_type == "rmvpe":
            state = torch.load(str(file_path), map_location="cpu")
            if not isinstance(state, dict):
                return False, "Not a valid PyTorch dictionary checkpoint"
            keys = list(state.keys())
            if not any("mel_extractor" in k or "unet" in k or "conv" in k for k in keys):
                return False, f"Missing expected RMVPE layers. Sample keys: {keys[:5]}"

            if str(APPLIO_ROOT) not in sys.path:
                sys.path.insert(0, str(APPLIO_ROOT))
            from rvc.lib.predictors.RMVPE import RMVPE0Predictor

            predictor = RMVPE0Predictor(str(file_path), is_half=False, device="cpu")
            del predictor
            return True, "Valid RMVPE0Predictor checkpoint (loaded successfully on CPU)"

        elif asset_type == "contentvec_weights":
            state = torch.load(str(file_path), map_location="cpu")
            if not isinstance(state, dict):
                return False, "Not a valid PyTorch dictionary checkpoint"
            keys = list(state.keys())
            if not any("encoder" in k or "feature_extractor" in k for k in keys):
                return False, f"Missing expected Hubert/ContentVec layers. Sample keys: {keys[:5]}"
            return True, "Valid ContentVec transformer weights (loaded successfully on CPU)"

        return True, "Verified"
    except Exception as err:
        return False, f"Compatibility check failed: {err}"


def main() -> int:
    print("=" * 70)
    print("RVC Auxiliary Asset Downloader & Verifier")
    print(f"HuggingFace Repository: {HF_REPO} @ commit {HF_COMMIT[:10]}")
    print("=" * 70)

    all_passed = True

    for asset in ASSETS:
        dest = asset.dest_path
        sync_dest = asset.applio_sync_path
        exp_size = asset.expected_size
        exp_sha = asset.expected_sha256

        print(f"\nProcessing: {asset.name} ({asset.filename})")
        print(f"  Target Destination : {dest}")

        if dest.exists():
            curr_size = dest.stat().st_size
            hasher = hashlib.sha256()
            with open(dest, "rb") as f:
                while chunk := f.read(1024 * 1024):
                    hasher.update(chunk)
            curr_sha = hasher.hexdigest()

            if curr_size == exp_size and curr_sha == exp_sha:
                print(f"  [EXISTS & VALID] Size: {curr_size} bytes | SHA-256 matches.")
                if not sync_dest.exists():
                    sync_dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(dest), str(sync_dest))
                    print(f"  [SYNCED] Copied to {sync_dest.relative_to(REPO_ROOT)}")
                continue
            else:
                print(f"  [CORRUPT/MISMATCH] Existing file invalid ({curr_size} bytes). Re-downloading...")

        tmp_dest = dest.with_suffix(dest.suffix + ".tmp")
        print(f"  Source URL : {asset.url}")
        print(f"  Downloading to temporary file: {tmp_dest.name} ...")

        try:
            actual_sha = download_with_progress(asset.url, tmp_dest, exp_size)
            actual_size = tmp_dest.stat().st_size

            if actual_size != exp_size:
                print(f"  [FAIL] Size mismatch: got {actual_size} bytes, expected {exp_size} bytes.")
                tmp_dest.unlink(missing_ok=True)
                all_passed = False
                continue

            if actual_sha != exp_sha:
                print(f"  [FAIL] SHA-256 mismatch:\n    Got:      {actual_sha}\n    Expected: {exp_sha}")
                tmp_dest.unlink(missing_ok=True)
                all_passed = False
                continue

            print(f"  [PASS] Size ({actual_size} bytes) & SHA-256 verified.")

            print("  Verifying model structure and Applio v3.2.8 compatibility...")
            ok, msg = verify_asset_compatibility(asset.asset_type, tmp_dest)
            if not ok:
                print(f"  [FAIL] {msg}")
                tmp_dest.unlink(missing_ok=True)
                all_passed = False
                continue
            print(f"  [PASS] {msg}")

            if dest.exists():
                dest.unlink()
            tmp_dest.replace(dest)
            print(f"  [INSTALLED] Successfully placed at {dest.relative_to(REPO_ROOT)}")

            sync_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(dest), str(sync_dest))
            print(f"  [SYNCED] Copied to {sync_dest.relative_to(REPO_ROOT)}")

        except Exception as err:
            print(f"  [FAIL] Error during processing: {err}")
            tmp_dest.unlink(missing_ok=True)
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("All auxiliary assets verified, downloaded, and installed successfully.")
        return 0
    else:
        print("Some auxiliary assets failed verification. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
