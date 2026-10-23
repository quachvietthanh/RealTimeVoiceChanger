"""Point a registered voice profile at copies of its files outside third_party/applio/logs/ (a training cache).

Dry run by default: lists the profiles that still read from the cache and, for --profile, the copies it would
make. --copy copies the registered files to models/weights/ and models/indices/ (never moving, modifying or
deleting the originals, never overwriting an existing file) and proves each copy identical to the pinned
SHA-256. --write then repoints model_path/index_path in config/voice_profiles.json (a timestamped backup is
kept); status, defaults, data rights, withdrawal and hashes are not changed. No model is loaded onto the GPU;
validation reads the checkpoint and index on the CPU.

Examples (repo root, inference venv):
    .venv\\Scripts\\python scripts/relocate_profile_assets.py
    .venv\\Scripts\\python scripts/relocate_profile_assets.py --profile voice_b_preview
    .venv\\Scripts\\python scripts/relocate_profile_assets.py --profile voice_b_preview --copy
    .venv\\Scripts\\python scripts/relocate_profile_assets.py --profile voice_b_preview --copy --write

See docs/voice-profiles.md ("Moving a profile off the training cache").
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from voice_changer.profiles.registration import (  # noqa: E402
    RegistrationError,
    profiles_in_training_cache,
    relocate_profile_assets,
)
from voice_changer.profiles.registry import DEFAULT_REGISTRY_PATH, ProfileRegistry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", help="profile id to relocate (omit to only list affected profiles)")
    ap.add_argument("--model-dest", type=Path, default=None, help="destination .pth (default models/weights/<name>)")
    ap.add_argument("--index-dest", type=Path, default=None, help="destination .index (default models/indices/<name>)")
    ap.add_argument("--copy", action="store_true", help="copy missing destination files (verified by SHA-256)")
    ap.add_argument("--write", action="store_true", help="repoint the registry entry (default: dry run)")
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    args = ap.parse_args()

    affected = profiles_in_training_cache(ProfileRegistry.from_file(args.registry))
    print("profiles reading from third_party/applio/logs/: " + (", ".join(affected) if affected else "none"))
    if not args.profile:
        return 0

    try:
        result = relocate_profile_assets(args.profile, args.model_dest, args.index_dest, registry_path=args.registry,
                                         copy=args.copy, write=args.write)
    except RegistrationError as err:
        print(f"REFUSED: {err}", file=sys.stderr)
        return 1

    for s in result.steps:
        print(f"{s.kind}: {s.action:9s} {s.source} -> {s.destination}")
    if result.validation is not None:
        v = result.validation
        print(f"validation ok: model {v.model_version} {v.model_sample_rate} Hz, index dim {v.index_dim} "
              f"({v.index_vectors} vectors)")
        for w in v.warnings:
            print(f"warning: {w}")
    print(json.dumps({k: result.entry.get(k) for k in ("id", "status", "model_path", "index_path", "provenance")},
                     indent=2, ensure_ascii=False))
    if result.written:
        print(f"written: {args.registry} (previous version: {result.backup_path})")
        print("the files under third_party/applio/logs/ were not touched; keep them until the profile has been "
              "selected and listened to from its new location.")
    elif any(s.action == "copy" for s in result.steps):
        print("dry run: nothing copied or written (add --copy, then --write)")
    else:
        print("dry run: registry not written (add --write)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
