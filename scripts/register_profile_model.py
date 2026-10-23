"""Register a trained RVC model + index for a voice profile (e.g. 'jett' or 'sage', status 'planned').

Dry run by default: validates everything and prints the entry that would be written. Add --write to update
config/voice_profiles.json (a timestamped backup of the old file is kept next to it). Model files are never
copied, moved or modified; put them in models/weights/ and models/indices/ first.

Examples (repo root, inference venv):
    .venv\\Scripts\\python scripts/register_profile_model.py --profile jett --model models/weights/jett_v1.pth --index models/indices/jett_v1.index --data-rights licensed
    .venv\\Scripts\\python scripts/register_profile_model.py --profile jett --model ... --index ... --data-rights licensed --write

See docs/voice-profiles.md for the rules (rights, shared files, embedder, sample rate).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from voice_changer.profiles.registration import RegistrationError, register_model  # noqa: E402
from voice_changer.profiles.registry import DATA_RIGHTS, DEFAULT_REGISTRY_PATH  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True, help="profile id, e.g. jett or sage")
    ap.add_argument("--model", required=True, type=Path, help="RVC inference checkpoint (.pth)")
    ap.add_argument("--index", type=Path, default=None, help="matching Faiss index (.index)")
    ap.add_argument("--no-index", action="store_true", help="register without an index (index_rate has no effect)")
    ap.add_argument("--data-rights", choices=DATA_RIGHTS, default=None,
                    help="set the profile's data_rights (default: keep the registry value)")
    ap.add_argument("--allow-unverified-rights", action="store_true",
                    help="allow unverified rights for a local test model that is never distributed")
    ap.add_argument("--replace", action="store_true", help="replace the model of a 'preview' profile")
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    ap.add_argument("--write", action="store_true", help="write the registry (default: dry run)")
    args = ap.parse_args()
    if args.index is None and not args.no_index:
        ap.error("--index is required (or pass --no-index)")

    try:
        result = register_model(
            args.profile, args.model, args.index, registry_path=args.registry, data_rights=args.data_rights,
            allow_unverified_rights=args.allow_unverified_rights, allow_no_index=args.no_index,
            replace=args.replace, write=args.write)
    except RegistrationError as err:
        print(f"REFUSED: {err}", file=sys.stderr)
        return 1

    v = result.validation
    print(json.dumps(result.entry, indent=2, ensure_ascii=False))
    print(f"validation ok: model {v.model_version} {v.model_sample_rate} Hz, index dim {v.index_dim} "
          f"({v.index_vectors} vectors)")
    for w in v.warnings:
        print(f"warning: {w}")
    if result.written:
        print(f"written: {args.registry} (previous version: {result.backup_path})")
        print("next: run `python -m unittest tests.integration.test_profile_assets`, then select the profile "
              "in the GUI or with VoiceChangerService and listen before changing its status.")
    else:
        print("dry run: nothing written (add --write)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
