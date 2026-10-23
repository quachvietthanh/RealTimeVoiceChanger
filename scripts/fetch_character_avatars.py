"""Download personal avatar pictures for the Voice Library's VALORANT catalog into the user's avatars folder.

The pictures are Riot Games artwork, fetched from the community API at valorant-api.com for the user's own,
local use. They are written to %LOCALAPPDATA%\\RealTimeVoiceChanger\\avatars\\valorant\\<agent>.png, outside the
repository: they are never bundled, shipped or recorded in gui/assets/asset_manifest.json. Each picture is
cropped to a circle and reduced to SIZE px so the GUI decodes it quickly. Requires Pillow.

    .venv\\Scripts\\python scripts/fetch_character_avatars.py [--out DIR] [--force]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voice_changer.gui.presentation import load_character_catalog, user_avatars_dir  # noqa: E402

AGENTS_URL = "https://valorant-api.com/v1/agents?isPlayableCharacter=true"
SIZE = 256
TIMEOUT_S = 30


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:
        return response.read()


def _circle(png: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(png)).convert("RGBA")
    image.thumbnail((SIZE, SIZE), Image.LANCZOS)
    square = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    square.paste(image, ((SIZE - image.width) // 2, (SIZE - image.height) // 2))
    mask = Image.new("L", (SIZE * 4, SIZE * 4), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, SIZE * 4 - 1, SIZE * 4 - 1), fill=255)
    alpha = Image.composite(square.getchannel("A"), Image.new("L", (SIZE, SIZE), 0),
                            mask.resize((SIZE, SIZE), Image.LANCZOS))
    square.putalpha(alpha)
    return square


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=user_avatars_dir() / "valorant",
                        help="destination folder (default: %(default)s)")
    parser.add_argument("--force", action="store_true", help="replace pictures that already exist")
    args = parser.parse_args()

    catalog = load_character_catalog()
    agents = {a["displayName"]: a for a in json.loads(_fetch(AGENTS_URL))["data"]}
    args.out.mkdir(parents=True, exist_ok=True)
    missing = []
    for entry in catalog.entries:
        franchise, slug = entry.catalog_id.split(".", 1)
        if franchise != "valorant":
            continue
        target = args.out / f"{slug}.png"
        agent = agents.get(entry.name)
        if agent is None or not agent.get("displayIcon"):
            missing.append(entry.name)
            continue
        if target.exists() and not args.force:
            print(f"keep     {target.name}")
            continue
        _circle(_fetch(agent["displayIcon"])).save(target, optimize=True)
        print(f"saved    {target.name}")
    if missing:
        print(f"no picture found for: {', '.join(missing)} (shown with a monogram)")
    print(f"folder: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
