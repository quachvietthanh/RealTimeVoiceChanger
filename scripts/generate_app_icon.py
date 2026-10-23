"""Generate the RealTimeVoiceChanger logo/app icon into src/voice_changer/gui/assets/icons/.

Original artwork, drawn here from simple geometry (no third-party images): a dark rounded square with
sound-wave bars running from the theme's purple (#7c3aed) to its cyan (#06b6d4).

Outputs: logo.svg (master), app_icon_<size>.png for the window/taskbar and sidebar, app_icon.ico (multi-size).
Requires Pillow, which the GUI itself does not need at runtime (Tk reads the PNGs directly).

    .venv\\Scripts\\python scripts/generate_app_icon.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

OUT_DIR = Path(__file__).resolve().parent.parent / "src" / "voice_changer" / "gui" / "assets" / "icons"

BACKGROUND = "#16151f"  # theme Colors.BG_SIDEBAR
BORDER = "#2e2a40"  # theme Colors.BORDER
PURPLE = (0x7C, 0x3A, 0xED)  # theme Colors.PRIMARY
CYAN = (0x06, 0xB6, 0xD4)  # theme Colors.ACCENT_CYAN

# Bar heights as a fraction of the icon size; small icons get fewer, wider bars so they stay legible.
BARS_LARGE = (0.30, 0.56, 0.80, 0.52, 0.28)
BARS_SMALL = (0.42, 0.76, 0.46)
PNG_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)
SUPERSAMPLE = 8


def _mix(t: float) -> tuple[int, int, int]:
    return tuple(round(a + (b - a) * t) for a, b in zip(PURPLE, CYAN))  # type: ignore[return-value]


def _bar_layout(size: float, heights: tuple[float, ...]) -> list[tuple[float, float, float, float]]:
    """(x0, y0, x1, y1) of each bar, centred in a square of `size`."""
    n = len(heights)
    width = size * (0.105 if n == 5 else 0.15)
    gap = size * (0.06 if n == 5 else 0.09)
    total = n * width + (n - 1) * gap
    x = (size - total) / 2
    bars = []
    for h in heights:
        bar_h = size * h
        y0 = (size - bar_h) / 2
        bars.append((x, y0, x + width, y0 + bar_h))
        x += width + gap
    return bars


def render_png(size: int) -> Image.Image:
    big = size * SUPERSAMPLE
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    radius = big * 0.22
    draw.rounded_rectangle((0, 0, big - 1, big - 1), radius=radius, fill=BACKGROUND,
                           outline=BORDER if size >= 32 else None, width=max(1, big // 64))
    heights = BARS_LARGE if size >= 32 else BARS_SMALL
    bars = _bar_layout(big, heights)
    for i, (x0, y0, x1, y1) in enumerate(bars):
        draw.rounded_rectangle((x0, y0, x1, y1), radius=(x1 - x0) / 2, fill=_mix(i / (len(bars) - 1)))
    return img.resize((size, size), Image.LANCZOS)


def render_svg(size: int = 256) -> str:
    bars = _bar_layout(size, BARS_LARGE)
    rects = "\n".join(
        f'    <rect x="{x0:.2f}" y="{y0:.2f}" width="{x1 - x0:.2f}" height="{y1 - y0:.2f}" '
        f'rx="{(x1 - x0) / 2:.2f}" fill="url(#wave)"/>'
        for x0, y0, x1, y1 in bars
    )
    # One gradient across the whole bar group (userSpaceOnUse), not one per bar.
    gx0, gx1 = bars[0][0], bars[-1][2]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">\n'
        "  <title>RealTimeVoiceChanger</title>\n"
        "  <defs>\n"
        f'    <linearGradient id="wave" x1="{gx0:.2f}" y1="0" x2="{gx1:.2f}" y2="0" gradientUnits="userSpaceOnUse">\n'
        '      <stop offset="0" stop-color="#7c3aed"/>\n'
        '      <stop offset="1" stop-color="#06b6d4"/>\n'
        "    </linearGradient>\n"
        "  </defs>\n"
        f'  <rect x="0.5" y="0.5" width="{size - 1}" height="{size - 1}" rx="{size * 0.22:.2f}" '
        f'fill="{BACKGROUND}" stroke="{BORDER}" stroke-width="{size / 64:.2f}"/>\n'
        f"{rects}\n"
        "</svg>\n"
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    images = {}
    for size in PNG_SIZES:
        images[size] = render_png(size)
        images[size].save(OUT_DIR / f"app_icon_{size}.png")
    # The SVG fills the bars with one continuous gradient; the PNGs give each bar one color along it.
    (OUT_DIR / "logo.svg").write_text(render_svg(), encoding="utf-8")
    ico_sizes = [(s, s) for s in (16, 20, 24, 32, 40, 48, 64, 256)]
    images[256].save(OUT_DIR / "app_icon.ico", sizes=ico_sizes, append_images=[images[s] for s, _ in ico_sizes])
    print(f"Wrote {len(PNG_SIZES)} PNGs, logo.svg and app_icon.ico to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
