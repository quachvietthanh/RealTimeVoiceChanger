"""Windows DPI awareness and logical-pixel scaling for the Tkinter GUI.

Tk reads the screen DPI once, when the first root window is created. The process must therefore be
marked DPI-aware *before* ``tk.Tk()``; marking it afterwards leaves Tk at 96 DPI (fonts drawn at 100%
on a 125%/150% display) while Windows stops bitmap-scaling the window, so everything looks undersized.

Once Tk knows the real DPI, point-sized fonts scale automatically, but distances given as plain integers
are physical pixels. ``px()`` converts the GUI's fixed sizes, written for 96 DPI, into physical pixels.
"""

from __future__ import annotations

import ctypes
import logging
import sys

import tkinter as tk

logger = logging.getLogger(__name__)

# PROCESS_SYSTEM_DPI_AWARE: Tk 8.6 does not handle WM_DPICHANGED, so per-monitor awareness would leave the
# window at the wrong size after moving it to a monitor with another scale. System awareness is crisp on the
# primary display and lets Windows scale the window on others.
_PROCESS_SYSTEM_DPI_AWARE = 1
_E_ACCESSDENIED = -2147024891  # 0x80070005 as a signed HRESULT: awareness was already set for this process

_awareness_configured = False
_ui_scale = 1.0


def enable_dpi_awareness() -> bool:
    """Mark the process system-DPI-aware. Call once, before the first ``tk.Tk()``.

    Returns True when the process is DPI-aware afterwards. Idempotent; a no-op off Windows.
    """
    global _awareness_configured
    if _awareness_configured:
        return True
    if sys.platform != "win32":
        return False
    try:
        result = ctypes.windll.shcore.SetProcessDpiAwareness(_PROCESS_SYSTEM_DPI_AWARE)
        aware = result in (0, _E_ACCESSDENIED)
    except (AttributeError, OSError):  # shcore is Windows 8.1+
        try:
            aware = bool(ctypes.windll.user32.SetProcessDPIAware())
        except (AttributeError, OSError):
            aware = False
    if not aware:
        logger.warning("Could not enable DPI awareness; the GUI may be blurry on scaled displays.")
    _awareness_configured = aware
    return aware


def init_ui_scale(root: tk.Misc) -> float:
    """Derive the logical-pixel scale (1.0 = 96 DPI) from the DPI Tk detected and store it for ``px()``.

    Rounded to the 25% steps Windows offers so a 96.09 DPI measurement does not shift layouts by a pixel.
    """
    global _ui_scale
    measured = root.winfo_fpixels("1i") / 96.0
    _ui_scale = max(1.0, round(measured * 4) / 4)
    return _ui_scale


def ui_scale() -> float:
    """The scale set by the last ``init_ui_scale()`` call (1.0 before any)."""
    return _ui_scale


def px(value: float) -> int:
    """Convert a size designed at 96 DPI (100% scaling) to physical pixels at the current scale."""
    return int(round(value * _ui_scale))


def fit_window(root: tk.Tk, width: int, height: int, min_width: int, min_height: int) -> None:
    """Size the window for the current scale, kept within the screen so the title bar and edges stay reachable.

    Sizes are 96-DPI values. On a small or highly scaled screen the minimum shrinks to the screen; the pages
    scroll, so content below the fold stays reachable.
    """
    screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
    avail_w = screen_w - px(16)
    avail_h = screen_h - px(80)  # taskbar and title bar
    w, h = min(px(width), avail_w), min(px(height), avail_h)
    root.minsize(min(px(min_width), avail_w), min(px(min_height), avail_h))
    root.geometry(f"{w}x{h}")
