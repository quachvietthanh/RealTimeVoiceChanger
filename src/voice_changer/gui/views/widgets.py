"""Small shared widgets: monogram avatars, collapsible sections and keyboard-focusable text links."""

from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
from typing import Callable, Optional

from voice_changer.gui.dpi import px
from voice_changer.gui.presentation import AvatarSpec
from voice_changer.gui.theme import Colors, Fonts

logger = logging.getLogger(__name__)


def _lighten(color: str, amount: float = 0.35) -> str:
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return "#{:02x}{:02x}{:02x}".format(*(round(c + (255 - c) * amount) for c in (r, g, b)))


def _load_scaled_image(master: tk.Misc, path: Path, size_px: int) -> tk.PhotoImage:
    """`path` fitted to `size_px`: smoothly resampled with Pillow when it is installed, else Tk's integer subsample."""
    try:
        from PIL import Image, ImageTk
    except ImportError:
        image = tk.PhotoImage(master=master, file=str(path))
        factor = max(1, round(image.width() / size_px))
        return image.subsample(factor) if factor > 1 else image
    with Image.open(path) as src:
        src = src.convert("RGBA")
        src.thumbnail((size_px, size_px), Image.LANCZOS)
        return ImageTk.PhotoImage(src, master=master)  # type: ignore[return-value]


def create_avatar(parent: tk.Misc, spec: AvatarSpec, size: int = 40, bg_parent: str = Colors.BG_CARD) -> tk.Canvas:
    """A circular avatar `size` logical pixels wide: the spec's local PNG if it loads, else its monogram.

    The glyph font is in points, which Tk already scales with the DPI, so it is sized from the 96-DPI `size`.
    """
    font_size = max(8, int(size * 0.35))
    size_px = px(size)
    canvas = tk.Canvas(parent, width=size_px, height=size_px, bg=bg_parent, highlightthickness=0)
    canvas.avatar_spec = spec  # type: ignore[attr-defined]
    if spec.image is not None and spec.image.is_file():
        try:
            image = _load_scaled_image(canvas, spec.image, size_px)
            canvas.create_image(size_px / 2, size_px / 2, image=image)
            canvas.image_ref = image  # type: ignore[attr-defined]  # keep the image alive with the canvas
            return canvas
        except (tk.TclError, OSError, ValueError) as err:
            logger.warning("Avatar image %s could not be loaded (%s); using the monogram", spec.image, err)
    canvas.create_oval(2, 2, size_px - 2, size_px - 2, fill=spec.color, outline=_lighten(spec.color), width=2)
    canvas.create_text(size_px / 2, size_px / 2, text=spec.glyph, fill=Colors.TEXT_ON_PRIMARY,
                       font=(Fonts.FAMILY, font_size, "bold"))
    return canvas


def make_link(label: tk.Label, on_activate: Callable[[], None], bg: str) -> None:
    """Make a Label behave like a link: hand cursor, Tab focus, Return/Space/click activation, focus ring."""
    label.config(cursor="hand2", takefocus=True, highlightthickness=1, highlightbackground=bg,
                 highlightcolor=Colors.ACCENT_CYAN)

    def _activate(event: Optional[tk.Event] = None) -> str:
        on_activate()
        return "break"

    def _focus_in(event: Optional[tk.Event] = None) -> None:
        label.config(highlightbackground=Colors.ACCENT_CYAN)

    def _focus_out(event: Optional[tk.Event] = None) -> None:
        label.config(highlightbackground=bg)

    label._activate = _activate  # type: ignore[attr-defined]
    label._on_focus_in = _focus_in  # type: ignore[attr-defined]
    label._on_focus_out = _focus_out  # type: ignore[attr-defined]
    for seq in ("<Button-1>", "<Return>", "<space>"):
        label.bind(seq, _activate)
    label.bind("<FocusIn>", _focus_in)
    label.bind("<FocusOut>", _focus_out)


class CollapsibleSection(tk.Frame):
    """A titled card whose body can be shown or hidden. Put content in ``self.body``."""

    def __init__(self, parent: tk.Misc, title: str, expanded: bool = False, bg: str = Colors.BG_SURFACE) -> None:
        super().__init__(parent, bg=bg, highlightbackground=Colors.BORDER, highlightthickness=1)
        self._bg = bg
        self.expanded = expanded

        self.header = tk.Frame(self, bg=bg, padx=14, pady=10, cursor="hand2", takefocus=True,
                               highlightthickness=1, highlightbackground=bg, highlightcolor=Colors.ACCENT_CYAN)
        self.header.pack(fill="x")
        self.lbl_arrow = tk.Label(self.header, text="", font=Fonts.BODY_BOLD, bg=bg, fg=Colors.TEXT_MUTED, width=2)
        self.lbl_arrow.pack(side="left")
        self.lbl_title = tk.Label(self.header, text=title, font=Fonts.HEADING, bg=bg, fg=Colors.TEXT_PRIMARY,
                                  anchor="w")
        self.lbl_title.pack(side="left")
        # Right-hand summary of the hidden content (e.g. "Đang tắt"), kept short by the owner.
        self.lbl_summary = tk.Label(self.header, text="", font=Fonts.CAPTION, bg=bg, fg=Colors.TEXT_MUTED,
                                    anchor="e")
        self.lbl_summary.pack(side="right")

        self.body = tk.Frame(self, bg=bg)

        for w in (self.header, self.lbl_arrow, self.lbl_title, self.lbl_summary):
            w.bind("<Button-1>", lambda e: self.toggle())
        self.header.bind("<Return>", lambda e: self.toggle())
        self.header.bind("<space>", lambda e: self.toggle())
        self.header.bind("<FocusIn>", lambda e: self.header.config(highlightbackground=Colors.ACCENT_CYAN))
        self.header.bind("<FocusOut>", lambda e: self.header.config(highlightbackground=bg))
        self._apply()

    def set_summary(self, text: str, fg: str = Colors.TEXT_MUTED) -> None:
        self.lbl_summary.config(text=text, fg=fg)

    def toggle(self) -> None:
        self.set_expanded(not self.expanded)

    def set_expanded(self, expanded: bool) -> None:
        self.expanded = expanded
        self._apply()

    def _apply(self) -> None:
        self.lbl_arrow.config(text="▾" if self.expanded else "▸")
        if self.expanded:
            self.body.pack(fill="x")
        else:
            self.body.pack_forget()
