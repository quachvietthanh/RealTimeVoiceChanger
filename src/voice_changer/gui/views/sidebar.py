"""Sidebar navigation: product identity, the three main pages, and Advanced / Troubleshooting at the bottom."""

from __future__ import annotations

import logging
import tkinter as tk
from typing import Callable, Optional

from voice_changer.gui.dpi import px
from voice_changer.gui.presentation import APP_NAME, ICONS_DIR, app_version
from voice_changer.gui.theme import Colors, Fonts

logger = logging.getLogger(__name__)

LOGO_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)


def logo_file_for(size_px: int) -> Optional[str]:
    """The generated logo PNG closest to `size_px` (screen pixels), or None if none exists."""
    candidates = [s for s in LOGO_SIZES if (ICONS_DIR / f"app_icon_{s}.png").is_file()]
    if not candidates:
        return None
    best = min(candidates, key=lambda s: (abs(s - size_px), -s))
    return str(ICONS_DIR / f"app_icon_{best}.png")


class Sidebar(tk.Frame):
    """Navigation only: engine state is shown on the Studio page, beside the action that changes it."""

    NAV_ITEMS = [
        ("studio", "🎙️", "Studio", "Chọn giọng và chuyển giọng"),
        ("library", "🎭", "Thư viện giọng", "Tất cả giọng và trạng thái"),
        ("audio_settings", "🔌", "Âm thanh & Kết nối", "Micro, ngõ ra, Discord/OBS/game"),
    ]
    BOTTOM_ITEMS = [
        ("diagnostics", "🛠️", "Nâng cao / Khắc phục sự cố", "Hiệu năng, thông số, log"),
    ]

    def __init__(
        self,
        parent: tk.Widget,
        on_navigate: Callable[[str], None],
        current_view: str = "studio",
    ) -> None:
        super().__init__(parent, bg=Colors.BG_SIDEBAR, width=px(240), highlightbackground=Colors.BORDER,
                         highlightthickness=1)
        self.pack_propagate(False)
        self.on_navigate = on_navigate
        self._current_view = current_view
        self._buttons: dict[str, tuple[tk.Frame, tk.Label, tk.Label, tk.Frame, tk.Frame]] = {}
        self.logo_image: Optional[tk.PhotoImage] = None

        self._build_ui()

    def _build_ui(self) -> None:
        brand_box = tk.Frame(self, bg=Colors.BG_SIDEBAR, padx=16, pady=18)
        brand_box.pack(fill="x")

        logo_path = logo_file_for(px(32))
        if logo_path:
            try:
                self.logo_image = tk.PhotoImage(file=logo_path)
            except tk.TclError as err:
                logger.warning("Could not load sidebar logo %s: %s", logo_path, err)
                self.logo_image = None
        if self.logo_image is not None:
            self.lbl_logo = tk.Label(brand_box, image=self.logo_image, bg=Colors.BG_SIDEBAR)
        else:
            self.lbl_logo = tk.Label(brand_box, text="RT", font=Fonts.BADGE, bg=Colors.PRIMARY,
                                     fg=Colors.TEXT_ON_PRIMARY, padx=6, pady=2)
        self.lbl_logo.pack(side="left", padx=(0, 10))

        title_col = tk.Frame(brand_box, bg=Colors.BG_SIDEBAR)
        title_col.pack(side="left", fill="x", expand=True)
        self.lbl_app_title = tk.Label(title_col, text=APP_NAME, font=Fonts.BODY_BOLD, bg=Colors.BG_SIDEBAR,
                                      fg=Colors.TEXT_PRIMARY, anchor="w", justify="left", wraplength=px(150))
        self.lbl_app_title.pack(fill="x")
        self.lbl_version = tk.Label(title_col, text=f"Phiên bản {app_version()}", font=Fonts.CAPTION,
                                    bg=Colors.BG_SIDEBAR, fg=Colors.TEXT_MUTED, anchor="w")
        self.lbl_version.pack(fill="x")

        tk.Frame(self, bg=Colors.BORDER_SUBTLE, height=1).pack(fill="x", padx=14, pady=(0, 12))

        bottom = tk.Frame(self, bg=Colors.BG_SIDEBAR, padx=10, pady=12)
        bottom.pack(fill="x", side="bottom")
        tk.Frame(bottom, bg=Colors.BORDER_SUBTLE, height=1).pack(fill="x", padx=4, pady=(0, 8))
        for view_id, icon, label, subtitle in self.BOTTOM_ITEMS:
            self._create_nav_button(bottom, view_id, icon, label, subtitle)

        nav_container = tk.Frame(self, bg=Colors.BG_SIDEBAR, padx=10)
        nav_container.pack(fill="both", expand=True)
        for view_id, icon, label, subtitle in self.NAV_ITEMS:
            self._create_nav_button(nav_container, view_id, icon, label, subtitle)

        self.set_active_view(self._current_view)

    def _create_nav_button(
        self,
        parent: tk.Widget,
        view_id: str,
        icon: str,
        label: str,
        subtitle: str,
    ) -> None:
        btn_frame = tk.Frame(
            parent,
            bg=Colors.BG_SIDEBAR,
            cursor="hand2",
            padx=10,
            pady=8,
            takefocus=True,
            highlightthickness=1,
            highlightbackground=Colors.BG_SIDEBAR,
            highlightcolor=Colors.ACCENT_CYAN,
        )
        btn_frame.pack(fill="x", pady=2)

        # Active indicator stripe (left border)
        indicator = tk.Frame(btn_frame, bg=Colors.BG_SIDEBAR, width=3)
        indicator.pack(side="left", fill="y", padx=(0, 8))

        content_box = tk.Frame(btn_frame, bg=Colors.BG_SIDEBAR)
        content_box.pack(side="left", fill="both", expand=True)

        lbl_top = tk.Label(content_box, text=f"{icon}  {label}", font=Fonts.BODY_BOLD, bg=Colors.BG_SIDEBAR,
                           fg=Colors.TEXT_MUTED, anchor="w", justify="left", wraplength=px(170))
        lbl_top.pack(fill="x")

        lbl_sub = tk.Label(
            content_box,
            text=subtitle,
            font=Fonts.CAPTION,
            bg=Colors.BG_SIDEBAR,
            fg=Colors.TEXT_MUTED,
            anchor="w",
            justify="left",
            wraplength=px(170),  # the sidebar has a fixed width; wrap rather than clip
        )
        lbl_sub.pack(fill="x")

        self._buttons[view_id] = (btn_frame, lbl_top, lbl_sub, indicator, content_box)

        def on_click(e: Optional[tk.Event] = None) -> None:
            self.set_active_view(view_id)
            self.on_navigate(view_id)

        def on_enter(e: Optional[tk.Event] = None) -> None:
            if self._current_view != view_id:
                for w in (btn_frame, content_box, lbl_sub):
                    w.config(bg=Colors.BG_SURFACE)
                lbl_top.config(bg=Colors.BG_SURFACE, fg=Colors.TEXT_PRIMARY)

        def on_leave(e: Optional[tk.Event] = None) -> None:
            if self._current_view != view_id:
                for w in (btn_frame, content_box, lbl_sub):
                    w.config(bg=Colors.BG_SIDEBAR)
                lbl_top.config(bg=Colors.BG_SIDEBAR, fg=Colors.TEXT_MUTED)

        def on_focus_in(e: Optional[tk.Event] = None) -> None:
            btn_frame.config(highlightbackground=Colors.ACCENT_CYAN)

        def on_focus_out(e: Optional[tk.Event] = None) -> None:
            btn_frame.config(highlightbackground=Colors.BG_SIDEBAR)

        for w in (btn_frame, indicator, content_box, lbl_top, lbl_sub):
            w.bind("<Button-1>", on_click)
            w.bind("<Enter>", on_enter)
            w.bind("<Leave>", on_leave)

        btn_frame.bind("<Return>", on_click)
        btn_frame.bind("<space>", on_click)
        btn_frame.bind("<FocusIn>", on_focus_in)
        btn_frame.bind("<FocusOut>", on_focus_out)

    def set_active_view(self, view_id: str) -> None:
        """Update navigation styling when current page changes."""
        self._current_view = view_id
        for vid, (frame, lbl_top, lbl_sub, indicator, content_box) in self._buttons.items():
            if vid == view_id:
                frame.config(bg=Colors.BG_CARD_ACTIVE)
                content_box.config(bg=Colors.BG_CARD_ACTIVE)
                lbl_top.config(bg=Colors.BG_CARD_ACTIVE, fg=Colors.TEXT_PRIMARY, font=Fonts.BODY_BOLD)
                lbl_sub.config(bg=Colors.BG_CARD_ACTIVE, fg=Colors.PRIMARY_LIGHT)
                indicator.config(bg=Colors.PRIMARY)
            else:
                frame.config(bg=Colors.BG_SIDEBAR)
                content_box.config(bg=Colors.BG_SIDEBAR)
                lbl_top.config(bg=Colors.BG_SIDEBAR, fg=Colors.TEXT_MUTED, font=Fonts.BODY)
                lbl_sub.config(bg=Colors.BG_SIDEBAR, fg=Colors.TEXT_MUTED)
                indicator.config(bg=Colors.BG_SIDEBAR)
