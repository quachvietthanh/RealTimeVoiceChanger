"""Modern sleek dark theme tokens and styles for RealTimeVoiceChanger desktop GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional

from voice_changer.service import EngineState


class Colors:
    """Refined color palette tokens with high contrast and modern aesthetics."""

    # Backgrounds - Refined Charcoal Surfaces
    BG_DARK = "#121118"        # Deep charcoal canvas
    BG_SIDEBAR = "#16151f"     # Distinct charcoal sidebar
    BG_SURFACE = "#1c1a27"     # Card & container background
    BG_ELEVATED = "#232132"    # Input background / elevated surfaces
    BG_CARD = "#201d2e"        # Voice cards
    BG_CARD_HOVER = "#2a263c"  # Voice card hover state
    BG_CARD_ACTIVE = "#2d2745" # Voice card selected state
    BG_HOVER = "#2c283e"       # General hover states

    # Borders & Dividers
    BORDER = "#2e2a40"         # Subtle card border
    BORDER_SUBTLE = "#211e2f"  # Hairline dividers
    BORDER_LIGHT = "#4c4569"   # Active focus border
    BORDER_ACTIVE = "#7c3aed"  # Selected item border (Purple)

    # Accent Colors - Controlled Purple & Precision Cyan
    PRIMARY = "#7c3aed"        # Controlled Royal Purple
    PRIMARY_HOVER = "#6d28d9"  # Deeper Purple
    PRIMARY_LIGHT = "#a78bfa"  # Soft Lavender highlight

    ACCENT_CYAN = "#06b6d4"    # Precision Pro-Audio Cyan
    ACCENT_PURPLE = "#8b5cf6"  # AI Studio Purple

    SUCCESS = "#047857"        # Emerald green (WCAG AA compliant > 5.4:1 with white text)
    SUCCESS_HOVER = "#065f46"  # Darker emerald (> 7.6:1 with white text)
    SUCCESS_BG = "#064e3b"     # Dark green container
    SUCCESS_TEXT = "#6ee7b7"   # Bright emerald text (> 6.3:1 on SUCCESS_BG)

    WARNING = "#f59e0b"        # Amber warning
    WARNING_BG = "#451a03"     # Dark amber container
    WARNING_TEXT = "#fcd34d"   # Bright amber text

    DANGER = "#be123c"         # Rose/Crimson (WCAG AA compliant > 6.2:1 with white text)
    DANGER_HOVER = "#9f1239"   # Darker Rose (> 8.0:1 with white text)
    DANGER_BG = "#4c0519"      # Dark rose container
    DANGER_TEXT = "#fda4af"    # Bright rose text (> 8.3:1 on BG_ELEVATED)

    # Text Colors (WCAG AA Compliant High Contrast)
    TEXT_PRIMARY = "#f8fafc"   # High contrast crisp white (> 12:1)
    TEXT_SECONDARY = "#cbd5e1" # Soft slate (high contrast > 7:1)
    TEXT_MUTED = "#94a3b8"     # Subtle supporting text (high contrast > 4.5:1)
    TEXT_DISABLED = "#64748b"  # Disabled control state text
    TEXT_ON_PRIMARY = "#ffffff"# Pure white text on accents


class Fonts:
    """Standard typography configuration scaled for readability on Windows."""

    FAMILY = "Segoe UI"

    HERO = (FAMILY, 20, "bold")
    DISPLAY = (FAMILY, 16, "bold")
    TITLE = (FAMILY, 14, "bold")
    SUBTITLE = (FAMILY, 12, "bold")
    HEADING = (FAMILY, 11, "bold")
    BODY = (FAMILY, 10, "normal")
    BODY_BOLD = (FAMILY, 10, "bold")
    CAPTION = (FAMILY, 9, "normal")
    CAPTION_BOLD = (FAMILY, 9, "bold")
    BADGE = (FAMILY, 8, "bold")
    METRIC_VAL = (FAMILY, 16, "bold")
    METRIC_LABEL = (FAMILY, 9, "bold")
    BUTTON = (FAMILY, 10, "bold")
    CODE = ("Consolas", 9, "normal")


def setup_theme(root: tk.Tk) -> ttk.Style:
    """Configure modern TTK styles to match dark theme tokens."""
    root.configure(bg=Colors.BG_DARK)
    style = ttk.Style(root)
    style.theme_use("clam")

    # General Frames
    style.configure("TFrame", background=Colors.BG_DARK)
    style.configure("Sidebar.TFrame", background=Colors.BG_SIDEBAR)
    style.configure("Surface.TFrame", background=Colors.BG_SURFACE)
    style.configure("Elevated.TFrame", background=Colors.BG_ELEVATED)
    style.configure("Card.TFrame", background=Colors.BG_CARD)

    # General Labels
    style.configure("TLabel", background=Colors.BG_DARK, foreground=Colors.TEXT_PRIMARY, font=Fonts.BODY)
    style.configure("Surface.TLabel", background=Colors.BG_SURFACE, foreground=Colors.TEXT_PRIMARY, font=Fonts.BODY)
    style.configure("Title.TLabel", background=Colors.BG_DARK, foreground=Colors.TEXT_PRIMARY, font=Fonts.TITLE)
    style.configure("Subtitle.TLabel", background=Colors.BG_SURFACE, foreground=Colors.TEXT_PRIMARY, font=Fonts.SUBTITLE)
    style.configure("Secondary.TLabel", background=Colors.BG_SURFACE, foreground=Colors.TEXT_SECONDARY, font=Fonts.CAPTION)
    style.configure("Muted.TLabel", background=Colors.BG_SURFACE, foreground=Colors.TEXT_MUTED, font=Fonts.CAPTION)

    # Primary Action Buttons
    style.configure(
        "Primary.TButton",
        background=Colors.PRIMARY,
        foreground=Colors.TEXT_ON_PRIMARY,
        font=Fonts.BODY_BOLD,
        borderwidth=0,
        focuscolor=Colors.TEXT_ON_PRIMARY,
        focusthickness=2,
        padding=(16, 8),
    )
    style.map(
        "Primary.TButton",
        background=[("active", Colors.PRIMARY_HOVER), ("disabled", Colors.BG_HOVER)],
        foreground=[("disabled", Colors.TEXT_DISABLED)],
    )

    # Success (Start) Action Button
    style.configure(
        "Success.TButton",
        background=Colors.SUCCESS,
        foreground=Colors.TEXT_ON_PRIMARY,
        font=Fonts.HEADING,
        borderwidth=0,
        focuscolor=Colors.TEXT_ON_PRIMARY,
        focusthickness=2,
        padding=(18, 10),
    )
    style.map(
        "Success.TButton",
        background=[("active", Colors.SUCCESS_HOVER), ("disabled", Colors.BG_HOVER)],
        foreground=[("disabled", Colors.TEXT_DISABLED)],
    )

    # Danger (Stop) Action Button
    style.configure(
        "Danger.TButton",
        background=Colors.DANGER,
        foreground=Colors.TEXT_ON_PRIMARY,
        font=Fonts.HEADING,
        borderwidth=0,
        focuscolor=Colors.TEXT_ON_PRIMARY,
        focusthickness=2,
        padding=(18, 10),
    )
    style.map(
        "Danger.TButton",
        background=[("active", Colors.DANGER_HOVER), ("disabled", Colors.BG_HOVER)],
        foreground=[("disabled", Colors.TEXT_DISABLED)],
    )

    # Studio's single primary action: the same three colors, larger.
    for name, bg, hover in (("Primary", Colors.PRIMARY, Colors.PRIMARY_HOVER),
                            ("Success", Colors.SUCCESS, Colors.SUCCESS_HOVER),
                            ("Danger", Colors.DANGER, Colors.DANGER_HOVER)):
        style.configure(
            f"Hero{name}.TButton",
            background=bg,
            foreground=Colors.TEXT_ON_PRIMARY,
            font=(Fonts.FAMILY, 13, "bold"),
            borderwidth=0,
            focuscolor=Colors.TEXT_ON_PRIMARY,
            focusthickness=2,
            padding=(24, 14),
        )
        style.map(
            f"Hero{name}.TButton",
            background=[("active", hover), ("disabled", Colors.BG_HOVER)],
            foreground=[("disabled", Colors.TEXT_DISABLED)],
        )

    # Secondary Outline Button
    style.configure(
        "Secondary.TButton",
        background=Colors.BG_ELEVATED,
        foreground=Colors.TEXT_PRIMARY,
        font=Fonts.CAPTION_BOLD,
        borderwidth=1,
        bordercolor=Colors.BORDER,
        focuscolor=Colors.ACCENT_CYAN,
        focusthickness=2,
        padding=(10, 5),
    )
    style.map(
        "Secondary.TButton",
        background=[("active", Colors.BG_HOVER), ("disabled", Colors.BG_SURFACE)],
        foreground=[("disabled", Colors.TEXT_DISABLED)],
    )

    # Combobox style
    style.configure(
        "TCombobox",
        background=Colors.BG_ELEVATED,
        foreground=Colors.TEXT_PRIMARY,
        fieldbackground=Colors.BG_ELEVATED,
        bordercolor=Colors.BORDER,
        darkcolor=Colors.BG_ELEVATED,
        lightcolor=Colors.BG_ELEVATED,
        arrowcolor=Colors.TEXT_PRIMARY,
        padding=6,
        font=Fonts.BODY,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", Colors.BG_ELEVATED), ("disabled", Colors.BG_SURFACE)],
        foreground=[("disabled", Colors.TEXT_DISABLED)],
    )

    # Notebook (Tabs)
    style.configure("TNotebook", background=Colors.BG_DARK, borderwidth=0)
    style.configure(
        "TNotebook.Tab",
        background=Colors.BG_SURFACE,
        foreground=Colors.TEXT_MUTED,
        font=Fonts.BODY_BOLD,
        padding=(20, 10),
        borderwidth=0,
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", Colors.BG_ELEVATED), ("active", Colors.BG_HOVER)],
        foreground=[("selected", Colors.TEXT_PRIMARY), ("active", Colors.TEXT_PRIMARY)],
    )

    # Hidden Notebook (tab headers hidden, page switching controlled by Sidebar)
    try:
        style.layout("Hidden.TNotebook", [("Notebook.client", {"sticky": "nswe"})])
        style.layout("Hidden.TNotebook.Tab", [])
        style.configure("Hidden.TNotebook", background=Colors.BG_DARK, borderwidth=0)
    except Exception:
        pass


    # Sliders (Scale)
    style.configure(
        "Horizontal.TScale",
        background=Colors.BG_SURFACE,
        troughcolor=Colors.BG_ELEVATED,
        slidercolor=Colors.PRIMARY,
        sliderthickness=16,
        borderwidth=0,
    )

    # Checkbuttons
    style.configure(
        "TCheckbutton",
        background=Colors.BG_SURFACE,
        foreground=Colors.TEXT_PRIMARY,
        font=Fonts.BODY_BOLD,
        focuscolor=Colors.ACCENT_CYAN,
    )
    style.map(
        "TCheckbutton",
        background=[("active", Colors.BG_SURFACE)],
        foreground=[("disabled", Colors.TEXT_DISABLED)],
    )

    # Separator
    style.configure("TSeparator", background=Colors.BORDER)

    # Secondary Outline Button
    style.configure(
        "AccentOutline.TButton",
        background=Colors.BG_CARD,
        foreground=Colors.PRIMARY_LIGHT,
        font=Fonts.CAPTION_BOLD,
        borderwidth=1,
        bordercolor=Colors.PRIMARY,
        focuscolor=Colors.PRIMARY_LIGHT,
        focusthickness=2,
        padding=(12, 6),
    )
    style.map(
        "AccentOutline.TButton",
        background=[("active", Colors.PRIMARY), ("disabled", Colors.BG_SURFACE)],
        foreground=[("active", Colors.TEXT_ON_PRIMARY), ("disabled", Colors.TEXT_DISABLED)],
    )

    return style


# Authoritative Engine State Visual Presentation (label, background_color, foreground_color)
ENGINE_STATE_VISUALS: dict[EngineState, tuple[str, str, str]] = {
    EngineState.IDLE: ("● CHƯA NẠP GIỌNG", Colors.BG_ELEVATED, Colors.TEXT_MUTED),
    EngineState.LOADING: ("● ĐANG NẠP…", Colors.WARNING_BG, Colors.WARNING_TEXT),
    EngineState.READY: ("● SẴN SÀNG", Colors.SUCCESS_BG, Colors.SUCCESS_TEXT),
    EngineState.STARTING: ("● ĐANG KHỞI ĐỘNG…", Colors.WARNING_BG, Colors.WARNING_TEXT),
    EngineState.RUNNING: ("● ĐANG CHUYỂN GIỌNG", Colors.SUCCESS, Colors.TEXT_ON_PRIMARY),
    EngineState.STOPPING: ("● ĐANG DỪNG…", Colors.WARNING_BG, Colors.WARNING_TEXT),
    EngineState.ERROR: ("● CÓ LỖI", Colors.DANGER_BG, Colors.DANGER_TEXT),
}


def get_engine_state_visual(state: Optional[EngineState]) -> tuple[str, str, str]:
    """Return authoritative (label, bg_color, fg_color) for an engine state."""
    if state in ENGINE_STATE_VISUALS:
        return ENGINE_STATE_VISUALS[state]
    return ENGINE_STATE_VISUALS[EngineState.IDLE]


def get_rtf_color(rtf: float) -> str:
    """Text color for a real-time factor; the same thresholds in Voice Studio and Diagnostics.

    RTF = inference time / chunk length: >= 1.0 cannot keep up with real time, near 1.0 has no headroom.
    """
    if rtf < 0.85:
        return Colors.SUCCESS_TEXT
    if rtf <= 1.0:
        return Colors.WARNING_TEXT
    return Colors.DANGER_TEXT
