"""Modern sleek dark theme tokens and styles for RealTimeVoiceChanger desktop GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class Colors:
    """Refined color palette tokens with high contrast and modern aesthetics."""

    # Backgrounds
    BG_DARK = "#0d131f"        # Deep slate background
    BG_SURFACE = "#161f2e"     # Card & container background
    BG_ELEVATED = "#1f2b3e"    # Input background / elevated surfaces
    BG_HOVER = "#2a3a52"       # Hover states

    # Borders & Dividers
    BORDER = "#283548"         # Subtle card border
    BORDER_LIGHT = "#3d4f68"   # Active focus border

    # Accent Colors
    PRIMARY = "#3b82f6"        # Modern Blue / Indigo
    PRIMARY_HOVER = "#2563eb"  # Darker Blue
    PRIMARY_LIGHT = "#60a5fa"  # Soft Blue highlight

    SUCCESS = "#10b981"        # Emerald green
    SUCCESS_HOVER = "#059669"  # Darker emerald
    SUCCESS_BG = "#064e3b"     # Dark green container
    SUCCESS_TEXT = "#6ee7b7"   # Bright emerald text

    WARNING = "#f59e0b"        # Amber warning
    WARNING_BG = "#451a03"     # Dark amber container
    WARNING_TEXT = "#fcd34d"   # Bright amber text

    DANGER = "#f43f5e"         # Modern Rose/Crimson
    DANGER_HOVER = "#e11d48"   # Darker Rose
    DANGER_BG = "#4c0519"      # Dark rose container
    DANGER_TEXT = "#fda4af"    # Bright rose text

    # Text Colors (WCAG AA Compliant High Contrast)
    TEXT_PRIMARY = "#f8fafc"   # High contrast crisp white
    TEXT_SECONDARY = "#cbd5e1" # Soft slate (high contrast > 7:1)
    TEXT_MUTED = "#94a3b8"     # Subtle supporting text (high contrast > 4.5:1)
    TEXT_DISABLED = "#64748b"  # Disabled text
    TEXT_ON_PRIMARY = "#ffffff"# Pure white text on accents


class Fonts:
    """Standard typography configuration scaled for readability on Windows."""

    FAMILY = "Segoe UI"

    TITLE = (FAMILY, 15, "bold")
    SUBTITLE = (FAMILY, 12, "bold")
    HEADING = (FAMILY, 11, "bold")
    BODY = (FAMILY, 10, "normal")
    BODY_BOLD = (FAMILY, 10, "bold")
    CAPTION = (FAMILY, 9, "normal")
    CAPTION_BOLD = (FAMILY, 9, "bold")
    BADGE = (FAMILY, 9, "bold")
    METRIC_VAL = (FAMILY, 16, "bold")
    METRIC_LABEL = (FAMILY, 9, "bold")


def setup_theme(root: tk.Tk) -> ttk.Style:
    """Configure modern TTK styles to match dark theme tokens."""
    root.configure(bg=Colors.BG_DARK)
    style = ttk.Style(root)
    style.theme_use("clam")

    # General Frames
    style.configure("TFrame", background=Colors.BG_DARK)
    style.configure("Surface.TFrame", background=Colors.BG_SURFACE)
    style.configure("Elevated.TFrame", background=Colors.BG_ELEVATED)

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
        focuscolor=Colors.PRIMARY,
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
        focuscolor=Colors.SUCCESS,
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
        focuscolor=Colors.DANGER,
        padding=(18, 10),
    )
    style.map(
        "Danger.TButton",
        background=[("active", Colors.DANGER_HOVER), ("disabled", Colors.BG_HOVER)],
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
        focuscolor=Colors.BG_ELEVATED,
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
        focuscolor=Colors.BG_SURFACE,
    )
    style.map(
        "TCheckbutton",
        background=[("active", Colors.BG_SURFACE)],
        foreground=[("disabled", Colors.TEXT_DISABLED)],
    )

    # Separator
    style.configure("TSeparator", background=Colors.BORDER)

    return style
