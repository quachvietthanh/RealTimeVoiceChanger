"""Conversion configuration and tuning parameters panel."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from voice_changer.conversion.base import ConversionConfig
from voice_changer.gui.theme import Colors, Fonts


class TuningPanel(ttk.Frame):
    """Clean, balanced panel for adjusting pitch shift, feature retrieval, and smoothing."""

    def __init__(
        self,
        parent: tk.Widget,
        on_config_changed: Callable[[dict[str, float | int]], None],
        on_reset_requested: Callable[[], None],
    ) -> None:
        super().__init__(parent, style="Surface.TFrame")
        self.on_config_changed = on_config_changed
        self.on_reset_requested = on_reset_requested

        self._active_config: Optional[ConversionConfig] = None
        self._updating = False

        self._build_ui()

    def _build_ui(self) -> None:
        card = tk.Frame(
            self,
            bg=Colors.BG_SURFACE,
            highlightbackground=Colors.BORDER,
            highlightthickness=1,
            padx=16,
            pady=14,
        )
        card.pack(fill="x", expand=True)

        # Header with Reset button
        header_row = tk.Frame(card, bg=Colors.BG_SURFACE)
        header_row.pack(fill="x", pady=(0, 8))

        lbl_title = tk.Label(
            header_row,
            text="Tinh chỉnh giọng (Voice Tuning)",
            font=Fonts.SUBTITLE,
            bg=Colors.BG_SURFACE,
            fg=Colors.TEXT_PRIMARY,
        )
        lbl_title.pack(side="left")

        self.btn_reset = ttk.Button(
            header_row,
            text="↩️ Mặc định",
            style="Secondary.TButton",
            command=self.on_reset_requested,
        )
        self.btn_reset.pack(side="right")

        # 1. Pitch Shift Slider (-24 to +24 semitones)
        self.val_pitch = tk.DoubleVar(value=12.0)
        self.scale_pitch, self.lbl_pitch_val = self._create_slider(
            card,
            title="Độ dịch cao độ (Pitch Shift):",
            subtext="Nam ➔ Nữ: thường từ +12 đến +16 bán cung",
            var=self.val_pitch,
            from_val=-24.0,
            to_val=24.0,
            formatter=lambda v: f"{v:+.0f} bán cung",
        )

        # 2. Index Rate Slider (0.00 to 1.00)
        self.val_index = tk.DoubleVar(value=0.5)
        self.scale_index, self.lbl_index_val = self._create_slider(
            card,
            title="Tỷ lệ trích xuất đặc trưng (Index Rate):",
            subtext="0.0: thuần giọng gốc | 1.0: tối đa đặc trưng giọng đích",
            var=self.val_index,
            from_val=0.0,
            to_val=1.0,
            formatter=lambda v: f"{v:.2f}",
        )

        # 3. Filter Radius Slider (0 to 7)
        self.val_filter = tk.IntVar(value=3)
        self.scale_filter, self.lbl_filter_val = self._create_slider(
            card,
            title="Lọc trung vị F0 (Filter Radius):",
            subtext=">= 3 giúp làm mượt và giảm độ run giọng nói",
            var=self.val_filter,
            from_val=0,
            to_val=7,
            formatter=lambda v: f"{int(v)}",
        )

        # 4. Protect Rate Slider (0.00 to 0.50)
        self.val_protect = tk.DoubleVar(value=0.33)
        self.scale_protect, self.lbl_protect_val = self._create_slider(
            card,
            title="Bảo vệ phụ âm & hơi thở (Protect Rate):",
            subtext="0.33: bảo vệ tự nhiên | 0.50: tắt bảo vệ",
            var=self.val_protect,
            from_val=0.0,
            to_val=0.50,
            formatter=lambda v: f"{v:.2f}",
        )

        # Footer footnote
        footer_row = tk.Frame(card, bg=Colors.BG_SURFACE)
        footer_row.pack(fill="x", pady=(6, 0))

        tk.Label(
            footer_row,
            text="Thuật toán F0: RMVPE (Cố định theo chuẩn dự án)",
            font=Fonts.CAPTION,
            bg=Colors.BG_SURFACE,
            fg=Colors.TEXT_MUTED,
        ).pack(side="left")

    def _create_slider(
        self,
        parent: tk.Frame,
        title: str,
        subtext: str,
        var: tk.Variable,
        from_val: float,
        to_val: float,
        formatter: Callable[[float], str],
    ) -> tuple[ttk.Scale, tk.Label]:
        frame = tk.Frame(parent, bg=Colors.BG_SURFACE)
        frame.pack(fill="x", pady=4)

        # Label and value header
        row = tk.Frame(frame, bg=Colors.BG_SURFACE)
        row.pack(fill="x")

        tk.Label(
            row,
            text=title,
            font=Fonts.BODY_BOLD,
            bg=Colors.BG_SURFACE,
            fg=Colors.TEXT_PRIMARY,
        ).pack(side="left")

        lbl_val = tk.Label(
            row,
            text=formatter(var.get()),
            font=Fonts.BODY_BOLD,
            bg=Colors.BG_SURFACE,
            fg=Colors.PRIMARY_LIGHT,
        )
        lbl_val.pack(side="right")

        # Subtext hint
        tk.Label(
            frame,
            text=subtext,
            font=Fonts.CAPTION,
            bg=Colors.BG_SURFACE,
            fg=Colors.TEXT_MUTED,
        ).pack(anchor="w", pady=(0, 1))

        # Slider Scale
        scale = ttk.Scale(
            frame,
            from_=from_val,
            to=to_val,
            variable=var,
            orient="horizontal",
            style="Horizontal.TScale",
        )
        scale.pack(fill="x", pady=2)

        def on_change(*args: object) -> None:
            if not self._updating:
                val = var.get()
                lbl_val.config(text=formatter(val))

        var.trace_add("write", on_change)
        scale.bind("<ButtonRelease-1>", lambda e: self._on_slider_released())

        return scale, lbl_val

    def set_config(self, config: Optional[ConversionConfig]) -> None:
        """Update slider values to reflect active config."""
        self._active_config = config
        if config is None:
            return

        self._updating = True
        try:
            self.val_pitch.set(round(config.pitch_shift))
            self.lbl_pitch_val.config(text=f"{config.pitch_shift:+.0f} bán cung")

            self.val_index.set(round(config.index_rate, 2))
            self.lbl_index_val.config(text=f"{config.index_rate:.2f}")

            self.val_filter.set(int(config.filter_radius))
            self.lbl_filter_val.config(text=str(config.filter_radius))

            self.val_protect.set(round(config.protect_rate, 2))
            self.lbl_protect_val.config(text=f"{config.protect_rate:.2f}")
        finally:
            self._updating = False

    def _on_slider_released(self) -> None:
        if self._updating:
            return
        changes: dict[str, float | int] = {
            "pitch_shift": float(round(self.val_pitch.get())),
            "index_rate": float(round(self.val_index.get(), 2)),
            "filter_radius": int(self.val_filter.get()),
            "protect_rate": float(round(self.val_protect.get(), 2)),
        }
        self.on_config_changed(changes)
