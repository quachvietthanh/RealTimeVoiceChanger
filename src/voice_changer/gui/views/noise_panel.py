"""Microphone noise suppression control panel."""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Optional

from voice_changer.audio.noise_suppression import NOISE_SUPPRESSION_MODES
from voice_changer.gui.theme import Colors, Fonts
from voice_changer.service import NoiseSuppressionStatus

logger = logging.getLogger(__name__)

# Display mapping for supported modes
MODE_DISPLAY_MAP: dict[str, str] = {
    "light": "Nhẹ (-6 dB) — Phòng yên tĩnh, bảo toàn giọng thì thào",
    "medium": "Vừa (-12 dB) — Quạt máy / PC thông thường (Khuyên dùng)",
    "strong": "Mạnh (-18 dB) — Môi trường ồn lớn, quạt mạnh",
}
DISPLAY_TO_MODE: dict[str, str] = {v: k for k, v in MODE_DISPLAY_MAP.items()}


class NoiseSuppressionPanel(ttk.Frame):
    """Panel for controlling real-time microphone noise suppression before RVC conversion."""

    def __init__(
        self,
        parent: tk.Widget,
        on_noise_changed: Callable[[Optional[bool], Optional[str]], None],
        on_rearm_requested: Optional[Callable[[], None]] = None,
        framed: bool = True,
    ) -> None:
        super().__init__(parent, style="Surface.TFrame")
        self.on_noise_changed = on_noise_changed
        self.on_rearm_requested = on_rearm_requested
        # framed=False: no border or title, for use inside a section that already has both (Voice Studio).
        self._framed = framed

        self.var_enabled = tk.BooleanVar(value=False)
        self._current_mode = "medium"
        self._is_updating = False

        self._build_ui()

    def _build_ui(self) -> None:
        card = tk.Frame(
            self,
            bg=Colors.BG_SURFACE,
            highlightbackground=Colors.BORDER,
            highlightthickness=1 if self._framed else 0,
            padx=14 if self._framed else 16,
            pady=10 if self._framed else 4,
        )
        card.pack(fill="x", expand=True)

        # 1. Header with Title, Status Pill, and Prominent Toggle Button
        header_row = tk.Frame(card, bg=Colors.BG_SURFACE)
        header_row.pack(fill="x", pady=(0, 6))

        # Right side actions: Status indicator + Quick Action Toggle Button
        header_right = tk.Frame(header_row, bg=Colors.BG_SURFACE)
        header_right.pack(side="right")

        self.lbl_status = tk.Label(
            header_right,
            text="● ĐANG TẮT",
            font=Fonts.BADGE,
            bg=Colors.BG_ELEVATED,
            fg=Colors.TEXT_MUTED,
            padx=8,
            pady=3,
        )
        self.lbl_status.pack(side="left", padx=(0, 8))

        self.btn_toggle = tk.Button(
            header_right,
            text="⚪ TẮT",
            font=Fonts.CAPTION_BOLD,
            bg=Colors.BG_ELEVATED,
            fg=Colors.TEXT_MUTED,
            activebackground=Colors.BG_HOVER,
            activeforeground=Colors.TEXT_PRIMARY,
            borderwidth=1,
            relief="flat",
            highlightthickness=2,
            highlightbackground=Colors.BG_SURFACE,
            highlightcolor=Colors.ACCENT_CYAN,
            padx=10,
            pady=3,
            cursor="hand2",
            command=self._on_toggle_button_clicked,
        )
        self.btn_toggle.pack(side="left")

        def _focus_in_toggle(e: Optional[tk.Event] = None) -> None:
            self.btn_toggle.config(highlightbackground=Colors.ACCENT_CYAN)

        def _focus_out_toggle(e: Optional[tk.Event] = None) -> None:
            self.btn_toggle.config(highlightbackground=Colors.BG_SURFACE)

        self.btn_toggle._focus_in = _focus_in_toggle  # type: ignore[attr-defined]
        self.btn_toggle._focus_out = _focus_out_toggle  # type: ignore[attr-defined]
        self.btn_toggle.bind("<FocusIn>", _focus_in_toggle)
        self.btn_toggle.bind("<FocusOut>", _focus_out_toggle)
        self.btn_toggle.bind("<Return>", lambda e: (self.btn_toggle.invoke(), "break")[1])

        lbl_title = tk.Label(
            header_row,
            text="Khử tiếng ồn microphone" if self._framed else "",
            font=Fonts.SUBTITLE,
            bg=Colors.BG_SURFACE,
            fg=Colors.TEXT_PRIMARY,
            anchor="w",
        )
        lbl_title.pack(side="left", fill="x", expand=True)

        # 2. Short functional explanation
        self.lbl_desc = tk.Label(
            card,
            text=(
                "Lọc tiếng ồn nền đều đặn (quạt gió, ù điện, tiếng rít) từ microphone "
                "trước khi đưa vào chuyển đổi giọng nói RVC. Hoạt động trên CPU (không tốn VRAM)."
            ),
            font=Fonts.CAPTION,
            bg=Colors.BG_SURFACE,
            fg=Colors.TEXT_MUTED,
            justify="left",
            anchor="w",
        )
        self.lbl_desc.pack(fill="x", pady=(0, 6))

        # 3. Error Banner (Shown when failed or configuration error occurs)
        self.frame_error = tk.Frame(
            card,
            bg=Colors.DANGER_BG,
            highlightbackground=Colors.DANGER,
            highlightthickness=1,
            padx=10,
            pady=6,
        )
        self.lbl_error_msg = tk.Label(
            self.frame_error,
            text="",
            font=Fonts.CAPTION_BOLD,
            bg=Colors.DANGER_BG,
            fg=Colors.DANGER_TEXT,
            anchor="w",
            justify="left",
        )
        self.lbl_error_msg.pack(fill="x")

        self.lbl_error_hint = tk.Label(
            self.frame_error,
            text="",
            font=Fonts.CAPTION,
            bg=Colors.DANGER_BG,
            fg=Colors.TEXT_PRIMARY,
            anchor="w",
            justify="left",
        )
        self.lbl_error_hint.pack(fill="x", pady=(2, 4))

        self.btn_rearm = tk.Button(
            self.frame_error,
            text="Kích hoạt lại bộ lọc",
            font=Fonts.CAPTION_BOLD,
            bg=Colors.DANGER,
            fg=Colors.TEXT_ON_PRIMARY,
            borderwidth=0,
            highlightthickness=2,
            highlightbackground=Colors.DANGER_BG,
            highlightcolor=Colors.ACCENT_CYAN,
            padx=8,
            pady=3,
            cursor="hand2",
            command=self._on_rearm,
        )
        self.btn_rearm.pack(anchor="w", pady=(2, 0))

        def _focus_in_rearm(e: Optional[tk.Event] = None) -> None:
            self.btn_rearm.config(highlightbackground=Colors.ACCENT_CYAN)

        def _focus_out_rearm(e: Optional[tk.Event] = None) -> None:
            self.btn_rearm.config(highlightbackground=Colors.DANGER_BG)

        self.btn_rearm._focus_in = _focus_in_rearm  # type: ignore[attr-defined]
        self.btn_rearm._focus_out = _focus_out_rearm  # type: ignore[attr-defined]
        self.btn_rearm.bind("<FocusIn>", _focus_in_rearm)
        self.btn_rearm.bind("<FocusOut>", _focus_out_rearm)
        self.btn_rearm.bind("<Return>", lambda e: (self.btn_rearm.invoke(), "break")[1])

        # 4. Controls: Checkbox and Mode Selector
        ctrl_frame = tk.Frame(card, bg=Colors.BG_SURFACE)
        ctrl_frame.pack(fill="x", pady=(0, 6))

        # Checkbutton as secondary direct toggle
        self.chk_enabled = ttk.Checkbutton(
            ctrl_frame,
            text="Kích hoạt bộ lọc tiếng ồn trên micro",
            variable=self.var_enabled,
            command=self._on_toggle,
            style="TCheckbutton",
        )
        self.chk_enabled.pack(anchor="w", pady=(0, 4))

        # Mode Selector
        mode_box = tk.Frame(ctrl_frame, bg=Colors.BG_SURFACE)
        mode_box.pack(fill="x")

        lbl_mode = tk.Label(
            mode_box,
            text="Mức độ khử ồn (Suppression Mode):",
            font=Fonts.BODY_BOLD,
            bg=Colors.BG_SURFACE,
            fg=Colors.TEXT_PRIMARY,
        )
        lbl_mode.pack(anchor="w", pady=(0, 2))

        combo_values = [MODE_DISPLAY_MAP.get(m, m) for m in NOISE_SUPPRESSION_MODES.keys()]
        self.combo_mode = ttk.Combobox(
            mode_box,
            values=combo_values,
            state="readonly",
            style="TCombobox",
        )
        self.combo_mode.set(MODE_DISPLAY_MAP.get("medium", "medium"))
        self.combo_mode.pack(fill="x", ipady=2)
        self.combo_mode.bind("<<ComboboxSelected>>", self._on_mode_selected)

        # 5. Latency & Algorithm Details
        self.lbl_latency_info = tk.Label(
            card,
            text="Độ trễ bổ sung: +20.0 ms (cố định) | Thuật toán: STFT + Wiener Filter (MCRA)",
            font=Fonts.CAPTION,
            bg=Colors.BG_SURFACE,
            fg=Colors.TEXT_MUTED,
            anchor="w",
        )
        self.lbl_latency_info.pack(fill="x", pady=(3, 6))

        # 6. Quality Warning Box
        self.frame_warn = tk.Frame(
            card,
            bg=Colors.WARNING_BG,
            highlightbackground=Colors.WARNING,
            highlightthickness=1,
            padx=8,
            pady=4,
        )
        self.frame_warn.pack(fill="x")

        self.lbl_warn = tk.Label(
            self.frame_warn,
            text=(
                "⚠️ Lưu ý: Mức lọc quá mạnh (-18 dB) có thể làm mờ âm gió, phụ âm nhẹ "
                "và biến dạng các âm sắc thì thào tự nhiên."
            ),
            font=Fonts.CAPTION,
            bg=Colors.WARNING_BG,
            fg=Colors.WARNING_TEXT,
            justify="left",
            anchor="w",
        )
        self.lbl_warn.pack(fill="x")

        # Responsive wrapping
        card.bind("<Configure>", lambda e: self._on_resize(e.width))

    def _on_resize(self, width: int) -> None:
        """Dynamically adjust label wrap lengths on window resize."""
        avail_w = max(200, width - 36)
        self.lbl_desc.config(wraplength=avail_w)
        self.lbl_warn.config(wraplength=max(200, width - 56))
        self.lbl_error_msg.config(wraplength=max(200, width - 56))
        self.lbl_error_hint.config(wraplength=max(200, width - 56))

    def _update_toggle_button_visual(self, enabled: bool) -> None:
        """Update toggle button appearance to clearly reflect ON/OFF state."""
        if enabled:
            self.btn_toggle.config(
                text="🟢 ĐÃ BẬT",
                bg=Colors.SUCCESS,
                fg=Colors.TEXT_ON_PRIMARY,
                activebackground=Colors.SUCCESS_HOVER,
                activeforeground=Colors.TEXT_ON_PRIMARY,
            )
        else:
            self.btn_toggle.config(
                text="⚪ TẮT",
                bg=Colors.BG_ELEVATED,
                fg=Colors.TEXT_MUTED,
                activebackground=Colors.BG_HOVER,
                activeforeground=Colors.TEXT_PRIMARY,
            )

    def set_status(self, status: NoiseSuppressionStatus) -> None:
        """Update UI controls to match backend NoiseSuppressionStatus."""
        self._is_updating = True
        try:
            self.var_enabled.set(status.enabled)
            self._current_mode = status.mode
            self._update_toggle_button_visual(status.enabled)

            display_text = MODE_DISPLAY_MAP.get(status.mode, status.mode)
            self.combo_mode.set(display_text)

            # Update status badge
            if status.failed:
                self.lbl_status.config(
                    text="⚠️ BỘ LỌC TỰ NGẮT DO LỖI",
                    bg=Colors.DANGER_BG,
                    fg=Colors.DANGER_TEXT,
                )
                self.show_error(
                    "Bộ lọc đã tự ngắt sau nhiều lần lỗi liên tiếp.",
                    "Nhấn 'Kích hoạt lại' hoặc khởi động lại luồng âm thanh.",
                )
            elif status.active:
                self.lbl_status.config(
                    text="● ĐANG LỌC TIẾNG ỒN",
                    bg=Colors.SUCCESS_BG,
                    fg=Colors.SUCCESS_TEXT,
                )
                self.hide_error()
            elif status.enabled:
                self.lbl_status.config(
                    text="● ĐÃ BẬT (CHỜ STREAMING)",
                    bg=Colors.PRIMARY,
                    fg=Colors.TEXT_ON_PRIMARY,
                )
                self.hide_error()
            else:
                self.lbl_status.config(
                    text="● ĐANG TẮT",
                    bg=Colors.BG_ELEVATED,
                    fg=Colors.TEXT_MUTED,
                )
                self.hide_error()

            # Latency display
            lat = status.latency_ms if status.latency_ms is not None else 20.0
            self.lbl_latency_info.config(
                text=f"Độ trễ bổ sung: +{lat:.1f} ms (cố định) | Thuật toán: STFT + Wiener Filter (MCRA)"
            )
        finally:
            self._is_updating = False

    def sync_live_state(
        self,
        status: NoiseSuppressionStatus,
        metrics: Optional[dict[str, Any]] = None,
    ) -> None:
        """Sync live streaming metrics (CPU time) and status without disrupting user interaction."""
        self.set_status(status)

        if metrics and status.active:
            avg_ms = metrics.get("avg_noise_suppression_ms")
            if avg_ms is not None and avg_ms > 0:
                lat = status.latency_ms if status.latency_ms is not None else 20.0
                self.lbl_latency_info.config(
                    text=f"Độ trễ bổ sung: +{lat:.1f} ms | CPU lọc ồn: ~{avg_ms:.2f} ms/chunk | STFT + Wiener"
                )

    def show_error(self, message: str, hint: Optional[str] = None) -> None:
        """Display actionable error banner."""
        self.lbl_error_msg.config(text=f"❌ {message}")
        self.lbl_error_hint.config(text=f"Gợi ý: {hint}" if hint else "")
        self.frame_error.pack(fill="x", pady=(0, 6), before=self.chk_enabled.master)

    def hide_error(self) -> None:
        """Hide error banner."""
        self.frame_error.pack_forget()

    def _on_toggle_button_clicked(self) -> None:
        """Invoked when user clicks the prominent toggle button in header."""
        if self._is_updating:
            return
        new_val = not self.var_enabled.get()
        self.var_enabled.set(new_val)
        self._on_toggle()

    def _on_toggle(self) -> None:
        if self._is_updating:
            return
        new_enabled = self.var_enabled.get()
        self._update_toggle_button_visual(new_enabled)
        self.on_noise_changed(new_enabled, None)

    def _on_mode_selected(self, event: Any = None) -> None:
        if self._is_updating:
            return
        selected_label = self.combo_mode.get()
        mode = DISPLAY_TO_MODE.get(selected_label, "medium")
        self._current_mode = mode
        self.on_noise_changed(None, mode)

    def _on_rearm(self) -> None:
        """User clicked re-arm button after failures."""
        self.hide_error()
        if self.on_rearm_requested:
            self.on_rearm_requested()
        else:
            self.on_noise_changed(True, None)
