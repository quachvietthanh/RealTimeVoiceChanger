"""Live engine control, status monitor, and metrics dashboard."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable, Optional

from voice_changer.gui.theme import Colors, Fonts
from voice_changer.service import EngineState, EngineStatus, ErrorInfo


class ControlsPanel(ttk.Frame):
    """Clean, balanced panel for conversion action, status indicator, and live metrics."""

    def __init__(
        self,
        parent: tk.Widget,
        on_start_requested: Callable[[], None],
        on_stop_requested: Callable[[], None],
    ) -> None:
        super().__init__(parent, style="Surface.TFrame")
        self.on_start_requested = on_start_requested
        self.on_stop_requested = on_stop_requested

        self._current_state = EngineState.IDLE
        self._last_error: Optional[ErrorInfo] = None

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

        # Header Row: Title & Engine State Pill
        header_row = tk.Frame(card, bg=Colors.BG_SURFACE)
        header_row.pack(fill="x", pady=(0, 10))

        lbl_title = tk.Label(
            header_row,
            text="Điều khiển & Giám sát (Controls & Status)",
            font=Fonts.SUBTITLE,
            bg=Colors.BG_SURFACE,
            fg=Colors.TEXT_PRIMARY,
        )
        lbl_title.pack(side="left")

        self.pill_state = tk.Label(
            header_row,
            text="● CHƯA NẠP MODEL",
            font=Fonts.BADGE,
            padx=10,
            pady=3,
            bg=Colors.BG_ELEVATED,
            fg=Colors.TEXT_MUTED,
            relief="flat",
        )
        self.pill_state.pack(side="right")

        # Action Button (Start / Stop)
        btn_box = tk.Frame(card, bg=Colors.BG_SURFACE)
        btn_box.pack(fill="x", pady=(0, 12))

        self.btn_action = ttk.Button(
            btn_box,
            text="▶ BẮT ĐẦU CHUYỂN ĐỔI",
            style="Success.TButton",
            command=self._on_action_clicked,
        )
        self.btn_action.pack(fill="x")

        # Error notification banner (hidden by default)
        self.banner_error = tk.Frame(
            card,
            bg=Colors.DANGER_BG,
            highlightbackground=Colors.DANGER,
            highlightthickness=1,
            padx=12,
            pady=8,
        )
        self.lbl_error_msg = tk.Label(
            self.banner_error,
            text="",
            font=Fonts.BODY_BOLD,
            bg=Colors.DANGER_BG,
            fg=Colors.TEXT_PRIMARY,
            anchor="w",
            justify="left",
        )
        self.lbl_error_msg.pack(fill="x")

        self.lbl_error_hint = tk.Label(
            self.banner_error,
            text="",
            font=Fonts.CAPTION,
            bg=Colors.DANGER_BG,
            fg=Colors.DANGER_TEXT,
            anchor="w",
            justify="left",
        )
        self.lbl_error_hint.pack(fill="x", pady=(2, 4))

        btn_err_details = tk.Button(
            self.banner_error,
            text="Xem chi tiết kỹ thuật",
            font=Fonts.CAPTION_BOLD,
            bg=Colors.DANGER,
            fg=Colors.TEXT_ON_PRIMARY,
            borderwidth=0,
            padx=8,
            pady=2,
            cursor="hand2",
            command=self._show_error_dialog,
        )
        btn_err_details.pack(anchor="w", pady=(2, 0))

        # Metrics Dashboard Grid
        lbl_metrics = tk.Label(
            card,
            text="Chỉ số hiệu năng thời gian thực (Live Metrics):",
            font=Fonts.CAPTION_BOLD,
            bg=Colors.BG_SURFACE,
            fg=Colors.TEXT_MUTED,
        )
        lbl_metrics.pack(anchor="w", pady=(0, 6))

        metrics_grid = tk.Frame(card, bg=Colors.BG_SURFACE)
        metrics_grid.pack(fill="x")
        metrics_grid.columnconfigure(0, weight=1, uniform="m")
        metrics_grid.columnconfigure(1, weight=1, uniform="m")
        metrics_grid.columnconfigure(2, weight=1, uniform="m")

        # 6 Metric Cards
        self.card_rtf, self.val_rtf = self._create_metric_card(
            metrics_grid, row=0, col=0, label="TỐC ĐỘ RTF", default_val="--"
        )
        self.card_gpu, self.val_gpu = self._create_metric_card(
            metrics_grid, row=0, col=1, label="GPU SUY LUẬN", default_val="--"
        )
        self.card_e2e, self.val_e2e = self._create_metric_card(
            metrics_grid, row=0, col=2, label="ĐỘ TRỄ ƯỚC TÍNH", default_val="--"
        )
        self.card_buf, self.val_buf = self._create_metric_card(
            metrics_grid, row=1, col=0, label="BUFFER PHÁT", default_val="--"
        )
        self.card_chunks, self.val_chunks = self._create_metric_card(
            metrics_grid, row=1, col=1, label="CHUNKS (OK/RƠI)", default_val="--"
        )
        self.card_und, self.val_und = self._create_metric_card(
            metrics_grid, row=1, col=2, label="GIÁN ĐOẠN ÂM", default_val="--"
        )

    def _create_metric_card(
        self,
        parent: tk.Frame,
        row: int,
        col: int,
        label: str,
        default_val: str,
    ) -> tuple[tk.Frame, tk.Label]:
        card = tk.Frame(
            parent,
            bg=Colors.BG_ELEVATED,
            highlightbackground=Colors.BORDER,
            highlightthickness=1,
            padx=6,
            pady=8,
        )
        card.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")

        lbl_val = tk.Label(
            card,
            text=default_val,
            font=Fonts.METRIC_VAL,
            bg=Colors.BG_ELEVATED,
            fg=Colors.TEXT_PRIMARY,
        )
        lbl_val.pack()

        lbl_desc = tk.Label(
            card,
            text=label,
            font=Fonts.METRIC_LABEL,
            bg=Colors.BG_ELEVATED,
            fg=Colors.TEXT_MUTED,
        )
        lbl_desc.pack(pady=(2, 0))

        return card, lbl_val

    def _on_action_clicked(self) -> None:
        if self._current_state in (EngineState.READY, EngineState.ERROR):
            self.on_start_requested()
        elif self._current_state == EngineState.RUNNING:
            self.on_stop_requested()

    def update_status(self, status: EngineStatus) -> None:
        """Update button state and status pills based on EngineStatus."""
        self._current_state = status.state
        self._last_error = status.last_error

        # Update status pill and button
        state = status.state
        if state == EngineState.IDLE:
            self.pill_state.config(text="● CHƯA NẠP MODEL", bg=Colors.BG_ELEVATED, fg=Colors.TEXT_MUTED)
            self.btn_action.config(text="Vui lòng nạp Voice Profile trước", state="disabled", style="Primary.TButton")
        elif state == EngineState.LOADING:
            self.pill_state.config(text="● ĐANG NẠP GPU...", bg=Colors.WARNING_BG, fg=Colors.WARNING_TEXT)
            self.btn_action.config(text="Đang nạp mô hình vào GPU...", state="disabled", style="Primary.TButton")
        elif state == EngineState.READY:
            self.pill_state.config(text="● SẴN SÀNG", bg=Colors.SUCCESS_BG, fg=Colors.SUCCESS_TEXT)
            self.btn_action.config(text="▶ BẮT ĐẦU CHUYỂN ĐỔI", state="normal", style="Success.TButton")
        elif state == EngineState.STARTING:
            self.pill_state.config(text="● ĐANG MỞ LUỒNG...", bg=Colors.WARNING_BG, fg=Colors.WARNING_TEXT)
            self.btn_action.config(text="Đang kết nối thiết bị...", state="disabled", style="Success.TButton")
        elif state == EngineState.RUNNING:
            self.pill_state.config(text="● ĐANG CHẠY (STREAMING)", bg=Colors.SUCCESS_BG, fg=Colors.SUCCESS_TEXT)
            self.btn_action.config(text="⏹ DỪNG CHUYỂN ĐỔI", state="normal", style="Danger.TButton")
        elif state == EngineState.STOPPING:
            self.pill_state.config(text="● ĐANG DỪNG...", bg=Colors.WARNING_BG, fg=Colors.WARNING_TEXT)
            self.btn_action.config(text="Đang dừng luồng âm thanh...", state="disabled", style="Danger.TButton")
        elif state == EngineState.ERROR:
            self.pill_state.config(text="● LỖI HỆ THỐNG", bg=Colors.DANGER_BG, fg=Colors.DANGER_TEXT)
            self.btn_action.config(text="▶ THỬ LẠI", state="normal", style="Success.TButton")

        # Show or hide error banner
        if status.last_error and (state == EngineState.ERROR or status.health_problems):
            self.lbl_error_msg.config(text=f"❌ {status.last_error.message}")
            self.lbl_error_hint.config(text=status.last_error.hint or "Hãy nhấn 'Dừng' và kiểm tra lại thiết bị.")
            self.banner_error.pack(fill="x", pady=(0, 10), before=self.btn_action.master)
        else:
            self.banner_error.pack_forget()

    def update_metrics(self, metrics: Optional[dict[str, Any]]) -> None:
        """Update metrics cards with live numerical data."""
        if not metrics:
            self.val_rtf.config(text="--", fg=Colors.TEXT_PRIMARY)
            self.val_gpu.config(text="--")
            self.val_e2e.config(text="--")
            self.val_buf.config(text="--")
            self.val_chunks.config(text="--", fg=Colors.TEXT_PRIMARY)
            self.val_und.config(text="--", fg=Colors.TEXT_PRIMARY)
            return

        # 1. RTF
        rtf = metrics.get("rtf")
        if rtf is not None:
            self.val_rtf.config(text=f"{rtf:.3f}")
            if rtf < 0.85:
                self.val_rtf.config(fg=Colors.SUCCESS)
            elif rtf <= 1.0:
                self.val_rtf.config(fg=Colors.WARNING)
            else:
                self.val_rtf.config(fg=Colors.DANGER)
        else:
            self.val_rtf.config(text="--", fg=Colors.TEXT_PRIMARY)

        # 2. GPU Inference time
        avg_inf = metrics.get("avg_inference_ms")
        if avg_inf is not None:
            self.val_gpu.config(text=f"{avg_inf:.0f} ms")
        else:
            self.val_gpu.config(text="--")

        # 3. Estimated E2E Latency
        e2e = metrics.get("avg_software_e2e_ms")
        buf = metrics.get("current_buffer_ms", 0.0)
        if e2e is not None:
            total_est = e2e + buf
            self.val_e2e.config(text=f"~{total_est:.0f} ms")
        else:
            self.val_e2e.config(text="--")

        # 4. Playback Buffer
        if buf is not None:
            self.val_buf.config(text=f"{buf:.0f} ms")
        else:
            self.val_buf.config(text="--")

        # 5. Chunks converted / dropped
        conv = metrics.get("chunks_converted", 0)
        drop = metrics.get("chunks_dropped", 0)
        self.val_chunks.config(text=f"{conv} / {drop}")
        if drop > 0:
            self.val_chunks.config(fg=Colors.WARNING)
        else:
            self.val_chunks.config(fg=Colors.TEXT_PRIMARY)

        # 6. Steady underflows (audible gaps)
        und = metrics.get("steady_underflows", 0)
        self.val_und.config(text=str(und))
        if und > 0:
            self.val_und.config(fg=Colors.DANGER)
        else:
            self.val_und.config(fg=Colors.SUCCESS)

    def _show_error_dialog(self) -> None:
        if not self._last_error:
            return
        details = (
            f"Mã lỗi (Code): {self._last_error.code}\n"
            f"Phân loại (Category): {self._last_error.category}\n"
            f"Thời điểm: {self._last_error.timestamp}\n\n"
            f"Chi tiết kỹ thuật:\n{self._last_error.details or 'Không có chi tiết thêm.'}\n\n"
            f"File log chi tiết ghi tại:\n%LOCALAPPDATA%\\RealTimeVoiceChanger\\logs\\backend.log"
        )
        messagebox.showerror(
            title="Chi tiết lỗi hệ thống",
            message=self._last_error.message,
            detail=details,
        )
