"""Advanced / Troubleshooting: detailed metrics, stream parameters, the last error and the log file.

Metrics are labelled with where they come from (presentation.telemetry_label): the running session, or the
last session that ended. Values that are not measurements (averages before the first chunk, the playback
buffer after stop) are shown as "--". Stream parameters are read from the session's AudioSettings, or from
the AudioSettings defaults when nothing is running.
"""

from __future__ import annotations

import logging
import os
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Optional

from voice_changer.gui.dpi import px
from voice_changer.gui.presentation import stream_parameters, telemetry_label
from voice_changer.gui.theme import Colors, Fonts, get_rtf_color
from voice_changer.service import EngineStatus, ErrorInfo

logger = logging.getLogger(__name__)


class DiagnosticsPanel(ttk.Frame):
    """Performance telemetry, stream parameters, last error and log access for troubleshooting."""

    def __init__(self, parent: tk.Widget, log_path: Optional[str] = None) -> None:
        super().__init__(parent, style="TFrame")
        self.log_path = log_path or "logs/backend.log"
        self._last_error: Optional[ErrorInfo] = None
        self._last_card_error_width: int = 0

        self._build_ui()

    def _build_ui(self) -> None:
        canvas = tk.Canvas(self, bg=Colors.BG_DARK, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        page = tk.Frame(canvas, bg=Colors.BG_DARK)
        page.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas_window = canvas.create_window((0, 0), window=page, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_mousewheel(event: tk.Event) -> None:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        canvas.pack(side="left", fill="both", expand=True, padx=(20, 0), pady=14)
        scrollbar.pack(side="right", fill="y", padx=(0, 12), pady=14)
        bg = Colors.BG_SURFACE

        # 1. Telemetry
        card_telemetry = tk.Frame(page, bg=bg, highlightbackground=Colors.BORDER, highlightthickness=1,
                                  padx=16, pady=14)
        card_telemetry.pack(fill="x", pady=(0, 12))
        head = tk.Frame(card_telemetry, bg=bg)
        head.pack(fill="x", pady=(0, 10))
        tk.Label(head, text="Hiệu năng", font=Fonts.HEADING, bg=bg, fg=Colors.TEXT_PRIMARY).pack(side="left")
        self.lbl_telemetry_source = tk.Label(head, text=telemetry_label(None), font=Fonts.CAPTION_BOLD, bg=bg,
                                             fg=Colors.TEXT_MUTED)
        self.lbl_telemetry_source.pack(side="right")

        grid_m = tk.Frame(card_telemetry, bg=bg)
        grid_m.pack(fill="x")
        for c in range(3):
            grid_m.columnconfigure(c, weight=1, uniform="diag_m")
        self.val_diag_rtf = self._build_stat_box(grid_m, 0, 0, "RTF (CẦN < 1.0)", "--")
        self.val_diag_gpu = self._build_stat_box(grid_m, 0, 1, "SUY LUẬN TB / CHUNK", "--")
        self.val_diag_e2e = self._build_stat_box(grid_m, 0, 2, "ĐỘ TRỄ PHẦN MỀM (ƯỚC TÍNH)", "--")
        self.val_diag_buf = self._build_stat_box(grid_m, 1, 0, "BUFFER PHÁT HIỆN TẠI", "--")
        self.val_diag_chunks = self._build_stat_box(grid_m, 1, 1, "CHUNK (THÀNH CÔNG / RƠI)", "--")
        self.val_diag_und = self._build_stat_box(grid_m, 1, 2, "GIÁN ĐOẠN (UNDERFLOW)", "--")
        self.lbl_latency_note = tk.Label(
            card_telemetry, font=Fonts.CAPTION, bg=bg, fg=Colors.TEXT_MUTED, anchor="w", justify="left",
            text="Độ trễ phần mềm không gồm độ trễ của driver/thiết bị âm thanh; độ trễ thực tế nghe được lớn hơn.")
        self.lbl_latency_note.pack(fill="x", pady=(8, 0))

        # 2. Stream parameters (session values, or the defaults the next session will use)
        card_sys = tk.Frame(page, bg=bg, highlightbackground=Colors.BORDER, highlightthickness=1, padx=16, pady=14)
        card_sys.pack(fill="x", pady=(0, 12))
        head2 = tk.Frame(card_sys, bg=bg)
        head2.pack(fill="x", pady=(0, 8))
        tk.Label(head2, text="Thông số luồng", font=Fonts.HEADING, bg=bg, fg=Colors.TEXT_PRIMARY).pack(side="left")
        self.lbl_params_source = tk.Label(head2, text="", font=Fonts.CAPTION_BOLD, bg=bg, fg=Colors.TEXT_MUTED)
        self.lbl_params_source.pack(side="right")
        self.params_grid = tk.Frame(card_sys, bg=bg)
        self.params_grid.pack(fill="x")
        self.params_grid.columnconfigure(0, weight=1, uniform="sys_col")
        self.params_grid.columnconfigure(1, weight=1, uniform="sys_col")
        self._param_values: dict[str, tk.Label] = {}
        rows = stream_parameters() + [("Thuật toán F0", "--"), ("Độ trễ ghép chunk + khử ồn", "--")]
        for i, (k, v) in enumerate(rows):
            box = tk.Frame(self.params_grid, bg=bg, pady=3)
            box.grid(row=i // 2, column=i % 2, sticky="w", padx=4)
            tk.Label(box, text=f"{k}:", font=Fonts.CAPTION_BOLD, bg=bg, fg=Colors.TEXT_MUTED).pack(side="left")
            val = tk.Label(box, text=f" {v}", font=Fonts.CAPTION, bg=bg, fg=Colors.TEXT_PRIMARY)
            val.pack(side="left")
            self._param_values[k] = val
        self._show_params(None)

        # 3. Last error
        self.card_error = tk.Frame(page, bg=Colors.DANGER_BG, highlightbackground=Colors.DANGER,
                                   highlightthickness=1, padx=16, pady=14)
        self.card_error.bind("<Configure>", lambda e: self._on_card_error_configure(e.width))
        tk.Label(self.card_error, text="Lỗi gần nhất", font=Fonts.HEADING, bg=Colors.DANGER_BG,
                 fg=Colors.TEXT_PRIMARY).pack(anchor="w", pady=(0, 6))
        self.lbl_diag_err_msg = tk.Label(self.card_error, text="", font=Fonts.BODY_BOLD, bg=Colors.DANGER_BG,
                                         fg=Colors.TEXT_PRIMARY, anchor="w", justify="left", wraplength=600)
        self.lbl_diag_err_msg.pack(fill="x")
        self.lbl_diag_err_hint = tk.Label(self.card_error, text="", font=Fonts.CAPTION, bg=Colors.DANGER_BG,
                                          fg=Colors.DANGER_TEXT, anchor="w", justify="left", wraplength=600)
        self.lbl_diag_err_hint.pack(fill="x", pady=(2, 6))
        self.lbl_diag_err_detail = tk.Label(self.card_error, text="", font=Fonts.CODE, bg="#2a1215", fg="#fca5a5",
                                            padx=8, pady=6, anchor="w", justify="left", wraplength=600)
        self.lbl_diag_err_detail.pack(fill="x")

        # 4. Log file
        card_log = tk.Frame(page, bg=bg, highlightbackground=Colors.BORDER, highlightthickness=1, padx=16, pady=14)
        card_log.pack(fill="x", pady=(0, 12))
        self._card_log = card_log
        tk.Label(card_log, text="Tập tin log", font=Fonts.HEADING, bg=bg, fg=Colors.TEXT_PRIMARY).pack(
            anchor="w", pady=(0, 8))
        tk.Label(card_log, text=self.log_path, font=Fonts.CODE, bg=Colors.BG_ELEVATED, fg=Colors.TEXT_SECONDARY,
                 anchor="w", justify="left", padx=10, pady=8).pack(fill="x", pady=(0, 8))
        buttons = tk.Frame(card_log, bg=bg)
        buttons.pack(fill="x")
        self.btn_copy_log = ttk.Button(buttons, text="Sao chép đường dẫn", style="Secondary.TButton",
                                       command=self._copy_log_path)
        self.btn_copy_log.pack(side="left")
        self.btn_open_log_dir = ttk.Button(buttons, text="Mở thư mục log", style="Secondary.TButton",
                                           command=self._open_log_folder)
        self.btn_open_log_dir.pack(side="left", padx=(8, 0))
        self.lbl_log_note = tk.Label(
            card_log, font=Fonts.CAPTION, bg=bg, fg=Colors.TEXT_MUTED, anchor="w", justify="left", wraplength=600,
            text="Lỗi chi tiết, cảnh báo PortAudio và stack trace được ghi vào file log xoay vòng. Gửi file này "
                 "kèm mô tả khi báo lỗi.")
        self.lbl_log_note.pack(fill="x", pady=(8, 0))

        def _wrap(e: tk.Event) -> None:
            for lab in (self.lbl_log_note, self.lbl_latency_note):
                lab.config(wraplength=max(200, e.width - 36))

        card_log.bind("<Configure>", _wrap)

    def _build_stat_box(self, parent: tk.Frame, row: int, col: int, title: str, default_val: str) -> tk.Label:
        box = tk.Frame(parent, bg=Colors.BG_ELEVATED, highlightbackground=Colors.BORDER, highlightthickness=1,
                       padx=10, pady=10)
        box.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
        lbl_val = tk.Label(box, text=default_val, font=Fonts.METRIC_VAL, bg=Colors.BG_ELEVATED,
                           fg=Colors.TEXT_PRIMARY)
        lbl_val.pack()
        # Fixed wraplength: at the 980 px minimum window width one-line titles were clipped.
        tk.Label(box, text=title, font=Fonts.METRIC_LABEL, bg=Colors.BG_ELEVATED, fg=Colors.TEXT_MUTED,
                 justify="center", wraplength=px(150)).pack(pady=(4, 0))
        return lbl_val

    # ----------------------------------------------------------------- updates
    def update_metrics(self, metrics: Optional[dict[str, Any]]) -> None:
        """Show metrics with their source; averages before the first chunk and a stopped buffer are "--"."""
        self.lbl_telemetry_source.config(
            text=telemetry_label(metrics),
            fg=Colors.SUCCESS_TEXT if metrics and metrics.get("live") is True else Colors.TEXT_MUTED)
        if not metrics:
            self.val_diag_rtf.config(text="--", fg=Colors.TEXT_PRIMARY)
            for lab in (self.val_diag_gpu, self.val_diag_e2e, self.val_diag_buf, self.val_diag_chunks,
                        self.val_diag_und):
                lab.config(text="--", fg=Colors.TEXT_PRIMARY)
            self._param_values["Độ trễ ghép chunk + khử ồn"].config(text=" --")
            return

        no_chunks = metrics.get("chunks_converted") == 0
        not_live = metrics.get("live") is False

        rtf = None if no_chunks else metrics.get("rtf")
        if rtf is not None:
            self.val_diag_rtf.config(text=f"{rtf:.3f}", fg=get_rtf_color(rtf))
        else:
            self.val_diag_rtf.config(text="--", fg=Colors.TEXT_PRIMARY)

        gpu_ms = None if no_chunks else metrics.get("avg_inference_ms")
        self.val_diag_gpu.config(text=f"{gpu_ms:.0f} ms" if gpu_ms is not None else "--")

        est_lat = None if no_chunks else metrics.get("estimated_software_latency_ms")
        if est_lat is not None:
            self.val_diag_e2e.config(text=f"~{est_lat:.0f} ms")
        else:
            e2e_ms = None if no_chunks else metrics.get("avg_software_e2e_ms")
            self.val_diag_e2e.config(text=f"~{e2e_ms:.0f} ms" if e2e_ms is not None else "--")

        buf_ms = None if not_live else metrics.get("current_buffer_ms")
        self.val_diag_buf.config(text=f"{buf_ms:.0f} ms" if buf_ms is not None else "--")

        chunks_ok = metrics.get("chunks_converted", 0)
        chunks_drop = metrics.get("chunks_dropped", 0)
        self.val_diag_chunks.config(text=f"{chunks_ok} / {chunks_drop}",
                                    fg=Colors.WARNING_TEXT if chunks_drop else Colors.TEXT_PRIMARY)

        und = metrics.get("steady_underflows", 0)
        self.val_diag_und.config(text=f"{und}", fg=Colors.DANGER_TEXT if und > 0 else Colors.TEXT_PRIMARY)

        delay = metrics.get("algorithmic_delay_ms")
        self._param_values["Độ trễ ghép chunk + khử ồn"].config(
            text=f" {delay:.0f} ms" if isinstance(delay, (int, float)) else " --")

    def update_status(self, status: EngineStatus) -> None:
        """Stream parameters of the session and the last recorded error."""
        self._show_params(status)
        if status.last_error:
            self._last_error = status.last_error
            self.lbl_diag_err_msg.config(text=f"[{status.last_error.category}] {status.last_error.message}")
            self.lbl_diag_err_hint.config(text=f"Gợi ý: {status.last_error.hint}")
            self.lbl_diag_err_detail.config(text=status.last_error.details or "Không có trace kỹ thuật")
            self.card_error.pack(fill="x", pady=(0, 12), before=self._card_log)
            if self.card_error.winfo_width() > 1:
                self._on_card_error_configure(self.card_error.winfo_width())
        else:
            self.card_error.pack_forget()

    def _show_params(self, status: Optional[EngineStatus]) -> None:
        audio = status.audio if status is not None else None
        for k, v in stream_parameters(audio):
            self._param_values[k].config(text=f" {v}")
        self.lbl_params_source.config(text="Phiên hiện tại" if audio is not None else "Mặc định cho phiên tiếp theo")
        config = status.config if status is not None else None
        self._param_values["Thuật toán F0"].config(text=f" {config.f0_method.upper()}" if config else " --")

    def _on_card_error_configure(self, width: int) -> None:
        """Dynamically update error labels wrap length based on card container width."""
        if width <= 1 or abs(width - self._last_card_error_width) < 4:
            return
        self._last_card_error_width = width
        msg_wrap = max(200, width - 36)
        detail_wrap = max(200, width - 52)
        self.lbl_diag_err_msg.config(wraplength=msg_wrap)
        self.lbl_diag_err_hint.config(wraplength=msg_wrap)
        self.lbl_diag_err_detail.config(wraplength=detail_wrap)

    def _copy_log_path(self) -> None:
        self.clipboard_clear()
        self.clipboard_append(self.log_path)
        messagebox.showinfo("Đã sao chép", "Đã sao chép đường dẫn file log vào bộ nhớ tạm.")

    def _open_log_folder(self) -> None:
        folder = Path(self.log_path).parent
        try:
            os.startfile(str(folder))  # type: ignore[attr-defined]  # Windows only
        except (AttributeError, OSError) as err:
            logger.warning("Could not open log folder %s: %s", folder, err)
            messagebox.showwarning("Không mở được thư mục", f"Thư mục log: {folder}\n\n{err}")
