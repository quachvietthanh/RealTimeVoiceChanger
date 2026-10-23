"""Voice Studio action area: one state-dependent action, status, inline errors and the audio route summary.

The action comes from presentation.primary_action(): "Nạp giọng" for a selected voice that is not loaded,
"Bắt đầu chuyển giọng" once the backend reports it loaded and READY, "Dừng" while RUNNING, a disabled busy
label during LOADING/STARTING/STOPPING, and "Thử lại" after a stream failure. Detailed metrics live in the
Advanced page; Studio shows a one-line performance note only while measurements are live.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable, Mapping, Optional

from voice_changer.audio.devices import AudioDevice
from voice_changer.gui.presentation import PrimaryAction, performance_summary, primary_action, route_summary
from voice_changer.gui.theme import Colors, Fonts, get_engine_state_visual
from voice_changer.gui.views.widgets import make_link
from voice_changer.service import EngineState, EngineStatus, ErrorInfo, ProfileSummary


class ControlsPanel(ttk.Frame):
    """Primary action, engine status, inline error banner and route summary for Voice Studio."""

    def __init__(
        self,
        parent: tk.Widget,
        on_start_requested: Callable[[], None],
        on_stop_requested: Callable[[], None],
        log_path: Optional[str] = None,
        on_load_requested: Optional[Callable[[], None]] = None,
        on_open_audio_settings: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(parent, style="Surface.TFrame")
        self.on_start_requested = on_start_requested
        self.on_stop_requested = on_stop_requested
        self.on_load_requested = on_load_requested
        self.on_open_audio_settings = on_open_audio_settings
        self.log_path = log_path or r"%LOCALAPPDATA%\RealTimeVoiceChanger\logs\backend.log"

        self._current_state = EngineState.IDLE
        self._status: Optional[EngineStatus] = None
        self._last_error: Optional[ErrorInfo] = None
        self._op_error: Optional[ErrorInfo] = None  # a failed load/start/stop call (raised, not a stream failure)
        self._selected: Optional[ProfileSummary] = None
        self._loaded_id: Optional[str] = None
        self._loaded_name: str = ""
        self._loading = False
        self.action: PrimaryAction = primary_action(EngineState.IDLE, None, None)
        self._last_card_width: int = 0
        self._last_banner_width: int = 0

        self._build_ui()
        self._refresh()

    def _build_ui(self) -> None:
        bg = Colors.BG_SURFACE
        card = tk.Frame(self, bg=bg, highlightbackground=Colors.BORDER, highlightthickness=1, padx=20, pady=18)
        card.pack(fill="x", expand=True)
        card.bind("<Configure>", lambda e: self._on_card_configure(e.width))
        self._card = card

        status_row = tk.Frame(card, bg=bg)
        status_row.pack(fill="x", pady=(0, 12))
        init_text, init_bg, init_fg = get_engine_state_visual(EngineState.IDLE)
        self.pill_state = tk.Label(status_row, text=init_text, font=Fonts.BADGE, padx=10, pady=4, bg=init_bg,
                                   fg=init_fg)
        self.pill_state.pack(side="left")
        self.lbl_perf = tk.Label(status_row, text="", font=Fonts.CAPTION_BOLD, bg=bg, fg=Colors.TEXT_MUTED)
        self.lbl_perf.pack(side="right")

        self.btn_action = ttk.Button(card, text="", style="HeroPrimary.TButton", command=self._on_action_clicked)
        self.btn_action.pack(fill="x")
        self._action_box = self.btn_action  # the error banner is packed before it

        self.lbl_status_line = tk.Label(card, text="", font=Fonts.BODY, bg=bg, fg=Colors.TEXT_SECONDARY,
                                        anchor="w", justify="left")
        self.lbl_status_line.pack(fill="x", pady=(10, 0))

        # Shown while streaming with another loadable voice selected: switching stops the session first.
        self.btn_switch = ttk.Button(card, text="", style="Secondary.TButton", command=self._on_switch_clicked)

        # Inline error banner (engine failure or a failed operation), with the actions that can fix it.
        self.banner_error = tk.Frame(card, bg=Colors.DANGER_BG, highlightbackground=Colors.DANGER,
                                     highlightthickness=1, padx=12, pady=8)
        self.banner_error.bind("<Configure>", lambda e: self._on_banner_configure(e.width))
        self.lbl_error_msg = tk.Label(self.banner_error, text="", font=Fonts.BODY_BOLD, bg=Colors.DANGER_BG,
                                      fg=Colors.TEXT_PRIMARY, anchor="w", justify="left", wraplength=350)
        self.lbl_error_msg.pack(fill="x")
        self.lbl_error_hint = tk.Label(self.banner_error, text="", font=Fonts.CAPTION, bg=Colors.DANGER_BG,
                                       fg=Colors.DANGER_TEXT, anchor="w", justify="left", wraplength=350)
        self.lbl_error_hint.pack(fill="x", pady=(2, 6))
        err_actions = tk.Frame(self.banner_error, bg=Colors.DANGER_BG)
        err_actions.pack(fill="x")
        self.btn_err_details = self._banner_button(err_actions, "Xem chi tiết kỹ thuật", self._show_error_dialog)
        self.btn_err_details.pack(side="left")
        self.btn_err_devices = self._banner_button(err_actions, "Kiểm tra thiết bị âm thanh",
                                                   self._open_audio_settings)

        # Route summary: the devices actually selected, and what that means for other applications.
        sep = tk.Frame(card, bg=Colors.BORDER_SUBTLE, height=1)
        sep.pack(fill="x", pady=(16, 12))
        route_row = tk.Frame(card, bg=bg)
        route_row.pack(fill="x")
        self.btn_change_devices = tk.Label(route_row, text="Đổi thiết bị →", font=Fonts.CAPTION_BOLD, bg=bg,
                                           fg=Colors.ACCENT_CYAN, padx=4)
        make_link(self.btn_change_devices, self._open_audio_settings, bg)
        self.btn_change_devices.pack(side="right", anchor="n")
        self.lbl_route = tk.Label(route_row, text="", font=Fonts.BODY_BOLD, bg=bg, fg=Colors.TEXT_PRIMARY,
                                  anchor="w", justify="left")
        self.lbl_route.pack(fill="x")
        self.lbl_route_hint = tk.Label(card, text="", font=Fonts.CAPTION, bg=bg, fg=Colors.TEXT_MUTED, anchor="w",
                                       justify="left")
        self.lbl_route_hint.pack(fill="x", pady=(4, 0))

    def _banner_button(self, parent: tk.Widget, text: str, command: Callable[[], None]) -> tk.Button:
        btn = tk.Button(parent, text=text, font=Fonts.CAPTION_BOLD, bg=Colors.DANGER, fg=Colors.TEXT_ON_PRIMARY,
                        activebackground=Colors.DANGER_HOVER, activeforeground=Colors.TEXT_ON_PRIMARY,
                        borderwidth=0, highlightthickness=2, highlightbackground=Colors.DANGER_BG,
                        highlightcolor=Colors.ACCENT_CYAN, padx=8, pady=2, cursor="hand2", command=command)

        def _focus_in(e: Optional[tk.Event] = None) -> None:
            btn.config(highlightbackground=Colors.ACCENT_CYAN)

        def _focus_out(e: Optional[tk.Event] = None) -> None:
            btn.config(highlightbackground=Colors.DANGER_BG)

        btn._focus_in = _focus_in  # type: ignore[attr-defined]
        btn._focus_out = _focus_out  # type: ignore[attr-defined]
        btn.bind("<FocusIn>", _focus_in)
        btn.bind("<FocusOut>", _focus_out)
        btn.bind("<Return>", lambda e: (btn.invoke(), "break")[1])
        return btn

    # ----------------------------------------------------------------- inputs from the app
    def update_status(self, status: EngineStatus) -> None:
        """Follow the live engine status (state pill, action, engine-error banner)."""
        self._status = status
        self._current_state = status.state
        self._last_error = status.last_error
        self._loaded_id = status.profile_id if status.model_loaded else None
        text, bg, fg = get_engine_state_visual(status.state)
        self.pill_state.config(text=text, bg=bg, fg=fg)
        self._refresh()

    def set_selection(self, selected: Optional[ProfileSummary], loaded_name: str = "",
                      loading: Optional[bool] = None) -> None:
        """The voice chosen in Studio, the display name of the loaded one, and whether a load is in progress."""
        self._selected = selected
        self._loaded_name = loaded_name
        if loading is not None:
            self._loading = loading
        self._refresh()

    def set_route(self, inp: Optional[AudioDevice], out: Optional[AudioDevice],
                  missing_input: Optional[str] = None, missing_output: Optional[str] = None) -> None:
        summary = route_summary(inp, out)
        self.lbl_route.config(text=summary.text)
        notes = []
        for name in (missing_input, missing_output):
            if name:
                notes.append(f"⚠ Thiết bị đã lưu “{name}” không khả dụng; đang tạm dùng thiết bị trên.")
        self.lbl_route_hint.config(text="\n".join(notes + [summary.hint]),
                                   fg=Colors.WARNING_TEXT if notes else Colors.TEXT_MUTED)

    def update_performance(self, metrics: Optional[Mapping[str, Any]]) -> None:
        summary = performance_summary(metrics)
        if summary is None:
            self.lbl_perf.config(text="")
        else:
            self.lbl_perf.config(text=summary[0], fg=summary[1])

    def show_operation_error(self, message: str, hint: str = "", details: str = "", code: str = "",
                             category: str = "") -> None:
        """Show a failed load/start/stop call inline (the engine may be IDLE or READY, not ERROR)."""
        self._op_error = ErrorInfo(category=category or "operation", code=code or "operation_failed",
                                   message=message, hint=hint, details=details, timestamp="")
        self._refresh()

    def clear_operation_error(self) -> None:
        self._op_error = None
        self._refresh()

    # ----------------------------------------------------------------- rendering
    def _refresh(self) -> None:
        loaded = self._loaded_id
        self.action = primary_action(self._current_state, self._selected, loaded, self._loading)
        self.btn_action.config(text=self.action.text, style=f"Hero{self.action.style}.TButton",
                               state="normal" if self.action.enabled else "disabled")
        self.lbl_status_line.config(text=self._status_line())

        sel = self._selected
        if (self._current_state == EngineState.RUNNING and sel is not None and sel.profile_id != loaded
                and sel.selectable):
            self.btn_switch.config(text=f"Dừng phiên và chuyển sang “{sel.display_name}”")
            self.btn_switch.pack(fill="x", pady=(8, 0), after=self.lbl_status_line)
        else:
            self.btn_switch.pack_forget()

        err = self._engine_error() or self._op_error
        if err is not None:
            self.lbl_error_msg.config(text=f"❌ {err.message}")
            # In ERROR the action offers "Thử lại", not "Dừng"; the fallback must name an action that exists.
            self.lbl_error_hint.config(text=err.hint or
                                       "Kiểm tra lại thiết bị âm thanh rồi thử lại; xem chi tiết kỹ thuật bên dưới.")
            if err.category == "audio_device" and self.on_open_audio_settings is not None:
                self.btn_err_devices.pack(side="left", padx=(8, 0))
            else:
                self.btn_err_devices.pack_forget()
            self.banner_error.pack(fill="x", pady=(0, 12), before=self._action_box)
        else:
            self.banner_error.pack_forget()

    def _engine_error(self) -> Optional[ErrorInfo]:
        s = self._status
        if s is not None and s.last_error and (s.state == EngineState.ERROR or s.health_problems):
            return s.last_error
        return None

    def _status_line(self) -> str:
        state, sel = self._current_state, self._selected
        loaded = self._loaded_name or (self._loaded_id or "")
        if self._loading or state == EngineState.LOADING:
            return "Đang nạp và kiểm tra tệp mô hình. Việc này có thể mất vài giây."
        if state == EngineState.STARTING:
            return "Đang mở microphone và ngõ ra…"
        if state == EngineState.STOPPING:
            return "Đang dừng phiên chuyển giọng…"
        if state == EngineState.RUNNING:
            return f"Đang chuyển giọng bằng “{loaded}”." if loaded else "Đang chuyển giọng."
        if state == EngineState.ERROR:
            return "Phiên chuyển giọng đã dừng vì lỗi."
        if self.action.kind == "load" and sel is not None:
            if self._loaded_id:
                return f"Đang nạp sẵn “{loaded}”. Nhấn “Nạp giọng” để chuyển sang “{sel.display_name}”."
            return f"“{sel.display_name}” chưa được nạp. Nhấn “Nạp giọng” để chuẩn bị."
        if self.action.kind == "start":
            return f"Đã nạp “{loaded}”. Nhấn để bắt đầu." if loaded else "Đã nạp giọng. Nhấn để bắt đầu."
        if self.action.kind == "none" and sel is not None:
            return "Giọng đang chọn không thể nạp. Chọn giọng khác trong danh sách hoặc Thư viện."
        return "Chọn một giọng để bắt đầu."

    # ----------------------------------------------------------------- user actions
    def _on_action_clicked(self) -> None:
        kind = self.action.kind
        if kind == "load" and self.on_load_requested is not None:
            self.on_load_requested()
        elif kind in ("start", "retry"):
            self.on_start_requested()
        elif kind == "stop":
            self.on_stop_requested()

    def _on_switch_clicked(self) -> None:
        if self.on_load_requested is not None:
            self.on_load_requested()

    def _open_audio_settings(self) -> None:
        if self.on_open_audio_settings is not None:
            self.on_open_audio_settings()

    def _show_error_dialog(self) -> None:
        err = self._engine_error() or self._op_error or self._last_error
        if not err:
            return
        details = (
            f"Mã lỗi (Code): {err.code}\n"
            f"Phân loại (Category): {err.category}\n"
            + (f"Thời điểm: {err.timestamp}\n" if err.timestamp else "")
            + f"\nChi tiết kỹ thuật:\n{err.details or 'Không có chi tiết thêm.'}\n\n"
            f"File log chi tiết ghi tại:\n{self.log_path}"
        )
        messagebox.showerror(title="Chi tiết lỗi", message=err.message, detail=details)

    def _on_card_configure(self, width: int) -> None:
        """Wrap the long labels to the card width (ignoring sub-4 px changes so it cannot oscillate)."""
        if width <= 1 or abs(width - self._last_card_width) < 4:
            return
        self._last_card_width = width
        target_wrap = max(180, width - 58)
        for lab in (self.lbl_error_msg, self.lbl_error_hint, self.lbl_status_line, self.lbl_route_hint):
            lab.config(wraplength=target_wrap)
        self.lbl_route.config(wraplength=max(180, target_wrap - self.btn_change_devices.winfo_reqwidth()))

    def _on_banner_configure(self, width: int) -> None:
        """Fine-tune error label wraplength when banner container resizes."""
        if width <= 1 or abs(width - self._last_banner_width) < 4:
            return
        self._last_banner_width = width
        target_wrap = max(180, width - 28)
        self.lbl_error_msg.config(wraplength=target_wrap)
        self.lbl_error_hint.config(wraplength=target_wrap)
