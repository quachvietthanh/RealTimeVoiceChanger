"""Main Tkinter desktop application for RealTimeVoiceChanger.

Pages: Studio (selected voice + one state-dependent action), Voice Library, Audio & Connections (the only
place devices are chosen) and Advanced / Troubleshooting. Selecting a voice never loads it: loading happens
only through Studio's "Nạp giọng" action (or the startup load of the profile the backend resolves).
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Callable, Optional

import tkinter as tk
from tkinter import messagebox, ttk

from voice_changer.audio.devices import AudioDevice
from voice_changer.conversion.base import ConversionConfig
from voice_changer.diagnostics import configure_logging
from voice_changer.gui.dpi import enable_dpi_awareness, fit_window, init_ui_scale, px
from voice_changer.gui.presentation import APP_NAME, icon_files, windows_icon_file
from voice_changer.gui.settings import GuiSettings
from voice_changer.gui.theme import Colors, Fonts, setup_theme
from voice_changer.gui.views import (
    AudioSettingsView,
    ControlsPanel,
    DiagnosticsPanel,
    NoiseSuppressionPanel,
    ProfilesPanel,
    Sidebar,
    TuningPanel,
    VoiceLibraryView,
)
from voice_changer.gui.views.widgets import CollapsibleSection
from voice_changer.service import (
    AudioSettings,
    BackendError,
    DeviceInventory,
    EngineState,
    EngineStatus,
    NoiseSuppressionStatus,
    ProfileSummary,
    VoiceChangerService,
)

logger = logging.getLogger(__name__)

STUDIO_MAX_WIDTH = 760  # logical px; the Studio column is centred and never wider than this

VIEWS: dict[str, tuple[int, str, str]] = {
    "studio": (0, "Studio", "Chọn giọng, nạp giọng rồi bắt đầu chuyển giọng."),
    "library": (1, "Thư viện giọng", "Tất cả giọng đã đăng ký với trạng thái thật, và bộ sưu tập dự kiến."),
    "audio_settings": (2, "Âm thanh & Kết nối",
                       "Chọn micro, ngõ ra và cách dùng giọng trong Discord, OBS hoặc game."),
    "diagnostics": (3, "Nâng cao / Khắc phục sự cố",
                    "Số liệu hiệu năng, thông số luồng, lỗi gần nhất và file log."),
}

NOISE_MODE_LABELS = {"light": "Nhẹ", "medium": "Vừa", "strong": "Mạnh"}


def tuning_summary(config: Optional[ConversionConfig]) -> str:
    if config is None:
        return "Chưa nạp giọng"
    return f"Cao độ {config.pitch_shift:+.0f} · Index {config.index_rate:.2f}"


def noise_summary(status: Optional[NoiseSuppressionStatus]) -> tuple[str, str]:
    if status is None:
        return "", Colors.TEXT_MUTED
    if status.failed:
        return "Đã tự ngắt — mở để bật lại", Colors.DANGER_TEXT
    if not status.enabled:
        return "Đang tắt", Colors.TEXT_MUTED
    mode = NOISE_MODE_LABELS.get(status.mode, status.mode)
    if status.active:
        return f"Đang lọc · {mode}", Colors.SUCCESS_TEXT
    return f"Bật · {mode} (áp dụng khi chạy)", Colors.TEXT_SECONDARY


class VoiceChangerApp:
    """Desktop GUI orchestrator: wires the views to VoiceChangerService and keeps them in sync."""

    def __init__(self, root: tk.Tk, service: Optional[VoiceChangerService] = None) -> None:
        self.root = root
        self.root.title(APP_NAME)
        # DPI awareness must be set before tk.Tk() (see launch_gui); here only the scale Tk detected is read.
        init_ui_scale(self.root)
        fit_window(self.root, 1140, 780, min_width=980, min_height=680)
        self._icon_images: list[tk.PhotoImage] = []
        self._apply_window_icon()

        # Configure rotating file logging
        try:
            self.log_path = configure_logging()
        except Exception as err:
            logger.warning("Could not initialize logging: %s", err)
            self.log_path = None

        # Backend service instance
        self.service = service if service is not None else VoiceChangerService()

        # Settings persistence
        self.settings = GuiSettings.load()

        # Thread marshaling queue
        self._action_queue: queue.Queue[Callable[[], None]] = queue.Queue()
        self._inventory: Optional[DeviceInventory] = None
        self._profiles: list[ProfileSummary] = []
        self._active_status: Optional[EngineStatus] = None
        self._current_view: str = "studio"
        self._loading: bool = False

        # Build theme and window UI
        self.style = setup_theme(self.root)
        self._build_layout()

        # Hook status listener from backend
        self.service.add_status_listener(self._on_backend_status_changed)

        # Window close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)

        # Start periodic queue checking & polling
        self.root.after(40, self._process_queue)
        self.root.after(300, self._poll_metrics)

        # Initial data loading in background
        self._run_async(self._initial_load)

    def _apply_window_icon(self) -> None:
        """Title-bar/taskbar icon from the generated assets; a missing or unreadable file is only logged."""
        try:
            self._icon_images = [tk.PhotoImage(file=str(p)) for p in icon_files()]
            if self._icon_images:
                self.root.iconphoto(True, *self._icon_images)
        except tk.TclError as err:
            logger.warning("Could not set window icon: %s", err)
            self._icon_images = []
        ico = windows_icon_file()
        if ico is not None:
            try:
                self.root.iconbitmap(default=str(ico))
            except tk.TclError as err:
                logger.warning("Could not set Windows icon %s: %s", ico, err)

    def _build_layout(self) -> None:
        workspace = tk.Frame(self.root, bg=Colors.BG_DARK)
        workspace.pack(fill="both", expand=True)

        self.sidebar = Sidebar(workspace, on_navigate=self._navigate_to, current_view="studio")
        self.sidebar.pack(side="left", fill="y")

        main_area = tk.Frame(workspace, bg=Colors.BG_DARK)
        main_area.pack(side="right", fill="both", expand=True)

        # Page title only: engine state lives beside the Studio action, devices in Audio & Connections.
        top_header = tk.Frame(main_area, bg=Colors.BG_DARK, padx=24, pady=14)
        top_header.pack(fill="x")
        _, title, subtitle = VIEWS["studio"]
        self.lbl_breadcrumb = tk.Label(top_header, text=title, font=Fonts.TITLE, bg=Colors.BG_DARK,
                                       fg=Colors.TEXT_PRIMARY, anchor="w", justify="left")
        self.lbl_breadcrumb.pack(anchor="w", fill="x")
        self.lbl_breadcrumb_sub = tk.Label(top_header, text=subtitle, font=Fonts.CAPTION, bg=Colors.BG_DARK,
                                           fg=Colors.TEXT_MUTED, anchor="w", justify="left")
        self.lbl_breadcrumb_sub.pack(anchor="w", fill="x")

        def _wrap_breadcrumb(event: tk.Event) -> None:
            wrap = max(120, event.width - px(48))
            self.lbl_breadcrumb.config(wraplength=wrap)
            self.lbl_breadcrumb_sub.config(wraplength=wrap)

        top_header.bind("<Configure>", _wrap_breadcrumb)

        self.notebook = ttk.Notebook(main_area, style="Hidden.TNotebook")
        self.notebook.pack(fill="both", expand=True)
        tabs: dict[str, ttk.Frame] = {}
        for view_id, (_, title, _) in VIEWS.items():
            tab = ttk.Frame(self.notebook, style="TFrame")
            self.notebook.add(tab, text=title)
            tabs[view_id] = tab

        log_str = str(self.log_path) if self.log_path else "logs/backend.log"
        self._build_studio(tabs["studio"], log_str)

        self.panel_library = VoiceLibraryView(tabs["library"], on_profile_selected=self._handle_profile_selected)
        self.panel_library.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        # The only device selection in the app; Studio shows a summary of it.
        self.panel_audio_settings = AudioSettingsView(
            tabs["audio_settings"],
            on_devices_changed=lambda i, o: self._handle_devices_changed(
                i, o, changed=self._take_user_change(self.panel_audio_settings)),
            on_refresh_requested=self._handle_refresh_devices,
        )
        self.panel_audio_settings.pack(fill="both", expand=True)
        self.panel_guide = self.panel_audio_settings  # backwards-compatible alias

        self.panel_diagnostics = DiagnosticsPanel(tabs["diagnostics"], log_path=log_str)
        self.panel_diagnostics.pack(fill="both", expand=True)

    def _build_studio(self, tab: ttk.Frame, log_str: str) -> None:
        """One centred column: selected voice, the action card, then collapsed tuning and noise sections."""
        canvas = tk.Canvas(tab, bg=Colors.BG_DARK, highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        column = tk.Frame(canvas, bg=Colors.BG_DARK)
        column.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        window = canvas.create_window((0, 0), window=column, anchor="n")

        def _on_canvas_configure(event: tk.Event) -> None:
            width = max(1, min(event.width - px(24), px(STUDIO_MAX_WIDTH)))
            canvas.coords(window, event.width / 2, 0)
            canvas.itemconfig(window, width=width)

        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_mousewheel(event: tk.Event) -> None:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        canvas.pack(side="left", fill="both", expand=True, pady=(0, 8))
        scrollbar.pack(side="right", fill="y", padx=(0, 4), pady=(0, 8))
        self.studio_column = column

        self.panel_profiles = ProfilesPanel(
            column,
            service=self.service,
            on_profile_selected=self._handle_profile_selected,
            on_open_library=lambda: self._navigate_to("library"),
        )
        self.panel_profiles.pack(fill="x", pady=(4, 16))

        self.panel_controls = ControlsPanel(
            column,
            on_start_requested=self._handle_start_requested,
            on_stop_requested=self._handle_stop_requested,
            log_path=log_str,
            on_load_requested=self._handle_load_requested,
            on_open_audio_settings=lambda: self._navigate_to("audio_settings"),
        )
        self.panel_controls.pack(fill="x", pady=(0, 12))

        self.section_tuning = CollapsibleSection(column, "Tinh chỉnh giọng", expanded=False)
        self.section_tuning.pack(fill="x", pady=(0, 8))
        self.panel_tuning = TuningPanel(
            self.section_tuning.body,
            on_config_changed=self._handle_config_changed,
            on_reset_requested=self._handle_reset_config,
            framed=False,
        )
        self.panel_tuning.pack(fill="x", pady=(0, 8))
        self.section_tuning.set_summary(tuning_summary(None))

        self.section_noise = CollapsibleSection(column, "Khử tiếng ồn micro", expanded=False)
        self.section_noise.pack(fill="x", pady=(0, 8))
        self.panel_noise = NoiseSuppressionPanel(
            self.section_noise.body,
            on_noise_changed=self._handle_noise_suppression_changed,
            on_rearm_requested=self._handle_noise_rearm,
            framed=False,
        )
        self.panel_noise.pack(fill="x", pady=(0, 8))

    # ----------------------------------------------------------------- Navigation
    def _navigate_to(self, view_id: str) -> None:
        """Switch pages (sidebar, in-page links, and after choosing a voice in the Library)."""
        if view_id not in VIEWS:
            logger.warning("Unknown view '%s'", view_id)
            return
        self._current_view = view_id
        self.sidebar.set_active_view(view_id)
        index, title, subtitle = VIEWS[view_id]
        self.notebook.select(index)
        self.lbl_breadcrumb.config(text=title)
        self.lbl_breadcrumb_sub.config(text=subtitle)

    # ----------------------------------------------------------------- Async Execution
    def _run_async(self, func: Callable[..., None], *args: Any) -> None:
        """Run long operation in background thread without freezing GUI."""
        def worker() -> None:
            try:
                func(*args)
            except Exception as err:
                logger.exception("Async worker failed")
                self._dispatch(lambda err=err: self._handle_async_error(err))

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _dispatch(self, callback: Callable[[], None]) -> None:
        """Marshal a callback to the main GUI thread."""
        self._action_queue.put(callback)

    def _process_queue(self) -> None:
        """Drain action queue on Tkinter main thread."""
        while not self._action_queue.empty():
            try:
                fn = self._action_queue.get_nowait()
                fn()
            except Exception:
                logger.exception("Error executing queued action")
        self.root.after(40, self._process_queue)

    def _handle_async_error(self, err: Exception) -> None:
        """Show a failed load/start/stop inline in Studio, with its hint and technical details."""
        if isinstance(err, BackendError):
            self.panel_controls.show_operation_error(err.message, hint=err.hint, details=err.details,
                                                     code=err.code, category=err.category)
        else:
            self.panel_controls.show_operation_error(str(err) or type(err).__name__, details=repr(err))
        self._sync_status(self.service.get_status())

    # ----------------------------------------------------------------- Initialization
    def _initial_load(self) -> None:
        """List profiles and devices, then load the profile the backend resolves (the only automatic load)."""
        profiles = self.service.list_profiles(include_withdrawn=True)
        inventory = self.service.list_devices()
        ns_status = self.service.get_noise_suppression()
        # A saved profile that is no longer selectable (withdrawn, removed, files missing) falls back to the
        # default, or to nothing; the backend owns that rule.
        target_profile_id = self.service.resolve_profile_id(self.settings.profile_id)

        def ui_init() -> None:
            self._profiles = profiles
            self._inventory = inventory
            self._apply_inventory(inventory)  # never saves; see _apply_inventory
            self._apply_noise_status(ns_status)
            self.panel_profiles.set_profiles(profiles, target_profile_id)
            self.panel_library.set_profiles(profiles, target_profile_id)
            self._sync_status(self.service.get_status())

        self._dispatch(ui_init)

        if target_profile_id:
            self._load_profile_worker(target_profile_id)

    # ----------------------------------------------------------------- Voice selection and loading
    def _find_profile(self, profile_id: str) -> Optional[ProfileSummary]:
        return next((p for p in self._profiles if p.profile_id == profile_id), None)

    def _explain_unselectable(self, profile: ProfileSummary) -> None:
        """Why a profile cannot be used (withdrawn / no model yet / files missing)."""
        withdrawn_reason = getattr(profile, "withdrawn_reason", "")
        status = getattr(profile, "status", "")
        readiness = getattr(profile, "readiness", None)
        setup_hint = getattr(profile, "setup_hint", "")
        if withdrawn_reason:
            messagebox.showwarning(
                "Hồ sơ đã bị thu hồi",
                f"Hồ sơ '{profile.display_name}' đã được rút khỏi danh sách sử dụng thông thường.\n\n"
                f"Lý do: {withdrawn_reason}",
            )
        elif status == "planned" or readiness == "no_model":
            detail_hint = setup_hint or "Chưa có dataset và checkpoint trong hệ thống."
            messagebox.showwarning(
                "Hồ sơ chưa có mô hình",
                f"Hồ sơ '{profile.display_name}' đang ở trạng thái dự kiến.\n"
                "Chưa thể chuyển giọng: chưa có checkpoint và index hợp lệ.\n\n"
                f"Gợi ý: {detail_hint}",
            )
        else:
            detail_hint = setup_hint or "Kiểm tra lại đường dẫn tệp trong config/voice_profiles.json."
            messagebox.showerror(
                "Thiếu tệp mô hình",
                f"Hồ sơ '{profile.display_name}' không tìm thấy tệp checkpoint hoặc index trên máy tính.\n\n"
                f"Gợi ý: {detail_hint}",
            )

    def _handle_profile_selected(self, profile_id: str) -> None:
        """Select a voice (Studio picker or Library). Shows it in Studio; never loads it or touches the engine."""
        target_profile = self._find_profile(profile_id)
        if target_profile is not None and not getattr(target_profile, "selectable", False):
            self._explain_unselectable(target_profile)
            self.panel_profiles.set_profiles(self._profiles)  # the picker returns to the current choice
            return
        self.panel_profiles.select_profile(profile_id)
        if self._current_view != "studio":
            self._navigate_to("studio")
        self._refresh_selection()

    def _handle_load_requested(self) -> None:
        """Studio's "Nạp giọng" (or switch) action for the selected voice."""
        selected = self.panel_profiles.get_selected_profile()
        if selected is None:
            return
        if not getattr(selected, "selectable", False):
            self._explain_unselectable(selected)
            return
        profile_id = selected.profile_id
        self.panel_controls.clear_operation_error()
        status = self.service.get_status()
        if status.state == EngineState.RUNNING:
            ans = messagebox.askyesno(
                "Đang chuyển đổi giọng nói",
                f"Cần dừng phiên hiện tại để nạp “{selected.display_name}”.\nBạn có muốn dừng và tiếp tục?",
            )
            if not ans:
                return
            self._run_async(self._switch_profile_worker, profile_id)
        elif status.state == EngineState.ERROR:
            # The failed session must be stopped before select_profile() accepts a new profile.
            self._run_async(self._switch_profile_worker, profile_id)
        else:
            self._run_async(self._load_profile_worker, profile_id)

    def _switch_profile_worker(self, profile_id: str) -> None:
        self._dispatch(lambda: self._set_profiles_loading(True))
        try:
            # Check the new profile's files before stopping, so an invalid profile never interrupts the stream.
            validation = self.service.validate_profile(profile_id)
            if not validation.ok:
                problems = "\n".join(validation.errors)
                self._dispatch(lambda: self.panel_controls.show_operation_error(
                    f"Không thể nạp hồ sơ '{profile_id}': hồ sơ không hợp lệ; phiên đang chạy được giữ nguyên.",
                    hint=problems or "Kiểm tra lại tệp mô hình của hồ sơ này.",
                    details=problems,
                    code="profile_invalid",
                    category="profile",
                ))
                return
            self.service.stop()
            self._load_profile_worker(profile_id)
        except Exception as err:
            self._dispatch(lambda err=err: self._handle_async_error(err))
        finally:
            self._dispatch(lambda: self._set_profiles_loading(False))

    def _load_profile_worker(self, profile_id: str) -> None:
        self._dispatch(lambda: self._set_profiles_loading(True))
        try:
            self.service.select_profile(profile_id)
            self.settings.profile_id = profile_id
            self.settings.save()
            cfg = self.service.get_config()
            updated_profiles = self.service.list_profiles(include_withdrawn=True)
            self._profiles = updated_profiles

            def _update_ui() -> None:
                self.panel_tuning.set_config(cfg)
                self.section_tuning.set_summary(tuning_summary(cfg))
                self.panel_profiles.set_profiles(updated_profiles, current_id=profile_id)
                self.panel_library.set_profiles(updated_profiles, active_id=profile_id)
                # No clear_operation_error() here: a late startup load must not hide a newer error. Errors are
                # cleared when the user starts the next load/start/stop.

            self._dispatch(_update_ui)
        except Exception as err:
            self._dispatch(lambda err=err: self._handle_async_error(err))
        finally:
            self._dispatch(lambda: self._set_profiles_loading(False))

    def _set_profiles_loading(self, is_loading: bool) -> None:
        self._loading = is_loading
        self.panel_profiles.set_loading(is_loading)
        self.panel_library.set_loading(is_loading)
        self._refresh_selection()

    def _refresh_selection(self) -> None:
        """Tell the Studio action card what is selected, what is loaded, and whether a load is running."""
        status = self._active_status
        loaded_name = ""
        if status is not None and status.model_loaded and status.profile_id:
            loaded = self._find_profile(status.profile_id)
            loaded_name = loaded.display_name if loaded else status.profile_id
        self.panel_controls.set_selection(self.panel_profiles.get_selected_profile(), loaded_name, self._loading)

    # ----------------------------------------------------------------- Devices
    @staticmethod
    def _take_user_change(panel: Any) -> Optional[str]:
        """The side ('input'/'output') a panel reports the user changed, consumed so it is never reused."""
        changed = getattr(panel, "last_user_change", None)
        panel.last_user_change = None
        return changed

    def _apply_inventory(self, inventory: DeviceInventory) -> None:
        """Show `inventory`, selecting the saved devices when present.

        Never saves settings: an unplugged saved device keeps its saved name (so a later refresh restores it), and the
        fallback chosen for this session is only the runtime selection, which the views show and start() uses.
        """
        in_id = self.settings.resolve_input_device_id(inventory)
        out_id = self.settings.resolve_output_device_id(inventory)
        self.panel_audio_settings.set_inventory(inventory, in_id, out_id)
        self._show_missing_saved_devices()

    def _show_missing_saved_devices(self) -> None:
        missing_in = missing_out = None
        if self._inventory:
            missing_in = self.settings.missing_saved_input(self._inventory)
            missing_out = self.settings.missing_saved_output(self._inventory)
            self.panel_audio_settings.show_saved_device_unavailable(missing_in, missing_out)
        inp, out = self.panel_audio_settings.get_selected_devices()
        self.panel_controls.set_route(inp, out, missing_in, missing_out)

    def _handle_devices_changed(self, inp: Optional[AudioDevice], out: Optional[AudioDevice],
                                changed: Optional[str] = None) -> None:
        """The user picked a device. Saves only the side they changed (`changed`; None = both),
        so a session fallback on the other side never replaces that side's saved preference."""
        if changed in (None, "input"):
            self.settings.update_input(inp)
        if changed in (None, "output"):
            self.settings.update_output(out)
        self.settings.save()
        if self._inventory:
            curr_in, curr_out = self.panel_audio_settings.get_selected_devices()
            in_id = inp.device_id if inp else None
            out_id = out.device_id if out else None
            if (curr_in.device_id if curr_in else None, curr_out.device_id if curr_out else None) != (in_id, out_id):
                self.panel_audio_settings.set_inventory(self._inventory, in_id, out_id)
        self._show_missing_saved_devices()

    def _handle_refresh_devices(self) -> None:
        def worker() -> None:
            inv = self.service.list_devices()

            def update_ui() -> None:
                self._inventory = inv
                self._apply_inventory(inv)  # a reconnected saved device is selected again; nothing is saved

            self._dispatch(update_ui)

        self._run_async(worker)

    # ----------------------------------------------------------------- Tuning and noise suppression
    def _handle_config_changed(self, changes: dict[str, float | int]) -> None:
        try:
            new_cfg = self.service.update_config(**changes)
            self.settings.pitch_shift = new_cfg.pitch_shift
            self.settings.index_rate = new_cfg.index_rate
            self.settings.filter_radius = new_cfg.filter_radius
            self.settings.protect_rate = new_cfg.protect_rate
            self.settings.save()
            self.section_tuning.set_summary(tuning_summary(new_cfg))
        except BackendError as err:
            self.panel_tuning.set_config(self.service.get_config())
            messagebox.showwarning("Cài đặt không hợp lệ", err.message)

    def _handle_reset_config(self) -> None:
        try:
            new_cfg = self.service.reset_config()
            self.panel_tuning.set_config(new_cfg)
            self.section_tuning.set_summary(tuning_summary(new_cfg))
        except Exception as err:
            self._handle_async_error(err)

    def _apply_noise_status(self, status: NoiseSuppressionStatus) -> None:
        self.panel_noise.set_status(status)
        self.section_noise.set_summary(*noise_summary(status))

    def _handle_noise_suppression_changed(
        self, enabled: Optional[bool], mode: Optional[str]
    ) -> None:
        """Handle noise suppression toggle or mode changes from the GUI."""
        try:
            self._apply_noise_status(self.service.set_noise_suppression(enabled=enabled, mode=mode))
        except BackendError as err:
            logger.warning("Noise suppression configuration error: %s", err)
            self._apply_noise_status(self.service.get_noise_suppression())
            self.panel_noise.show_error(err.message, err.hint)
        except Exception as err:
            logger.exception("Unexpected error setting noise suppression")
            self._apply_noise_status(self.service.get_noise_suppression())
            self.panel_noise.show_error("Không thể cập nhật khử tiếng ồn", str(err))

    def _handle_noise_rearm(self) -> None:
        """Handle re-arm action after stage disabled itself due to errors."""
        self._handle_noise_suppression_changed(enabled=True, mode=None)

    # ----------------------------------------------------------------- Start / stop
    def _handle_start_requested(self) -> None:
        inp, out = self.panel_audio_settings.get_selected_devices()
        settings = AudioSettings(input_device_id=inp.device_id if inp else None,
                                 output_device_id=out.device_id if out else None)
        self.panel_controls.clear_operation_error()

        def worker() -> None:
            try:
                # "Retry" after a stream failure: start() only accepts READY, so stop the failed session first.
                if self.service.get_status().state == EngineState.ERROR:
                    self.service.stop()
                self.service.start(settings)
            except Exception as err:
                self._dispatch(lambda err=err: self._handle_async_error(err))

        self._run_async(worker)

    def _handle_stop_requested(self) -> None:
        self.panel_controls.clear_operation_error()

        def worker() -> None:
            try:
                self.service.stop()
            except Exception as err:
                self._dispatch(lambda err=err: self._handle_async_error(err))

        self._run_async(worker)

    # ----------------------------------------------------------------- Status & Polling
    def _on_backend_status_changed(self, status: EngineStatus) -> None:
        """Called by service listener on worker thread; marshal to GUI thread."""
        self._dispatch(lambda: self._sync_status(status))

    def _poll_metrics(self) -> None:
        """Periodically poll metrics & status (~3.3 Hz). Also detects background stream errors."""
        try:
            status = self.service.get_status()
            self._sync_status(status)
            metrics = self.service.get_metrics()
            self.panel_controls.update_performance(metrics)
            self.panel_diagnostics.update_metrics(metrics)

            ns_status = self.service.get_noise_suppression()
            self.panel_noise.sync_live_state(ns_status, metrics)
            self.section_noise.set_summary(*noise_summary(ns_status))
        except Exception:
            logger.exception("Error during metric polling")
        finally:
            self.root.after(300, self._poll_metrics)

    def _sync_status(self, status: EngineStatus) -> None:
        """Synchronize all UI elements with current EngineStatus."""
        self._active_status = status
        self.panel_controls.update_status(status)
        self.panel_diagnostics.update_status(status)

        is_streaming = status.state in (EngineState.RUNNING, EngineState.STARTING, EngineState.STOPPING)
        self.panel_audio_settings.set_streaming(is_streaming)

        loaded_id = status.profile_id if (status.profile_id and status.model_loaded) else None
        self.panel_library.set_active_profile(loaded_id)
        self.panel_profiles.set_active_profile(loaded_id)
        self._refresh_selection()

    def _on_window_close(self) -> None:
        """Gracefully shutdown service and release GPU memory upon exit."""
        self.root.title(f"{APP_NAME} — đang giải phóng tài nguyên…")
        self.root.update_idletasks()

        def worker() -> None:
            try:
                self.service.shutdown()
            except Exception:
                logger.exception("Shutdown raised")
            finally:
                self._dispatch(self.root.destroy)

        threading.Thread(target=worker, daemon=True).start()


def launch_gui(service: Optional[VoiceChangerService] = None) -> None:
    """Launch the RealTimeVoiceChanger desktop GUI."""
    enable_dpi_awareness()  # before tk.Tk(): Tk reads the screen DPI once, when the root is created
    root = tk.Tk()
    app = VoiceChangerApp(root, service=service)
    root.mainloop()
