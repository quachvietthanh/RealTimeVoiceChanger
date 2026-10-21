"""Main Tkinter desktop application for RealTimeVoiceChanger."""

from __future__ import annotations

import ctypes
import logging
import queue
import sys
import threading
from typing import Any, Callable, Optional

import tkinter as tk
from tkinter import messagebox, ttk

from voice_changer.audio.devices import AudioDevice
from voice_changer.conversion.base import ConversionConfig
from voice_changer.diagnostics import configure_logging
from voice_changer.gui.settings import GuiSettings
from voice_changer.gui.theme import Colors, Fonts, setup_theme
from voice_changer.gui.views import (
    ControlsPanel,
    DevicesPanel,
    GuidePanel,
    NoiseSuppressionPanel,
    ProfilesPanel,
    TuningPanel,
)
from voice_changer.service import (
    AudioSettings,
    BackendError,
    DeviceInventory,
    EngineState,
    EngineStatus,
    ProfileSummary,
    VoiceChangerService,
)

logger = logging.getLogger(__name__)


class VoiceChangerApp:
    """Desktop GUI orchestrator for RealTimeVoiceChanger."""

    def __init__(self, root: tk.Tk, service: Optional[VoiceChangerService] = None) -> None:
        self.root = root
        self.root.title("RealTimeVoiceChanger — Chuyển đổi giọng nói thời gian thực")
        self.root.geometry("1080x760")
        self.root.minsize(960, 680)

        # Enable Windows DPI awareness if supported
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

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
        self._worker_thread: Optional[threading.Thread] = None
        self._inventory: Optional[DeviceInventory] = None
        self._profiles: list[ProfileSummary] = []
        self._active_status: Optional[EngineStatus] = None

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

    def _build_layout(self) -> None:
        # Top branding bar
        top_bar = tk.Frame(self.root, bg=Colors.BG_SURFACE, padx=20, pady=12)
        top_bar.pack(fill="x")

        title_box = tk.Frame(top_bar, bg=Colors.BG_SURFACE)
        title_box.pack(side="left")

        tk.Label(
            title_box,
            text="🎙️ RealTimeVoiceChanger",
            font=Fonts.TITLE,
            bg=Colors.BG_SURFACE,
            fg=Colors.TEXT_PRIMARY,
        ).pack(anchor="w")

        tk.Label(
            title_box,
            text="Hệ thống chuyển đổi giọng nói Nam ➔ Nữ thời gian thực sử dụng RVC v2 & RMVPE",
            font=Fonts.CAPTION,
            bg=Colors.BG_SURFACE,
            fg=Colors.TEXT_MUTED,
        ).pack(anchor="w")

        # Top right active profile indicator (single clean badge)
        top_right = tk.Frame(top_bar, bg=Colors.BG_SURFACE)
        top_right.pack(side="right")

        self.lbl_active_profile_badge = tk.Label(
            top_right,
            text="Hồ sơ: Đang khởi tạo...",
            font=Fonts.CAPTION_BOLD,
            bg=Colors.BG_ELEVATED,
            fg=Colors.TEXT_MUTED,
            padx=12,
            pady=4,
        )
        self.lbl_active_profile_badge.pack(side="right")

        # Tab navigation notebook
        self.notebook = ttk.Notebook(self.root, style="TNotebook")
        self.notebook.pack(fill="both", expand=True, padx=16, pady=(10, 4))

        # Tab 1: Bảng điều khiển chính
        tab_dashboard = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(tab_dashboard, text="  ⚙️ BẢNG ĐIỀU KHIỂN & CÀI ĐẶT  ")

        # Tab 2: Hướng dẫn kết nối
        tab_guide = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(tab_guide, text="  📖 HƯỚNG DẪN KẾT NỐI (DISCORD & OBS)  ")

        # --- Populate Tab 1: Dashboard with scrollable container ---
        dash_canvas = tk.Canvas(tab_dashboard, bg=Colors.BG_DARK, highlightthickness=0)
        dash_scrollbar = ttk.Scrollbar(tab_dashboard, orient="vertical", command=dash_canvas.yview)
        dash_container = ttk.Frame(dash_canvas, style="TFrame")

        dash_container.bind(
            "<Configure>",
            lambda e: dash_canvas.configure(scrollregion=dash_canvas.bbox("all")),
        )
        dash_canvas_window = dash_canvas.create_window((0, 0), window=dash_container, anchor="nw")

        def _on_dash_canvas_configure(event: tk.Event) -> None:
            dash_canvas.itemconfig(dash_canvas_window, width=event.width)

        dash_canvas.bind("<Configure>", _on_dash_canvas_configure)
        dash_canvas.configure(yscrollcommand=dash_scrollbar.set)

        def _on_dash_mousewheel(event: tk.Event) -> None:
            dash_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        dash_canvas.bind("<Enter>", lambda e: dash_canvas.bind_all("<MouseWheel>", _on_dash_mousewheel))
        dash_canvas.bind("<Leave>", lambda e: dash_canvas.unbind_all("<MouseWheel>"))

        dash_canvas.pack(side="left", fill="both", expand=True, padx=(4, 0), pady=4)
        dash_scrollbar.pack(side="right", fill="y", padx=(0, 4), pady=4)

        dash_container.columnconfigure(0, weight=1, uniform="col")
        dash_container.columnconfigure(1, weight=1, uniform="col")

        # Left Column: Setup & Audio Routing
        left_col = ttk.Frame(dash_container, style="TFrame")
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        self.panel_profiles = ProfilesPanel(
            left_col,
            service=self.service,
            on_profile_selected=self._handle_profile_selected,
        )
        self.panel_profiles.pack(fill="x", pady=(0, 10))

        self.panel_devices = DevicesPanel(
            left_col,
            on_devices_changed=self._handle_devices_changed,
            on_refresh_requested=self._handle_refresh_devices,
        )
        self.panel_devices.pack(fill="x", pady=(0, 10))

        self.panel_noise = NoiseSuppressionPanel(
            left_col,
            on_noise_changed=self._handle_noise_suppression_changed,
            on_rearm_requested=self._handle_noise_rearm,
        )
        self.panel_noise.pack(fill="x")

        # Right Column: Tuning & Live Session Monitoring
        right_col = ttk.Frame(dash_container, style="TFrame")
        right_col.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        self.panel_tuning = TuningPanel(
            right_col,
            on_config_changed=self._handle_config_changed,
            on_reset_requested=self._handle_reset_config,
        )
        self.panel_tuning.pack(fill="x", pady=(0, 10))

        self.panel_controls = ControlsPanel(
            right_col,
            on_start_requested=self._handle_start_requested,
            on_stop_requested=self._handle_stop_requested,
        )
        self.panel_controls.pack(fill="x")

        # --- Populate Tab 2: Guide ---
        self.panel_guide = GuidePanel(tab_guide)
        self.panel_guide.pack(fill="both", expand=True)

        # Bottom status bar
        self.bottom_bar = tk.Frame(self.root, bg=Colors.BG_SURFACE, padx=16, pady=6)
        self.bottom_bar.pack(fill="x", side="bottom")

        self.lbl_status_text = tk.Label(
            self.bottom_bar,
            text="Sẵn sàng",
            font=Fonts.CAPTION,
            bg=Colors.BG_SURFACE,
            fg=Colors.TEXT_MUTED,
        )
        self.lbl_status_text.pack(side="left")

        log_str = str(self.log_path) if self.log_path else "logs/backend.log"
        tk.Label(
            self.bottom_bar,
            text=f"Log: {log_str}",
            font=Fonts.CAPTION,
            bg=Colors.BG_SURFACE,
            fg=Colors.TEXT_MUTED,
        ).pack(side="right")

    # ----------------------------------------------------------------- Async Execution
    def _run_async(self, func: Callable[..., None], *args: Any) -> None:
        """Run long operation in background thread without freezing GUI."""
        def worker() -> None:
            try:
                func(*args)
            except Exception as err:
                logger.exception("Async worker failed")
                self._dispatch(lambda: self._handle_async_error(err))

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
        """Present background worker error to the user."""
        if isinstance(err, BackendError):
            messagebox.showerror(
                title=f"Lỗi: {err.category}",
                message=err.message,
                detail=f"Gợi ý: {err.hint}\n\nChi tiết kỹ thuật:\n{err.details}",
            )
        else:
            messagebox.showerror(
                title="Lỗi xử lý",
                message=str(err),
            )
        self._sync_status(self.service.get_status())

    # ----------------------------------------------------------------- Initialization
    def _initial_load(self) -> None:
        """List profiles, devices, and optionally load default/saved profile."""
        self._dispatch(lambda: self.lbl_status_text.config(text="Đang tải danh sách hồ sơ và thiết bị âm thanh..."))
        profiles = self.service.list_profiles()
        inventory = self.service.list_devices()
        ns_status = self.service.get_noise_suppression()
        target_profile_id = self.settings.profile_id or self.service.default_profile_id

        def ui_init() -> None:
            self._profiles = profiles
            self._inventory = inventory

            # Configure devices from settings
            in_id = self.settings.resolve_input_device_id(inventory)
            out_id = self.settings.resolve_output_device_id(inventory)
            self.panel_devices.set_inventory(inventory, in_id, out_id)

            # Configure noise suppression (defaults to OFF from backend)
            self.panel_noise.set_status(ns_status)

            # Configure profiles
            self.panel_profiles.set_profiles(profiles, target_profile_id)
            self._sync_status(self.service.get_status())
            self.lbl_status_text.config(text="Khởi tạo hoàn tất. Sẵn sàng hoạt động.")

        self._dispatch(ui_init)

        # Automatically load the initial profile into GPU
        if target_profile_id:
            self._load_profile_worker(target_profile_id)

    # ----------------------------------------------------------------- Event Handlers
    def _handle_profile_selected(self, profile_id: str) -> None:
        status = self.service.get_status()
        if status.state == EngineState.RUNNING:
            ans = messagebox.askyesno(
                "Đang chuyển đổi giọng nói",
                "Hệ thống đang hoạt động. Cần dừng phiên hiện tại để nạp hồ sơ giọng mới.\nBạn có muốn dừng và tiếp tục?",
            )
            if not ans:
                return
            self._run_async(self._switch_profile_worker, profile_id)
        else:
            self._run_async(self._load_profile_worker, profile_id)

    def _switch_profile_worker(self, profile_id: str) -> None:
        self._dispatch(lambda: self.panel_profiles.set_loading(True))
        try:
            self.service.stop()
            self._load_profile_worker(profile_id)
        except Exception as err:
            self._dispatch(lambda: self._handle_async_error(err))
        finally:
            self._dispatch(lambda: self.panel_profiles.set_loading(False))

    def _load_profile_worker(self, profile_id: str) -> None:
        self._dispatch(lambda: self.panel_profiles.set_loading(True))
        try:
            self.service.select_profile(profile_id)
            self.settings.profile_id = profile_id
            self.settings.save()
            cfg = self.service.get_config()
            self._dispatch(lambda: self.panel_tuning.set_config(cfg))
        except Exception as err:
            self._dispatch(lambda: self._handle_async_error(err))
        finally:
            self._dispatch(lambda: self.panel_profiles.set_loading(False))

    def _handle_devices_changed(self, inp: Optional[AudioDevice], out: Optional[AudioDevice]) -> None:
        self.settings.update_input(inp)
        self.settings.update_output(out)
        self.settings.save()

    def _handle_refresh_devices(self) -> None:
        def worker() -> None:
            inv = self.service.list_devices()
            in_id = self.settings.resolve_input_device_id(inv)
            out_id = self.settings.resolve_output_device_id(inv)
            self._dispatch(lambda: self.panel_devices.set_inventory(inv, in_id, out_id))

        self._run_async(worker)

    def _handle_config_changed(self, changes: dict[str, float | int]) -> None:
        try:
            new_cfg = self.service.update_config(**changes)
            self.settings.pitch_shift = new_cfg.pitch_shift
            self.settings.index_rate = new_cfg.index_rate
            self.settings.filter_radius = new_cfg.filter_radius
            self.settings.protect_rate = new_cfg.protect_rate
            self.settings.save()
        except BackendError as err:
            self.panel_tuning.set_config(self.service.get_config())
            messagebox.showwarning("Cài đặt không hợp lệ", err.message)

    def _handle_reset_config(self) -> None:
        try:
            new_cfg = self.service.reset_config()
            self.panel_tuning.set_config(new_cfg)
        except Exception as err:
            self._handle_async_error(err)

    def _handle_noise_suppression_changed(
        self, enabled: Optional[bool], mode: Optional[str]
    ) -> None:
        """Handle noise suppression toggle or mode changes from the GUI."""
        try:
            new_status = self.service.set_noise_suppression(enabled=enabled, mode=mode)
            self.panel_noise.set_status(new_status)
        except BackendError as err:
            logger.warning("Noise suppression configuration error: %s", err)
            current = self.service.get_noise_suppression()
            self.panel_noise.set_status(current)
            self.panel_noise.show_error(err.message, err.hint)
        except Exception as err:
            logger.exception("Unexpected error setting noise suppression")
            current = self.service.get_noise_suppression()
            self.panel_noise.set_status(current)
            self.panel_noise.show_error("Không thể cập nhật khử tiếng ồn", str(err))

    def _handle_noise_rearm(self) -> None:
        """Handle re-arm action after stage disabled itself due to errors."""
        self._handle_noise_suppression_changed(enabled=True, mode=None)

    def _handle_start_requested(self) -> None:
        inp, out = self.panel_devices.get_selected_devices()
        in_id = inp.device_id if inp else None
        out_id = out.device_id if out else None

        settings = AudioSettings(input_device_id=in_id, output_device_id=out_id)

        def worker() -> None:
            try:
                self.service.start(settings)
            except Exception as err:
                self._dispatch(lambda: self._handle_async_error(err))

        self._run_async(worker)

    def _handle_stop_requested(self) -> None:
        def worker() -> None:
            try:
                self.service.stop()
            except Exception as err:
                self._dispatch(lambda: self._handle_async_error(err))

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
            self.panel_controls.update_metrics(metrics)

            ns_status = self.service.get_noise_suppression()
            self.panel_noise.sync_live_state(ns_status, metrics)
        except Exception:
            logger.exception("Error during metric polling")
        finally:
            self.root.after(300, self._poll_metrics)

    def _sync_status(self, status: EngineStatus) -> None:
        """Synchronize UI elements with current EngineStatus."""
        self._active_status = status
        self.panel_controls.update_status(status)

        is_streaming = status.state in (EngineState.RUNNING, EngineState.STARTING, EngineState.STOPPING)
        self.panel_devices.set_streaming(is_streaming)

        # Update top badge (single clean label without duplicate status tags)
        if status.profile_id:
            profile_name = status.profile_id
            for p in self._profiles:
                if p.profile_id == status.profile_id:
                    profile_name = p.display_name
                    break
            self.lbl_active_profile_badge.config(
                text=f"Hồ sơ: {profile_name}",
                bg=Colors.SUCCESS_BG if status.model_loaded else Colors.BG_ELEVATED,
                fg=Colors.SUCCESS_TEXT if status.model_loaded else Colors.TEXT_MUTED,
            )
        else:
            self.lbl_active_profile_badge.config(
                text="Chưa nạp hồ sơ",
                bg=Colors.BG_ELEVATED,
                fg=Colors.TEXT_MUTED,
            )

        # Update bottom status text
        if status.state == EngineState.RUNNING:
            self.lbl_status_text.config(text="Đang chuyển đổi âm thanh thời gian thực (Micro ➔ RVC ➔ Cáp ảo)", fg=Colors.SUCCESS_TEXT)
        elif status.state == EngineState.READY:
            self.lbl_status_text.config(text="Mô hình đã nạp sẵn trong GPU. Nhấn 'Bắt đầu' để sử dụng.", fg=Colors.TEXT_MUTED)
        elif status.state == EngineState.ERROR:
            self.lbl_status_text.config(text="Hệ thống phát hiện lỗi. Xem chi tiết trong khung thông báo đỏ.", fg=Colors.DANGER_TEXT)

    def _on_window_close(self) -> None:
        """Gracefully shutdown service and release GPU memory upon exit."""
        self.lbl_status_text.config(text="Đang giải phóng tài nguyên GPU...")
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
    root = tk.Tk()
    app = VoiceChangerApp(root, service=service)
    root.mainloop()
