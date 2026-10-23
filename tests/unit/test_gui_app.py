"""Unit tests for VoiceChangerApp and view panels (headless execution)."""

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import tkinter as tk

src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from voice_changer.audio.devices import AudioDevice
from voice_changer.conversion.base import ConversionConfig
from voice_changer.gui.app import VoiceChangerApp
from voice_changer.gui.settings import GuiSettings, _get_settings_path
from voice_changer.gui.theme import Colors
from voice_changer.service import (
    AudioSettings,
    ConfigurationError,
    DeviceInventory,
    EngineState,
    EngineStatus,
    ErrorInfo,
    NoiseSuppressionStatus,
    ProfileError,
    ProfileSummary,
    VoiceChangerService,
)


class TestVoiceChangerApp(unittest.TestCase):
    def setUp(self) -> None:
        # The app saves GuiSettings on profile/device/tuning changes. Without this, every test run overwrote the
        # owner's real %LOCALAPPDATA%\RealTimeVoiceChanger\gui_settings.json with the fake devices below.
        settings_dir = tempfile.TemporaryDirectory()
        self.addCleanup(settings_dir.cleanup)
        env = mock.patch.dict(os.environ, {"LOCALAPPDATA": settings_dir.name})
        env.start()
        self.addCleanup(env.stop)
        self.settings_dir = Path(settings_dir.name)

        self.root = tk.Tk()
        self.root.withdraw()  # Headless: do not show window

        # Mock VoiceChangerService
        self.mock_service = mock.MagicMock(spec=VoiceChangerService)
        self.mock_service.default_profile_id = "vctk_p231"

        self.mock_profiles = [
            ProfileSummary(
                profile_id="vctk_p231",
                display_name="VCTK Female p231",
                description="English Prototype",
                status="prototype",
                language="en",
                defaults=ConversionConfig(pitch_shift=12.0, index_rate=0.75, filter_radius=3, protect_rate=0.33),
                files_present=True,
                is_active=True,
                notes="Standard test model",
            ),
            ProfileSummary(
                profile_id="voice_a_preview",
                display_name="Speaker A (Northern)",
                description="Northern Vietnamese Female",
                status="preview",
                language="vi",
                defaults=ConversionConfig(pitch_shift=12.0, index_rate=0.50, filter_radius=2, protect_rate=0.40),
                files_present=True,
                is_active=False,
                notes="Preview e50",
            ),
        ]
        self.mock_service.list_profiles.return_value = self.mock_profiles
        self.mock_service.resolve_profile_id.return_value = "vctk_p231"

        dev_in = AudioDevice(0, "Test Mic", "MME", 1, 0, 40000.0)
        dev_out = AudioDevice(1, "CABLE Input", "MME", 0, 2, 40000.0)
        self.mock_inventory = DeviceInventory(
            inputs=(dev_in,),
            outputs=(dev_out,),
            default_input_id=0,
            default_output_id=1,
        )
        self.mock_service.list_devices.return_value = self.mock_inventory

        self.mock_service.get_status.return_value = EngineStatus(
            state=EngineState.READY,
            profile_id="vctk_p231",
            profile_status="prototype",
            model_loaded=True,
            config=self.mock_profiles[0].defaults,
            audio=None,
            session_id=1,
            health_problems=(),
            last_error=None,
        )
        self.mock_service.get_config.return_value = self.mock_profiles[0].defaults
        self.mock_service.get_metrics.return_value = None
        self.mock_service.get_noise_suppression.return_value = NoiseSuppressionStatus(
            enabled=False,
            mode="medium",
            available_modes=("light", "medium", "strong"),
            max_attenuation_db=-12.0,
            latency_ms=20.0,
            active=False,
            failed=False,
            last_error="",
        )

        with mock.patch("voice_changer.gui.app.configure_logging", return_value=Path("test.log")):
            self.app = VoiceChangerApp(self.root, service=self.mock_service)

    def tearDown(self) -> None:
        try:
            self.root.update_idletasks()
        except Exception:
            pass
        self.root.destroy()

    def test_settings_are_written_to_the_test_directory(self) -> None:
        self.assertEqual(_get_settings_path().parent, self.settings_dir / "RealTimeVoiceChanger")

    def test_startup_loads_the_profile_resolved_by_the_backend(self) -> None:
        # A saved profile that is no longer selectable (e.g. withdrawn voice A) must not be loaded at startup.
        self.app.settings.profile_id = "voice_a_preview"
        self.mock_service.resolve_profile_id.reset_mock()
        self.mock_service.select_profile.reset_mock()
        self.app._initial_load()
        self.mock_service.resolve_profile_id.assert_called_once_with("voice_a_preview")
        self.mock_service.select_profile.assert_called_once_with("vctk_p231")

    def test_startup_loads_nothing_when_no_profile_is_selectable(self) -> None:
        self.mock_service.resolve_profile_id.return_value = None
        self.mock_service.select_profile.reset_mock()
        self.app._initial_load()
        self.mock_service.select_profile.assert_not_called()

    def test_switch_while_running_to_an_invalid_profile_keeps_the_stream(self) -> None:
        self.mock_service.validate_profile.return_value = mock.MagicMock(
            ok=False, errors=("index file does not match its registered SHA-256",))
        self.mock_service.stop.reset_mock()
        self.mock_service.select_profile.reset_mock()
        with mock.patch("tkinter.messagebox.showerror") as mock_err:
            self.app._switch_profile_worker("voice_a_preview")
            self.app._process_queue()
            mock_err.assert_not_called()  # shown inline in Studio, not as a modal dialog
        self.assertEqual(self.app.panel_controls.banner_error.winfo_manager(), "pack")
        self.assertIn("SHA-256", self.app.panel_controls.lbl_error_hint.cget("text"))
        self.assertIn("giữ nguyên", self.app.panel_controls.lbl_error_msg.cget("text"))
        self.mock_service.stop.assert_not_called()
        self.mock_service.select_profile.assert_not_called()

    def test_switch_while_running_to_a_valid_profile_stops_then_loads(self) -> None:
        self.mock_service.validate_profile.return_value = mock.MagicMock(ok=True, errors=())
        calls = mock.MagicMock()
        self.mock_service.stop.side_effect = lambda: calls.stop()
        self.mock_service.select_profile.side_effect = lambda pid: calls.select(pid)
        self.app._switch_profile_worker("voice_a_preview")
        self.assertEqual(calls.mock_calls, [mock.call.stop(), mock.call.select("voice_a_preview")])

    def test_app_initialization_components(self) -> None:
        """Verify that all tabs and panels are cleanly constructed."""
        self.assertIsNotNone(self.app.panel_profiles)
        self.assertFalse(hasattr(self.app, "panel_devices"))  # devices are chosen only in Audio & Connections
        self.assertIsNotNone(self.app.panel_noise)
        self.assertIsNotNone(self.app.panel_tuning)
        self.assertIsNotNone(self.app.panel_controls)
        self.assertIsNotNone(self.app.panel_guide)
        self.assertIsNotNone(self.app.panel_library)
        self.assertIsNotNone(self.app.panel_diagnostics)
        self.assertEqual(self.app.notebook.index("end"), 4)

    def test_studio_quick_link_keyboard_accessibility(self) -> None:
        """Verify the Studio link to the Library is keyboard focusable, activates on Return/Space, and handles FocusIn/FocusOut."""
        btn = self.app.panel_profiles.btn_library
        self.assertTrue(btn.cget("takefocus") in (1, "1", True))
        self.assertEqual(int(btn.cget("highlightthickness")), 1)

        # Initial unfocused border matches background
        self.assertEqual(btn.cget("highlightbackground"), Colors.BG_DARK)

        # Focus in triggers accent border
        btn._on_focus_in()
        self.assertEqual(btn.cget("highlightbackground"), Colors.ACCENT_CYAN)

        # Focus out restores background border
        btn._on_focus_out()
        self.assertEqual(btn.cget("highlightbackground"), Colors.BG_DARK)

        # Verify Return activates navigation exactly once
        with mock.patch.object(self.app, "_navigate_to") as mock_nav:
            btn._activate()
            self.assertEqual(mock_nav.call_count, 1)
            mock_nav.assert_called_with("library")

        # Verify bindings exist for keyboard and focus
        self.assertTrue(bool(btn.bind("<Return>")))
        self.assertTrue(bool(btn.bind("<space>")))
        self.assertTrue(bool(btn.bind("<Button-1>")))
        self.assertTrue(bool(btn.bind("<FocusIn>")))
        self.assertTrue(bool(btn.bind("<FocusOut>")))


    def test_sync_status_states(self) -> None:
        """Verify status sync for different engine states."""
        # 1. State: IDLE
        status_idle = EngineStatus(
            state=EngineState.IDLE,
            profile_id=None,
            profile_status=None,
            model_loaded=False,
            config=None,
            audio=None,
            session_id=0,
            health_problems=(),
            last_error=None,
        )
        self.app._sync_status(status_idle)
        self.assertIn("CHƯA NẠP GIỌNG", str(self.app.panel_controls.pill_state.cget("text")))
        self.assertEqual(str(self.app.panel_controls.btn_action.cget("state")), "disabled")  # nothing selected

        # 2. State: READY
        status_ready = EngineStatus(
            state=EngineState.READY,
            profile_id="vctk_p231",
            profile_status="prototype",
            model_loaded=True,
            config=self.mock_profiles[0].defaults,
            audio=None,
            session_id=1,
            health_problems=(),
            last_error=None,
        )
        self.app._sync_status(status_ready)
        self.assertIn("SẴN SÀNG", str(self.app.panel_controls.pill_state.cget("text")))
        self.assertEqual(str(self.app.panel_controls.btn_action.cget("state")), "normal")
        self.assertIn("Bắt đầu chuyển giọng", str(self.app.panel_controls.btn_action.cget("text")))

        # 3. State: RUNNING
        status_running = EngineStatus(
            state=EngineState.RUNNING,
            profile_id="vctk_p231",
            profile_status="prototype",
            model_loaded=True,
            config=self.mock_profiles[0].defaults,
            audio=AudioSettings(),
            session_id=1,
            health_problems=(),
            last_error=None,
        )
        self.app._sync_status(status_running)
        self.assertIn("ĐANG CHUYỂN GIỌNG", str(self.app.panel_controls.pill_state.cget("text")))
        self.assertIn("Dừng", str(self.app.panel_controls.btn_action.cget("text")))

        # 4. State: ERROR
        status_error = EngineStatus(
            state=EngineState.ERROR,
            profile_id="vctk_p231",
            profile_status="prototype",
            model_loaded=True,
            config=self.mock_profiles[0].defaults,
            audio=AudioSettings(),
            session_id=1,
            health_problems=("test stream lost",),
            last_error=ErrorInfo(
                category="audio_device",
                code="stream_failure",
                message="Micro bị ngắt kết nối",
                hint="Kiểm tra lại dây cắm",
                details="PortAudio error",
                timestamp="2026-10-21T21:00:00",
            ),
        )
        self.app._sync_status(status_error)
        self.assertIn("CÓ LỖI", str(self.app.panel_controls.pill_state.cget("text")))
        self.assertIn("Micro bị ngắt kết nối", self.app.panel_controls.lbl_error_msg.cget("text"))

    def test_window_resizing_and_layout(self) -> None:
        """Verify window layout responds to common dimensions without errors."""
        self.app.section_tuning.set_expanded(True)
        self.app.section_noise.set_expanded(True)
        for width, height in [(960, 680), (1080, 760), (1280, 800)]:
            self.root.geometry(f"{width}x{height}")
            self.root.update_idletasks()
            # Verify panels remain mapped and populated
            self.assertGreater(self.app.panel_profiles.winfo_width(), 0)
            self.assertGreater(self.app.panel_tuning.winfo_width(), 0)
            self.assertGreater(self.app.panel_noise.winfo_width(), 0)
            self.assertGreater(self.app.panel_controls.winfo_width(), 0)

    def test_metrics_dashboard_update(self) -> None:
        """Detailed metrics are formatted in Advanced; Studio shows only a one-line note while live."""
        metrics_sample = {
            "rtf": 0.452,
            "avg_inference_ms": 226.0,
            "avg_software_e2e_ms": 500.0,
            "estimated_software_latency_ms": 575.0,
            "current_buffer_ms": 150.0,
            "chunks_converted": 45,
            "chunks_dropped": 0,
            "steady_underflows": 0,
        }
        pd = self.app.panel_diagnostics
        pd.update_metrics(metrics_sample)

        self.assertEqual(pd.val_diag_rtf.cget("text"), "0.452")
        self.assertEqual(pd.val_diag_gpu.cget("text"), "226 ms")
        self.assertEqual(pd.val_diag_e2e.cget("text"), "~575 ms")
        self.assertEqual(pd.val_diag_buf.cget("text"), "150 ms")
        self.assertEqual(pd.val_diag_chunks.cget("text"), "45 / 0")
        self.assertEqual(pd.val_diag_und.cget("text"), "0")

        # Fallback when estimated_software_latency_ms is absent
        pd.update_metrics({"avg_software_e2e_ms": 500.0})
        self.assertEqual(pd.val_diag_e2e.cget("text"), "~500 ms")

        # Studio: no "live" flag means no claim of live performance.
        self.app.panel_controls.update_performance(metrics_sample)
        self.assertEqual(self.app.panel_controls.lbl_perf.cget("text"), "")
        self.app.panel_controls.update_performance(dict(metrics_sample, live=True))
        self.assertIn("ổn định", self.app.panel_controls.lbl_perf.cget("text"))
        self.app.panel_controls.update_performance(dict(metrics_sample, live=True, steady_underflows=2))
        self.assertIn("gián đoạn", self.app.panel_controls.lbl_perf.cget("text"))

    def test_tuning_panel_values(self) -> None:
        """Verify that tuning sliders read and write config correctly."""
        cfg = ConversionConfig(pitch_shift=14.0, index_rate=0.45, filter_radius=4, protect_rate=0.25)
        self.app.panel_tuning.set_config(cfg)

        self.assertEqual(self.app.panel_tuning.val_pitch.get(), 14.0)
        self.assertEqual(self.app.panel_tuning.val_index.get(), 0.45)
        self.assertEqual(self.app.panel_tuning.val_filter.get(), 4)
        self.assertEqual(self.app.panel_tuning.val_protect.get(), 0.25)

    def test_noise_suppression_initial_default_state(self) -> None:
        """Verify noise suppression defaults to OFF and medium mode from backend."""
        self.assertFalse(self.app.panel_noise.var_enabled.get())
        self.assertEqual(self.app.panel_noise.lbl_status.cget("text"), "● ĐANG TẮT")
        self.assertIn("Vừa (-12 dB)", self.app.panel_noise.combo_mode.get())
        self.assertIn("20.0 ms", self.app.panel_noise.lbl_latency_info.cget("text"))
        self.assertIn("Mức lọc quá mạnh", self.app.panel_noise.lbl_warn.cget("text"))

    def test_noise_suppression_toggle_on_off(self) -> None:
        """Verify toggle ON and OFF updates backend and status pill."""
        # 1. Toggle ON via prominent header toggle button
        self.mock_service.set_noise_suppression.return_value = NoiseSuppressionStatus(
            enabled=True,
            mode="medium",
            available_modes=("light", "medium", "strong"),
            max_attenuation_db=-12.0,
            latency_ms=20.0,
            active=False,
            failed=False,
            last_error="",
        )
        self.app.panel_noise.btn_toggle.invoke()
        self.mock_service.set_noise_suppression.assert_called_with(enabled=True, mode=None)
        self.assertEqual(self.app.panel_noise.lbl_status.cget("text"), "● ĐÃ BẬT (CHỜ STREAMING)")
        self.assertIn("🟢 ĐÃ BẬT", self.app.panel_noise.btn_toggle.cget("text"))

        # 2. Toggle OFF via prominent header toggle button
        self.mock_service.set_noise_suppression.return_value = NoiseSuppressionStatus(
            enabled=False,
            mode="medium",
            available_modes=("light", "medium", "strong"),
            max_attenuation_db=-12.0,
            latency_ms=20.0,
            active=False,
            failed=False,
            last_error="",
        )
        self.app.panel_noise.btn_toggle.invoke()
        self.mock_service.set_noise_suppression.assert_called_with(enabled=False, mode=None)
        self.assertEqual(self.app.panel_noise.lbl_status.cget("text"), "● ĐANG TẮT")
        self.assertIn("⚪ TẮT", self.app.panel_noise.btn_toggle.cget("text"))

    def test_noise_suppression_mode_selection(self) -> None:
        """Verify selecting a mode in combobox updates backend."""
        self.mock_service.set_noise_suppression.return_value = NoiseSuppressionStatus(
            enabled=False,
            mode="strong",
            available_modes=("light", "medium", "strong"),
            max_attenuation_db=-18.0,
            latency_ms=20.0,
            active=False,
            failed=False,
            last_error="",
        )
        # Select "strong" (index 2 in ["light", "medium", "strong"])
        self.app.panel_noise.combo_mode.current(2)
        self.app.panel_noise._on_mode_selected()
        self.mock_service.set_noise_suppression.assert_called_with(enabled=None, mode="strong")
        self.assertIn("Mạnh (-18 dB)", self.app.panel_noise.combo_mode.get())

    def test_noise_suppression_error_handling_and_rollback(self) -> None:
        """Verify error banner is displayed and control rolls back when backend raises error."""
        self.mock_service.set_noise_suppression.side_effect = ConfigurationError(
            "invalid_setting",
            "Chế độ lọc không hợp lệ",
            hint="Chỉ hỗ trợ: light, medium, strong",
        )
        self.app.panel_noise.var_enabled.set(True)
        self.app.panel_noise._on_toggle()

        # Banner should display error message and hint
        self.assertIn("Chế độ lọc không hợp lệ", self.app.panel_noise.lbl_error_msg.cget("text"))
        self.assertIn("Chỉ hỗ trợ", self.app.panel_noise.lbl_error_hint.cget("text"))
        # Control rolled back to match get_noise_suppression (False)
        self.assertFalse(self.app.panel_noise.var_enabled.get())

    def test_noise_suppression_failed_stage_and_rearm(self) -> None:
        """Verify failed stage shows error status and re-arm button re-enables it."""
        failed_status = NoiseSuppressionStatus(
            enabled=True,
            mode="medium",
            available_modes=("light", "medium", "strong"),
            max_attenuation_db=-12.0,
            latency_ms=20.0,
            active=False,
            failed=True,
            last_error="Stage failed 3 consecutive chunks",
        )
        self.app.panel_noise.set_status(failed_status)
        self.assertIn("TỰ NGẮT", self.app.panel_noise.lbl_status.cget("text"))

        # Re-arm via button
        self.mock_service.set_noise_suppression.side_effect = None
        self.mock_service.set_noise_suppression.return_value = NoiseSuppressionStatus(
            enabled=True,
            mode="medium",
            available_modes=("light", "medium", "strong"),
            max_attenuation_db=-12.0,
            latency_ms=20.0,
            active=False,
            failed=False,
            last_error="",
        )
        self.app.panel_noise._on_rearm()
        self.mock_service.set_noise_suppression.assert_called_with(enabled=True, mode=None)
        self.assertEqual(self.app.panel_noise.lbl_status.cget("text"), "● ĐÃ BẬT (CHỜ STREAMING)")

    def test_noise_suppression_live_metrics_sync(self) -> None:
        """Verify live streaming metrics update CPU time and active status."""
        active_status = NoiseSuppressionStatus(
            enabled=True,
            mode="medium",
            available_modes=("light", "medium", "strong"),
            max_attenuation_db=-12.0,
            latency_ms=20.0,
            active=True,
            failed=False,
            last_error="",
        )
        metrics = {"avg_noise_suppression_ms": 0.38}
        self.app.panel_noise.sync_live_state(active_status, metrics)
        self.assertEqual(self.app.panel_noise.lbl_status.cget("text"), "● ĐANG LỌC TIẾNG ỒN")
        self.assertIn("0.38 ms", self.app.panel_noise.lbl_latency_info.cget("text"))

    def test_handle_profile_selected_planned_warns_and_blocks_load(self) -> None:
        """Verify selecting planned profile Jett or Sage pops warning and blocks backend load."""
        jett_profile = ProfileSummary(
            profile_id="jett",
            display_name="Jett",
            description="Planned Valorant agent.",
            status="planned",
            language="en",
            defaults=ConversionConfig(pitch_shift=12.0, index_rate=0.50, filter_radius=3, protect_rate=0.33),
            files_present=False,
            is_active=False,
            notes="Planned profile",
            readiness="no_model",
            setup_hint="Chưa có dataset và checkpoint trong hệ thống.",
        )
        self.app._profiles.append(jett_profile)
        self.mock_service.select_profile.reset_mock()

        with mock.patch("tkinter.messagebox.showwarning") as mock_warn:
            self.app._handle_profile_selected("jett")
            mock_warn.assert_called_once()
            args, _ = mock_warn.call_args
            self.assertEqual(args[0], "Hồ sơ chưa có mô hình")
            self.assertIn("Chưa thể chuyển giọng", args[1])
            self.mock_service.select_profile.assert_not_called()

    def test_handle_profile_selected_missing_files_blocks_load(self) -> None:
        """Verify selecting a profile with missing files pops error and blocks backend load."""
        missing_profile = ProfileSummary(
            profile_id="missing_voice",
            display_name="Missing Voice",
            description="Profile with missing files.",
            status="preview",
            language="vi",
            defaults=ConversionConfig(pitch_shift=12.0, index_rate=0.50, filter_radius=3, protect_rate=0.33),
            files_present=False,
            is_active=False,
            notes="",
            readiness="files_missing",
            setup_hint="Kiểm tra lại đường dẫn tệp.",
        )
        self.app._profiles.append(missing_profile)
        self.mock_service.select_profile.reset_mock()

        with mock.patch("tkinter.messagebox.showerror") as mock_err:
            self.app._handle_profile_selected("missing_voice")
            mock_err.assert_called_once()
            args, _ = mock_err.call_args
            self.assertEqual(args[0], "Thiếu tệp mô hình")
            self.assertIn("không tìm thấy tệp checkpoint", args[1])
            self.mock_service.select_profile.assert_not_called()

    def test_handle_profile_selected_while_running_does_not_stop_stream_or_unload_model(self) -> None:
        """Verify Case 1: Selecting Jett while stream is RUNNING does not stop stream or unload model."""
        jett_profile = ProfileSummary(
            profile_id="jett",
            display_name="Jett",
            description="Planned Valorant agent.",
            status="planned",
            language="en",
            defaults=ConversionConfig(pitch_shift=12.0, index_rate=0.50, filter_radius=3, protect_rate=0.33),
            files_present=False,
            is_active=False,
            notes="Planned profile",
            readiness="no_model",
            setup_hint="Chưa có dataset và checkpoint trong hệ thống.",
        )
        self.app._process_queue()
        self.app._profiles.append(jett_profile)

        # Set engine running with voice_a_preview
        running_status = EngineStatus(
            state=EngineState.RUNNING,
            profile_id="voice_a_preview",
            profile_status="preview",
            model_loaded=True,
            config=self.mock_profiles[1].defaults,
            audio=AudioSettings(),
            session_id=1,
            health_problems=(),
            last_error=None,
        )
        self.mock_service.get_status.return_value = running_status
        self.app._sync_status(running_status)

        self.assertIn("Speaker A (Northern)", self.app.panel_controls.lbl_status_line.cget("text"))
        self.mock_service.stop.reset_mock()
        self.mock_service.select_profile.reset_mock()

        with mock.patch("tkinter.messagebox.showwarning") as mock_warn, \
             mock.patch("tkinter.messagebox.askyesno") as mock_ask:
            self.app._handle_profile_selected("jett")

            # Must warn and return without stopping stream or asking confirmation
            mock_warn.assert_called_once()
            mock_ask.assert_not_called()
            self.mock_service.stop.assert_not_called()
            self.mock_service.select_profile.assert_not_called()

        # The running session and its description remain unchanged
        self.assertIn("Speaker A (Northern)", self.app.panel_controls.lbl_status_line.cget("text"))
        self.assertEqual(self.app.panel_controls.action.kind, "stop")

    def test_handle_profile_selected_malformed_readiness_blocks_load(self) -> None:
        """Verify Case 4: Malformed readiness field fails safely and blocks load."""
        malformed_profile = ProfileSummary(
            profile_id="malformed_voice",
            display_name="Malformed Voice",
            description="Malformed profile with null readiness.",
            status="preview",
            language="vi",
            defaults=ConversionConfig(pitch_shift=12.0, index_rate=0.50, filter_radius=3, protect_rate=0.33),
            files_present=True,
            is_active=False,
            notes="",
        )
        object.__setattr__(malformed_profile, "readiness", None)
        self.app._profiles.append(malformed_profile)
        self.mock_service.select_profile.reset_mock()

        with mock.patch("tkinter.messagebox.showerror") as mock_err:
            self.app._handle_profile_selected("malformed_voice")
            mock_err.assert_called_once()
            args, _ = mock_err.call_args
            self.assertEqual(args[0], "Thiếu tệp mô hình")
            self.mock_service.select_profile.assert_not_called()



    def test_startup_resolve_profile_id_fallback(self) -> None:
        """Verify startup uses resolve_profile_id and falls back safely if saved profile is unselectable."""
        self.mock_service.resolve_profile_id.return_value = "vctk_p231"
        self.app.settings.profile_id = "jett"  # Stale/unselectable saved profile

        with mock.patch.object(self.app, "_load_profile_worker") as mock_load:
            self.app._initial_load()
            self.mock_service.resolve_profile_id.assert_called_with("jett")
            mock_load.assert_called_once_with("vctk_p231")

    def test_handle_profile_selected_withdrawn_profile_warns_and_blocks_load(self) -> None:
        """Verify withdrawn profile triggers warning and refuses load."""
        withdrawn_profile = ProfileSummary(
            profile_id="withdrawn_voice",
            display_name="Withdrawn Voice",
            description="Withdrawn preview model.",
            status="preview",
            language="vi",
            defaults=ConversionConfig(pitch_shift=12.0, index_rate=0.50, filter_radius=3, protect_rate=0.33),
            files_present=True,
            is_active=False,
            notes="",
            readiness="files_present",
            withdrawn_reason="Chất lượng âm thanh bị giật/không đạt chuẩn.",
        )
        self.app._profiles.append(withdrawn_profile)
        self.mock_service.select_profile.reset_mock()

        with mock.patch("tkinter.messagebox.showwarning") as mock_warn:
            self.app._handle_profile_selected("withdrawn_voice")
            mock_warn.assert_called_once()
            args, _ = mock_warn.call_args
            self.assertEqual(args[0], "Hồ sơ đã bị thu hồi")
            self.assertIn("rút khỏi danh sách", args[1])
            self.mock_service.select_profile.assert_not_called()

    def test_profile_switching_failed_validation_keeps_stream_running(self) -> None:
        """Verify that when switching profiles while running, failed validation preserves the active stream."""
        # Validation returns errors
        mock_val = mock.MagicMock()
        mock_val.ok = False
        mock_val.errors = ("SHA-256 hash mismatch for model checkpoint",)
        self.mock_service.validate_profile.return_value = mock_val

        self.mock_service.stop.reset_mock()
        self.mock_service.select_profile.reset_mock()

        with mock.patch("tkinter.messagebox.showerror") as mock_err:
            self.app._switch_profile_worker("corrupt_profile")
            self.app._process_queue()
            mock_err.assert_not_called()
        self.assertIn("không hợp lệ", self.app.panel_controls.lbl_error_msg.cget("text"))
        self.assertIn("SHA-256", self.app.panel_controls.lbl_error_hint.cget("text"))

        # Active stream must NOT be stopped!
        self.mock_service.stop.assert_not_called()
        self.mock_service.select_profile.assert_not_called()

    def test_sync_status_idle_state_honest_representation(self) -> None:
        """Verify truthful display of IDLE engine state and unloaded model."""
        status_idle = EngineStatus(
            state=EngineState.IDLE,
            profile_id=None,
            profile_status=None,
            model_loaded=False,
            config=None,
            audio=None,
            session_id=0,
            health_problems=(),
            last_error=None,
        )
        self._settle_initial_load()  # profiles listed; vctk_p231 selected in Studio
        self.app._sync_status(status_idle)

        pc = self.app.panel_controls
        self.assertEqual(pc.pill_state.cget("text"), "● CHƯA NẠP GIỌNG")
        # The selection is offered for loading, never presented as loaded or startable.
        self.assertEqual(pc.action.kind, "load")
        self.assertEqual(pc.btn_action.cget("text"), "Nạp giọng")
        self.assertIn("chưa được nạp", pc.lbl_status_line.cget("text"))
        self.assertIsNone(self.app.panel_profiles._active_profile_id)
        self.assertNotIn("Đang dùng", self.app.panel_profiles.pill_status.cget("text"))

    def test_audio_settings_view_device_synchronization(self) -> None:
        """Verify that AudioSettingsView synchronizes selected endpoints with app state."""
        dev_in = AudioDevice(2, "Studio USB Mic", "MME", 1, 0, 40000.0)
        dev_out = AudioDevice(3, "CABLE Input", "MME", 0, 2, 40000.0)
        inv = DeviceInventory(
            inputs=(dev_in,),
            outputs=(dev_out,),
            default_input_id=2,
            default_output_id=3,
        )
        self.app._inventory = inv
        self.app.panel_audio_settings.set_inventory(inv, selected_input_id=2, selected_output_id=3)

        # Trigger device change through AudioSettingsView
        self.app.panel_audio_settings.on_devices_changed(dev_in, dev_out)

        # App state, settings, and the Studio route summary should be in sync
        self.assertEqual(self.app.settings.input_device_name, "Studio USB Mic")
        self.assertEqual(self.app.settings.output_device_name, "CABLE Input")
        self.assertIn("Studio USB Mic", self.app.panel_controls.lbl_route.cget("text"))
        self.assertIn("CABLE Input", self.app.panel_controls.lbl_route.cget("text"))

    # ------------------------------------------------------------- review regressions (2026-10-22)
    def _status(self, state: EngineState, **kw) -> EngineStatus:
        fields = dict(state=state, profile_id="vctk_p231", profile_status="prototype", model_loaded=True,
                      config=self.mock_profiles[0].defaults, audio=None, session_id=1, health_problems=(),
                      last_error=None)
        fields.update(kw)
        return EngineStatus(**fields)

    def _run_async_inline(self) -> None:
        patcher = mock.patch.object(self.app, "_run_async", lambda func, *args: func(*args))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_background_error_reaches_the_user(self) -> None:
        # The dispatched lambda used to reference `err` after the except block had unbound it, so every
        # load/start/stop error raised NameError in _process_queue and no dialog was shown.
        self.mock_service.select_profile.side_effect = ConfigurationError("x", "Loading voice profile failed.",
                                                                          hint="Check the files.")
        with mock.patch("tkinter.messagebox.showerror") as mock_err:
            self.app._load_profile_worker("voice_a_preview")
            self.app._process_queue()
        mock_err.assert_not_called()  # inline in Studio, next to the action that failed
        pc = self.app.panel_controls
        self.assertEqual(pc.banner_error.winfo_manager(), "pack")
        self.assertIn("Loading voice profile failed.", pc.lbl_error_msg.cget("text"))
        self.assertEqual(pc.lbl_error_hint.cget("text"), "Check the files.")
        # The details dialog names the backend error code and category.
        with mock.patch("voice_changer.gui.views.controls_panel.messagebox.showerror") as dialog:
            pc._show_error_dialog()
        self.assertIn("configuration", dialog.call_args.kwargs["detail"])

    def test_retry_after_stream_failure_stops_then_starts(self) -> None:
        # service.start() only accepts READY; the Retry button shown in ERROR must stop the failed session first.
        self._run_async_inline()
        self.mock_service.get_status.return_value = self._status(EngineState.ERROR, health_problems=("stream lost",))
        self.app._sync_status(self.mock_service.get_status())
        calls = mock.MagicMock()
        self.mock_service.stop.side_effect = lambda: calls.stop()
        self.mock_service.start.side_effect = lambda settings: calls.start()
        self.app.panel_controls.btn_action.invoke()
        self.assertEqual(calls.mock_calls, [mock.call.stop(), mock.call.start()])

    def test_start_from_ready_does_not_call_stop(self) -> None:
        self._run_async_inline()
        self.mock_service.stop.reset_mock()
        self.app._handle_start_requested()
        self.mock_service.stop.assert_not_called()
        self.mock_service.start.assert_called_once()

    def test_load_requested_in_error_state_validates_stops_then_selects(self) -> None:
        # select_profile() refuses ERROR; loading a voice after a stream failure must go through the switch path.
        self._settle_initial_load()
        self._run_async_inline()
        self.mock_service.get_status.return_value = self._status(EngineState.ERROR)
        self.app._sync_status(self.mock_service.get_status())
        calls = mock.MagicMock()
        self.mock_service.validate_profile.side_effect = lambda pid: (calls.validate(pid), mock.MagicMock(ok=True))[1]
        self.mock_service.stop.side_effect = lambda: calls.stop()
        self.mock_service.select_profile.side_effect = lambda pid: calls.select(pid)
        with mock.patch("tkinter.messagebox.askyesno") as mock_ask:
            self.app._handle_profile_selected("voice_a_preview")
            self.assertEqual(calls.mock_calls, [])  # selecting alone touches nothing
            self.assertEqual(self.app.panel_controls.action.kind, "load")
            self.app.panel_controls.btn_action.invoke()
        mock_ask.assert_not_called()
        self.assertEqual(calls.mock_calls, [mock.call.validate("voice_a_preview"), mock.call.stop(),
                                            mock.call.select("voice_a_preview")])

    def test_metrics_before_first_chunk_show_placeholders_not_zero(self) -> None:
        # Real get_metrics() right after start(): averages are 0.0 because nothing was measured yet.
        metrics = {"live": True, "chunks_converted": 0, "chunks_dropped": 0, "steady_underflows": 0, "rtf": 0.0,
                   "avg_inference_ms": 0.0, "avg_software_e2e_ms": 0.0, "estimated_software_latency_ms": 0.0,
                   "current_buffer_ms": 0.0}
        self.app.panel_controls.update_performance(metrics)
        self.app.panel_diagnostics.update_metrics(metrics)
        pc, pd = self.app.panel_controls, self.app.panel_diagnostics
        for label in (pd.val_diag_rtf, pd.val_diag_gpu, pd.val_diag_e2e):
            self.assertEqual(label.cget("text"), "--")
        self.assertEqual(pd.val_diag_buf.cget("text"), "0 ms")  # a live buffer reading of 0 ms is a real value
        self.assertEqual(pd.val_diag_chunks.cget("text"), "0 / 0")
        self.assertEqual(pc.lbl_perf.cget("text"), "")  # no performance verdict before anything was measured
        self.assertTrue(pd.lbl_telemetry_source.cget("text").startswith("Trực tiếp"))

    def test_metrics_after_stop_do_not_show_stale_buffer(self) -> None:
        metrics = {"live": False, "chunks_converted": 40, "chunks_dropped": 0, "steady_underflows": 0, "rtf": 0.5,
                   "avg_inference_ms": 250.0, "estimated_software_latency_ms": 900.0, "current_buffer_ms": 480.0,
                   "session_id": 3}
        self.app.panel_controls.update_performance(metrics)
        self.app.panel_diagnostics.update_metrics(metrics)
        pd = self.app.panel_diagnostics
        self.assertEqual(pd.val_diag_buf.cget("text"), "--")
        self.assertEqual(self.app.panel_controls.lbl_perf.cget("text"), "")  # Studio never shows a stale verdict
        # Final-session averages are real measurements and stay visible, labelled as the previous session.
        self.assertEqual(pd.val_diag_e2e.cget("text"), "~900 ms")
        self.assertEqual(pd.val_diag_rtf.cget("text"), "0.500")
        self.assertIn("Phiên trước (#3)", pd.lbl_telemetry_source.cget("text"))

    def test_studio_action_stays_disabled_while_loading(self) -> None:
        # select_profile() broadcasts LOADING with no profile; that status update used to re-enable the button.
        self.app.panel_profiles.set_profiles(self.mock_profiles, "voice_a_preview")
        self.app._sync_status(self._status(EngineState.READY))  # vctk_p231 active
        self.app._set_profiles_loading(True)
        self.app._sync_status(self._status(EngineState.LOADING, profile_id=None, model_loaded=False))
        self.assertEqual(str(self.app.panel_controls.btn_action.cget("state")), "disabled")
        self.app._sync_status(self._status(EngineState.READY, profile_id="voice_a_preview"))
        self.assertEqual(str(self.app.panel_controls.btn_action.cget("state")), "disabled")  # still loading
        self.app._set_profiles_loading(False)
        self.assertEqual(self.app.panel_controls.action.kind, "start")
        self.assertEqual(str(self.app.panel_controls.btn_action.cget("state")), "normal")

    def test_audio_settings_change_updates_voice_studio_selection(self) -> None:
        mic_a = AudioDevice(2, "Mic A", "MME", 1, 0, 44100.0)
        mic_b = AudioDevice(5, "Mic B", "MME", 1, 0, 44100.0)
        spk = AudioDevice(3, "Speakers", "MME", 0, 2, 44100.0)
        cable = AudioDevice(6, "CABLE Input", "MME", 0, 2, 44100.0)
        inv = DeviceInventory(inputs=(mic_a, mic_b), outputs=(spk, cable), default_input_id=2, default_output_id=3)
        self.app._inventory = inv
        self.app.panel_audio_settings.set_inventory(inv, 2, 3)

        self.app.panel_audio_settings.input_combo.current(1)
        self.app.panel_audio_settings._on_input_selected(None)
        self.app.panel_audio_settings.output_combo.current(1)
        self.app.panel_audio_settings._on_output_selected(None)

        route = self.app.panel_controls.lbl_route.cget("text")
        self.assertIn("Mic B", route)
        self.assertIn("CABLE Input", route)
        self.assertIn("cáp âm thanh ảo", self.app.panel_controls.lbl_route_hint.cget("text"))
        self._run_async_inline()
        self.app._handle_start_requested()
        sent = self.mock_service.start.call_args[0][0]
        self.assertEqual((sent.input_device_id, sent.output_device_id), (5, 6))

    def test_voice_library_choice_synchronizes_voice_studio_without_loading(self) -> None:
        """Choosing a voice in the Library selects it in Studio; only Studio's action loads it."""
        self._settle_initial_load()
        self._run_async_inline()
        self.mock_service.get_config.return_value = self.mock_profiles[1].defaults
        self.mock_service.list_profiles.return_value = self.mock_profiles

        self.app.panel_library._on_card_load_clicked("voice_a_preview")
        self.app._process_queue()

        selected_in_studio = self.app.panel_profiles.get_selected_profile()
        self.assertIsNotNone(selected_in_studio)
        self.assertEqual(selected_in_studio.profile_id, "voice_a_preview")
        self.mock_service.select_profile.assert_not_called()
        self.assertEqual(self.app.panel_controls.btn_action.cget("text"), "Nạp giọng")

        self.mock_service.get_status.return_value = self._status(EngineState.READY, profile_id="voice_a_preview")
        self.app.panel_controls.btn_action.invoke()
        self.app._process_queue()
        self.mock_service.select_profile.assert_called_once_with("voice_a_preview")
        self.app._sync_status(self.mock_service.get_status())
        self.assertEqual(self.app.panel_profiles.pill_status.cget("text"), "● Đang dùng")
        self.assertIn("Bắt đầu chuyển giọng", self.app.panel_controls.btn_action.cget("text"))

    def test_voice_library_selection_navigates_to_studio(self) -> None:
        """Verify that selecting a voice from Library navigates the user directly to Voice Studio."""
        self._settle_initial_load()  # the Library lists only profiles the backend has reported
        self._run_async_inline()
        self.mock_service.get_config.return_value = self.mock_profiles[1].defaults
        self.mock_service.list_profiles.return_value = self.mock_profiles
        self.mock_service.get_status.return_value = self._status(EngineState.READY, profile_id="voice_a_preview")

        # Navigate to Library first
        self.app._navigate_to("library")
        self.assertEqual(self.app._current_view, "library")

        # Select profile via Library card action
        self.app.panel_library._on_card_load_clicked("voice_a_preview")
        self.app._process_queue()

        # View must automatically switch to Studio
        self.assertEqual(self.app._current_view, "studio")
        self.assertEqual(self.app.notebook.index("current"), 0)
        self.assertEqual(self.app.lbl_breadcrumb.cget("text"), "Studio")

    def _settle_initial_load(self) -> None:
        """Let the startup worker thread finish and apply its UI updates, then forget its service calls."""
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and any(
                t.daemon and t.is_alive() for t in threading.enumerate() if t is not threading.current_thread()):
            time.sleep(0.01)
        self.app._process_queue()
        self.mock_service.select_profile.reset_mock()

    def test_library_use_in_studio_selects_the_voice_without_claiming_it_is_loaded(self) -> None:
        self._settle_initial_load()
        deferred = []
        with mock.patch.object(self.app, "_run_async", lambda func, *args: deferred.append((func, args))):
            self.app._navigate_to("library")
            self.app.panel_library._on_card_load_clicked("voice_a_preview")

        # Studio shows the chosen voice as selected while the backend still has vctk_p231 loaded.
        panel = self.app.panel_profiles
        pc = self.app.panel_controls
        self.assertEqual(self.app._current_view, "studio")
        self.assertEqual(panel.get_selected_profile().profile_id, "voice_a_preview")
        self.assertEqual(panel.lbl_hero_name.cget("text"), "Speaker A (Northern)")
        self.assertNotEqual(panel.pill_status.cget("text"), "● Đang dùng")
        self.assertIn("VCTK Female p231", pc.lbl_status_line.cget("text"))  # still the loaded voice
        self.assertEqual(pc.btn_action.cget("text"), "Nạp giọng")
        self.assertEqual(self.app.panel_library._active_profile_id, "vctk_p231")

        # Selecting never loads: nothing was scheduled and the service was not called.
        self.assertEqual(deferred, [])
        self.mock_service.select_profile.assert_not_called()

        # The load uses the app's one loading path, from Studio's action.
        with mock.patch.object(self.app, "_run_async", lambda func, *args: deferred.append((func, args))):
            pc.btn_action.invoke()
        self.assertEqual(deferred, [(self.app._load_profile_worker, ("voice_a_preview",))])

        # Only the backend reporting the new profile as loaded marks it loaded.
        self.mock_service.get_config.return_value = self.mock_profiles[1].defaults
        self.mock_service.get_status.return_value = self._status(EngineState.READY, profile_id="voice_a_preview")
        func, args = deferred[0]
        func(*args)
        self.app._process_queue()
        self.app._sync_status(self.mock_service.get_status())
        self.mock_service.select_profile.assert_called_once_with("voice_a_preview")
        self.assertEqual(panel.pill_status.cget("text"), "● Đang dùng")
        self.assertIn("Speaker A (Northern)", pc.lbl_status_line.cget("text"))
        self.assertEqual(self.app.panel_library._active_profile_id, "voice_a_preview")

    def test_library_load_failure_leaves_the_voice_selected_but_not_loaded(self) -> None:
        self._settle_initial_load()
        self._run_async_inline()
        # The old model was released before the load failed: the engine is IDLE with nothing loaded.
        self.mock_service.select_profile.side_effect = ProfileError("load_failed", "Không nạp được mô hình.")
        self.mock_service.get_status.return_value = self._status(
            EngineState.IDLE, profile_id=None, profile_status=None, model_loaded=False, config=None)

        with mock.patch("tkinter.messagebox.showerror") as mock_err:
            self.app._navigate_to("library")
            self.app.panel_library._on_card_load_clicked("voice_a_preview")
            self.app.panel_controls.btn_action.invoke()
            self.app._process_queue()
        mock_err.assert_not_called()
        self.mock_service.select_profile.assert_called_once_with("voice_a_preview")

        panel = self.app.panel_profiles
        pc = self.app.panel_controls
        self.assertIn("Không nạp được mô hình.", pc.lbl_error_msg.cget("text"))  # inline error
        self.assertEqual(panel.get_selected_profile().profile_id, "voice_a_preview")
        self.assertEqual(str(pc.btn_action.cget("state")), "normal")  # retry is possible
        self.assertEqual(pc.btn_action.cget("text"), "Nạp giọng")
        self.assertNotEqual(panel.pill_status.cget("text"), "● Đang dùng")
        self.assertIn("CHƯA NẠP GIỌNG", pc.pill_state.cget("text"))
        self.assertIsNone(self.app.panel_library._active_profile_id)
        card_btn = self.app.panel_library._rendered_cards["voice_a_preview"]["btn_action_box"].winfo_children()[0]
        self.assertIn("Dùng trong Studio", card_btn.cget("text"))

        # vctk_p231's ProfileSummary from the last list_profiles() still says is_active=True; it is not loaded.
        self.assertTrue(self.mock_profiles[0].is_active)
        panel.profile_combo.current(0)
        panel._on_combo_changed(None)
        self.assertEqual(panel.get_selected_profile().profile_id, "vctk_p231")
        self.assertNotEqual(panel.pill_status.cget("text"), "● Đang dùng")
        self.assertEqual(pc.action.kind, "load")
        self.assertEqual(str(pc.btn_action.cget("state")), "normal")

    def test_declining_to_stop_a_running_session_keeps_the_stream(self) -> None:
        self._settle_initial_load()
        self.mock_service.get_status.return_value = self._status(EngineState.RUNNING)
        self.app._sync_status(self.mock_service.get_status())
        with mock.patch("tkinter.messagebox.askyesno", return_value=False) as ask, \
                mock.patch.object(self.app, "_run_async") as run_async:
            self.app.panel_library._on_card_load_clicked("voice_a_preview")
            ask.assert_not_called()  # selecting alone never interrupts the session
            pc = self.app.panel_controls
            self.assertEqual(pc.action.kind, "stop")
            self.assertEqual(pc.btn_switch.winfo_manager(), "pack")
            self.assertIn("Speaker A (Northern)", pc.btn_switch.cget("text"))
            pc.btn_switch.invoke()
            ask.assert_called_once()
        run_async.assert_not_called()
        self.mock_service.stop.assert_not_called()
        self.assertEqual(self.app.panel_profiles._active_profile_id, "vctk_p231")

    def test_studio_is_one_column_hero_then_action_then_collapsed_sections(self) -> None:
        slaves = self.app.panel_controls.master.pack_slaves()
        self.assertEqual(slaves, [self.app.panel_profiles, self.app.panel_controls, self.app.section_tuning,
                                  self.app.section_noise])
        self.assertIs(self.app.panel_tuning.master, self.app.section_tuning.body)
        self.assertIs(self.app.panel_noise.master, self.app.section_noise.body)
        self.assertFalse(self.app.section_tuning.expanded)
        self.assertFalse(self.app.section_noise.expanded)
        self.assertEqual(self.app.section_noise.lbl_summary.cget("text"), "")  # until the backend reports
        self.app._apply_noise_status(self.mock_service.get_noise_suppression())
        self.assertEqual(self.app.section_noise.lbl_summary.cget("text"), "Đang tắt")

    def test_action_button_calls_the_existing_start_and_stop_handlers(self) -> None:
        self._settle_initial_load()
        self._run_async_inline()
        self.app._sync_status(self._status(EngineState.READY))
        self.app.panel_controls.btn_action.invoke()
        self.mock_service.start.assert_called_once()
        self.mock_service.stop.assert_not_called()

        self.app._sync_status(self._status(EngineState.RUNNING))
        self.app.panel_controls.btn_action.invoke()
        self.mock_service.stop.assert_called_once()
        self.assertEqual(self.mock_service.start.call_count, 1)

        for state in (EngineState.IDLE, EngineState.LOADING, EngineState.STARTING, EngineState.STOPPING):
            self.app._sync_status(self._status(state))
            self.assertEqual(str(self.app.panel_controls.btn_action.cget("state")), "disabled")

    def test_engine_state_is_shown_once_beside_the_action(self) -> None:
        from voice_changer.gui.theme import get_engine_state_visual

        self.assertFalse(hasattr(self.app.sidebar, "lbl_engine_state"))  # sidebar is navigation only
        self.assertFalse(hasattr(self.app, "lbl_header_engine_pill"))  # no duplicate in the header
        for state in EngineState:
            self.app._sync_status(self._status(state))
            expected = get_engine_state_visual(state)
            controls = self.app.panel_controls.pill_state
            self.assertEqual((controls.cget("text"), controls.cget("bg"), controls.cget("fg")), expected, state)

    def test_error_details_dialog_names_the_actual_log_file(self) -> None:
        err = ErrorInfo(category="audio_device", code="stream_failure", message="Micro bị ngắt",
                        hint="", details="PortAudio error", timestamp="2026-10-23T10:00:00")
        self.app._sync_status(self._status(EngineState.ERROR, last_error=err, health_problems=("lost",)))
        self.assertNotIn("'Dừng'", self.app.panel_controls.lbl_error_hint.cget("text"))  # ERROR offers "Thử lại"
        with mock.patch("voice_changer.gui.views.controls_panel.messagebox.showerror") as mock_err:
            self.app.panel_controls._show_error_dialog()
        self.assertIn("test.log", mock_err.call_args.kwargs["detail"])


class TestSavedDevicePreferences(unittest.TestCase):
    """Loading or refreshing the device list must never replace the user's saved devices (review finding M5)."""

    REALTEK = AudioDevice(1, "Realtek Mic", "MME", 1, 0, 44100.0)  # system default input
    USB = AudioDevice(2, "USB Mic", "MME", 1, 0, 44100.0)
    SPEAKERS = AudioDevice(4, "Speakers", "MME", 0, 2, 44100.0)  # system default output
    CABLE = AudioDevice(7, "CABLE Input (VB-Audio Virtual C", "MME", 0, 2, 44100.0)

    def setUp(self) -> None:
        settings_dir = tempfile.TemporaryDirectory()
        self.addCleanup(settings_dir.cleanup)
        env = mock.patch.dict(os.environ, {"LOCALAPPDATA": settings_dir.name})
        env.start()
        self.addCleanup(env.stop)
        self.settings_path = _get_settings_path()
        self.assertTrue(str(self.settings_path).startswith(settings_dir.name))
        GuiSettings(input_device_name="USB Mic", input_device_host_api="MME",
                    output_device_name=self.CABLE.name, output_device_host_api="MME").save()
        self.saved_bytes = self.settings_path.read_bytes()

        self.service = mock.MagicMock(spec=VoiceChangerService)
        self.service.list_profiles.return_value = []
        self.service.resolve_profile_id.return_value = None
        self.service.get_status.return_value = self._status(EngineState.IDLE)
        self.service.get_metrics.return_value = None
        self.service.get_noise_suppression.return_value = NoiseSuppressionStatus(
            enabled=False, mode="medium", available_modes=("light", "medium", "strong"), max_attenuation_db=-12.0,
            latency_ms=20.0, active=False, failed=False, last_error="")
        inline = mock.patch.object(VoiceChangerApp, "_run_async", lambda app, func, *args: func(*args))
        inline.start()
        self.addCleanup(inline.stop)
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)

    @staticmethod
    def _status(state: EngineState) -> EngineStatus:
        return EngineStatus(state=state, profile_id=None, profile_status=None, model_loaded=False, config=None,
                            audio=None, session_id=0, health_problems=(), last_error=None)

    def _inventory(self, inputs, outputs) -> DeviceInventory:
        return DeviceInventory(inputs=tuple(inputs), outputs=tuple(outputs),
                               default_input_id=self.REALTEK.device_id, default_output_id=self.SPEAKERS.device_id)

    def _start_app(self, inventory: DeviceInventory) -> VoiceChangerApp:
        self.service.list_devices.return_value = inventory
        with mock.patch("voice_changer.gui.app.configure_logging", return_value=Path("test.log")):
            app = VoiceChangerApp(self.root, service=self.service)
        app._process_queue()
        return app

    def _ids(self, panel) -> tuple:
        inp, out = panel.get_selected_devices()
        return (inp.device_id if inp else None, out.device_id if out else None)

    def _saved(self) -> tuple:
        s = GuiSettings.load()
        return (s.input_device_name, s.output_device_name)

    def _started_with(self, app: VoiceChangerApp) -> tuple:
        app._handle_start_requested()
        sent = self.service.start.call_args[0][0]
        return (sent.input_device_id, sent.output_device_id)

    def _studio_route(self, app: VoiceChangerApp) -> str:
        return app.panel_controls.lbl_route.cget("text") + "\n" + app.panel_controls.lbl_route_hint.cget("text")

    def test_startup_with_saved_devices_available_selects_them_without_saving(self) -> None:
        app = self._start_app(self._inventory([self.REALTEK, self.USB], [self.SPEAKERS, self.CABLE]))
        self.assertEqual(self._ids(app.panel_audio_settings), (2, 7))
        self.assertNotIn("không khả dụng", app.panel_audio_settings.lbl_in_specs.cget("text"))
        self.assertIn("USB Mic", self._studio_route(app))
        self.assertNotIn("không khả dụng", self._studio_route(app))
        self.assertEqual(self.settings_path.read_bytes(), self.saved_bytes)

    def test_startup_with_saved_mic_missing_uses_default_for_session_and_keeps_preference(self) -> None:
        app = self._start_app(self._inventory([self.REALTEK], [self.SPEAKERS, self.CABLE]))
        self.assertEqual(self.settings_path.read_bytes(), self.saved_bytes)
        self.assertEqual(self._ids(app.panel_audio_settings), (1, 7))
        self.assertIn("'USB Mic' không khả dụng", app.panel_audio_settings.lbl_in_specs.cget("text"))
        self.assertIn("“USB Mic” không khả dụng", self._studio_route(app))
        self.assertIn("Realtek Mic", app.panel_controls.lbl_route.cget("text"))
        self.assertNotIn("không khả dụng", app.panel_audio_settings.lbl_out_specs.cget("text"))
        # The fallback is the runtime device; the saved preference is untouched.
        self.assertEqual(self._started_with(app), (1, 7))
        self.assertEqual(self._saved(), ("USB Mic", self.CABLE.name))

    def test_startup_with_saved_output_missing_uses_default_for_session_and_keeps_preference(self) -> None:
        app = self._start_app(self._inventory([self.REALTEK, self.USB], [self.SPEAKERS]))
        self.assertEqual(self.settings_path.read_bytes(), self.saved_bytes)
        self.assertEqual(self._ids(app.panel_audio_settings), (2, 4))
        self.assertIn(f"'{self.CABLE.name}' không khả dụng", app.panel_audio_settings.lbl_out_specs.cget("text"))
        self.assertIn(f"“{self.CABLE.name}” không khả dụng", self._studio_route(app))
        self.assertEqual(self._started_with(app), (2, 4))

    def test_changing_one_side_does_not_save_the_other_sides_fallback(self) -> None:
        app = self._start_app(self._inventory([self.REALTEK], [self.SPEAKERS, self.CABLE]))  # USB Mic missing
        app.panel_audio_settings.output_combo.current(0)  # user picks Speakers
        app.panel_audio_settings._on_output_selected(None)
        self.assertEqual(self._saved(), ("USB Mic", "Speakers"))
        self.assertEqual(self._ids(app.panel_audio_settings), (1, 4))
        # The still-missing mic keeps its notice, in Audio & Connections and in the Studio route summary.
        self.assertIn("'USB Mic' không khả dụng", app.panel_audio_settings.lbl_in_specs.cget("text"))
        self.assertIn("“USB Mic” không khả dụng", self._studio_route(app))
        self.assertIn("Speakers", app.panel_controls.lbl_route.cget("text"))

    def test_explicit_selection_in_audio_settings_is_saved_and_used_for_start(self) -> None:
        app = self._start_app(self._inventory([self.REALTEK, self.USB], [self.SPEAKERS, self.CABLE]))
        app.panel_audio_settings.input_combo.current(0)  # Realtek Mic
        app.panel_audio_settings._on_input_selected(None)
        self.assertEqual(self._saved(), ("Realtek Mic", self.CABLE.name))
        self.assertEqual(self._ids(app.panel_audio_settings), (1, 7))
        self.assertIn("Realtek Mic", app.panel_controls.lbl_route.cget("text"))
        self.assertEqual(self._started_with(app), (1, 7))

    def test_explicitly_choosing_the_fallback_device_adopts_it(self) -> None:
        app = self._start_app(self._inventory([self.REALTEK], [self.SPEAKERS, self.CABLE]))  # USB Mic missing
        app.panel_audio_settings.input_combo.current(0)  # the user confirms the fallback that is already shown
        app.panel_audio_settings._on_input_selected(None)
        self.assertEqual(self._saved(), ("Realtek Mic", self.CABLE.name))
        self.assertNotIn("không khả dụng", app.panel_audio_settings.lbl_in_specs.cget("text"))
        self.assertNotIn("không khả dụng", self._studio_route(app))

    def test_refresh_does_not_overwrite_preferences(self) -> None:
        app = self._start_app(self._inventory([self.REALTEK], [self.SPEAKERS]))  # both saved devices missing
        app._handle_refresh_devices()
        app._process_queue()
        self.assertEqual(self.settings_path.read_bytes(), self.saved_bytes)
        self.assertEqual(self._ids(app.panel_audio_settings), (1, 4))

    def test_reconnecting_the_saved_device_restores_it_on_refresh(self) -> None:
        app = self._start_app(self._inventory([self.REALTEK], [self.SPEAKERS]))
        self.assertEqual(self._ids(app.panel_audio_settings), (1, 4))
        # USB Mic and the cable come back (PortAudio ids may change after replugging).
        usb_again = AudioDevice(9, "USB Mic", "MME", 1, 0, 44100.0)
        cable_again = AudioDevice(11, self.CABLE.name, "MME", 0, 2, 44100.0)
        self.service.list_devices.return_value = self._inventory([self.REALTEK, usb_again],
                                                                 [self.SPEAKERS, cable_again])
        app._handle_refresh_devices()
        app._process_queue()
        self.assertEqual(self._ids(app.panel_audio_settings), (9, 11))
        self.assertNotIn("không khả dụng", app.panel_audio_settings.lbl_in_specs.cget("text"))
        self.assertNotIn("không khả dụng", app.panel_audio_settings.lbl_out_specs.cget("text"))
        self.assertNotIn("không khả dụng", self._studio_route(app))
        self.assertEqual(self._started_with(app), (9, 11))
        self.assertEqual(self.settings_path.read_bytes(), self.saved_bytes)

    def test_device_controls_stay_locked_while_streaming(self) -> None:
        app = self._start_app(self._inventory([self.REALTEK], [self.SPEAKERS, self.CABLE]))
        panel = app.panel_audio_settings
        for state in (EngineState.STARTING, EngineState.RUNNING, EngineState.STOPPING):
            app._sync_status(self._status(state))
            self.assertEqual(str(panel.input_combo.cget("state")), "disabled")
            self.assertEqual(str(panel.output_combo.cget("state")), "disabled")
            self.assertEqual(str(panel.btn_refresh.cget("state")), "disabled")
        app._sync_status(self._status(EngineState.READY))
        self.assertEqual(str(panel.input_combo.cget("state")), "readonly")
        self.assertEqual(str(panel.btn_refresh.cget("state")), "normal")
        self.assertEqual(self.settings_path.read_bytes(), self.saved_bytes)

    def test_status_pill_harmony(self) -> None:
        """Verify that the Studio status pill wording and colors track get_engine_state_visual exactly."""
        from voice_changer.gui.theme import get_engine_state_visual

        app = self._start_app(self._inventory([self.REALTEK], [self.SPEAKERS, self.CABLE]))
        for state in (
            EngineState.IDLE,
            EngineState.LOADING,
            EngineState.READY,
            EngineState.STARTING,
            EngineState.RUNNING,
            EngineState.STOPPING,
            EngineState.ERROR,
        ):
            app._sync_status(self._status(state))
            exp_text, exp_bg, exp_fg = get_engine_state_visual(state)
            pill = app.panel_controls.pill_state
            self.assertEqual(pill.cget("text"), exp_text)
            self.assertEqual(pill.cget("bg"), exp_bg)
            self.assertEqual(pill.cget("fg"), exp_fg)


if __name__ == "__main__":
    unittest.main()
