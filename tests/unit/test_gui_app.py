"""Unit tests for VoiceChangerApp and view panels (headless execution)."""

import sys
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
from voice_changer.service import (
    AudioSettings,
    ConfigurationError,
    DeviceInventory,
    EngineState,
    EngineStatus,
    ErrorInfo,
    NoiseSuppressionStatus,
    ProfileSummary,
    VoiceChangerService,
)


class TestVoiceChangerApp(unittest.TestCase):
    def setUp(self) -> None:
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

    def test_app_initialization_components(self) -> None:
        """Verify that all tabs and panels are cleanly constructed."""
        self.assertIsNotNone(self.app.panel_profiles)
        self.assertIsNotNone(self.app.panel_devices)
        self.assertIsNotNone(self.app.panel_noise)
        self.assertIsNotNone(self.app.panel_tuning)
        self.assertIsNotNone(self.app.panel_controls)
        self.assertIsNotNone(self.app.panel_guide)
        self.assertEqual(self.app.notebook.index("end"), 2)

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
        self.assertIn("CHƯA NẠP MODEL", str(self.app.panel_controls.pill_state.cget("text")))
        self.assertEqual(str(self.app.panel_controls.btn_action.cget("state")), "disabled")

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
        self.assertIn("BẮT ĐẦU", str(self.app.panel_controls.btn_action.cget("text")))

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
        self.assertIn("ĐANG CHẠY", str(self.app.panel_controls.pill_state.cget("text")))
        self.assertIn("DỪNG", str(self.app.panel_controls.btn_action.cget("text")))

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
        self.assertIn("LỖI HỆ THỐNG", str(self.app.panel_controls.pill_state.cget("text")))
        self.assertIn("Micro bị ngắt kết nối", self.app.panel_controls.lbl_error_msg.cget("text"))

    def test_window_resizing_and_layout(self) -> None:
        """Verify window layout responds to common dimensions without errors."""
        for width, height in [(960, 680), (1080, 760), (1280, 800)]:
            self.root.geometry(f"{width}x{height}")
            self.root.update_idletasks()
            # Verify panels remain mapped and populated
            self.assertGreater(self.app.panel_profiles.winfo_width(), 0)
            self.assertGreater(self.app.panel_devices.winfo_width(), 0)
            self.assertGreater(self.app.panel_noise.winfo_width(), 0)
            self.assertGreater(self.app.panel_controls.winfo_width(), 0)

    def test_metrics_dashboard_update(self) -> None:
        """Verify metric cards update with formatted values."""
        metrics_sample = {
            "rtf": 0.452,
            "avg_inference_ms": 226.0,
            "avg_software_e2e_ms": 500.0,
            "current_buffer_ms": 150.0,
            "chunks_converted": 45,
            "chunks_dropped": 0,
            "steady_underflows": 0,
        }
        self.app.panel_controls.update_metrics(metrics_sample)

        self.assertEqual(self.app.panel_controls.val_rtf.cget("text"), "0.452")
        self.assertEqual(self.app.panel_controls.val_gpu.cget("text"), "226 ms")
        self.assertEqual(self.app.panel_controls.val_e2e.cget("text"), "~650 ms")
        self.assertEqual(self.app.panel_controls.val_buf.cget("text"), "150 ms")
        self.assertEqual(self.app.panel_controls.val_chunks.cget("text"), "45 / 0")
        self.assertEqual(self.app.panel_controls.val_und.cget("text"), "0")

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


if __name__ == "__main__":
    unittest.main()
