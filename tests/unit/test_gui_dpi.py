"""Unit tests for GUI DPI awareness and logical-pixel scaling (headless; no real DPI change)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tkinter as tk

src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from voice_changer.audio.devices import AudioDevice
from voice_changer.conversion.base import ConversionConfig
from voice_changer.gui import app as app_module
from voice_changer.gui import dpi
from voice_changer.gui.app import VoiceChangerApp
from voice_changer.gui.presentation import AvatarSpec
from voice_changer.gui.views.widgets import create_avatar
from voice_changer.service import (
    DeviceInventory,
    EngineState,
    EngineStatus,
    NoiseSuppressionStatus,
    ProfileSummary,
    VoiceChangerService,
)


def _reset_dpi_state() -> None:
    dpi._ui_scale = 1.0
    dpi._awareness_configured = False


class TestDpiHelpers(unittest.TestCase):
    def setUp(self) -> None:
        _reset_dpi_state()
        self.addCleanup(_reset_dpi_state)

    def test_scale_is_rounded_to_windows_scaling_steps(self) -> None:
        root = mock.MagicMock()
        for measured_dpi, expected in [(96.09, 1.0), (120.12, 1.25), (144.0, 1.5), (168.0, 1.75), (72.0, 1.0)]:
            root.winfo_fpixels.return_value = measured_dpi
            self.assertEqual(dpi.init_ui_scale(root), expected)
            self.assertEqual(dpi.ui_scale(), expected)

    def test_px_converts_96_dpi_sizes(self) -> None:
        self.assertEqual(dpi.px(240), 240)
        dpi._ui_scale = 1.25
        self.assertEqual(dpi.px(240), 300)
        self.assertEqual(dpi.px(42), 52)
        dpi._ui_scale = 1.5
        self.assertEqual(dpi.px(980), 1470)

    def test_fit_window_scales_and_stays_on_screen(self) -> None:
        root = mock.MagicMock()
        root.winfo_screenwidth.return_value = 1920
        root.winfo_screenheight.return_value = 1080
        dpi._ui_scale = 1.25
        dpi.fit_window(root, 1140, 780, min_width=980, min_height=680)
        root.geometry.assert_called_with("1425x975")
        root.minsize.assert_called_with(1225, 850)

        # 150% on a 1080p panel: the designed height no longer fits, so both sizes are capped to the screen.
        dpi._ui_scale = 1.5
        dpi.fit_window(root, 1140, 780, min_width=980, min_height=680)
        root.geometry.assert_called_with("1710x960")
        root.minsize.assert_called_with(1470, 960)

    @unittest.skipUnless(sys.platform == "win32", "Windows DPI API")
    def test_awareness_is_set_once_with_system_awareness(self) -> None:
        windll = mock.MagicMock()
        windll.shcore.SetProcessDpiAwareness.return_value = 0
        with mock.patch.object(dpi.ctypes, "windll", windll, create=True):
            self.assertTrue(dpi.enable_dpi_awareness())
            self.assertTrue(dpi.enable_dpi_awareness())
        windll.shcore.SetProcessDpiAwareness.assert_called_once_with(1)

    @unittest.skipUnless(sys.platform == "win32", "Windows DPI API")
    def test_awareness_already_set_counts_as_aware(self) -> None:
        windll = mock.MagicMock()
        windll.shcore.SetProcessDpiAwareness.return_value = dpi._E_ACCESSDENIED
        with mock.patch.object(dpi.ctypes, "windll", windll, create=True):
            self.assertTrue(dpi.enable_dpi_awareness())

    @unittest.skipUnless(sys.platform == "win32", "Windows DPI API")
    def test_awareness_falls_back_to_user32_before_windows_8_1(self) -> None:
        windll = mock.MagicMock()
        windll.shcore.SetProcessDpiAwareness.side_effect = AttributeError("no shcore")
        windll.user32.SetProcessDPIAware.return_value = 1
        with mock.patch.object(dpi.ctypes, "windll", windll, create=True):
            self.assertTrue(dpi.enable_dpi_awareness())
        windll.user32.SetProcessDPIAware.assert_called_once_with()

    def test_awareness_is_a_no_op_off_windows(self) -> None:
        with mock.patch.object(dpi.sys, "platform", "linux"):
            self.assertFalse(dpi.enable_dpi_awareness())

    def test_launch_gui_sets_awareness_before_creating_the_root(self) -> None:
        calls: list[str] = []
        root = mock.MagicMock()
        with mock.patch.object(app_module, "enable_dpi_awareness", side_effect=lambda: calls.append("aware")), \
                mock.patch.object(app_module.tk, "Tk", side_effect=lambda: calls.append("tk") or root), \
                mock.patch.object(app_module, "VoiceChangerApp") as app_cls:
            app_module.launch_gui(service=mock.sentinel.service)
        self.assertEqual(calls, ["aware", "tk"])
        app_cls.assert_called_once_with(root, service=mock.sentinel.service)
        root.mainloop.assert_called_once_with()


class TestScaledLayout(unittest.TestCase):
    """Builds the app with Tk reporting 144 DPI (150%); the test process itself is not made DPI-aware."""

    def setUp(self) -> None:
        _reset_dpi_state()
        self.addCleanup(_reset_dpi_state)
        settings_dir = tempfile.TemporaryDirectory()
        self.addCleanup(settings_dir.cleanup)
        env = mock.patch.dict(os.environ, {"LOCALAPPDATA": settings_dir.name})
        env.start()
        self.addCleanup(env.stop)

        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.root.tk.call("tk", "scaling", 144 / 72)

        defaults = ConversionConfig(pitch_shift=12.0, index_rate=0.75, filter_radius=3, protect_rate=0.33)
        service = mock.MagicMock(spec=VoiceChangerService)
        service.list_profiles.return_value = [
            ProfileSummary(profile_id="vctk_p231", display_name="VCTK Female p231", description="English Prototype",
                           status="prototype", language="en", defaults=defaults, files_present=True,
                           is_active=False, notes=""),
        ]
        service.resolve_profile_id.return_value = None
        service.list_devices.return_value = DeviceInventory(
            inputs=(AudioDevice(0, "Test Mic", "MME", 1, 0, 40000.0),),
            outputs=(AudioDevice(1, "CABLE Input", "MME", 0, 2, 40000.0),),
            default_input_id=0, default_output_id=1)
        service.get_status.return_value = EngineStatus(
            state=EngineState.IDLE, profile_id=None, profile_status=None, model_loaded=False, config=defaults,
            audio=None, session_id=0, health_problems=(), last_error=None)
        service.get_config.return_value = defaults
        service.get_metrics.return_value = None
        service.get_noise_suppression.return_value = NoiseSuppressionStatus(
            enabled=False, mode="medium", available_modes=("light", "medium", "strong"), max_attenuation_db=-12.0,
            latency_ms=20.0, active=False, failed=False, last_error="")
        with mock.patch("voice_changer.gui.app.configure_logging", return_value=Path("test.log")):
            self.app = VoiceChangerApp(self.root, service=service)

    def test_fixed_sizes_follow_the_detected_dpi(self) -> None:
        self.assertEqual(dpi.ui_scale(), 1.5)
        self.assertEqual(int(self.app.sidebar.cget("width")), 360)
        self.assertEqual(int(self.app.panel_profiles.avatar_canvas.cget("width")), 108)  # hero avatar: px(72)
        self.assertIsNotNone(self.app.sidebar.logo_image)
        # The sidebar logo is the generated PNG closest to px(32) = 48 screen pixels.
        self.assertEqual(int(self.app.sidebar.logo_image.width()), 48)
        min_w, min_h = self.root.minsize()
        self.assertEqual(min_w, min(1470, self.root.winfo_screenwidth() - 24))
        self.assertEqual(min_h, min(1020, self.root.winfo_screenheight() - 120))

    def test_avatar_glyph_is_not_scaled_twice(self) -> None:
        canvas = create_avatar(self.root, AvatarSpec("VB", "#7c3aed"), size=40)
        self.assertEqual(int(canvas.cget("width")), 60)
        text_item = next(i for i in canvas.find_all() if canvas.type(i) == "text")
        # Font sizes are points, which Tk already scales with the DPI: 0.35 * 40, not 0.35 * 60.
        self.assertEqual(canvas.itemcget(text_item, "font").split()[-2], "14")


if __name__ == "__main__":
    unittest.main()
