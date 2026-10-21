"""Unit tests for GUI settings persistence and device matching."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from voice_changer.audio.devices import AudioDevice
from voice_changer.gui.settings import GuiSettings
from voice_changer.service import DeviceInventory



class TestGuiSettings(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings_path = Path(self.temp_dir.name) / "test_gui_settings.json"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @mock.patch("voice_changer.gui.settings._get_settings_path")
    def test_save_and_load(self, mock_get_path: mock.MagicMock) -> None:
        mock_get_path.return_value = self.settings_path

        settings = GuiSettings(
            profile_id="voice_b_preview",
            input_device_name="Microphone (Realtek)",
            input_device_host_api="MME",
            output_device_name="CABLE Input",
            output_device_host_api="MME",
            pitch_shift=14.0,
            index_rate=0.25,
            filter_radius=2,
            protect_rate=0.4,
        )
        settings.save()

        loaded = GuiSettings.load()
        self.assertEqual(loaded.profile_id, "voice_b_preview")
        self.assertEqual(loaded.input_device_name, "Microphone (Realtek)")
        self.assertEqual(loaded.input_device_host_api, "MME")
        self.assertEqual(loaded.output_device_name, "CABLE Input")
        self.assertEqual(loaded.output_device_host_api, "MME")
        self.assertEqual(loaded.pitch_shift, 14.0)
        self.assertEqual(loaded.index_rate, 0.25)

    def test_resolve_devices_matching(self) -> None:
        dev_in1 = AudioDevice(0, "Mic USB", "WASAPI", 1, 0, 48000.0)
        dev_in2 = AudioDevice(1, "Mic USB", "MME", 1, 0, 40000.0)
        dev_out1 = AudioDevice(2, "Speakers", "MME", 0, 2, 44100.0)
        dev_out2 = AudioDevice(3, "CABLE Input", "MME", 0, 2, 40000.0)

        inv = DeviceInventory(
            inputs=(dev_in1, dev_in2),
            outputs=(dev_out1, dev_out2),
            default_input_id=0,
            default_output_id=2,
        )

        # Matched pair (name, host_api)
        settings = GuiSettings(
            input_device_name="Mic USB",
            input_device_host_api="MME",
            output_device_name="CABLE Input",
            output_device_host_api="MME",
        )
        self.assertEqual(settings.resolve_input_device_id(inv), 1)
        self.assertEqual(settings.resolve_output_device_id(inv), 3)

        # Fallback when device is not found
        settings_unfound = GuiSettings(
            input_device_name="Nonexistent Mic",
            input_device_host_api="DirectSound",
        )
        self.assertEqual(settings_unfound.resolve_input_device_id(inv), 0)
        self.assertEqual(settings_unfound.resolve_output_device_id(inv), 2)

    def test_update_helpers(self) -> None:
        settings = GuiSettings()
        dev = AudioDevice(5, "Headphones", "DirectSound", 0, 2, 44100.0)
        settings.update_output(dev)

        self.assertEqual(settings.output_device_name, "Headphones")
        self.assertEqual(settings.output_device_host_api, "DirectSound")

        settings.update_output(None)
        self.assertIsNone(settings.output_device_name)
        self.assertIsNone(settings.output_device_host_api)


if __name__ == "__main__":
    unittest.main()
