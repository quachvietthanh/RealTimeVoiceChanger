"""Unit tests for AudioDeviceManager with a mocked sounddevice (no audio hardware needed)."""

import sys
import unittest
from pathlib import Path
from unittest import mock

src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from voice_changer.audio import devices as dev_mod
from voice_changer.audio.devices import AudioDeviceManager

DEVICES = [
    {"name": "Microphone (USB)", "hostapi": 0, "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 44100.0},
    {"name": "Speakers (USB)", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 44100.0},
    {"name": "CABLE Input (VB-Audio Virtual C", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 8, "default_samplerate": 44100.0},
    {"name": "CABLE Output (VB-Audio Virtual ", "hostapi": 0, "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 44100.0},
]


class DeviceManagerTests(unittest.TestCase):
    def setUp(self):
        sd = mock.MagicMock()
        sd.query_hostapis.return_value = [{"name": "MME"}]
        sd.query_devices.side_effect = lambda *a, **k: list(DEVICES)
        self.sd = sd
        self.patch = mock.patch.object(dev_mod, "sd", sd)
        self.patch.start()
        self.dm = AudioDeviceManager()

    def tearDown(self):
        self.patch.stop()

    def test_enumeration_and_virtual_cable_flag(self):
        outs = self.dm.list_output_devices()
        self.assertEqual([d.device_id for d in outs], [1, 2])
        self.assertEqual([d.is_virtual_cable for d in outs], [False, True])
        self.assertTrue(self.dm.list_input_devices()[1].is_virtual_cable)

    def test_get_and_find_device(self):
        self.assertEqual(self.dm.get_device(2).name, "CABLE Input (VB-Audio Virtual C")
        self.assertIsNone(self.dm.get_device(42))
        self.assertEqual(self.dm.find_device("Speakers (USB)", "MME", is_input=False).device_id, 1)
        self.assertIsNone(self.dm.find_device("Speakers (USB)", "WASAPI", is_input=False))

    def test_check_device(self):
        self.assertIsNone(self.dm.check_device(1, is_input=False, sample_rate=40000, channels=2))
        self.assertIn("does not exist", self.dm.check_device(42, is_input=False, sample_rate=40000, channels=2))
        self.assertIn("no input channels", self.dm.check_device(1, is_input=True, sample_rate=40000, channels=1))
        self.sd.check_output_settings.side_effect = RuntimeError("Invalid sample rate")
        self.assertIn("rejects 40000 Hz", self.dm.check_device(1, is_input=False, sample_rate=40000, channels=2))


if __name__ == "__main__":
    unittest.main()
