"""Unit tests verifying project scaffold, package metadata, and interfaces."""

import sys
from pathlib import Path

# Ensure src/ is on sys.path for test execution without package installation
src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import unittest
from voice_changer import __version__
from voice_changer.conversion.base import ConversionConfig, VoiceConverter
from voice_changer.conversion.rvc.adapter import RVCAdapter
from voice_changer.diagnostics.latency import LatencyMonitor
from voice_changer.tuning.presets import get_default_presets


class TestScaffold(unittest.TestCase):
    def test_package_version(self):
        self.assertEqual(__version__, "0.1.0")

    def test_conversion_config_defaults(self):
        config = ConversionConfig()
        self.assertEqual(config.pitch_shift, 12.0)
        self.assertEqual(config.f0_method, "rmvpe")
        self.assertEqual(config.index_rate, 0.75)

    def test_rvc_adapter_implements_voice_converter_protocol(self):
        adapter = RVCAdapter()
        self.assertIsInstance(adapter, VoiceConverter)
        self.assertFalse(adapter.is_loaded)

    def test_default_presets_structure(self):
        presets = get_default_presets()
        self.assertIn("natural_feminine", presets)
        self.assertEqual(presets["natural_feminine"].config.pitch_shift, 12.0)

    def test_latency_monitor_initial_state(self):
        monitor = LatencyMonitor()
        stats = monitor.get_stats()
        self.assertEqual(stats.sample_count, 0)
        self.assertEqual(stats.underrun_count, 0)

        monitor.record_inference(0.025)  # 25ms
        monitor.record_inference(0.035)  # 35ms
        stats = monitor.get_stats()
        self.assertEqual(stats.sample_count, 2)
        self.assertAlmostEqual(stats.mean_ms, 30.0, places=3)


if __name__ == "__main__":
    unittest.main()

