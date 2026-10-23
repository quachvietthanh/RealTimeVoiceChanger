"""Contract tests for VoiceChangerService, the GUI-facing backend API (no GPU, devices or real models)."""

import hashlib
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import numpy as np

from voice_changer.audio.devices import AudioDevice
from voice_changer.audio.noise_suppression import NoiseSuppressionSettings
from voice_changer.audio.pipeline import RealTimeVoiceChangerPipeline
from voice_changer.conversion.rvc.assets import CheckpointInfo, IndexInfo
from voice_changer.profiles.registry import ProfileRegistry
from voice_changer.service import (
    AudioDeviceError,
    AudioSettings,
    ConfigurationError,
    EngineState,
    InvalidStateError,
    ProfileError,
    VoiceChangerService,
)

CKPT = CheckpointInfo(version="v2", f0=1, sample_rate=40000, vocoder="HiFi-GAN", embedder_dim=768, speaker_count=1)
IDX = IndexInfo(dim=768, ntotal=100, is_trained=True)
CHUNK = 20000


class FakeConverter:
    loaded_now = 0  # models currently loaded across all instances (VRAM proxy)
    max_loaded = 0

    def __init__(self, fail_load=False):
        self.fail_load = fail_load
        self.model = None

    def load_model(self, model_path, index_path=None, *, model_sha256=None, index_sha256=None):
        if self.fail_load:
            raise RuntimeError("CUDA out of memory")
        self.model = (model_path, index_path)
        self.pins = (model_sha256, index_sha256)
        FakeConverter.loaded_now += 1
        FakeConverter.max_loaded = max(FakeConverter.max_loaded, FakeConverter.loaded_now)

    def unload(self):
        if self.model is not None:
            FakeConverter.loaded_now -= 1
        self.model = None

    def convert_chunk(self, audio_data, config=None, input_sr=40000, target_sr=None):
        if self.model is None:
            raise RuntimeError("model not loaded")
        return np.asarray(audio_data)


class FakeDevices:
    def __init__(self):
        self.inputs = [AudioDevice(1, "Mic", "MME", 1, 0, 44100.0)]
        self.outputs = [AudioDevice(4, "Speakers", "MME", 0, 2, 44100.0),
                        AudioDevice(7, "CABLE Input (VB-Audio Virtual C", "MME", 0, 2, 44100.0)]

    def list_input_devices(self):
        return list(self.inputs)

    def list_output_devices(self):
        return list(self.outputs)

    def get_default_device(self, is_input=True):
        return self.inputs[0] if is_input else self.outputs[0]

    def check_device(self, device_id, is_input, sample_rate, channels):
        if device_id is None:
            return None
        pool = self.inputs if is_input else self.outputs
        if not any(d.device_id == device_id for d in pool):
            return f"{'input' if is_input else 'output'} device {device_id} does not exist"
        return None


class FakeStream:
    def __init__(self):
        self.active = False
        self.callback = None
        self.overflow_count = self.status_flag_count = 0
        self.underflow_count = self.startup_underflow_count = self.steady_underflow_count = 0
        self.shutdown_underflow_count = 0
        self.buffered_seconds = 0.0
        self.written = []

    @property
    def is_active(self):
        return self.active

    def start(self, callback=None):
        self.callback = callback
        self.active = True

    def stop(self):
        self.active = False

    def prepare_shutdown(self):
        pass

    def write_chunk(self, arr):
        self.written.append(arr)


class ServiceTestBase(unittest.TestCase):
    def setUp(self):
        FakeConverter.loaded_now = FakeConverter.max_loaded = 0
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        profiles = []
        for pid in ("a", "b"):
            (d / f"{pid}.pth").write_bytes(pid.encode())
            (d / f"{pid}.index").write_bytes(pid.encode() * 2)
            profiles.append({
                "id": pid, "display_name": pid.upper(), "status": "preview", "model_path": f"{pid}.pth",
                "model_sha256": hashlib.sha256(pid.encode()).hexdigest(), "index_path": f"{pid}.index",
                "index_sha256": hashlib.sha256(pid.encode() * 2).hexdigest(),
                "defaults": {"pitch_shift": 12.0 if pid == "a" else 14.0, "index_rate": 0.5 if pid == "a" else 0.25},
            })
        profiles.append({"id": "jett", "display_name": "Jett", "status": "planned", "data_rights": "unverified"})
        (d / "p.json").write_text(json.dumps({"schema_version": 1, "default_profile": "a", "profiles": profiles}))
        self.registry = ProfileRegistry.from_file(d / "p.json", base_dir=d)
        self.dir = d
        self.patches = [mock.patch("voice_changer.conversion.rvc.assets.inspect_checkpoint", return_value=CKPT),
                        mock.patch("voice_changer.conversion.rvc.assets.inspect_index", return_value=IDX)]
        for p in self.patches:
            p.start()
        self.streams = []
        self.fail_next_load = False

        def converter_factory():
            fail, self.fail_next_load = self.fail_next_load, False
            return FakeConverter(fail_load=fail)

        def pipeline_factory(**kw):
            inp, out = FakeStream(), FakeStream()
            self.streams.append((inp, out))
            return RealTimeVoiceChangerPipeline(input_stream=inp, output_stream=out, **kw)

        self.converter_factory, self.pipeline_factory = converter_factory, pipeline_factory
        self.svc = VoiceChangerService(self.registry, FakeDevices(), converter_factory, pipeline_factory)
        self.events = []
        self.svc.add_status_listener(lambda s: self.events.append(s.state))

    def tearDown(self):
        self.svc.shutdown()
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()


class LifecycleContractTests(ServiceTestBase):
    def test_initial_state(self):
        st = self.svc.get_status()
        self.assertEqual(st.state, EngineState.IDLE)
        self.assertFalse(st.model_loaded)
        self.assertIsNone(self.svc.get_config())
        self.assertIsNone(self.svc.get_metrics())
        self.assertEqual(self.svc.default_profile_id, "a")

    def test_select_start_stop_transitions(self):
        self.svc.select_profile("a")
        st = self.svc.get_status()
        self.assertEqual((st.state, st.profile_id, st.model_loaded), (EngineState.READY, "a", True))
        self.assertEqual(self.svc.get_config().pitch_shift, 12.0)
        self.svc.start(AudioSettings(input_device_id=1, output_device_id=7))
        self.assertEqual(self.svc.get_status().state, EngineState.RUNNING)
        self.assertEqual(self.svc.get_status().session_id, 1)
        self.svc.stop()
        self.assertEqual(self.svc.get_status().state, EngineState.READY)
        self.assertEqual(self.events, [EngineState.LOADING, EngineState.READY, EngineState.STARTING,
                                       EngineState.RUNNING, EngineState.STOPPING, EngineState.READY])

    def test_only_one_model_loaded_when_switching(self):
        self.svc.select_profile("a")
        self.svc.select_profile("b")
        self.svc.select_profile("a")
        self.assertEqual(FakeConverter.max_loaded, 1)
        self.assertEqual(FakeConverter.loaded_now, 1)
        self.assertEqual(self.svc.get_status().profile_id, "a")
        self.svc.unload_profile()
        self.assertEqual(FakeConverter.loaded_now, 0)
        self.assertEqual(self.svc.get_status().state, EngineState.IDLE)

    def test_switch_uses_the_new_profiles_model_index_and_defaults(self):
        self.svc.select_profile("a")
        self.svc.update_config(pitch_shift=10)
        self.svc.select_profile("b")
        self.assertEqual(self.svc.get_config().pitch_shift, 14.0)  # b's defaults, not a's edits
        self.assertEqual(self.svc.get_config().index_rate, 0.25)
        conv = self.svc._converter
        self.assertTrue(conv.model[0].endswith("b.pth") and conv.model[1].endswith("b.index"))

    def test_converter_receives_the_profiles_pins_to_recheck_on_the_loaded_bytes(self):
        self.svc.select_profile("b")
        b = self.registry.get("b")
        self.assertEqual(self.svc._converter.pins, (b.model_sha256, b.index_sha256))
        self.assertIsNotNone(b.model_sha256)

    def test_switch_while_running_is_refused(self):
        self.svc.select_profile("a")
        self.svc.start()
        with self.assertRaises(InvalidStateError):
            self.svc.select_profile("b")
        self.assertEqual(self.svc.get_status().profile_id, "a")
        self.assertEqual(self.svc.get_status().state, EngineState.RUNNING)

    def test_start_without_profile_is_refused(self):
        with self.assertRaises(InvalidStateError):
            self.svc.start()

    def test_invalid_profile_keeps_current_model(self):
        self.svc.select_profile("a")
        (self.dir / "b.index").write_bytes(b"another speaker's index")
        with self.assertRaises(ProfileError) as ctx:
            self.svc.select_profile("b")
        self.assertEqual(ctx.exception.code, "profile_invalid")
        st = self.svc.get_status()
        self.assertEqual((st.state, st.profile_id), (EngineState.READY, "a"))
        self.assertEqual(st.last_error.code, "profile_invalid")

    def test_planned_profile_is_refused_and_keeps_current_model(self):
        self.svc.select_profile("a")
        with self.assertRaises(ProfileError) as ctx:
            self.svc.select_profile("jett")
        self.assertEqual(ctx.exception.code, "profile_not_ready")
        self.assertIn("No model yet", ctx.exception.hint)
        st = self.svc.get_status()
        self.assertEqual((st.state, st.profile_id, st.model_loaded), (EngineState.READY, "a", True))
        self.assertEqual(st.last_error.code, "profile_not_ready")
        self.assertEqual(FakeConverter.loaded_now, 1)

    def test_unknown_profile(self):
        with self.assertRaises(ProfileError) as ctx:
            self.svc.select_profile("zzz")
        self.assertEqual(ctx.exception.code, "unknown_profile")

    def test_failed_load_leaves_idle_with_nothing_loaded(self):
        self.svc.select_profile("a")
        self.fail_next_load = True
        with self.assertRaises(ProfileError) as ctx:
            self.svc.select_profile("b")
        self.assertEqual(ctx.exception.code, "model_load_failed")
        self.assertIn("CUDA out of memory", ctx.exception.details)
        st = self.svc.get_status()
        self.assertEqual((st.state, st.profile_id, st.model_loaded), (EngineState.IDLE, None, False))
        self.assertEqual(FakeConverter.loaded_now, 0)
        self.svc.select_profile("b")  # recoverable
        self.assertEqual(self.svc.get_status().state, EngineState.READY)


class ConfigContractTests(ServiceTestBase):
    def test_update_config_validates_and_applies_live(self):
        self.svc.select_profile("a")
        self.svc.start()
        cfg = self.svc.update_config(pitch_shift=15, index_rate=0.0)
        self.assertEqual((cfg.pitch_shift, cfg.index_rate), (15, 0.0))
        self.assertEqual(self.svc._pipeline.config.pitch_shift, 15)
        with self.assertRaises(ConfigurationError):
            self.svc.update_config(index_rate=1.5)
        with self.assertRaises(ConfigurationError):
            self.svc.update_config(volume=3)
        self.assertEqual(self.svc.get_config().index_rate, 0.0)  # unchanged by the rejected updates
        self.assertEqual(self.svc.reset_config().pitch_shift, 12.0)

    def test_update_config_requires_profile(self):
        with self.assertRaises(InvalidStateError):
            self.svc.update_config(pitch_shift=12)

    def test_get_config_returns_a_copy(self):
        self.svc.select_profile("a")
        cfg = self.svc.get_config()
        cfg.pitch_shift = 99
        self.assertEqual(self.svc.get_config().pitch_shift, 12.0)


class DeviceContractTests(ServiceTestBase):
    def test_list_devices_marks_virtual_cable(self):
        inv = self.svc.list_devices()
        self.assertEqual(inv.default_input_id, 1)
        self.assertEqual([d.is_virtual_cable for d in inv.outputs], [False, True])

    def test_missing_device_is_an_audio_device_error(self):
        self.svc.select_profile("a")
        with self.assertRaises(AudioDeviceError) as ctx:
            self.svc.start(AudioSettings(output_device_id=99))
        self.assertEqual(ctx.exception.code, "audio_device_unavailable")
        self.assertEqual(self.svc.get_status().state, EngineState.READY)

    def test_bad_buffering_is_a_configuration_error(self):
        self.svc.select_profile("a")
        with self.assertRaises(ConfigurationError):
            self.svc.start(AudioSettings(chunk_duration_s=5.0))

    def test_noise_suppression_defaults_off_and_validates(self):
        st = self.svc.get_noise_suppression()
        self.assertEqual((st.enabled, st.mode, st.active, st.failed), (False, "medium", False, False))
        self.assertEqual(st.available_modes, ("light", "medium", "strong"))
        self.assertEqual(st.latency_ms, 20.0)
        with self.assertRaises(ConfigurationError):
            self.svc.set_noise_suppression(mode="extreme")
        with self.assertRaises(ConfigurationError):
            self.svc.set_noise_suppression(enabled="yes")
        self.assertFalse(self.svc.get_noise_suppression().enabled)  # nothing changed

    def test_noise_suppression_applies_live_and_persists_across_sessions_and_profiles(self):
        st = self.svc.set_noise_suppression(enabled=True, mode="light")  # allowed with nothing loaded
        self.assertEqual((st.enabled, st.mode, st.max_attenuation_db), (True, "light", -6.0))
        self.svc.select_profile("a")
        self.svc.start()
        stage = self.svc._pipeline._noise_stage
        self.assertEqual(stage.settings, NoiseSuppressionSettings(True, "light"))
        self.svc.set_noise_suppression(mode="strong")
        self.assertEqual(stage.settings, NoiseSuppressionSettings(True, "strong"))
        self.svc.stop()
        self.svc.select_profile("b")
        self.svc.start()
        self.assertEqual(self.svc._pipeline._noise_stage.settings, NoiseSuppressionSettings(True, "strong"))
        self.svc.stop()

    def test_bad_seam_settings_are_configuration_errors(self):
        self.svc.select_profile("a")
        for bad in (AudioSettings(context_s=0.03), AudioSettings(crossfade_s=0.5),
                    AudioSettings(crossfade_s=0.0, sola_search_s=0.01), AudioSettings(sola_search_s=0.1)):
            with self.subTest(bad=bad), self.assertRaises(ConfigurationError):
                self.svc.start(bad)
        self.assertEqual(self.svc.validate_audio_settings(
            AudioSettings(context_s=0.0, crossfade_s=0.0, sola_search_s=0.0)), [])  # legacy joining allowed

    def test_stream_open_failure_returns_to_ready(self):
        self.svc.select_profile("a")

        def broken_factory(**kw):
            inp = FakeStream()
            inp.start = mock.Mock(side_effect=RuntimeError("Device unavailable [PaErrorCode -9985]"))
            return RealTimeVoiceChangerPipeline(input_stream=inp, output_stream=FakeStream(), **kw)

        self.svc._pipeline_factory = broken_factory
        with self.assertRaises(AudioDeviceError) as ctx:
            self.svc.start()
        self.assertEqual(ctx.exception.code, "stream_open_failed")
        self.assertEqual(self.svc.get_status().state, EngineState.READY)


class HealthAndMetricsTests(ServiceTestBase):
    def test_device_loss_moves_to_error_then_stop_recovers(self):
        self.svc.select_profile("a")
        self.svc.start()
        self.streams[-1][0].active = False  # input device unplugged
        st = self.svc.get_status()
        self.assertEqual(st.state, EngineState.ERROR)
        self.assertEqual(st.last_error.code, "stream_failure")
        self.assertTrue(st.health_problems)
        self.svc.stop()
        self.assertEqual(self.svc.get_status().state, EngineState.READY)

    def test_metrics_are_per_session_and_cleared_on_switch(self):
        self.svc.select_profile("a")
        self.svc.start()
        inp = self.streams[-1][0]
        inp.callback(np.ones((CHUNK, 1), dtype=np.float32))
        deadline = time.monotonic() + 2
        while self.svc.get_metrics()["chunks_converted"] < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        live = self.svc.get_metrics()
        self.assertTrue(live["live"])
        self.assertEqual((live["chunks_converted"], live["profile_id"], live["session_id"]), (1, "a", 1))
        self.assertIn("rtf", live)
        for key in ("estimated_software_latency_ms", "algorithmic_delay_ms", "stitch_resets", "p99_inference_ms",
                    "p99_software_e2e_ms", "max_buffer_occupancy_ms", "input_device_latency_ms"):
            self.assertIn(key, live)
        self.assertEqual(live["algorithmic_delay_ms"], 75.0)  # 55 ms seam + 20 ms noise suppression stage
        with self.assertLogs("voice_changer.service.service", level="INFO") as logs:
            self.svc.stop()
        self.assertTrue(any("session 1 metrics" in line for line in logs.output))
        final = self.svc.get_metrics()
        self.assertFalse(final["live"])
        self.assertEqual(final["chunks_converted"], 1)
        self.svc.select_profile("b")
        self.assertIsNone(self.svc.get_metrics())

    def test_status_is_readable_during_a_slow_load(self):
        gate = threading.Event()
        original = FakeConverter.load_model

        def slow_load(conv, *a, **k):
            gate.wait(2)
            original(conv, *a, **k)

        with mock.patch.object(FakeConverter, "load_model", slow_load):
            t = threading.Thread(target=self.svc.select_profile, args=("a",))
            t.start()
            deadline = time.monotonic() + 2
            while self.svc.get_status().state != EngineState.LOADING and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertEqual(self.svc.get_status().state, EngineState.LOADING)
            self.assertEqual(len(self.svc.list_profiles()), len(self.registry.list_profiles()))
            gate.set()
            t.join()
        self.assertEqual(self.svc.get_status().state, EngineState.READY)

    def test_start_waits_for_a_worker_left_in_slow_inference(self):
        self.svc.select_profile("a")
        self.svc.start()
        conv = self.svc._converter
        in_flight = []
        original = conv.convert_chunk

        def slow(*a, **k):
            in_flight.append(1)
            time.sleep(2.6)  # longer than the pipeline's 2 s stop() join
            in_flight.pop()
            return original(*a, **k)

        conv.convert_chunk = slow
        self.streams[-1][0].callback(np.ones((CHUNK, 1), dtype=np.float32))
        deadline = time.monotonic() + 2
        while not in_flight and time.monotonic() < deadline:
            time.sleep(0.01)
        self.svc.stop()
        self.assertTrue(in_flight)  # stop() returned while the chunk is still converting
        self.svc.start()  # must wait for that worker instead of sharing the converter with it
        self.assertFalse(in_flight)
        self.assertEqual(self.svc.get_status().state, EngineState.RUNNING)

    def test_list_profiles(self):
        self.svc.select_profile("b")
        summaries = {p.profile_id: p for p in self.svc.list_profiles()}
        self.assertTrue(summaries["b"].is_active and not summaries["a"].is_active)
        self.assertTrue(all(summaries[pid].files_present for pid in ("a", "b")))
        self.assertEqual({summaries[pid].readiness for pid in ("a", "b")}, {"files_present"})
        self.assertEqual(summaries["b"].defaults.pitch_shift, 14.0)

    def test_selectable_and_startup_resolution(self):
        summaries = {p.profile_id: p for p in self.svc.list_profiles()}
        self.assertEqual({pid: s.selectable for pid, s in summaries.items()}, {"a": True, "b": True, "jett": False})
        self.assertEqual(self.svc.resolve_profile_id("b"), "b")
        for stale in ("jett", "removed_profile", None):  # planned, unknown, nothing saved -> default
            self.assertEqual(self.svc.resolve_profile_id(stale), "a")
        (self.dir / "a.pth").unlink()  # default no longer on disk
        self.assertEqual(self.svc.resolve_profile_id("jett"), None)
        self.assertEqual(self.svc.resolve_profile_id("b"), "b")
        self.assertIsNone(self.svc.get_status().last_error)  # resolution never raises or records errors

    def test_planned_profile_is_listed_but_not_ready(self):
        jett = {p.profile_id: p for p in self.svc.list_profiles()}["jett"]
        self.assertEqual((jett.status, jett.readiness, jett.files_present, jett.is_active, jett.data_rights),
                         ("planned", "no_model", False, False, "unverified"))
        self.assertIn("No model yet", jett.setup_hint)
        self.assertFalse(self.svc.validate_profile("jett").ok)

    def test_missing_files_are_reported_with_a_hint(self):
        (self.dir / "a.index").unlink()
        a = {p.profile_id: p for p in self.svc.list_profiles()}["a"]
        self.assertEqual((a.readiness, a.files_present), ("files_missing", False))
        self.assertTrue(a.setup_hint)


class WithdrawnProfileTests(ServiceTestBase):
    """A withdrawn profile (Voice A since 2026-10-22) keeps its files but is out of normal selection."""

    REASON = "Echo during live use; under investigation."

    def setUp(self):
        super().setUp()
        data = json.loads((self.dir / "p.json").read_text())
        data["default_profile"] = "b"
        data["profiles"][0]["withdrawn_reason"] = self.REASON
        (self.dir / "p_w.json").write_text(json.dumps(data))
        self.svc.shutdown()
        self.registry = ProfileRegistry.from_file(self.dir / "p_w.json", base_dir=self.dir)
        self.svc = VoiceChangerService(self.registry, FakeDevices(), self.converter_factory, self.pipeline_factory)

    def test_hidden_from_normal_listing(self):
        self.assertEqual([p.profile_id for p in self.svc.list_profiles()], ["b", "jett"])
        listed = {p.profile_id: p for p in self.svc.list_profiles(include_withdrawn=True)}
        self.assertEqual(listed["a"].withdrawn_reason, self.REASON)
        self.assertEqual((listed["a"].readiness, listed["a"].files_present), ("files_present", True))
        self.assertEqual(listed["b"].withdrawn_reason, "")

    def test_selection_is_refused_and_keeps_the_current_model(self):
        self.svc.select_profile("b")
        with self.assertRaises(ProfileError) as ctx:
            self.svc.select_profile("a")
        self.assertEqual(ctx.exception.code, "profile_withdrawn")
        self.assertIn(self.REASON, ctx.exception.hint)
        st = self.svc.get_status()
        self.assertEqual((st.state, st.profile_id, st.model_loaded), (EngineState.READY, "b", True))
        self.assertEqual(self.loaded_model(), "b.pth")
        self.assertEqual(FakeConverter.loaded_now, 1)

    def test_saved_withdrawn_profile_falls_back_to_the_default(self):
        self.assertEqual(self.svc.resolve_profile_id("a"), "b")
        self.svc.select_profile("a", allow_withdrawn=True)  # loaded for diagnostics: still not offered
        self.assertFalse({p.profile_id: p for p in self.svc.list_profiles()}["a"].selectable)
        self.assertEqual(self.svc.resolve_profile_id("a"), "b")

    def test_explicit_diagnostic_selection_still_works(self):
        self.svc.select_profile("a", allow_withdrawn=True)
        self.assertEqual((self.svc.get_status().profile_id, self.loaded_model()), ("a", "a.pth"))
        self.assertTrue(self.svc.validate_profile("a").ok)
        self.assertTrue(any("withdrawn" in w for w in self.svc.validate_profile("a").warnings))
        # While loaded it stays listed, so a client can still show what is active.
        self.assertTrue({p.profile_id: p for p in self.svc.list_profiles()}["a"].is_active)

    def loaded_model(self):
        return Path(self.svc._converter.model[0]).name


class JettSageLifecycleTests(ServiceTestBase):
    """Jett and Sage with (fake) assets of their own: selection, refusal and switching rules."""

    def setUp(self):
        super().setUp()
        for pid in ("jett", "sage"):
            (self.dir / f"{pid}.pth").write_bytes(f"{pid}-model".encode())
            (self.dir / f"{pid}.index").write_bytes(f"{pid}-index".encode())
        self.build({})

    def build(self, overrides, **ckpt_overrides):
        """Rebuild the service with a, b, jett, sage; `overrides` = {pid: {field: value}}."""
        data = json.loads((self.dir / "p.json").read_text())
        data["profiles"] = [p for p in data["profiles"] if p["id"] in ("a", "b")]
        for pid, pitch in (("jett", 12.0), ("sage", 10.0)):
            entry = {"id": pid, "display_name": pid.title(), "status": "preview", "data_rights": "licensed",
                     "model_path": f"{pid}.pth", "model_sha256": hashlib.sha256(f"{pid}-model".encode()).hexdigest(),
                     "index_path": f"{pid}.index", "index_sha256": hashlib.sha256(f"{pid}-index".encode()).hexdigest(),
                     "defaults": {"pitch_shift": pitch}}
            entry.update(overrides.get(pid, {}))
            data["profiles"].append(entry)
        (self.dir / "p_js.json").write_text(json.dumps(data))
        self.svc.shutdown()
        self.registry = ProfileRegistry.from_file(self.dir / "p_js.json", base_dir=self.dir)
        self.svc = VoiceChangerService(self.registry, FakeDevices(), self.converter_factory, self.pipeline_factory)
        if ckpt_overrides:
            self.patches[0].stop()
            self.patches[0] = mock.patch("voice_changer.conversion.rvc.assets.inspect_checkpoint",
                                         return_value=CheckpointInfo(**{**CKPT.__dict__, **ckpt_overrides}))
            self.patches[0].start()

    def loaded_files(self):
        model, index = self.svc._converter.model
        return Path(model).name, Path(index).name if index else None

    def test_registry_lists_jett_and_sage_separately(self):
        summaries = {p.profile_id: p for p in self.svc.list_profiles()}
        self.assertEqual((summaries["jett"].display_name, summaries["sage"].display_name), ("Jett", "Sage"))
        self.assertEqual({summaries[p].readiness for p in ("jett", "sage")}, {"files_present"})
        self.assertNotEqual(self.registry.get("jett").model_path, self.registry.get("sage").model_path)

    def test_switch_jett_to_sage_loads_only_sages_files_and_releases_jett(self):
        self.svc.select_profile("jett")
        self.assertEqual(self.loaded_files(), ("jett.pth", "jett.index"))
        self.svc.start()
        self.svc.stop()
        self.assertIsNotNone(self.svc.get_metrics())
        self.svc.select_profile("sage")
        self.assertEqual(self.loaded_files(), ("sage.pth", "sage.index"))
        st = self.svc.get_status()
        self.assertEqual((st.state, st.profile_id, st.config.pitch_shift), (EngineState.READY, "sage", 10.0))
        self.assertEqual((FakeConverter.loaded_now, FakeConverter.max_loaded), (1, 1))
        self.assertIsNone(self.svc.get_metrics())  # Jett's session metrics never describe Sage

    def test_switch_while_running_is_refused_and_jett_keeps_streaming(self):
        self.svc.select_profile("jett")
        self.svc.start()
        with self.assertRaises(InvalidStateError):
            self.svc.select_profile("sage")
        st = self.svc.get_status()
        self.assertEqual((st.state, st.profile_id), (EngineState.RUNNING, "jett"))
        self.assertEqual(self.loaded_files(), ("jett.pth", "jett.index"))

    def test_hash_mismatch_is_refused_and_active_profile_survives(self):
        self.svc.select_profile("jett")
        (self.dir / "sage.pth").write_bytes(b"a different checkpoint")
        with self.assertRaises(ProfileError) as ctx:
            self.svc.select_profile("sage")
        self.assertEqual(ctx.exception.code, "profile_invalid")
        self.assertIn("does not match its registered SHA-256", ctx.exception.hint)
        st = self.svc.get_status()
        self.assertEqual((st.state, st.profile_id, st.model_loaded), (EngineState.READY, "jett", True))
        self.assertEqual(self.loaded_files(), ("jett.pth", "jett.index"))

    def test_missing_checkpoint_is_not_ready_with_its_path(self):
        (self.dir / "jett.pth").unlink()
        jett = {p.profile_id: p for p in self.svc.list_profiles()}["jett"]
        self.assertEqual((jett.readiness, jett.files_present), ("files_missing", False))
        with self.assertRaises(ProfileError) as ctx:
            self.svc.select_profile("jett")
        self.assertEqual(ctx.exception.code, "profile_invalid")
        self.assertIn(f"model file not found: {self.dir / 'jett.pth'}", ctx.exception.hint)
        self.assertEqual(FakeConverter.loaded_now, 0)

    def test_missing_declared_index_is_not_ready(self):
        (self.dir / "sage.index").unlink()
        sage = {p.profile_id: p for p in self.svc.list_profiles()}["sage"]
        self.assertEqual(sage.readiness, "files_missing")
        with self.assertRaises(ProfileError) as ctx:
            self.svc.select_profile("sage")
        self.assertIn("index file not found", ctx.exception.hint)

    def test_sage_pointing_at_jetts_files_is_refused(self):
        self.build({"sage": {"model_path": "jett.pth", "model_sha256": None,
                             "index_path": "jett.index", "index_sha256": None}})
        v = self.svc.validate_profile("sage")
        self.assertFalse(v.ok)
        self.assertTrue(any("also registered for profile 'jett'" in e for e in v.errors), v.errors)
        with self.assertRaises(ProfileError):
            self.svc.select_profile("sage")

    def test_copy_of_another_voices_model_is_refused(self):
        # A copy of voice A's checkpoint registered as Jett, under a new file name and with its own hash pinned.
        (self.dir / "jett_copy.pth").write_bytes(b"a")
        self.build({"jett": {"model_path": "jett_copy.pth", "model_sha256": hashlib.sha256(b"a").hexdigest()}})
        v = self.svc.validate_profile("jett")
        self.assertFalse(v.ok)
        self.assertTrue(any("same content as profile 'a'" in e for e in v.errors), v.errors)
        self.assertTrue(self.svc.validate_profile("a").ok)  # the earlier voice keeps working

    def test_unsupported_embedder_is_refused(self):
        self.build({}, embedder_model="chinese-hubert-base")
        with self.assertRaises(ProfileError) as ctx:
            self.svc.select_profile("jett")
        self.assertIn("embedder 'chinese-hubert-base'", ctx.exception.hint)
        self.assertEqual(FakeConverter.loaded_now, 0)

    def test_non_40k_model_loads_with_a_warning(self):
        self.build({}, sample_rate=48000, embedder_model="contentvec")
        v = self.svc.validate_profile("jett")
        self.assertTrue(v.ok, v.errors)
        self.assertTrue(any("48000 Hz" in w for w in v.warnings))
        self.svc.select_profile("jett")
        self.assertEqual(self.svc.get_status().state, EngineState.READY)


if __name__ == "__main__":
    unittest.main()
