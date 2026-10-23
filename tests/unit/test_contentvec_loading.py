"""ContentVec is built from hash-verified bytes, weights-only, without transformers reopening any file.

CPU only. Uses a tiny HubertModelWithFinalProj saved to a temporary directory, never the real asset.
"""

import hashlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import numpy as np
import torch
from transformers import HubertConfig

from voice_changer.conversion.rvc import contentvec as cv_mod
from voice_changer.conversion.rvc.adapter import RVCAdapter
from voice_changer.conversion.rvc.assets import CheckpointInfo, IndexInfo
from voice_changer.conversion.rvc.contentvec import (
    CONTENTVEC_CONFIG_SHA256,
    CONTENTVEC_WEIGHTS_SHA256,
    load_contentvec,
)
from voice_changer.conversion.rvc.rmvpe import VerifiedRMVPE0Predictor

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PAYLOAD_CALLS = []


def _payload(*args):
    PAYLOAD_CALLS.append(args)
    return {}


class _Malicious:
    # Unpickling this calls _payload: stands in for code execution from a crafted checkpoint.
    def __reduce__(self):
        return (_payload, ("executed",))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tiny_config() -> HubertConfig:
    return HubertConfig(
        hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
        conv_dim=(32, 32), conv_stride=(5, 2), conv_kernel=(10, 3), num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2, classifier_proj_size=8, architectures=["HubertModelWithFinalProj"],
    )


def _write_embedder(directory: Path, seed: int = 0) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = cv_mod._hubert_class()(_tiny_config())
    state = model.state_dict()
    torch.save(state, directory / "pytorch_model.bin")
    (directory / "config.json").write_text(_tiny_config().to_json_string(), encoding="utf-8")
    return state


class ContentVecLoadingTests(unittest.TestCase):
    def setUp(self):
        PAYLOAD_CALLS.clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name) / "contentvec"
        self.state = _write_embedder(self.dir)
        self.bin, self.cfg = self.dir / "pytorch_model.bin", self.dir / "config.json"

    def _load(self, **kw):
        kw.setdefault("weights_sha256", _sha(self.bin))
        kw.setdefault("config_sha256", _sha(self.cfg))
        return load_contentvec(self.dir, **kw)

    def test_pins_match_the_repository_sources(self):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        try:
            import download_auxiliary_assets as dl
        finally:
            sys.path.remove(str(REPO_ROOT / "scripts"))
        by_type = {a.asset_type: a.expected_sha256 for a in dl.ASSETS}
        self.assertEqual(by_type["contentvec_weights"], CONTENTVEC_WEIGHTS_SHA256)
        self.assertEqual(by_type["json_config"], CONTENTVEC_CONFIG_SHA256)
        doc = (REPO_ROOT / "docs" / "model-setup.md").read_text(encoding="utf-8")
        for pin in (CONTENTVEC_WEIGHTS_SHA256, CONTENTVEC_CONFIG_SHA256):
            self.assertIn(pin, doc)
            self.assertRegex(pin, r"^[0-9a-f]{64}$")

    def test_correct_hashes_load_the_same_model_as_from_pretrained(self):
        model = self._load()
        self.assertFalse(model.training)
        self.assertEqual(model.config._name_or_path, str(self.dir))
        for k, v in self.state.items():
            self.assertTrue(torch.equal(model.state_dict()[k], v), k)
        # Reference: Applio's path-based load of the same directory.
        ref = cv_mod._hubert_class().from_pretrained(str(self.dir)).eval()
        self.assertEqual(model.config.to_dict(), ref.config.to_dict())
        x = torch.randn(1, 4000)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(x)["last_hidden_state"], ref(x)["last_hidden_state"]))
            self.assertTrue(torch.equal(model.final_proj.weight, ref.final_proj.weight))

    def test_missing_expected_hash_is_refused_before_reading(self):
        with mock.patch("torch.load") as load, \
                mock.patch.object(Path, "read_bytes", autospec=True) as rb:
            for kw in ({"weights_sha256": ""}, {"config_sha256": ""}, {"weights_sha256": None}):
                with self.subTest(kw):
                    with self.assertRaisesRegex(ValueError, "expected SHA-256 values .* are required"):
                        load_contentvec(self.dir, **{"weights_sha256": "a" * 64, "config_sha256": "b" * 64, **kw})
        load.assert_not_called()
        rb.assert_not_called()

    def test_wrong_hashes_are_refused_before_deserialization(self):
        for kw in ({"weights_sha256": "0" * 64}, {"config_sha256": "0" * 64}):
            with self.subTest(kw), mock.patch("torch.load") as load:
                with self.assertRaisesRegex(ValueError, "does not match its registered SHA-256"):
                    self._load(**kw)
                load.assert_not_called()

    def test_default_pins_are_the_real_asset_hashes_so_another_file_is_refused(self):
        with mock.patch("torch.load") as load:
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_contentvec(self.dir)
        load.assert_not_called()

    def test_verified_bytes_are_what_the_weights_only_loader_receives(self):
        expected = self.bin.read_bytes()
        pins = {"weights_sha256": _sha(self.bin), "config_sha256": _sha(self.cfg)}
        real_read, real_load = Path.read_bytes, torch.load
        with mock.patch.object(Path, "read_bytes", autospec=True, side_effect=real_read) as rb, \
                mock.patch("torch.load", side_effect=real_load) as load, \
                mock.patch("transformers.modeling_utils.cached_file", side_effect=AssertionError("file lookup")):
            load_contentvec(self.dir, **pins)
        self.assertEqual(sorted(c.args[0].name for c in rb.call_args_list), ["config.json", "pytorch_model.bin"])
        self.assertEqual(load.call_count, 1)
        (source,), kwargs = load.call_args
        self.assertIsInstance(source, io.BytesIO)
        self.assertEqual(source.getvalue(), expected)
        self.assertIs(kwargs.get("weights_only"), True)

    def test_a_file_replaced_after_verification_is_never_read(self):
        pins = {"weights_sha256": _sha(self.bin), "config_sha256": _sha(self.cfg)}
        other = Path(self.tmp.name) / "other"
        _write_embedder(other, seed=1)
        real_read = Path.read_bytes

        def read_then_swap(path):
            data = real_read(path)
            if path.name == "pytorch_model.bin":  # attacker swaps the file right after it was read
                shutil.copyfile(other / "pytorch_model.bin", path)
            return data

        with mock.patch.object(Path, "read_bytes", autospec=True, side_effect=read_then_swap) as rb:
            model = load_contentvec(self.dir, **pins)
        self.assertEqual(rb.call_count, 2)
        for k, v in self.state.items():  # the verified weights, not the swapped-in ones
            self.assertTrue(torch.equal(model.state_dict()[k], v), k)
        # The swapped file itself is refused on the next load.
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_contentvec(self.dir, **pins)

    def test_safe_load_failure_does_not_fall_back(self):
        with mock.patch("torch.load", side_effect=RuntimeError("Weights only load failed")) as load:
            with self.assertRaisesRegex(ValueError, "cannot be loaded weights-only"):
                self._load()
        self.assertEqual(load.call_count, 1)
        self.assertIs(load.call_args.kwargs.get("weights_only"), True)

    def test_pickled_code_is_not_executed(self):
        torch.save({"x": torch.zeros(1), "extra": _Malicious()}, self.bin)
        with self.assertRaisesRegex(ValueError, "cannot be loaded weights-only"):
            self._load()
        self.assertEqual(PAYLOAD_CALLS, [])

    def test_empty_malformed_and_invalid_weights_fail_closed(self):
        partial = dict(self.state)
        partial.pop("final_proj.weight")
        wrong_shape = dict(self.state)
        wrong_shape["final_proj.weight"] = torch.zeros(3, 3)
        cases = {
            "empty file": b"",
            "garbage": b"not a checkpoint",
            "tensor": torch.zeros(3),
            "empty dict": {},
            "non-tensor values": {"a": 1},
            "missing key": partial,
            "wrong shape": wrong_shape,
        }
        for name, content in cases.items():
            with self.subTest(name):
                if isinstance(content, bytes):
                    self.bin.write_bytes(content)
                else:
                    torch.save(content, self.bin)
                with self.assertRaises(ValueError):
                    self._load()

    def test_malformed_or_foreign_config_fails_closed(self):
        for name, text in {"not json": "{", "not an object": "[]",
                           "other architecture": json.dumps({"architectures": ["Wav2Vec2Model"]})}.items():
            with self.subTest(name):
                self.cfg.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    self._load()

    def test_missing_file_is_reported(self):
        self.bin.unlink()
        with self.assertRaises(FileNotFoundError):
            self._load()


class ServiceBoundaryTests(unittest.TestCase):
    """A ContentVec failure inside RVCAdapter.load_model surfaces as model_load_failed with nothing loaded."""

    def setUp(self):
        import faiss

        from voice_changer.profiles.registry import ProfileRegistry
        from voice_changer.service import VoiceChangerService

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        d = Path(self.tmp.name)
        torch.save({"weight": {"emb_g.weight": torch.zeros(109, 4)}, "config": [1, 2, 40000],
                    "version": "v2", "f0": 1}, d / "v.pth")
        idx = faiss.IndexFlatL2(768)
        idx.add(np.random.default_rng(0).normal(size=(5, 768)).astype(np.float32))
        faiss.write_index(idx, str(d / "v.index"))
        (d / "p.json").write_text(json.dumps({"schema_version": 1, "default_profile": "v", "profiles": [{
            "id": "v", "display_name": "V", "status": "preview", "model_path": "v.pth",
            "model_sha256": _sha(d / "v.pth"), "index_path": "v.index", "index_sha256": _sha(d / "v.index")}]}))
        ckpt = CheckpointInfo(version="v2", f0=1, sample_rate=40000, vocoder="HiFi-GAN", embedder_dim=768,
                              speaker_count=109)
        for p in (mock.patch("voice_changer.conversion.rvc.assets.inspect_checkpoint", return_value=ckpt),
                  mock.patch("voice_changer.conversion.rvc.assets.inspect_index",
                             return_value=IndexInfo(dim=768, ntotal=5, is_trained=True))):
            p.start()
            self.addCleanup(p.stop)

        self.applio = d / "applio"
        (self.applio / "rvc" / "models" / "predictors").mkdir(parents=True)
        (self.applio / "rvc" / "models" / "predictors" / "rmvpe.pt").write_bytes(b"x")
        self.cv_dir = self.applio / "rvc" / "models" / "embedders" / "contentvec"
        _write_embedder(self.cv_dir)
        self.vcs = []

        def converter_factory():
            adapter = RVCAdapter(device="cpu")
            adapter._applio_root = self.applio
            vc = mock.MagicMock()
            vc.hubert_model = None
            vc.config.device, vc.config.is_half = "cpu", False
            vc.vc.model_rmvpe = VerifiedRMVPE0Predictor.__new__(VerifiedRMVPE0Predictor)
            vc.vc.pipeline.return_value = np.zeros(100, dtype=np.float32)
            adapter._sync_auxiliary_assets = lambda: None
            adapter._ensure_applio_initialized = lambda: vc
            self.vcs.append(vc)
            return adapter

        self.svc = VoiceChangerService(registry=ProfileRegistry.from_file(d / "p.json", base_dir=d),
                                       converter_factory=converter_factory)
        self.addCleanup(self.svc.shutdown)

    def test_contentvec_failure_is_model_load_failed_and_nothing_is_active(self):
        from voice_changer.service import EngineState, ProfileError

        # The tiny test embedder is not the pinned real asset, so the adapter's verified load refuses it.
        with self.assertRaises(ProfileError) as ctx:
            self.svc.select_profile("v")
        self.assertEqual(ctx.exception.code, "model_load_failed")
        self.assertIn("ContentVec", ctx.exception.details)
        st = self.svc.get_status()
        self.assertEqual((st.state, st.profile_id, st.model_loaded), (EngineState.IDLE, None, False))
        self.assertIsNone(self.vcs[-1].hubert_model)
        self.vcs[-1].load_hubert.assert_not_called()

    def test_verified_embedder_is_installed_when_its_pins_match(self):
        from voice_changer.service import EngineState

        pins = (_sha(self.cv_dir / "pytorch_model.bin"), _sha(self.cv_dir / "config.json"))
        # The adapter relies on load_contentvec's default pins; point them at the test embedder.
        with mock.patch.object(cv_mod.load_contentvec, "__defaults__", pins):
            self.svc.select_profile("v")
        self.assertEqual(self.svc.get_status().state, EngineState.READY)
        vc = self.vcs[-1]
        self.assertIsInstance(vc.hubert_model, cv_mod._hubert_class())
        self.assertFalse(vc.hubert_model.training)
        self.assertEqual(vc.last_embedder_model, "contentvec")
        vc.load_hubert.assert_not_called()


if __name__ == "__main__":
    unittest.main()
