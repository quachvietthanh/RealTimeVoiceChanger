"""RVCAdapter.load_model reads each file once, checks its pin on those bytes and never unpickles freely.

CPU only: Applio's VoiceConverter is replaced by a mock, so no network is built and no GPU is used.
"""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import faiss
import numpy as np
import torch

from voice_changer.conversion.rvc import adapter as adapter_mod
from voice_changer.conversion.rvc.adapter import RVCAdapter
from voice_changer.conversion.rvc.rmvpe import VerifiedRMVPE0Predictor

PAYLOAD_CALLS = []


def _payload(*args):
    PAYLOAD_CALLS.append(args)
    return {"weight": {}, "config": [40000]}


class _Malicious:
    # Unpickling this calls _payload: stands in for code execution from a crafted checkpoint.
    def __reduce__(self):
        return (_payload, ("executed",))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AdapterLoadingTests(unittest.TestCase):
    def setUp(self):
        PAYLOAD_CALLS.clear()
        self.tmp = Path(tempfile.mkdtemp())
        self.model = self.tmp / "voice.pth"
        torch.save({"weight": {"emb_g.weight": torch.zeros(109, 4)}, "config": [1, 2, 40000],
                    "version": "v2", "f0": 1}, self.model)
        self.index = self.tmp / "voice.index"
        idx = faiss.IndexFlatL2(768)
        idx.add(np.random.default_rng(0).normal(size=(5, 768)).astype(np.float32))
        faiss.write_index(idx, str(self.index))

        self.adapter = RVCAdapter(device="cpu")
        applio_root = self.tmp / "applio"
        (applio_root / "rvc" / "models" / "predictors").mkdir(parents=True)
        (applio_root / "rvc" / "models" / "predictors" / "rmvpe.pt").write_bytes(b"x")
        self.adapter._applio_root = applio_root
        self.vc = mock.MagicMock()
        self.vc.hubert_model = object()
        # What a real setup_vc_instance() builds once the adapter has installed the verified predictor.
        self.vc.vc.model_rmvpe = VerifiedRMVPE0Predictor.__new__(VerifiedRMVPE0Predictor)
        patches = [
            mock.patch.object(self.adapter, "_sync_auxiliary_assets"),
            mock.patch.object(self.adapter, "_ensure_applio_initialized", return_value=self.vc),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_installs_the_verified_checkpoint_without_applio_reading_the_file(self):
        self.adapter.load_model(str(self.model), str(self.index),
                                model_sha256=_sha(self.model), index_sha256=_sha(self.index))
        self.assertTrue(self.adapter.is_loaded)
        self.vc.get_vc.assert_not_called()
        self.vc.load_model.assert_not_called()
        self.vc.setup_network.assert_called_once_with()
        self.vc.setup_vc_instance.assert_called_once_with()
        self.assertEqual(self.vc.cpt["config"][-1], 40000)
        self.assertEqual(self.vc.loaded_model, str(self.model.resolve()))
        self.assertEqual(self.adapter._cached_index.ntotal, 5)
        self.assertEqual(self.adapter._cached_big_npy.shape, (5, 768))

    def test_checkpoint_is_deserialized_weights_only_from_the_hashed_bytes(self):
        real_load, real_read = torch.load, Path.read_bytes
        pin = _sha(self.model)
        with mock.patch.object(Path, "read_bytes", autospec=True, side_effect=real_read) as rb, \
                mock.patch("torch.load", side_effect=real_load) as load:
            self.adapter.load_model(str(self.model), model_sha256=pin)
        self.assertEqual(rb.call_count, 1)  # one read; the path is never handed to torch.load
        (source,), kwargs = load.call_args
        self.assertNotIsInstance(source, (str, Path))
        self.assertIs(kwargs.get("weights_only"), True)

    def test_model_hash_mismatch_is_refused_before_deserializing(self):
        with mock.patch("torch.load") as load:
            with self.assertRaisesRegex(ValueError, "does not match its registered SHA-256"):
                self.adapter.load_model(str(self.model), model_sha256="0" * 64)
        load.assert_not_called()
        self.assertFalse(self.adapter.is_loaded)
        self.vc.setup_network.assert_not_called()

    def test_index_hash_mismatch_is_refused(self):
        with self.assertRaisesRegex(ValueError, "index file .* does not match"):
            self.adapter.load_model(str(self.model), str(self.index),
                                    model_sha256=_sha(self.model), index_sha256="0" * 64)
        self.assertFalse(self.adapter.is_loaded)
        self.vc.setup_network.assert_not_called()

    def test_pickled_code_in_a_checkpoint_is_not_executed(self):
        torch.save({"weight": {}, "config": [40000], "extra": _Malicious()}, self.model)
        with self.assertRaisesRegex(ValueError, "Failed to load PyTorch checkpoint"):
            self.adapter.load_model(str(self.model), model_sha256=_sha(self.model))
        self.assertEqual(PAYLOAD_CALLS, [])
        self.assertFalse(self.adapter.is_loaded)

    def test_model_is_refused_if_the_pitch_predictor_was_not_verified(self):
        self.vc.vc.model_rmvpe = mock.MagicMock()  # e.g. built by Applio's unrestricted RMVPE0Predictor
        with self.assertRaisesRegex(RuntimeError, "VerifiedRMVPE0Predictor"):
            self.adapter.load_model(str(self.model), model_sha256=_sha(self.model))
        self.assertFalse(self.adapter.is_loaded)

    def test_missing_embedder_is_installed_from_the_verified_loader(self):
        self.vc.hubert_model = None
        self.vc.config.device, self.vc.config.is_half = "cpu", False
        model = mock.MagicMock()
        with mock.patch.object(adapter_mod, "load_contentvec", return_value=model) as load:
            self.adapter.load_model(str(self.model), model_sha256=_sha(self.model))
        load.assert_called_once_with(self.adapter._applio_root / "rvc" / "models" / "embedders" / "contentvec")
        model.to.assert_called_once_with("cpu")
        self.assertIs(self.vc.hubert_model, model.float.return_value)
        self.vc.hubert_model.eval.assert_called_once_with()
        self.assertEqual(self.vc.last_embedder_model, "contentvec")
        self.vc.load_hubert.assert_not_called()
        self.assertTrue(self.adapter.is_loaded)

    def test_embedder_failure_leaves_the_adapter_unloaded(self):
        self.vc.hubert_model = None
        with mock.patch.object(adapter_mod, "load_contentvec", side_effect=ValueError("ContentVec weights …")):
            with self.assertRaisesRegex(ValueError, "ContentVec"):
                self.adapter.load_model(str(self.model), model_sha256=_sha(self.model))
        self.assertFalse(self.adapter.is_loaded)
        self.vc.load_hubert.assert_not_called()

    def test_applio_embedder_loader_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "unverified embedder load"):
            adapter_mod._refuse_embedder_load("contentvec")

    def test_applio_path_loader_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "unrestricted torch.load"):
            adapter_mod._refuse_path_load(str(self.model))


class ApplioInstanceHardeningTests(unittest.TestCase):
    def test_initialized_applio_converter_cannot_load_from_a_path(self):
        fake_module, fake_pipeline = mock.MagicMock(), mock.MagicMock()
        adapter = RVCAdapter(device="cpu")
        with mock.patch.dict(sys.modules, {"rvc.infer.infer": fake_module, "rvc.infer.pipeline": fake_pipeline}), \
                mock.patch.object(adapter, "_sync_auxiliary_assets"):
            vc = adapter._ensure_applio_initialized()
        self.assertIs(vc.load_model, adapter_mod._refuse_path_load)
        self.assertIs(vc.load_hubert, adapter_mod._refuse_embedder_load)
        # Applio's Pipeline builds its pitch predictor through this name.
        self.assertIs(fake_pipeline.RMVPE0Predictor, VerifiedRMVPE0Predictor)


if __name__ == "__main__":
    unittest.main()
