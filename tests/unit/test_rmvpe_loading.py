"""rmvpe.pt is hash-verified and deserialized weights-only from the verified bytes, with no unrestricted fallback.

CPU only. Uses temporary checkpoints (Applio's E2E architecture with random weights), never the real asset.
"""

import hashlib
import io
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

from voice_changer.conversion.rvc import rmvpe as rmvpe_mod
from voice_changer.conversion.rvc.rmvpe import (
    RMVPE_SHA256,
    VerifiedRMVPE0Predictor,
    load_rmvpe_state_dict,
)

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


class RmvpeLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        torch.manual_seed(0)
        cls.state = rmvpe_mod.E2E(4, 1, (2, 2)).state_dict()
        cls.ckpt = cls.dir / "rmvpe.pt"
        torch.save(cls.state, cls.ckpt)
        cls.pin = _sha(cls.ckpt)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        PAYLOAD_CALLS.clear()

    def _write(self, name, obj=None, raw=None):
        path = self.dir / name
        if raw is not None:
            path.write_bytes(raw)
        else:
            torch.save(obj, path)
        return path, _sha(path)

    def test_pin_matches_the_repository_sources(self):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        try:
            import download_auxiliary_assets as dl
        finally:
            sys.path.remove(str(REPO_ROOT / "scripts"))
        spec = next(a for a in dl.ASSETS if a.asset_type == "rmvpe")
        self.assertEqual(spec.expected_sha256, RMVPE_SHA256)
        self.assertIn(RMVPE_SHA256, (REPO_ROOT / "docs" / "model-setup.md").read_text(encoding="utf-8"))
        self.assertRegex(RMVPE_SHA256, r"^[0-9a-f]{64}$")

    def test_hash_mismatch_is_refused_before_any_deserialization(self):
        with mock.patch("torch.load") as load:
            with self.assertRaisesRegex(ValueError, "does not match its registered SHA-256"):
                load_rmvpe_state_dict(self.ckpt, "0" * 64)
        load.assert_not_called()

    def test_default_pin_is_the_real_asset_hash_so_another_file_is_refused(self):
        with mock.patch("torch.load") as load:
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_rmvpe_state_dict(self.ckpt)  # temp checkpoint vs RMVPE_SHA256
        load.assert_not_called()

    def test_an_expected_hash_is_mandatory(self):
        with mock.patch("torch.load") as load:
            for empty in ("", None):
                with self.assertRaisesRegex(ValueError, "expected SHA-256 is required"):
                    load_rmvpe_state_dict(self.ckpt, empty)
        load.assert_not_called()

    def test_verified_bytes_are_what_the_weights_only_loader_receives(self):
        real_read, real_load = Path.read_bytes, torch.load
        expected = self.ckpt.read_bytes()
        with mock.patch.object(Path, "read_bytes", autospec=True, side_effect=real_read) as rb, \
                mock.patch("torch.load", side_effect=real_load) as load:
            state = load_rmvpe_state_dict(self.ckpt, self.pin)
        self.assertEqual(rb.call_count, 1)
        self.assertEqual(load.call_count, 1)
        (source,), kwargs = load.call_args
        self.assertIsInstance(source, io.BytesIO)
        self.assertEqual(source.getvalue(), expected)
        self.assertIs(kwargs.get("weights_only"), True)
        self.assertEqual(state.keys(), self.state.keys())

    def test_safe_load_failure_does_not_fall_back(self):
        with mock.patch("torch.load", side_effect=RuntimeError("Weights only load failed")) as load:
            with self.assertRaisesRegex(ValueError, "cannot be loaded weights-only"):
                load_rmvpe_state_dict(self.ckpt, self.pin)
        self.assertEqual(load.call_count, 1)
        self.assertIs(load.call_args.kwargs.get("weights_only"), True)

    def test_pickled_code_is_not_executed(self):
        path, pin = self._write("evil.pt", {"x": torch.zeros(1), "extra": _Malicious()})
        with self.assertRaisesRegex(ValueError, "cannot be loaded weights-only"):
            load_rmvpe_state_dict(path, pin)
        self.assertEqual(PAYLOAD_CALLS, [])

    def test_malformed_and_unsupported_files_fail_closed(self):
        cases = {
            "garbage.pt": dict(raw=b"not a checkpoint"),
            "tensor.pt": dict(obj=torch.zeros(3)),
            "empty.pt": dict(obj={}),
            "nontensor.pt": dict(obj={"a": 1}),
        }
        for name, kw in cases.items():
            with self.subTest(name):
                path, pin = self._write(name, **kw)
                with self.assertRaises(ValueError):
                    load_rmvpe_state_dict(path, pin)

    def test_wrong_layout_is_rejected_by_the_strict_state_dict_load(self):
        state = dict(self.state)
        state.pop(next(iter(state)))
        path, pin = self._write("partial.pt", state)
        with self.assertRaises(RuntimeError):
            VerifiedRMVPE0Predictor(path, is_half=False, device="cpu", expected_sha256=pin)

    def test_predictor_initializes_like_applios_from_verified_weights(self):
        ours = VerifiedRMVPE0Predictor(str(self.ckpt), is_half=False, device="cpu", expected_sha256=self.pin)
        # Reference: Applio's own constructor, its loader stubbed to return the same state dict (test only).
        with mock.patch("torch.load", return_value=self.state):
            ref = rmvpe_mod.RMVPE0Predictor("unused", is_half=False, device="cpu")
        self.assertEqual(sorted(vars(ours)), sorted(vars(ref)))
        self.assertTrue(np.array_equal(ours.cents_mapping, ref.cents_mapping))
        self.assertEqual((ours.is_half, ours.device), (ref.is_half, ref.device))
        self.assertFalse(ours.model.training)
        for k, v in ref.model.state_dict().items():
            self.assertTrue(torch.equal(ours.model.state_dict()[k], v), k)
        x = np.random.default_rng(0).normal(0, 0.1, 16000).astype(np.float32)
        self.assertTrue(np.array_equal(ours.infer_from_audio(x), ref.infer_from_audio(x)))

    def test_predictor_refuses_a_mismatched_file_before_building_a_model(self):
        with mock.patch.object(rmvpe_mod, "E2E") as e2e:
            with self.assertRaises(ValueError):
                VerifiedRMVPE0Predictor(str(self.ckpt), is_half=False, device="cpu")
        e2e.assert_not_called()


if __name__ == "__main__":
    unittest.main()
