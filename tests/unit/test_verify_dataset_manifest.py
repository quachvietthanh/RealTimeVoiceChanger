"""Unit tests for scripts/verify_dataset_manifest.py using synthetic audio."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

script_path = Path(__file__).resolve().parent.parent.parent / "scripts" / "verify_dataset_manifest.py"
spec = importlib.util.spec_from_file_location("verify_dataset_manifest", script_path)
assert spec is not None and spec.loader is not None
vdm = importlib.util.module_from_spec(spec)
sys.modules["verify_dataset_manifest"] = vdm
spec.loader.exec_module(vdm)

SR = 8000


def _write(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, SR, subtype="PCM_16")


class TestVerifyDatasetManifest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        rng = np.random.default_rng(0)
        self.source = (rng.uniform(-0.5, 0.5, size=(1000, 2)) * 32767).astype(np.int16)
        # [0, 600) training, [600, 700) guard, [700, 1000) held-out
        self.bounds = {"training": (0, 600), "guard": (600, 700), "held_out_test": (700, 1000)}
        _write(self.root / "data/source_audio/src.wav", self.source)
        _write(self.root / "data/dataset_raw/train.wav", self.source[0:600])
        _write(self.root / "data/held_out_test/holdout.wav", self.source[700:1000])
        self.manifest_path = self.root / "data/source_audio/manifest.json"
        self._write_manifest()

    def tearDown(self):
        self._tmp.cleanup()

    def _file_meta(self, rel: str) -> dict:
        path = self.root / rel
        return {"sha256": vdm.sha256_of(path), "file_size_bytes": path.stat().st_size}

    def _segment(self, name: str, output_path: str | None = None) -> dict:
        start, end = self.bounds[name]
        seg = {
            "start_seconds": start / SR,
            "end_seconds": end / SR,
            "start_frame": start,
            "end_frame": end,
            "frame_count": end - start,
        }
        if output_path:
            seg.update({"output_path": output_path, **self._file_meta(output_path)})
        return seg

    def _write_manifest(self, train_rel: str = "data/dataset_raw/train.wav",
                        holdout_rel: str = "data/held_out_test/holdout.wav") -> dict:
        manifest = {
            "source_audio": {
                "path": "data/source_audio/src.wav",
                **self._file_meta("data/source_audio/src.wav"),
                "sample_rate_hz": SR,
                "channels": 2,
                "sample_format": "PCM_16",
                "total_frames": 1000,
            },
            "split_specifications": {
                "sample_rate_hz": SR,
                "partition_policy": "half_open_frame_intervals",
                "segments": {
                    "training": self._segment("training", train_rel),
                    "guard": self._segment("guard"),
                    "held_out_test": self._segment("held_out_test", holdout_rel),
                },
            },
        }
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    def _failed(self) -> list[str]:
        return [r.name for r in vdm.verify_manifest(self.manifest_path, self.root) if not r.passed]

    def test_valid_manifest_passes_every_check(self):
        self.assertEqual(self._failed(), [])

    def test_hash_mismatch_fails(self):
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["split_specifications"]["segments"]["held_out_test"]["sha256"] = "0" * 64
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertEqual(self._failed(), ["held_out_test: sha256"])

    def test_overlapping_intervals_fail(self):
        self.bounds["held_out_test"] = (550, 1000)
        _write(self.root / "data/held_out_test/holdout.wav", self.source[550:1000])
        self._write_manifest()
        failed = self._failed()
        self.assertIn("segments: no overlap", failed)
        self.assertIn("segments: cover source without gaps", failed)

    def test_modified_samples_fail_even_with_updated_hash(self):
        tampered = self.source[700:1000].copy()
        tampered[10, 0] += 1
        _write(self.root / "data/held_out_test/holdout.wav", tampered)
        self._write_manifest()  # records the tampered file's true hash
        self.assertEqual(self._failed(), ["held_out_test: sample-exact copy of source interval"])

    def test_held_out_inside_training_directory_fails(self):
        _write(self.root / "data/dataset_raw/holdout.wav", self.source[700:1000])
        self._write_manifest(holdout_rel="data/dataset_raw/holdout.wav")
        self.assertEqual(self._failed(), ["training/held-out: separate directories"])

    def test_missing_file_fails(self):
        (self.root / "data/dataset_raw/train.wav").unlink()
        self.assertEqual(self._failed(), ["training: exists"])

    def test_malformed_manifest_fails(self):
        self.manifest_path.write_text("{not json", encoding="utf-8")
        self.assertEqual(self._failed(), ["manifest: parse"])


derive_path = script_path.parent / "derive_gain_dataset.py"
derive_spec = importlib.util.spec_from_file_location("derive_gain_dataset", derive_path)
assert derive_spec is not None and derive_spec.loader is not None
dgd = importlib.util.module_from_spec(derive_spec)
sys.modules["derive_gain_dataset"] = dgd
derive_spec.loader.exec_module(dgd)


class TestDerivedGainDataset(TestVerifyDatasetManifest):
    """Reuses the split fixture with a loud source (peak 0.99) and derives a 0.88-peak copy."""

    def setUp(self):
        super().setUp()
        # Rewrite the fixture with a loud source so the derivation is a gain reduction
        self.source[0, 0] = int(0.99 * 32768)
        _write(self.root / "data/source_audio/src.wav", self.source)
        _write(self.root / "data/dataset_raw/train.wav", self.source[0:600])
        self._write_manifest()
        self.out_dir = self.root / "data/dataset_train_peak0p88"
        self.derived_manifest = self.out_dir / "derived_manifest.json"

    def _derive(self, **kwargs):
        return dgd.derive(self.manifest_path, self.out_dir, kwargs.get("target", 0.88),
                          "2026-10-21", "2026-10-21", repo_root=self.root)

    def _derived_failed(self) -> list[str]:
        return [r.name for r in vdm.verify_derived_manifest(self.derived_manifest, self.root) if not r.passed]

    def test_derived_dataset_passes_and_leaves_parent_untouched(self):
        before = vdm.sha256_of(self.root / "data/dataset_raw/train.wav")
        m = self._derive()
        self.assertEqual(self._derived_failed(), [])
        self.assertAlmostEqual(m["output"]["peak"], 0.88, places=6)
        self.assertEqual(m["output"]["sample_format"], "FLOAT")
        self.assertEqual(vdm.sha256_of(self.root / "data/dataset_raw/train.wav"), before)

    def test_refuses_to_overwrite_existing_output(self):
        self._derive()
        with self.assertRaises(FileExistsError):
            self._derive()

    def test_refuses_target_at_or_above_limit_and_amplification(self):
        with self.assertRaises(ValueError):
            self._derive(target=0.90)
        with self.assertRaises(ValueError):
            self._derive(target=0.995)  # above the 0.99 source peak -> would amplify

    def test_refuses_when_parent_manifest_fails(self):
        (self.root / "data/held_out_test/holdout.wav").unlink()
        with self.assertRaises(RuntimeError):
            self._derive()
        self.assertFalse(self.out_dir.exists())

    def test_tampered_derived_audio_fails(self):
        m = self._derive()
        out = self.root / m["output"]["path"]
        data, sr = sf.read(str(out), dtype="float32", always_2d=True)
        data[5, 1] += 1e-3
        sf.write(str(out), data, sr, subtype="FLOAT")
        failed = self._derived_failed()
        self.assertIn("derived output: sha256", failed)
        self.assertIn("derived: sample-exact float32(input * gain)", failed)

    def test_derived_inside_held_out_directory_fails(self):
        self.out_dir = self.root / "data/held_out_test/derived"
        self.derived_manifest = self.out_dir / "derived_manifest.json"
        self._derive()
        self.assertEqual(self._derived_failed(), ["derived: separate from held-out directory"])


if __name__ == "__main__":
    unittest.main()
