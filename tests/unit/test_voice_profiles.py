"""Unit tests for the voice profile registry and conversion config validation (no GPU, no real models)."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from voice_changer.conversion.base import ConversionConfig, validate_conversion_config
from voice_changer.conversion.rvc.assets import CheckpointInfo, IndexInfo, check_pair
from voice_changer.profiles.registry import ProfileRegistry, ProfileRegistryError

V2_CKPT = CheckpointInfo(version="v2", f0=1, sample_rate=40000, vocoder="HiFi-GAN", embedder_dim=768, speaker_count=1)


def _profile(pid, model="a.pth", index="a.index", **extra):
    p = {"id": pid, "display_name": pid, "status": "preview", "model_path": model, "index_path": index}
    p.update(extra)
    return p


class RegistryParsingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, data):
        path = self.dir / "profiles.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def load(self, profiles, **top):
        data = {"schema_version": 1, "profiles": profiles}
        data.update(top)
        return ProfileRegistry.from_file(self.write(data), base_dir=self.dir)

    def test_loads_profiles_and_resolves_relative_paths(self):
        reg = self.load([_profile("a"), _profile("b", "sub/b.pth", None)], default_profile="a")
        self.assertEqual([p.profile_id for p in reg.list_profiles()], ["a", "b"])
        self.assertEqual(reg.default_profile_id, "a")
        self.assertEqual(reg.get("a").model_path, self.dir / "a.pth")
        self.assertIsNone(reg.get("b").index_path)
        self.assertEqual(reg.get("a").defaults, ConversionConfig())

    def test_rejects_duplicate_ids(self):
        with self.assertRaises(ProfileRegistryError):
            self.load([_profile("a"), _profile("a")])

    def test_rejects_unknown_default(self):
        with self.assertRaises(ProfileRegistryError):
            self.load([_profile("a")], default_profile="zzz")

    def test_rejects_bad_status_and_bad_defaults(self):
        with self.assertRaises(ProfileRegistryError):
            self.load([_profile("a", status="production")])
        with self.assertRaises(ProfileRegistryError):
            self.load([_profile("a", defaults={"index_rate": 2.0})])
        with self.assertRaises(ProfileRegistryError):
            self.load([_profile("a", defaults={"no_such_field": 1})])

    def test_rejects_wrong_schema_and_missing_file(self):
        with self.assertRaises(ProfileRegistryError):
            ProfileRegistry.from_file(self.write({"schema_version": 99, "profiles": []}))
        with self.assertRaises(ProfileRegistryError):
            ProfileRegistry.from_file(self.dir / "missing.json")

    def test_unknown_profile_raises_key_error(self):
        reg = self.load([_profile("a")])
        with self.assertRaises(KeyError):
            reg.get("b")

    def test_repository_registry_parses(self):
        reg = ProfileRegistry.from_file()
        ids = {p.profile_id for p in reg.list_profiles()}
        self.assertIn(reg.default_profile_id, ids)
        # No preview may be marked approved without the owner.
        self.assertTrue(all(p.status != "approved" for p in reg.list_profiles()))
        # Each profile pins its own model and index, and no index is shared between profiles.
        indexes = [p.index_sha256 for p in reg.list_profiles() if p.index_path]
        self.assertEqual(len(indexes), len(set(indexes)))
        self.assertTrue(all(p.model_sha256 for p in reg.list_profiles()))


class RegistryValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / "a.pth").write_bytes(b"model-a")
        (self.dir / "a.index").write_bytes(b"index-a")
        self.sha_model = hashlib.sha256(b"model-a").hexdigest()
        self.sha_index = hashlib.sha256(b"index-a").hexdigest()

    def tearDown(self):
        self.tmp.cleanup()

    def registry(self, **extra):
        data = {"schema_version": 1, "profiles": [_profile("a", model_sha256=self.sha_model,
                                                           index_sha256=self.sha_index, **extra)]}
        path = self.dir / "p.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return ProfileRegistry.from_file(path, base_dir=self.dir)

    def validate(self, reg, ckpt=V2_CKPT, idx=IndexInfo(dim=768, ntotal=100, is_trained=True)):
        with mock.patch("voice_changer.conversion.rvc.assets.inspect_checkpoint", return_value=ckpt), \
             mock.patch("voice_changer.conversion.rvc.assets.inspect_index", return_value=idx):
            return reg.validate("a")

    def test_valid_profile(self):
        v = self.validate(self.registry())
        self.assertTrue(v.ok, v.errors)
        self.assertEqual(v.index_dim, 768)
        self.assertTrue(any("not approved" in w for w in v.warnings))

    def test_hash_mismatch_is_an_error(self):
        (self.dir / "a.index").write_bytes(b"index-of-another-speaker")
        v = self.validate(self.registry())
        self.assertFalse(v.ok)
        self.assertTrue(any("index file does not match" in e for e in v.errors))

    def test_missing_files_are_errors(self):
        (self.dir / "a.pth").unlink()
        (self.dir / "a.index").unlink()
        v = self.validate(self.registry())
        self.assertFalse(v.ok)
        self.assertEqual(len([e for e in v.errors if "not found" in e]), 2)

    def test_dimension_mismatch_is_an_error(self):
        v = self.validate(self.registry(), idx=IndexInfo(dim=256, ntotal=100, is_trained=True))
        self.assertFalse(v.ok)
        self.assertTrue(any("dimension" in e for e in v.errors))

    def test_unreadable_checkpoint_is_an_error(self):
        with mock.patch("voice_changer.conversion.rvc.assets.inspect_checkpoint", side_effect=ValueError("bad")), \
             mock.patch("voice_changer.conversion.rvc.assets.inspect_index", return_value=IndexInfo(768, 1, True)):
            v = self.registry().validate("a")
        self.assertFalse(v.ok)

    def test_unpinned_hash_is_a_warning(self):
        data = {"schema_version": 1, "profiles": [_profile("a")]}
        path = self.dir / "p2.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        v = self.validate(ProfileRegistry.from_file(path, base_dir=self.dir))
        self.assertTrue(v.ok)
        self.assertTrue(any("not pinned" in w for w in v.warnings))


class ConfigValidationTests(unittest.TestCase):
    def test_defaults_are_valid(self):
        self.assertEqual(validate_conversion_config(ConversionConfig()), [])

    def test_out_of_range_and_wrong_types(self):
        cases = [
            ConversionConfig(pitch_shift=30),
            ConversionConfig(index_rate=-0.1),
            ConversionConfig(protect_rate=0.6),
            ConversionConfig(filter_radius=2.5),  # type: ignore[arg-type]
            ConversionConfig(filter_radius=9),
            ConversionConfig(f0_method="harvest"),
            ConversionConfig(pitch_shift=True),  # type: ignore[arg-type]
        ]
        for cfg in cases:
            with self.subTest(cfg=cfg):
                self.assertNotEqual(validate_conversion_config(cfg), [])

    def test_check_pair(self):
        self.assertEqual(check_pair(V2_CKPT, None), [])
        self.assertEqual(check_pair(V2_CKPT, IndexInfo(768, 10, True)), [])
        self.assertTrue(check_pair(V2_CKPT, IndexInfo(768, 0, True)))


if __name__ == "__main__":
    unittest.main()
