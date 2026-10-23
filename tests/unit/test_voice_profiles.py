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
from voice_changer.profiles.registry import REPO_ROOT, ProfileRegistry, ProfileRegistryError

V2_CKPT = CheckpointInfo(version="v2", f0=1, sample_rate=40000, vocoder="HiFi-GAN", embedder_dim=768, speaker_count=1)
# Third-party Jett/Sage files registered 2026-10-22 (sources in config/voice_profiles.json provenance).
JETT_SAGE_FILES = (
    ("jett", "83745f80f560c727367f7e172a35605fbb5ca2f50749953e4a847c0111062bc7",
     "d014f9dc0b5f4649ad84f5c9070a71b64b604738f234666d21cfa3587edfd8ed"),
    ("sage", "ef95e555298bac728d637174ebe7263fc1b12e7f6ed60b66b411ad96883045d2",
     "35d408eea2a70e4f7e4f54bb4093f7c0fd9499246accdc7d6c076560e7d88db7"),
)


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
        self.assertTrue(all(p.model_sha256 for p in reg.list_profiles() if not p.is_planned))
        # Jett and Sage use third-party models (2026-10-22): preview only, pinned to their own local files, and
        # neither the data rights nor the model license are verified (local evaluation only).
        for pid, model_sha, index_sha in JETT_SAGE_FILES:
            p = reg.get(pid)
            self.assertEqual((p.status, p.data_rights, p.provenance.get("license_status")),
                             ("preview", "unverified", "LICENSE_UNVERIFIED"))
            self.assertEqual((p.model_path, p.model_sha256, p.index_path, p.index_sha256),
                             (REPO_ROOT / "models" / "weights" / f"{pid}_valorant_500e.pth", model_sha,
                              REPO_ROOT / "models" / "indices" / f"{pid}_valorant_500e.index", index_sha))
            self.assertTrue(p.provenance.get("download", "").startswith("https://huggingface.co/"))
        self.assertNotEqual(reg.get("jett").dataset["path"], reg.get("sage").dataset["path"])

    def test_repository_jett_and_sage_are_unselectable_without_their_local_files(self):
        # Model files are not in Git: on a machine without them, the profiles must say so instead of loading.
        from voice_changer.service import VoiceChangerService

        data = json.loads((REPO_ROOT / "config" / "voice_profiles.json").read_text(encoding="utf-8"))
        reg = ProfileRegistry.from_data(data, base_dir=self.dir)  # same entries, files absent under this dir
        summaries = {s.profile_id: s for s in VoiceChangerService(registry=reg).list_profiles()}
        for pid in ("jett", "sage"):
            s = summaries[pid]
            self.assertEqual((s.readiness, s.selectable), ("files_missing", False))
            self.assertIn("config/voice_profiles.json", s.setup_hint)
            errors = reg.validate(pid).errors
            self.assertTrue(any(e.startswith("model file not found") and f"{pid}_valorant_500e.pth" in e
                                for e in errors), errors)

    def test_repository_voice_b_defaults_are_unchanged(self):
        p = ProfileRegistry.from_file().get("voice_b_preview")
        self.assertEqual(p.defaults, ConversionConfig(pitch_shift=14.0, f0_method="rmvpe", index_rate=0.25,
                                                      filter_radius=2, protect_rate=0.4))

    def test_repository_voice_a_and_b_are_unchanged(self):
        # Guard: Jett/Sage work must not move or re-pin the human voices. Change only with the owner's choice.
        reg = ProfileRegistry.from_file()
        expected = {
            "voice_a_preview": ("nvf_short_v1_50e_5350s.pth", "d6ed4d41fed6b30e6f3b95916865607343ecf782b6351af8947dbff768b3cc28",
                                "added_nvf_short_v1_v2.index", "798371f59f8719ea264c985329dc1c6d0755c7505da61f8143e5d2a04eba33e0"),
            "voice_b_preview": ("spkb_preview_v2_25e_1975s.pth", "06f6ec38e1d87336b012eb4a2b7edb0b8457e8e48391e6a08d732c36ec244532",
                                "added_spkb_preview_v2_v2.index", "fbd308fe60329dcbf159641669e2210362ab8115ceabd13ca70e61dd9ab13f7f"),
        }
        for pid, (model, model_sha, index, index_sha) in expected.items():
            p = reg.get(pid)
            self.assertEqual((p.status, p.model_path.name, p.model_sha256, p.index_path.name, p.index_sha256),
                             ("preview", model, model_sha, index, index_sha))

    def test_repository_voice_a_is_withdrawn_and_voice_b_is_not(self):
        # Owner request 2026-10-22: Voice A out of normal selection, files kept for diagnostics and rollback.
        reg = ProfileRegistry.from_file()
        self.assertTrue(reg.get("voice_a_preview").is_withdrawn)
        self.assertEqual([p.profile_id for p in reg.list_profiles() if p.is_withdrawn], ["voice_a_preview"])
        self.assertFalse(reg.get(reg.default_profile_id).is_withdrawn)

    def test_withdrawn_reason_rules(self):
        reg = self.load([_profile("a", withdrawn_reason="echo"), _profile("b", withdrawn_reason="  ")])
        self.assertEqual((reg.get("a").is_withdrawn, reg.get("a").withdrawn_reason), (True, "echo"))
        self.assertFalse(reg.get("b").is_withdrawn)  # blank = not withdrawn
        self.assertFalse(self.load([_profile("a")]).get("a").is_withdrawn)
        with self.assertRaises(ProfileRegistryError):
            self.load([_profile("a", withdrawn_reason=True)])
        with self.assertRaises(ProfileRegistryError):
            self.load([_profile("a", status="approved", data_rights="owner_recorded", withdrawn_reason="x")])
        with self.assertRaises(ProfileRegistryError):  # the default must stay selectable
            self.load([_profile("a", withdrawn_reason="x"), _profile("b")], default_profile="a")

    def test_planned_profile_has_no_model(self):
        reg = self.load([{"id": "jett", "display_name": "Jett", "status": "planned", "data_rights": "unverified"}])
        p = reg.get("jett")
        self.assertTrue(p.is_planned)
        self.assertIsNone(p.model_path)
        self.assertFalse(p.files_present())
        with self.assertRaises(ProfileRegistryError):  # a planned profile must not point at a model
            self.load([_profile("jett", status="planned")])
        with self.assertRaises(ProfileRegistryError):  # every other status needs one
            self.load([{"id": "a", "display_name": "a", "status": "preview"}])

    def test_data_rights_values(self):
        self.assertEqual(self.load([_profile("a")]).get("a").data_rights, "unspecified")
        with self.assertRaises(ProfileRegistryError):
            self.load([_profile("a", data_rights="public")])
        with self.assertRaises(ProfileRegistryError):
            self.load([_profile("a", status="approved", data_rights="unverified")])
        with self.assertRaises(ProfileRegistryError):
            self.load([_profile("a", dataset="data/x")])
        self.load([_profile("a", status="approved", data_rights="owner_recorded")])


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

    def test_unverified_data_rights_is_a_warning(self):
        v = self.validate(self.registry(data_rights="unverified"))
        self.assertTrue(v.ok)
        self.assertTrue(any("rights are unverified" in w for w in v.warnings))

    def test_planned_profile_is_never_valid(self):
        data = {"schema_version": 1, "profiles": [{"id": "a", "display_name": "a", "status": "planned"}]}
        path = self.dir / "p3.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        v = self.validate(ProfileRegistry.from_file(path, base_dir=self.dir))
        self.assertFalse(v.ok)
        self.assertIn("no model registered", v.errors[0])


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
