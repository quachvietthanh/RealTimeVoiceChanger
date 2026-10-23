"""Integration test: every profile in config/voice_profiles.json validates against the real files.

CPU only (reads checkpoints and indexes, hashes them). Skips a profile whose files are absent on this
machine (model files are not in Git); a present file with a wrong hash or shape fails. A 'planned'
profile (no model yet) must stay unloadable.
"""

import sys
import unittest
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from voice_changer.profiles.registry import REPO_ROOT, ProfileRegistry


class RepositoryProfileAssetsTests(unittest.TestCase):
    def test_registered_profiles_validate(self):
        reg = ProfileRegistry.from_file()
        for profile in reg.list_profiles():
            with self.subTest(profile=profile.profile_id):
                if profile.is_planned:
                    self.assertIsNone(profile.model_path)
                    self.assertFalse(reg.validate(profile.profile_id).ok)
                    continue
                if not profile.files_present():
                    self.skipTest(f"assets of '{profile.profile_id}' are not on this machine")
                v = reg.validate(profile.profile_id)
                self.assertTrue(v.ok, v.errors)
                self.assertEqual((v.model_version, v.model_sample_rate, v.index_dim), ("v2", 40000, 768))

    def test_jett_and_sage_refuse_files_that_do_not_match_their_pins(self):
        # Third-party downloads: a replaced or corrupted file must never load under the profile's name.
        import json

        from voice_changer.service import ProfileError, VoiceChangerService

        data = json.loads((REPO_ROOT / "config" / "voice_profiles.json").read_text(encoding="utf-8"))
        for pid in ("jett", "sage"):
            with self.subTest(profile=pid):
                entry = next(p for p in data["profiles"] if p["id"] == pid)
                if not ProfileRegistry.from_data(data).get(pid).files_present():
                    self.skipTest(f"assets of '{pid}' are not on this machine")
                for field in ("model_sha256", "index_sha256"):
                    original = entry[field]
                    entry[field] = "0" * 64
                    try:
                        reg = ProfileRegistry.from_data(data)
                        self.assertFalse(reg.validate(pid).ok)
                        svc = VoiceChangerService(registry=reg, converter_factory=lambda: self.fail("must not load"))
                        with self.assertRaises(ProfileError) as ctx:
                            svc.select_profile(pid)
                        self.assertEqual(ctx.exception.code, "profile_invalid")
                        self.assertFalse(svc.get_status().model_loaded)
                    finally:
                        entry[field] = original


if __name__ == "__main__":
    unittest.main()
