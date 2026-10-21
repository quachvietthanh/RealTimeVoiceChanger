"""Integration test: every profile in config/voice_profiles.json validates against the real files.

CPU only (reads checkpoints and indexes, hashes them). Skips a profile whose files are absent on this
machine (model files are not in Git); a present file with a wrong hash or shape fails.
"""

import sys
import unittest
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from voice_changer.profiles.registry import ProfileRegistry


class RepositoryProfileAssetsTests(unittest.TestCase):
    def test_registered_profiles_validate(self):
        reg = ProfileRegistry.from_file()
        for profile in reg.list_profiles():
            with self.subTest(profile=profile.profile_id):
                present = profile.model_path.is_file() and (profile.index_path is None or profile.index_path.is_file())
                if not present:
                    self.skipTest(f"assets of '{profile.profile_id}' are not on this machine")
                v = reg.validate(profile.profile_id)
                self.assertTrue(v.ok, v.errors)
                self.assertEqual((v.model_version, v.model_sample_rate, v.index_dim), ("v2", 40000, 768))


if __name__ == "__main__":
    unittest.main()
