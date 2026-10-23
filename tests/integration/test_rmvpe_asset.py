"""Integration test: the real rmvpe.pt passes the pinned SHA-256 and loads weights-only into a working predictor.

CPU only, read-only. Skips a copy that is absent on this machine (assets are not in Git).
"""

import sys
import unittest
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import numpy as np

from voice_changer.conversion.rvc.rmvpe import VerifiedRMVPE0Predictor, load_rmvpe_state_dict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
COPIES = (
    REPO_ROOT / "models" / "checkpoints" / "rmvpe.pt",
    REPO_ROOT / "third_party" / "applio" / "rvc" / "models" / "predictors" / "rmvpe.pt",  # the one inference uses
)


class RealRmvpeAssetTests(unittest.TestCase):
    def test_copies_verify_and_load_weights_only(self):
        for path in COPIES:
            with self.subTest(path=str(path)):
                if not path.is_file():
                    self.skipTest(f"{path} is not on this machine")
                state = load_rmvpe_state_dict(path)
                self.assertGreater(len(state), 100)

    def test_predictor_tracks_the_pitch_of_a_tone(self):
        path = COPIES[1] if COPIES[1].is_file() else COPIES[0]
        if not path.is_file():
            self.skipTest("rmvpe.pt is not on this machine")
        predictor = VerifiedRMVPE0Predictor(str(path), is_half=False, device="cpu")
        t = np.arange(16000) / 16000
        f0 = predictor.infer_from_audio((0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32), thred=0.03)
        voiced = f0[f0 > 0]
        self.assertGreater(len(voiced), len(f0) // 2)
        self.assertAlmostEqual(float(np.median(voiced)), 220.0, delta=5.0)


if __name__ == "__main__":
    unittest.main()
