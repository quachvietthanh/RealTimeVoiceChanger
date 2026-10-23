"""Integration test: the real ContentVec files pass their pinned SHA-256 and build the same model as Applio.

CPU only, read-only. Skips a copy that is absent on this machine (assets are not in Git).
"""

import sys
import unittest
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch

from voice_changer.conversion.rvc import contentvec as cv_mod
from voice_changer.conversion.rvc.contentvec import load_contentvec

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
COPIES = (
    REPO_ROOT / "third_party" / "applio" / "rvc" / "models" / "embedders" / "contentvec",  # the one inference uses
    REPO_ROOT / "models" / "checkpoints" / "contentvec",
)


class RealContentVecAssetTests(unittest.TestCase):
    def test_verified_load_matches_applios_path_based_load(self):
        path = COPIES[0]
        if not (path / "pytorch_model.bin").is_file():
            self.skipTest(f"{path} is not on this machine")
        ours = load_contentvec(path)
        # Reference: what Applio's load_embedding() ends with (transformers reads the files itself).
        ref = cv_mod._hubert_class().from_pretrained(str(path)).eval()
        self.assertEqual(ours.config.to_dict(), ref.config.to_dict())
        ref_state, our_state = ref.state_dict(), ours.state_dict()
        self.assertEqual(ref_state.keys(), our_state.keys())
        for k in ref_state:
            self.assertTrue(torch.equal(ref_state[k], our_state[k]), k)
        x = torch.randn(1, 16000, generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            self.assertTrue(torch.equal(ours(x)["last_hidden_state"], ref(x)["last_hidden_state"]))

    def test_sync_source_copy_also_verifies(self):
        path = COPIES[1]
        if not (path / "pytorch_model.bin").is_file():
            self.skipTest(f"{path} is not on this machine")
        self.assertEqual(load_contentvec(path).config.hidden_size, 768)


if __name__ == "__main__":
    unittest.main()
