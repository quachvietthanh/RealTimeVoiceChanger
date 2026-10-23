"""Verified, weights-only loading of the RMVPE pitch model (rmvpe.pt).

Applio's RMVPE0Predictor.__init__ calls torch.load(model_path) without weights_only, which under
PyTorch 2.5 unpickles arbitrary objects (code execution from a crafted file), and nothing checks the
file's hash at runtime. VerifiedRMVPE0Predictor is a drop-in replacement with the same constructor:
it reads rmvpe.pt once, checks the pinned SHA-256 on those bytes and deserializes them with
torch.load(weights_only=True). There is no fallback to an unrestricted load.

RVCAdapter installs it as rvc.infer.pipeline.RMVPE0Predictor, the name Applio's Pipeline uses.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from voice_changer.conversion.rvc.assets import read_verified_bytes

_APPLIO_ROOT = Path(__file__).resolve().parents[4] / "third_party" / "applio"
if str(_APPLIO_ROOT) not in sys.path:
    sys.path.insert(0, str(_APPLIO_ROOT))

from rvc.lib.predictors.RMVPE import E2E, N_CLASS, N_MELS, MelSpectrogram, RMVPE0Predictor  # noqa: E402

# IAHispano/Applio @ e908bdd, Resources/predictors/rmvpe.pt (181,184,272 bytes). Same value in
# docs/model-setup.md and scripts/download_auxiliary_assets.py.
RMVPE_SHA256 = "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193"


def load_rmvpe_state_dict(path: Any, expected_sha256: str = RMVPE_SHA256) -> dict[str, torch.Tensor]:
    """Read rmvpe.pt once, verify its SHA-256 on those bytes and deserialize them weights-only.

    Raises:
        ValueError: Hash mismatch, a file torch cannot load weights-only, or not a tensor state dict.
    """
    if not expected_sha256:
        raise ValueError("an expected SHA-256 is required to load rmvpe.pt")
    path = Path(path)
    data = read_verified_bytes(path, expected_sha256, "RMVPE pitch model")
    try:
        state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except Exception as err:
        raise ValueError(f"RMVPE pitch model '{path}' cannot be loaded weights-only: {err}") from err
    if not isinstance(state, dict) or not state or not all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
    ):
        raise ValueError(f"RMVPE pitch model '{path}' is not a state dict of tensors")
    return state


class VerifiedRMVPE0Predictor(RMVPE0Predictor):
    """RMVPE0Predictor whose checkpoint is hash-verified and loaded weights-only."""

    def __init__(self, model_path: Any, is_half: bool, device: Optional[str] = None,
                 expected_sha256: str = RMVPE_SHA256) -> None:
        # Mirrors RMVPE0Predictor.__init__ of the pinned Applio v3.2.8, with the verified state dict in
        # place of its torch.load(model_path). load_state_dict is strict: a wrong layout fails here.
        state = load_rmvpe_state_dict(model_path, expected_sha256)
        model = E2E(4, 1, (2, 2))
        model.load_state_dict(state)
        model.eval()
        if is_half:
            model = model.half()
        self.model = model
        self.resample_kernel = {}
        self.is_half = is_half
        self.device = device
        self.mel_extractor = MelSpectrogram(
            is_half, N_MELS, 16000, 1024, 160, None, 30, 8000
        ).to(device)
        self.model = self.model.to(device)
        cents_mapping = 20 * np.arange(N_CLASS) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))
