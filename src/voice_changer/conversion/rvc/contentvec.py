"""Verified loading of the ContentVec embedder (pytorch_model.bin + config.json).

Applio's load_embedding() downloads a missing file from an unpinned URL, then calls
HubertModelWithFinalProj.from_pretrained(<directory>), which resolves and re-reads the files itself
(and would prefer a model.safetensors placed next to them); nothing checks a hash. load_contentvec()
reads exactly the two files once, checks each pinned SHA-256 on those bytes, deserializes the weights
with torch.load(weights_only=True) and builds the model from the in-memory state dict, so transformers
never opens a file. There is no download and no fallback.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import torch

from voice_changer.conversion.rvc.assets import read_verified_bytes

_APPLIO_ROOT = Path(__file__).resolve().parents[4] / "third_party" / "applio"
if str(_APPLIO_ROOT) not in sys.path:
    sys.path.insert(0, str(_APPLIO_ROOT))

# IAHispano/Applio @ e908bdd, Resources/embedders/contentvec/. Same values in docs/model-setup.md and
# scripts/download_auxiliary_assets.py.
CONTENTVEC_WEIGHTS_SHA256 = "d8dd400e054ddf4e6be75dab5a2549db748cc99e756a097c496c099f65a4854e"
CONTENTVEC_CONFIG_SHA256 = "2ddde063b795d38d9051a7215a092fecf4cfe148b54251e38de51d88d356898b"
ARCHITECTURE = "HubertModelWithFinalProj"


def _hubert_class() -> Any:
    """Applio's HubertModelWithFinalProj (rvc/lib/utils.py), imported without a side effect on Applio."""
    if "rvc.lib.utils" not in sys.modules:
        # rvc.lib.utils records os.getcwd() at import as Applio's root directory.
        with contextlib.chdir(_APPLIO_ROOT):
            import rvc.lib.utils  # noqa: F401
    return sys.modules["rvc.lib.utils"].HubertModelWithFinalProj


def load_contentvec(
    model_dir: Any,
    weights_sha256: str = CONTENTVEC_WEIGHTS_SHA256,
    config_sha256: str = CONTENTVEC_CONFIG_SHA256,
) -> Any:
    """Build HubertModelWithFinalProj (CPU, float32, eval mode) from hash-verified files in model_dir.

    Equivalent to Applio's HubertModelWithFinalProj.from_pretrained(model_dir) for the pinned files
    (same weights and outputs; checked in tests/unit/test_contentvec_loading.py and against the real asset).

    Raises:
        FileNotFoundError: config.json or pytorch_model.bin is missing.
        ValueError: A hash is missing or does not match, or a file is malformed or has the wrong layout.
    """
    if not weights_sha256 or not config_sha256:
        raise ValueError("expected SHA-256 values for ContentVec weights and config are required")
    model_dir = Path(model_dir)
    config_bytes = read_verified_bytes(model_dir / "config.json", config_sha256, "ContentVec config")
    weights_bytes = read_verified_bytes(model_dir / "pytorch_model.bin", weights_sha256, "ContentVec weights")

    try:
        config_dict = json.loads(config_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise ValueError(f"ContentVec config '{model_dir / 'config.json'}' is not valid JSON: {err}") from err
    if not isinstance(config_dict, dict) or ARCHITECTURE not in (config_dict.get("architectures") or []):
        raise ValueError(f"ContentVec config '{model_dir / 'config.json'}' does not describe {ARCHITECTURE}")

    try:
        state = torch.load(io.BytesIO(weights_bytes), map_location="cpu", weights_only=True)
    except Exception as err:
        raise ValueError(f"ContentVec weights '{model_dir / 'pytorch_model.bin'}' cannot be loaded "
                         f"weights-only: {err}") from err
    del weights_bytes
    if not isinstance(state, dict) or not state or not all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
    ):
        raise ValueError(f"ContentVec weights '{model_dir / 'pytorch_model.bin'}' are not a state dict of tensors")

    from transformers import HubertConfig

    config = HubertConfig.from_dict(config_dict)
    try:
        model, info = _hubert_class().from_pretrained(
            None, config=config, state_dict=state, output_loading_info=True
        )
    except Exception as err:
        raise ValueError(f"ContentVec weights do not fit {ARCHITECTURE}: {err}") from err
    # from_pretrained only warns about missing keys and initializes them randomly; refuse instead.
    problems = {k: v for k, v in info.items() if v}
    if problems:
        raise ValueError(f"ContentVec weights do not match the {ARCHITECTURE} layout: {problems}")
    model.config._name_or_path = str(model_dir)  # what from_pretrained(model_dir) would record
    return model
