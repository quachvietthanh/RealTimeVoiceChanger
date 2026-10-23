"""Static inspection of RVC model and Faiss index files, without loading anything onto the GPU.

Used to validate a voice profile before it is loaded: the checkpoint must be an RVC inference
export, and the index dimension must match the embedder dimension the checkpoint expects.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Embedder output dimension by RVC version (Applio rvc/infer/infer.py: text_enc_hidden_dim).
EMBEDDER_DIM_BY_VERSION = {"v1": 256, "v2": 768}
# RVCAdapter always loads ContentVec; a model trained on another embedder would run but sound wrong.
SUPPORTED_EMBEDDERS = ("contentvec",)
# Model output rate verified in streaming. Other rates run (the adapter resamples each chunk to the
# stream rate) but have not been measured for quality or inference time.
VERIFIED_MODEL_SAMPLE_RATE = 40000


@dataclass(frozen=True)
class CheckpointInfo:
    """Metadata read from an RVC inference checkpoint (.pth)."""

    version: str
    f0: int
    sample_rate: int
    vocoder: str
    embedder_dim: int
    speaker_count: int
    # Feature extractor the model was trained with, as recorded by Applio 3.x exports
    # ("contentvec", "chinese-hubert-base", ...). None for older exports that do not record it.
    embedder_model: Optional[str] = None


@dataclass(frozen=True)
class IndexInfo:
    """Metadata read from a Faiss retrieval index (.index)."""

    dim: int
    ntotal: int
    is_trained: bool


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def read_verified_bytes(path: Path, expected_sha256: Optional[str], kind: str) -> bytes:
    """Read a file once and check its SHA-256 on those same bytes.

    The caller deserializes the returned bytes, never the path, so a file replaced after an earlier
    validation is either rejected here or not loaded at all.

    Raises:
        ValueError: The hash does not match.
    """
    data = path.read_bytes()
    if expected_sha256:
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected_sha256.lower():
            raise ValueError(
                f"{kind} file '{path}' does not match its registered SHA-256 "
                f"(expected {expected_sha256[:12]}…, read {actual[:12]}…)"
            )
    return data


def inspect_checkpoint(path: Path) -> CheckpointInfo:
    """Read an RVC checkpoint on the CPU.

    Raises:
        ValueError: If the file is not a loadable RVC inference checkpoint.
    """
    import torch

    try:
        cpt = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as err:
        raise ValueError(f"not a loadable PyTorch checkpoint: {err}") from err
    if not isinstance(cpt, dict) or "weight" not in cpt or "config" not in cpt:
        raise ValueError("not an RVC inference checkpoint (missing 'weight' or 'config')")
    config = cpt["config"]
    if not isinstance(config, (list, tuple)) or not config:
        raise ValueError("checkpoint 'config' is not a list")
    version = str(cpt.get("version", "v1"))
    if version not in EMBEDDER_DIM_BY_VERSION:
        raise ValueError(f"unsupported RVC version '{version}'")
    emb = cpt["weight"].get("emb_g.weight")
    return CheckpointInfo(
        version=version,
        f0=int(cpt.get("f0", 1)),
        sample_rate=int(config[-1]),
        vocoder=str(cpt.get("vocoder", "HiFi-GAN")),
        embedder_dim=EMBEDDER_DIM_BY_VERSION[version],
        speaker_count=int(emb.shape[0]) if emb is not None else 0,
        embedder_model=str(cpt["embedder_model"]) if cpt.get("embedder_model") else None,
    )


def inspect_index(path: Path) -> IndexInfo:
    """Read a Faiss index header and size.

    Raises:
        ValueError: If the file cannot be read as a Faiss index.
    """
    import faiss

    try:
        index = faiss.read_index(str(path))
    except Exception as err:
        raise ValueError(f"not a readable Faiss index: {err}") from err
    return IndexInfo(dim=int(index.d), ntotal=int(index.ntotal), is_trained=bool(index.is_trained))


def check_pair(checkpoint: CheckpointInfo, index: Optional[IndexInfo]) -> list[str]:
    """Return problems that make the checkpoint and index unusable together (empty list if none)."""
    problems = []
    if index is not None:
        if index.dim != checkpoint.embedder_dim:
            problems.append(
                f"index dimension {index.dim} does not match the {checkpoint.version} checkpoint's "
                f"embedder dimension {checkpoint.embedder_dim}"
            )
        if index.ntotal <= 0 or not index.is_trained:
            problems.append("index is empty or untrained")
    return problems


def check_runtime(checkpoint: CheckpointInfo) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) about running the checkpoint with RVCAdapter in the streaming pipeline."""
    errors, warnings = [], []
    if checkpoint.embedder_model is not None and checkpoint.embedder_model not in SUPPORTED_EMBEDDERS:
        errors.append(f"checkpoint was trained with embedder '{checkpoint.embedder_model}'; the adapter only "
                      f"loads {', '.join(SUPPORTED_EMBEDDERS)}")
    if checkpoint.sample_rate != VERIFIED_MODEL_SAMPLE_RATE:
        warnings.append(f"model outputs {checkpoint.sample_rate} Hz; it is resampled to the stream rate on every "
                        f"chunk, which is untested (run scripts/service_simulation.py before relying on it)")
    return errors, warnings
