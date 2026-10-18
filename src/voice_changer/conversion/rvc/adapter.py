"""RVC engine adapter using Applio Core v3.2.8.

Isolates Applio's RVC inference engine behind the VoiceConverter protocol.
"""

from __future__ import annotations

import contextlib
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from voice_changer.conversion.base import ConversionConfig, VoiceConverter


class RVCAdapter(VoiceConverter):
    """Adapter bridging RealTimeVoiceChanger to Applio Core v3.2.8.

    Keeps Applio/RVC dependencies isolated from the rest of the application.
    """

    def __init__(
        self,
        device: str = "cuda",
        checkpoints_dir: Optional[str] = None,
    ) -> None:
        self.device = device
        self._model_loaded = False
        self._model_path: Optional[str] = None
        self._index_path: Optional[str] = None
        self._cached_index: Optional[Any] = None
        self._cached_big_npy: Optional[Any] = None
        self._model_info: Dict[str, Any] = {}
        self._applio_vc: Optional[Any] = None

        # Resolve paths
        self._repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
        self._applio_root = self._repo_root / "third_party" / "applio"
        self._checkpoints_dir = (
            Path(checkpoints_dir).resolve()
            if checkpoints_dir
            else self._repo_root / "models" / "checkpoints"
        )

        # Ensure third_party/applio is resolvable on sys.path
        if str(self._applio_root) not in sys.path:
            sys.path.insert(0, str(self._applio_root))

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded

    @property
    def model_info(self) -> Dict[str, Any]:
        """Metadata of the currently loaded RVC voice model."""
        return self._model_info

    def _sync_auxiliary_assets(self) -> None:
        """Synchronize predictor and embedder assets from project checkpoints into Applio layout."""
        applio_rmvpe = self._applio_root / "rvc" / "models" / "predictors" / "rmvpe.pt"
        local_rmvpe = self._checkpoints_dir / "rmvpe.pt"
        if not applio_rmvpe.exists() and local_rmvpe.exists():
            applio_rmvpe.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(local_rmvpe), str(applio_rmvpe))

        applio_cv_dir = self._applio_root / "rvc" / "models" / "embedders" / "contentvec"
        local_cv_dir = self._checkpoints_dir / "contentvec"

        if not (applio_cv_dir / "pytorch_model.bin").exists():
            if (local_cv_dir / "pytorch_model.bin").exists():
                applio_cv_dir.mkdir(parents=True, exist_ok=True)
                for f in ["pytorch_model.bin", "config.json"]:
                    src_f = local_cv_dir / f
                    if src_f.exists():
                        shutil.copy2(str(src_f), str(applio_cv_dir / f))
            elif (self._checkpoints_dir / "pytorch_model.bin").exists():
                applio_cv_dir.mkdir(parents=True, exist_ok=True)
                for f in ["pytorch_model.bin", "config.json"]:
                    src_f = self._checkpoints_dir / f
                    if src_f.exists():
                        shutil.copy2(str(src_f), str(applio_cv_dir / f))

    def _ensure_applio_initialized(self) -> Any:
        """Lazily import and initialize Applio's VoiceConverter."""
        if self._applio_vc is None:
            if not self._applio_root.exists():
                raise FileNotFoundError(
                    f"Applio repository not found at {self._applio_root}. "
                    "Ensure Applio v3.2.8 is cloned into third_party/applio."
                )
            self._sync_auxiliary_assets()
            try:
                with contextlib.chdir(self._applio_root):
                    from rvc.infer.infer import VoiceConverter as ApplioVoiceConverter

                    self._applio_vc = ApplioVoiceConverter()
            except ImportError as err:
                raise ImportError(
                    f"Failed to import Applio inference core: {err}. "
                    "Verify that librosa, faiss-cpu, and other inference dependencies are installed."
                ) from err
        return self._applio_vc

    def load_model(self, model_path: str, index_path: Optional[str] = None) -> None:
        """Load RVC voice model weights (.pth) and optional Faiss feature index (.index)."""
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"RVC model weight file not found: {model_path}")

        # Validate PyTorch checkpoint integrity
        import torch

        try:
            cpt = torch.load(str(model_file.resolve()), map_location="cpu", weights_only=True)
        except Exception as err:
            raise ValueError(f"Failed to load PyTorch checkpoint '{model_path}': {err}") from err

        if not isinstance(cpt, dict) or "weight" not in cpt or "config" not in cpt:
            raise ValueError(
                f"Invalid RVC model checkpoint '{model_path}'. "
                "Checkpoint must be a dictionary containing 'weight' and 'config' keys."
            )

        self._model_info = {
            "version": cpt.get("version", "v1"),
            "f0": cpt.get("f0", 1),
            "target_sr": cpt["config"][-1] if isinstance(cpt["config"], (list, tuple)) else None,
            "vocoder": cpt.get("vocoder", "HiFi-GAN"),
        }

        # Validate index path if provided
        if index_path is not None and index_path != "":
            idx_file = Path(index_path)
            if not idx_file.exists():
                raise FileNotFoundError(f"RVC feature index file not found: {index_path}")
            self._index_path = str(idx_file.resolve())
        else:
            self._index_path = ""

        self._model_path = str(model_file.resolve())

        # Sync and verify auxiliary assets before loading pipeline
        self._sync_auxiliary_assets()
        applio_rmvpe = self._applio_root / "rvc" / "models" / "predictors" / "rmvpe.pt"
        if not applio_rmvpe.exists():
            raise FileNotFoundError(
                f"RMVPE pitch model 'rmvpe.pt' not found. "
                f"Place 'rmvpe.pt' into '{self._checkpoints_dir}' or '{applio_rmvpe.parent}' "
                "before loading the voice model. Refer to docs/model-setup.md."
            )

        # Initialize Applio VoiceConverter and preload model checkpoint inside Applio directory
        vc = self._ensure_applio_initialized()
        with contextlib.chdir(self._applio_root):
            vc.get_vc(self._model_path, sid=0)
            if not vc.hubert_model:
                vc.load_hubert("contentvec")

        # Preload and cache Faiss feature index into RAM to eliminate disk reload overhead (~165 ms/chunk)
        if self._index_path:
            import faiss

            self._cached_index = faiss.read_index(self._index_path)
            self._cached_big_npy = self._cached_index.reconstruct_n(0, self._cached_index.ntotal)
        else:
            self._cached_index = None
            self._cached_big_npy = None

        # Configure low-latency streaming edge padding (0.1s instead of 1.0s/3.0s default)
        model_sr = self._model_info.get("target_sr", 40000) or 40000
        if hasattr(vc.vc, "set_padding"):
            vc.vc.set_padding(pad_seconds=0.1, tgt_sr=model_sr)
        else:
            vc.vc.t_pad = int(16000 * 0.1)
            vc.vc.t_pad_tgt = int(model_sr * 0.1)
            vc.vc.t_pad2 = vc.vc.t_pad * 2

        self._model_loaded = True

    def convert_chunk(
        self,
        audio_data: Any,
        config: Optional[ConversionConfig] = None,
        input_sr: int = 40000,
        target_sr: Optional[int] = None,
    ) -> Any:
        """Convert an in-memory audio chunk (float32 numpy array).

        Args:
            audio_data: 1D NumPy array of float32 samples.
            config: Optional conversion configuration.
            input_sr: Sample rate of incoming audio chunk (default 40000 Hz).
            target_sr: Target sample rate for output (defaults to model's target_sr).

        Returns:
            Converted 1D float32 NumPy array.
        """
        if not self._model_loaded or self._model_path is None:
            raise RuntimeError("Model is not loaded. Call load_model() before converting.")

        import numpy as np
        import torch

        cfg = config or ConversionConfig()
        vc = self._ensure_applio_initialized()

        audio = np.asarray(audio_data, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if len(audio) == 0:
            return np.zeros(0, dtype=np.float32)

        # Resample to 16000 Hz for HuBERT and RMVPE feature extraction
        if input_sr != 16000:
            import librosa
            audio_16k = librosa.resample(audio, orig_sr=input_sr, target_sr=16000)
        else:
            audio_16k = audio

        audio_max = np.abs(audio_16k).max() / 0.95
        if audio_max > 1.0:
            audio_16k = audio_16k / audio_max

        # Ensure embedder is loaded
        if not vc.hubert_model:
            with contextlib.chdir(self._applio_root):
                vc.load_hubert("contentvec")

        if self._cached_index is not None and cfg.index_rate > 0:
            index_input: Any = (self._cached_index, self._cached_big_npy)
        else:
            index_input = self._index_path or ""
        model_sr = self._model_info.get("target_sr", 40000) or 40000
        final_sr = target_sr or model_sr
        expected_len = round(len(audio) * (final_sr / input_sr))

        # Symmetrical 5ms reflection padding (80 samples at 16 kHz) to prevent HuBERT convolutional
        # encoder and HiFi-GAN from dropping boundary frames due to receptive field truncation.
        audio_16k_padded = np.pad(audio_16k, (80, 80), mode="reflect")

        with torch.no_grad():
            with contextlib.chdir(self._applio_root):
                opt = vc.vc.pipeline(
                    model=vc.hubert_model,
                    net_g=vc.net_g,
                    sid=0,
                    audio=audio_16k_padded,
                    pitch=round(cfg.pitch_shift),
                    f0_method=cfg.f0_method,
                    file_index=index_input,
                    index_rate=cfg.index_rate,
                    pitch_guidance=vc.use_f0,
                    filter_radius=cfg.filter_radius,
                    volume_envelope=1.0,
                    version=vc.version,
                    protect=cfg.protect_rate,
                    hop_length=128,
                    f0_autotune=False,
                    f0_autotune_strength=1.0,
                    f0_file=None,
                )

        output_audio = opt.astype(np.float32)

        if final_sr != model_sr:
            import librosa
            output_audio = librosa.resample(output_audio, orig_sr=model_sr, target_sr=final_sr)

        # Enforce exact temporal length alignment to prevent continuous ring buffer drift
        if len(output_audio) > expected_len:
            output_audio = output_audio[:expected_len]
        elif len(output_audio) < expected_len:
            pad_needed = expected_len - len(output_audio)
            output_audio = np.pad(output_audio, (0, pad_needed), mode="edge")

        return output_audio

    def convert(self, audio_chunk: bytes, config: ConversionConfig) -> bytes:
        """Convert a single audio chunk (PCM float32 bytes)."""
        import numpy as np

        arr = np.frombuffer(audio_chunk, dtype=np.float32)
        out_arr = self.convert_chunk(arr, config)
        return out_arr.tobytes()

    def convert_file(
        self,
        input_audio_path: str,
        output_audio_path: str,
        config: Optional[ConversionConfig] = None,
    ) -> Dict[str, Any]:
        """Perform offline voice conversion on an input WAV file.

        Args:
            input_audio_path: Path to the clean input WAV file.
            output_audio_path: Destination path for the converted WAV file.
            config: Optional conversion configuration (pitch, f0_method, index_rate, etc.).

        Returns:
            Dictionary containing execution statistics (elapsed_time, input_path, output_path).
        """
        if not self._model_loaded or self._model_path is None:
            raise RuntimeError("Model is not loaded. Call load_model() before converting.")

        in_file = Path(input_audio_path).resolve()
        if not in_file.exists():
            raise FileNotFoundError(f"Input audio file not found: {input_audio_path}")

        cfg = config or ConversionConfig()
        vc = self._ensure_applio_initialized()

        out_file = Path(output_audio_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        start_time = time.perf_counter()

        with contextlib.chdir(self._applio_root):
            vc.convert_audio(
                audio_input_path=str(in_file.resolve()),
                audio_output_path=str(out_file.resolve()),
                model_path=self._model_path,
                index_path=self._index_path or "",
                pitch=round(cfg.pitch_shift),
                f0_method=cfg.f0_method,
                index_rate=cfg.index_rate,
                filter_radius=cfg.filter_radius,
                protect=cfg.protect_rate,
                export_format="WAV",
                clean_audio=False,
                post_process=False,
            )

        if not out_file.exists():
            raise RuntimeError(
                f"Applio convert_audio completed without generating output file at '{out_file}'."
            )

        elapsed_time = time.perf_counter() - start_time

        return {
            "elapsed_time_seconds": elapsed_time,
            "input_audio_path": str(in_file.resolve()),
            "output_audio_path": str(out_file.resolve()),
            "pitch_shift": cfg.pitch_shift,
            "f0_method": cfg.f0_method,
            "index_rate": cfg.index_rate,
        }

    def reset(self) -> None:
        """Reset internal buffer states."""
        pass

