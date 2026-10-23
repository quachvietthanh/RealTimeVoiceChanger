"""RVC engine adapter using Applio Core v3.2.8.

Isolates Applio's RVC inference engine behind the VoiceConverter protocol.
"""

from __future__ import annotations

import contextlib
import io
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from voice_changer.conversion.base import ConversionConfig, VoiceConverter
from voice_changer.conversion.rvc.assets import read_verified_bytes
from voice_changer.conversion.rvc.contentvec import load_contentvec

# CPU threads for PyTorch and Faiss when inference runs on CUDA. The CPU-side work per chunk is small,
# and with the default pools (torch: one thread per core, Faiss: OpenMP) the idle threads spin-wait between
# calls: measured 4.3 CPU cores busy while streaming vs 0.56 with one thread each, with the same
# inference time (output/noise_suppression_2026-10-21/README.md, "CPU use").
CUDA_CPU_THREADS = 1


def limit_cpu_threads(n: int) -> None:
    """Set PyTorch intra-op and Faiss OpenMP thread counts (process-wide)."""
    import torch

    torch.set_num_threads(n)
    try:
        import faiss

        faiss.omp_set_num_threads(n)
    except ImportError:
        pass


def _refuse_path_load(weight_root: Any) -> None:
    """Replaces Applio's VoiceConverter.load_model on the adapter's instance.

    Applio's version calls torch.load() without weights_only, which under PyTorch 2.5 unpickles
    arbitrary objects (code execution from a crafted .pth). The adapter installs checkpoints it has
    verified and loaded with weights_only=True instead; any other path into Applio's loader fails here.
    """
    raise RuntimeError(
        f"refusing Applio's unrestricted torch.load of '{weight_root}'; load models through "
        "RVCAdapter.load_model()"
    )


def _refuse_embedder_load(embedder_model: Any, embedder_model_custom: Any = None) -> None:
    """Replaces Applio's VoiceConverter.load_hubert on the adapter's instance.

    Applio's version downloads a missing ContentVec from an unpinned URL and loads the files without a
    hash check. The adapter installs a verified embedder instead (RVCAdapter._install_verified_embedder).
    """
    raise RuntimeError(
        f"refusing Applio's unverified embedder load of '{embedder_model}'; the adapter installs a "
        "hash-verified ContentVec"
    )


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
        # (left, right) reflection padding in 16 kHz samples applied before feature extraction
        self.pad_16k: tuple[int, int] = (80, 80)

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
                    import importlib

                    from rvc.infer.infer import VoiceConverter as ApplioVoiceConverter

                    from voice_changer.conversion.rvc.rmvpe import VerifiedRMVPE0Predictor

                    # Applio's Pipeline builds its pitch predictor from this module-level name; the
                    # replacement verifies rmvpe.pt's SHA-256 and loads it weights-only (see rmvpe.py).
                    importlib.import_module("rvc.infer.pipeline").RMVPE0Predictor = VerifiedRMVPE0Predictor
                    self._applio_vc = ApplioVoiceConverter()
                self._applio_vc.load_model = _refuse_path_load
                self._applio_vc.load_hubert = _refuse_embedder_load
            except ImportError as err:
                raise ImportError(
                    f"Failed to import Applio inference core: {err}. "
                    "Verify that librosa, faiss-cpu, and other inference dependencies are installed."
                ) from err
        return self._applio_vc

    def _install_verified_embedder(self, vc: Any) -> None:
        """Load ContentVec from hash-verified bytes and set it up as Applio's load_hubert() would."""
        model = load_contentvec(self._applio_root / "rvc" / "models" / "embedders" / "contentvec")
        model.to(vc.config.device)
        model = model.half() if vc.config.is_half else model.float()
        model.eval()
        vc.hubert_model = model
        vc.last_embedder_model = "contentvec"  # Applio's convert_audio() then keeps this embedder

    def load_model(
        self,
        model_path: str,
        index_path: Optional[str] = None,
        *,
        model_sha256: Optional[str] = None,
        index_sha256: Optional[str] = None,
    ) -> None:
        """Load RVC voice model weights (.pth) and optional Faiss feature index (.index).

        Each file is read once; when a SHA-256 is given it is checked on exactly the bytes that are
        then deserialized. The checkpoint is unpickled only with torch.load(weights_only=True).

        The adapter is marked unloaded first, so a failed load never leaves the previous model's
        state (path, index, model_info) reported as loaded.

        Raises:
            FileNotFoundError: A file is missing.
            ValueError: A hash does not match, or the checkpoint is not a weights-only RVC checkpoint.
        """
        self._model_loaded = False
        model_file = Path(model_path)
        if self.device.startswith("cuda") and CUDA_CPU_THREADS:
            import torch

            if torch.cuda.is_available():
                limit_cpu_threads(CUDA_CPU_THREADS)
        if not model_file.exists():
            raise FileNotFoundError(f"RVC model weight file not found: {model_path}")

        # Validate PyTorch checkpoint integrity
        import torch

        model_bytes = read_verified_bytes(model_file, model_sha256, "RVC model")
        try:
            cpt = torch.load(io.BytesIO(model_bytes), map_location="cpu", weights_only=True)
        except Exception as err:
            raise ValueError(f"Failed to load PyTorch checkpoint '{model_path}': {err}") from err

        if not isinstance(cpt, dict) or "weight" not in cpt or "config" not in cpt:
            raise ValueError(
                f"Invalid RVC model checkpoint '{model_path}'. "
                "Checkpoint must be a dictionary containing 'weight' and 'config' keys."
            )

        model_info = {
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
            resolved_index = str(idx_file.resolve())
            index_bytes: Optional[bytes] = read_verified_bytes(idx_file, index_sha256, "RVC index")
        else:
            resolved_index = ""
            index_bytes = None

        self._model_info = model_info
        self._index_path = resolved_index
        self._model_path = str(model_file.resolve())
        self._cached_index = None
        self._cached_big_npy = None

        # Sync and verify auxiliary assets before loading pipeline
        self._sync_auxiliary_assets()
        applio_rmvpe = self._applio_root / "rvc" / "models" / "predictors" / "rmvpe.pt"
        if not applio_rmvpe.exists():
            raise FileNotFoundError(
                f"RMVPE pitch model 'rmvpe.pt' not found. "
                f"Place 'rmvpe.pt' into '{self._checkpoints_dir}' or '{applio_rmvpe.parent}' "
                "before loading the voice model. Refer to docs/model-setup.md."
            )

        # Initialize Applio VoiceConverter and install the checkpoint read above. Applio's own get_vc()
        # would re-read the file with an unrestricted torch.load (see _refuse_path_load).
        vc = self._ensure_applio_initialized()
        with contextlib.chdir(self._applio_root):
            vc.cpt = cpt
            vc.setup_network()
            vc.setup_vc_instance()
            from voice_changer.conversion.rvc.rmvpe import VerifiedRMVPE0Predictor

            if not isinstance(getattr(vc.vc, "model_rmvpe", None), VerifiedRMVPE0Predictor):
                raise RuntimeError("Applio's pitch predictor was not built by VerifiedRMVPE0Predictor; "
                                   "refusing a model whose rmvpe.pt was loaded without verification")
            vc.loaded_model = self._model_path  # convert_file() -> get_vc() then skips reloading
            if not vc.hubert_model:
                self._install_verified_embedder(vc)

        # Preload and cache Faiss feature index into RAM to eliminate disk reload overhead (~165 ms/chunk)
        if index_bytes is not None:
            import faiss
            import numpy as np

            self._cached_index = faiss.deserialize_index(np.frombuffer(index_bytes, dtype=np.uint8))
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
            self._install_verified_embedder(vc)

        if self._cached_index is not None and cfg.index_rate > 0:
            index_input: Any = (self._cached_index, self._cached_big_npy)
        else:
            index_input = self._index_path or ""
        model_sr = self._model_info.get("target_sr", 40000) or 40000
        final_sr = target_sr or model_sr
        expected_len = round(len(audio) * (final_sr / input_sr))

        # Symmetrical 5ms reflection padding (80 samples at 16 kHz) to prevent HuBERT convolutional
        # encoder and HiFi-GAN from dropping boundary frames due to receptive field truncation.
        audio_16k_padded = np.pad(audio_16k, self.pad_16k, mode="reflect")

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

    def unload(self) -> None:
        """Release the voice model, embedder and cached index, and return GPU memory.

        Safe to call when nothing is loaded. After unload() the adapter can load another model.
        """
        self._model_loaded = False
        self._model_path = None
        self._index_path = None
        self._cached_index = None
        self._cached_big_npy = None
        self._model_info = {}
        vc = self._applio_vc
        self._applio_vc = None
        if vc is not None:
            try:
                vc.cleanup_model()
            except Exception:  # cleanup is best effort; references are dropped below anyway
                pass
            del vc
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
