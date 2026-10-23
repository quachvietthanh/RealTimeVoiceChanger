"""In-process backend API for the desktop GUI.

VoiceChangerService is the only object the GUI needs. It owns the voice profile registry, at most
one loaded voice model, and at most one streaming session, and exposes them through typed,
thread-safe methods. The GUI never touches Applio, Faiss, PyTorch or sounddevice directly.

Lifecycle (EngineState):

    IDLE --select_profile--> LOADING --ok--> READY --start--> STARTING --ok--> RUNNING
                                 \\--fail--> IDLE         \\--fail--> READY        |
    READY --select_profile--> LOADING (the old model is released first)          |
    READY --unload_profile--> IDLE                                                |
    RUNNING --stop--> STOPPING --> READY                                          |
    RUNNING --(failure detected by get_status)--> ERROR --stop--> STOPPING --> READY

Only one voice model is in GPU memory at a time (4 GB VRAM target). Switching profiles while
streaming is refused: call stop(), select_profile(), start().
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any, Callable, Optional, Protocol

import numpy as np

from voice_changer.audio.devices import AudioDevice, AudioDeviceManager
from voice_changer.audio.noise_suppression import (
    NOISE_SUPPRESSION_MODES,
    NoiseSuppressionSettings,
    validate_noise_suppression,
)
from voice_changer.audio.pipeline import RealTimeVoiceChangerPipeline
from voice_changer.conversion.base import ConversionConfig, validate_conversion_config
from voice_changer.profiles.registry import ProfileRegistry, ProfileValidation, VoiceProfile
from voice_changer.service.errors import (
    AudioDeviceError,
    BackendError,
    ConfigurationError,
    ErrorInfo,
    InvalidStateError,
    ProfileError,
)

logger = logging.getLogger(__name__)

# Streaming sample rate. The pipeline converts at this rate end to end (40 kHz RVC v2 models).
STREAM_SAMPLE_RATE = 40000
CHUNK_DURATION_RANGE_S = (0.25, 2.0)
PREBUFFER_RANGE_S = (0.0, 1.0)
QUEUE_SIZE_RANGE = (1, 8)
CONTEXT_RANGE_S = (0.0, 1.0)
CROSSFADE_RANGE_S = (0.0, 0.2)
SOLA_SEARCH_RANGE_S = (0.0, 0.02)
# A session is reported as failed after this many inference errors in a row.
MAX_CONSECUTIVE_INFERENCE_ERRORS = 5
# How long start/select/unload wait for a worker that stop() left finishing a slow chunk.
WORKER_DRAIN_TIMEOUT_S = 10.0


class EngineState(str, Enum):
    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass(frozen=True)
class AudioSettings:
    """Audio route and buffering for one streaming session.

    Device ids are PortAudio indices from list_devices(); they are only valid for the current
    enumeration. None means the system default device.

    context_s / crossfade_s / sola_search_s control how consecutive chunks are joined (see
    audio/crossfade.py); all three 0 gives the old end-to-end joining with a seam at every chunk.
    """

    input_device_id: Optional[int] = None
    output_device_id: Optional[int] = None
    chunk_duration_s: float = 0.5
    prebuffer_s: float = 0.15
    max_queue_size: int = 3
    context_s: float = 0.2
    crossfade_s: float = 0.05
    sola_search_s: float = 0.01


@dataclass(frozen=True)
class NoiseSuppressionStatus:
    """Microphone noise suppression state (see set_noise_suppression)."""

    enabled: bool
    mode: str
    available_modes: tuple[str, ...]
    max_attenuation_db: float  # of the selected mode
    latency_ms: float  # fixed delay of the stage while streaming, on or off
    active: bool  # denoised audio is being output right now
    failed: bool  # disabled itself this session after repeated errors; set_noise_suppression() re-arms it
    last_error: str


@dataclass(frozen=True)
class DeviceInventory:
    inputs: tuple[AudioDevice, ...]
    outputs: tuple[AudioDevice, ...]
    default_input_id: Optional[int]
    default_output_id: Optional[int]


@dataclass(frozen=True)
class ProfileSummary:
    """Cheap description of a registered profile (no hashing or model reads).

    readiness: "no_model" (status planned, nothing to load), "files_missing" (model/index declared but
    not on this machine) or "files_present" (call validate_profile() to confirm it can be loaded).
    setup_hint says what to do when readiness is not "files_present" (empty otherwise).
    withdrawn_reason is non-empty only for a withdrawn profile, which list_profiles() returns only when
    include_withdrawn=True.
    """

    profile_id: str
    display_name: str
    description: str
    status: str
    language: str
    defaults: ConversionConfig
    files_present: bool
    is_active: bool
    notes: str
    readiness: str = "files_present"
    setup_hint: str = ""
    data_rights: str = "unspecified"
    withdrawn_reason: str = ""

    @property
    def selectable(self) -> bool:
        """The backend's answer to "offer a load action?": files on disk, not planned, not withdrawn.

        select_profile() can still refuse it (hash mismatch, incompatible files: validate_profile()).
        """
        return self.readiness == "files_present" and not self.withdrawn_reason


@dataclass(frozen=True)
class EngineStatus:
    state: EngineState
    profile_id: Optional[str]
    profile_status: Optional[str]
    model_loaded: bool
    config: Optional[ConversionConfig]
    audio: Optional[AudioSettings]
    session_id: int
    health_problems: tuple[str, ...]
    last_error: Optional[ErrorInfo]


class _Converter(Protocol):
    def load_model(self, model_path: str, index_path: Optional[str] = None, *,
                   model_sha256: Optional[str] = None, index_sha256: Optional[str] = None) -> None: ...
    def unload(self) -> None: ...
    def convert_chunk(self, audio_data: Any, config: Optional[ConversionConfig] = None,
                      input_sr: int = 40000, target_sr: Optional[int] = None) -> Any: ...


def _default_converter_factory() -> _Converter:
    from voice_changer.conversion.rvc.adapter import RVCAdapter

    return RVCAdapter()


class VoiceChangerService:
    """Thread-safe facade over profiles, model loading, devices and streaming.

    Long operations (select_profile, start, stop) are serialized and run on the caller's thread;
    call them off the GUI thread. get_status(), get_metrics() and list_profiles() return quickly
    and may be called from any thread at any time, including during a long operation.
    """

    def __init__(
        self,
        registry: Optional[ProfileRegistry] = None,
        device_manager: Optional[AudioDeviceManager] = None,
        converter_factory: Callable[[], _Converter] = _default_converter_factory,
        pipeline_factory: Callable[..., RealTimeVoiceChangerPipeline] = RealTimeVoiceChangerPipeline,
    ) -> None:
        self._registry = registry if registry is not None else ProfileRegistry.from_file()
        self._devices = device_manager
        self._converter_factory = converter_factory
        self._pipeline_factory = pipeline_factory

        self._op_lock = threading.Lock()  # serializes lifecycle operations (may be held for seconds)
        self._state_lock = threading.RLock()  # guards the fields below (held briefly; re-entrant for _fail)
        self._state = EngineState.IDLE
        self._profile: Optional[VoiceProfile] = None
        self._converter: Optional[_Converter] = None
        self._config: Optional[ConversionConfig] = None
        self._expected_inference_s: Optional[float] = None  # timed at load, for the default chunk length
        self._audio: Optional[AudioSettings] = None
        self._noise = NoiseSuppressionSettings()  # kept across sessions and profile switches
        self._pipeline: Optional[RealTimeVoiceChangerPipeline] = None
        self._draining: Optional[RealTimeVoiceChangerPipeline] = None  # stopped, worker still in inference
        self._session_id = 0
        self._last_metrics: Optional[dict[str, Any]] = None
        self._health: tuple[str, ...] = ()
        self._last_error: Optional[ErrorInfo] = None
        self._listeners: list[Callable[[EngineStatus], None]] = []

    # ----------------------------------------------------------------- internal helpers
    def _device_manager(self) -> AudioDeviceManager:
        if self._devices is None:
            self._devices = AudioDeviceManager()
        return self._devices

    def _snapshot(self) -> EngineStatus:
        return EngineStatus(
            state=self._state,
            profile_id=self._profile.profile_id if self._profile else None,
            profile_status=self._profile.status if self._profile else None,
            model_loaded=self._converter is not None,
            config=self._config,
            audio=self._audio,
            session_id=self._session_id,
            health_problems=self._health,
            last_error=self._last_error,
        )

    def _set(self, state: EngineState, error: Optional[BackendError] = None, **fields: Any) -> None:
        with self._state_lock:
            self._state = state
            for k, v in fields.items():
                setattr(self, f"_{k}", v)
            if error is not None:
                self._last_error = error.to_info()
            status = self._snapshot()
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(status)
            except Exception:
                logger.exception("status listener raised")

    def _require(self, *allowed: EngineState, action: str) -> None:
        if self._state not in allowed:
            raise InvalidStateError(
                "invalid_state",
                f"Cannot {action} while the engine is {self._state.value}.",
                hint=f"Allowed states: {', '.join(s.value for s in allowed)}.",
            )

    def _drain_previous_worker(self) -> None:
        """Make sure no worker of a stopped session still uses the converter (call under _op_lock)."""
        if self._draining is None:
            return
        if not self._draining.wait_for_worker(WORKER_DRAIN_TIMEOUT_S):
            raise self._fail(BackendError(
                "worker_busy", "The previous session is still finishing a conversion.",
                hint="Wait a few seconds and retry. If it persists, restart the application (GPU may be hung).",
            ))
        self._draining = None

    def _fail(self, err: BackendError) -> BackendError:
        """Record err as last_error (keeping the state) and return it for raising."""
        with self._state_lock:
            self._last_error = err.to_info()
        logger.error("%s [%s]: %s %s", err.category, err.code, err.message, err.details)
        return err

    # ----------------------------------------------------------------- events
    def add_status_listener(self, listener: Callable[[EngineStatus], None]) -> None:
        """Call `listener(status)` after every state transition.

        The listener runs on the thread that caused the transition (not the GUI thread); it must
        return quickly and marshal to the GUI thread itself. Exceptions are logged and ignored.
        """
        with self._state_lock:
            self._listeners.append(listener)

    def remove_status_listener(self, listener: Callable[[EngineStatus], None]) -> None:
        with self._state_lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    # ----------------------------------------------------------------- profiles
    @property
    def default_profile_id(self) -> Optional[str]:
        """The registry's default profile. Not loaded automatically."""
        return self._registry.default_profile_id

    @staticmethod
    def _readiness(profile: VoiceProfile) -> tuple[bool, str, str]:
        if profile.is_planned:
            return False, "no_model", (
                "No model yet. Prepare a dataset with verified usage rights, train it, then register the "
                "model and index in config/voice_profiles.json (docs/voice-profiles.md).")
        if not profile.files_present():
            return False, "files_missing", (
                "Model or index file not found on this machine. Copy the registered files to the paths "
                "in config/voice_profiles.json.")
        return True, "files_present", ""

    def list_profiles(self, include_withdrawn: bool = False) -> list[ProfileSummary]:
        """Registered profiles in registry order. Withdrawn profiles are left out unless include_withdrawn."""
        with self._state_lock:
            active = self._profile.profile_id if self._profile and self._converter else None
        summaries = []
        for p in self._registry.list_profiles():
            if p.is_withdrawn and not include_withdrawn and p.profile_id != active:
                continue
            present, readiness, hint = self._readiness(p)
            summaries.append(ProfileSummary(
                profile_id=p.profile_id,
                display_name=p.display_name,
                description=p.description,
                status=p.status,
                language=p.language,
                defaults=p.defaults,
                files_present=present,
                is_active=p.profile_id == active,
                notes=p.notes,
                readiness=readiness,
                setup_hint=hint,
                data_rights=p.data_rights,
                withdrawn_reason=p.withdrawn_reason,
            ))
        return summaries

    def resolve_profile_id(self, preferred: Optional[str]) -> Optional[str]:
        """The profile a client should load at startup: `preferred` (e.g. the last one saved in GUI settings)
        if it is still selectable, else the registry default if selectable, else None (load nothing).

        Cheap (no hashing). A preferred id that was removed, withdrawn, planned or whose files are missing
        never reaches select_profile(), so a stale saved setting cannot cause a startup error.
        """
        selectable = {p.profile_id for p in self.list_profiles() if p.selectable}
        for candidate in (preferred, self.default_profile_id):
            if candidate in selectable:
                return candidate
        return None

    def validate_profile(self, profile_id: str) -> ProfileValidation:
        """Full asset check (existence, SHA-256, checkpoint/index compatibility). CPU only; may take ~1 s."""
        try:
            return self._registry.validate(profile_id)
        except KeyError as err:
            raise self._fail(ProfileError("unknown_profile", str(err).strip("'\""),
                                          hint="Use list_profiles() for valid ids.")) from None

    def select_profile(self, profile_id: str, allow_withdrawn: bool = False) -> None:
        """Validate and load a profile's model and index; its defaults become the active config.

        Allowed in IDLE and READY (stop() first when streaming). If validation fails, the currently
        loaded profile stays loaded. If loading fails after the old model was released, the engine
        is IDLE with no model loaded. A withdrawn profile is refused unless allow_withdrawn=True
        (diagnostic scripts only; the GUI never passes it).

        Raises:
            InvalidStateError: while streaming (RUNNING/ERROR). Concurrent lifecycle calls wait, not raise.
            ProfileError: unknown profile, planned profile without a model ("profile_not_ready"),
                withdrawn profile ("profile_withdrawn"), invalid assets, or load failure.
            BackendError: code "worker_busy" (see _drain_previous_worker).
        """
        with self._op_lock:
            self._require(EngineState.IDLE, EngineState.READY, action="select a voice profile")
            self._drain_previous_worker()
            validation = self.validate_profile(profile_id)
            profile = self._registry.get(profile_id)
            if profile.is_withdrawn and not allow_withdrawn:
                raise self._fail(ProfileError(
                    "profile_withdrawn",
                    f"Voice profile '{profile.display_name}' has been withdrawn from normal use.",
                    hint=f"{profile.withdrawn_reason} Choose another profile.",
                ))
            if profile.is_planned:
                raise self._fail(ProfileError(
                    "profile_not_ready",
                    f"Voice profile '{profile.display_name}' has no model yet.",
                    hint=self._readiness(profile)[2],
                ))
            if not validation.ok:
                raise self._fail(ProfileError(
                    "profile_invalid",
                    f"Voice profile '{profile_id}' cannot be loaded.",
                    hint="; ".join(validation.errors),
                    details=repr(validation),
                ))
            assert profile.model_path is not None  # validation.ok implies a model is registered
            previous_state = self._state

            old = self._converter
            self._set(EngineState.LOADING, profile=None, converter=None, config=None, last_metrics=None)
            if old is not None:
                old.unload()
            converter = self._converter_factory()
            try:
                # The pins are re-checked on the bytes the converter actually deserializes: the
                # validation above read the files separately, and they could change in between.
                converter.load_model(str(profile.model_path), str(profile.index_path) if profile.index_path else None,
                                     model_sha256=profile.model_sha256, index_sha256=profile.index_sha256)
                # Warm-up: the first inference after a load is ~4x slower (CUDA/cuDNN initialisation).
                # Done here, the delay never reaches a session, where it would leave a permanent
                # ~0.6 s backlog in the playback buffer (measured 2026-10-21). The length must match the
                # streaming calls (chunk + context): a new input length is slow again on its first call.
                warm_s = AudioSettings.chunk_duration_s + AudioSettings.context_s
                noise = np.random.default_rng(0).normal(0.0, 1e-3, int(STREAM_SAMPLE_RATE * warm_s)).astype(np.float32)
                converter.convert_chunk(noise, config=profile.defaults, input_sr=STREAM_SAMPLE_RATE,
                                        target_sr=STREAM_SAMPLE_RATE)
                # A second, timed call measures the steady conversion time. The pipeline compares a
                # session's first chunk with it: the first chunk is still slow after the GPU idles
                # (435-554 ms vs ~270 ms after an 8 s pause, measured 2026-10-22).
                t0 = time.perf_counter()
                converter.convert_chunk(noise, config=profile.defaults, input_sr=STREAM_SAMPLE_RATE,
                                        target_sr=STREAM_SAMPLE_RATE)
                expected_inference_s = time.perf_counter() - t0
            except Exception as err:
                try:
                    converter.unload()
                except Exception:
                    logger.exception("unload after failed load raised")
                error = ProfileError(
                    "model_load_failed",
                    f"Loading voice profile '{profile_id}' failed.",
                    hint="Check GPU memory (close other GPU apps) and the files listed in validate_profile().",
                    details=f"{type(err).__name__}: {err}",
                )
                logger.exception("model load failed for profile %s", profile_id)
                self._set(EngineState.IDLE, error, profile=None, converter=None, config=None)
                raise error from err
            logger.info("Loaded voice profile %s (previous state %s); steady inference %.1f ms",
                        profile_id, previous_state.value, expected_inference_s * 1000.0)
            self._set(EngineState.READY, profile=profile, converter=converter,
                      config=replace(profile.defaults), last_metrics=None, health=(),
                      expected_inference_s=expected_inference_s)

    def unload_profile(self) -> None:
        """Release the loaded model and return GPU memory. READY -> IDLE; no-op in IDLE."""
        with self._op_lock:
            if self._state == EngineState.IDLE:
                return
            self._require(EngineState.READY, action="unload the voice profile")
            self._drain_previous_worker()
            old = self._converter
            self._set(EngineState.IDLE, profile=None, converter=None, config=None, last_metrics=None)
            if old is not None:
                old.unload()

    # ----------------------------------------------------------------- configuration
    def get_config(self) -> Optional[ConversionConfig]:
        """The active conversion settings (a copy), or None when no profile is loaded."""
        with self._state_lock:
            return replace(self._config) if self._config else None

    @staticmethod
    def validate_config(config: ConversionConfig) -> list[str]:
        """Problems with `config` (empty if valid). Does not change anything."""
        return validate_conversion_config(config)

    def update_config(self, **changes: Any) -> ConversionConfig:
        """Change conversion settings, e.g. update_config(pitch_shift=14, index_rate=0.25).

        Allowed in READY and RUNNING; while running, applies from the next chunk (<= one chunk delay).
        Invalid values change nothing.

        Raises:
            ConfigurationError: unknown field or out-of-range value.
            InvalidStateError: no profile loaded.
        """
        with self._state_lock:
            self._require(EngineState.READY, EngineState.RUNNING, action="change settings")
            assert self._config is not None
            try:
                new = replace(self._config, **changes)
            except TypeError as err:
                raise self._fail(ConfigurationError("unknown_setting", str(err),
                                                    hint=f"Valid fields: {list(asdict(self._config))}")) from None
            problems = validate_conversion_config(new)
            if problems:
                raise self._fail(ConfigurationError("invalid_setting", "; ".join(problems)))
            self._config = new
            if self._pipeline is not None:
                self._pipeline.update_config(replace(new))
            return replace(new)

    def reset_config(self) -> ConversionConfig:
        """Restore the active profile's default settings."""
        with self._state_lock:
            self._require(EngineState.READY, EngineState.RUNNING, action="reset settings")
            assert self._profile is not None
            self._config = replace(self._profile.defaults)
            if self._pipeline is not None:
                self._pipeline.update_config(replace(self._config))
            return replace(self._config)

    # ----------------------------------------------------------------- noise suppression
    def get_noise_suppression(self) -> NoiseSuppressionStatus:
        """Current noise suppression settings and, while streaming, its live state."""
        with self._state_lock:
            settings, pipeline = self._noise, self._pipeline
        stage = getattr(pipeline, "_noise_stage", None) if pipeline is not None else None
        latency_ms = stage.latency_samples / STREAM_SAMPLE_RATE * 1000.0 if stage is not None else 20.0
        return NoiseSuppressionStatus(
            enabled=settings.enabled,
            mode=settings.mode,
            available_modes=tuple(NOISE_SUPPRESSION_MODES),
            max_attenuation_db=NOISE_SUPPRESSION_MODES[settings.mode],
            latency_ms=round(latency_ms, 1),
            active=bool(stage is not None and stage.active),
            failed=bool(stage is not None and stage.failed),
            last_error=stage.last_error if stage is not None else "",
        )

    def set_noise_suppression(self, enabled: Optional[bool] = None,
                              mode: Optional[str] = None) -> NoiseSuppressionStatus:
        """Turn microphone noise suppression on/off and/or choose its mode. Allowed in any state.

        Applies to the running session from the next chunk (on: after one priming chunk, ~0.5 s; then a
        20 ms fade), otherwise to the next start(). Kept across sessions and profile switches. Calling it
        also re-arms a stage that disabled itself after errors.

        Raises:
            ConfigurationError: unknown mode or non-bool enabled (nothing changes).
        """
        with self._state_lock:
            new = replace(self._noise, **{k: v for k, v in (("enabled", enabled), ("mode", mode)) if v is not None})
            problems = validate_noise_suppression(new)
            if problems:
                raise self._fail(ConfigurationError(
                    "invalid_setting", "; ".join(problems),
                    hint=f"Noise suppression modes: {list(NOISE_SUPPRESSION_MODES)}"))
            self._noise = new
            if self._pipeline is not None:
                self._pipeline.update_noise_suppression(new)
        logger.info("noise suppression set: enabled=%s mode=%s", new.enabled, new.mode)
        return self.get_noise_suppression()

    # ----------------------------------------------------------------- devices
    def list_devices(self) -> DeviceInventory:
        """Enumerate audio endpoints now. Ids can change after devices are added or removed."""
        dm = self._device_manager()
        try:
            inputs, outputs = dm.list_input_devices(), dm.list_output_devices()
        except Exception as err:
            raise self._fail(AudioDeviceError("enumeration_failed", "Could not list audio devices.",
                                              hint="Check that the Windows audio service is running.",
                                              details=str(err))) from err
        din, dout = dm.get_default_device(True), dm.get_default_device(False)
        return DeviceInventory(tuple(inputs), tuple(outputs),
                               din.device_id if din else None, dout.device_id if dout else None)

    def validate_audio_settings(self, audio: AudioSettings) -> list[str]:
        """Problems with the route/buffering (empty if valid). Queries devices; opens no stream."""
        problems = []
        lo, hi = CHUNK_DURATION_RANGE_S
        if not lo <= audio.chunk_duration_s <= hi:
            problems.append(f"chunk_duration_s must be between {lo} and {hi}, got {audio.chunk_duration_s}")
        lo, hi = PREBUFFER_RANGE_S
        if not lo <= audio.prebuffer_s <= hi:
            problems.append(f"prebuffer_s must be between {lo} and {hi}, got {audio.prebuffer_s}")
        qlo, qhi = QUEUE_SIZE_RANGE
        if not qlo <= audio.max_queue_size <= qhi:
            problems.append(f"max_queue_size must be between {qlo} and {qhi}, got {audio.max_queue_size}")
        for name, (lo, hi) in (("context_s", CONTEXT_RANGE_S), ("crossfade_s", CROSSFADE_RANGE_S),
                               ("sola_search_s", SOLA_SEARCH_RANGE_S)):
            value = getattr(audio, name)
            if not lo <= value <= hi:
                problems.append(f"{name} must be between {lo} and {hi}, got {value}")
        if audio.crossfade_s == 0 and audio.sola_search_s > 0:
            problems.append("sola_search_s needs crossfade_s > 0")
        if audio.context_s + 1e-9 < audio.crossfade_s + audio.sola_search_s:
            problems.append("context_s must be >= crossfade_s + sola_search_s")
        if audio.crossfade_s + audio.sola_search_s > audio.chunk_duration_s:
            problems.append("crossfade_s + sola_search_s must not exceed chunk_duration_s")
        dm = self._device_manager()
        for dev, is_input, ch in ((audio.input_device_id, True, 1), (audio.output_device_id, False, 2)):
            problem = dm.check_device(dev, is_input=is_input, sample_rate=STREAM_SAMPLE_RATE, channels=ch)
            if problem:
                problems.append(problem)
        return problems

    # ----------------------------------------------------------------- streaming
    def start(self, audio: AudioSettings = AudioSettings()) -> None:
        """Start real-time conversion with the loaded profile. READY -> RUNNING.

        Raises:
            InvalidStateError: no profile loaded, or already running.
            ConfigurationError: buffering values out of range.
            AudioDeviceError: a device is missing, rejects 40 kHz, or fails to open.
        """
        with self._op_lock:
            self._require(EngineState.READY, action="start streaming")
            self._drain_previous_worker()
            problems = self.validate_audio_settings(audio)
            if problems:
                device_problem = any("device" in p for p in problems)
                cls = AudioDeviceError if device_problem else ConfigurationError
                raise self._fail(cls(
                    "audio_device_unavailable" if device_problem else "invalid_audio_settings",
                    "; ".join(problems),
                    hint="Refresh list_devices() and pick an available device; for VB-Cable output choose 'CABLE Input'."
                    if device_problem else "",
                ))
            assert self._converter is not None and self._config is not None
            # The load-time timing is only valid for the input length it was measured with.
            default_length = (audio.chunk_duration_s, audio.context_s) == (AudioSettings.chunk_duration_s,
                                                                         AudioSettings.context_s)
            self._set(EngineState.STARTING)
            pipeline = self._pipeline_factory(
                converter=self._converter,
                conversion_config=replace(self._config),
                input_device_id=audio.input_device_id,
                output_device_id=audio.output_device_id,
                sample_rate=STREAM_SAMPLE_RATE,
                chunk_duration_s=audio.chunk_duration_s,
                max_queue_size=audio.max_queue_size,
                prebuffer_duration_s=audio.prebuffer_s,
                context_duration_s=audio.context_s,
                crossfade_duration_s=audio.crossfade_s,
                sola_search_s=audio.sola_search_s,
                noise_suppression=self._noise,
                expected_inference_s=self._expected_inference_s if default_length else None,
            )
            try:
                pipeline.start()
            except Exception as err:
                error = AudioDeviceError("stream_open_failed", "Could not open the audio streams.",
                                         hint="The device may be in use, unplugged, or not support 40 kHz.",
                                         details=f"{type(err).__name__}: {err}")
                logger.exception("pipeline start failed")
                self._set(EngineState.READY, error)
                raise error from err
            self._set(EngineState.RUNNING, pipeline=pipeline, audio=audio, session_id=self._session_id + 1,
                      health=(), last_metrics=None)

    def stop(self) -> None:
        """Stop streaming. RUNNING/ERROR -> READY. No-op when not streaming. Keeps the model loaded."""
        with self._op_lock:
            if self._state not in (EngineState.RUNNING, EngineState.ERROR):
                return
            pipeline = self._pipeline
            self._set(EngineState.STOPPING)
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    logger.exception("pipeline stop raised")
                metrics = self._metrics_from(pipeline)
                # One line per session so a reported glitch can be checked against the numbers later.
                logger.info("session %d metrics: %s", self._session_id,
                            json.dumps({k: v for k, v in metrics.items() if not isinstance(v, (list, dict))}))
                if pipeline.worker_stuck:
                    self._draining = pipeline
            else:
                metrics = None
            self._set(EngineState.READY, pipeline=None, audio=None, last_metrics=metrics)

    def shutdown(self) -> None:
        """Stop streaming and release the model. Call before the application exits."""
        self.stop()
        self.unload_profile()

    # ----------------------------------------------------------------- status and metrics
    def get_status(self) -> EngineStatus:
        """Current state. While RUNNING, also checks stream/worker health and moves to ERROR on failure."""
        with self._state_lock:
            pipeline, state = self._pipeline, self._state
        if state == EngineState.RUNNING and pipeline is not None:
            problems = list(pipeline.health())
            errors = pipeline.metrics.consecutive_inference_errors
            if errors >= MAX_CONSECUTIVE_INFERENCE_ERRORS:
                problems.append(f"{errors} consecutive inference errors: {pipeline.metrics.last_inference_error}")
            if problems:
                stream_issue = any("stream" in p for p in problems)
                cls = AudioDeviceError if stream_issue else BackendError
                error = cls("stream_failure" if stream_issue else "inference_failure",
                            "Real-time conversion stopped working.",
                            hint="Call stop(), check the devices, then start() again."
                            if stream_issue else "Call stop(); check GPU memory and the log, then start() again.",
                            details="; ".join(problems))
                logger.error("session failure: %s", error.details)
                with self._state_lock:
                    if self._state == EngineState.RUNNING and self._pipeline is pipeline:
                        self._state = EngineState.ERROR
                        self._health = tuple(problems)
                        self._last_error = error.to_info()
                        status, listeners = self._snapshot(), list(self._listeners)
                    else:
                        listeners = []
                for fn in listeners:
                    try:
                        fn(status)
                    except Exception:
                        logger.exception("status listener raised")
        with self._state_lock:
            return self._snapshot()

    @staticmethod
    def _metrics_from(pipeline: RealTimeVoiceChangerPipeline) -> dict[str, Any]:
        pipeline.refresh_stream_counters()
        m = pipeline.metrics.summary()
        chunk_ms = pipeline.chunk_duration_s * 1000.0
        m["chunk_duration_ms"] = chunk_ms
        m["rtf"] = round(m["avg_inference_ms"] / chunk_ms, 3) if chunk_ms else 0.0
        m["current_buffer_ms"] = round(pipeline.output_stream.buffered_seconds * 1000.0, 1)
        # Capture of a chunk's first sample -> that sample leaves the software playback buffer:
        # e2e (ends when the chunk is written) + the audio queued ahead of it + the seam delay.
        # avg_buffer_occupancy_ms is sampled right after each write, so it includes the chunk itself.
        queued_ahead = max(m["avg_buffer_occupancy_ms"] - chunk_ms, 0.0)
        m["estimated_software_latency_ms"] = round(
            m["avg_software_e2e_ms"] + queued_ahead + m["algorithmic_delay_ms"], 1) if m["chunks_converted"] else 0.0
        in_lat = getattr(pipeline.input_stream, "device_latency_s", None)
        out_lat = getattr(pipeline.output_stream, "device_latency_s", None)
        m["input_device_latency_ms"] = round(in_lat * 1000.0, 1) if in_lat is not None else None
        m["output_device_latency_ms"] = round(out_lat * 1000.0, 1) if out_lat is not None else None
        return m

    def get_metrics(self) -> Optional[dict[str, Any]]:
        """Live metrics of the current session, the final metrics of the last session, or None.

        Reset when a new session starts and cleared when the profile changes, so metrics never
        describe a different model. Includes "session_id" and "live" (True while streaming).
        """
        with self._state_lock:
            pipeline, last, sid, profile = self._pipeline, self._last_metrics, self._session_id, self._profile
        if pipeline is not None:
            m, live = self._metrics_from(pipeline), True
        elif last is not None:
            m, live = dict(last), False
        else:
            return None
        m.update(session_id=sid, live=live, profile_id=profile.profile_id if profile else None)
        return m
