# GUI ↔ Backend Contract

**Audience:** the desktop GUI developer (Gemini Flash 3.8). **Status:** implemented and covered by
headless tests (`tests/unit/test_service_contract.py`). GPU inference and profile switching were exercised
through this API in a simulated-device run (see "Verification"). Physical device and VB-Cable routing are
**not** verified on hardware.

The GUI talks to one in-process Python object, `VoiceChangerService`. It never imports Applio, Faiss,
PyTorch or sounddevice, never builds file paths to models, and never runs inference itself.

```python
from voice_changer.service import (
    VoiceChangerService, AudioSettings, EngineState, EngineStatus,
    BackendError, ConfigurationError, ProfileError, AudioDeviceError, InvalidStateError,
)
```

Run from the repository root with the inference environment (`.venv`) and `src/` on `sys.path`
(or the package installed with `pip install -e .`).

## 1. Quick start

```python
svc = VoiceChangerService()                      # reads config/voice_profiles.json; loads nothing
svc.add_status_listener(on_status)               # called after every state change (worker thread!)

profiles = svc.list_profiles()                   # cheap
check = svc.validate_profile("voice_b_preview")  # full check, ~1 s, CPU only
devices = svc.list_devices()

svc.select_profile("voice_b_preview")            # loads + warms up the model (2–8 s)  IDLE -> READY
svc.start(AudioSettings(input_device_id=1, output_device_id=4))           # READY -> RUNNING
svc.update_config(pitch_shift=15, index_rate=0.0)  # live, applies from the next 0.5 s chunk
svc.set_noise_suppression(enabled=True, mode="medium")  # mic noise suppression, any state, live
status, metrics = svc.get_status(), svc.get_metrics()   # poll ~1-4x per second
svc.stop()                                       # RUNNING -> READY (model stays loaded)
svc.shutdown()                                   # on application exit: stop + release GPU memory
```

A runnable version: `scripts/backend_api_example.py` (inspect only by default; `--run SECONDS` streams).

## 2. Threading rules

| Call | Duration | Thread |
|---|---|---|
| `select_profile`, `start`, `stop`, `unload_profile`, `shutdown` | 0.1–8 s, serialized | **Call from a worker thread**, not the GUI thread |
| `get_status`, `get_metrics`, `get_config`, `list_profiles`, `update_config`, `reset_config`, `get_noise_suppression`, `set_noise_suppression` | < 5 ms | any thread, any time (also during a long operation) |
| `validate_profile` | ~1 s first time (hashes files; cached afterwards) | worker thread recommended |
| `list_devices`, `validate_audio_settings` | 10–200 ms | worker thread recommended |

- Lifecycle operations are serialized: a second `start()` while `select_profile()` runs waits for it.
- Status listeners run on the thread that caused the transition (the thread that called
  `select_profile`/`start`/`stop`, or the thread that called `get_status()` when it detects a failure).
  Marshal to the GUI thread yourself (e.g. a Qt signal). A listener must not call lifecycle methods.
  Exceptions in listeners are logged and ignored.
- The backend never calls the GUI from an audio callback.

### Logging

Call once at application start:

```python
from voice_changer.diagnostics import configure_logging
log_path = configure_logging()   # %LOCALAPPDATA%\RealTimeVoiceChanger\logsackend.log, rotating 5 x 2 MB
```

Error details and tracebacks go to this file; show users `message`/`hint` and offer the log path for
bug reports. Audio callbacks never log.

## 3. Lifecycle

```
IDLE --select_profile--> LOADING --ok--> READY --start--> STARTING --ok--> RUNNING
                            \--fail--> IDLE          \--fail--> READY          |
READY --select_profile--> LOADING  (the previous model is released first)       |
READY --unload_profile--> IDLE                                                  |
RUNNING --stop--> STOPPING --> READY                                            |
RUNNING --(failure detected by get_status)--> ERROR --stop--> STOPPING --> READY
```

`EngineState` values (string enum): `idle`, `loading`, `ready`, `starting`, `running`, `stopping`, `error`.

| Method | Allowed in | Otherwise |
|---|---|---|
| `select_profile(id)` | `idle`, `ready` | `InvalidStateError` (stop first when streaming) |
| `unload_profile()` | `ready` (no-op in `idle`) | `InvalidStateError` |
| `start(audio)` | `ready` | `InvalidStateError` |
| `stop()` | `running`, `error` | no-op |
| `update_config(...)`, `reset_config()` | `ready`, `running` | `InvalidStateError` |
| `shutdown()` | any | – |

- **One voice model in GPU memory at a time.** `select_profile` validates the new profile first (CPU
  only). If validation fails, the current profile stays loaded and the state is unchanged. If validation
  passes, the old model is released, then the new one is loaded. If that load fails, the state is `idle`
  with no model loaded.
- **Switching while streaming is refused.** Do `stop()` → `select_profile()` → `start()`; expect a gap of
  about 2–3 s (model load) plus the startup buffering below.
- **ERROR** is entered only from RUNNING, when `get_status()` finds an audio stream that stopped on its
  own (device unplugged, driver error), a dead inference worker, or 5 consecutive inference errors. The
  streams are not closed automatically: call `stop()`, which returns to `ready`. Poll `get_status()`
  while running (1–4 Hz), or failures are not detected.

## 4. Data types

### `EngineStatus` (frozen dataclass, from `get_status()`)
| Field | Type | Meaning |
|---|---|---|
| `state` | `EngineState` | see above |
| `profile_id` | `str \| None` | loaded profile |
| `profile_status` | `str \| None` | `prototype`, `preview` or `approved` (quality status; see §6) |
| `model_loaded` | `bool` | a model is in GPU memory |
| `config` | `ConversionConfig \| None` | active settings |
| `audio` | `AudioSettings \| None` | route of the running session |
| `session_id` | `int` | increments on every successful `start()` |
| `health_problems` | `tuple[str, ...]` | why the session is in `error` (empty otherwise) |
| `last_error` | `ErrorInfo \| None` | most recent error (kept until the next error) |

### `ProfileSummary` (from `list_profiles()`)
`profile_id`, `display_name`, `description`, `status`, `language` (`vi`/`en`, informational),
`defaults: ConversionConfig`, `files_present: bool` (cheap existence check), `is_active: bool`, `notes`,
`readiness: str`, `setup_hint: str`, `data_rights: str`, `withdrawn_reason: str`.

`list_profiles()` omits withdrawn profiles (registry field `withdrawn_reason`; currently `voice_a_preview`)
unless it is the loaded profile; `list_profiles(include_withdrawn=True)` returns them too, with a non-empty
`withdrawn_reason`. The GUI should call the default form.

- `selectable` (property, not a constructor field): `readiness == "files_present"` and not withdrawn. This is
  the one rule for "offer a load action"; Voice Studio and Voice Library should both use it instead of
  combining `status`/`readiness`/`files_present` themselves. `select_profile()` can still refuse a selectable
  profile whose files fail the hash/compatibility check (`profile_invalid`).
- `resolve_profile_id(preferred) -> str | None` (cheap, never raises): `preferred` if selectable, else the
  registry default if selectable, else `None`. Call it at startup with the saved profile id and load the
  result (load nothing for `None`). The GUI does this since 2026-10-22.

- `readiness`: `no_model` (status `planned`: registered, nothing to load), `files_missing` (model/index
  declared but not on this machine) or `files_present` (call `validate_profile()` to confirm it is loadable).
  Show `setup_hint` when it is not `files_present`; disable "load" for `no_model`.
- `data_rights`: `unspecified`, `unverified`, `owner_recorded`, `consented` or `licensed` (see
  `docs/voice-profiles.md`). Informational; `unverified` also appears as a `validate_profile()` warning.

### `ProfileValidation` (from `validate_profile(id)`)
`profile_id`, `ok: bool` (loadable), `errors: tuple[str]`, `warnings: tuple[str]`, `model_version`,
`model_sample_rate`, `index_dim`, `index_vectors`. Checks: files exist, SHA-256 equal to the registered
values, checkpoint is an RVC inference export with pitch guidance, index dimension matches the
checkpoint's embedder (768 for v2), index not empty. Every non-approved profile carries a warning.

### `ConversionConfig` (settings; `get_config()` returns a copy)
| Field | Type | Valid range | Notes |
|---|---|---|---|
| `pitch_shift` | float | −24 … +24 semitones | Applied rounded to whole semitones. Male→female: +12…+16 |
| `index_rate` | float | 0.0 … 1.0 | Retrieval blend; no effect if the profile has no index |
| `filter_radius` | int | 0 … 7 | F0 median filter; only active above 2 |
| `protect_rate` | float | 0.0 … 0.5 | Consonant/breath protection; 0.5 disables it |
| `f0_method` | str | `"rmvpe"` | Only RMVPE is verified in this project |

Defaults come from the selected profile (`ProfileSummary.defaults`). `update_config(**fields)` changes
only the named fields and returns the new config; invalid input raises `ConfigurationError` and changes
nothing. `reset_config()` restores the profile defaults. `VoiceChangerService.validate_config(cfg)` returns
the list of problems without applying anything.

### `AudioSettings` (argument of `start()`)
| Field | Type | Default | Valid range | Notes |
|---|---|---|---|---|
| `input_device_id` | int \| None | None | id from `list_devices().inputs` | None = Windows default mic |
| `output_device_id` | int \| None | None | id from `list_devices().outputs` | None = Windows default output; VB-Cable: "CABLE Input" |
| `chunk_duration_s` | float | 0.5 | 0.25 … 2.0 | Only 0.5 s is verified (baseline). Smaller chunks may underflow |
| `prebuffer_s` | float | 0.15 | 0.0 … 1.0 | Playback starts this long after the first converted chunk; shortened by a late first chunk's lateness (see `prebuffer_credit_ms`) |
| `max_queue_size` | int | 3 | 1 … 8 | Captured chunks waiting for inference (see overload policy) |
| `context_s` | float | 0.2 | 0.0 … 1.0, ≥ `crossfade_s + sola_search_s` | Previous input each model call also sees |
| `crossfade_s` | float | 0.05 | 0.0 … 0.2 | Overlap blended between consecutive chunks |
| `sola_search_s` | float | 0.01 | 0.0 … 0.02, needs `crossfade_s` > 0 | Alignment search before the blend |

`context_s` / `crossfade_s` / `sola_search_s` join consecutive chunks without a seam (see
`docs/architecture.md`, "Chunk seams"). They are engineering settings, not user controls; keep the defaults.
All three 0 restores the old end-to-end joining, which leaves a click/level step every chunk (for A/B tests only).

Streaming runs at 40 kHz (`STREAM_SAMPLE_RATE`), mono capture, stereo playback (the same signal on both
channels). The device must accept 40 kHz; MME and DirectSound resample, WASAPI shared mode usually does not.

### Noise suppression (`get_noise_suppression()` / `set_noise_suppression(enabled=None, mode=None)`)

Optional suppression of steady background noise in the **microphone signal before voice conversion**
(fans, PC hum, mains hum, hiss, room tone). **Off by default.** It is a speech enhancer, not a gate: it
does not remove keyboard clicks, other voices or music, and it never mutes the microphone.

| `mode` | Max. noise attenuation | Use |
|---|---|---|
| `"light"` | 6 dB | quiet rooms; least risk to breathy/soft speech |
| `"medium"` (default mode) | 12 dB | typical fan/PC noise |
| `"strong"` | 18 dB | loud steady noise; most risk to soft consonants and breath |

These three modes are the only setting (no continuous slider). `set_noise_suppression` changes only the
arguments given, works in every engine state, and returns a `NoiseSuppressionStatus`:
`enabled`, `mode`, `available_modes` (`("light", "medium", "strong")`), `max_attenuation_db`,
`latency_ms` (20.0), `active` (denoised audio is being heard right now), `failed`, `last_error`.

- **Lifecycle:** the setting is kept across sessions and profile switches (not across application
  restarts: persist it in GUI settings and call `set_noise_suppression` after creating the service). While
  streaming, a change applies from the next chunk. Turning it on primes the denoiser for one chunk
  (~0.5 s) and then fades in over 20 ms; turning it off fades out over 20 ms. Each `start()` begins with a
  fresh noise estimate, which needs about 1–1.5 s of audio to settle (speaking right at start can be
  slightly attenuated during that time).
- **Latency:** the stage adds a fixed 20 ms whether it is on or off (so toggling never shifts or repeats
  audio); included in `algorithmic_delay_ms`. CPU cost ≈ 4 ms per 500 ms chunk on one core; no GPU use.
- **Errors:** an invalid mode or non-bool `enabled` raises `ConfigurationError("invalid_setting")` and
  changes nothing. If the denoiser fails while streaming, that chunk is passed through un-denoised (never
  corrupted or dropped); after 3 failures in a row it stops for the session (`failed=True`, `last_error`,
  `noise_suppression_failed` in metrics) while conversion continues. Calling `set_noise_suppression` again
  re-arms it. The engine does not go to `error` for a denoiser failure.
- **Rates:** runs at the 40 kHz stream rate on mono input; no extra resampling. Device rates (e.g. 44.1 kHz)
  are converted by Windows (MME/DirectSound) before the backend sees the audio.

### `DeviceInventory` and `AudioDevice` (from `list_devices()`)
`inputs`, `outputs`: tuples of `AudioDevice(device_id, name, host_api, max_input_channels,
max_output_channels, default_sample_rate)` with properties `is_input`, `is_output`, `is_virtual_cable`
(name heuristic: VB-Audio, CABLE Input/Output, Hi-Fi Cable, VoiceMeeter). `default_input_id`,
`default_output_id`: Windows defaults or None.

- **Device ids are not stable.** They are PortAudio indices, valid only until devices are added or
  removed. Persist the pair `(name, host_api)` in GUI settings and re-resolve it to an id after
  `list_devices()` at the next launch.
- The same physical device appears once per host API (MME, DirectSound, WASAPI, WDM-KS). MME names
  are truncated to 31 characters. Prefer MME or DirectSound entries.
- `validate_audio_settings(audio)` returns a list of problems (missing device, no channels, 40 kHz
  rejected, values out of range) without opening a stream; `start()` runs the same checks.

### Metrics (`get_metrics()` → `dict | None`)
`None` before the first session of the loaded profile. While streaming, `live=True`, with values that
update continuously. After `stop()`, the final values of that session (`live=False`). Reset on every
`start()` and cleared by `select_profile()`, so metrics never describe a different model.

| Key | Meaning |
|---|---|
| `session_id`, `live`, `profile_id` | which session the numbers belong to |
| `chunks_captured`, `chunks_converted` | chunks taken from the mic / converted |
| `chunks_dropped` | chunks discarded by the overload policy (see below) |
| `avg_inference_ms`, `median_…`, `p95_…`, `p99_…`, `min_…`, `max_inference_ms` | GPU conversion time per chunk (includes the context) |
| `rtf` | `avg_inference_ms / chunk_duration_ms`; must stay well below 1.0 |
| `avg_queue_wait_ms`, `median_…`, `p95_…`, `p99_…`, `max_queue_wait_ms` | time a chunk waited for the worker |
| `avg_software_e2e_ms`, `median_…`, `p95_…`, `p99_…`, `max_software_e2e_ms` | first captured sample → converted chunk in the playback buffer (chunk window + queue + inference) |
| `speech_onset_latency_ms` | software e2e of the first chunk (includes model warm-up) |
| `startup_chunks` | list, first 5 converted chunks: `seq`, `queue_wait_ms`, `noise_suppression_ms` (None without a stage), `inference_ms`, `turnaround_ms`, `software_e2e_ms`, `buffer_before_write_ms`, `buffer_after_write_ms` (software timings only; also logged once per session) |
| `expected_inference_ms` | steady conversion time measured at profile load (0 when not used, e.g. non-default chunk/context length) |
| `prebuffer_credit_ms` | how much of the prebuffer wait was skipped because the first chunk was late (first-chunk processing − expected − 50 ms tolerance, capped at `prebuffer_s` − one 51.2 ms output block). 0 when the first chunk took longer than the earliest next chunk (460.8 ms at 0.5 s chunks / 2048-frame input blocks): the worker is then behind and the full prebuffer is kept. No audio is discarded; lateness above the credit stays in the session latency |
| `current_buffer_ms` | converted audio waiting to be played, right now (swings by one chunk as chunks arrive and play) |
| `avg_buffer_occupancy_ms`, `p95_…`, `max_buffer_occupancy_ms` | the same, sampled right after each write (so it **includes** the chunk just written) |
| `algorithmic_delay_ms` | fixed delays inside the backend: chunk-seam crossfade (55 ms) + noise suppression stage (20 ms) = 75 ms with defaults |
| `noise_suppression_enabled`, `noise_suppression_mode`, `noise_suppression_active`, `noise_suppression_failed`, `noise_suppression_errors`, `noise_suppression_last_error`, `noise_suppression_latency_ms` | noise suppression state this session |
| `avg_noise_suppression_ms`, `p95_…`, `p99_…`, `max_noise_suppression_ms` | CPU time of the noise stage per chunk (near 0 when off) |
| `estimated_software_latency_ms` | **show this as the latency figure**: `avg_software_e2e_ms` + (`avg_buffer_occupancy_ms` − chunk) + `algorithmic_delay_ms` |
| `input_device_latency_ms`, `output_device_latency_ms` | latency PortAudio reports for the opened devices (driver buffers; None for simulated streams) |
| `stitch_resets` | times the seam state was reset because a chunk was dropped or failed (should be 0) |
| `startup_underflows` | silent output blocks before playback started (expected) |
| `steady_underflows` | **audible gaps** during streaming (should be 0) |
| `shutdown_underflows` | silent blocks while stopping (expected) |
| `output_overflows` | writes that exceeded the 4 s playback buffer; oldest audio discarded |
| `input_overflows` | PortAudio reported lost microphone samples |
| `input_status_flags`, `output_status_flags` | callbacks with any PortAudio status flag |
| `inference_errors`, `consecutive_inference_errors`, `last_inference_error` | failed chunks |
| `sample_count_mismatches`, `expected_chunk_samples`, `last_converted_chunk_samples` | output length checks |
| `chunk_duration_ms`, `session_duration_s` | context |

Latency terms are separate quantities; do not add them up into a claim of "acoustic latency". The
backend does not measure physical microphone-to-speaker latency. Display `estimated_software_latency_ms`
("software latency, estimate"); optionally add `input_device_latency_ms + output_device_latency_ms` as
"+ device buffers (reported by the driver)".

Do **not** compute `avg_software_e2e_ms + current_buffer_ms` (an earlier version of this document
suggested it). `current_buffer_ms` includes the chunk that was just written, which `avg_software_e2e_ms`
already covers, so the sum overstates latency by up to one chunk and jumps by 500 ms as chunks arrive. A
GUI showing ~1600 ms that way corresponds to roughly 1000 ms of software latency.

**Overload policy:** if inference falls behind and `max_queue_size` chunks are waiting, the **oldest**
waiting chunk is discarded so latency stays bounded. Every discard is counted in `chunks_dropped`; show
it to the user. Nothing is discarded silently.

## 5. Errors

All errors raised by the service derive from `BackendError` and carry:
`category` (class-level), `code` (stable string), `message` (one line for the user), `hint` (what to do),
`details` (technical; also written to the log). The same data is stored as `EngineStatus.last_error`
(`ErrorInfo` with an added `timestamp`). Show `message` + `hint`; put `details` behind a "details" control.

| Class (`category`) | `code` | When | Engine state afterwards |
|---|---|---|---|
| `InvalidStateError` (`state`) | `invalid_state` | method not allowed in the current state | unchanged |
| `ProfileError` (`profile`) | `unknown_profile` | id not in the registry | unchanged |
| | `profile_not_ready` | profile has status `planned` (no model yet; `hint` says what to do) | unchanged; the current model stays loaded |
| | `profile_withdrawn` | profile withdrawn from normal use (`hint` gives the reason); only diagnostics pass `select_profile(id, allow_withdrawn=True)` | unchanged; the current model stays loaded |
| | `profile_invalid` | file missing, SHA-256 mismatch, incompatible index (`hint` lists the problems) | unchanged; the current model stays loaded |
| | `model_load_failed` | GPU load failed (e.g. out of memory) | `idle`, nothing loaded |
| `ConfigurationError` (`configuration`) | `invalid_setting` | value out of range | unchanged |
| | `unknown_setting` | field name not in `ConversionConfig` | unchanged |
| | `invalid_audio_settings` | chunk/prebuffer/queue value out of range | unchanged (`ready`) |
| `AudioDeviceError` (`audio_device`) | `audio_device_unavailable` | device id missing, no channels, or 40 kHz rejected | unchanged (`ready`) |
| | `stream_open_failed` | PortAudio could not open a stream (device busy, unplugged) | `ready` |
| | `enumeration_failed` | device listing failed | unchanged |
| | `stream_failure` | (status only) a stream stopped while running | `error` |
| `BackendError` (`backend`) | `inference_failure` | (status only) 5 consecutive inference errors or dead worker | `error` |
| | `worker_busy` | `start`/`select_profile`/`unload_profile`/`shutdown` waited 10 s for a conversion left running by the previous `stop()` | unchanged; retry, or restart the app |

`stream_failure` and `inference_failure` are not raised; they appear in `get_status().last_error` when
`get_status()` moves the engine to `error`.

## 6. Voice profiles

Profiles are declared in `config/voice_profiles.json` (format: `docs/voice-profiles.md`). Current entries:

| `profile_id` | Voice | `status` | Default settings |
|---|---|---|---|
| `vctk_p231` | English VCTK p231 prototype (registry default) | `prototype` | p+12, index 0.75, filter 3, protect 0.33 |
| `voice_a_preview` | Speaker A, Northern Vietnamese (nvf_short_v1 e50); **withdrawn**, not in `list_profiles()` | `preview` | p+12, index 0.50, filter 2, protect 0.40 |
| `voice_b_preview` | Speaker B, warm/soft (spkb_preview_v2 e25) | `preview` | p+14, index 0.25, filter 2, protect 0.40 |
| `jett` | Jett (no model yet) | `planned` | placeholders: p+12, index 0.50, filter 3, protect 0.33 |
| `sage` | Sage (no model yet) | `planned` | placeholders: p+12, index 0.50, filter 3, protect 0.33 |

- `planned` profiles are listed so the user can see them, but `select_profile` refuses them
  (`profile_not_ready`). Show them as "not ready" with `ProfileSummary.setup_hint`.
  The GUI now does this (checked 2026-10-22: planned badge, load disabled). GUI follow-ups, not done here:
  (1) the "can it be loaded" rule is re-derived from `status`/`readiness`/`files_present` in five places
  (`profiles_panel.py` x2, `library_view.py` x2, `app.py`); `readiness == "files_present"` alone is the
  backend's answer (`files_present` is equivalent, and a planned profile is always `no_model`).
  Use `ProfileSummary.selectable` for that.
- Switching while streaming: the GUI calls `validate_profile(id)` first and only then `stop()` →
  `select_profile(id)`, so a profile with bad files never interrupts the running session (since 2026-10-22).
  If the GUI is redesigned, keep that order.
- `validate_profile()` also rejects a model trained with an embedder other than ContentVec and files already
  registered for an earlier profile, and warns for non-40 kHz models (`docs/voice-profiles.md`, Validation).
- For the other values, `status` is the owner's quality verdict, not technical readiness: `preview` profiles load and run, but
  the owner has not approved how they sound. Show it, e.g. as a "Preview" badge. Only the project owner
  changes a status to `approved`.
- `default_profile_id` (`vctk_p231`) is not loaded automatically; the GUI decides when to call
  `select_profile`. Remember the user's last profile in GUI settings if desired.
- Voice B's output is about 9 dB quieter than voice A's on the same input; there is no output gain
  control in the backend yet (see §8).

## 7. Startup and buffering behaviour

- `select_profile`: measured 7.6 s for the first load in a process (CUDA initialisation, embedder) and
  2.2–2.4 s for later switches, including one warm-up inference. The warm-up keeps the slow first
  inference out of the session; without it, the startup backlog stayed in the playback buffer for the whole
  session (+0.67 s latency, measured). A second, timed warm-up call (~0.3 s more, measured since
  2026-10-22) gives `expected_inference_ms`.
- The first chunk of a session is still slower after the GPU has idled (435–594 ms vs ~270 ms after an
  8 s pause, simulated devices, 2026-10-22). If the first chunk finishes before the next one can be ready
  (< 460.8 ms), up to `prebuffer_s` − 51.2 ms of its lateness is absorbed by starting playback earlier
  (`prebuffer_credit_ms`); later chunks no slower than the first then cannot underflow (deterministic
  model, evenly spaced callbacks). A slower first chunk gets no credit: when the next chunks are also
  slower than real time, only the full prebuffer prevents underflows (530 ms + 530 ms underflowed with the
  credit, 2026-10-23). The rest of the lateness remains for the whole session.
- After `start()`, the first converted audio is heard after about one chunk (0.5 s) + inference (~0.3 s with
  the 0.2 s context) + `prebuffer_s` (0.15 s) + the 75 ms fixed delay (seams 55 ms, noise stage 20 ms).
- `startup_underflows` > 0 is normal (silence before the first chunk arrives).
- Loading a profile on CUDA sets PyTorch and Faiss CPU thread pools to 1 thread for the whole process
  (streaming CPU use 4.4 → 0.6 cores, same speed). Do not run CPU-heavy PyTorch work in the GUI process.

## 8. Known gaps (not implemented; do not assume)

- No output gain/volume control and no input gate in the backend.
- No offline "convert this file" method on the service (use `scripts/offline_conversion_test.py`).
- No hot profile switching while streaming (refused by design).
- No automatic recovery after device loss: the GUI must `stop()` and `start()` (possibly with re-resolved device ids).
- No persistence of GUI settings or device choices in the backend.
- Device change notifications are not pushed; refresh with `list_devices()`.
- Only RMVPE pitch extraction and 40 kHz streaming are supported.
- `validate_profile` hashes model/index files (~0.2–1 s per profile the first time; cached by size/mtime).

## 9. Verification

| What | How | Result (2026-10-21) |
|---|---|---|
| Contract (states, errors, single-model switching, config validation, metrics reset, device errors, ERROR detection, status during a slow load) | `python -m unittest tests.unit.test_service_contract` (fakes, no GPU) | 20/20 pass |
| Real profile files (hash, checkpoint, index dimension) | `python -m unittest tests.integration.test_profile_assets` | pass for all 3 profiles |
| GPU inference + A→B→A switching through this API, simulated devices | `scripts/service_simulation.py --profiles voice_a_preview,voice_b_preview,voice_a_preview --seconds 32` | `output/backend_validation_2026-10-21/switch_a_b_a_r2.json` |
| Physical microphone, speakers, VB-Cable | – | **not verified**; see `docs/audio-routing.md` |
