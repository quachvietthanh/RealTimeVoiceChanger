# Test Suite

This directory contains the automated test suite for RealTimeVoiceChanger.

## Directory Structure

```text
tests/
├── README.md               # Test suite guidelines and instructions
├── fixtures/               # Small, version-controlled audio test samples (clean, non-private)
├── integration/            # Multi-component and stream pipeline tests
└── unit/                   # Isolated component and logic tests
```

## Testing Guidelines

1. **Unit Tests (`tests/unit/`)**:
   - Verify isolated logic, configuration schema parsing, data structures, tuning presets, and protocol definitions.
   - Must run fast and without requiring external hardware or GPU access.

2. **Integration Tests (`tests/integration/`)**:
   - Verify audio buffer ingestion, mock conversion adapters, pipeline stream flows, and device listing handlers.
   - Test end-to-end routing using mock streams or virtual loopback devices.

3. **Audio Fixture Safety**:
   - Only small, non-private, synthetic, or public domain speech test samples may be committed to `tests/fixtures/`.
   - Never commit personal voice recordings or sensitive audio files.

## Running Tests

```powershell
# Run all unit tests
python -m unittest discover -s tests

# Run offline conversion regression test
python scripts/offline_conversion_test.py
```

| Test module | Covers | Needs |
|---|---|---|
| `unit/test_service_contract.py` | GUI backend API: lifecycle states, errors, single-model switching, config validation, metrics reset, device errors, ERROR detection | nothing (fakes) |
| `unit/test_pipeline_lifecycle.py` | start/stop, stale-chunk drain, start rollback, overload drops, inference errors, health, live config, seam context/reset on drops and errors | nothing (fake streams) |
| `unit/test_noise_suppression.py` | suppressor: exact reconstruction/latency for any call sizes, per-mode attenuation, state across chunks, reset; stage: pure delay when off, priming + ramp, live disable/mode change, failure fallback, session reset | nothing |
| `unit/test_adapter_threads.py` | CPU thread pools limited to 1 for CUDA inference only | nothing |
| `unit/test_crossfade.py` | `ChunkStitcher`: context, exact chunk length, SOLA lag compensation, crossfade continuity, reset, validation | nothing |
| `unit/test_voice_profiles.py` | profile registry parsing and validation, config ranges | nothing (mocked model inspection) |
| `unit/test_devices.py` | device enumeration, virtual-cable flag, 40 kHz checks | nothing (mocked sounddevice) |
| `integration/test_simulated_streaming.py` | sample continuity through the real ring buffer (old joining and context + crossfade), real noise suppressor in the real-time path, device-loss detection | CPU, ~17 s real time |
| `integration/test_profile_assets.py` | every registered profile's real files: hashes, checkpoint, index dimension | model files (skips if absent) |

GPU and hardware checks are scripts, not unit tests: `scripts/service_simulation.py` (real inference,
simulated devices) and the manual procedure in `docs/audio-routing.md` (physical devices, VB-Cable).
