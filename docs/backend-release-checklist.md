# Backend Release-Readiness Checklist (2026-10-21)

Scope: the Python backend for GUI integration and source-run packaging. Not the GUI, not voice quality.
**Verdict: ready for GUI integration against the documented contract (source run). Not release-ready:**
voice quality is unapproved, physical audio/VB-Cable is unverified, and dependency declarations disagree
with the verified environment.

| # | Area | Status | Evidence / command | Notes |
|---|---|---|---|---|
| 1 | Unit + integration tests | PASS | `.venv\Scripts\python -m unittest discover -s tests` → 136 tests OK (`output/noise_suppression_2026-10-21/unittest_full.log`) | no GPU/device needed; profile asset test reads real model files on CPU |
| 2 | Model/aux asset validation | PASS | `tests/integration/test_profile_assets.py`; `ProfileRegistry.validate` | SHA-256, RVC checkpoint keys, v2/40 kHz, index dim 768 for all 3 profiles |
| 3 | Custom model loading + index compatibility | PASS | GPU: `scripts/service_simulation.py` A→B→A (`switch_a_b_a_warmup.json`) | index pinned per profile; A and B never share an index |
| 4 | Inference init / shutdown, VRAM release | PASS | same runs: torch allocated ~437 MiB with one model, flat across switches, 8–16 MiB after `shutdown()` | one model at a time by design |
| 5 | Streaming regression (600 chunks, simulated devices) | PASS | `output/live_audio_diag_2026-10-21/long_run_5min_vctk_stitch_warm700.json` (with chunk-seam stitching): 601/601 chunks, 0 drops, 0 steady underflows, 0 mismatches, 0 stitch resets, RTF 0.598, p95 inference 307.8 ms, software latency estimate 1089.6 ms (old joining: RTF 0.513, 969.5 ms) | simulated devices only; seams verified offline on all 3 profiles |
| 5a | Noise suppression (optional, default off) | IMPLEMENTED, PENDING LISTENING | real-time path connected; 5-min runs off/medium: 601/601, 0 drops/underflows/errors; stage ≈ 4–5 ms CPU per chunk, +20 ms fixed delay; `output/noise_suppression_2026-10-21/README.md` | owner listening and a real-microphone test still needed |
| 5b | CPU use while streaming | PASS | 0.57 cores (was 4.4) with CPU thread pools limited for CUDA inference | same inference time |
| 6 | Configuration validation | PASS | `tests/unit/test_voice_profiles.py`, `test_service_contract.py` | ranges in `docs/gui-backend-contract.md` §4 |
| 7 | Backend API stability (contract) | PASS | `tests/unit/test_service_contract.py` (21 tests) | API is v0: no version field yet; additive changes only from now on |
| 8 | Error reporting + diagnostics logging | PASS | typed errors with codes/hints; `configure_logging()` (`tests/unit/test_logging_setup.py`); callbacks do not log | the app must call `configure_logging()` |
| 9 | Audio device enumeration + failure handling | PASS (mocked) / BLOCKED (hardware) | `tests/unit/test_devices.py`; 40 kHz checks per host API run on the dev machine (MME/DirectSound OK, WASAPI/WDM-KS rejected) | live unplug test not run |
| 10 | Physical mic → speaker streaming | BLOCKED | needs an operator with headphones | procedure: `docs/audio-routing.md` |
| 11 | VB-Cable / Discord / OBS | BLOCKED | VB-Cable not installed on the dev machine | procedure: `docs/audio-routing.md` |
| 12 | Voice quality (A and B) | BLOCKED | owner listening pending | `output/spkb_preview_v2_2026-10-21/README.md`, `output/nvf_short_v1_audit_2026-10-21/README.md` |
| 13 | Dependency declarations | FAIL | `pyproject.toml` declares `numpy>=1.23,<2.0`; the verified inference venv has NumPy 2.4.6 (faiss 1.15.1, torch 2.5.1+cu121). `pip install -e .` into a fresh venv would install a different NumPy than the one tested | owner decision: declare the tested versions (e.g. a lock/constraints file) |
| 14 | Applio pin and local patch | PASS | submodule at `0c92ac5` (3.2.8-bugfix); working-tree diff equals `patches/applio_streaming.patch` (changed lines compared) | |
| 15 | Windows path handling | PASS (source run) / FAIL (wheel) | repo-relative paths via `pathlib`; `REPO_ROOT` / `RVCAdapter._repo_root` are derived from `__file__` | works from the source tree or `pip install -e .`; a regular wheel install would resolve `models/`, `config/`, `third_party/` inside site-packages |
| 16 | Model assets location | FAIL (risk) | A and B profiles point into `third_party/applio/logs/` | a training cache, already hit by one cleanup incident; copy chosen checkpoints to `models/` after owner selection (`docs/voice-profiles.md`) |
| 17 | Static type check | NOT RUN | pyright is declared in `[dev]` extras but not installed in `.venv` | not installed, to avoid a dependency change |
| 18 | CLI entry point | FAIL (minor) | `voice-changer` (`src/voice_changer/main.py`) still prints the scaffold message | the GUI uses the service API; low priority |
| 19 | Setup/troubleshooting docs | PASS | README quickstart, `docs/model-setup.md`, `docs/audio-routing.md`, `docs/voice-profiles.md`, `docs/gui-backend-contract.md` | |

## Packaging notes (source-run)
- Run from the repository with `.venv` (Python 3.11.9, torch 2.5.1+cu121 from the CUDA index, then `pip install -e .`
  and `scripts/setup_applio_patch.py`), as in the README. The GUI process imports `voice_changer.service`.
- Ship: `src/`, `config/voice_profiles.json`, `third_party/applio` (pinned + patch), `models/checkpoints/` (rmvpe, contentvec),
  the chosen profile `.pth`/`.index` files. Do **not** ship `data/`, `output/`, `third_party/applio/logs/` caches,
  G/D training checkpoints, or the training venv.
- Freezing (PyInstaller etc.) is not evaluated; items 13 and 15 must be fixed first.

## Priority of open issues
1. **High:** owner listening and checkpoint choice for A and B (#12), then copy the chosen files out of `logs/` (#16).
2. **High:** physical streaming + VB-Cable test (#10, #11).
3. **Medium:** dependency declarations vs the tested venv (#13).
4. **Medium:** asset path resolution for non-editable installs (#15).
5. **Low:** static type check (#17), CLI entry point (#18), output gain control (B is ~9 dB quieter).
