# Backend Release-Readiness Checklist (2026-10-21, rows 13-18 and 16a updated 2026-10-23)

Product-level readiness, packaging and release gates: `docs/release-acceptance.md`.

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
| 12 | Voice quality (A and B) | BLOCKED | owner listening pending. Objective comparison 2026-10-22: 300 s streaming each, 0 drops/underflows/errors, same RTF (0.55/0.54) and VRAM; tone-contour proxy on Vietnamese input A 0.997, B ≥ 0.991; B ~9 dB quieter | `output/checkpoint_eval_2026-10-22/README.md`, `output/spkb_preview_v2_2026-10-21/README.md`, `output/nvf_short_v1_audit_2026-10-21/README.md` |
| 13 | Dependency declarations | PARTIAL | 2026-10-23: the Applio modules loaded for inference import `torchcrepe`, `noisereduce`, `pedalboard`, `pydub`, `wget`, `einops` and `local-attention` at module level; they are now runtime dependencies in `pyproject.toml` (several were only in the `train` extra, others nowhere), guarded by `tests/unit/test_preflight.py` (static import closure vs declarations). The verified venv is recorded in `requirements/inference-win-py311-cu121.txt` (87 exact pins, `pip check` clean); a test checks every runtime dependency is pinned within its specifier | install into a clean venv from the constraints file not yet run; `transformers` major still unpinned in `pyproject.toml` (owner decision) |
| 14 | Applio pin and local patch | PASS | separate git checkout (not a submodule; `third_party/` is git-ignored) at `0c92ac5` (3.2.8-bugfix); working-tree diff equals `patches/applio_streaming.patch` (`git apply --check -R`, 2026-10-23); `--preflight` checks the patch marker | `docs/rvc-engine-specification.md` still names `b65c325` / 3.2.8 |
| 15 | Windows path handling | PASS (source run, app folder) / FAIL (wheel, frozen) | paths derive from `__file__`; `scripts/build_app_folder.py` keeps the repository layout, and `--preflight` run from a built folder resolved everything inside it (2026-10-23) | a regular wheel install or a PyInstaller freeze would still look for `models/`, `config/`, `third_party/` in the wrong place (`docs/release-acceptance.md` §2) |
| 16 | Model assets location | PARTIAL | B migrated 2026-10-23 (owner-approved, local only): `scripts/relocate_profile_assets.py --profile voice_b_preview --copy`, then `--write`. `voice_b_preview` now reads `models/weights/spkb_preview_v2_25e_1975s.pth` (SHA-256 `06f6ec38…4532`) and `models/indices/added_spkb_preview_v2_v2.index` (`fbd308fe…3f7f`); copies match source and pins. The registry diff is only B's `model_path`, `index_path` and `provenance.relocated_from`, and the backup is byte-exact. `--preflight --verify-hashes`: 27 passed, 3 warnings, 0 failed | A (withdrawn) still reads from `third_party/applio/logs/`; the `logs/` originals of B are kept. B's `data_rights` is unspecified (verbal consent only), so `scripts/build_app_folder.py` refuses to package it, like every profile without verified rights |
| 16a | Voice B GUI load from new paths | PASS (load only) | 2026-10-23: the real GUI (`VoiceChangerApp` + `VoiceChangerService`) loaded `voice_b_preview` from the paths in row 16. Pinned SHA-256 checks passed on the exact bytes read for loading. CUDA initialized on the Quadro T2000; the synthetic-noise warm-up succeeded (steady inference ≈ 286 ms). State went `loading` → `ready` with `model_loaded=true` and `last_error=null`, and the GUI showed Voice B as active and ready. Active config was the registry defaults: pitch 14.0, RMVPE, index rate 0.25, filter radius 2, protect 0.4. Shutdown returned to `idle` with ≈ 8.1 MiB CUDA still allocated | Isolated run: `LOCALAPPDATA` pointed to a scratch folder holding a copied settings file, audio stream/play/record calls and `service.start()` were blocked, and no microphone, playback or real-time conversion ran. The real `gui_settings.json`, registry, code, model/index files and protected files were unchanged. The `pydub` "ffmpeg not found" warning did not affect loading or warm-up; audio-file operations without FFmpeg are untested. Not covered: normal launch with the real settings, switching to B from another loaded voice, live conversion and listening (#10–#12) |
| 17 | Static type check | BASELINE | pyright 1.1.389 (installed outside the repository, 2026-10-23): 105 files, 245 errors, all pre-existing (124 `reportArgumentType`, 59 `reportOptionalMemberAccess`, 39 `reportAttributeAccessIssue`) | not installed in `.venv`; errors not yet triaged |
| 18 | CLI entry point | PASS | `voice-changer` launches the GUI; `--preflight [--verify-hashes]` checks the installation; `--check-env`, `--version` | – |
| 19 | Setup/troubleshooting docs | PASS | README quickstart, `docs/model-setup.md`, `docs/audio-routing.md`, `docs/voice-profiles.md`, `docs/gui-backend-contract.md` | |

## Packaging notes (source-run)
- Run from the repository with `.venv` (Python 3.11.9, torch 2.5.1+cu121 from the CUDA index, then `pip install -e .`
  and `scripts/setup_applio_patch.py`), as in the README. The GUI process imports `voice_changer.service`.
- Ship: `src/`, `config/voice_profiles.json`, `third_party/applio` (pinned + patch), `models/checkpoints/` (rmvpe, contentvec),
  the chosen profile `.pth`/`.index` files. Do **not** ship `data/`, `output/`, `third_party/applio/logs/` caches,
  G/D training checkpoints, or the training venv.
- Freezing (PyInstaller etc.) is not evaluated; items 13 and 15 must be fixed first.

## Priority of open issues
1. **High:** owner listening and checkpoint choice for A and B (#12). B's files are already out of `logs/` and load in the GUI (#16, #16a). B's data rights are still unverified.
2. **High:** physical streaming + VB-Cable test (#10, #11).
3. **Medium:** dependency declarations vs the tested venv (#13).
4. **Medium:** asset path resolution for non-editable installs (#15).
5. **Low:** static type check (#17), CLI entry point (#18), output gain control (B is ~9 dB quieter).
