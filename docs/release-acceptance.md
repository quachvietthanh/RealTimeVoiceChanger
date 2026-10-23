# Release Acceptance: Readiness, Packaging and Gates (2026-10-23)

Status of the Windows desktop application as a product, what was verified and how, how it is packaged, and
the gates a release candidate must pass. The backend-only checklist is `docs/backend-release-checklist.md`.

**Verdict: not release-ready.** The source run works and is well covered by automated tests, but no gate
that needs real hardware, a loaded model or a clean machine has been passed in this cycle, no voice is
approved, and no Python runtime package or installer has been built.

Labels: **Verified** (tested here, today, with the command shown) · **Reported** (measured earlier, see the
cited file; not re-run today) · **Mocked** (automated test with simulated devices or a fake converter) ·
**Not verified** (implemented, never exercised in a real environment) · **Planned** (not implemented).

## 1. Production-readiness matrix

| Capability | Implementation / evidence | Automated coverage | Real device / model | Risk | Needed | Acceptance criterion |
|---|---|---|---|---|---|---|
| Profile registry, SHA-256 pinning | `profiles/registry.py`; `--preflight --verify-hashes`: all 5 profiles and 3 auxiliary files match (2026-10-23) | Verified (unit) | Hashes verified on real files; CPU checkpoint validation Reported | Low | – | preflight 0 FAIL with `--verify-hashes` |
| Model loading, VRAM release, profile switch | `service.select_profile`, `RVCAdapter` | Mocked | Reported (`output/backend_validation_2026-10-21/`) | Medium | re-run G2 | G2 |
| Streaming pipeline (bounded queue, drop-oldest, seam stitching) | `audio/pipeline.py` | Mocked (`test_pipeline_lifecycle`, `test_simulated_streaming`) | Reported: 601/601 chunks, 0 drops, 0 steady underflows, RTF 0.598 (simulated devices, `output/live_audio_diag_2026-10-21/`) | Medium | re-run G4 on hardware | G4 |
| Mic to output on physical devices | same | – | **Not verified** | **High** | G3 | G3 |
| Device disconnection | `pipeline.health()` → `EngineState.ERROR`, `stream_failure` | Mocked | **Not verified** | High | G5 | G5 |
| Start / stop / restart, worker drain | `service.start/stop`, `WORKER_DRAIN_TIMEOUT_S` | Mocked | Reported (simulated devices) | Medium | G5 | G5 |
| Latency reporting | `estimated_software_latency_ms` + components in `get_metrics()` | Mocked | Software estimate Reported (1089.6 ms); acoustic **not measured** | High | G4 | G4 |
| Noise suppression (optional, default off) | `audio/noise_suppression.py` | Mocked | Reported stable; listening pending | Medium | owner listening | G6 |
| Virtual cable routing (VB-Cable) | generic output device; no driver bundled | – | **Not verified** (VB-Cable not installed) | High | G3 with cable | G3 |
| GUI truthfulness (file presence vs loaded) | `gui/presentation.py` `availability()`, `readiness_text()` | Verified (unit, headless Tk) | Not verified on screen this cycle | Low | G1 | G1 |
| GUI assets, local only | `gui/assets/asset_manifest.json`, `verify_asset_manifest()` | Verified (unit, no-network test) | – | Low | – | manifest 0 problems |
| Settings persistence | `gui/settings.py` (`%LOCALAPPDATA%\RealTimeVoiceChanger\gui_settings.json`) | Verified (unit, byte comparison) | Not verified across a packaged relaunch | Medium | G7 | G7 |
| Logging | `diagnostics/logging_setup.py` (`%LOCALAPPDATA%\RealTimeVoiceChanger\logs\backend.log`) | Verified (unit) | in use by the running app | Low | – | – |
| Installation check | `voice-changer --preflight` (`diagnostics/preflight.py`) | Verified (unit + real run, 0 FAIL) | Verified in the repository and in a built app folder | Low | – | 0 FAIL |
| Dependency declaration | `pyproject.toml` (runtime imports of Applio now declared), `requirements/inference-win-py311-cu121.txt` | Verified (unit: import closure vs declarations; pins vs specifiers) | Clean-venv install **not run** | Medium | G8 | G8 |
| Application folder build | `scripts/build_app_folder.py` + `BUILD_MANIFEST.json` | Verified (unit) | Built and preflighted in a scratch folder (125 files, 1325 MB with RMVPE/ContentVec and `vctk_p231`) | Medium | runtime + installer | G8 |
| Python runtime package / installer / code signing | – | – | – | **High** | Planned | G8 |
| Custom Vietnamese voice | Voice A withdrawn; Voice B preview | – | owner listening pending | **High** | owner decision | G6 |

## 2. Packaging decision

**Application folder now, installer later; no frozen executable yet.**

- The code finds its files from `__file__`: `profiles/registry.py` (`REPO_ROOT`, 3 levels up) and
  `RVCAdapter._repo_root` (5 levels up) expect the repository layout. A PyInstaller/Nuitka freeze moves
  modules into an archive and breaks both, so freezing would require a path layer inside model loading.
- On the first load, `RVCAdapter._sync_auxiliary_assets()` **copies** RMVPE and ContentVec into
  `third_party/applio/rvc/models/` when they are missing there. Installed under `Program Files` (read-only
  for users) this write fails. `build_app_folder.py --with-aux-models` therefore places both files at the
  Applio location at build time; the cost is a second 559 MB copy inside the folder. Removing the duplicate
  needs a change to model loading (separate review).
- Applio opens `rvc/configs/*.json` relative to the working directory; the adapter wraps every Applio call
  in `contextlib.chdir(applio_root)`, so the application does not depend on the shortcut's start folder.
- PyTorch 2.5.1+cu121 is ~4.3 GB installed (bundles CUDA 12.1 runtime libraries); a working NVIDIA driver
  is still required. Per-user installation under `%LOCALAPPDATA%\Programs\RealTimeVoiceChanger` avoids both
  admin rights and the write problem above.

### Folder layout (`scripts/build_app_folder.py`)

```
RealTimeVoiceChanger\
  BUILD_MANIFEST.json          version, source commit, dirty flag, SHA-256 of every file
  src\voice_changer\           code + gui\assets (icons, presentation data, asset manifest)
  config\voice_profiles.json   registry, unchanged
  third_party\applio\rvc\      inference code with the streaming patch; no logs\, no pretraineds, no UI
  models\checkpoints\          RMVPE + ContentVec (--with-aux-models)
  models\weights|indices\      only profiles named with --profile (refused unless data_rights is owner_recorded, consented
                                or licensed, and while files are in the training cache)
  requirements\                pinned constraints for the runtime install
```

User data stays outside the folder: `%LOCALAPPDATA%\RealTimeVoiceChanger\gui_settings.json` and `logs\`.

### Model distribution

| Asset | Size | Shipped by default | Reason |
|---|---|---|---|
| RMVPE `rmvpe.pt` | 181 MB | with `--with-aux-models` | pinned SHA-256; upstream Applio resources. Redistribution terms not yet reviewed (owner) |
| ContentVec weights + config | 378 MB | with `--with-aux-models` | same |
| `vctk_p231` model + index | 55 + 150 MB | no | registry default; documented use is research/personal testing (`docs/selected-voice-model.md`) |
| Voice A / Voice B | 55 MB + 41/28 MB | no | still in the training cache; Voice A withdrawn, Voice B not approved |
| Jett / Sage | 55 + 72 MB each | **never** | third-party character models, data rights unverified |
| VALORANT agent artwork | – | **never** | redistribution rights not established (`gui/assets/asset_manifest.json`, `excluded`) |
| VB-Cable driver | – | **never** | separate vendor licence; user installs it (`docs/audio-routing.md`) |

Recommendation: ship RMVPE/ContentVec only after their terms are reviewed; otherwise run
`scripts/download_auxiliary_assets.py` (hash-verified) at first setup. Install voice models separately.

### Building and checking the folder

```powershell
.venv\Scripts\python scripts/build_app_folder.py --out dist\RealTimeVoiceChanger --with-aux-models
# provide a Python 3.11 runtime and packages for that folder, e.g. a venv inside it:
py -3.11 -m venv dist\RealTimeVoiceChanger\runtime
dist\RealTimeVoiceChanger\runtime\Scripts\python -m pip install -c dist\RealTimeVoiceChanger\requirements\inference-win-py311-cu121.txt ^
    --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchaudio==2.5.1 dist\RealTimeVoiceChanger
dist\RealTimeVoiceChanger\runtime\Scripts\python dist\RealTimeVoiceChanger\src\voice_changer\main.py --preflight --verify-hashes
```

The runtime step has **not** been executed (≈4.5 GB download). Verified on 2026-10-23: the folder build and
`--preflight --verify-hashes` run from inside the built folder with the repository interpreter (`-I`), which
resolved every path inside the folder: 26 PASS, 4 WARN (profiles without shipped files), 0 FAIL.

### Uninstallation

Delete the application folder. User data is separate: delete `%LOCALAPPDATA%\RealTimeVoiceChanger\` to
remove settings and logs as well. Voice models installed inside the folder go with it. VB-Cable, if
installed, is removed through its own uninstaller.

## 3. Release gates

Thresholds are proposals derived from the reported baseline; the owner confirms them before a release
decision. A gate passes only with the evidence named; anything less is BLOCKED, not PASS.

| Gate | Criterion (pass/fail) | Evidence to record | Status 2026-10-23 |
|---|---|---|---|
| G0 Automated | `python -m unittest discover -s tests/unit -t .` all pass; `--preflight --verify-hashes` 0 FAIL | command output | **PASS** (405 tests; preflight 26/4/0) |
| G1 First launch | clean Windows account: app opens within 30 s, shows the default voice as not loaded, no error dialog, no network access needed | screenshot, `backend.log` | BLOCKED (needs approval: launching a second GUI instance) |
| G2 Model load | default profile loads; state READY; torch allocated after `shutdown()` ≤ 32 MiB; profile switch A→B→A without error | `service_simulation.py` JSON | BLOCKED (GPU; Reported pass 2026-10-21) |
| G3 Mic → output | 2 min live on headphones: intelligible converted voice, no audible dropouts; with VB-Cable: another app records it from "CABLE Output" | operator notes, metrics JSON | BLOCKED (hardware, operator) |
| G4 Stability and latency | 600 chunks: 0 dropped, 0 steady underflows, 0 sample mismatches, 0 inference errors, mean RTF ≤ 0.75, p95 inference ≤ 450 ms; software latency estimate ≤ 1200 ms; acoustic latency measured and recorded | `service_simulation.py --seconds 300` JSON; clap test (`docs/audio-routing.md`) | BLOCKED (GPU; simulated run Reported pass) |
| G5 Lifecycle and recovery | 20 start/stop cycles without error; unplug input while running → ERROR (`stream_failure`) within 2 s, `stop()` → READY, replug + start works; app close leaves no python process | log, metrics, Task Manager | BLOCKED (hardware) |
| G6 Voice quality | owner listening approves a voice; status set to `approved` by the owner | owner note in registry | BLOCKED (owner) |
| G7 Persistence | close and relaunch: device and voice choices restored; unrelated keys in `gui_settings.json` unchanged | before/after file copies | BLOCKED (G1) |
| G8 Clean machine package | runtime installed from the constraints file in a clean VM; `--preflight --verify-hashes` 0 FAIL; G1–G5 repeated on the package | VM notes, preflight output | BLOCKED (runtime/installer not built) |

## 4. Procedures that need explicit approval

None of these were run in this cycle. Each loads a model onto the GPU, opens audio devices, or launches the GUI.

1. **G2 model load (GPU, simulated devices):**
   `.venv\Scripts\python scripts/service_simulation.py --profiles vctk_p231,voice_b_preview,vctk_p231 --seconds 32 --json-output output/<new_dir>/switch.json`
2. **G4 600-chunk regression (GPU, simulated devices, 5 min):**
   `.venv\Scripts\python scripts/service_simulation.py --profiles vctk_p231 --seconds 300 --pitch 10 --index-rate 0.75 --json-output output/<new_dir>/long_run.json`
3. **Device probe (opens real devices, plays silence):**
   `.venv\Scripts\python scripts/device_probe.py --input <id> --output <id> --no-load`
4. **G3 / G5 physical streaming, VB-Cable and acoustic latency:** `docs/audio-routing.md`, "Manual test procedure"
   and "Choppy / robotic live audio: physical check" (headphones, operator present).
5. **G1 / G7 GUI launch:** close the running instance first (it is one process tree: the `.venv` launcher and its
   child interpreter), back up `%LOCALAPPDATA%\RealTimeVoiceChanger\gui_settings.json`, then
   `.venv\Scripts\python src\voice_changer\main.py`.
6. **Voice B relocation (writes `models/` and the registry):** `docs/voice-profiles.md`, "Moving a profile off the
   training cache".

## 5. Known limitations

- Windows only; NVIDIA GPU with CUDA 12.1-capable driver required; tested on one machine (Quadro T2000, 4 GB).
- 40 kHz streams only; WASAPI and WDM-KS endpoints usually reject 40 kHz (use MME or DirectSound).
- Latency is about one second of software delay at default settings (0.5 s chunks); not suitable for
  real-time conversation that needs sub-200 ms response.
- No voice activity detection, echo cancellation, virtual microphone driver or direct integration with
  Discord, OBS, VALORANT or any other application: routing is through a generic output device (VB-Cable).
- Noise suppression is optional and spectral; it is not echo cancellation.
- No approved custom Vietnamese voice yet; the default profile is an English prototype.
- Two preview profiles still read from the training cache until relocated.
- The repository has no top-level LICENSE file although `pyproject.toml` declares MIT; Applio's MIT licence
  is shipped with its code.
