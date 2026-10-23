# Voice Profiles

A voice profile binds one RVC checkpoint to **its own** retrieval index and default settings. Profiles
are declared in `config/voice_profiles.json` and loaded by `voice_changer.profiles.ProfileRegistry`; the
GUI uses them through `VoiceChangerService` (`docs/gui-backend-contract.md`).

## Current profiles (2026-10-22)

| id | Model | Index | Status |
| --- | --- | --- | --- |
| `vctk_p231` | `models/weights/vctk_female_p231.pth` | `models/indices/vctk_female_p231.index` | prototype (English placeholder, registry default) |
| `voice_a_preview` | `…/logs/nvf_short_v1/nvf_short_v1_50e_5350s.pth` | `…/logs/nvf_short_v1/added_nvf_short_v1_v2.index` | preview, **withdrawn 2026-10-22** (hidden from normal selection; see Withdrawn profiles) |
| `voice_b_preview` | `…/logs/spkb_preview_v2/spkb_preview_v2_25e_1975s.pth` | `…/logs/spkb_preview_v2/added_spkb_preview_v2_v2.index` | preview: pending owner listening |
| `jett` | `models/weights/jett_valorant_500e.pth` | `models/indices/jett_valorant_500e.index` | preview: third-party model, **LICENSE_UNVERIFIED, local evaluation only**; pending owner listening |
| `sage` | `models/weights/sage_valorant_500e.pth` | `models/indices/sage_valorant_500e.index` | preview: third-party model, **LICENSE_UNVERIFIED, local evaluation only**; pending owner listening |

(`…` = `third_party/applio`.) Model and index SHA-256 values are pinned in the JSON.

The A and B checkpoints are **not** owner selections: e50 for A and e25 for B are provisional technical
choices so that each profile can be loaded. Evaluation material:
`output/nvf_short_v1_audit_2026-10-21/README.md` (A) and `output/spkb_preview_v2_2026-10-21/README.md` (B).

## File format

```json
{
  "schema_version": 1,
  "default_profile": "vctk_p231",
  "profiles": [
    {
      "id": "voice_b_preview",              // [A-Za-z0-9_-], unique, stable (the GUI stores it)
      "display_name": "Voice B (preview)",
      "description": "…",
      "status": "preview",                  // planned | prototype | preview | approved (approved = owner only)
      "language": "vi",                     // informational
      "model_path": "relative/to/repo.pth", // or absolute
      "model_sha256": "…",                  // optional but recommended
      "index_path": "relative/to/repo.index", // optional; omit or null for no index
      "index_sha256": "…",                  // optional but recommended
      "defaults": {"pitch_shift": 14.0, "f0_method": "rmvpe", "index_rate": 0.25, "filter_radius": 2, "protect_rate": 0.4},
      "provenance": {"experiment": "…", "checkpoint": "…"},  // free-form
      "data_rights": "consented",           // optional, default "unspecified"; see below
      "dataset": {"path": "data/speakers/b", "manifest": "…"}, // optional, free-form object
      "notes": "…",
      "withdrawn_reason": "…"               // optional; non-empty = withdrawn from normal selection
    }
  ]
}
```

(JSON has no comments; they are annotations here.) Missing `defaults` fields take the `ConversionConfig`
defaults. The registry rejects the file at load time if ids repeat, the default id is unknown, a status is
not one of the four, or a default value is out of range (`ProfileRegistryError`).

### Planned profiles

`"status": "planned"` registers a voice before it has a model. It must **not** have `model_path` or
`index_path`; every other status must have `model_path`. A planned profile is listed
(`ProfileSummary.readiness == "no_model"`, with a `setup_hint`), always fails validation, and
`select_profile` refuses it with `profile_not_ready`. Readiness stages, and where each one is recorded:

| Stage | Evidence |
| --- | --- |
| Registered | entry exists (`status: planned`) |
| Dataset prepared | `dataset.manifest` points at a split manifest with SHA-256 hashes (`scripts/validate_dataset.py`) |
| Model trained | experiment under `third_party/applio/logs/<name>/` |
| Model checked | checkpoint copied to `models/`, entry switched to `preview` with pinned hashes, `validate_profile().ok` |
| Ready for inference | `select_profile` succeeds (GPU load + warm-up) |
| Tested in real time | `scripts/service_simulation.py` report under `output/`, linked in `provenance.evaluation` |

### Data rights

`data_rights` records whether the training data may be used: `unspecified` (not recorded; older entries),
`unverified` (a source is known, its terms are not checked), `owner_recorded`, `consented`, `licensed`.
Only the project owner sets one of the last three. `unverified` adds a validation warning, and a profile
cannot be `approved` while its rights are `unverified`. Publicly downloadable is not the same as licensed:
the candidate source for `jett`/`sage` (Hugging Face `NgThVinh/ValorantAgentVoiceLines`, checked
2026-10-22) declares **no license**, and its audio is game voice lines collected from a fan wiki. Do not
download it, train on it, or ship a model from it until the owner has verified the right to use it.

### Withdrawn profiles

A non-empty `"withdrawn_reason"` takes a profile out of normal use without touching its files: the entry,
paths, SHA-256 pins, status and data rights stay as they are, and validation still checks the files (with a
`withdrawn from normal selection` warning). `VoiceChangerService.list_profiles()` leaves it out (it is still
listed while it is the loaded profile), `list_profiles(include_withdrawn=True)` includes it with
`ProfileSummary.withdrawn_reason`, and `select_profile(id)` refuses it with `profile_withdrawn` (the loaded
model stays loaded). Diagnostic scripts (`service_simulation.py`, `stream_quality_compare.py`,
`noise_suppression_compare.py`) pass `allow_withdrawn=True`, so a withdrawn voice can still be measured on
request; the GUI never does. The `default_profile` and an `approved` profile cannot be withdrawn. To restore
the profile, delete the field.

**Voice A** (`voice_a_preview`) is withdrawn since 2026-10-22 at the owner's request (live use: unstable speech,
echo/repeated sound, choppy playback). Nothing is fixed. Evidence so far:

- Owner's live sessions (`%LOCALAPPDATA%\RealTimeVoiceChanger\logsackend.log`, 2026-10-21/22, 12 sessions
  with Voice A): 0 dropped chunks, 0 steady-state underflows, 0 inference errors, 0 stitch resets,
  RTF 0.555, estimated software latency ~1.25 s. The software pipeline did not drop or repeat chunks.
- Simulated 300 s stream (`output/checkpoint_eval_2026-10-22/`): 601/601 chunks, no drops, no clipping.
- An offline echo/repetition scan (cepstral peak 20-300 ms, envelope autocorrelation, decay after speech
  offsets) of A's whole-file and streamed outputs shows no A-specific signature compared with B and vctk.
- A's output is ~9 dB louder than B's. With ~1.25 s of software latency, converted audio played on
  speakers and picked up again by the microphone would return as a distinct repeat about 1.2 s later, and
  more strongly for the louder voice. **Unverified**: depends on the owner's output device at the time
  (not recorded; the GUI settings file was later overwritten by the unit tests, now fixed).
- A was trained on 4.2 min of audio from one recording (50 epochs); a model trained on that little data
  can produce unstable timbre on unseen input.

Next checks, in order: (1) listen to `output/live_audio_diag_2026-10-21/offline_compare/voice_a_preview__whole.wav`
(no streaming) — if it already sounds unstable/echoey, the cause is the model; (2) repeat a live test with
headphones (no speaker-to-mic path) and noise suppression off; (3) compare with Voice B on the same route.

## Validation

`ProfileRegistry.validate(id)` / `VoiceChangerService.validate_profile(id)` (CPU only, no GPU load):

1. Model and index files exist.
2. Their SHA-256 equals the pinned values. This is what ties an index to its own model: another
   speaker's index, or a rebuilt one, has a different hash and is rejected. An unpinned hash is only a warning.
3. The checkpoint is an RVC inference export (`weight` + `config`), version v1/v2, with pitch guidance.
4. The index dimension equals the checkpoint's embedder dimension (v2: 768, v1: 256); the index is trained and not empty.
5. The checkpoint's recorded embedder (`embedder_model`, written by Applio 3.x exports) is `contentvec`, the only
   one `RVCAdapter` loads. Older exports without the field (e.g. `vctk_p231`) are accepted.
6. The model and index are not registered for a profile declared **earlier** in the file, by path or by content
   (SHA-256). A copy of another voice's file therefore never makes a profile look ready, and appending a bad
   entry never breaks an existing voice.

Warnings (do not block loading): unpinned hashes, no index, status not `approved`, data rights `unverified`, and a
model rate other than 40 kHz (the adapter resamples each chunk to the 40 kHz stream; untested in streaming).

Any error makes `ok` false and `select_profile` refuses the profile.

## Adding a profile or promoting a checkpoint

1. Pick the checkpoint by listening (owner decision). Do not pick by training loss.
2. Copy the chosen `.pth` and its index out of `third_party/applio/logs/` (a training cache that may be
   cleaned) into `models/weights/` and `models/indices/` with descriptive names. Copying is safe; do not
   move or delete anything under `logs/` (see `GEMINI.md`).
3. Compute hashes: `.venv\Scripts\python -c "from voice_changer.conversion.rvc.assets import sha256_file as s; from pathlib import Path; print(s(Path('models/weights/x.pth')))"`
   (with `src` on `PYTHONPATH`).
4. Add or update the entry. Keep `status` at `preview` unless the owner approved it.
5. Run `.venv\Scripts\python -m unittest tests.integration.test_profile_assets`.

Changing `default_profile` changes what the GUI offers first; it needs the owner's approval.

For a `planned` profile, use the registration script (next section) instead of steps 3-4.

### Moving a profile off the training cache

`voice_a_preview` (withdrawn) and `voice_b_preview` still read their `.pth` and `.index` from
`third_party/applio/logs/`. That folder is a git-ignored training cache: a cleanup of `logs/` would make both
profiles unloadable, and a packaged build would not contain them. `scripts/relocate_profile_assets.py`
(`relocate_profile_assets()` in `profiles/registration.py`) moves a registered profile to `models/` without
changing what it loads:

```powershell
.venv\Scripts\python scripts/relocate_profile_assets.py                                     # list affected profiles
.venv\Scripts\python scripts/relocate_profile_assets.py --profile voice_b_preview           # dry run: planned copies
.venv\Scripts\python scripts/relocate_profile_assets.py --profile voice_b_preview --copy    # copy + verify, registry unchanged
.venv\Scripts\python scripts/relocate_profile_assets.py --profile voice_b_preview --write   # repoint (needs both copies)
```

- `--copy` copies to `models/weights/<name>` / `models/indices/<name>` through a `.partial` file, checks the copy
  against the pinned SHA-256 and renames it. The registered files are never moved, modified or deleted, an
  existing destination is never overwritten, and a source that no longer matches its pin is not copied.
- `--write` only runs when every destination matches its pin and the result validates (CPU only). It changes
  `model_path`/`index_path`, records the old paths in `provenance.relocated_from` and keeps a timestamped backup.
  `status`, `defaults` (Voice B's tuning), `data_rights`, `withdrawn_reason` and the hashes are not touched.
- Profiles without pinned hashes are refused: identity of the copy could not be proven.
- Afterwards, select the profile and listen once before removing anything under `logs/` (an owner decision).
- Disk cost (measured 2026-10-23): Voice A 55.2 MB + 40.5 MB, Voice B 55.2 MB + 28.3 MB.

## Jett and Sage

**Status (2026-10-22): third-party preview models, local evaluation only.** Both profiles now use RVC v2 models
downloaded from Hugging Face (not trained by this project) and are selectable in the GUI. `data_rights` stays
`"unverified"` and the provenance records `license_status: LICENSE_UNVERIFIED`: each repository is tagged
`openrail` by its uploader, with no license text or terms, and an uploader cannot license Riot's or the voice
actor's rights to the character voice. Do not distribute, publish or use these models commercially. Quality is
not owner-reviewed; the defaults (p+12, index 0.5, filter 3, protect 0.33) are untuned placeholders.

| | Jett | Sage |
| --- | --- | --- |
| Model page | <https://voice-models.com/model/1m4kd2zR2TO> ("Jett Valorant (RVC v2) 500 Epochs", 2023-08-07, "Trained on Crepe") | <https://voice-models.com/model/1mqq9wQSusT> ("Sage- VALORANT (EN) (RVC v2) 500 Epochs", 2023-08-20) |
| Download | `https://huggingface.co/SecretRetriever/RVC2_Models/resolve/main/Jett500Epoch.zip` (repo rev `a6a8398`) | `https://huggingface.co/lafual/Sage_VALORANT/resolve/main/Sage.zip` (repo rev `ff2819b`) |
| Archive SHA-256 (= Hugging Face LFS hash) | `ebfee2e6bcd578efa72dd854e1a9ac504f7ccfe4514623cdba92fbb696860745` (96,347,872 B) | `344d368032ce29057f3b3d0a357904e8548a759757cceddf38a844baf155a5a2` (96,030,222 B) |
| Archive contents | `Jett500epoch.pth`, `added_IVF589_Flat_nprobe_1_Jetttest_v2.index` (nothing else) | `Sage.pth`, `added_IVF582_Flat_nprobe_1_Sage_v2.index` (nothing else) |
| Checkpoint | v2, f0, 40 kHz, fp16, info `500epoch`; pickle references only `OrderedDict`/tensor rebuild | same |
| Index | Faiss IVF, dim 768, 22,992 vectors (same run name `Jetttest` as the checkpoint) | Faiss IVF, dim 768, 22,728 vectors (run name `Sage`) |
| Attribution | SecretRetriever | lafual ("Give credit as long as you can") |

Verified 2026-10-22 (Quadro T2000): validation and `select_profile` (real `RVCAdapter` load + warm-up) succeed;
offline conversion of `tests/fixtures/input_male.wav` in 0.5 s chunks gives finite, unclipped 40 kHz mono output
at ~238 ms/chunk (RTF ~0.48), ~600 MiB peak allocated VRAM. The `dataset` field in the registry is the earlier
training plan, unrelated to these models.

Output level (measured 2026-10-22, defaults, same fixture at +0/+10/+20 dB): the output level is set by the model,
not the input. Active-speech RMS is ~-16 dBFS (Jett) and ~-18 dBFS (Sage) at every input level (change <=0.1 dB
for +20 dB input); Voice B gives ~-29 dBFS on the same input, so Jett/Sage are ~11-13 dB louder than Voice B.
Peaks stay at 0.82-0.94 and Applio's peak limiter (scales down only above 0.99) never engaged. The adapter calls
Applio with `volume_envelope=1.0`, so the input envelope is not followed and there is no gain stage. The fixture
itself is quiet (peak -23 dBFS, speech ~-44 dBFS RMS), which is why output/input looked like "+27 dB". The Jett
page warns "DEMOS ARE LOUD!".

Loading: `RVCAdapter.load_model` reads each file once, checks the profile's SHA-256 on those bytes, unpickles the
checkpoint only with `torch.load(weights_only=True)` and hands the result to Applio; Applio's own path loader
(an unrestricted `torch.load` under PyTorch 2.5) is disabled on the adapter's instance.

### Local setup (model files are not in Git)

```powershell
# from the repo root; downloads to a temp folder, verifies, then copies only the two files of each archive
$tmp = New-Item -ItemType Directory -Force "$env:TEMP\rvc_jett_sage"
curl.exe -L --fail -o "$tmp\Jett500Epoch.zip" https://huggingface.co/SecretRetriever/RVC2_Models/resolve/main/Jett500Epoch.zip
curl.exe -L --fail -o "$tmp\Sage.zip" https://huggingface.co/lafual/Sage_VALORANT/resolve/main/Sage.zip
Get-FileHash "$tmp\*.zip" -Algorithm SHA256   # must equal the archive SHA-256 above
Expand-Archive "$tmp\Jett500Epoch.zip" "$tmp\jett"; Expand-Archive "$tmp\Sage.zip" "$tmp\sage"
Copy-Item "$tmp\jett\Jett500epoch.pth" models\weights\jett_valorant_500e.pth
Copy-Item "$tmp\jett\added_IVF589_Flat_nprobe_1_Jetttest_v2.index" models\indices\jett_valorant_500e.index
Copy-Item "$tmp\sage\Sage.pth" models\weights\sage_valorant_500e.pth
Copy-Item "$tmp\sage\added_IVF582_Flat_nprobe_1_Sage_v2.index" models\indices\sage_valorant_500e.index
.venv\Scripts\python -m unittest tests.integration.test_profile_assets   # checks the pinned file hashes
```

Without these files the profiles are listed as `files_missing` (not selectable) with a setup hint, and
`validate_profile` reports `model file not found: …\models\weights\<id>_valorant_500e.pth`.

Earlier sources checked 2026-10-22 for training data: Hugging Face `NgThVinh/ValorantAgentVoiceLines` (no
license; game audio from a fan wiki); `MikuX/Jett_Valorant` (tagged apache-2.0 but empty card and no data
provenance); Riot's fan-content policy ("Legal Jibber Jabber", 2018) does not address voice cloning. A model that
may be distributed needs data whose right to train is documented (own recordings with consent, or a license that
explicitly allows it).

### Adding a model

1. Put the files in the supported layout (gitignored, never under `third_party/applio/logs/`):
   `models/weights/<id>_<name>.pth` and `models/indices/<id>_<name>.index` (e.g. `jett_v1`). Each voice has
   its own files; never reuse another profile's checkpoint or index.
2. Requirements: RVC v1/v2 inference export with pitch guidance, trained with ContentVec, the index built from the
   same voice's features. 40 kHz is the verified rate; 32/48 kHz load with a warning and need a streaming test.
3. Record the verified rights, then dry-run and write:

   ```powershell
   .venv\Scripts\python scripts/register_profile_model.py --profile jett --model models/weights/jett_v1.pth --index models/indices/jett_v1.index --data-rights licensed
   .venv\Scripts\python scripts/register_profile_model.py --profile jett --model models/weights/jett_v1.pth --index models/indices/jett_v1.index --data-rights licensed --write
   ```

   The script pins both SHA-256 values, sets `status` to `preview`, validates exactly like `select_profile`,
   keeps `config/voice_profiles.json.bak-<timestamp>` and changes no other entry. `--allow-unverified-rights`
   exists for a local test model that is never distributed.
4. Check: `.venv\Scripts\python -m unittest tests.integration.test_profile_assets` (hashes, format, pairing).
5. GPU checks (not covered by unit tests): `.venv\Scripts\python scripts/service_simulation.py --profiles jett --seconds 300`
   (simulated devices: RTF, drops, VRAM), then the physical procedure in `docs/audio-routing.md` and owner
   listening. Only the owner changes `status` to `approved`.

### Common errors

| Error (code / message) | Cause | Fix |
| --- | --- | --- |
| `profile_not_ready` | status `planned`, no model | add a model (above) |
| `profile_invalid`: `model file not found` / `index file not found` | file not at the registered path | copy the registered file there, or re-register |
| `profile_invalid`: `does not match its registered SHA-256` | file replaced or corrupted | restore the original, or re-register with `--replace` |
| `profile_invalid`: `also registered for profile` / `same content as profile` | another voice's file | use this voice's own checkpoint/index |
| `profile_invalid`: `trained with embedder '...'` | model not trained with ContentVec | retrain with ContentVec |
| `profile_invalid`: `index dimension ... does not match` | v1 index with v2 model (or reverse) | use the index built for this model |
| script `REFUSED: ... training cache` | file under `third_party/applio/logs/` | copy it to `models/` first |
| script `REFUSED: data rights are 'unverified'` | rights not verified | verify, then `--data-rights ...` |
