# Voice Profiles

A voice profile binds one RVC checkpoint to **its own** retrieval index and default settings. Profiles
are declared in `config/voice_profiles.json` and loaded by `voice_changer.profiles.ProfileRegistry`; the
GUI uses them through `VoiceChangerService` (`docs/gui-backend-contract.md`).

## Current profiles (2026-10-21)

| id | Model | Index | Status |
|---|---|---|---|
| `vctk_p231` | `models/weights/vctk_female_p231.pth` | `models/indices/vctk_female_p231.index` | prototype (English placeholder, registry default) |
| `voice_a_preview` | `…/logs/nvf_short_v1/nvf_short_v1_50e_5350s.pth` | `…/logs/nvf_short_v1/added_nvf_short_v1_v2.index` | preview: pending owner listening |
| `voice_b_preview` | `…/logs/spkb_preview_v2/spkb_preview_v2_25e_1975s.pth` | `…/logs/spkb_preview_v2/added_spkb_preview_v2_v2.index` | preview: pending owner listening |

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
      "status": "preview",                  // prototype | preview | approved (approved = owner only)
      "language": "vi",                     // informational
      "model_path": "relative/to/repo.pth", // or absolute
      "model_sha256": "…",                  // optional but recommended
      "index_path": "relative/to/repo.index", // optional; omit or null for no index
      "index_sha256": "…",                  // optional but recommended
      "defaults": {"pitch_shift": 14.0, "f0_method": "rmvpe", "index_rate": 0.25, "filter_radius": 2, "protect_rate": 0.4},
      "provenance": {"experiment": "…", "checkpoint": "…"},  // free-form
      "notes": "…"
    }
  ]
}
```

(JSON has no comments; they are annotations here.) Missing `defaults` fields take the `ConversionConfig`
defaults. The registry rejects the file at load time if ids repeat, the default id is unknown, a status is
not one of the three, or a default value is out of range (`ProfileRegistryError`).

## Validation

`ProfileRegistry.validate(id)` / `VoiceChangerService.validate_profile(id)` (CPU only, no GPU load):

1. Model and index files exist.
2. Their SHA-256 equals the pinned values. This is what ties an index to its own model: another
   speaker's index, or a rebuilt one, has a different hash and is rejected. An unpinned hash is only a warning.
3. The checkpoint is an RVC inference export (`weight` + `config`), version v1/v2, with pitch guidance.
4. The index dimension equals the checkpoint's embedder dimension (v2: 768, v1: 256); the index is trained and not empty.

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
