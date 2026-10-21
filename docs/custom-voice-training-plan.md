# Safe Custom RVC v2 Voice Training Plan: Warm Northern Vietnamese Female Voice

## 1. Target Voice Profile, Objective & Experiment Nomenclature

- **Identity**: Single adult female speaker.
- **Dialect / Accent**: Natural Northern Vietnamese (Hà Nội or Red River Delta standard conversational pronunciation and intonation).
- **Vocal Timbre**: Warm, soft, mellow, gentle, and clear; free from metallic, robotic, or strained characteristics.
- **Acoustic Objective**: Replace the foreign-sounding and sharp VCTK p231 baseline with an authentic, natural Northern Vietnamese feminine voice identity suitable for conversational voice conversion.
- **Baseline Preservation**: The existing VCTK p231 model (`models/weights/vctk_female_p231.pth`), its index (`models/indices/vctk_female_p231.index`), and all baseline audio outputs (`output/offline_test_result.wav`, `output/quality_tests/`) remain permanently preserved as reference benchmarks.

### 1.1 Strict Experiment Nomenclature & Checkpoint Resumption Isolation

Applio's training engine scans the experiment directory (`logs/<model_name>/`) for existing checkpoints matching `G_*.pth` and `D_*.pth`. **Resumption occurs only when both the matching G and D checkpoints load successfully.** If both load, training advances from those checkpoints, continues epoch numbering, and ignores the specified pretrained base weights (`pretrainG` / `pretrainD`). Otherwise, if either checkpoint is missing, truncated, or fails to deserialize, Applio's broad exception handler catches the failure, resets `epoch_str = 1` and `global_step = 0`, and silently falls back to pretrained weights.

> [!WARNING]
> **Silent Fallback Hazard & Asymmetric Partial-Load Risk on Resumption Failure**:
> In `third_party/applio/rvc/train/train.py` (lines 393–431), checkpoint resumption is wrapped in a broad `try...except:` block. If a previous run suffered a save interruption (e.g. process cancellation or crash during checkpoint serialization), an invalid, partial, or truncated checkpoint file may remain on disk. When resumption is attempted:
> 1. `load_checkpoint` attempts to load `D_*.pth` first into `net_d` and populates `optim_d`.
> 2. Next, `load_checkpoint` attempts to load `G_*.pth` into `net_g` and `optim_g`.
> 3. If either checkpoint is missing, truncated, or fails to deserialize (or if D succeeds but G fails), the broad `except:` block catches the failure, resets `epoch_str = 1` and `global_step = 0`, and **silently falls back to loading the pretrained base weights (`pretrainG` / `pretrainD`)** without halting or warning the user.
> 4. Crucially, if D loads successfully but G fails, the fallback path reloads pretrained weights into `net_d` and `net_g` **without resetting the discriminator optimizer state (`optim_d`)**, leaving `optim_d` populated with historical momentum buffers from the abandoned checkpoint rather than re-initialized (note: this partial-load asymmetry is derived from structural source inspection of `train.py` and has not been runtime-tested).
> Maintaining verified, intact matching checkpoint pairs is therefore mandatory to prevent silent fallback to base weights and optimizer state corruption.

Furthermore, training reuses existing `config.json` and `filelist.txt` files found in that directory (though running `extract.py` regenerates `filelist.txt`).

To ensure strict reproducibility and prevent accidental checkpoint inheritance:
1. **Pipeline Smoke Test**: Must use the dedicated experiment name **`nvf_smoke_v0`**.
   - Purpose: Strict **pipeline correctness verification only** (exercises data loading, forward pass, discriminator/generator backward passes, GradScaler, checkpoint saving, inference extraction, and index building across 3 epochs).
   - Non-Goal: It is **not** a voice quality evaluation. A 3-epoch run on a 253.58-second audio segment (~165 s active / 162 s voiced audio) will not produce perceptual voice conversion quality.
2. **Experimental Short-Data Model**: Must use a separate experiment name, such as **`nvf_short_v1`**.
   - Planned only after `nvf_smoke_v0` passes and the user explicitly reviews and approves proceeding.
   - Operates with completely independent preprocessing, extraction, configuration, training, and indexing steps (never inheriting or depending on artifacts or configs from `nvf_smoke_v0`).
   - Tests acoustic transfer feasibility on the limited available preview dataset.
3. **Eventual Conversational Voice Target**: Must use a new production identifier, such as **`nvf_full_v1`**.
   - Requires a 20–30 minute continuous studio speech recording corpus with broad phonetic coverage (accounting for speech utterances, unvoiced consonants, and natural inter-utterance pauses; not net voiced speech).
4. **Resumption Isolation Rule**: Any new training run intended to start fresh from pretrained base weights **must never reuse an existing experiment directory**. If re-running from scratch, either supply a new experiment name (e.g. bumping the version tag) or delete the existing `logs/<model_name>/` folder.

---

## 2. Dataset, Consent & Data Governance

### 2.1 Speaker Consent & Usage Rights

- **Mandatory Consent**: The target voice dataset must originate from a single consenting adult female Northern Vietnamese speaker who provides explicit authorization for her voice recordings to be collected, processed, and used for local AI voice conversion model training, offline benchmarking, and real-time inference.
- **Intended Scope**: Personal use in gaming, Discord communication, and local development.
- **Prohibitions**: Commercial impersonation, deceptive deepfakes, or public redistribution of the speaker's biometric identity without separate written agreement are strictly prohibited.
- **Public Speech Corpora (VIVOS, ViVoice, Common Voice)**:
  - Public datasets must NOT be scraped or combined automatically.
  - VIVOS is licensed for non-commercial research by VNU-HCM/AILab and contains multiple speakers across accents; extracting a single Northern female speaker requires verified speaker metadata and rights clearance.
  - ViVoice contains commercial and research tiers; licensing must be confirmed before any local use.
  - A research-only license does NOT clear commercial or unrestricted deployment.

### 2.2 Dataset Scopes: Smoke Test vs. Short-Data vs. Full Target

| Scope Tier | Experiment Name | Audio Duration & Characterization | Primary Objective | Quality Expectation |
| :--- | :---: | :---: | :--- | :--- |
| **Pipeline Smoke Test** | `nvf_smoke_v0` | 253.58 s segment (~165 s active audio / ~162 s voiced; 0.00–253.58 s) | Verify pipeline execution, VRAM headroom, checkpointing & extraction across 3 epochs. | Non-evaluable (pipeline check only). |
| **Experimental Short Model** | `nvf_short_v1` | 253.58 s segment (~165 s active audio / ~162 s voiced; 0.00–253.58 s) | Preliminary fine-tuning to observe timbre transfer on limited preview data. | Constrained prosody, limited emotional range. |
| **Full Conversational Target** | `nvf_full_v1` | **20 to 30 minutes** continuous studio recording corpus | High-fidelity conversational voice model with authentic Northern intonation. | Warm, natural, rich prosody across all 6 tones. |

- **Voicing vs. Transcript Disclaimer**: Voiced-frame estimates (~162 s in the 253.58 s preview segment) reflect energy and pitch periodicity analysis from audio tools, not a semantic speech transcript or a count of net spoken words.
- **Recording Format**: WAV PCM (16-bit or 24-bit), mono or stereo, 44.1 kHz or 48.0 kHz.
- **Acoustic Environment**: Quiet room or studio booth; zero background music, zero fan/AC hum, no room echo/reverberation, and no digital clipping (peak amplitude < 0.90 / -1.0 dBFS).
- **Content Diversity**: Natural conversational declarative sentences, questions, varied pacing, varied sentence lengths, expressive tone variation across all 6 Vietnamese tones (ngang, huyền, sắc, hỏi, ngã, nặng).
- **Speaker Purity**: Strictly **single-speaker**. Never mix recordings of multiple speakers into one training dataset.

### 2.3 Anti-Leakage Train/Held-Out Split Protocol & Verified Audio Timeline

In Applio's `preprocess.py` (`third_party/applio/rvc/train/preprocess/preprocess.py`), audio slicing operates with a sliding window defined by `OVERLAP = 0.3` (300 ms). When `cut_preprocess` is enabled, the slicer computes successive segment start positions as:
```python
start = int(self.sr * (self.per - OVERLAP) * i)
```
For 3.0-second target slices (`self.per = 3.0`), adjacent slices share **300 ms of identical audio waveform and pitch trajectory** (a 10% temporal overlap).

> [!CAUTION]
> **Data Leakage Risk**: If raw audio is preprocessed first and the resulting sliced `.wav` clips are partitioned into train and validation sets (randomly or sequentially), adjacent overlapping clips would appear in both sets. This results in severe data leakage, artificially deflating loss and corrupting validation objectivity.

**Mandatory Partition & Verified Timeline Rules**:
1. **Pre-Slicing Partition**: The train vs. held-out test split **MUST be executed on the continuous raw audio recordings BEFORE running `preprocess.py`**.
2. **Verified Audio Timeline** (Measured Source Duration: **295.172 seconds** / 13,017,088 frames @ 44.1 kHz in `female_source.wav` / `female_source_vocal_preview.wav`) `[Measured]`:
   - **Training Segment (0.00 – 253.58 s)**: A 253.58-second segment containing approximately 165 seconds of active audio and 162 seconds classified as voiced by acoustic analysis (clarifying that voiced-frame estimates reflect energy/pitch detection rather than a semantic speech transcript), assigned exclusively to `data/dataset_raw/`.
   - **Natural-Silence Guard (253.58 – 256.28 s)**: A verified **2.70-second** interval of natural acoustic silence between utterances. *(Note: The previously proposed 255.0–290.0 s interval is explicitly removed because it cuts speech mid-utterance. Any arbitrary "3–5 second guard" requirement is replaced with splitting at a verified natural-silence boundary).*
   - **Held-Out Test Segment (256.28 – 291.09 s)**: Approximately **34.8 seconds** of continuous speech assigned to `data/held_out_test/`. *(Caution: This held-out segment has not been semantically verified; it still requires an explicit human listening check to confirm phonetic consistency, absence of music artifacts, and intelligible utterance boundaries before use in model validation).*
   - **Excluded Silent Tail (291.09 – 295.172 s)**: 4.08 seconds of trailing ambient silence excluded from both sets.
3. **Partition Locations & Workflow Rule**:
   - Held-out audio is saved directly to `data/held_out_test/`.
   - Dedicated training audio is placed into `data/dataset_raw/`.
   - `preprocess.py` is pointed exclusively at `../../data/dataset_raw`. The held-out subset is never exposed to slicing, feature extraction, or model training.
   - *Constraint*: Do not create, move, or split any audio files during this planning task.
4. **Generalization Caveat**:
   - A short held-out segment taken from the same continuous recording session provides an internal sanity check for reconstruction and vowel clarity, **not evidence of generalizability** to new microphones, different room acoustics, varied emotional registers, or unconstrained conversational environments.

---

## 3. Verified Applio v3.2.8 Training Architecture & Command Semantics

All training components have been verified directly against the pinned source code in `third_party/applio/rvc/train/`:

### 3.1 Verification of Preprocessing Arguments (`rvc/train/preprocess/preprocess.py`)

| Argument Index | Parameter Name | Verified Type | Verified Behavior in Pinned v3.2.8 Source |
| :---: | :--- | :---: | :--- |
| `sys.argv[1]` | `exp_dir` | `str` | Destination directory for processed audio under `logs/<model_name>`. |
| `sys.argv[2]` | `input_root` | `str` | Source directory containing raw training audio files (`../../data/dataset_raw`). |
| `sys.argv[3]` | `sr` | `int` | Target sampling rate (`40000` for 40 kHz models). Resamples audio to 40 kHz (`sliced_audios`) and 16 kHz (`sliced_audios_16k`). |
| `sys.argv[4]` | `percentage` | `float` | Target segment duration in seconds (`3.0`). Defines `self.per`. |
| `sys.argv[5]` | `num_processes` | `str` / `int` | Number of parallel worker processes (`none` auto-selects `cpu_count()`, or an explicit integer like `4`). |
| `sys.argv[6]` | `cut_preprocess` | `bool` (`strtobool`) | **Dataset Slicing Toggle (CRITICAL FIX)**: Controls `if cut_preprocess:` at line 110. When `True`, passes audio through `self.slicer.slice()` and cuts into 3.0s overlapping windows. When `False`, **slicing is completely bypassed** and the raw unsliced file is passed as one giant chunk. **Must be `True`**. |
| `sys.argv[7]` | `process_effects` | `bool` (`strtobool`) | When `True`, applies a 48 Hz high-pass Butterworth filter (`signal.butter(N=5, Wn=48, btype="high", fs=sr)`) and peak normalization. Set to `False` if dataset is already clean and normalized to avoid double-filtering. |
| `sys.argv[8]` | `noise_reduction` | `bool` (`strtobool`) | When `True`, runs spectral noise gating via `noisereduce`. Set to `False` for clean studio recordings to avoid phase smearing. |
| `sys.argv[9]` | `reduction_strength` | `float` | Noise reduction strength multiplier (`0.0` to `1.0`). |

### 3.2 Verification of Feature Extraction Arguments (`rvc/train/extract/extract.py`)

| Argument Index | Parameter Name | Verified Type | Verified Behavior in Pinned v3.2.8 Source |
| :---: | :--- | :---: | :--- |
| `sys.argv[1]` | `exp_dir` | `str` | Experiment directory (`logs/<model_name>`). Reads from `sliced_audios_16k`. |
| `sys.argv[2]` | `f0_method` | `str` | Pitch estimation algorithm (`"rmvpe"` recommended, or `"crepe"`). |
| `sys.argv[3]` | `hop_length` | `int` | Pitch extraction hop length (`128` for RMVPE / 40 kHz). |
| `sys.argv[4]` | `num_processes` | `int` | Number of worker processes for CPU/GPU multiprocessing (`4`). |
| `sys.argv[5]` | `gpus` | `str` | **Active Device Selector**: Unlike `train.py`, `extract.py` directly parses this argument: `devices = ["cpu"] if gpus == "-" else [f"cuda:{idx}" for idx in gpus.split("-")]`. Pass `"0"` for `cuda:0` or `"-"` for CPU. |
| `sys.argv[6]` | `version` | `str` | Architecture version (`"v2"` for 768-dim ContentVec). |
| `sys.argv[7]` | `sample_rate` | `str` | Target sampling rate string (`"40000"`). |
| `sys.argv[8]` | `embedder_model` | `str` | Speech representation model (`"contentvec"`). |
| `sys.argv[9]` | `embedder_model_custom` | `str` / `None` | Optional custom embedder path (default `None`). |

*Note on Configuration & Filelist Generation*: Upon completing feature and F0 extraction, `extract.py` automatically executes:
1. `generate_config(version, sample_rate, exp_dir)`: Copies `rvc/configs/<version>/<sample_rate>.json` into `<exp_dir>/config.json` **only if `<exp_dir>/config.json` does not already exist**. If a config file already exists in `<exp_dir>`, `generate_config` leaves it intact and does not overwrite it. (Maintaining `fp16_run: true` in `rvc/configs/v2/40000.json` ensures fresh experiments inherit mixed-precision training).
2. `generate_filelist(exp_dir, version, sample_rate)`: Formats, shuffles, and regenerates `<exp_dir>/filelist.txt` from all extracted features.

### 3.3 Verification of Training Arguments (`rvc/train/train.py`)

| Argument Index | Parameter Name | Verified Type | Verified Behavior in Pinned v3.2.8 Source |
| :---: | :--- | :---: | :--- |
| `sys.argv[1]` | `model_name` | `str` | Name of the experiment directory under `logs/<model_name>`. **Resumption Behavior**: Scans for `G_*.pth` and `D_*.pth`. Resumption occurs only when both matching G and D checkpoints load successfully, advancing epoch numbering and ignoring base pretrains. If either checkpoint fails to load, training falls back to pretrained base weights. |
| `sys.argv[2]` | `save_every_epoch` | `int` | **Checkpoint Frequency Interval**. Controls: (1) TensorBoard logging frequency, (2) saving full training checkpoints (`G_*.pth`, `D_*.pth`), and (3) exporting intermediate inference models if `save_every_weights=True`. (Use `1` for the 3-epoch smoke test). |
| `sys.argv[3]` | `total_epoch` | `int` | Total maximum training epochs to execute. (Use `3` for the smoke test). |
| `sys.argv[4]` | `pretrainG` | `str` | Filepath to pretrained generator base weights (`f0G40k.pth`). Skipped if resuming from existing checkpoints. |
| `sys.argv[5]` | `pretrainD` | `str` | Filepath to pretrained discriminator base weights (`f0D40k.pth`). Skipped if resuming from existing checkpoints. |
| `sys.argv[6]` | `version` | `str` | Architecture version (`v2` for 768-dim ContentVec). |
| `sys.argv[7]` | `gpus` | `str` | **Unused GPU Argument (Applio Quirk)**: While parsed on line 63, `train.py` never uses this argument to bind CUDA devices. Instead, it queries `torch.cuda.device_count()` and assigns devices by rank in DDP (`torch.cuda.set_device(rank)`). **GPU selection must be configured via environment variable `$env:CUDA_VISIBLE_DEVICES = "0"`**. |
| `sys.argv[8]` | `batch_size` | `int` | Dataloader batch size per GPU (`1` for smoke test and default for short model; `2` permitted only after explicit VRAM gating conditions are met). |
| `sys.argv[9]` | `sample_rate` | `int` | Target sampling rate (`40000`). |
| `sys.argv[10]` | `pitch_guidance` | `bool` (`strtobool`) | F0 guidance flag (`True` for pitch-guided voice conversion). |
| `sys.argv[11]` | `save_only_latest` | `bool` (`strtobool`) | **Disk Space Guard**: When `True`, checkpoints are named `G_2333333.pth` and `D_2333333.pth`, overwriting earlier checkpoints. Steady-state G/D checkpoint pair storage is approximately **1.29 GB [Calculated: G (~438 MB) + D (~857 MB)]**. During checkpoint writes, Applio saves G and D sequentially using a two-write sequence: the destination path is truncated when the first `torch.save` begins writing it, followed by writing a temporary replacement file (`_old_version.pth`), and finally `os.replace` replaces the destination with that temporary file. The maximum transient save overhead is determined by the larger D checkpoint, adding approximately **+857 MB [Calculated]** temporary storage during its write before atomic replacement. When `False`, saves `G_<step>.pth` and `D_<step>.pth` persistently at every save interval without deleting prior steps. |
| `sys.argv[12]` | `save_every_weights` | `bool` (`strtobool`) | **Inference Checkpoint Export**: When `True`, calls `extract_model` to save a standalone inference `.pth` (`<model>_<epoch>e_<step>s.pth`, ~55 MB; measured baseline `models/weights/vctk_female_p231.pth` is **55,224,656 bytes [Measured]**) **at every epoch divisible by `save_every_epoch`**. |
| `sys.argv[13]` | `cache_data_in_gpu` | `bool` (`strtobool`) | **VRAM Guard**: When `True`, pre-loads all dataset tensors into VRAM, causing immediate CUDA OOM on a 4 GB card. **Must remain `False`**. |
| `sys.argv[14]` | `overtraining_detector` | `bool` (`strtobool`) | Tracks smoothed exponential moving averages (EMA) of Generator and Discriminator losses. When `True`, exports `<model>_<epoch>e_<step>s_best_epoch.pth` during non-overtrained epochs (`epoch > 1`). |
| `sys.argv[15]` | `overtraining_threshold` | `int` | Consecutive epochs without loss improvement before early stopping trigger (default `50`). |
| `sys.argv[16]` | `cleanup` | `bool` (`strtobool`) | Removes temporary training files upon completion. |

### 3.4 Production of Latest Checkpoint and `_best_epoch.pth`

Traced directly from `third_party/applio/rvc/train/train.py` (lines 735–878) and `rvc/train/utils.py`:

1. **Latest Training Checkpoint Structure & Mechanics**:
   - Each checkpoint dictionary contains:
     - `model`: Generator or Discriminator neural weights in FP32.
     - `iteration`: Current global step counter.
     - `optimizer`: Full AdamW optimizer state (first and second momentum buffers for each parameter).
     - `learning_rate`: Current learning rate float.
     - **No Scheduler State**: Schedulers are re-instantiated upon resumption and are not serialized in checkpoint files.
   - At every epoch where `epoch % save_every_epoch == 0`:
     - If `save_only_latest = True`: Applio's `save_checkpoint` saves G (~438 MB) and D (~857 MB) sequentially. In pinned source (`rvc/train/utils.py`, lines 105–118), each checkpoint save executes three sequential steps:
       1. `torch.save` writes to the destination checkpoint path (e.g. `D_2333333.pth`), truncating and replacing its contents. The prior valid checkpoint does not remain intact during this first write.
       2. A second `torch.save` writes the `_old_version.pth` temporary file (e.g. `D_2333333_old_version.pth`).
       3. `os.replace` replaces the destination with that temporary file.
       Because G and D are saved sequentially, the two temporary files never exist simultaneously; the maximum transient save increment is associated with saving the larger D checkpoint, adding approximately **+857 MB [Calculated]** of temporary storage during its write before returning to the **~1.29 GB steady-state** pair footprint (438 MB + 857 MB = 1,295 MB) `[Calculated]`.
       > [!WARNING]
       > **Save-Interruption Risk**: Because the destination checkpoint is truncated as soon as the first `torch.save` begins, an interrupted Mode A save (e.g. process termination or crash during serialization) may leave an invalid or truncated checkpoint on disk. Upon a subsequent resume attempt, failing to load both intact G and D checkpoints will cause Applio's broad exception handler to silently fall back to pretrained base weights.
     - If `save_only_latest = False`: Writes `os.path.join(experiment_dir, f"G_{global_step}.pth")` and `D_{global_step}.pth` persistently for every save interval, accumulating ~1.29 GB of additional storage at each save interval.
   - Resuming training occurs only when **both matching G and D checkpoint files** (`G_*.pth` and `D_*.pth`) load successfully. If either checkpoint fails to load, Applio falls back to pretrained base weights.
2. **Intermediate Inference Exports**:
   - If `save_every_weights = True`: At every `save_every_epoch`, `extract_model()` extracts the Generator weights into `<experiment_dir>/<model_name>_<epoch>e_<step>s.pth` (~55 MB, FP16, stripped of discriminator, optimizer, and `enc_q`; baseline `vctk_female_p231.pth` measured at 55,224,656 bytes `[Measured]`).
3. **`_best_epoch.pth` Inference Checkpoint**:
   - Produced **ONLY IF `overtraining_detector = True`** and `epoch > 1`.
   - In each epoch, EMA smoothed losses are updated:
     ```python
     smoothed_value_gen = update_exponential_moving_average(smoothed_loss_gen_history, current_loss_gen)
     smoothed_value_disc = update_exponential_moving_average(smoothed_loss_disc_history, current_loss_disc)
     ```
   - If overtraining is not triggered:
     - All previous files matching `<model_name>_*e_*s_best_epoch.pth` in `logs/<model_name>/` are deleted.
     - A new file `<model_name>_{epoch}e_{global_step}s_best_epoch.pth` is extracted via `extract_model()`.
   - If `overtraining_detector = False`, **no `_best_epoch.pth` file is ever produced**.
4. **Final Model Export**:
   - When `epoch >= total_epoch` (or when overtraining early stopping fires), `train.py` extracts the final model `<model_name>_{epoch}e_{global_step}s.pth`.

### 3.5 Applio Configuration Precision Behavior & Mandatory Preflight Precision Check

> [!WARNING]
> **Applio Configuration Precision Behavior**:
> In `third_party/applio/rvc/configs/config.py`, initializing `Config()` without a visible GPU or on a GPU listed in Applio's low-end device table (where `is_half` evaluates to `False`) executes `set_precision("fp32")`. This method iterates across all six JSON configuration files in `third_party/applio/rvc/configs/**` (`v1/32000.json`, `v1/40000.json`, `v1/48000.json`, `v2/32000.json`, `v2/40000.json`, `v2/48000.json`) and overwrites `"fp16_run": false`. (Note: this modifies the config JSON files; it does not mutate `preprocess.py`).
>
> When `extract.py` subsequently runs `generate_config()`, it copies the template config into `logs/<experiment>/config.json`. If `fp16_run` is `false`, `train.py` disables PyTorch AMP (`use_amp = config.train.fp16_run and device.type == "cuda"`), executing training entirely in full-precision FP32. On a 4 GB GPU (such as the Quadro T2000 Mobile), full FP32 training significantly elevates VRAM utilization and sharply increases the risk of a CUDA Out-of-Memory (OOM) error.

**Mandatory Preflight Precision Check**:
Before starting training on any experiment (`nvf_smoke_v0`, `nvf_short_v1`, etc.), developers must verify that the generated experiment config has `fp16_run: true`:
```powershell
$cfg = Get-Content "third_party/applio/logs/<experiment>/config.json" | ConvertFrom-Json
if (-not $cfg.train.fp16_run) {
    throw "CRITICAL ERROR: fp16_run is False in logs/<experiment>/config.json. Elevated VRAM consumption and high OOM risk!"
}
Write-Host "Preflight Passed: fp16_run is True in logs/<experiment>/config.json." -ForegroundColor Green
```

**Non-Destructive Config Restoration Rule**:
`generate_config` in Applio checks `if not os.path.exists(config_path):` and **will not overwrite an existing `config.json`**. If a fresh experiment fails the preflight check because `config.json` was generated with `fp16_run: false`:
1. Restore the template configs under `third_party/applio/rvc/configs/` to the pinned upstream state:
   ```powershell
   git -C third_party/applio checkout HEAD -- rvc/configs/
   ```
2. Remove **only that fresh experiment's generated `config.json`** (`logs/<experiment>/config.json`).
3. Re-run `generate_config` (or re-run `extract.py`) to regenerate the correct `config.json` with `fp16_run: true`.
4. **Never** indiscriminately delete an existing experiment directory or training checkpoints.

### 3.6 Correct Inference-Model Loading & Key Validation Protocol

In RVC / Applio, `extract_model()` explicitly prunes training-only components to produce compact runtime inference weights:
```python
opt = OrderedDict(
    weight={key: value.half() for key, value in ckpt.items() if "enc_q" not in key}
)
```
The posterior encoder (`enc_q.*`) computes the variational latent distribution $q(z|spec)$ during training, but is entirely bypassed during inference (which runs via the prior encoder from ContentVec embeddings).

> [!IMPORTANT]
> **Validation Rule**: Calling `model.load_state_dict(checkpoint["weight"], strict=True)` on an exported inference checkpoint will **fail with RuntimeError** due to missing `enc_q.*` keys.

**Correct Verification Procedure**:
1. Load state dict with `strict=False`:
   ```python
   missing_keys, unexpected_keys = net_g.load_state_dict(checkpoint["weight"], strict=False)
   ```
2. **Assert zero unexpected keys**: `assert len(unexpected_keys) == 0, f"Unexpected keys found: {unexpected_keys}"`
3. **Assert missing keys are exclusively posterior encoder keys**:
   `assert all(k.startswith("enc_q.") for k in missing_keys), f"Non-enc_q keys missing: {missing_keys}"`
4. **Assert shape matching**: Treat any tensor shape or dimension mismatch as an immediate hard failure.
5. **Perceptual Quality Disclaimers**: Successful structural loading and key verification confirms architectural integrity, but **does not prove voice naturalness, perceptual quality, or absence of acoustic artifacts**.

---

## 4. Hardware Constraints & VRAM Safety Strategy

### Target GPU: NVIDIA Quadro T2000 Mobile (4 GB VRAM)

Available VRAM on desktop is typically **~2.5 to 2.7 GiB** due to Windows DWM and running applications.

### Checkpoint File Sizes: Training States vs. Inference Exports

- **Training Checkpoints (`G_*.pth`, `D_*.pth`)**:
  - Each checkpoint stores the model (`model`), iteration counter (`iteration`), AdamW optimizer states (`optimizer` momentum and variance buffers for all FP32 parameters: 4 bytes weight + 4 bytes first momentum + 4 bytes second momentum = 12 bytes/param plus state metadata), and learning rate (`learning_rate`).
  - Schedulers are **not** saved in checkpoints; they are re-instantiated upon resumption.
  - Resuming training requires **both G and D checkpoint files** (`G_*.pth` and `D_*.pth`).
  - **Generator G Checkpoint**: Approximately **438 MB [Calculated]** (~36.5M FP32 parameters $\times$ 12 bytes/param + metadata).
  - **Discriminator D Checkpoint**: Approximately **857 MB [Calculated]** (~71.4M FP32 parameters across Multi-Period and Multi-Scale Discriminators $\times$ 12 bytes/param + metadata).
  - **Combined G/D Checkpoint Pair**: Approximately **1.29 GB [Calculated: 438 MB + 857 MB = 1,295 MB ≈ 1.295 GB]**.
  - Used exclusively for resuming training state. Cannot be loaded by `RVCAdapter` for inference.
- **Inference Exports (`<model>_<epoch>e_<step>s.pth`)**:
  - Extracted via `extract_model`. Discriminator, optimizer states, and posterior encoder (`enc_q`) are stripped. Weights are cast to FP16.
  - Size: **55,224,656 bytes (about 55 MB)** `[Measured: models/weights/vctk_female_p231.pth]`.
  - Ready for immediate loading and conversion in `RVCAdapter`.
- **Pretrained Base Models**:
  - Base FP16 Generator (`rvc/models/pretraineds/pretrained_v2/f0G40k.pth`): **approximately 73.1 MB** `[Measured: 73,106,273 bytes]`.
  - Base FP16 Discriminator (`rvc/models/pretraineds/pretrained_v2/f0D40k.pth`): **approximately 142.9 MB** `[Measured: 142,875,703 bytes]`.

### Two-Phase Training Strategy

1. **Phase 1: Pipeline Smoke Test (`nvf_smoke_v0`, batch_size = 1, 3 epochs)**:
   - Run a strict **3-epoch correctness check** with `save_every_epoch = 1` and `batch_size = 1`.
   - Exercises data loading, forward pass, discriminator/generator backward passes, GradScaler, saving checkpoint `G_2333333.pth` / `D_2333333.pth`, exporting inference model `nvf_smoke_v0_1e_<step>s.pth`, overtraining detector initialization on epoch 2, and final completion export `nvf_smoke_v0_3e_<step>s.pth`.
   - Monitor actual peak VRAM consumption via `nvidia-smi` in terminal (the current training script does not report `torch.cuda.max_memory_allocated`).
   - Purpose: Strict **pipeline correctness verification only**, not voice quality evaluation.
2. **Phase 2: Experimental Short Run (`nvf_short_v1`, 100 epochs, default batch_size = 1)**:
   - **Default Execution**: Configure with **`batch_size = 1` by default** to preserve maximum VRAM headroom on the 4 GB GPU.
   - **Prerequisites for Running the Short Experiment**: Running `nvf_short_v1` at `batch_size = 1` requires the Phase 1 smoke test (`nvf_smoke_v0`) to pass and the user to explicitly approve proceeding. Free VRAM is not required to be demonstrated $\ge 1.5\text{ GiB}$ prior to launching at `batch_size = 1`.
   - **Considering `batch_size = 2`**: Considering, testing, or transitioning to `batch_size = 2` additionally requires observing at least **1.5 GiB of free VRAM headroom** during the `batch_size = 1` run using `nvidia-smi`, plus explicit user approval.
   - Do NOT run either configuration during this planning phase.
   - Do NOT use `batch_size = 4` on this 4 GB card.

### Summary Disk Budget by Experiment Tier

| Experiment Tier | Documented Epochs & Interval | Checkpoint Retention Mode | Feature Cache & Index | Model Exports | Checkpoints Retained | Steady-State Total | Transient Write Peak | Disk Headroom Required |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pipeline Smoke Test** (`nvf_smoke_v0`) | 3 epochs<br>save_interval = 1 | Mode A (`save_only_latest=True`) | ~250 MB | ~221 MB<br>(4 exports) | ~1.29 GB<br>(1 G/D pair) | **~1.78 GB** | **+857 MB**<br>(D write) | **$\ge 2.7\text{ GB}$** |
| **Experimental Short** (`nvf_short_v1`) | 100 epochs<br>save_interval = 25 | Mode A (`save_only_latest=True`, default) | ~250 MB | ~276 MB<br>(5 exports) | ~1.29 GB<br>(1 G/D pair) | **~1.87 GB** | **+857 MB**<br>(D write) | **$\ge 3.0\text{ GB}$** |
| **Experimental Short** (`nvf_short_v1`) | 100 epochs<br>save_interval = 25 | Mode B (`save_only_latest=False`, optional) | ~250 MB | ~276 MB<br>(5 exports) | ~5.18 GB<br>(4 G/D pairs) | **~5.76 GB** | **+857 MB**<br>(D write) | **$\ge 7.0\text{ GB}$** |
| **Full Target [Projection]** (`nvf_full_v1`) | 250 epochs<br>save_interval = 25 | Mode A (`save_only_latest=True`, default) | ~1.2 – 1.8 GB | ~607 MB<br>(11 exports) | ~1.29 GB<br>(1 G/D pair) | **~3.2 – 3.9 GB** | **+857 MB**<br>(D write) | **$\ge 5.0\text{ GB}$** |
| **Full Target [Projection]** (`nvf_full_v1`) | 250 epochs<br>save_interval = 25 | Mode B (`save_only_latest=False`, optional) | ~1.2 – 1.8 GB | ~607 MB<br>(11 exports) | ~12.95 GB<br>(10 G/D pairs) | **~14.9 – 15.6 GB** | **+857 MB**<br>(D write) | **$\ge 16.5\text{ GB}$** |

### Itemized Disk Space Estimates

| Storage Component | Smoke Test (`nvf_smoke_v0`)<br>(3 ep, interval 1)<br>`save_only_latest=True` | Experimental Short (`nvf_short_v1`)<br>(100 ep, interval 25)<br>`save_only_latest=True` | Full Target [Projection] (`nvf_full_v1`)<br>(250 ep, interval 25)<br>**Mode A: `save_only_latest=True`** | Full Target [Projection] (`nvf_full_v1`)<br>(250 ep, interval 25)<br>**Mode B: `save_only_latest=False`** | Basis & Arithmetic |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Cached Audio & Features** | ~210 MB | ~210 MB | ~1.0 – 1.5 GB | ~1.0 – 1.5 GB | `[Calculated]`: ~0.83 MB/s of raw speech (40k slices, 16k slices, RMVPE F0, 768-dim ContentVec). Smoke / Short: 253.58 s preview segment (~210 MB); Full [Projection]: 20–30 min continuous corpus (1,000–1,500 MB). |
| **FAISS Feature Index** | ~38 – 45 MB | ~38 – 45 MB | ~200 – 330 MB | ~200 – 330 MB | `[Estimated]`: IVFFlat/Flat index (all slice counts, vector counts, and index sizes are approximate estimates until measured during a real extraction/indexing run). Smoke / Short: estimated ~90–95 variable-length slices (~12.5k–13k vectors, ~38–45 MB); Full [Projection]: estimated ~440–670 slices (~66.7k–100k vectors, ~200–330 MB). MiniBatchKMeans downsampling skipped ($N \le 200,000$); coarse quantizer trained via Faiss. |
| **Inference Model Exports** | ~221 MB<br>(4 exports) | ~276 MB<br>(5 exports) | ~607 MB<br>(11 exports) | ~607 MB<br>(11 exports) | `[Calculated]`: ~55.2 MB per model (measured baseline `models/weights/vctk_female_p231.pth`: 55,224,656 bytes `[Measured]`).<br>• Smoke: ep 1, 2, 3 interval exports + ep 3 rolling best = 4 exports (4 × 55.2 MB = 220.8 MB).<br>• Short: ep 25, 50, 75, 100 interval exports + 1 rolling best = 5 exports (5 × 55.2 MB = 276.1 MB).<br>• Full [Projection]: 10 interval exports + 1 rolling best = 11 exports (11 × 55.2 MB = 607.2 MB). |
| **TensorBoard Event Logs** | ~15 MB | ~50 – 70 MB | ~150 – 200 MB | ~150 – 200 MB | `[Calculated]`: Approximate estimates. Traced from `train.py` (lines 701–728): scalar metrics and 3 spectrogram images (`slice/mel_org`, `slice/mel_gen`, `all/mel`) are logged each epoch; audio waveforms (`gen/audio_*`) are synthesized and logged only at epochs divisible by `save_every_epoch`. |
| **Training Checkpoints** | ~1.29 GB<br>(1 G/D pair) | ~1.29 GB<br>(1 G/D pair) | ~1.29 GB<br>(1 G/D pair) | **~12.95 GB**<br>(10 G/D pairs) | `[Calculated]`: G (~438 MB) + D (~857 MB) = 1,295 MB (~1.295 GB) per pair.<br>• Mode A (`save_only_latest=True`): 1 pair retained steady-state.<br>• Short Mode B: 4 pairs retained (4 × 1,295 MB = 5,180 MB ≈ 5.18 GB).<br>• Full Mode B: 10 pairs retained (10 × 1,295 MB = 12,950 MB ≈ 12.95 GB). |
| **Steady-State Total** | **~1.78 GB** | **~1.87 GB** | **~3.2 – 3.9 GB** | **~14.9 – 15.6 GB** | `[Calculated]`: Sum of feature cache, index, exports, logs, and steady-state checkpoints.<br>• Smoke: 210 + 40 + 221 + 1,295 + 15 = 1,781 MB (~1.78 GB).<br>• Short Mode A: 210 + 40 + 276 + 1,295 + 50 = 1,871 MB (~1.87 GB).<br>• Full Mode A: (1,000 to 1,500) + (200 to 330) + 607 + 1,295 + (150 to 200) = 3,252 to 3,932 MB (~3.2–3.9 GB).<br>• Full Mode B: (1,000 to 1,500) + (200 to 330) + 607 + 12,950 + (150 to 200) = 14,907 to 15,587 MB (~14.9–15.6 GB). |
| **Transient Checkpoint-Write Overhead** | **+857 MB**<br>(D write) | **+857 MB**<br>(D write) | **+857 MB**<br>(D write) | **+857 MB**<br>(final D write) | `[Calculated]`: Traced from `rvc/train/utils.py:save_checkpoint`. Each checkpoint save executes: (1) `torch.save` writes to the destination path, truncating its contents when writing begins, (2) a second `torch.save` writes `<name>_old_version.pth`, and (3) `os.replace` replaces the destination with that temporary file. Sequential save of G then D means the largest transient file is `D_*_old_version.pth` (+857 MB). In Mode A, the destination is truncated when the first `torch.save` begins writing it, followed by the temporary-file write and `os.replace`. An interrupted save can leave an invalid checkpoint, risking silent fallback to pretrained weights. In Mode B, the temporary file coexists momentarily with the newly written versioned checkpoint before rename. |
| **Peak Disk Headroom Required** | **~2.64 GB** | **~2.73 GB** | **~4.1 – 4.8 GB** | **~15.7 – 16.5 GB** | `[Calculated]`: Steady-state total plus transient write overhead (+857 MB).<br>• Smoke: 1.78 GB + 0.86 GB = ~2.64 GB (recommend $\ge 2.7$ GB free).<br>• Short Mode A: 1.87 GB + 0.86 GB = ~2.73 GB (recommend $\ge 3.0$ GB free).<br>• Full Mode A: 3.2–3.9 GB + 0.86 GB = ~4.1–4.8 GB (recommend $\ge 5.0$ GB free).<br>• Full Mode B: 14.9–15.6 GB + 0.86 GB = ~15.7–16.5 GB (recommend $\ge 16.5$ GB, conservative $\ge 17$ GB free). |

### FAISS Vector-Count & Index Storage Arithmetic

Traced directly from Applio's ContentVec configuration (`models/checkpoints/contentvec/config.json`) and index extraction script (`third_party/applio/rvc/train/process/extract_index.py`):
1. **Effective ContentVec Frame Rate**:
   - The convolution stride list in `config.json` is `"conv_stride": [5, 2, 2, 2, 2, 2, 2]`.
   - Stride product: $5 \times 2 \times 2 \times 2 \times 2 \times 2 \times 2 = 5 \times 64 = 320$ audio samples per frame.
   - For 16 kHz input audio (`sliced_audios_16k`): $16,000 / 320 = \mathbf{50\text{ vectors/second}}$ `[Verified from source]`.
2. **Preprocessing Slicing & Overlap Accounting**:
   - In `preprocess.py`, audio slicing is handled by `self.slicer.slice(audio)`, which is silence-aware and splits audio dynamically on detected quiet intervals (`db_thresh`), producing variable-length segments rather than a fixed grid of identical clips. Segments exceeding target duration `self.per = 3.0` s are then partitioned with sliding overlap `OVERLAP = 0.3` s (300 ms), advancing by 2.7 s per step.
   - For a nominal 3.0-second slice, the nominal frame count based on 50 vectors/second is approximately 150 frames ($3.0 \times 50$). However, because ContentVec's convolutional receptive field, padding, and edge framing conditions affect boundary feature extraction, the precise output frame count depends on model framing and should not be presented as an exact invariant without empirical verification.
   - For a **20–30 minute continuous studio recording corpus** (1,200 to 1,800 seconds):
     - Net unique continuous audio: $1,200 \times 50 = 60,000$ to $1,800 \times 50 = 90,000$ vectors.
     - Extracted slice count: approximately 440 to 670 variable-length slices.
     - Total vectors extracted across all slices (counting overlapping 300 ms boundary windows across generated `.npy` feature files): $\approx \mathbf{66,700\text{ to }100,000\text{ vectors}}$ `[Calculated]`.
     - *Characterization Note*: The 20–30 minute recording span is a continuous studio recording corpus, **not net voiced speech**; it contains voiced phonemes, unvoiced consonants, fricatives, breaths, and natural speech pauses, all of which are sliced and embedded by ContentVec.
   - For the **preview segment** (smoke test & `nvf_short_v1`):
     - 253.58 s continuous segment: yields an approximate estimate of **~90 to 95 variable-length slices** (nominally ~94 slices).
     - Total extracted vectors: estimated at **about 12,500 to 13,000 ContentVec vectors** (subject to the silence-aware slicer's actual segmentation and how overlapping windows contribute repeated frames into the saved `.npy` feature arrays, compared to ~12,679 raw 50 Hz frames across 253.58 s).
     - *Estimation Note*: All slice counts, vector counts, and resulting index file sizes are approximate estimates until measured during a real extraction and index run.
3. **Raw Float32 Vector Storage**:
   - Vector dimensionality is 768 in FP32: $768 \times 4\text{ bytes} = \mathbf{3,072\text{ bytes per vector}}$ (~3.072 KB).
   - Preview segment (~12,500–13,000 vectors) `[Estimated]`: ~38.4 to 40.0 MB.
   - 20–30 min corpus (~66.7k–100k vectors) `[Estimated]`: $66,700 \times 3,072 \approx 205\text{ MB}$ to $100,000 \times 3,072 \approx 307.2\text{ MB}$.
4. **Index-Building Logic, K-Means Operations & `n_ivf` Centroid Arithmetic**:
   - In `third_party/applio/rvc/train/process/extract_index.py`, two distinct clustering/k-means operations occur:
     1. **`MiniBatchKMeans` Downsampling Pass (scikit-learn)** (lines 40–53):
        ```python
        if big_npy.shape[0] > 2e5 and (index_algorithm == "Auto" or index_algorithm == "KMeans"):
            big_npy = MiniBatchKMeans(n_clusters=10000, ...).fit(big_npy).cluster_centers_
        ```
        This optional pre-clustering step activates only when vector count $N > 200,000$ (`2e5`), where it compresses the training set down to 10,000 cluster centers. Because our 20–30 minute corpus produces an estimated ~66.7k–100k vectors (and the preview segment produces an estimated ~12.5k–13k vectors), this **`MiniBatchKMeans` downsampling step is skipped** under `"Auto"`, preserving all extracted feature vectors.
     2. **FAISS IVF Coarse-Quantizer Training (Faiss internal k-means)** (lines 72–77):
        ```python
        n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
        index_added = faiss.index_factory(256 if version == "v1" else 768, f"IVF{n_ivf},Flat")
        index_added.train(big_npy)
        ```
        An `IVF{n_ivf},Flat` index requires a trained coarse quantizer (`IndexFlatL2`) containing `n_ivf` Voronoi centroids. The training call `index_added.train(big_npy)` executes Faiss's internal k-means clustering on the training vectors to compute these `n_ivf` centroid locations. This operation is **never skipped**; it is required so subsequent additions (`index_added.add(...)`, in batches of 8,192) can assign all raw vectors to their respective inverted lists.
   - **Corrected `n_ivf` and Centroid Storage Arithmetic**:
     - The formula `n_ivf = min(int(16 * np.sqrt(N)), N // 39)` evaluates both terms:
       - For **$N \approx 66,700$** (lower bound of 20–30 min corpus):
         - `16 * sqrt(N)`: $\text{int}(16 \times \sqrt{66,700}) = \text{int}(4,132.22) = \mathbf{4,132}$
         - `N // 39` = $66,700 // 39 = 1,710$
         - `n_ivf = min(4132, 1710) = 1,710`
         - Float32 centroid table ($1,710 \times 768 \times 4\text{ bytes}$): **$\approx 5.3\text{ MB}$** ($5,253,120\text{ bytes}$).
       - For **$N \approx 100,000$** (upper bound of 20–30 min corpus):
         - `16 * sqrt(N)`: $\text{int}(16 \times \sqrt{100,000}) = \text{int}(5,059.64) = \mathbf{5,059}$
         - `N // 39` = $100,000 // 39 = 2,564$
         - `n_ivf = min(5059, 2564) = 2,564` (the second term limits `n_ivf`, not 5,059)
         - Float32 centroid table ($2,564 \times 768 \times 4\text{ bytes}$): **$\approx 7.9\text{ MB}$** ($7,876,608\text{ bytes}$).
       - For **$N \approx 12,500\text{ to }13,000$** (smoke test / short-preview segment estimate):
         - `16 * sqrt(N)`: for $N = 13,000$, $\text{int}(16 \times \sqrt{13,000}) = \text{int}(1,824.28) = \mathbf{1,824}$ (or $1,788$ for $N = 12,500$)
         - `N // 39` = $13,000 // 39 = 333$ (or $320$ for $N = 12,500$)
         - `n_ivf = min(1824, 333) = 333` (the second term limits `n_ivf`)
         - Float32 centroid table ($333 \times 768 \times 4\text{ bytes}$): **$\approx 1.02\text{ MB}$** ($1,022,976\text{ bytes}$).
   - **Overall FAISS Index Storage Composition**:
     - Inverted list vectors ($N \times 768 \times 4\text{ bytes}$): ~205.0 to 307.2 MB (full target) or ~38.4 to 40.0 MB (smoke test).
     - Coarse-quantizer centroid table: ~5.3 to 7.9 MB (full target) or ~1.0 MB (smoke test).
     - Inverted list 64-bit entry IDs ($N \times 8\text{ bytes}$): ~0.53 to 0.80 MB (full target) or ~0.10 MB (smoke test).
     - Inverted list header offsets and metadata: ~0.1 to 0.2 MB.
     - **Total FAISS Index Size on Disk**: **~211 to 316 MB [Estimated]**, which firmly supports the estimated range of **~200 to 330 MB [Estimated]** for the full target corpus and **~38 to 45 MB [Estimated]** for the preview segment. (All index sizes remain estimates until measured during an actual run).

### Checkpoint File States & Retention Arithmetic (Mode A vs. Mode B)

Traced directly from `third_party/applio/rvc/train/utils.py` (`save_checkpoint`, lines 105–118) and `train.py` (lines 735–752):
```python
# utils.py:save_checkpoint
torch.save(checkpoint_data, checkpoint_path)
old_version_path = checkpoint_path.replace(".pth", "_old_version.pth")
checkpoint_data = replace_keys_in_dict(...)
torch.save(checkpoint_data, old_version_path)
os.replace(old_version_path, checkpoint_path)
```
- In `train.py`, G and D are saved sequentially: `save_checkpoint(net_g, ...)` finishes before `save_checkpoint(net_d, ...)` begins.
- **Mode A (`save_only_latest = True`)**:
  - Checkpoint filenames are fixed: `G_2333333.pth` (~438 MB) and `D_2333333.pth` (~857 MB).
  - Pinned source execution order (`rvc/train/utils.py:save_checkpoint`, lines 105–118):
    1. `torch.save` writes to the destination checkpoint path, truncating its contents as soon as writing begins (the prior valid checkpoint does not remain intact during this first write).
    2. A second `torch.save` writes the `_old_version.pth` temporary file.
    3. `os.replace` replaces the destination with that temporary file.
  - When saving G: The destination `G_2333333.pth` is truncated when the first `torch.save` begins writing it (~438 MB); a second `torch.save` writes the temporary backwards-compatible file `G_2333333_old_version.pth` (+438 MB transient); then `os.replace` replaces `_old_version.pth` into the destination file.
  - When saving D: The destination `D_2333333.pth` is truncated when the first `torch.save` begins writing it (~857 MB); a second `torch.save` writes the temporary file `D_2333333_old_version.pth` (+857 MB transient, which coexists momentarily with the newly written/truncated destination file, rather than an intact previous checkpoint); then `os.replace` replaces `_old_version.pth` into the destination file.
  - An interrupted Mode A save (e.g. process termination during writing) may leave an invalid or truncated checkpoint on disk; upon a later resumption attempt, failing to load both G and D will cause Applio to silently fall back to pretrained base weights.
  - Because G and D are saved sequentially, the maximum transient overhead above the steady-state pair footprint is **+857 MB [Calculated]** during the D save.
- **Mode B (`save_only_latest = False`)**:
  - Checkpoint filenames are versioned: `G_<global_step>.pth` and `D_<global_step>.pth`.
  - At each save interval, the target file is a new file that does not yet exist on disk.
  - However, because `save_checkpoint` executes `torch.save` twice (first writing the target file, then writing `_old_version.pth` before `os.replace`), both `D_<step>.pth` (857 MB) and `D_<step>_old_version.pth` (857 MB) exist simultaneously on disk between line 116 and line 118.
  - At the final (10th) interval of Full Target:
    - 9 prior checkpoint pairs are already accumulated on disk ($9 \times 1,295\text{ MB} = 11,655\text{ MB}$).
    - Interval 10 G is saved: writes `G_10` (438 MB).
    - Interval 10 D is saved: writes `D_10` (857 MB), then writes `D_10_old_version` (857 MB).
    - Momentary peak storage reaches: $11,655 + 438 + 857 + 857 = \mathbf{13,807\text{ MB}}$ (~13.81 GB for checkpoints alone; total disk ~15.7 to 16.5 GB).
    - Upon `os.replace`, `D_10_old_version` is renamed over `D_10`, and checkpoint storage settles at the steady-state final total of $12,950\text{ MB}$ (~12.95 GB).
    - Thus, because `save_checkpoint` performs duplicate writes rather than a single temp-and-rename, even in Mode B a transient overhead of **+857 MB** occurs during the final interval D write.
    - Recommending $\ge 16.5\text{ GB}$ (conservative $\ge 17\text{ GB}$) free disk space ensures safe operation.

*Details on Arithmetic and Settings Assumptions*:
1. **Smoke Test (`nvf_smoke_v0`)**:
   - Configuration: 253.58 s preview segment (~165 s active / 162 s voiced), `batch_size = 1`, `total_epoch = 3`, `save_every_epoch = 1`, `save_only_latest = True`, `save_every_weights = True`, `overtraining_detector = True`.
   - Produces exactly 4 exported `.pth` models:
     - 3 interval exports: `nvf_smoke_v0_1e_<step1>s.pth`, `nvf_smoke_v0_2e_<step2>s.pth`, `nvf_smoke_v0_3e_<step3>s.pth`.
     - 1 rolling best-epoch export: `nvf_smoke_v0_3e_<step3>s_best_epoch.pth` (epoch 2's best epoch is deleted on epoch 3 via `model_del` / `os.remove`).
     - Total: 4 × 55.2 MB = **220.8 MB (~221 MB) [Calculated]**.
   - Checkpoints: 1 G (`G_2333333.pth`, ~438 MB) and 1 D (`D_2333333.pth`, ~857 MB) = 1,295 MB (~1.295 GB).
   - Transient write peak: 1,781 MB + 857 MB = **2,638 MB (~2.64 GB) [Calculated]**.
2. **Experimental Short Model (`nvf_short_v1`, Mode A: `save_only_latest = True`)**:
   - Configuration: 253.58 s preview segment, default `batch_size = 1`, `total_epoch = 100`, `save_every_epoch = 25`, `save_only_latest = True`, `save_every_weights = True`, `overtraining_detector = True`.
   - Produces 5 exported `.pth` models:
     - 4 interval exports: epochs 25, 50, 75, 100.
     - 1 rolling `_best_epoch.pth` (overwritten/replaced rather than accumulated; prior best-epoch files deleted via `model_del`).
     - Total: 5 × 55.2 MB = **276.1 MB (~276 MB) [Calculated]**.
   - Checkpoints: Retains only the latest G and D pair = **1,295 MB (~1.30 GB) [Calculated]**.
   - Steady-state disk footprint: **~1.87 GB [Calculated]**.
   - Transient write peak (+857 MB during D write): **~2.73 GB [Calculated]** (requires $\ge 3.0$ GB free disk space).
3. **Full Target Dataset [Projection] (`nvf_full_v1`, Mode A: `save_only_latest = True`)**:
   - Configuration [Projection]: 20–30 minutes continuous studio recording corpus, `batch_size = 2` (if gated), `total_epoch = 250`, `save_every_epoch = 25`, `save_only_latest = True`, `save_every_weights = True`, `overtraining_detector = True`.
   - Produces 11 exported `.pth` models (epochs 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, plus 1 rolling `_best_epoch.pth`) = 11 × 55.2 MB = **~607 MB [Calculated]**.
   - Checkpoints: Retains only the latest G and D pair = **1,295 MB (~1.30 GB) [Calculated]**.
   - Steady-state disk footprint: **~3.2 to 3.9 GB [Calculated]**.
   - Transient write peak (+857 MB during D write): **~4.1 to 4.8 GB [Calculated]** (requires $\ge 5.0$ GB free disk space on drive C:).
4. **Full Target Dataset [Projection] (`nvf_full_v1`, Mode B: `save_only_latest = False`)**:
   - If the developer disables `save_only_latest` to retain all historical training checkpoints:
   - Checkpoints accumulate across all 10 save intervals (epochs 25, 50, ..., 250): 10 × 1,295 MB = **~12,950 MB (~12.95 GB) [Calculated]**.
   - Combined with feature cache (1.0–1.5 GB), FAISS index (200–330 MB), model exports (607 MB), and logs (150–200 MB), total steady-state storage reaches **~14.9 to 15.6 GB [Calculated]**.
   - Transient write peak during the final interval D write (+857 MB) reaches **~15.7 to 16.5 GB [Calculated]** (requires $\ge 16.5$ GB, conservative $\ge 17$ GB free disk space on drive C:).

---

## 5. Quality Evaluation & Epoch Selection Principles

### 5.1 Why Training Loss & `_best_epoch.pth` Are Insufficient

- Applio's `_best_epoch.pth` is produced purely by an exponential moving average heuristic over training batch losses.
- Lower generator loss frequently correlates with over-fitting to the training speaker's micro-prosody, producing a robotic, buzzy, or strained timbre.
- Training loss cannot evaluate downstream streaming stability, consonant intelligibility, tone contour preservation, or artifact suppression in silent intervals.
- The 3-epoch smoke test (`nvf_smoke_v0`) is strictly a technical check; **voice quality evaluation is deferred entirely to Phase 2**.

### 5.2 Step-by-Step Held-Out Listening & Checkpoint Selection Protocol

For subsequent quality experiments (`nvf_short_v1` or `nvf_full_v1`):

1. **Save Intermediate Inference Weights**:
   - Set `save_every_weights = True` with `save_every_epoch = 25` (or `20`) during training. This saves inference models at epochs 25, 50, 75, 100, 125, 150, 175, 200...
2. **Held-Out Test Conversion**:
   - For each candidate checkpoint, run offline conversion on:
     - **Target Female Speech (`data/held_out_test/`)**: Unseen recordings reserved before slicing. Tests identity retention, vowel clarity, and Vietnamese tone contour fidelity (ngang, huyền, sắc, hỏi, ngã, nặng).
     - **Male Fixture (`tests/fixtures/input_male.wav`)**: Standard project benchmark. Tests cross-gender pitch shifting (+12 semitones) and naturalness under real-time conditions.
3. **Blind Listening & Objective Criteria**:
   - Evaluate converted outputs using four explicit criteria:
     1. **Warmth & Naturalness**: Absence of metallic distortion, unnatural vibrato, or robotic harmonics.
     2. **Intelligibility**: Crisp consonants (especially aspirated stops /tʰ/, affricates /c/, /tɕ/, and fricatives /s/, /x/) without slurring.
     3. **Tone Contour Fidelity**: Natural contour trajectories across all 6 tones without pitch cracks or octave jumping.
     4. **Silence Stability**: Absence of breath noise amplification, breathing gasp artifacts, or background static during pauses.
4. **Checkpoint Selection**:
   - Declare the winning model based on human listening consensus and spectral cleanliness across the candidate set, rather than accepting the final epoch or `_best_epoch.pth` uncritically.

---

## 6. Execution Command Reference

All commands must be executed with working directory set to `third_party/applio` to ensure consistent resolution of `logs/` and pretrained models:

### Pre-requisite Base Model Placement

Download official Applio v2 base checkpoints into:

- `third_party/applio/rvc/models/pretraineds/pretrained_v2/f0G40k.pth` (**approximately 73.1 MB** `[Measured: 73,106,273 bytes]`)
- `third_party/applio/rvc/models/pretraineds/pretrained_v2/f0D40k.pth` (**approximately 142.9 MB** `[Measured: 142,875,703 bytes]`)

### Step 1: Dataset Quality Audit (from project root)

```powershell
.venv\Scripts\python scripts/validate_dataset.py --dataset-dir data/dataset_raw
```

---

### Step 2A: Audio Preprocessing for Smoke Test (`nvf_smoke_v0`)

```powershell
Push-Location third_party/applio
..\..\.venv\Scripts\python rvc/train/preprocess/preprocess.py `
    "logs/nvf_smoke_v0" `
    "../../data/dataset_raw" `
    40000 `
    3.0 `
    none `
    True `
    False `
    False `
    0.0
Pop-Location
```
*Argument Breakdown*:
- `logs/nvf_smoke_v0`: Dedicated experiment directory for the pipeline smoke test.
- `../../data/dataset_raw`: Input audio directory (strictly pre-split training audio).
- `40000`: Target sample rate (40 kHz).
- `3.0`: Target slice duration in seconds (`percentage`).
- `none`: Number of processes (`none` auto-selects CPU core count).
- `True`: `cut_preprocess` (**MUST BE `True`** to enable slicing via Slicer).
- `False`: `process_effects` (keep `False` for pre-cleaned studio audio).
- `False`: `noise_reduction` (keep `False` to prevent phase smearing).
- `0.0`: `reduction_strength`.

### Step 3A: Feature & Pitch Extraction for Smoke Test (`nvf_smoke_v0`)

```powershell
Push-Location third_party/applio
..\..\.venv\Scripts\python rvc/train/extract/extract.py `
    "logs/nvf_smoke_v0" `
    "rmvpe" `
    128 `
    4 `
    "0" `
    "v2" `
    "40000" `
    "contentvec"
Pop-Location
```
*Argument Breakdown*:
- `logs/nvf_smoke_v0`: Experiment directory.
- `"rmvpe"`: F0 pitch extractor.
- `128`: F0 hop length.
- `4`: Number of extraction processes.
- `"0"`: GPU ID (`"0"` for CUDA, `"-"` for CPU — actively parsed by `extract.py`).
- `"v2"`: Architecture version (v2 768-dim).
- `"40000"`: Sample rate.
- `"contentvec"`: Embedder model.
*(Note: `extract.py` automatically generates `logs/nvf_smoke_v0/config.json` via `generate_config` and formats `logs/nvf_smoke_v0/filelist.txt`).*

### Step 4A: Pipeline Correctness Smoke Test (3 Epochs, `save_every_epoch=1`)

> [!IMPORTANT]
> **Preflight Checklist**: Verify that `CUDA_VISIBLE_DEVICES` is set and that `logs/nvf_smoke_v0/config.json` has `fp16_run = true` before launching.

```powershell
# Mandatory Preflight Precision Check
$cfg = Get-Content "third_party/applio/logs/nvf_smoke_v0/config.json" | ConvertFrom-Json
if (-not $cfg.train.fp16_run) {
    throw "CRITICAL: fp16_run is false in config.json! Refusing to run in FP32 on 4 GB GPU."
}

# Explicitly isolate GPU 0
$env:CUDA_VISIBLE_DEVICES = "0"

Push-Location third_party/applio
..\..\.venv\Scripts\python rvc/train/train.py `
    "nvf_smoke_v0" `
    1 `
    3 `
    "rvc/models/pretraineds/pretrained_v2/f0G40k.pth" `
    "rvc/models/pretraineds/pretrained_v2/f0D40k.pth" `
    "v2" `
    "0" `
    1 `
    40000 `
    True `
    True `
    True `
    False `
    True `
    50 `
    False
Pop-Location
```
*Argument Breakdown (Smoke Test)*:
- `"nvf_smoke_v0"`: Dedicated experiment name (`sys.argv[1]`).
- `1`: `save_every_epoch` (exercises checkpointing and inference extraction on every epoch, `sys.argv[2]`).
- `3`: `total_epoch` (3 epochs strictly for pipeline verification, `sys.argv[3]`).
- `"rvc/models/pretraineds/pretrained_v2/f0G40k.pth"`: Pretrained G (`sys.argv[4]`).
- `"rvc/models/pretraineds/pretrained_v2/f0D40k.pth"`: Pretrained D (`sys.argv[5]`).
- `"v2"`: Version (`sys.argv[6]`).
- `"0"`: Unused GPU argument (`sys.argv[7]`, device binding handled by `$env:CUDA_VISIBLE_DEVICES`).
- `1`: `batch_size` (1 for safe VRAM verification, `sys.argv[8]`).
- `40000`: Sample rate (`sys.argv[9]`).
- `True`: `pitch_guidance` (`sys.argv[10]`).
- `True`: `save_only_latest` (cap checkpoint disk footprint, `sys.argv[11]`).
- `True`: `save_every_weights` (export inference `.pth` at every epoch, `sys.argv[12]`).
- `False`: `cache_data_in_gpu` (prevent VRAM exhaustion, `sys.argv[13]`).
- `True`: `overtraining_detector` (`sys.argv[14]`).
- `50`: `overtraining_threshold` (`sys.argv[15]`).
- `False`: `cleanup` (`sys.argv[16]`).

### Step 5A: Faiss Feature Index Extraction for Smoke Test (`nvf_smoke_v0`)

```powershell
Push-Location third_party/applio
..\..\.venv\Scripts\python rvc/train/process/extract_index.py `
    "logs/nvf_smoke_v0" "v2" "Auto"
Pop-Location
```

---

### Step 2B: Audio Preprocessing for Quality Run (`nvf_short_v1`)

> [!NOTE]
> **Complete Workflow Isolation**: The experimental quality run `nvf_short_v1` is an independent experiment. It runs its own preprocessing, extraction, preflight check, training, and indexing steps into `logs/nvf_short_v1/`, completely decoupled from `nvf_smoke_v0`. It does NOT share or depend on `config.json`, `filelist.txt`, features, checkpoints, or exports from `nvf_smoke_v0`.
>
> **Execution Gate**: Running the short experiment (`batch_size = 1` default) requires the Step 4A smoke test (`nvf_smoke_v0`) to pass and the user to explicitly review and approve proceeding. Free VRAM is not required to be demonstrated $\ge 1.5\text{ GiB}$ prior to launching at `batch_size = 1`. Considering `batch_size = 2` additionally requires observing at least 1.5 GiB free VRAM during the batch-size-1 run using `nvidia-smi`, plus explicit user approval.

```powershell
Push-Location third_party/applio
..\..\.venv\Scripts\python rvc/train/preprocess/preprocess.py `
    "logs/nvf_short_v1" `
    "../../data/dataset_raw" `
    40000 `
    3.0 `
    none `
    True `
    False `
    False `
    0.0
Pop-Location
```

### Step 3B: Feature & Pitch Extraction for Quality Run (`nvf_short_v1`)

```powershell
Push-Location third_party/applio
..\..\.venv\Scripts\python rvc/train/extract/extract.py `
    "logs/nvf_short_v1" `
    "rmvpe" `
    128 `
    4 `
    "0" `
    "v2" `
    "40000" `
    "contentvec"
Pop-Location
```
*(Extracts features into `logs/nvf_short_v1/`, generates `logs/nvf_short_v1/config.json`, and generates `logs/nvf_short_v1/filelist.txt`).*

### Step 4B: Mandatory Preflight Precision Check & Quality Fine-Tuning (`nvf_short_v1`)

```powershell
# Mandatory Preflight Precision Check for nvf_short_v1
$cfg = Get-Content "third_party/applio/logs/nvf_short_v1/config.json" | ConvertFrom-Json
if (-not $cfg.train.fp16_run) {
    throw "CRITICAL: fp16_run is false in logs/nvf_short_v1/config.json! Apply the non-destructive restoration rule before launching."
}

# Explicitly isolate GPU 0
$env:CUDA_VISIBLE_DEVICES = "0"

Push-Location third_party/applio
..\..\.venv\Scripts\python rvc/train/train.py `
    "nvf_short_v1" `
    25 `
    100 `
    "rvc/models/pretraineds/pretrained_v2/f0G40k.pth" `
    "rvc/models/pretraineds/pretrained_v2/f0D40k.pth" `
    "v2" `
    "0" `
    1 `
    40000 `
    True `
    True `
    True `
    False `
    True `
    50 `
    False
Pop-Location
```
*Argument Breakdown (Quality Run)*:
- `"nvf_short_v1"`: Dedicated experiment name (`sys.argv[1]`).
- `25`: `save_every_epoch` (exports candidate models at epochs 25, 50, 75, 100, `sys.argv[2]`).
- `100`: `total_epoch` (100 epochs fine-tuning, `sys.argv[3]`).
- `"rvc/models/pretraineds/pretrained_v2/f0G40k.pth"`: Pretrained G (`sys.argv[4]`).
- `"rvc/models/pretraineds/pretrained_v2/f0D40k.pth"`: Pretrained D (`sys.argv[5]`).
- `"v2"`: Version (`sys.argv[6]`).
- `"0"`: Unused GPU argument (`sys.argv[7]`, device binding handled by `$env:CUDA_VISIBLE_DEVICES`).
- `1`: `batch_size` (**Safe Default**: configured with `batch_size = 1` by default to preserve VRAM safety, `sys.argv[8]`).
  - Running `nvf_short_v1` at the default `batch_size = 1` requires only that the Step 4A smoke test (`nvf_smoke_v0`) has passed and the user has explicitly approved proceeding. Free VRAM is not required to be demonstrated $\ge 1.5\text{ GiB}$ prior to launching at `batch_size = 1`.
  - **Considering `batch_size = 2`**: Transitioning to or testing `batch_size = 2` additionally requires observing at least **1.5 GiB of free VRAM headroom** during the `batch_size = 1` run using `nvidia-smi`, plus explicit user approval.
  - Do NOT run either configuration during this planning phase.
- `40000`: Sample rate (`sys.argv[9]`).
- `True`: `pitch_guidance` (`sys.argv[10]`).
- `True`: `save_only_latest` (cap checkpoint disk footprint, `sys.argv[11]`).
- `True`: `save_every_weights` (export candidate inference models at every 25 epochs, `sys.argv[12]`).
- `False`: `cache_data_in_gpu` (prevent VRAM exhaustion, `sys.argv[13]`).
- `True`: `overtraining_detector` (`sys.argv[14]`).
- `50`: `overtraining_threshold` (`sys.argv[15]`).
- `False`: `cleanup` (`sys.argv[16]`).

### Step 5B: Faiss Feature Index Extraction for Quality Run (`nvf_short_v1`)

```powershell
Push-Location third_party/applio
..\..\.venv\Scripts\python rvc/train/process/extract_index.py `
    "logs/nvf_short_v1" "v2" "Auto"
Pop-Location
```

---

### Step 6: Model Validation & Deployment Guidelines

- Validate exported weights using the protocol in Section 3.6:
  - Load state dict with `strict=False`.
  - Assert zero unexpected keys.
  - Assert missing keys are exclusively posterior encoder (`enc_q.*`) keys.
  - Treat shape or dimension mismatches as immediate hard failures.
- For final production deployment:
  - Copy `third_party/applio/logs/<model_name>/<model_name>_<epoch>e_<step>s.pth` to `models/weights/<model_name>.pth`.
  - Copy `third_party/applio/logs/<model_name>/added_<model_name>_v2.index` to `models/indices/<model_name>.index`.

---

## 7. Dependency Baseline & Verified Fresh-Install Procedure

### 7.1 Python & NumPy Version Constraints

- **Python Baseline**: **Python 3.11** is the documented project baseline. In `pyproject.toml`, `requires-python = ">=3.10,<3.12"` explicitly excludes unverified Python 3.12+ environments.
- **NumPy Baseline**: **NumPy 1.x** is the intended training baseline, with **NumPy 1.26.4** as the initial verification target. The package manifest maintains `numpy>=1.23,<2.0`.
  > [!IMPORTANT]
  > **Existing Environment Status**: The currently existing local virtual environment (`.venv`) runs Python 3.11 with **NumPy 2.4.6** (installed prior to Milestone 4B). While previously reported unit tests passed, this existing environment **has not yet passed a training-path runtime test under the intended NumPy version (1.26.4)**. NumPy 2.x support on the Applio training path is unverified.
- **Module-Level Training Imports**: `pydub>=0.25.0` and `wget>=3.2` are required in `[project.optional-dependencies] train` because both are imported at module level in `third_party/applio/rvc/lib/utils.py` (`from pydub import AudioSegment` and `import wget`), which is loaded on the preprocessing and extraction paths whenever `preprocess.py` or `extract.py` imports `load_audio` or `load_embedding` (rather than being imported directly by `preprocess.py`). `stftpitchshift` is excluded from training extras because it is imported lazily for inference only.
- **Matplotlib Compatibility Constraint (`matplotlib==3.7.2`) & NumPy Relationship**:
  - In `third_party/applio/rvc/train/utils.py` (line 188), the training logging function `plot_spectrogram_to_numpy()` converts Matplotlib figures to raw RGB byte buffers via:
    ```python
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    ```
  - In `train.py` (lines 701–728), `plot_spectrogram_to_numpy()` is called on **every epoch** to generate three spectrogram images (`slice/mel_org`, `slice/mel_gen`, `all/mel`) for TensorBoard event logging.
  - **Compatibility Break**: `fig.canvas.tostring_rgb()` was deprecated in Matplotlib 3.8 and removed in Matplotlib 3.10 (in favor of `fig.canvas.buffer_rgba()`). The currently active development environment was previously reported as having Matplotlib 3.11.2, which is expected to fail with an `AttributeError` on epoch 1 of training (inferred from source code inspection and installed package metadata, not observed in a training run).
  - **Dependency Pin & Baseline Alignment**: To ensure reproducible compatibility with Applio's unpatched training logger, `pyproject.toml` pins `matplotlib==3.7.2` (matching Applio's pinned upstream `requirements.txt`). This pin is specifically intended for the declared NumPy 1.x baseline (`numpy>=1.23,<2.0`, with 1.26.4 as the planned verification target).
  - **Environment Status & Disclaimers**: The active environment was previously reported as running NumPy 2.4.6 and Matplotlib 3.11.2, and neither the dependency change nor logger compatibility has yet been runtime-tested there. Updating the dependency declaration in `pyproject.toml` does not modify the active virtual environment or prove runtime compatibility. The `matplotlib==3.7.2` pin should be validated together with NumPy < 2.0 in the intended training environment.
  - **Smoke-Test Blocker**: Execution of the Phase 1 smoke test (`nvf_smoke_v0`) **remains strictly blocked** until the training logger / Matplotlib compatibility is verified and resolved in the runtime environment.

### 7.2 Fresh-Install Execution Order (Documented Specification)

For setting up a clean, reproducible training environment from scratch:

1. **Create and Activate Python 3.11 Virtual Environment**:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
2. **Install PyTorch 2.5.1 with CUDA 12.1**:
   ```powershell
   pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
   ```
3. **Install Project Training Extras**:
   ```powershell
   pip install -e ".[train]"
   ```
4. **Apply Applio Streaming Patch**:
   ```powershell
   python scripts/setup_applio_patch.py --apply
   ```
5. **Validate Environment & Importability Before Training**:
   ```powershell
   python scripts/check_environment.py
   python -c "import torch, torchaudio, numpy, scipy, soundfile, librosa, faiss, sounddevice, transformers, tensorboard, noisereduce, torchcrepe, sklearn, matplotlib, pydub, wget; print('All training dependencies successfully imported!')"
   ```

*Specification Disclaimer*: This fresh-install sequence represents the documented installation specification. It has not yet been executed or validated on a completely clean environment during this task. No packages have been installed, uninstalled, or downgraded in the active environment.

---

## 8. Pre-Flight Readiness Checklist

- [ ] Target speaker provided explicit written/recorded consent for local voice-model training and personal voice conversion use.
- [ ] Dataset contains exclusively the single consenting female Northern Vietnamese speaker.
- [ ] Speech partitioned before preprocessing: training segment (253.58 s, ~165 s active / ~162 s voiced audio) in `data/dataset_raw/`, held-out segment (256.28–291.09 s) in `data/held_out_test/`, separated by verified 2.70 s natural-silence boundary (253.58–256.28 s).
- [ ] Training audio in `data/dataset_raw/` passes `scripts/validate_dataset.py` with zero clipping and zero decode errors.
- [ ] Pretrained `f0G40k.pth` (~73.1 MB `[Measured]`) and `f0D40k.pth` (~142.9 MB `[Measured]`) placed in `third_party/applio/rvc/models/pretraineds/pretrained_v2/`.
- [ ] Dedicated experiment directory (`logs/nvf_smoke_v0` or `logs/nvf_short_v1`) initialized cleanly to prevent accidental checkpoint resumption.
- [ ] `$env:CUDA_VISIBLE_DEVICES = "0"` configured in environment.
- [ ] Mandatory preflight precision check confirms `fp16_run: true` in `logs/<experiment>/config.json`.
- [ ] Matplotlib version confirmed compatible with Applio logger (requires Matplotlib < 3.10, pinned to 3.7.2 for NumPy < 2.0; smoke test blocked under Matplotlib >= 3.10 where logger is expected to fail due to `tostring_rgb()` removal).
- [ ] Idle VRAM confirmed $\ge 2.5\text{ GiB}$ via `nvidia-smi` prior to launching.
- [ ] Peak disk headroom confirmed: $\ge 2.7\text{ GB}$ for smoke test; for `nvf_short_v1`: $\ge 3.0\text{ GB}$ with `save_only_latest=True` (or $\ge 7.0\text{ GB}$ if `save_only_latest=False`); for full target projection: $\ge 5.0\text{ GB}$ with `save_only_latest=True` (or $\ge 16.5\text{ GB}$ if `save_only_latest=False`), accounting for steady-state checkpoints plus transient write overhead (+857 MB for D checkpoint).
