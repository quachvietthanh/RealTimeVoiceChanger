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
   - `preprocess.py` is pointed exclusively at the training directory of the dataset version assigned to the experiment (section 2.4). It is never pointed at `data/held_out_test/`. The held-out subset is never exposed to slicing, feature extraction, or model training.
   - *Constraint*: Do not create, move, or split any audio files during this planning task.
4. **Generalization Caveat**:
   - A short held-out segment taken from the same continuous recording session provides an internal sanity check for reconstruction and vowel clarity, **not evidence of generalizability** to new microphones, different room acoustics, varied emotional registers, or unconstrained conversational environments.

### 2.4 Dataset Versions and Experiment Assignment

The original split files are never modified. A dataset change creates a new directory with its own manifest that records its parent.

| Version | Training audio | Format | Peak | SHA-256 | Manifest |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `split_v1` (original) | `data/dataset_raw/female_train_source.wav` | 44.1 kHz, stereo, PCM_16, 253.58 s | 0.990845 (**fails** the < 0.90 rule) | `a13254f6…f09c853` | `data/source_audio/dataset_split_manifest.json` |
| `train_peak0p88` (derived from `split_v1`) | `data/dataset_train_peak0p88/female_train_source_peak0p88.wav` | 44.1 kHz, stereo, FLOAT32, 253.58 s | 0.880000 | `475af7f9…5e0bdf3` | `data/dataset_train_peak0p88/derived_manifest.json` |
| Held-out (shared, unchanged) | `data/held_out_test/female_holdout.wav` | 44.1 kHz, stereo, PCM_16, 34.81 s | 0.806 | `d025cd8d…fdf9129` | `data/source_audio/dataset_split_manifest.json` |

**`train_peak0p88` derivation (2026-10-21):**
- Created by `scripts/derive_gain_dataset.py`.
- Processing is a single linear gain: `float32(int16 / 32768 × 0.8881310829)`, which is −1.0305 dB. No clipping, limiting, resampling or channel mixing is applied.
- The gain comes from the measured source peak: 0.88 / (32468 / 32768).
- FLOAT32 storage avoids requantization. Because of that, `scripts/verify_dataset_manifest.py --derived <manifest>` can prove the derived file is sample-exact `input × gain`. It also re-verifies the full parent manifest first: 57/57 checks PASS.
- The derived manifest records:
  - the parent manifest SHA-256;
  - input and output SHA-256;
  - gain;
  - audio parameters;
  - tool versions (Python 3.11.9, NumPy 2.4.6, soundfile 0.14.0, libsndfile 1.2.2);
  - the owner's confirmation of speaker consent and of the listening check (2026-10-21).
- The training-only held-out audio is not derived; evaluation uses the original held-out file.

**Peaks along the Applio preprocess path** (`scripts/check_preprocess_peaks.py`, run with `rvc-train`, in memory) `[Measured]`:

| Stage | `split_v1` | `train_peak0p88` |
| :--- | :--- | :--- |
| Input file | 0.990845 | 0.880000 |
| `load_audio` mono @ 40 kHz | 0.992669 | 0.881620 |
| Slices @ 40 kHz | 0.992669 (106 slices) | 0.881620 (105 slices) |
| Slices @ 16 kHz (RMVPE/ContentVec input) | **1.004252** | 0.891908 |

Downmixing and resampling overshoot the input peak by up to +1.35 % at 16 kHz. That leaves `train_peak0p88` 0.008 below the limit. A higher target peak would not keep that margin.

The Slicer threshold (−42 dB) is absolute, so the −1.03 dB gain also changes slicing. `train_peak0p88` keeps 243.455 s instead of 244.880 s (−1.425 s, mostly quiet chunk edges in 5 of 55 chunks), which gives 105 slices instead of 106.

| Experiment | Dataset version | Status |
| :--- | :--- | :--- |
| `nvf_smoke_v0` | `split_v1` (one-time peak-rule exception, pipeline check only, ≤ 3 epochs) | Done 2026-10-21 |
| `nvf_short_v1` | `train_peak0p88` | Not run |

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
.venv\Scripts\python scripts/verify_dataset_manifest.py

# Derived dataset used by nvf_short_v1 (section 2.4)
.venv\Scripts\python scripts/validate_dataset.py --dataset-dir data/dataset_train_peak0p88
.venv\Scripts\python scripts/verify_dataset_manifest.py --derived data/dataset_train_peak0p88/derived_manifest.json
C:\Users\quach\venvs\rvc-train\Scripts\python scripts/check_preprocess_peaks.py data/dataset_train_peak0p88/female_train_source_peak0p88.wav
```

`verify_dataset_manifest.py` is read-only. It checks `data/source_audio/dataset_split_manifest.json` against the files on disk (SHA-256, size, sample rate, channels, format, frame counts, non-overlapping intervals, separate train/held-out directories, sample-exact copies of the source intervals) and exits non-zero on any failure. Do not preprocess if it fails.

---

### Step 2A: Audio Preprocessing for Smoke Test (`nvf_smoke_v0`)

```powershell
Push-Location third_party/applio
C:\Users\quach\venvs\rvc-train\Scripts\python rvc/train/preprocess/preprocess.py `
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
C:\Users\quach\venvs\rvc-train\Scripts\python rvc/train/extract/extract.py `
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
# Windows PyTorch builds lack libuv; without this, DDP init_process_group fails
$env:USE_LIBUV = "0"

Push-Location third_party/applio
C:\Users\quach\venvs\rvc-train\Scripts\python rvc/train/train.py `
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
> [!WARNING]
> **Exit code is not a success signal**: `train.py` runs training in a child process, and the parent returned exit code 0 even when the child crashed in `dist.init_process_group` (`use_libuv was requested but PyTorch was build without libuv support`, observed 2026-10-21 before `USE_LIBUV=0` was set). Confirm success from the log (`Saved model ... (epoch N)` for every epoch) and from the exported `<model>_<N>e_<step>s.pth` files.

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
C:\Users\quach\venvs\rvc-train\Scripts\python rvc/train/process/extract_index.py `
    "logs/nvf_smoke_v0" "v2" "Auto"
Pop-Location
```

### Recorded Run: `nvf_smoke_v0` (2026-10-21, pipeline check only)

Logs, VRAM samples, package freeze and the inference sample are in `output/nvf_smoke_v0_2026-10-21/`. The run used the Step 2A–5A commands with `rvc-train` and `USE_LIBUV=0`.

| Step | Result `[Measured]` |
| :--- | :--- |
| Input | `data/dataset_raw/female_train_source.wav` (manifest-verified, 44.1 kHz stereo PCM_16, 253.58 s, peak 0.9908, 0 full-scale samples, 0 flat-top runs) |
| 2A preprocess | 8.7 s; `load_audio` downmixes to mono and resamples to 40 kHz; no peak normalization (`process_effects=False`); 106 float32 slices (0.43–3.24 s), 40 kHz peak 0.9927 |
| 3A extract | 38 s; 106/106 f0, f0_voiced, v2 features; no errors or NaN; peak VRAM used 1315 MiB; `fp16_run: true` |
| 4A train (3 epochs, batch 1) | 486 s (~2.5 min/epoch); peak VRAM used 3201 MiB of 4096 (whole GPU, including ~620 MiB desktop baseline); loss_g 20.76 → 14.87, loss_d 4.34 → 5.47 (epochs 1–2 in `training_data.json`) |
| Exports | `nvf_smoke_v0_{1e_108s,2e_216s,3e_324s}.pth`, `nvf_smoke_v0_3e_324s_best_epoch.pth` (~55 MB, FP16); section 3.6 key validation PASS on all four (103 missing keys, all `enc_q.*`; 0 unexpected, 0 shape mismatch, 0 non-finite) |
| 5A index | `added_nvf_smoke_v0_v2.index`, IVFFlat, d=768, 12,920 vectors |
| Adapter inference (`.venv`) | `scripts/offline_conversion_test.py --language vi` on `tests/fixtures/input_male.wav`: RTF 0.351, 40 kHz mono, no clipping, no silence |

Limitations:
- The dataset peak (0.9908) does not meet the < 0.90 rule in section 2.2. The user accepted this exception only for `nvf_smoke_v0` (≤ 3 epochs). It does not apply to `nvf_short_v1` or later runs.
- Three epochs on 4 minutes of audio do not show anything about voice quality. Nobody has listened to the output.
- The first launch failed on libuv. That failed run left two empty TensorBoard event files and a `process_pids` field in `config.json`, but no G/D checkpoints.

---

### Step 2B: Audio Preprocessing for Quality Run (`nvf_short_v1`)

> [!NOTE]
> **Complete Workflow Isolation**: The experimental quality run `nvf_short_v1` is an independent experiment. It runs its own preprocessing, extraction, preflight check, training, and indexing steps into `logs/nvf_short_v1/`, completely decoupled from `nvf_smoke_v0`. It does NOT share or depend on `config.json`, `filelist.txt`, features, checkpoints, or exports from `nvf_smoke_v0`.
>
> **Execution Gate**: Running the short experiment (`batch_size = 1` default) requires the Step 4A smoke test (`nvf_smoke_v0`) to pass and the user to explicitly review and approve proceeding. Free VRAM is not required to be demonstrated $\ge 1.5\text{ GiB}$ prior to launching at `batch_size = 1`. Considering `batch_size = 2` additionally requires observing at least 1.5 GiB free VRAM during the batch-size-1 run using `nvidia-smi`, plus explicit user approval.

```powershell
Push-Location third_party/applio
C:\Users\quach\venvs\rvc-train\Scripts\python rvc/train/preprocess/preprocess.py `
    "logs/nvf_short_v1" `
    "../../data/dataset_train_peak0p88" `
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
C:\Users\quach\venvs\rvc-train\Scripts\python rvc/train/extract/extract.py `
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
# Windows PyTorch builds lack libuv; without this, DDP init_process_group fails
$env:USE_LIBUV = "0"

Push-Location third_party/applio
C:\Users\quach\venvs\rvc-train\Scripts\python rvc/train/train.py `
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
C:\Users\quach\venvs\rvc-train\Scripts\python rvc/train/process/extract_index.py `
    "logs/nvf_short_v1" "v2" "Auto"
Pop-Location
```

### Recorded Run: `nvf_short_v1` (2026-10-21, short experiment)

Launcher, logs, VRAM samples, evaluation outputs and metrics are in `output/nvf_short_v1_2026-10-21/` (`run_4b_train.sh`, `4b_train.log`, `4b_vram_samples.csv`, `5b_index.log`, `run_eval.sh`, `eval/`). The run used the Step 2B–5B commands with `rvc-train` and `USE_LIBUV=0`. The owner confirmed voice-use consent and a listening check of the training audio before launch.

| Step | Result `[Measured]` |
| :--- | :--- |
| Input | `data/dataset_train_peak0p88/female_train_source_peak0p88.wav` (derived, gain 0.8881, peak 0.8800; `--derived` manifest check PASS) |
| 2B preprocess | 105 slices; peak 0.8816 (40 kHz), 0.8919 (16 kHz) |
| 4B train (100 epochs, batch 1) | Completed 100/100 epochs, 10,700 steps (`Training has been successfully completed` in log; overtraining detector never triggered). Wall time 11:15:00–15:45:59 including a ~36 min laptop sleep (12:59–13:35); ~2:20 per epoch, ~3 h 55 min of actual training. Peak GPU memory 3340 MiB of 4096 (whole GPU); watchdog never fired. Smoothed loss_gen 15.39 (e25) → 13.17 (e50) → 11.26 (e75) → 9.74 (e100); loss_disc ~4.4 → 4.09. No error/traceback/NaN in the log |
| Exports | `nvf_short_v1_{25e_2675s,50e_5350s,75e_8025s,100e_10700s}.pth` and `nvf_short_v1_100e_10700s_best_epoch.pth`; section 3.6 key validation PASS on all five |
| 5B index | `added_nvf_short_v1_v2.index`, IVFFlat, d=768, 12,835 vectors |

Offline evaluation (`.venv`, `scripts/offline_conversion_test.py --language vi`, index above; baseline `vctk_female_p231`). Metrics from `analyze_conversions.py` (RMVPE F0); these are objective proxies, not a quality judgement:

| Input | Model | RTF | f0_corr_st | median shift (st) | octave err % | voiced recall % | peak |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| holdout, pitch 0 | e25 | 0.303 | 0.997 | 0.00 | 0.00 | 98.9 | 0.629 |
| | e50 | 0.301 | 0.997 | 0.00 | 0.00 | 99.0 | 0.860 |
| | e75 | 0.304 | 0.998 | 0.01 | 0.00 | 98.7 | 0.898 |
| | e100 | 0.302 | 0.988 | -0.01 | 0.18 | 98.5 | 0.803 |
| | baseline | 0.309 | 0.994 | 0.01 | 0.04 | 97.4 | 0.809 |
| `input_male`, pitch +12 | e25 | 0.306 | 0.982 | 11.99 | 0.10 | 99.0 | 0.599 |
| | e50 | 0.307 | 0.981 | 11.99 | 0.10 | 98.8 | 0.870 |
| | e75 | 0.307 | 0.982 | 12.01 | 0.11 | 98.1 | 0.863 |
| | e100 | 0.307 | 0.980 | 11.98 | 0.11 | 98.1 | 0.816 |
| | baseline | 0.313 | 0.989 | 12.02 | 0.05 | 98.1 | 0.695 |

No output clipped or went silent; dropout (output < -50 dBFS on input-voiced frames) was 0.41–0.78 % for all outputs.

Limitations:
- Nobody has listened to these outputs. F0 correlation measures contour tracking, not timbre, naturalness or how close the result is to the target speaker. The owner needs to listen and choose a checkpoint.
- The holdout is recorded by the target speaker, so pitch-0 conversion is close to reconstruction and favours every model. On `input_male`, the custom checkpoints scored slightly lower f0_corr than the vctk baseline (0.980–0.982 vs 0.989). The audit below shows this gap is within run-to-run sampling noise, so it does not rank the models.
- `_best_epoch` is Applio's own pick (epoch 100 here); the lowest single-step loss_gen was at epoch 93, which was not exported.
- Only one male input clip was tested.

Owner listening (2026-10-21): `baseline_male` sounds robotic; e25/e50/e75 sound nearly the same; e100 sounds less natural. No checkpoint is accepted. `e50` is a temporary preview checkpoint only.

### Inference Audit: `nvf_short_v1` (2026-10-21)

Outputs, configs and scripts are in `output/nvf_short_v1_audit_2026-10-21/` (see its `README.md` for the listening order). All runs use `tests/fixtures/input_male.wav` and `RVCAdapter.convert_file`, starting from the `vi` profile (pitch +12, rmvpe, index_rate 0.50, filter_radius 2, protect 0.40, hop 128), with torch seeded before each conversion and one parameter changed at a time. `[Measured]`:

- The original `eval/` comparison was valid. Every model ran through the same script and profile. The only differences were the model file and its index: the custom index for the custom models, and the VCTK index for the baseline. Each old `eval/*_male.wav` is closest to a new seeded output of the same model (log-mel distance 1.8–2.6 dB vs ≥ 3.5 dB to other models), so no file was mixed up or reused.
- The checkpoints really differ. Weight rel-L2 from pretrained is 0.058 (e25), 0.081 (e50), 0.099 (e75) and 0.114 (e100); between neighbouring exports it is 0.046–0.049. `_best_epoch` is bit-identical in weights to e100.
- Output differences between e25/e50/e75/e100 are small: 3.1–3.7 dB log-mel, against a same-model seed-to-seed noise floor of 1.8–2.3 dB. Custom-vs-baseline is 6.7–7.3 dB. This matches the owner hearing e25–e75 as nearly the same.
- RVC inference is stochastic (`torch.randn_like` in `synthesizers.py` infer). Unseeded f0_corr moved by up to 0.01 between runs: e100 scored 0.980 in `eval/` and 0.992 seeded. f0_corr differences below this do not rank models.
- index_rate 0–0.75 and protect 0.40 vs 0.50 change the output by 0.5–2.5 dB, about the same as the seed noise floor, for both e50 and the baseline. index_rate 0 is bit-identical to loading no index. protect ≥ 0.5 disables protection in Applio (`pipeline.py`, `if protect < 0.5`).
- Crude timbre proxy (MFCC-mean distance to the holdout speaker; unconverted input is 117): custom models 20–29, baseline 65–68. Lower pitch moves e50 further from the target (+10: 37, +8: 48). The input median F0 is 132 Hz and the holdout median is 279 Hz; +12 gives about 264 Hz.
- No clipping or silence. Durations were 32.20 s for every output. RTF was 0.23–0.32 (includes the first-run warm-up).

The `vi` profile lowers index_rate to 0.50 because the VCTK index is English. That reason does not apply to the Vietnamese `nvf_short_v1` index. The audit did not change the default.

### Speaker B Dataset (2026-10-21, prepared, not trained)

Speaker B is a separate dataset. Never mix it with speaker A. The owner confirmed consent verbally on 2026-10-21; no consent document is stored in the repository. The owner states the speech is mainly English. The goal is to borrow B's timbre for Vietnamese male input, not to reproduce a native Northern Vietnamese voice.

- Manifest `data/speaker_b/dataset_split_manifest.json` (verify with `scripts/verify_dataset_manifest.py --manifest ...`: 39/39 PASS). Train `data/speaker_b/dataset_raw/speaker_b_train.wav` covers 0–177.52 s. A 0.40 s guard pause follows. Holdout `data/speaker_b/held_out_test/speaker_b_holdout.wav` covers 177.92–205.10 s. Both are sample-exact PCM_16 48 kHz mono copies.
- The source WAV is a decode of a 128 kbps MP3 (`output/speaker_b_inspection_2026-10-21/inspection.txt`).
- No gain change: peak 0.279 at every preprocess stage (file, 40 kHz, slices, 16 kHz). Planned preprocess: 40 kHz, `per` 3.0, `process_effects=False`, no noise reduction. Expected 77 train slices.
- Train has 117 s of active speech and 120 s RMVPE-voiced; F0 median 331 Hz (p5–p95 261–418 Hz). Holdout has 17 s active.
- Preparation scripts and results: `output/speaker_b_prep_2026-10-21/`.

### Recorded Run: `spkb_preview_v2` (2026-10-21, speaker B preview, 25 epochs)

`spkb_preview_v1` crashed in epoch 8 because an external cleanup deleted `sliced_audios` (see `output/spkb_preview_v1_2026-10-21/INCIDENT_2026-10-21_file_deletion.md`). `spkb_preview_v2` is a fresh run from pretrained f0G40k/f0D40k with new preprocess and extract. Files are in `output/spkb_preview_v2_2026-10-21/`.

Cache protection, used for this run and reusable for later ones:
- `.training_active/<exp>.json` lock, plus the `GEMINI.md` cleanup rules.
- `cache_guard.py snapshot` covers 392 required files with SHA-256: the cache, `filelist.txt`, the referenced mute files and the pretrained G/D.
- `cache_guard.py protect` sets read-only on the 386 cache files the experiment owns and records the list in `readonly_files.txt`. `unprotect` restores only those files. Read-only blocks Python `shutil.rmtree`/`os.remove` and `Remove-Item` without `-Force`. It does not block `Remove-Item -Force` or `rm -rf`.
- `run_train.sh` checks activity and cache integrity before launch. Every 10 s it runs a cache quick-check, stopping training and logging the missing files and the last epoch; it also warns about cleanup-like processes. The VRAM watchdog is unchanged.
- The cache watchdog was dry-run on a fake experiment: a deleted slice was detected and the fake trainer was stopped within 4 s.

| Step | Result `[Measured]` |
| :--- | :--- |
| Preprocess / extract | 77 slices; 77/77 f0, f0_voiced and v2 features consistent (f0 length − 2 × feature length = 1–3, same as `nvf_short_v1`); f0 identical to `spkb_preview_v1`; extract peak VRAM 1552 MiB |
| Train (25 epochs, batch 1, save/export every 5) | 17:35:53–18:20:59, 1975 steps, about 1:46 per epoch. Peak GPU memory 3536 MiB. No watchdog alarm. Smoothed loss_gen 15.93 (e2) → 15.69 (e5) → 15.20 (e10) → 14.65 (e15) → 14.12 (e20) → 13.63 (e25); loss_disc about 4.2 |
| Exports | `spkb_preview_v2_{5e_395s,10e_790s,15e_1185s,20e_1580s,25e_1975s}.pth` and `_25e_1975s_best_epoch.pth`; key validation PASS on all six |
| Integrity | Cache verify after training 392/392 unchanged (training added 77 `.spec.pt`, as Applio does); 948/948 baseline hashes unchanged (data A/B, manifests, models, all other experiments' checkpoints and indexes, evaluation WAVs) |

No index has been built and no evaluation has been run yet. Loss values say nothing about voice quality.

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
  > **Environment Split (verified 2026-10-21)**: Training runs only in the dedicated environment `C:\Users\quach\venvs\rvc-train` (Python 3.11.9, NumPy 1.26.4, Matplotlib 3.7.2, PyTorch 2.5.1+cu121). The inference environment `.venv` keeps NumPy 2.4.6 / Matplotlib 3.11.2 and must not be used for training: `scripts/check_training_env.py` fails there on the Matplotlib logger check.
- **Module-Level Training Imports**: `pydub>=0.25.0` and `wget>=3.2` are required in `[project.optional-dependencies] train` because both are imported at module level in `third_party/applio/rvc/lib/utils.py` (`from pydub import AudioSegment` and `import wget`), which is loaded on the preprocessing and extraction paths whenever `preprocess.py` or `extract.py` imports `load_audio` or `load_embedding` (rather than being imported directly by `preprocess.py`). `stftpitchshift` is excluded from training extras because it is imported lazily for inference only.
- **Matplotlib Compatibility Constraint (`matplotlib==3.7.2`) & NumPy Relationship**:
  - In `third_party/applio/rvc/train/utils.py` (line 188), the training logging function `plot_spectrogram_to_numpy()` converts Matplotlib figures to raw RGB byte buffers via:
    ```python
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    ```
  - In `train.py` (lines 701–728), `plot_spectrogram_to_numpy()` is called on **every epoch** to generate three spectrogram images (`slice/mel_org`, `slice/mel_gen`, `all/mel`) for TensorBoard event logging.
  - **Compatibility Break**: `fig.canvas.tostring_rgb()` was deprecated in Matplotlib 3.8 and removed in Matplotlib 3.10 (in favor of `fig.canvas.buffer_rgba()`). In `.venv` (Matplotlib 3.11.2) the call raises `AttributeError: 'FigureCanvasAgg' object has no attribute 'tostring_rgb'` `[Measured 2026-10-21 via scripts/check_training_env.py]`.
  - **Dependency Pin & Baseline Alignment**: To ensure reproducible compatibility with Applio's unpatched training logger, `pyproject.toml` pins `matplotlib==3.7.2` (matching Applio's pinned upstream `requirements.txt`) together with NumPy 1.x.
  - **Verified Status**: In `rvc-train` (Matplotlib 3.7.2, NumPy 1.26.4), `plot_spectrogram_to_numpy()` returns a `uint8 (200, 1000, 3)` image `[Measured 2026-10-21]`.

### 7.2 Training Environment Install (`rvc-train`, executed 2026-10-21)

The training environment lives outside the repository and is separate from the inference `.venv`. Never install training packages into `.venv`.

1. **Create the venv with Python 3.11.9**:
   ```powershell
   & "C:\Users\quach\AppData\Local\Programs\Python311\python.exe" -m venv C:\Users\quach\venvs\rvc-train
   $py = "C:\Users\quach\venvs\rvc-train\Scripts\python.exe"
   ```
2. **Install PyTorch 2.5.1 with CUDA 12.1**:
   ```powershell
   & $py -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
   ```
3. **Install training dependencies** (shared packages pinned to the versions verified in `.venv`; NumPy and Matplotlib pinned for the Applio logger):
   ```powershell
   & $py -m pip install numpy==1.26.4 matplotlib==3.7.2 scipy==1.17.1 soundfile==0.14.0 librosa==0.11.0 `
       faiss-cpu==1.15.1 transformers==4.49.0 tensorboard==2.21.0 noisereduce==3.0.3 torchcrepe==0.0.24 `
       scikit-learn==1.9.1 pydub==0.25.1 wget==3.2 tqdm
   & $py -m pip check
   ```
   The resolver kept torch 2.5.1+cu121, numpy 1.26.4 and matplotlib 3.7.2 without conflicts, and `pip check` reported no broken requirements. The full resolved package list is saved in `output/nvf_smoke_v0_2026-10-21/rvc-train-freeze.txt`.
4. **Run the training preflight** (from the repository root):
   ```powershell
   & $py scripts/check_training_env.py
   ```
   It checks the interpreter, pinned versions, a small CUDA FP32/FP16 workload, the training-path imports, the Applio spectrogram logger, `fp16_run`, and the pretrained/auxiliary SHA-256 values. Result on 2026-10-21: 10/10 PASS in `rvc-train`. In `.venv` it fails as expected (NumPy, Matplotlib and logger checks).

The Applio streaming patch (`scripts/setup_applio_patch.py`) affects only inference and is applied to the shared `third_party/applio` checkout, so it needs no step here.

`pydub` prints `Couldn't find ffmpeg or avconv` on import. This does not matter for WAV training input, because `rvc/lib/utils.py:load_audio` reads audio with `soundfile`.

---

## 8. Pre-Flight Readiness Checklist

- [ ] Target speaker provided explicit written/recorded consent for local voice-model training and personal voice conversion use.
- [ ] Dataset contains exclusively the single consenting female Northern Vietnamese speaker.
- [ ] Speech partitioned before preprocessing: training segment (253.58 s, ~165 s active / ~162 s voiced audio) in `data/dataset_raw/`, held-out segment (256.28–291.09 s) in `data/held_out_test/`, separated by verified 2.70 s natural-silence boundary (253.58–256.28 s).
- [ ] Training audio for the experiment's dataset version (section 2.4) passes `scripts/validate_dataset.py` with zero clipping and zero decode errors. Derived versions must also pass `scripts/verify_dataset_manifest.py --derived`, and `scripts/check_preprocess_peaks.py` must report all stages, including 16 kHz slices, below 0.90.
- [ ] Pretrained `f0G40k.pth` (~73.1 MB `[Measured]`) and `f0D40k.pth` (~142.9 MB `[Measured]`) placed in `third_party/applio/rvc/models/pretraineds/pretrained_v2/`.
- [ ] Dedicated experiment directory (`logs/nvf_smoke_v0` or `logs/nvf_short_v1`) initialized cleanly to prevent accidental checkpoint resumption.
- [ ] `$env:CUDA_VISIBLE_DEVICES = "0"` configured in environment.
- [ ] Mandatory preflight precision check confirms `fp16_run: true` in `logs/<experiment>/config.json`.
- [ ] Matplotlib version confirmed compatible with Applio logger (requires Matplotlib < 3.10, pinned to 3.7.2 for NumPy < 2.0; smoke test blocked under Matplotlib >= 3.10 where logger is expected to fail due to `tostring_rgb()` removal).
- [ ] Idle VRAM confirmed $\ge 2.5\text{ GiB}$ via `nvidia-smi` prior to launching.
- [ ] Peak disk headroom confirmed: $\ge 2.7\text{ GB}$ for smoke test; for `nvf_short_v1`: $\ge 3.0\text{ GB}$ with `save_only_latest=True` (or $\ge 7.0\text{ GB}$ if `save_only_latest=False`); for full target projection: $\ge 5.0\text{ GB}$ with `save_only_latest=True` (or $\ge 16.5\text{ GB}$ if `save_only_latest=False`), accounting for steady-state checkpoints plus transient write overhead (+857 MB for D checkpoint).
