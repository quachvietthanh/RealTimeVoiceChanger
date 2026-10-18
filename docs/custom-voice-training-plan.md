# Safe Custom RVC v2 Voice Training Plan: Warm Northern Vietnamese Female Voice

## 1. Target Voice Profile & Objective

- **Identity**: Single adult female speaker.
- **Dialect / Accent**: Natural Northern Vietnamese (Hà Nội or Red River Delta standard conversational pronunciation and intonation).
- **Vocal Timbre**: Warm, soft, mellow, gentle, and clear; free from metallic, robotic, or strained characteristics.
- **Acoustic Objective**: Replace the foreign-sounding and sharp VCTK p231 baseline with an authentic, natural Northern Vietnamese feminine voice identity suitable for conversational voice conversion.
- **Baseline Preservation**: The existing VCTK p231 model (`models/weights/vctk_female_p231.pth`), its index (`models/indices/vctk_female_p231.index`), and all baseline audio outputs (`output/offline_test_result.wav`, `output/quality_tests/`) remain permanently preserved as reference benchmarks.

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

### 2.2 Recording Requirements & Purity

- **Volume**: **20 to 30 minutes** of net spoken audio (excluding long silences).
- **Format**: WAV PCM (16-bit or 24-bit), mono or stereo, 44.1 kHz or 48.0 kHz.
- **Acoustic Environment**: Quiet room or studio booth; zero background music, zero fan/AC hum, no room echo/reverberation, and no digital clipping (peak amplitude < 0.90 / -1.0 dBFS).
- **Content Diversity**: Natural conversational declarative sentences, questions, varied pacing, varied sentence lengths, expressive tone variation across all 6 Vietnamese tones (ngang, huyền, sắc, hỏi, ngã, nặng).
- **Speaker Purity**: Strictly **single-speaker**. Never mix recordings of multiple speakers into one training dataset.
- **Held-Out Test Subset (Mandatory)**:
  - Before preprocessing, reserve **1 to 2 minutes** of clean recordings (5–10 sentences) in `data/held_out_test/`.
  - This subset is **strictly excluded** from training to serve as an objective ground truth for post-training listening evaluations.

---

## 3. Verified Applio v3.2.8 Training Architecture & Command Semantics

All training components have been verified directly against the pinned source code in `third_party/applio/rvc/train/`:

### 3.1 Verification of Command Arguments (`train.py`)

| Argument Index | Parameter Name | Verified Type | Verified Behavior in Pinned v3.2.8 Source |
| :---: | :--- | :---: | :--- |
| `sys.argv[1]` | `model_name` | `str` | Name of the experiment directory under `logs/<model_name>`. |
| `sys.argv[2]` | `save_every_epoch` | `int` | **Checkpoint Frequency Interval**. Controls: (1) TensorBoard logging frequency, (2) saving full training checkpoints (`G_*.pth`, `D_*.pth`), and (3) if `save_every_weights=True`, exporting an inference model. |
| `sys.argv[3]` | `total_epoch` | `int` | Total maximum training epochs to execute. |
| `sys.argv[4]` | `pretrainG` | `str` | Filepath to pretrained generator weights (`f0G40k.pth`). |
| `sys.argv[5]` | `pretrainD` | `str` | Filepath to pretrained discriminator weights (`f0D40k.pth`). |
| `sys.argv[6]` | `version` | `str` | Architecture version (`v2` for 768-dim ContentVec). |
| `sys.argv[7]` | `gpus` | `str` | GPU identifier (`0` for `cuda:0`, or `-` for CPU). |
| `sys.argv[8]` | `batch_size` | `int` | Dataloader batch size per GPU. |
| `sys.argv[9]` | `sample_rate` | `int` | Target sampling rate (`40000`). |
| `sys.argv[10]` | `pitch_guidance` | `bool` | F0 guidance flag (`True` for voice conversion). |
| `sys.argv[11]` | `save_only_latest` | `bool` | **Disk Space Guard**: When `True`, checkpoints are named `G_2333333.pth` and `D_2333333.pth`, overwriting the previous interval's training weights. Prevents tens of GBs of checkpoint bloat. When `False`, saves `G_<step>.pth` for every interval. |
| `sys.argv[12]` | `save_every_weights` | `bool` | When `True`, calls `extract_model` to save a lightweight inference `.pth` **at every epoch divisible by `save_every_epoch`**. (It does *not* mean "every 25 epochs" by default; it strictly follows the interval set in `save_every_epoch`). |
| `sys.argv[13]` | `cache_data_in_gpu` | `bool` | **VRAM Guard**: When `True`, pre-loads all dataset tensors into VRAM, causing immediate CUDA OOM on a 4 GB card. **Must remain `False`** so batches are streamed dynamically from system RAM/disk. |
| `sys.argv[14]` | `overtraining_detector` | `bool` | Tracks smoothed exponential moving averages of Generator and Discriminator losses. |
| `sys.argv[15]` | `overtraining_threshold` | `int` | Number of consecutive evaluations without loss improvement before early stopping. |
| `sys.argv[16]` | `cleanup` | `bool` | Removes temporary training files upon completion. |

### 3.2 Checkpoint Types: Training vs. Inference Models

- **Training Checkpoints (`G_*.pth`, `D_*.pth`)**:
  - Contain full neural weights for Generator, Multi-Period Discriminator, Multi-Scale Discriminator, plus Adam optimizer states (momentum, variance, learning rate schedulers).
  - Size: ~160 MB each (**~320 MB per pair**).
  - Used exclusively for resuming training via `load_checkpoint`. Cannot be loaded by `RVCAdapter` for inference.
- **Inference Exports (`<model>_<epoch>e_<step>s.pth`)**:
  - Extracted via `extract_model`. Discriminator, optimizer states, and posterior encoder (`enc_q`) are stripped. Weights are cast to FP16. Architecture config (`config[4]=768`, 40 kHz) is packaged.
  - Size: **~55.2 MB**.
  - Ready for immediate loading and conversion in `RVCAdapter`.

### 3.3 Checkpointing & Recovery Mechanism

- Gradient checkpointing (`torch.utils.checkpoint`) is **not implemented** in Applio v3.2.8.
- "Checkpointing" in this engine refers strictly to state saving (`save_checkpoint`) and automatic resumption (`load_checkpoint` with `latest_checkpoint_path`). If training is interrupted, rerunning the command automatically detects `G_*.pth` and `D_*.pth` in the experiment directory and resumes from the exact epoch.

---

## 4. Hardware Constraints & VRAM Safety Strategy

### Target GPU: NVIDIA Quadro T2000 Mobile (4 GB VRAM)

Available VRAM on desktop is typically **~2.5 to 2.7 GiB** due to Windows DWM and running applications.

### Two-Phase Training Strategy

1. **Phase 1: Smoke Test (`batch_size = 1`)**:
   - Run a short 5-to-10 epoch smoke test with `batch_size = 1`.
   - Verify that data loading, forward pass, discriminator loss, generator loss, and FP16 GradScaler run without errors.
   - Measure actual peak VRAM consumption (`torch.cuda.max_memory_allocated`).
2. **Phase 2: Production Fine-Tuning (`batch_size = 2`)**:
   - Transition to `batch_size = 2` only if Phase 1 demonstrates at least **1.5 GiB of free VRAM headroom**.
   - Do NOT use `batch_size = 4` initially on this 4 GB card; keep it as a theoretical option only if batch size 2 shows large surplus margins.

### Itemized Disk Space Estimates

- Sliced audio & extracted features (`sliced_audios`, `sliced_audios_16k`, `f0`, `v2_extracted`): **~500–600 MB**.
- Training checkpoints (`G_2333333.pth`, `D_2333333.pth` with `save_only_latest = True`): **~320 MB fixed**.
- Intermediate inference models (~4–6 exported checkpoints at ~55 MB each): **~220–330 MB**.
- TensorBoard event logs: **~100–200 MB**.
- Faiss feature index: **~60–120 MB**.
- **Total Storage Required**: **~1.3 to 1.8 GB** (Well within the **48.8 GB free space** on drive C:).

---

## 5. Quality Evaluation & Epoch Selection Principles

1. **Do Not Promise That 250–300 Epochs is Optimal**:
   - Speech fine-tuning from a pretrained RVC v2 base often peaks in softness and naturalness between **100 and 200 epochs**.
   - Training beyond 200 epochs can over-fit the acoustic generator, causing the voice to sound metallic, robotic, or overly sharp.
2. **Do Not Rely Solely on the Overtraining Detector**:
   - The detector triggers on numerical loss smoothing, which does not evaluate human vocal warmth or Vietnamese tone contour naturalness.
3. **Mandatory Held-Out Listening Evaluation**:
   - At each intermediate export interval (e.g. every 25 epochs), perform offline conversions on:
     - The held-out target female speech in `data/held_out_test/` (evaluating vowel clarity and tone preservation).
     - The male source audio `tests/fixtures/input_male.wav` (evaluating cross-gender timbre transfer).
   - Human listening is the sole criterion for declaring the winning model checkpoint.

---

## 6. Execution Command Reference

All commands must be executed with working directory set to `third_party/applio` to ensure consistent resolution of `logs/` and pretrained models:

### Pre-requisite Base Model Placement

Download official Applio v2 base checkpoints into:

- `third_party/applio/rvc/models/pretraineds/pretrained_v2/f0G40k.pth` (~160 MB)
- `third_party/applio/rvc/models/pretraineds/pretrained_v2/f0D40k.pth` (~160 MB)

### Step 1: Dataset Quality Audit (from project root)

```powershell
.venv\Scripts\python scripts/validate_dataset.py --dataset-dir data/dataset_raw
```

### Step 2: Audio Preprocessing (Slicing & 40k Resampling)

```powershell
Push-Location third_party/applio
..\..\.venv\Scripts\python rvc/train/preprocess/preprocess.py `
    "logs/northern_vietnamese_female_v1" `
    "../../data/dataset_raw" `
    40000 3.0 none False False False 0.0
Pop-Location
```

### Step 3: Feature & Pitch Extraction (RMVPE + ContentVec 768-dim)

```powershell
Push-Location third_party/applio
..\..\.venv\Scripts\python rvc/train/extract/extract.py `
    "logs/northern_vietnamese_female_v1" `
    "rmvpe" 128 4 "0" "v2" "40000" "contentvec"
Pop-Location
```

### Step 4A: Initial Smoke Test (Batch Size 1, 10 Epochs)

```powershell
Push-Location third_party/applio
..\..\.venv\Scripts\python rvc/train/train.py `
    "northern_vietnamese_female_v1" `
    5 10 `
    "rvc/models/pretraineds/pretrained_v2/f0G40k.pth" `
    "rvc/models/pretraineds/pretrained_v2/f0D40k.pth" `
    "v2" "0" 1 40000 True True False False True 50 False
Pop-Location
```

### Step 4B: Production Fine-Tuning (Batch Size 2, up to 250 Epochs)

Note: Execute only after Step 4A verifies sufficient VRAM headroom.

```powershell
Push-Location third_party/applio
..\..\.venv\Scripts\python rvc/train/train.py `
    "northern_vietnamese_female_v1" `
    25 250 `
    "rvc/models/pretraineds/pretrained_v2/f0G40k.pth" `
    "rvc/models/pretraineds/pretrained_v2/f0D40k.pth" `
    "v2" "0" 2 40000 True True False False True 50 False
Pop-Location
```

### Step 5: Faiss Feature Index Extraction

```powershell
Push-Location third_party/applio
..\..\.venv\Scripts\python rvc/train/process/extract_index.py `
    "logs/northern_vietnamese_female_v1" "v2" "Auto"
Pop-Location
```

### Step 6: Deployment to Project Models Directory

- Trained weight: copy `third_party/applio/logs/northern_vietnamese_female_v1/northern_vietnamese_female_v1_<epoch>e_<step>s.pth` to:
  `models/weights/northern_vietnamese_female_v1.pth`
- Matching index: copy `third_party/applio/logs/northern_vietnamese_female_v1/added_northern_vietnamese_female_v1_v2.index` to:
  `models/indices/northern_vietnamese_female_v1.index`

---

## 7. Pre-Flight Readiness Checklist

- [ ] Target speaker provided explicit written/recorded consent for local voice-model training and personal voice conversion use.
- [ ] Dataset contains exclusively the single consenting female Northern Vietnamese speaker.
- [ ] 1–2 minutes of clean speech segregated into `data/held_out_test/` before preprocessing.
- [ ] Training audio in `data/dataset_raw/` passes `scripts/validate_dataset.py` with zero clipping and zero decode errors.
- [ ] Pretrained `f0G40k.pth` and `f0D40k.pth` verified and placed in `third_party/applio/rvc/models/pretraineds/pretrained_v2/`.
- [ ] Experiment name is unique (`northern_vietnamese_female_v1`) to prevent overwriting existing models.
- [ ] Idle VRAM confirmed $\ge 2.5\text{ GiB}$ via `nvidia-smi` prior to launching.
