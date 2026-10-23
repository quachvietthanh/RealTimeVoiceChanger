# Model Setup Guide: Model Assets & Prerequisites for Offline RVC Conversion

This guide specifies the exact filenames, directory structure, compatibility criteria, and legitimate download sources required for the first offline voice conversion in **RealTimeVoiceChanger**.

---

## 1. Required vs. Optional Asset Classification

| Asset Description | Filename | Expected Location | Status | Verified Exact Size | SHA-256 Checksum | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Target Voice Model** | `<model_name>.pth` | `models/weights/` | **REQUIRED** | ~50–120 MB | User-provided | RVC v2 generator weights for target female voice timbre. |
| **RMVPE Pitch Model** | `rmvpe.pt` | `models/checkpoints/` *(or `third_party/applio/rvc/models/predictors/`)* | **REQUIRED** | **181,184,272 bytes** (~181.18 MB) | `6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193` | High-precision fundamental frequency (F0) extraction. |
| **ContentVec Embedder Weights** | `pytorch_model.bin` | `models/checkpoints/contentvec/` *(or `third_party/applio/rvc/models/embedders/contentvec/`)* | **REQUIRED** | **378,342,945 bytes** (~378.34 MB) | `d8dd400e054ddf4e6be75dab5a2549db748cc99e756a097c496c099f65a4854e` | Speech representation extraction (HuggingFace Hubert format). |
| **ContentVec Embedder Config** | `config.json` | `models/checkpoints/contentvec/` *(or `third_party/applio/rvc/models/embedders/contentvec/`)* | **REQUIRED** | **1,388 bytes** (~1.39 KB) | `2ddde063b795d38d9051a7215a092fecf4cfe148b54251e38de51d88d356898b` | HubertModelWithFinalProj architecture config parameters. |
| **Input Speech Sample** | `<input_speech>.wav` | `tests/fixtures/` | **REQUIRED** | < 5 MB | User-provided | Clean, unclipped source male speech sample for testing. |
| **Feature Retrieval Index** | `<model_name>.index` | `models/indices/` | **OPTIONAL** | ~5–80 MB | User-provided | Faiss index for speaker feature matching. Operates without it. |
| **Training Pretraineds** | `f0G48k.pth`, `f0D48k.pth` | N/A | **NOT NEEDED** | N/A | N/A | Only used during model training, never during inference. |
| **Alternative Predictors** | `fcpe.pt`, `crepe` | N/A | **NOT NEEDED** | N/A | N/A | Only needed if alternative F0 methods are selected. |

> [!NOTE]
> **Resolution of Asset Size Discrepancies:**
>
> - **RMVPE (~16.8 MB vs. 181.18 MB)**: Earlier community references cited ~16.8 MB, which corresponds to quantized ONNX exports (`rmvpe.onnx`) or pruned F0 models. Applio Core v3.2.8 uses the full unquantized PyTorch E2E model (`rmvpe.pt`), which is verified at **181,184,272 bytes**.
> - **ContentVec (~180 MB vs. 378.34 MB)**: Legacy Fairseq checkpoints (`checkpoint_best_GID.pt` / `hubert_base.pt`) were ~180–190 MB. Applio v3.2.8 completely removed Fairseq in favor of HuggingFace `transformers` (`HubertModelWithFinalProj`). The full PyTorch HuggingFace checkpoint (`pytorch_model.bin`) includes all 12 transformer encoder layers, layer norms, and final projection weights in float32, verified at **378,342,945 bytes**.
> - **`hubert_base.pt` is NOT used**: Placing `hubert_base.pt` will have no effect in Applio v3.2.8. Only `pytorch_model.bin` + `config.json` in the `contentvec` directory are loaded.

---

## 2. Directory Structure

Place assets into the following local repository paths:

```text
RealTimeVoiceChanger/
├── models/
│   ├── weights/
│   │   └── <female_model_v2>.pth        # [REQUIRED] Target voice model weights
│   ├── indices/
│   │   └── <female_model_v2>.index      # [OPTIONAL] Matching Faiss index file
│   └── checkpoints/
│       ├── rmvpe.pt                     # [REQUIRED] RMVPE pitch predictor
│       └── contentvec/                  # [REQUIRED] HuggingFace ContentVec embedder
│           ├── pytorch_model.bin
│           └── config.json
└── tests/
    └── fixtures/
        └── male_speech_sample.wav       # [REQUIRED] Clean male input WAV fixture
```

---

## 3. Compatibility & Format Requirements

### A. Target Voice Model (`.pth`)

- **Container Format**: Serialized PyTorch state dictionary (`torch.save` output).
- **Mandatory Dictionary Keys**:
  - `weight`: State dictionary containing neural generator weights (specifically `emb_g.weight` and synthesizer layers).
  - `config`: Architecture tuple where `config[-1]` specifies target sample rate (`sr`) and `config[-3]` defines speaker count.
- **RVC Architecture Version**:
  - **RVC v2 (Strongly Recommended)**: Utilizes 768-dimensional ContentVec embeddings (`text_enc_hidden_dim = 768`).
  - **RVC v1**: Supported by Applio, but uses 256-dimensional embeddings with noticeably lower acoustic fidelity.
- **Pitch Guidance (`f0`)**:
  - Must have `f0 == 1` in the checkpoint dictionary.
  - Models trained with `f0 == 0` lack pitch predictor conditioning layers and **cannot** transpose male pitch (+12 semitones) to female pitch.
- **Sample Rate**: Typically 40,000 Hz (40k) or 48,000 Hz (48k).
- **VRAM / Precision**: Compatible with FP16 half-precision on NVIDIA Quadro T2000 (Turing 4 GB VRAM).

### B. Pitch Predictor (`rmvpe.pt`)

- **Format**: RMVPE E2E checkpoint (size: 16.8 MB).
- **Behavior**: Loaded unconditionally into GPU memory upon `Pipeline` initialization.

### C. Speech Embedder (`contentvec`)

- **Format**: HuggingFace pre-trained Hubert transformer checkpoint.
- **Required Files**:
  - `pytorch_model.bin`: ~180 MB model weights.
  - `config.json`: Model configuration parameters (hidden size 768, 12 layers, 12 heads).

### D. Feature Index (`.index` - Optional)

- **Format**: Faiss index (`IVF-PQ` or Flat index).
- **Omission Behavior**: Verified in Applio source code (`rvc/infer/pipeline.py` L558–566). When omitted or left empty, index feature blending (`_retrieve_speaker_embeddings`) is completely bypassed, and synthesis proceeds directly from ContentVec features.

### E. Input Speech WAV Fixture

- **Format**: Standard RIFF WAV format (PCM 16-bit or 32-bit float).
- **Channels**: Mono (1 channel).
- **Sample Rate**: 16,000 Hz, 44,100 Hz, or 48,000 Hz (automatically resampled to 16 kHz for F0 and feature extraction).
- **Duration**: 5 to 15 seconds of clean, spoken speech.
- **Acoustic Conditions**: Clean recording without background noise, music, heavy reverberation, or digital clipping (peak < -1.0 dBFS).

---

## 4. Legitimate & Verified Download Sources

Download official, verified weights directly from the upstream Applio repository on HuggingFace using the verified immutable commit revision `e908bdd5bf9f18cede26bbbec724b1a4bc7c2f14`:

### A. Auxiliary Predictor & Embedder Assets

| Asset | Immutable Download URL | Expected Size | Destination Path |
| :--- | :--- | :--- | :--- |
| `rmvpe.pt` | [Applio RMVPE (Commit e908bd)](https://huggingface.co/IAHispano/Applio/resolve/e908bdd5bf9f18cede26bbbec724b1a4bc7c2f14/Resources/predictors/rmvpe.pt) | 181,184,272 B (~181.18 MB) | `models/checkpoints/rmvpe.pt` |
| `pytorch_model.bin` | [ContentVec Weights (Commit e908bd)](https://huggingface.co/IAHispano/Applio/resolve/e908bdd5bf9f18cede26bbbec724b1a4bc7c2f14/Resources/embedders/contentvec/pytorch_model.bin) | 378,342,945 B (~378.34 MB) | `models/checkpoints/contentvec/pytorch_model.bin` |
| `config.json` | [ContentVec Config (Commit e908bd)](https://huggingface.co/IAHispano/Applio/resolve/e908bdd5bf9f18cede26bbbec724b1a4bc7c2f14/Resources/embedders/contentvec/config.json) | 1,388 B (~1.39 KB) | `models/checkpoints/contentvec/config.json` |

*Automated Downloader Script:*

A dedicated verification script is provided in the repository:

```powershell
.venv\Scripts\python scripts/download_auxiliary_assets.py
```

This script downloads to `.tmp` files, verifies SHA-256 and PyTorch architecture compatibility before committing files to their final destinations, and automatically syncs them to Applio's internal model directory.

At runtime `rmvpe.pt` is loaded by `VerifiedRMVPE0Predictor` (`src/voice_changer/conversion/rvc/rmvpe.py`): the file is read once, its SHA-256 must equal the value in the table above, and those bytes are deserialized with `torch.load(weights_only=True)`. A different or unreadable file stops model loading with an error; there is no fallback to Applio's unrestricted loader. Replacing `rmvpe.pt` with another build therefore also requires updating `RMVPE_SHA256`.

ContentVec is loaded the same way by `load_contentvec` (`src/voice_changer/conversion/rvc/contentvec.py`): `third_party/applio/rvc/models/embedders/contentvec/config.json` and `pytorch_model.bin` are each read once and must match the SHA-256 values in the table above; the weights are deserialized weights-only and the model is built from that in-memory state dict, so transformers never re-opens a file. Applio's own embedder loader (which downloads a missing file from an unpinned URL) is disabled on the adapter's Applio instance. A different file stops model loading with an error; updating ContentVec requires updating `CONTENTVEC_WEIGHTS_SHA256` / `CONTENTVEC_CONFIG_SHA256`.

### B. Female Voice Model (`.pth`)

- Obtain a verified, licensed RVC v2 female voice model (e.g., from an authorized public HuggingFace repository or personal training run).
- Save the `.pth` file into `models/weights/<model_name>.pth`.
- If an `.index` file accompanies the model, save it into `models/indices/<model_name>.index`.
- **Security Rule**: Never accept an unverified `.pth` file from untrusted forums or arbitrary links, as PyTorch checkpoints utilize Python pickle serialization.

---

## 5. Asset Validation Workflow

To audit whether all required assets are present and structurally valid without executing inference:

```powershell
.venv\Scripts\python scripts/offline_conversion_test.py
```

The script will perform:

1. Compatibility preflight check (Torch, NumPy bridge, SoundFile, Applio import).
2. Deep inspection of `.pth` header keys (`weight`, `config`, `version`, `f0`, `sr`).
3. Verification of `rmvpe.pt` and `contentvec` weights.
4. Input WAV inspection (duration, sample rate, channels, RMS, peak level).
5. Safe exit if any required component is missing or incompatible.

---

## 6. Security and Privacy Policy

1. **Git Exclusion**:
   - `models/weights/`, `models/checkpoints/`, `models/indices/`, and `tests/fixtures/*.wav` are strictly excluded in `.gitignore`.
   - Never force-add (`git add -f`) large binary models or private voice recordings to the repository.
2. **Preservation of Existing Assets**:
   - Setup scripts and adapter logic will never silently overwrite existing model files.

---

## 7. Applio Streaming Optimization Patch (Reproducibility)

Applio Core v3.2.8 is pinned at commit `0c92ac5` (`third_party/applio/`). For low-latency real-time inference (reducing chunk inference time from ~992 ms to ~269 ms), two surgical modifications are tracked in `patches/applio_streaming.patch`:

1. `set_padding()`: Dynamic streaming edge padding (0.1s instead of 1.0s/3.0s default), reducing HiFi-GAN Generator synthesis workload.
2. In-memory Faiss cache: Reusing preloaded `(index, big_npy)` tuple to eliminate the ~165 ms disk reload overhead per chunk.

### Reproducibility Commands

To verify, apply, or revert the patch on a clean checkout:

```powershell
# Check patch status
.venv\Scripts\python scripts/setup_applio_patch.py --check

# Apply patch to third_party/applio
.venv\Scripts\python scripts/setup_applio_patch.py --apply

# Revert to clean upstream Applio
.venv\Scripts\python scripts/setup_applio_patch.py --revert
```
