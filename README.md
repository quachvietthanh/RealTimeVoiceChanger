# RealTimeVoiceChanger

A real-time male-to-female voice changer application for Windows using Retrieval-based Voice Conversion (RVC v2).

## Project Purpose

The primary objective of this project is to create a high-quality, low-latency male-to-female voice changer for Windows using Retrieval-based Voice Conversion (RVC v2) with RMVPE pitch estimation and ContentVec feature extraction.

The system is designed for live conversational use in voice platforms like Discord, OBS, and games via virtual audio drivers (such as VB-Audio Virtual Cable), with specialized tuning for Vietnamese tonal speech.

---

## Current Project Status: Milestone 4 In Progress

> **Status:** **Operational Streaming Prototype & Custom Voice Training Preparation**.
>
> - **Inference Engine**: Applio Core v3.2.8 (`rvc.infer.infer.VoiceConverter`) running locally on PyTorch 2.5.1 + CUDA 12.1.
> - **Hardware Baseline**: NVIDIA Quadro T2000 Mobile (4 GB VRAM) on Windows 11.
> - **Real-Time Factor (RTF)**: **0.562** (Mean inference: ~278 ms per 500 ms audio chunk).
> - **5-Minute Continuous Validation**: **600/600 chunks** processed with **0 steady-state dropouts**, **0 sample mismatches**, and stable **~445 MB VRAM**.
> - **Active Inference Voice Model**: Technical baseline model `vctk_female_p231` (VCTK female speaker p231, RVC v2, 40 kHz, HiFi-GAN).
> - **Custom Voice Training (Milestone 4)**: Target Northern Vietnamese female vocal split prepared (`female_train_source.wav` 232.98s and `female_holdout.wav` 31.98s) with cryptographic manifest verification. Isolated training environment architecture (`rvc-train`) designed to maintain strict dependency isolation.
> - **Voice profiles (2026-10-21)**: three selectable profiles in `config/voice_profiles.json`: `vctk_p231` (prototype, default), `voice_a_preview` (speaker A, nvf_short_v1 e50) and `voice_b_preview` (speaker B, spkb_preview_v2 e25). A and B are **previews pending owner listening**; neither is approved for production. See [docs/voice-profiles.md](docs/voice-profiles.md).
> - **Backend API for the GUI**: in-process `VoiceChangerService` ([docs/gui-backend-contract.md](docs/gui-backend-contract.md)), headless contract tests. Physical microphone/VB-Cable routing not yet verified on hardware ([docs/audio-routing.md](docs/audio-routing.md)).

---

## Development Milestones

| Milestone | Description | Status |
| :--- | :--- | :--- |
| **Milestone 1** | **Environment & Hardware Assessment**: Python 3.11, CUDA 12.1, PyTorch 2.5.1, Applio v3.2.8 pinned. | **DONE** |
| **Milestone 2** | **Offline WAV Conversion**: RVCAdapter, RMVPE, ContentVec, offline baseline conversion (RTF = 0.345). | **DONE** |
| **Milestone 3** | **Real-Time Streaming Prototype**: Non-blocking I/O, async bounded worker thread, PortAudio output stream. | **DONE** |
| **Milestone 3A** | **Latency Reduction**: Faiss in-RAM caching, 0.1s streaming padding, 0.5s chunk configuration, RTF reduced from 0.88 to 0.56. | **DONE** |
| **Milestone 3A.1** | **Long-Run Streaming Validation**: 5-minute continuous test (600 chunks), 80-sample reflection padding, 0 steady dropouts. | **DONE** |
| **Milestone 4A** | **Custom Voice Sourcing & Vocal Separation**: High-quality Northern Vietnamese female vocal extraction, listening test approval. | **DONE** |
| **Milestone 4B** | **Training Preparation & Isolated Environment**: Dataset split, cryptographic manifest, isolated `rvc-train` venv engineering. | **DONE** (training runs executed in `rvc-train`) |
| **Milestone 4C** | **Custom Voice Model Training**: smoke `nvf_smoke_v0`; A preview `nvf_short_v1` (100 epochs + index); B preview `spkb_preview_v2` (25 epochs + index); full training `nvf_full_v1`. | **IN PROGRESS**: previews trained and evaluated offline; owner listening pending; no full training |
| **Milestone 5** | **Virtual Audio Cable Integration & Quality Tuning**: VB-Cable routing, Discord passthrough, acoustic tuning. | **IN PROGRESS**: backend device support done; not physically verified (no VB-Cable installed) |
| **Milestone 6** | **Minimal Desktop GUI & Packaging**: User-friendly control panel for device selection, pitch shifting, and monitoring. | **IN PROGRESS**: backend contract done; GUI (Gemini) not started |

---

## Environment Architecture

To ensure stability and prevent dependency conflicts (specifically between NumPy 2.x in modern inference and NumPy 1.x C-API ABI requirements in Applio training logger), the project uses two separated environments:

1. **Inference Environment (`.venv`)**:
   - Location: `.venv` (inside workspace).
   - Core stack: Python 3.11.9, PyTorch 2.5.1+cu121, NumPy 2.4.x, Matplotlib 3.11.x, SoundDevice, SoundFile, Librosa, Transformers.
   - Purpose: Running live real-time voice conversion, streaming prototype, latency profiling, and unit tests.
2. **Dedicated Training Environment (`rvc-train`)**:
   - Location: `C:\Users\quach\venvs\rvc-train` (isolated outside repository).
   - Core stack: Python 3.11.9, PyTorch 2.5.1+cu121, `numpy==1.26.4`, `matplotlib==3.7.2` (required for `fig.canvas.tostring_rgb()`), `scipy==1.17.1`, `tqdm>=4.66.0`.
   - Purpose: Offline preprocessing, ContentVec feature extraction, RMVPE F0 extraction, Faiss IVF index building, and G/D training loops.

---

## Quickstart Guide

### 1. Prerequisites & Inference Setup

- Windows 10/11 (x64)
- Python 3.11 (AMD64)
- NVIDIA GPU with CUDA Compute Capability >= 7.0 (Driver >= 530+)

```powershell
# Clone the repository
git clone https://github.com/quachvietthanh/RealTimeVoiceChanger.git
cd RealTimeVoiceChanger

# 1. Create and activate Python 3.11 virtual environment for inference
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install PyTorch 2.5.1 with CUDA 12.1 support
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 3. Install core project dependencies in editable mode
pip install -e .

# 4. Apply Applio streaming low-latency patch
python scripts/setup_applio_patch.py --apply

# 5. Verify hardware & dependencies
python scripts/check_environment.py
```

### 2. Download Auxiliary Models & Pretrained Checkpoints

Ensure auxiliary assets (`rmvpe.pt`, `contentvec/pytorch_model.bin`, `contentvec/config.json`) and the target voice model (`vctk_female_p231.pth` + `.index`) are present:

```powershell
# Download required checkpoints (RMVPE & ContentVec)
python scripts/download_auxiliary_assets.py

# Download base pretrained models for training (v2 40k f0G/f0D)
python scripts/download_pretraineds.py

# List and verify audio devices
python scripts/list_audio_devices.py
```

### 3. Run Offline Conversion Test

```powershell
python scripts/offline_conversion_test.py
```

*Expected output: Converts `tests/fixtures/input_male.wav` to `output/offline_test_result.wav` with RTF ~0.35.*

### 4. Run Real-Time Streaming Prototype

#### A. Simulated Streaming Validation (Recommended for automated benchmarking)

```powershell
# Run 5-minute continuous streaming validation (600 chunks @ 0.5s)
python scripts/realtime_prototype.py --simulated-input tests/fixtures/input_male.wav --duration 300.0 --chunk-size 0.5 --json-output output/long_run_5min_results.json
```

#### B. Live Microphone Real-Time Voice Conversion

```powershell
# Run with default microphone and speakers
python scripts/realtime_prototype.py --chunk-size 0.5 --pitch-shift 10.0 --index-rate 0.75 --duration 30.0

# Or specify device IDs (from python scripts/list_audio_devices.py)
python scripts/realtime_prototype.py --input-device 1 --output-device 3 --chunk-size 0.5

# Choose the speech language profile: vi (Vietnamese tones) or en (default)
python scripts/realtime_prototype.py --language vi --chunk-size 0.5
```

### 5. Run Test Suite

```powershell
python -m unittest discover -s tests
```

Unit and integration tests need no GPU or audio device (`tests/integration/test_profile_assets.py` reads the model files on the CPU and skips profiles whose files are absent).

### 6. Backend API (GUI integration) and headless GPU simulation

```powershell
# Inspect profiles (full validation), devices and status; loads nothing onto the GPU
.venv\Scripts\python scripts/backend_api_example.py

# Real GPU inference through the service API with simulated devices (nothing is played): profile switching A -> B -> A
.venv\Scripts\python scripts/service_simulation.py --profiles voice_a_preview,voice_b_preview,voice_a_preview --seconds 32

# 600-chunk / 5-minute regression through the service API (baseline conditions: vctk, +10, index 0.75)
.venv\Scripts\python scripts/service_simulation.py --profiles vctk_p231 --seconds 300 --pitch 10 --index-rate 0.75 --json-output output/<new_dir>/long_run.json
```

---

## Custom Voice Training (Northern Vietnamese Female)

The dataset preparation and model training workflow is documented in detail in:
- [Custom Voice Training Plan](file:///C:/Users/quach/Desktop/RealTimeVoiceChanger/docs/custom-voice-training-plan.md)
- [Hướng dẫn huấn luyện giọng Bắc Việt](file:///C:/Users/quach/Desktop/RealTimeVoiceChanger/docs/huong-dan-huan-luyen-giong-bac-viet.md)

### Current Dataset Split Status
Values below were verified on 2026-10-21 with `scripts/verify_dataset_manifest.py`.

- **Source Audio**: `data/source_audio/female_source_vocal_preview.wav` (isolated vocals, 44.1 kHz, stereo, PCM_16, 295.17 s).
- **Training Segment (`split_v1`, original)**: `data/dataset_raw/female_train_source.wav` (253.58 s, ~44.7 MB). Peak 0.9908, which fails the < 0.90 rule; it was used only for the `nvf_smoke_v0` pipeline check.
- **Derived Training Set (`train_peak0p88`)**: `data/dataset_train_peak0p88/female_train_source_peak0p88.wav` (the same audio × 0.88813 gain, FLOAT32, peak 0.88), with `derived_manifest.json`. This is the input for `nvf_short_v1`.
- **Held-Out Test Segment**: `data/held_out_test/female_holdout.wav` (34.81 s, ~6.1 MB), unchanged and shared by all versions.
- **Split Manifest**: `data/source_audio/dataset_split_manifest.json`. Dataset versions and experiment assignment are listed in section 2.4 of the training plan.

---

## Project Structure

```text
RealTimeVoiceChanger/
├── config/                         # Configuration guidelines and local overrides
├── data/                           # Dataset preparation & audio assets
│   ├── dataset_raw/                # Original training partition (female_train_source.wav)
│   ├── dataset_train_peak0p88/     # Derived training set (gain 0.888, peak 0.88) + derived_manifest.json
│   ├── held_out_test/              # Evaluation partition (female_holdout.wav)
│   └── source_audio/               # Clean vocal preview & dataset_split_manifest.json
├── docs/                           # Technical documentation & architecture
│   ├── ab-padding-report.md        # Edge padding & reflection artifact analysis
│   ├── architecture.md             # System architecture & component design
│   ├── custom-voice-training-plan.md # Vietnamese custom voice dataset & training plan
│   ├── huong-dan-huan-luyen-giong-bac-viet.md # Vietnamese training quick guide
│   ├── model-setup.md              # Checkpoint specification & asset checksums
│   ├── overlap-crossfade-report.md # Overlap and crossfade latency analysis
│   ├── quality-test-plan.md        # Voice quality and latency test methodology
│   ├── roadmap.md                  # Milestone-driven implementation plan
│   ├── rvc-engine-specification.md # Selected Applio Core inference specification
│   └── selected-voice-model.md     # VCTK p231 model metadata & verification
├── models/
│   ├── checkpoints/                # rmvpe.pt and contentvec/
│   ├── indices/                    # Faiss .index files
│   └── weights/                    # Voice model .pth weights
├── output/                         # Generated test audio and JSON telemetry logs
├── scripts/
│   ├── ab_padding_test.py          # A/B edge padding evaluation
│   ├── check_environment.py        # Environment & GPU preflight audit
│   ├── download_auxiliary_assets.py# Automated downloader for RMVPE & ContentVec
│   ├── download_pretraineds.py     # Base pretrained weights (f0G40k / f0D40k)
│   ├── list_audio_devices.py       # Audio device enumeration
│   ├── offline_conversion_test.py  # Offline WAV conversion benchmark
│   ├── overlap_crossfade_experiment.py # Overlap crossfade testing
│   ├── profile_stages.py           # Deep stage-by-stage latency profiler
│   ├── realtime_prototype.py       # Real-time prototype runner (live mic & simulated)
│   ├── separate_vocals_preview.py  # Vocal separation and extraction utility
│   └── setup_applio_patch.py       # Applio streaming patch manager
├── src/voice_changer/
│   ├── audio/                      # Audio I/O, device management & streaming pipeline
│   │   ├── devices.py              # SoundDevice enumeration
│   │   ├── input_stream.py         # Non-blocking microphone capture
│   │   ├── output_stream.py        # Ring buffer playback with underflow tracking
│   │   └── pipeline.py             # RealTimeVoiceChangerPipeline & PipelineMetrics
│   ├── conversion/                 # Decoupled voice conversion interfaces
│   │   ├── base.py                 # VoiceConverter protocol & ConversionConfig
│   │   └── rvc/adapter.py          # Applio Core RVC v2 adapter with Faiss caching
│   ├── diagnostics/                # Latency & system telemetry
│   └── tuning/                     # Presets & parameter management
├── tests/
│   ├── fixtures/                   # Audio fixtures for test reproducibility
│   └── unit/                       # Component test suite
└── third_party/
    └── applio/                     # Pinned Applio Core v3.2.8 submodule (commit 0c92ac5)
```

---

## Latency & Performance Profile (Quadro T2000 Baseline)

| Measurement | Value | Description |
| :--- | :--- | :--- |
| **Chunk Duration** | **500.0 ms** | 20,000 samples @ 40 kHz. |
| **Mean Inference Time** | **280.8 ms** | Total neural inference (ContentVec + RMVPE + HiFi-GAN). |
| **Real-Time Factor (RTF)** | **0.562** | Mean Inference / Chunk Duration (well below 1.0 threshold). |
| **Median Queue Wait** | **0.18 ms** | Async input handoff delay to worker thread. |
| **Software E2E Turnaround** | **778.7 ms** | First sample captured -> Converted chunk in output buffer. |
| **Output Buffer Post-Write** | **838.3 ms** | Peak ring buffer occupancy immediately after write. |
| **Estimated Acoustic Latency** | **~1117.0 ms** | Software E2E + Output buffer playback queue (excl. OS/DAC hardware). |
| **GPU VRAM Usage** | **444.5 MB** | Extremely lightweight footprint (<12% of 4 GB VRAM). |
| **Steady-State Dropouts** | **0** | Zero audio glitches over 5-minute continuous test (600 chunks). |
