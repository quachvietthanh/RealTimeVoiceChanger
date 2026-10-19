# RealTimeVoiceChanger

A real-time male-to-female voice changer application for Windows using Retrieval-based Voice Conversion (RVC v2).

## Project Purpose

The primary objective of this project is to create a high-quality, low-latency male-to-female voice changer for Windows using Retrieval-based Voice Conversion (RVC v2) with RMVPE pitch estimation and ContentVec feature extraction.

The system is designed for live conversational use in voice platforms like Discord, OBS, and games via virtual audio drivers (such as VB-Audio Virtual Cable).

---

## Current Project Status: Milestone 3A.1 Completed

> **Status:** **Operational Real-Time Pipeline Prototype**.
>
> - **Inference Engine**: Applio Core v3.2.8 (`rvc.infer.infer.VoiceConverter`) running locally on PyTorch 2.5.1 + CUDA 12.1.
> - **Hardware Baseline**: NVIDIA Quadro T2000 Mobile (4 GB VRAM) on Windows 11.
> - **Real-Time Factor (RTF)**: **0.562** (Inference: ~278 ms per 500 ms audio chunk).
> - **5-Minute Long-Run Validation**: **600/600 chunks** successfully processed with **0 steady-state dropouts**, **0 sample count mismatches**, and **stable ~445 MB VRAM**.
> - **Active Voice Model**: Technical placeholder model `vctk_female_p231` (VCTK female speaker p231, RVC v2, 40 kHz, HiFi-GAN). Custom conversational Northern Vietnamese female training is planned for the next milestone.

---

## Development Milestones

| Milestone | Description | Status |
| :--- | :--- | :--- |
| **Milestone 1** | **Environment & Hardware Assessment**: Python 3.11, CUDA 12.1, PyTorch 2.5.1, Applio v3.2.8 pinned. | **DONE** |
| **Milestone 2** | **Offline WAV Conversion**: RVCAdapter, RMVPE, ContentVec, offline baseline conversion (RTF = 0.345). | **DONE** |
| **Milestone 3** | **Real-Time Streaming Prototype**: Non-blocking I/O, async bounded worker thread, PortAudio output stream. | **DONE** |
| **Milestone 3A** | **Latency Reduction**: Faiss in-RAM caching, 0.1s streaming padding, 0.5s chunk configuration, RTF reduced from 0.88 to 0.56. | **DONE** |
| **Milestone 3A.1** | **Long-Run Streaming Validation**: 5-minute continuous test (600 chunks), 80-sample reflection padding, 0 steady dropouts. | **DONE** |
| **Milestone 4** | **Custom Voice Dataset & Model Training**: High-quality Northern Vietnamese female conversational voice model. | *Planned* |
| **Milestone 5** | **Virtual Audio Cable Integration & Quality Tuning**: VB-Cable routing, Discord passthrough, acoustic tuning. | *Planned* |
| **Milestone 6** | **Minimal Desktop GUI & Packaging**: User-friendly control panel for device selection, pitch shifting, and monitoring. | *Planned* |

---

## Quickstart Guide

### 1. Prerequisites & Environment Setup

- Windows 10/11 (x64)
- Python 3.11 (`python -m venv .venv`)
- NVIDIA GPU with CUDA Compute Capability >= 7.0 (Driver >= 530+)

```powershell
# Clone the repository
git clone https://github.com/quachvietthanh/RealTimeVoiceChanger.git
cd RealTimeVoiceChanger

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Verify hardware & dependencies
python scripts/check_environment.py
```

### 2. Download Models & Pretrained Checkpoints

Ensure auxiliary assets (`rmvpe.pt`, `contentvec/pytorch_model.bin`, `contentvec/config.json`) and the target voice model (`vctk_female_p231.pth` + `.index`) are present:

```powershell
# Download required checkpoints (RMVPE & ContentVec)
python scripts/download_auxiliary_assets.py

# List and verify audio devices
python scripts/list_audio_devices.py
```

### 3. Run Offline Conversion Test

```powershell
python scripts/offline_conversion_test.py
```

*Expected: Converts `tests/fixtures/input_male.wav` to `output/offline_test_result.wav` with RTF ~0.35.*

### 4. Run Real-Time Streaming Prototype

#### A. Simulated Streaming Regression Test (Recommended for automated validation)

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
```

### 5. Run Test Suite

```powershell
python -m unittest discover -s tests
```

---

## Project Structure

```text
RealTimeVoiceChanger/
├── config/                         # Configuration guidelines and local overrides
├── docs/                           # Technical documentation & architecture
│   ├── architecture.md             # System architecture & component design
│   ├── custom-voice-training-plan.md # Vietnamese custom voice dataset & training plan
│   ├── model-setup.md              # Checkpoint specification & asset checksums
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
│   ├── check_environment.py        # Environment & GPU preflight audit
│   ├── download_auxiliary_assets.py# Automated downloader for RMVPE & ContentVec
│   ├── list_audio_devices.py       # Audio device enumeration
│   ├── offline_conversion_test.py  # Offline WAV conversion benchmark
│   ├── profile_stages.py           # Deep stage-by-stage latency profiler
│   └── realtime_prototype.py       # Real-time prototype runner (live mic & simulated)
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
    └── applio/                     # Pinned Applio Core v3.2.8 submodule
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
