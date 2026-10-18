# RealTimeVoiceChanger

A real-time male-to-female voice changer application for Windows.

## Project Purpose

The primary objective of this project is to maximize natural feminine voice quality using an existing, compatible Retrieval-based Voice Conversion (RVC) inference implementation, integrated with low-latency real-time microphone processing.

The converted output will eventually route to a virtual audio driver (such as VB-Audio Virtual Cable) for direct use in voice chat platforms like Discord, streaming applications, and games.

## Current Project Status

> **Notice:** This repository is currently in the **initial scaffolding and architectural definition stage**. It is **not yet a working voice changer**. No ML models have been downloaded, and no runtime audio engines or ML dependencies are installed yet.

## Development Milestones

The project will proceed through the sequential phases outlined in [`docs/roadmap.md`](docs/roadmap.md):

1. **Phase 1: Environment & Hardware Assessment** - Evaluate host GPU/CUDA, audio interfaces, and select compatible RVC inference runtime.
2. **Phase 2: Offline WAV Conversion & Model Evaluation** - Benchmark voice quality on test audio clips without real-time constraints.
3. **Phase 3: Audio Passthrough Pipeline** - Implement microphone capture to virtual cable routing without AI (zero-inference latency baseline).
4. **Phase 4: Real-Time RVC Integration** - Connect the modular conversion adapter to the chunked audio pipeline.
5. **Phase 5: Feminine Voice Quality Tuning** - Fine-tune pitch transposition, formant parameters, index rates, and noise gates.
6. **Phase 6: Latency & Stability Optimization** - Minimize end-to-end processing delay and eliminate audio dropouts/buffer underruns.
7. **Phase 7: Minimal GUI & Packaging** - Provide a streamlined desktop user interface and distribution package.

## Project Structure

```text
RealTimeVoiceChanger/
├── .gitignore                      # Git exclusion rules (safeguards models & audio)
├── pyproject.toml                  # Project configuration and packaging
├── README.md                       # Project overview and status
├── config/
│   └── README.md                   # Configuration guidelines and local overrides
├── docs/
│   ├── architecture.md             # System architecture and module boundaries
│   ├── quality-test-plan.md        # Voice quality and latency test methodology
│   └── roadmap.md                  # Milestone-driven implementation plan
├── models/
│   └── README.md                   # Model placement instructions (.pth / .index)
├── scripts/
│   ├── check_environment.py        # Hardware and environment inspection script
│   └── list_audio_devices.py       # Audio device discovery script
├── src/
│   └── voice_changer/
│       ├── __init__.py
│       ├── main.py                 # Scaffold entry point
│       ├── audio/                  # Audio I/O and device abstraction
│       ├── conversion/             # Converter interface and RVC adapter
│       ├── diagnostics/            # Latency and environment diagnostics
│       ├── pipeline/               # Real-time chunking and orchestration
│       └── tuning/                 # Preset management and tuning parameters
└── tests/
    ├── README.md                   # Test suite documentation
    ├── integration/                # Integration tests (end-to-end pipelines)
    └── unit/                       # Unit tests (individual components)
```

## Local Model Storage

Voice model weights (`.pth`) and feature retrieval indices (`.index`) are stored locally in the [`models/`](models/) directory:

- Models are never committed to version control.
- Detailed instructions for setting up model checkpoints and indices are available in [`models/README.md`](models/README.md).
