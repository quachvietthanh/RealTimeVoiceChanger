# Selected RVC Inference Implementation Specification

This document records the specific RVC inference implementation selected for RealTimeVoiceChanger to ensure explicit, unmixed architectural assumptions.

---

## 1. Selected Implementation Details

- **Implementation Name**: Applio Core (Modular RVC v2)
- **Repository URL**: [https://github.com/IA-Hispano/Applio](https://github.com/IA-Hispano/Applio)
- **Target Revision / Release**: `v3.2.8` (Commit `b65c325` / stable tag `3.2.8`)
- **License**: MIT License
- **Supported Python Version**: Python 3.10 and 3.11 (Active host is Python 3.11.9)
- **PyTorch / CUDA Requirements**:
  - PyTorch >= 2.1.0 (Targeting PyTorch 2.5.1 with CUDA 12.1 runtime)
  - Torchaudio matching PyTorch release
  - CUDA Compute Capability >= 7.0 (Quadro T2000 is Turing Compute 7.5)

---

## 2. Model & Weight Formats

- **Target Voice Model**: Standard RVC v2 `.pth` PyTorch state dictionary (VITS generator weights, typically 40kHz or 48kHz).
- **Feature Retrieval Index**: Faiss `.index` file (IVF-PQ or Flat index extracted from target female speaker training corpus).
- **Pitch Predictor Model**: RMVPE (`rmvpe.pt`, ~16 MB PyTorch checkpoint).
- **Speech Representation Model**: ContentVec / HuBERT (`hubert_base.pt` or `checkpoint_best_GID.pt`, ~180 MB PyTorch checkpoint).

---

## 3. Inference Architecture & Entry Point

- **Core Module**: `rvc.infer.infer` / `VoiceConverter` pipeline.
- **Inference Entry Method**:

  ```python
  from rvc.infer.infer import VoiceConverter
  # Programmatic pipeline:
  # converter.convert_audio(audio_path, model_path, index_path, pitch, f0_method, ...)
  ```

- **Adapter Encapsulation**:
  In our application, this engine will be wrapped exclusively within [`RVCAdapter`](file:///c:/Users/quach/Desktop/RealTimeVoiceChanger/src/voice_changer/conversion/rvc/adapter.py#L15-L50), which implements the decoupled [`VoiceConverter`](file:///c:/Users/quach/Desktop/RealTimeVoiceChanger/src/voice_changer/conversion/base.py#L25-L42) protocol. The real-time streaming pipeline will never directly import or invoke Applio internals.

---

## 4. Architectural Rationale

1. **Elimination of Legacy C++ Compilation**: Original RVC-Project/Retrieval-based-Voice-Conversion-WebUI relies on pinned Fairseq 0.12.2, which requires C++ MSVC build tools to compile on Windows and frequently fails on Python 3.11. Applio refactored speech representation extraction to avoid Fairseq compilation hurdles.
2. **Modern Dependency Stack**: Applio natively supports PyTorch 2.x and CUDA 12.x wheels, matching our host NVIDIA Driver (610.47).
3. **Headless Modular Design**: Applio cleanly separates its CLI/core inference routines from its Gradio web UI, enabling direct integration into a streaming Python service.
