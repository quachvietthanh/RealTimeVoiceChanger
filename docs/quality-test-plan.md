# Quality & Performance Test Plan

This document defines the evaluation criteria, test protocols, and metrics for validating RealTimeVoiceChanger in future development stages.

> **Status Notice:** No tests have passed yet. The criteria below are specifications for future validation once the respective components and models are implemented.

---

## 1. Evaluation Dimensions

Future testing will assess the voice conversion across five critical dimensions:

| Dimension | Description | Target Standard |
| :--- | :--- | :--- |
| **Naturalness** | Perceived organic timbre of the converted feminine voice. | Free from metallic, synthetic, or robotic artifacts; preserves natural human inflection. |
| **Intelligibility** | Clarity of words, plosives, fricatives, and fast speech. | 100% sentence comprehension; no dropped syllables or muffled phonemes. |
| **Acoustic Artifacts** | Undesirable sound anomalies introduced during conversion. | Absence of pitch tracking squeaks, phase smearing, robotic fluttering, or gating clicks. |
| **Turnaround Latency** | End-to-end delay from user vocalization to virtual cable output. | Sub-300ms total latency for fluid conversational interaction in Discord/games. |
| **System Stability** | Sustained operational reliability under continuous processing. | Zero buffer underruns, audio dropouts, or GPU memory creep during a 60-minute session. |

---

## 2. Test Procedures

### Test Suite A: Voice Quality & Naturalness Evaluation (Offline & Real-Time)

- **Test A1: Phonetic & Pitch Coverage**
  - *Input:* Standardized speech samples containing diverse English phonemes, questions (rising intonation), statements (falling intonation), and whispers.
  - *Evaluation:* Compare converted audio against natural female voice references. Listen for unnatural robotic pitch plateaus or erratic octave leaps.
- **Test A2: Fricative & Consonant Preservation**
  - *Input:* Sentences dense with sibilants ("s", "sh", "z") and plosives ("p", "t", "k").
  - *Evaluation:* Verify that `protect_rate` settings retain consonant crispness without causing pitch leakage.
- **Test A3: Dynamic Range & Noise Handling**
  - *Input:* Quiet speech, vocal fry, energetic shouting, and background noise (keyboard clicks, fan hum).
  - *Evaluation:* Verify noise gate and index blending prevent background noise from triggering synthetic artifacts.

### Test Suite B: Turnaround Latency Benchmarking

- **Test B1: Audio Buffer Ingestion Latency**
  - Measure delay between microphone hardware capture and audio chunk delivery to the pipeline queue.
- **Test B2: Conversion Inference Latency**
  - Measure pure inference execution time per chunk in `conversion/rvc/adapter.py`.
  - Benchmark varying chunk sizes (e.g. 128ms, 256ms, 384ms, 512ms) against inference time to identify the real-time threshold (RTF < 1.0).
- **Test B3: Output Buffer & Crossfade Latency**
  - Measure time spent in crossfade windowing and delivery to VB-Audio Virtual Cable.
- **Test B4: End-to-End Turnaround Verification**
  - Output an acoustic click impulse, loop through microphone -> changer -> output, and measure physical round-trip delay.

### Test Suite C: Real-Time Streaming Stability

- **Test C1: Extended Run Endurance**
  - Run the audio pipeline continuously for 60 minutes with intermittent speech.
  - Monitor memory usage (RAM and VRAM) for leaks.
  - Track audio buffer underruns and dropped frame counts.
- **Test C2: Dynamic Device Disconnect & Reconnect**
  - Simulate microphone or virtual cable restart while the pipeline is active.
  - Verify graceful stream recovery without application crashes.

---

## 3. Test Recording & Reporting Template

Future test runs must record results according to this format:

```text
Run ID:               [e.g., 2026-XX-XX-RUN01]
Model:                [Model Name / Epoch / Sample Rate]
F0 Algorithm:         [RMVPE / PM / Harvest]
Pitch Transposition:  [+12 semitones]
Chunk Size:           [e.g., 256ms]
Inference Time/Chunk: [TBD ms]
Total Latency:        [TBD ms]
Underruns/Hour:       [TBD]
Quality Notes:        [Subjective listening feedback and artifact observations]
Result:               [PENDING / PASS / FAIL]
```
