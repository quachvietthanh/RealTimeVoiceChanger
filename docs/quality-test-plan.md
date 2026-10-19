# Quality & Performance Test Plan

This document defines the evaluation criteria, test protocols, and metrics for validating RealTimeVoiceChanger in future development stages.

> **Status Notice:** Milestone 2 (Offline WAV conversion) and Milestone 3A.1 (5-minute real-time streaming validation) have executed and passed successfully. Recorded test results are logged below.

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

---

## 4. Empirical Test Logs

### Run M2-OFFLINE-01 (Milestone 2 Offline Baseline)

```text
Run ID:               M2-OFFLINE-01
Timestamp:            2026-10-19
Model:                vctk_female_p231 (v2, 40kHz, HiFi-GAN)
F0 Algorithm:         RMVPE
Pitch Transposition:  +10.0 semitones
Input Audio:          tests/fixtures/input_male.wav (32.23s, 48kHz, 2ch)
Inference Time Total: 11.108 s
Real-Time Factor:     0.345 (Faster than real-time)
Output Audio:         output/offline_test_result.wav (32.20s, 40kHz, 1ch)
Clipping / Silence:   False / False (Peak: 0.80078, RMS: 0.10037)
Result:               PASS
```

### Run M3A1-LONG-01 (Milestone 3A.1 5-Minute Streaming Regression)

```text
Run ID:               M3A1-LONG-01
Timestamp:            2026-10-19
Model:                vctk_female_p231 (v2, 40kHz, HiFi-GAN, Faiss RAM cached)
F0 Algorithm:         RMVPE
Pitch Transposition:  +10.0 semitones
Index Rate:           0.75
Chunk Size:           500.0 ms (20,000 samples @ 40 kHz)
Session Duration:     301.0 s
Chunks Processed:     600 / 600 (100.0% completion)
Inference Errors:     0
Backlog Drops:        0
Input Overflows:      0
Sample Mismatches:    0 (Expected: 20,000, Actual: 20,000)
Steady-State Drops:   0 (Active speech buffer glitches)
Startup Silence:      21 callbacks (Expected initial priming delay ~1.0s)
Inference Time/Chunk: Mean: 280.8 ms | Median: 277.7 ms | Min: 249.7 ms | Max: 461.4 ms | P95: 340.4 ms
Real-Time Factor:     0.562 (280.8 ms / 500.0 ms)
Queue Wait Delay:     Median: 0.18 ms | P95: 0.27 ms
Software E2E Latency: Median: 778.7 ms | P95: 841.1 ms
Buffer Occupancy:     Peak: 838.3 ms | Ahead in Queue: ~338.3 ms
Estimated Acoustic:   ~1117.0 ms (Software E2E Median + Ahead in Queue; excl. hardware DAC/driver)
GPU Memory:           Start: 436.4 MB | Peak: 473.5 MB | End: 444.5 MB (Stable, no leaks)
Result:               PASS
```
