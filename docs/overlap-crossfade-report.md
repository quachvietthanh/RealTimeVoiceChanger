# Reconciling the A/B padding report and testing overlap-crossfade

Date: 2026-10-20. Scope: diagnostics only. Production padding `(80, 80)`, cropping, feature extraction and streaming are unchanged. Labels: **[measured]** = produced by a run in this session; **[code]** = derived from reading source; **[hypothesis]** = not verified.

## 1. Task 1: the §7.2 / §7.4 discrepancy in `docs/ab-padding-report.md`

### Evidence

| Item | Finding | Label |
| :--- | :--- | :--- |
| Origin of the report | §7.1–§7.5 were all added in one commit, `5e99061` (author `chau.le`, 2026-10-20 12:09:58). The reflog shows it reached this machine via `pull origin develop` at 12:12:29. It was not committed here. | [measured] `git reflog`, `git show` |
| Prior runs on this machine | `output/ab_padding/` does not exist. `scripts/__pycache__/ab_padding_test.cpython-311.pyc` was first written at 17:10 by this session's rerun, so the A/B harness had never been imported here before. | [measured] |
| Fixture on this machine | `tests/fixtures/input_male.wav` exists. It is gitignored (`.gitignore:47`), mtime 2026-10-18 22:29, NTFS birth time 2026-10-18 22:32. SHA-256 `59c412b1d0848ffe32feac53eb4402fd5353f4efb2bd1606a1dac12ea8e24ffb`, 48 kHz, 2-channel PCM_16, 32.235 s. Resampled to 40 kHz mono it is 1,289,387 samples. | [measured] |
| GPU | This machine has a Quadro T2000 (driver 610.47) and `torch.cuda.is_available()` is `True`. §7.2 says the machine had no NVIDIA GPU. | [measured] |
| §7.4 input hash | Not recorded. `ab_padding_test.py` hashes the model and index but not the input, and the §7.4 JSON was never committed (gitignored). | [code] |

### Resolution

- §7.2 ("file not in repo and not found on the machine") and §7.4 describe the author's machine, which is not this one. "Not in repo" is correct, since the file is gitignored. "Not found on the machine" is false for this machine, where the file predates the report by about 38 h.
- On the author's machine, §7.2 and §7.4 contradict each other inside a single commit. No log, JSON or hash from that machine is available. **This remains UNRESOLVED.** [hypothesis] §7.2 was written first, the file was added later, and §7.4 was appended without revising §7.2. Nothing confirms this.
- Indirect evidence that §7.4 used a file with the same content as the local fixture: the same 32.2 s duration and the same 1,289,387-sample count. The rerun below also reproduces the §7.4 lag and LSD values. This is consistent with identical content but does not prove it, because no hash was recorded.
- Code revision used for §7.4: unknown. It was probably the working tree that became `5e99061`. The report claims a CPU-only environment. No artifact confirms either point.

### Reproduction of §7.4 on this machine [measured]

Command: `CUDA_VISIBLE_DEVICES=-1 .venv\Scripts\python.exe scripts\ab_padding_test.py --input tests/fixtures/input_male.wav --out output/reconcile_ab_2026-10-20/ab_padding_speech_cpu_rerun.json`. Environment: HEAD `fb805bd`, CPU (i7-10850H), Python 3.11.9, torch 2.5.1+cu121, numpy 2.4.6 (note: `pyproject.toml` declares `numpy<2`, so the venv does not match the declared dependency), librosa 0.11.0, faiss 1.15.1. Model SHA `12f819e1…5a59`, index SHA `14c3a6fb…01ab`.

| Config | §7.4 lag / corr | Rerun lag (chunked / whole) / corr | §7.4 ratio mean/max | Rerun ratio mean/max | §7.4 LSD near/away | Rerun LSD near/away |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A (80,80) | +5 / 0.67 | +5 / +5 / 0.667 | 39 / 433 | 43.7 / 411 | 9.7 / 8.1 | 9.78 / 8.15 |
| B (0,160) | −1 / 0.67 | +2 / +1 / 0.663 | 118 / 2374 | 130 / 1756 | 10.5 / 8.1 | 10.27 / 8.12 |
| C (0,0) | −1 / 0.66 | +1 / −1 / 0.653 | 1.6·10⁵ / 4.9·10⁶ | 3.3·10⁵ / 9.7·10⁶ | 10.8 / 8.1 | 10.81 / 8.12 |

The qualitative §7.4 conclusions reproduce. The ordering of click ratios is A < B ≪ C, A shows the lowest near-boundary LSD, and all lengths match. Exact values do not reproduce. RVC synthesis adds unseeded noise (`torch.randn_like`, `third_party/applio/rvc/lib/algorithm/synthesizers.py:199`), so single runs vary. The "A − B ≈ 5 ms" lag claim measured 3–4 ms on this rerun. At 1 ms resolution from a single run, that claim is not stable. §7.3 (synthetic overlap) was not rerun. On real speech it is superseded by Task 2.

## 2. Task 2: controlled context / crossfade experiment

### Setup [measured]

- Script: `scripts/overlap_crossfade_experiment.py` (new). Command: `.venv\Scripts\python.exe scripts\overlap_crossfade_experiment.py --reps 6`. Log: `output/overlap_crossfade_2026-10-20.log`.
- Device: Quadro T2000, Applio fp16 (`is_half=True`), HEAD `fb805bd`, working tree clean apart from the new script. Desktop applications shared the GPU (utilisation 71–90 % during runs, 63–68 °C, SM clock 1875 MHz throughout).
- Input: the fixture above, using 64 full 500 ms chunks (1,280,000 samples). The 9,387-sample tail was dropped. That gives 63 internal boundaries, 41 of them with active speech (from input RMS). The file is processed linearly, so there are no file-loop boundaries.
- Fixed settings: model `vctk_female_p231`, pitch +10, rmvpe, index_rate 0.5, filter_radius 2, protect 0.4, pad_16k (80, 80) (asserted in the script).
- Controls: warm-up for every segment length plus one whole-file call. 6 repetitions with seeded random config order (orders recorded in `summary.json`). Each call is seeded with `torch.manual_seed(1000·rep + chunk)`, shared across configs within a rep. A whole-file reference is converted per rep, and a second reference measures the noise floor. That gives 384 timed calls per inference config, and 6 × 41 = 246 active boundary samples per config.

### Configurations [code]

| ID | Inference input | Output assembly | Extra algorithmic latency |
| :--- | :--- | :--- | :--- |
| A | chunk | as is (production) | 0 |
| B | 100 ms past input + chunk | discard context output | 0 |
| C | 200 ms past input + chunk | discard context output | 0 |
| D_xf50 | same calls as C | hold back the last 50 ms of each chunk and crossfade it (equal-gain sin²) with the next call's estimate of the same span | +50 ms |
| D_xf100 | same calls as C | as above, 100 ms crossfade | +100 ms |
| D_xf50_eqpow | same calls as C | 50 ms, equal-power sin/cos | +50 ms |

In every config each chunk emits exactly 20,000 samples, and output sample *i* maps to input sample *i*. Context output is never counted as new content. B and C remove the edge only on the right-hand side of the boundary. The left side is still the previous call's right edge, which had no lookahead, so the boundary is a junction between two independent inferences. D is the only config that blends two estimates of the same span, and it is continuous at the nominal boundary by construction. For that reason D is also judged on region metrics over [b − 100 ms, b + 10 ms), not only on the single boundary sample.

### Results (active boundaries, pooled over 6 reps; mean / median / P95 / max) [measured]

| Metric | A | B | C | D_xf50 | D_xf100 | D_xf50_eqpow | Within-chunk (mid) for C/D | Ref vs ref |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| n boundary samples | 246 | 246 | 246 | 246 | 246 | 246 | 246 | 41 |
| Abs jump at b | .132/.090/.359/.665 | .124/.104/.306/.594 | .133/.080/.371/.427 | .009/.007/.030/.080 | same as D_xf50 | same | .010/.004/.042/.085 | .006/.004/.015/.029 |
| Jump ÷ local median | 27.7/15.8/95/136 | 22.5/15.6/74/144 | 25.3/18.6/70/178 | 1.15/1.07/2.5/3.5 | 1.13/1.06/2.5/3.5 | 1.12/1.04/2.5/3.5 | 1.15/0.99/2.7/5.2 (A: 1.10/0.99/2.4/6.1) | 1.2/0.98/3.2/3.4 |
| 2nd-diff residual ÷ local median (median / P95) | 33 / 186 | 35 / 194 | 42 / 281 | 0.84 / 3.2 | 0.83 / 3.1 | 0.82 / 3.0 | 1.02 / 3.1 | 0.94 / 4.4 |
| Level step across b vs reference, dB (median / P95) | 3.70 / 23.2 | 2.70 / 17.1 | 2.53 / 12.2 | 0.79 / 7.7 | 0.78 / 8.0 | 1.13 / 7.8 | 0.99 / 6.5 (A mid: 1.21 / 6.3) | 0.42 / 3.3 |
| Envelope min dip vs reference in region, dB (median) | −4.11 | −3.77 | −2.41 | **−4.29** | **−4.85** | −2.53 | −3.07 (A mid: −1.81) | −1.01 |
| LSD in boundary region, dB (median / P95) | 8.24 / 12.6 | 8.29 / 11.8 | 8.09 / 10.7 | 7.04 / 8.9 | 6.92 / 8.4 | 7.12 / 8.8 | 7.02 / 9.9 (A mid: 7.44 / 9.3) | 5.36 / 6.0 |
| Crossfade coherence r (median / P95) | — | — | — | 0.04 / 0.85 | 0.02 / 0.87 | 0.04 / 0.85 | — | — |
| Global lag vs reference (6 reps) | 0 ms all | 0 ×5, −4 ×1 | 0 ms all | 0 ms all | 0 ms all | 0 ms all | — | 0 |
| Lag vs input (6 reps, 1 ms res.) | 5 all | 5 all | 5–8 | 5 all | 5–8 | 5–8 | — | 7 |
| Output length / clipped / NaN / spurious silence | 1,280,000 / 0 / 0 / 0 ms in every rep for every config | | | | | | | |
| Errors / length mismatches / edge-pad calls | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | inherited from C: 0 / 0 / 0 | | | | |
| F0 cross-boundary step, cents (rep 0, n) | 20 median (n=4) | n=0 | 25 (n=4) | 15 (n=12) | 15 (n=14) | 20 (n=12) | mid 10 | — |

### Inference cost (Quadro T2000, 384 calls each, includes resample and feature extraction inside `convert_chunk`) [measured]

| Config | Mean | Median | P95 | Max (ms) | RTF mean (÷ 500 ms new audio) | RTF P95 | Overhead vs A | Headroom at P95 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 240 | 239 | 249 | 278 | 0.480 | 0.499 | — | 251 ms |
| B | 255 | 254 | 266 | 453 | 0.511 | 0.532 | +6.4 % | 234 ms |
| C, D_* | 282 | 282 | 293 | 430 | 0.564 | 0.585 | +17.5 % | 207 ms |

Per-rep mean spread: A 239–241, B 252–261, C 280–286 ms. D's crossfade assembly costs about 0.05 ms per chunk. No call exceeded the 500 ms budget. Earlier CPU timings are not comparable and are not used here.

### What the measurements establish

1. **Left context alone (B, C) does not remove the boundary discontinuity.** The normalised boundary jump stays 15–19× the within-chunk median, against about 1× within chunks. The level step and region LSD improve slightly, by about 1 dB and 0.15 dB. The earlier synthetic claim of "200 ms reduces clicks ~35 %" does not hold on real speech: the median absolute jump fell 11 %, while the normalised median rose.
2. **Overlap-crossfade (D) brings the jump, the 2nd-difference and the boundary-region LSD down to within-chunk levels.** It also brings the level step to within-chunk levels. This holds across all 246 boundary samples, not only on average: D's P95 and max normalised jump (2.5 / 3.5) are at or below the within-chunk values (2.7 / 5.2).
3. **Equal-gain crossfade creates an envelope dip.** The two estimates of the overlap span are mostly uncorrelated at the waveform level (median r = 0.02–0.04). Summing incoherent signals with equal-gain weights loses energy mid-fade: median min-dip −4.3 / −4.9 dB, against −3.1 dB within chunks. The equal-power fade removes the excess dip (−2.5 dB) and keeps the same jump results.
4. **No temporal shift or content duplication.** Every config reports 0 ms global lag against the reference in every rep, except one B rep at −4 ms. All lengths are exact, with no silence and no clipping. The lag-vs-input estimator jumps between 5 and 8 ms even for reference vs reference (7 ms), so ±3 ms there is estimator ambiguity, not a measured shift.

### Limitations

- These are signal metrics only. Low waveform coherence means each crossfade blends two differently-phased renderings for 50–100 ms. That can sound like phasiness, comb filtering or doubling, and no metric here rules it out. **A listening test is required.**
- I could not explain why every chunked config is about 3 dB louder than the whole-file conversion (median chunk peak 0.62 vs 0.40; Applio's `/0.99` normalisation never triggered, max peak 0.85). Because of that offset, envelope-vs-reference values carry a bias, and only boundary-vs-mid comparisons are meaningful.
- F0 used pyin on rep 0 only. Very few voiced frames met the reliability gate on both sides (n = 0–14), so F0 continuity is inconclusive.
- The per-chunk envelope-lag estimator (`chunk_lags.csv`) saturates at ±10 ms for many chunks, so I don't use it as evidence of shift.
- One fixture (32 s, one male speaker) and one placeholder model, run offline rather than through the live audio pipeline. GPU timing was taken under desktop GPU sharing.

## 3. Task 3: does this support a production change?

- Context only (B/C): **no.** It costs 6–18 % more inference with no meaningful continuity gain.
- Overlap-crossfade (D, preferably equal-power): **promising, but not ready to ship.** Measured: discontinuities removed to within-chunk levels, no shift, no length change, no silence, no clipping, P95 293 ms out of a 500 ms budget. Not established: perceptual quality, given the low coherence; behaviour with a live microphone; other voices and models.
- Production cost if adopted [code]: the converter or pipeline would need state, namely a 200 ms input history and the held output tail. That adds +50 ms algorithmic latency on top of the current ~740 ms median software E2E (`output/realtime_test_vi_30s.json`) and +17.5 % inference. `convert_file` (the offline path) goes through Applio's `convert_audio` and does not use `convert_chunk`, so a streaming-side change would not touch the offline path.

**Smallest justified next step:** a blind listening comparison (A vs C vs D_xf50 vs D_xf50_eqpow vs reference) on the existing `*_rep0.wav` files. It needs no new code. Only if equal-power D passes listening should the next harness experiment add SOLA-style alignment (a small lag search before the fade) to raise coherence, and only after that should a production design be drafted.

## 4. Files

- New: `scripts/overlap_crossfade_experiment.py`, `docs/overlap-crossfade-report.md`.
- New, gitignored outputs: `output/reconcile_ab_2026-10-20/` (rerun JSON and log), `output/overlap_crossfade_2026-10-20/` (`summary.json`, `raw_results.json`, `boundary_rows.csv`, `chunk_lags.csv`, `input_40k.wav`, `reference_rep0.wav`, a `<config>_rep0.wav` per config), and `output/overlap_crossfade_2026-10-20.log`.
- Nothing existing was modified or overwritten. Nothing was committed or pushed.

## 5. Follow-up (2026-10-21)

After the owner reported choppy live audio on all profiles, the seam defect was re-measured on vctk, A and B and the
configuration-D approach was implemented with SOLA alignment (`src/voice_changer/audio/crossfade.py`, 200 ms context,
50 ms sin² crossfade, 0–10 ms lag search). Results and the remaining listening test: `output/live_audio_diag_2026-10-21/README.md`.
