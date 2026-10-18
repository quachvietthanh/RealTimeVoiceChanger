# Selected RVC Voice Model Documentation

## 1. Overview & Identity

- **Model Name**: VCTK Female Speaker p231 (`vctk_female_p231`)
- **Publisher / Repository**: `Nekochu/RVC-VCTK_Voice-sample` on Hugging Face
- **Repository URL**: [https://huggingface.co/Nekochu/RVC-VCTK_Voice-sample](https://huggingface.co/Nekochu/RVC-VCTK_Voice-sample)
- **Immutable Commit Hash**: `005c2f948ee9dafd7e3aa7f261b4c3a24beebeef`
- **Model Role**: Offline & real-time male-to-female voice conversion target model for `RealTimeVoiceChanger`.
- **Target Profile**: Natural, adult feminine speaking voice (acoustic chamber conversational/reading speech, not a celebrity imitation, fictional character, or singing-only model).

---

## 2. Source URLs & Asset Artifacts

| Asset | Source URL | Destination Path | Size (Bytes) | SHA-256 Checksum |
| :--- | :--- | :--- | :--- | :--- |
| **Inference Checkpoint (.pth)** | `https://huggingface.co/Nekochu/RVC-VCTK_Voice-sample/resolve/005c2f948ee9dafd7e3aa7f261b4c3a24beebeef/F/p231/rmvpe/Fp231rmvpe.pth` | `models/weights/vctk_female_p231.pth` | 55,224,656 | `12f819e129c6e48779439e491b85593b8819d094f751c3515ea13be6bd8b5a59` |
| **Feature Index (.index)** | `https://huggingface.co/Nekochu/RVC-VCTK_Voice-sample/resolve/005c2f948ee9dafd7e3aa7f261b4c3a24beebeef/F/p231/rmvpe/added_IVF1216_Flat_nprobe_1_Fp231rmvpe_v2.index` | `models/indices/vctk_female_p231.index` | 149,879,099 | `14c3a6fb421ed0c77654bb9ed5778307ff0204d7b047b6bbe37922af42f601ab` |

Both assets are excluded from Git version control via `.gitignore`.

---

## 3. Provenance & Licensing

### Voice & Training-Data Provenance

- **Dataset**: CSTR Edinburgh VCTK Corpus (Centre for Speech Technology Research, University of Edinburgh).
- **Speaker**: Speaker `p231` (Adult female, age 23, English with standard Southern England accent).
- **Acoustic Environment**: Hemianechoic chamber recording with high-fidelity studio microphones (omnidirectional DPA 4035 & Sennheiser MKH 800), clean speech without background noise, musical backing, or heavy studio compression.
- **Publisher History**: Model trained and published by Nekochu on Hugging Face (known community contributor of VCTK multi-speaker RVC benchmarking sets).

### Usage Rights & License Evaluation

- **Dataset License**: Open Data Commons Attribution License / Academic & Commercial use with attribution (University of Edinburgh).
- **Ethics & Privacy**: The speaker is an anonymized adult participant in a voluntary scientific speech corpus. Using this model does NOT involve impersonating private living individuals or celebrities without consent.
- **Permitted Use**: Permitted for academic, personal testing, development, and conversion pipeline benchmarking.

---

## 4. Technical Architecture & Verified Compatibility

### Independent Technical Verification

The following parameters were independently verified by inspecting checkpoint metadata and Faiss index binary headers (not relying solely on publisher claims):

| Property | Value | Verification Method | Compatibility Status |
| :--- | :--- | :--- | :--- |
| **Format** | Clean inference checkpoint | Inspected `ckpt.keys()`: contains `weight`, `config`, `info` (no `optimizer` or training-only states) | **PASS** |
| **RVC Version** | `v2` | `ckpt.get('version') == 'v2'` | **PASS** (Matches Applio Core v3.2.8) |
| **F0 Support** | `1` (F0-guided) | `ckpt.get('f0') == 1` | **PASS** (Requires F0 pitch estimator) |
| **Pitch Estimator** | RMVPE | Trained with RMVPE pitch extraction | **PASS** (Matches `models/checkpoints/rmvpe.pt`) |
| **Embedder Architecture** | ContentVec 768-dim | `ckpt['config'][4] == 768` | **PASS** (Matches `contentvec/pytorch_model.bin`) |
| **Sample Rate** | 40,000 Hz (40 kHz) | `ckpt['config'][-1] == 40000` | **PASS** (Supported natively by Applio) |
| **Vocoder** | HiFi-GAN | `ckpt.get('vocoder') == 'HiFi-GAN'` | **PASS** |
| **Index Dimensions** | `d = 768`, `ntotal = 47,446` | Inspected via `faiss.read_index` (`IVF1216,Flat`) | **PASS** (Exact 768-dim match with ContentVec) |
| **Hardware Suitability** | 55.2 MB (~120–150 MB VRAM) | FP16 weight allocation test | **PASS** (Safe for Quadro T2000 4GB VRAM) |

### RVCAdapter Compatibility

Verified via direct Python execution:

```python
adapter = RVCAdapter()
adapter.load_model('models/weights/vctk_female_p231.pth', 'models/indices/vctk_female_p231.index')
assert adapter.is_loaded is True
assert adapter.model_info == {'version': 'v2', 'f0': 1, 'target_sr': 40000, 'vocoder': 'HiFi-GAN'}
```

---

## 5. Candidate Comparison & Evaluation

During model selection, three candidate profiles were evaluated against the project requirements:

1. **Candidate 1: Vietnamese Female Speaking RVC Model**
   - *Status*: Rejected / Unavailable.
   - *Findings*: Public Hugging Face Vietnamese repositories (e.g., `AnhP/Vietnamese-RVC-Project`, anime/streamer models) exclusively contain celebrity/streamer voices, fictional anime characters, or models without clear training datasets, source licenses, or valid Faiss indices.
2. **Candidate 2: Nekochu VCTK Female p231 (`Fp231rmvpe`)**
   - *Status*: **SELECTED**.
   - *Strengths*: Highly documented provenance (CSTR VCTK), clear open dataset license, adult female conversational/reading speech, RMVPE-trained, RVC v2 768-dim ContentVec, matching 768-dim Faiss index, verified checksums.
3. **Candidate 3: Nekochu VCTK Female p249 (`Fp249rmvpe`)**
   - *Status*: Viable Alternative.
   - *Findings*: Same repo/architecture, Scottish English accent, slightly lower pitch baseline than p231.

---

## 6. Known Limitations & Remaining Uncertainties

1. **Acoustic / Language Priors**: Speaker p231 is a native English speaker. When converting Vietnamese speech, pitch shifting (+12 semitones) will provide a natural female vocal range, but timbre may carry subtle English vowel coloration rather than native Vietnamese regional phonetics.
2. **Inference Execution Unverified**: Offline voice conversion has not been run because no clean input speech fixture (`tests/fixtures/*.wav`) is present. Conversion quality must be validated once an authorized input audio fixture is supplied.
3. **Index Retrieval Ratio (`index_rate`)**: In real-time or low-latency mode, searching a 149 MB Faiss index may incur a small CPU overhead. Tuning `index_rate` (between 0.3 and 0.7) will be evaluated during Milestone 2 offline testing.

---

## 7. Investigation of Northern Vietnamese Female Target Models

Following feedback regarding the sharp and sour timbre of the VCTK p231 English baseline, an exhaustive search was conducted across public model repositories (Hugging Face, Voice-models, community archives) for a warm, soft, Northern Vietnamese female conversational RVC target model.

### Evaluated Candidates & Findings

1. **`AnhP/Vietnamese-RVC-Project`**:
   - *Type*: RVC tooling & pretrained base weights.
   - *Findings*: Contains 59 `.pth` files consisting exclusively of training base models (`f0G40k.pth`, `f0D40k.pth`), pitch predictors (`crepe`, `rmvpe`), and UVR stem separation models. Contains **zero trained target voice checkpoints** or Faiss indices.
   - *Verdict*: **REJECTED (Not an inference voice model)**.

2. **`Uchiha2026/Kurumi-RVC-v2-VietNamese`**:
   - *Type*: Anime character voice (`Kurumi.pth`).
   - *Findings*: Trained on Kurumi Tokisaki from the anime *Date A Live*. High-pitched, exaggerated anime vocalization without natural conversational cadence or documented speech corpus provenance.
   - *Verdict*: **REJECTED (Anime character voice; violates conversational adult female requirement)**.

3. **Community Celebrity & Streamer Checkpoints (Voice-Models.com / Weights.gg mirrors)**:
   - *Type*: Female streamer / celebrity imitation models (e.g. Nguyễn Phương Hằng, Như Quỳnh).
   - *Findings*: Public figure imitations lacking speaker consent or open licenses. Acoustically either singing-oriented or shouting/streamer style, lacking the required soft, mellow conversational quality. Missing verifiable dataset provenance and matching indices.
   - *Verdict*: **REJECTED (License violation, celebrity imitation, acoustic mismatch)**.

4. **Vietnamese Neural TTS Models (`toandev/F5-TTS-Vietnamese`, `hynt/F5-TTS-Vietnamese-ViVoice`, `Cong123779/matcha-tts-vietnamese`)**:
   - *Type*: Text-to-Speech (TTS) models.
   - *Findings*: Although trained on clean Northern Vietnamese female speech (e.g. VIVOS, ViVoice), these use diffusion transformer/Matcha text synthesis architectures, not RVC audio-to-audio voice conversion checkpoints.
   - *Verdict*: **INELIGIBLE (TTS model; cannot be loaded by RVCAdapter)**.

5. **Multi-Speaker Academic Corpora (VCTK p249, p244, p303)**:
   - *Type*: Licensed academic English reading speech.
   - *Findings*: While technically compatible and licensed, all speakers are native English speakers with Western phonetic priors, unable to resolve the linguistic mismatch with Vietnamese tones and vowels.
   - *Verdict*: **INSUFFICIENT (Lacks Northern Vietnamese accent and intonation)**.

### Conclusion & Recommended Path

No publicly shared pre-trained RVC model currently meets the strict criteria for an open-licensed, conversational Northern Vietnamese female speaking voice. In strict compliance with safety and licensing rules, no unverified or infringing checkpoint was downloaded.

**Safest Next Option**: Train a custom RVC v2 model on an open-licensed Vietnamese acoustic dataset (e.g. 15–30 minutes of clean Northern Vietnamese female speech from the open **VIVOS** corpus or an authorized voice recording) directly using the pinned Applio Core v3.2.8 training pipeline.
