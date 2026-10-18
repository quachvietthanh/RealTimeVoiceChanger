# Configuration Management

This directory stores configuration templates and parameter specifications for RealTimeVoiceChanger.

## Guidelines

- **Default Configuration**: Base defaults should be provided as version-controlled templates (e.g., `default_config.json` or `config.example.yaml`).
- **Local Overrides**: Machine-specific configurations, such as hardware device IDs, local audio buffer preferences, and local model paths, should be placed in `*.local.json` or `*.local.yaml`.
- **Git Exclusions**: The root `.gitignore` explicitly ignores `*.local.json`, `*.local.yaml`, and `.env` files to prevent machine-specific paths, device identifiers, and sensitive credentials from being committed to Git.

## Planned Configuration Schema

When fully implemented, configuration profiles will define:

1. **Audio Device Settings**:
   - `input_device_id` / `input_device_name`: Capture device (e.g. USB microphone).
   - `output_device_id` / `output_device_name`: Render device (e.g. CABLE Input / VB-Audio Virtual Cable).
   - `sample_rate`: Target sampling rate (e.g., 40000 Hz or 48000 Hz).
   - `buffer_size` / `block_size`: Audio chunk size in samples.

2. **Model Selection**:
   - `model_path`: Relative or absolute path to local `.pth` weight file.
   - `index_path`: Path to matching `.index` retrieval file (optional).

3. **Conversion & Tuning Parameters**:
   - `pitch_shift`: Semitone transposition (e.g. +12 semitones for male-to-female).
   - `f0_method`: Pitch extraction algorithm (e.g. `rmvpe`, `pm`, `harvest`).
   - `index_rate`: Search feature retrieval ratio (0.0 to 1.0).
   - `filter_radius`: Median filter radius for pitch.
   - `protect_rate`: Voiceless consonant and breath protection factor.
