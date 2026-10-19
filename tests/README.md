# Test Suite

This directory contains the automated test suite for RealTimeVoiceChanger.

## Directory Structure

```text
tests/
├── README.md               # Test suite guidelines and instructions
├── fixtures/               # Small, version-controlled audio test samples (clean, non-private)
├── integration/            # Multi-component and stream pipeline tests
└── unit/                   # Isolated component and logic tests
```

## Testing Guidelines

1. **Unit Tests (`tests/unit/`)**:
   - Verify isolated logic, configuration schema parsing, data structures, tuning presets, and protocol definitions.
   - Must run fast and without requiring external hardware or GPU access.

2. **Integration Tests (`tests/integration/`)**:
   - Verify audio buffer ingestion, mock conversion adapters, pipeline stream flows, and device listing handlers.
   - Test end-to-end routing using mock streams or virtual loopback devices.

3. **Audio Fixture Safety**:
   - Only small, non-private, synthetic, or public domain speech test samples may be committed to `tests/fixtures/`.
   - Never commit personal voice recordings or sensitive audio files.

## Running Tests

```powershell
# Run all unit tests
python -m unittest discover -s tests

# Run offline conversion regression test
python scripts/offline_conversion_test.py
```
