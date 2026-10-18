# Model Storage and Guidelines

This directory is the local repository for RVC voice models, feature retrieval indices, and related checkpoints.

## Directory Layout

```text
models/
├── README.md               # Model setup instructions and policies
├── weights/                # Model weight files (.pth) [git-ignored]
├── indices/                # Feature retrieval indices (.index) [git-ignored]
└── checkpoints/            # Intermediate checkpoints or ONNX models [git-ignored]
```

## Setup Instructions

1. **Placing Model Weights (`.pth`):**
   - Place trained RVC voice conversion weight files in `models/weights/`.
   - Ensure the filename is descriptive (e.g., `female_voice_v2_48k.pth`).

2. **Placing Feature Index Files (`.index`):**
   - Place the matching Faiss feature index file in `models/indices/`.
   - Ensure it corresponds to the exact model checkpoint being loaded.

3. **Pitch Predictor / Feature Extractor Weights:**
   - Any auxiliary models (such as `rmvpe.pt` for pitch extraction or `hubert_base.pt` for content features) should be placed in `models/checkpoints/` as required by the chosen inference implementation.

## Strict Git Policy

> **WARNING:** **Never commit model weights, indices, or checkpoints to Git.**
>
> All `.pth`, `.pt`, `.ckpt`, `.onnx`, and `.index` files are explicitly excluded via `.gitignore`.
> Voice model weights are large binary assets (often hundreds of megabytes) and must be managed locally or through dedicated model artifact storage, never in this Git repository.
