"""Check whether the np.pad(mode="edge") length fix-up in RVCAdapter.convert_chunk fires.

For each 0.5 s chunk, records:
    raw_len      - samples returned by Applio's pipeline (before any length fix-up)
    expected_len - length the adapter must return
    final_len    - length after the adapter's crop / edge fix-up
    edge_fired   - True when raw_len < expected_len (the np.pad(mode="edge") branch runs)

The pipeline call is wrapped on the Applio instance only; adapter code is not modified.

Usage (repository root, project venv):
    .venv\\Scripts\\python.exe scripts\\diagnose_edge_pad.py --pads "80,80" "0,160"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from voice_changer.conversion.base import ConversionConfig  # noqa: E402
from voice_changer.conversion.rvc.adapter import RVCAdapter  # noqa: E402
from ab_padding_test import CHUNK, SR, make_signal  # noqa: E402


def parse_pad(text: str) -> tuple[int, int]:
    left, right = (int(v) for v in text.split(","))
    return left, right


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the edge-pad branch of RVCAdapter")
    parser.add_argument("--pads", nargs="+", default=["80,80", "0,160"])
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", type=str, default="output/ab_padding/edge_branch_check.json")
    args = parser.parse_args()

    model = REPO_ROOT / "models" / "weights" / "vctk_female_p231.pth"
    index = REPO_ROOT / "models" / "indices" / "vctk_female_p231.index"
    adapter = RVCAdapter()
    adapter.load_model(str(model), str(index))
    vc = adapter._ensure_applio_initialized()
    original_pipeline = vc.vc.pipeline

    raw_lengths: list[int] = []

    def recording_pipeline(*a, **k):
        out = original_pipeline(*a, **k)
        raw_lengths.append(int(len(out)))
        return out

    vc.vc.pipeline = recording_pipeline

    cfg = ConversionConfig(pitch_shift=10.0, f0_method="rmvpe", index_rate=0.5, filter_radius=2, protect_rate=0.4)
    x = make_signal(args.seconds, seed=args.seed)
    results = {}
    for text in args.pads:
        adapter.pad_16k = parse_pad(text)
        rows = []
        for i, start in enumerate(range(0, len(x), CHUNK)):
            piece = x[start : start + CHUNK]
            raw_lengths.clear()
            final = np.asarray(adapter.convert_chunk(piece, cfg, input_sr=SR), dtype=np.float32)
            raw = raw_lengths[-1]
            expected = round(len(piece) * (SR / SR))
            rows.append(
                {
                    "chunk": i,
                    "raw_len": raw,
                    "expected_len": expected,
                    "final_len": int(len(final)),
                    "edge_fired": raw < expected,
                    "edge_pad_samples": max(0, expected - raw),
                }
            )
        fired = sum(r["edge_fired"] for r in rows)
        print(f"pad={text}: edge fired in {fired}/{len(rows)} chunks")
        for r in rows:
            print(f"  chunk {r['chunk']:2d}: raw={r['raw_len']} expected={r['expected_len']} final={r['final_len']} edge={r['edge_fired']} pad={r['edge_pad_samples']}")
        results[text] = rows

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seed": args.seed, "seconds": args.seconds, "results": results}, indent=2), encoding="utf-8")
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
