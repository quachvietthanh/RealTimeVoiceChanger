"""Minimal headless use of the GUI backend API (docs/gui-backend-contract.md).

By default it only inspects: lists profiles (with full validation), devices and the status, and loads
nothing onto the GPU. With --run SECONDS it loads --profile and streams from the default microphone to
the default output (or --output-device) for that many seconds, printing live metrics.

    .venv\\Scripts\\python scripts/backend_api_example.py
    .venv\\Scripts\\python scripts/backend_api_example.py --profile voice_b_preview --run 20 --output-device 7
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voice_changer.service import AudioSettings, BackendError, VoiceChangerService  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=None, help="profile id (default: the registry default)")
    ap.add_argument("--run", type=float, default=0.0, help="stream for this many seconds (0 = inspect only)")
    ap.add_argument("--input-device", type=int, default=None)
    ap.add_argument("--output-device", type=int, default=None)
    ap.add_argument("--noise-suppression", default="off", help="off, or a mode: light, medium, strong")
    ap.add_argument("--legacy-seams", action="store_true",
                    help="join chunks end to end (old behaviour, audible seam every chunk); for A/B listening only")
    args = ap.parse_args()

    svc = VoiceChangerService()
    svc.add_status_listener(lambda s: print(f"  -> state {s.state.value} profile={s.profile_id}"))

    print("Profiles:")
    for p in svc.list_profiles():
        v = svc.validate_profile(p.profile_id)
        print(f"  {p.profile_id:<18} {p.status:<9} ok={v.ok} files={p.files_present} defaults={p.defaults}")
        for e in v.errors:
            print(f"      error: {e}")

    inv = svc.list_devices()
    print(f"Devices (default in={inv.default_input_id}, out={inv.default_output_id}):")
    for d in inv.inputs:
        print(f"  in  {d.device_id:>3} [{d.host_api}] {d.name}")
    for d in inv.outputs:
        print(f"  out {d.device_id:>3} [{d.host_api}] {d.name}{'  (virtual cable)' if d.is_virtual_cable else ''}")
    print(f"Status: {svc.get_status()}")

    if args.run <= 0:
        return 0
    try:
        svc.select_profile(args.profile or svc.default_profile_id)
        seams = dict(context_s=0.0, crossfade_s=0.0, sola_search_s=0.0) if args.legacy_seams else {}
        if args.noise_suppression != "off":
            print(svc.set_noise_suppression(enabled=True, mode=args.noise_suppression))
        svc.start(AudioSettings(input_device_id=args.input_device, output_device_id=args.output_device, **seams))
        end = time.monotonic() + args.run
        while time.monotonic() < end:
            time.sleep(1.0)
            st, m = svc.get_status(), svc.get_metrics() or {}
            print(f"  {st.state.value} chunks={m.get('chunks_converted')} rtf={m.get('rtf')} "
                  f"buffer={m.get('current_buffer_ms')}ms steady_underflows={m.get('steady_underflows')}")
            if st.state.value == "error":
                print(f"  error: {st.last_error.message} | {st.last_error.hint} | {st.last_error.details}")
                break
    except BackendError as err:
        print(f"{err.category}/{err.code}: {err.message}\n  hint: {err.hint}\n  details: {err.details}")
        return 1
    finally:
        svc.shutdown()
    print(f"Final metrics: {svc.get_metrics()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
