# Audio Routing and VB-Cable

**Status:** backend support implemented (device enumeration, capability checks, virtual-cable detection,
stream-loss detection). **Not physically verified:** on 2026-10-21 no VB-Cable device was installed on the
development machine, and no microphone-to-speaker or Discord/OBS test was run. Follow the manual
procedure below and record the result.

## Signal path

```
Microphone ──> [input stream, 40 kHz mono] ──> RVC conversion ──> [output stream, 40 kHz stereo] ──> output device
```

- To hear yourself: output = speakers/headphones.
- To feed Discord/OBS/games: output = **"CABLE Input (VB-Audio Virtual Cable)"** (a playback device).
  In Discord, set the input device to **"CABLE Output (VB-Audio Virtual Cable)"** (a recording device).
- Choosing "CABLE Output" as *our* output, or "CABLE Input" as Discord's input, does not work: the names
  are from the cable's point of view.

## Device selection in the backend

- `VoiceChangerService.list_devices()` lists all endpoints; `AudioDevice.is_virtual_cable` marks VB-Audio /
  VoiceMeeter style endpoints by name.
- Each device appears once per Windows host API. Prefer **MME** or **DirectSound** entries: they resample
  to 40 kHz. WASAPI shared mode and WDM-KS reject 40 kHz (`audio_device_unavailable` with
  "rejects 40000 Hz"). Checked on 2026-10-21 for the USB microphone and speakers of the development
  machine with `sounddevice.check_*_settings`: MME and DirectSound accept, WASAPI and WDM-KS return
  `Invalid sample rate [PaErrorCode -9997]`. Other devices and the VB-Cable endpoints are unchecked.
- Ids change when devices are plugged/unplugged. Store `(name, host_api)` and re-resolve
  (`AudioDeviceManager.find_device`). MME truncates names to 31 characters.
- `validate_audio_settings()` / `start()` check that the device exists, has channels and accepts 40 kHz
  (`sounddevice.check_*_settings`) before opening streams.
- If a device disappears while streaming, `get_status()` reports `error` / `stream_failure`. Call
  `stop()`, refresh devices, `start()` again.

## Installing VB-Cable (owner)

1. Download VB-CABLE Driver from vb-audio.com; extract; run `VBCABLE_Setup_x64.exe` as administrator; reboot.
2. Do **not** make "CABLE Input" the Windows default playback device (system sounds would go into the cable).
3. Check: `.venv\Scripts\python scripts/backend_api_example.py`: a line ending with `(virtual cable)` must
   appear for "CABLE Input" under outputs.

## Manual test procedure (record results in the handoff)

Preconditions: VB-Cable installed; headphones (avoid speaker→mic feedback); GPU free of other heavy apps.

1. **Self-monitor:** `.venv\Scripts\python scripts/backend_api_example.py --profile voice_a_preview --run 30 --output-device <headphones id>`.
   Speak. Expect converted voice after ~1 s. Record: `steady_underflows`, `chunks_dropped`, `rtf`,
   `input_overflows` from the final metrics; audible clicks/gaps.
2. **Cable:** same with `--output-device <CABLE Input id>`. In Windows Sound settings → Recording →
   "CABLE Output" → Properties → Listen → "Listen to this device" via headphones to monitor, or use step 3.
3. **Discord:** Settings → Voice & Video → Input Device = "CABLE Output (VB-Audio Virtual Cable)"; disable
   Discord's noise suppression/echo cancellation for the first test; use "Let's Check" (mic test) or a
   test call. Input sensitivity: manual, low threshold.
4. **OBS:** Add "Audio Input Capture" → device "CABLE Output"; record 30 s; listen to the recording.
5. **Device loss:** while streaming, unplug the USB microphone. Expect `get_status().state == "error"` with
   code `stream_failure` within one status poll; `stop()` must return to `ready`.
6. **Profile switch:** stop, select `voice_b_preview`, start; confirm the voice changes and VRAM
   (`nvidia-smi`) does not grow by a second model.

Record: date, devices (name + host API), profile, settings, the metrics above, and what was heard.

### Choppy / robotic live audio: physical check (owner, ~10 min, headphones)

Each stop writes one `session N metrics: {...}` line to `%LOCALAPPDATA%\RealTimeVoiceChanger\logs\backend.log`;
copy those lines into the report.

1. **Devices:** `.venv\Scripts\python scripts/device_probe.py --input <mic id> --output <headphones id> --seconds 20 --json-output output/<new dir>/probe.json`
   (plays silence; stores no audio). Expect `measured_rate_hz` ≈ 40000 on both sides, no `status_flags`,
   `late_callbacks_gt_2x` 0. Speak normally in the idle phase: `input_peak_dbfs` should be about −20…−6 dBFS
   (lower = raise the Windows mic level; 0 = clipping).
2. **Seam A/B by ear:** in the GUI (or `backend_api_example.py`), same profile and settings, 30 s each:
   (a) default settings; (b) `AudioSettings(context_s=0, crossfade_s=0, sola_search_s=0)` (old joining).
   Read the same sentence. (b) should have a click/level jump twice per second; (a) should not.
3. **Model vs streaming:** listen to `output/live_audio_diag_2026-10-21/offline_compare/<profile>__whole.wav`
   (no chunking) vs `__stitch.wav` (current streaming path, offline). If `whole` already sounds robotic, the
   cause is the model/settings, not streaming.
4. **Acoustic latency (optional):** clap near the mic with headphones held to the mic, record the room with a
   phone; the time between the two claps in the recording is the acoustic latency.

### Noise suppression: owner check (headphones, ~10 min)

1. **Offline first:** `output/noise_suppression_2026-10-21/README.md` lists the files and listening order
   (denoiser alone, then A and B converted, off/light/medium/strong, clean and synthetic-noisy input).
2. **Live, same sentence each time,** with a steady noise source on (fan, PC): run
   `.venv\Scripts\python scripts/backend_api_example.py --profile voice_a_preview --run 30 --input-device <mic> --output-device <headphones>`
   once with `--noise-suppression off`, then `light`, `medium`, `strong` (or toggle it in the GUI once it
   exposes `set_noise_suppression`). Wait ~2 s after starting before judging (noise estimate settling).
3. Listen for: Vietnamese tones (sắc, hỏi, ngã, nặng) on soft syllables, final consonants and the
   fricatives s/x/th/ph, breathy or whispered words, "underwater"/warbling ("musical noise") in pauses.
4. Nothing is recorded unless you choose to; the backend never saves microphone audio.

Report: probe JSON, the session metrics lines (`steady_underflows`, `chunks_dropped`, `input_overflows`,
`input_status_flags`, `output_status_flags`, `stitch_resets`, `estimated_software_latency_ms`), and for 2–3
what was heard.

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `audio_device_unavailable … rejects 40000 Hz` | WASAPI/WDM-KS entry selected | pick the MME or DirectSound entry of the same device |
| `stream_open_failed` | device in exclusive use, or unplugged | close the other app / reconnect; refresh devices |
| Discord hears nothing | Discord input not set to "CABLE Output", or our output not "CABLE Input" | fix both sides; check Windows recording level of "CABLE Output" |
| Choppy: click/level jump about twice per second, `steady_underflows` 0 | chunk seams (fixed by the default `context_s`/`crossfade_s`/`sola_search_s`; check they are not 0) | keep the defaults |
| Robotic or choppy voice, `steady_underflows` rising | GPU overloaded (`rtf` near 1) | close GPU apps; keep chunk 0.5 s |
| Robotic in the offline `__whole.wav` too | the model or its settings, not streaming | owner listening / tuning (pitch, index rate, protect) |
| `chunks_dropped` rising | inference slower than real time | as above |
| Echo/feedback | speakers feeding the mic | use headphones |
| Voice B much quieter than A | B's model output level (~9 dB lower) | raise the Discord/OBS input gain; no backend gain control yet |
