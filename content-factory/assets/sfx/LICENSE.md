# SFX library – licence

All `.wav` files in this folder are **original works**, synthesised from scratch by
the ContentFactory project owner with the code in `factory/sfx_design.py`
(numpy oscillators, filtered noise, FM synthesis, envelopes and a convolution
reverb built from generated noise). **No recordings, samples or third-party sound
libraries were used.**

They are dedicated to the public domain under
**CC0 1.0 Universal** (https://creativecommons.org/publicdomain/zero/1.0/):
you may copy, modify, distribute and use them, also commercially and in
monetised videos, without asking permission and without attribution.

## How they were made

* Design method: layered sound design (transient + body + tail + sweetener,
  each in its own frequency lane, moving pitch, short lengths, high-passed to
  avoid mud, top end tamed), following the `game-sfx` workflow; every file was
  audited with its `doctor` (peak, LUFS, crest, spectral balance, spectrogram).
* Format: 48 kHz, 16-bit PCM, stereo, true-peak-normalised to -1 dBTP (4x oversampled), tail trimmed
  at -60 dB with a 15 ms fade-out. The onset stays at t = 0 so timed cues line
  up (whoosh peaks ~0.25 s in, risers end exactly at their nominal length).
* Rebuild (deterministic, identical output):
  `python -m factory.sfx_design` (writes into `assets/sfx/`).

## Contents

| Name | Variants | Use |
|---|---|---|
| `whoosh_1..3` / `whoosh_long_1..2` | 3 / 2 | scene transitions (peak at ~0.25 s / ~0.5 s) |
| `swoosh_reverse_1..2` | 2 | suck-in that slams into a cut |
| `impact_1..2` | 2 | cinematic boom (sub + punch + crack + debris + room) |
| `sub_drop_1..2` | 2 | sub-bass drop with harmonics (audible on phones) |
| `riser` / `riser_long` | 1.5 s / 3 s | build-ups ending on the cut |
| `pop_1..3`, `tick_1..3` | 3 / 3 | UI pops, clicks |
| `glitch_1..3` | 3 | digital stutter |
| `ding_1..2`, `shimmer` | 2 / 1 | FM bell, shimmering overtone cluster |
| `swish_text_1..4` | 4 | very short, subtle word-pop swish |
| `shutter` | 1 | camera shutter |
| `notification_1..2` | 2 | two-note chat ping |
| `heartbeat` | 1 | lub-dub ×2 |
| `typing` | 1 | short keyboard burst |
| `sparkle` | 1 | magic sparkle |
| `braam` | 1 | deep trailer braam |
| `zap_1..3` | 3 | transition zap |
