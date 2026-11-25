"""Original, procedurally synthesised background music and sound effects.

Everything here is generated from scratch with numpy -> no third-party copyright,
no Content-ID claims. Moods map to niches (space -> 'cosmic', tech -> 'pulse', ...).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
rng = np.random.default_rng()

NOTE = {n: i for i, n in enumerate(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}


def hz(note: str, octave: int) -> float:
    return 440.0 * 2 ** ((NOTE[note] - 9) / 12 + (octave - 4))


MOODS = {
    # chords: (root, minor?) progression; bpm; brightness
    "cosmic":     {"prog": [("A", 1), ("F", 0), ("C", 0), ("G", 0)], "bpm": 84, "bright": 0.35, "arp": True,  "kick": False},
    "mystery":    {"prog": [("D", 1), ("A#", 0), ("G", 1), ("A", 0)], "bpm": 78, "bright": 0.25, "arp": True,  "kick": False},
    "pulse":      {"prog": [("E", 1), ("C", 0), ("G", 0), ("D", 0)],  "bpm": 112, "bright": 0.55, "arp": True,  "kick": True},
    "epic":       {"prog": [("C", 1), ("G#", 0), ("D#", 0), ("A#", 0)], "bpm": 96, "bright": 0.45, "arp": False, "kick": True},
    "playful":    {"prog": [("C", 0), ("A", 1), ("F", 0), ("G", 0)],  "bpm": 118, "bright": 0.6, "arp": True,  "kick": True},
}

NICHE_MOOD = {"space": "cosmic", "mystery": "mystery", "ai": "pulse", "technology": "pulse", "future": "epic",
              "facts": "playful", "psychology": "mystery", "gaming": "pulse"}


def _env(n: int, a: float, r: float) -> np.ndarray:
    e = np.ones(n)
    na, nr = min(int(a * SR), n), min(int(r * SR), n)
    if na:
        e[:na] = np.linspace(0, 1, na)
    if nr:
        e[-nr:] *= np.linspace(1, 0, nr)
    return e


def _lowpass(x: np.ndarray, cutoff: float) -> np.ndarray:
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(x.shape[0], 1 / SR)
    resp = 1 / np.sqrt(1 + (f / cutoff) ** 4)
    return np.fft.irfft(X * (resp[:, None] if x.ndim == 2 else resp), n=x.shape[0], axis=0)


def _reverb(x: np.ndarray, decay: float = 2.2, wet: float = 0.35) -> np.ndarray:
    n = int(decay * SR)
    t = np.arange(n) / SR
    out = np.empty_like(x)
    for ch in range(x.shape[1]):
        ir = rng.standard_normal(n) * np.exp(-t * 6.9 / decay)
        ir[: int(0.02 * SR)] *= np.linspace(0, 1, int(0.02 * SR))
        ir /= np.sqrt((ir ** 2).sum())
        L = x.shape[0] + n
        y = np.fft.irfft(np.fft.rfft(x[:, ch], L) * np.fft.rfft(ir, L), L)[: x.shape[0]]
        out[:, ch] = (1 - wet) * x[:, ch] + wet * y
    return out


def _saw_stack(freq: float, n: int, detune=(-0.11, 0.0, 0.09)) -> np.ndarray:
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    for i, d in enumerate(detune):
        f = freq * 2 ** (d / 12)
        ph = rng.random()
        s = 2 * ((t * f + ph) % 1) - 1
        pan = 0.5 + 0.35 * (i - 1)
        out[:, 0] += s * (1 - pan)
        out[:, 1] += s * pan
    return out / len(detune)


def music(duration: float, mood: str = "cosmic", out: Path | None = None) -> np.ndarray:
    m = MOODS.get(mood, MOODS["cosmic"])
    beat = 60 / m["bpm"]
    bar = beat * 4
    n_total = int((duration + 2) * SR)
    mix = np.zeros((n_total, 2))
    bars = int(np.ceil((duration + 2) / bar))
    for b in range(bars):
        root, minor = m["prog"][b % len(m["prog"])]
        third = 3 if minor else 4
        start = int(b * bar * SR)
        n = min(int(bar * SR * 1.15), n_total - start)
        if n <= 0:
            break
        f0 = hz(root, 3)
        pad = sum(_saw_stack(f0 * 2 ** (iv / 12), n) for iv in (0, third, 7, 12)) * 0.16
        pad = _lowpass(pad, 500 + 2200 * m["bright"]) * _env(n, 0.6, 0.5)[:, None]
        mix[start:start + n] += pad
        # sub bass
        t = np.arange(n) / SR
        sub = np.sin(2 * np.pi * hz(root, 1 if NOTE[root] > 4 else 2) * t) * 0.22 * _env(n, 0.05, 0.3)
        mix[start:start + n] += sub[:, None]
        # arpeggio
        if m["arp"]:
            step = beat / 2
            notes = [0, third, 7, 12, 7, third]
            for k in range(int(bar / step)):
                s0 = start + int(k * step * SR)
                ln = int(step * SR * 0.9)
                if s0 + ln > n_total:
                    break
                tt = np.arange(ln) / SR
                f = f0 * 4 * 2 ** (notes[k % len(notes)] / 12)
                tone = (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt)) * np.exp(-tt * 7) * 0.06
                mix[s0:s0 + ln, 0] += tone * (0.6 if k % 2 else 0.4)
                mix[s0:s0 + ln, 1] += tone * (0.4 if k % 2 else 0.6)
        if m["kick"] and b > 0:
            for k in range(4):
                s0 = start + int(k * beat * SR)
                ln = int(0.35 * SR)
                if s0 + ln > n_total:
                    break
                tt = np.arange(ln) / SR
                kick = np.sin(2 * np.pi * (45 + 90 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 9) * 0.35
                mix[s0:s0 + ln] += kick[:, None]
    mix = _reverb(mix, 2.8, 0.3)
    mix = mix[: int(duration * SR)]
    mix *= _env(len(mix), 1.5, 2.5)[:, None]
    mix /= max(1e-6, np.abs(mix).max()) / 0.8
    if out:
        sf.write(out, mix.astype(np.float32), SR)
    return mix


# ---------------------------------------------------------------- SFX
def _noise(n):
    return rng.standard_normal(n)


def sfx_whoosh(d=0.6):
    n = int(d * SR)
    t = np.linspace(0, 1, n)
    x = _noise(n)
    # sweeping band via moving lowpass in blocks
    out = np.zeros(n)
    blocks = 24
    for i in range(blocks):
        a, b = i * n // blocks, (i + 1) * n // blocks
        cut = 300 + 5000 * np.sin(np.pi * (i + 0.5) / blocks) ** 2
        out[a:b] = _lowpass(x[a:b].copy(), cut)
    env = np.sin(np.pi * t) ** 2
    pan = t
    return np.stack([out * env * (1 - pan * 0.6), out * env * (0.4 + pan * 0.6)], 1) * 0.5


def sfx_impact(d=1.4):
    n = int(d * SR)
    t = np.arange(n) / SR
    boom = np.sin(2 * np.pi * (38 + 120 * np.exp(-t * 18)) * t) * np.exp(-t * 3.2)
    crack = _lowpass(_noise(n), 2500) * np.exp(-t * 25) * 0.6
    x = boom + crack
    return _reverb(np.stack([x, x], 1), 1.8, 0.25) * 0.8


def sfx_riser(d=1.5):
    n = int(d * SR)
    t = np.arange(n) / SR
    f = 200 * (1 + 6 * (t / d) ** 2)
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.3
    noise = _lowpass(_noise(n), 3000) * 0.4
    env = (t / d) ** 2
    x = (tone + noise) * env
    return np.stack([x, x], 1) * 0.6


def sfx_pop(d=0.12):
    n = int(d * SR)
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * (900 - 500 * t / d) * t) * np.exp(-t * 45)
    return np.stack([x, x], 1) * 0.35


def sfx_glitch(d=0.35):
    n = int(d * SR)
    x = np.zeros(n)
    for _ in range(9):
        a = rng.integers(0, n - 800)
        ln = rng.integers(200, 800)
        x[a:a + ln] += np.sign(np.sin(np.arange(ln) * rng.uniform(0.05, 0.6))) * rng.uniform(0.1, 0.4)
    return np.stack([x, np.roll(x, 60)], 1) * 0.5


def sfx_ding(d=1.2):
    n = int(d * SR)
    t = np.arange(n) / SR
    x = sum(np.sin(2 * np.pi * f * t) * a for f, a in ((1318.5, 0.5), (2637, 0.2), (3951, 0.08))) * np.exp(-t * 4)
    return _reverb(np.stack([x, x], 1), 1.2, 0.3) * 0.35


SFX = {"whoosh": sfx_whoosh, "impact": sfx_impact, "riser": sfx_riser, "pop": sfx_pop,
       "glitch": sfx_glitch, "ding": sfx_ding}


def build_sfx_library(folder: Path) -> dict[str, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, fn in SFX.items():
        p = folder / f"{name}.wav"
        if not p.exists():
            sf.write(p, fn().astype(np.float32), SR)
        out[name] = p
    return out


def sfx_track(events: list[tuple[float, str, float]], duration: float, lib: dict[str, Path], out: Path):
    """events: (time, name, gain) -> one stereo wav."""
    mix = np.zeros((int((duration + 2) * SR), 2), np.float32)
    for t, name, gain in events:
        if name not in lib:
            continue
        a, _ = sf.read(lib[name], dtype="float32")
        if a.ndim == 1:
            a = np.stack([a, a], 1)
        s = int(max(0, t) * SR)
        e = min(len(mix), s + len(a))
        mix[s:e] += a[: e - s] * gain
    mix = mix[: int(duration * SR)]
    peak = np.abs(mix).max()
    if peak > 0.95:
        mix *= 0.95 / peak
    sf.write(out, mix, SR)
