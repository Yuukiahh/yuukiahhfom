"""Sound design for the short-form-video SFX library (assets/sfx/*.wav).

Every sound is synthesised from scratch (noise, oscillators, filters, envelopes,
convolution reverb) -> original work, no samples, no licence questions.
Built with the layering rules of the game-sfx skill: transient + body + tail
(+ sweetener), each in its own frequency lane, moving pitch, short lengths, no
mud, tamed top end.

Regenerate the committed library (deterministic):
    python -m factory.sfx_design            # -> assets/sfx/
    python -m factory.sfx_design OUT_DIR
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from . import audio_dsp as d

SR = 48000
PEAK_DB = -1.0


# ---------------------------------------------------------------- building blocks
def _t(n: int, sr: int) -> np.ndarray:
    return np.arange(n) / sr


def _wide_noise(n: int, rng, width: float = 0.6) -> np.ndarray:
    """Partly decorrelated stereo noise (width 0 = mono, 1 = independent)."""
    c = rng.standard_normal(n)
    return np.stack([c * (1 - width) + rng.standard_normal(n) * width,
                     c * (1 - width) + rng.standard_normal(n) * width], 1)


def _pink(n: int, rng, sr: int) -> np.ndarray:
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / sr)
    X /= np.sqrt(np.maximum(f, 20))
    y = np.fft.irfft(X, n)
    return y / (np.std(y) + 1e-12)


def _env_peak(n: int, sr: int, peak: float, rise: float = 2.0, fall: float = 2.0) -> np.ndarray:
    """0 -> 1 at `peak` seconds -> 0 at the end, with power-curve shapes."""
    t = _t(n, sr)
    end = n / sr
    up = np.clip(t / peak, 0, 1) ** rise
    down = np.clip(1 - (t - peak) / max(1e-6, end - peak), 0, 1) ** fall
    return np.where(t < peak, up, down)


def _reverb(x: np.ndarray, sr: int, rng, decay: float, mix: float, hp: float = 200, damp: float = 6000) -> np.ndarray:
    ir = d.make_ir(sr, decay, rng, predelay=0.012, damp_hz=damp)
    tail = int(decay * sr * 0.8)
    xx = np.concatenate([x, np.zeros((tail, 2))])
    wet = d.convolve(xx, ir, hp=hp, sr=sr)
    return xx * (1 - mix * 0.5) + wet * mix


def _st(x: np.ndarray) -> np.ndarray:
    return np.stack([x, x], 1) if x.ndim == 1 else x


def _osc_sine(f: np.ndarray, sr: int, ph0: float = 0.0) -> np.ndarray:
    return np.sin(d.sweep_phase(f, sr) + ph0)


def _glide(n: int, sr: int, f0: float, f1: float, tau: float | None = None) -> np.ndarray:
    """Exponential glide f0 -> f1 (over the whole length, or time constant tau)."""
    t = _t(n, sr)
    if tau is None:
        return f0 * (f1 / f0) ** (t / (n / sr))
    return f1 + (f0 - f1) * np.exp(-t / tau)


# ---------------------------------------------------------------- the sounds
def whoosh(rng, sr=SR, dur=0.6, peak=0.25, lo=280.0, hi=2600.0, end=700.0, q=1.4, direction=1, body=0.5):
    n = int(dur * sr)
    t = _t(n, sr)
    noise = _wide_noise(n, rng, 0.45)
    curve = [(0, lo), (peak, hi), (dur, end)]
    air = d.tv_filter(noise, sr, "bandpass", curve, q=q)
    low = d.tv_filter(noise[:, ::-1].copy(), sr, "lowpass", [(0, lo * 0.6), (peak, hi * 0.25), (dur, end * 0.4)])
    x = air + body * low
    env = _env_peak(n, sr, peak, 2.2, 1.6)
    p = np.clip((t / dur - 0.5) * 1.6 * direction, -0.8, 0.8)  # moves across the stereo field
    gl, gr = np.cos((p + 1) * np.pi / 4), np.sin((p + 1) * np.pi / 4)
    out = x * env[:, None] * np.stack([gl, gr], 1) * np.sqrt(2)
    return d.fft_filter(out, sr, "highpass", 120)


def swoosh_reverse(rng, sr=SR, dur=0.8, hi=7000.0):
    """Reverse-cymbal style suck-in that slams to silence at the end."""
    n = int(dur * sr)
    t = _t(n, sr)
    noise = _wide_noise(n, rng, 0.7)
    x = d.tv_filter(noise, sr, "bandpass", [(0, 900), (dur, hi)], q=0.8)
    x += 0.35 * d.tv_filter(noise, sr, "lowpass", [(0, 300), (dur, 2500)])
    env = (t / dur) ** 3.2
    out = x * env[:, None]
    out[-int(0.012 * sr):] *= np.linspace(1, 0, int(0.012 * sr))[:, None]
    return d.fft_filter(d.fft_filter(out, sr, "lowpass", 10000, order=3), sr, "highpass", 150)


def impact(rng, sr=SR, dur=1.6, sub=(62, 34), punch=(150, 48), crack=1.0, tail=1.4):
    n = int(dur * sr)
    t = _t(n, sr)
    sub_f = _glide(n, sr, sub[0], sub[1], tau=0.35)
    rel = d.adsr(n, sr, 0.0005, 0, 1, dur * 0.5, 1.5)
    s = np.tanh(1.8 * _osc_sine(sub_f, sr)) * np.exp(-t * 3.2) * rel
    # driven punch: harmonics up to ~1 kHz so the hit reads on phone speakers
    p = np.tanh(2.5 * _osc_sine(_glide(n, sr, punch[0], punch[1], tau=0.05), sr)) * np.exp(-t * 11)
    cr = d.fft_filter(rng.standard_normal(n), sr, "highpass", 1500) * np.exp(-t * 70) * crack
    cr = d.fft_filter(cr, sr, "lowpass", 9000)
    thud = d.fft_filter(rng.standard_normal(n), sr, "bandpass", 320, q=0.9) * np.exp(-t * 16) * 1.6
    debris = d.tv_filter(rng.standard_normal(n), sr, "lowpass", [(0, 4000), (dur, 300)]) * np.exp(-t * 5) * 0.12
    x = _st(0.9 * s + 0.8 * p + 0.5 * cr + thud)
    x[:, 0] += debris
    x[:, 1] += np.roll(debris, 240)
    x = _reverb(x, sr, rng, tail, 0.35, hp=250)
    return d.fft_filter(x, sr, "highpass", 25)


def sub_drop(rng, sr=SR, dur=1.3, f0=95.0, f1=30.0):
    n = int(dur * sr)
    t = _t(n, sr)
    f = _glide(n, sr, f0, f1)
    s = _osc_sine(f, sr)
    # driven copy adds 2nd/3rd harmonics so phone speakers can hear the drop
    harm = np.tanh(3.0 * s) - 0.8 * s
    x = (s + 0.55 * harm) * d.adsr(n, sr, a=0.004, d=0.0, s=1.0, r=dur * 0.85, curve=1.6)
    click = d.fft_filter(rng.standard_normal(n), sr, "bandpass", 2500, q=1) * np.exp(-t * 200) * 0.25
    return d.fft_filter(_st(x + click), sr, "highpass", 22)


def riser(rng, sr=SR, dur=1.5, semis=19, root=110.0):
    """Builds to its loudest point exactly at the end (the cut)."""
    n = int(dur * sr)
    t = _t(n, sr)
    k = t / dur
    noise = _wide_noise(n, rng, 0.8)
    air = d.tv_filter(noise, sr, "bandpass", [(0, 500), (dur, 9000)], q=0.9)
    f = root * 2 ** (semis * k ** 1.5 / 12)
    tone = np.zeros((n, 2))
    for i, cents in enumerate((-14, 0, 11)):
        ff = f * 2 ** (cents / 1200)
        ph = np.cumsum(ff) / sr + rng.random()
        v = d.saw(ph, ff / sr) + 0.5 * d.saw(2 * ph, 2 * ff / sr)
        tone += d.pan(v, (i - 1) * 0.7)
    tone = d.tv_filter(tone, sr, "lowpass", [(0, 600), (dur, 7000)])
    rate = 4 + 18 * k ** 2  # accelerating tremolo
    trem = 0.75 + 0.25 * np.sin(2 * np.pi * np.cumsum(rate) / sr)
    x = (0.9 * air + 0.35 * tone * trem[:, None]) * (k ** 2.4)[:, None]
    x[-int(0.008 * sr):] *= np.linspace(1, 0, int(0.008 * sr))[:, None]
    return d.fft_filter(d.fft_filter(x, sr, "lowpass", 11000, order=3), sr, "highpass", 120)


def pop(rng, sr=SR, f0=1100.0, f1=380.0, dur=0.11):
    n = int(dur * sr)
    t = _t(n, sr)
    body = _osc_sine(_glide(n, sr, f0, f1, tau=0.02), sr) * d.adsr(n, sr, 0.0008, 0, 1, dur - 0.001, 3.0)
    click = d.fft_filter(rng.standard_normal(n), sr, "highpass", 3000) * np.exp(-t * 900) * 0.4
    x = body + click
    return d.fft_filter(np.stack([x, np.roll(x, 12)], 1), sr, "highpass", 180)


def tick(rng, sr=SR, f0=2000.0, f1=1700.0):
    n = int(0.04 * sr)
    t = _t(n, sr)
    nz = d.fft_filter(rng.standard_normal(n), sr, "bandpass", 5000, q=0.8) * np.exp(-t * 600) * 0.5
    ph = np.cumsum(_glide(n, sr, f0, f1)) / sr
    tri = 2 * np.abs(2 * (ph % 1) - 1) - 1
    x = nz + tri * np.exp(-t * 140)
    return d.fft_filter(_st(x), sr, "highpass", 300)


def glitch(rng, sr=SR, dur=0.4, n_seg=11):
    n = int(dur * sr)
    out = np.zeros((n, 2))
    src_len = int(0.06 * sr)
    tt = _t(src_len, sr)
    pos = 0
    while pos < n:
        seg = int(rng.uniform(0.012, 0.05) * sr)
        kind = rng.integers(0, 4 if pos else 3)  # never open with a dropout
        if kind == 0:  # square buzz
            f = rng.uniform(80, 900)
            s = np.sign(np.sin(2 * np.pi * f * tt[:seg]))
        elif kind == 1:  # bit-crushed noise
            s = np.round(rng.standard_normal(seg) * 2) / 4
            s = np.repeat(s[:: int(rng.integers(4, 24))], int(rng.integers(4, 24)))[:seg]
            s = np.pad(s, (0, seg - len(s)))
        elif kind == 2:  # FM chirp
            f = rng.uniform(300, 3000)
            s = np.sin(2 * np.pi * f * tt[:seg] + 6 * np.sin(2 * np.pi * f * 0.37 * tt[:seg]))
        else:  # gap / dropout
            s = np.zeros(seg)
        reps = int(rng.integers(1, 4))  # buffer-repeat stutter
        s = np.tile(s, reps)
        s = s * d.adsr(len(s), sr, 0.0008, 0, 1, 0.002) * rng.uniform(0.4, 1.0)
        e = min(n, pos + len(s))
        p = rng.uniform(-0.8, 0.8)
        out[pos:e] += d.pan(s[: e - pos], p)
        pos = e
    out = d.fft_filter(out, sr, "lowpass", 11000)
    return d.fft_filter(out, sr, "highpass", 150)


def ding(rng, sr=SR, f=1318.5, dur=1.3):
    n = int(dur * sr)
    t = _t(n, sr)
    idx = 2.2 * np.exp(-t * 9) + 0.25
    car = np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * 3 * f * t))
    part = np.sin(2 * np.pi * 2.0 * f * t) * 0.18 * np.exp(-t * 6) + np.sin(2 * np.pi * 4.07 * f * t) * 0.06 * np.exp(-t * 9)
    x = (car * np.exp(-t * 3.2) + part) * d.adsr(n, sr, 0.001, 0, 1, 0.3)
    x = np.stack([x, np.sin(2 * np.pi * f * 1.003 * t + idx * np.sin(2 * np.pi * 3 * f * t)) * np.exp(-t * 3.2) * 0.85 + part], 1)
    x = _reverb(x * 0.6, sr, rng, 1.0, 0.25, hp=500)[:n]
    x *= d.adsr(n, sr, 0.0, 0, 1, 0.25)[:, None]
    return d.fft_filter(x, sr, "highpass", 300)


def shimmer(rng, sr=SR, dur=1.5, root=1046.5):
    n = int(dur * sr)
    t = _t(n, sr)
    out = np.zeros((n, 2))
    for i, r in enumerate((1, 1.5, 2, 2.5, 3, 4)):  # major-ish overtone cluster
        for side, det in ((0, -4), (1, 5)):
            f = root * r * 2 ** (det / 1200)
            trem = 0.6 + 0.4 * np.sin(2 * np.pi * (5 + i) * t + rng.random() * 6)
            out[:, side] += np.sin(2 * np.pi * f * t) * trem / (1 + i * 0.6)
    env = _env_peak(n, sr, 0.25, 1.5, 2.2)
    out *= env[:, None]
    return d.fft_filter(_reverb(out, sr, rng, 1.2, 0.3, hp=800)[:n] * d.adsr(n, sr, 0, 0, 1, 0.3)[:, None], sr, "highpass", 500)


def swish_text(rng, sr=SR, dur=0.09, f0=2500.0, f1=6500.0):
    n = int(dur * sr)
    noise = _wide_noise(n, rng, 0.5)
    x = d.tv_filter(noise, sr, "bandpass", [(0, f0), (dur, f1)], q=1.6, frame=256)
    env = _env_peak(n, sr, dur * 0.35, 1.5, 2.0)
    x = x * env[:, None]
    return d.fft_filter(d.fft_filter(x, sr, "lowpass", 10000, order=3), sr, "highpass", 1500)


def shutter(rng, sr=SR):
    n = int(0.26 * sr)
    out = np.zeros(n)

    def clk(at, gain, ring_f):
        m = int(0.08 * sr)
        t = _t(m, sr)
        nz = rng.standard_normal(m)
        c = d.fft_filter(nz, sr, "bandpass", 4200, q=1.0) * np.exp(-t * 350)
        ring = d.fft_filter(nz, sr, "bandpass", ring_f, q=9) * np.exp(-t * 60) * 0.8
        th = np.sin(2 * np.pi * 160 * t) * np.exp(-t * 80) * 0.5
        s = int(at * sr)
        out[s:s + m] += (c + ring + th)[: n - s] * gain

    clk(0.0, 1.0, 3100)
    clk(0.075, 0.75, 2600)
    whirr = d.fft_filter(rng.standard_normal(n), sr, "bandpass", 1800, q=3) * 0.08
    whirr *= d.adsr(n, sr, 0.09, 0, 1, 0.1) * (np.arange(n) > int(0.09 * sr))
    out += whirr
    return d.fft_filter(np.stack([out, np.roll(out, 20)], 1), sr, "highpass", 200)


def notification(rng, sr=SR, notes=(987.8, 1318.5), gap=0.11):
    dur = gap + 0.55
    n = int(dur * sr)
    out = np.zeros((n, 2))
    for i, f in enumerate(notes):
        m = n - int(i * gap * sr)
        t = _t(m, sr)
        # marimba-like partials (wood) + soft sine body
        x = sum(np.sin(2 * np.pi * f * r * t) * d.db(g) * np.exp(-t * 9 / rs)
                for r, g, rs in ((1, 0, 1), (3.93, -14, 0.45), (9.2, -26, 0.25)))
        x = x * d.adsr(m, sr, 0.001, 0, 1, 0.05)
        out[int(i * gap * sr):] += d.pan(x, -0.15 + 0.3 * i)
    out = _reverb(out * 0.7, sr, rng, 0.6, 0.18, hp=600)[:n]
    out *= d.adsr(n, sr, 0, 0, 1, 0.1)[:, None]
    return d.fft_filter(out, sr, "highpass", 300)


def heartbeat(rng, sr=SR, bpm=70, beats=2):
    period = 60 / bpm
    dur = period * beats + 0.1
    n = int(dur * sr)
    out = np.zeros(n)

    def thump(at, gain, f0):
        m = int(0.22 * sr)
        t = _t(m, sr)
        s = _osc_sine(_glide(m, sr, f0, f0 * 0.6, tau=0.05), sr)
        s = np.tanh(2.2 * s) * np.exp(-t * 22) * d.adsr(m, sr, 0.006, 0, 1, 0.05)
        a = int(at * sr)
        out[a:a + m] += s[: n - a] * gain

    for b in range(beats):
        thump(b * period, 1.0, 72)        # lub
        thump(b * period + 0.26, 0.7, 64)  # dub
    out = d.fft_filter(out, sr, "lowpass", 900)
    return d.fft_filter(_st(out), sr, "highpass", 28)


def typing(rng, sr=SR, keys=12):
    times = np.cumsum(rng.uniform(0.055, 0.14, keys))
    times -= times[0]
    dur = times[-1] + 0.08
    n = int(dur * sr)
    out = np.zeros((n, 2))
    m = int(0.05 * sr)
    t = _t(m, sr)
    for i, at in enumerate(times):
        nz = rng.standard_normal(m)
        click = d.fft_filter(nz, sr, "bandpass", rng.uniform(2800, 4200), q=1.5) * np.exp(-t * 500)
        body = d.fft_filter(nz, sr, "bandpass", rng.uniform(700, 1200), q=3) * np.exp(-t * 120) * 0.7
        bottom = np.sin(2 * np.pi * 180 * t) * np.exp(-t * 150) * 0.3
        k = (click + body + bottom) * rng.uniform(0.6, 1.0) * (1.4 if i == keys - 1 else 1.0)
        s = int(at * sr)
        out[s:s + m] += d.pan(k, rng.uniform(-0.3, 0.3))[: n - s]
    return d.fft_filter(out, sr, "highpass", 250)


def sparkle(rng, sr=SR, dur=1.3, count=28):
    n = int(dur * sr)
    out = np.zeros((n, 2))
    penta = np.array([0, 2, 4, 7, 9])
    m = int(0.25 * sr)
    t = _t(m, sr)
    for i in range(count):
        at = (i / count) ** 1.3 * 0.75 + rng.uniform(-0.02, 0.02)
        semi = 12 * rng.integers(0, 3) + penta[rng.integers(0, 5)]
        f = 1760 * 2 ** (semi / 12)
        s = np.sin(2 * np.pi * f * t) * np.exp(-t * rng.uniform(18, 35)) * d.adsr(m, sr, 0.001, 0, 1, 0.02)
        a = int(max(0, at) * sr)
        seg = d.pan(s, rng.uniform(-0.9, 0.9)) * rng.uniform(0.3, 1.0) * (1 - 0.5 * i / count)
        out[a:a + m] += seg[: n - a]
    tt = _t(n, sr)
    pad = np.sin(2 * np.pi * 880 * tt) * _env_peak(n, sr, 0.3, 1.5, 2.5) * 0.15
    out += _st(pad)
    out = _reverb(out, sr, rng, 1.1, 0.35, hp=900)[:n]
    out *= d.adsr(n, sr, 0, 0, 1, 0.25)[:, None]
    return d.fft_filter(out, sr, "highpass", 600)


def braam(rng, sr=SR, dur=2.8, root=65.41):
    n = int(dur * sr)
    t = _t(n, sr)
    out = np.zeros((n, 2))
    for ratio, g in ((1, 1.0), (2, 0.7), (1.5, 0.4), (0.5, 0.25)):
        for i, cents in enumerate((-12, 0, 9)):
            f = root * ratio * 2 ** (cents / 1200) * (1 + 0.004 * np.exp(-t * 3))
            ph = np.cumsum(f) / sr + rng.random()
            out += d.pan(d.saw(ph, f / sr) * g, (i - 1) * 0.6)
    out = np.tanh(out * 0.9)  # brassy growl
    out = d.tv_filter(out, sr, "lowpass", [(0, 300), (0.06, 3200), (1.0, 1400), (dur, 500)], q=0.9)
    env = d.adsr(n, sr, 0.015, 0.4, 0.55, dur * 0.65, 1.8)
    sub = _osc_sine(np.full(n, root), sr) * env * 0.35
    out = out * env[:, None] + _st(sub)
    hit = impact(rng, sr, dur=0.6, crack=0.6, tail=0.4)[: n]
    out[: len(hit)] += hit * 0.5
    out = _reverb(out, sr, rng, 2.0, 0.3, hp=200)[:n]
    out *= d.adsr(n, sr, 0, 0, 1, 0.4)[:, None]
    return d.fft_filter(out, sr, "highpass", 28)


def zap(rng, sr=SR, dur=0.35, f0=3200.0, f1=180.0):
    n = int(dur * sr)
    t = _t(n, sr)
    f = _glide(n, sr, f0, f1, tau=dur / 4)
    mod = np.sin(2 * np.pi * np.cumsum(f * 0.5) / sr)
    s = np.sin(d.sweep_phase(f, sr) + 3.0 * mod)
    ring = s * np.sign(np.sin(2 * np.pi * 63 * t))  # ring-mod grit
    crackle = d.fft_filter(rng.standard_normal(n), sr, "highpass", 3000) * (rng.random(n) < 0.02) * 2
    x = (0.7 * s + 0.3 * ring) * d.adsr(n, sr, 0.001, 0, 1, dur * 0.8, 2.0) + crackle * np.exp(-t * 12)
    st = np.stack([x, np.roll(x, int(0.004 * sr))], 1)
    st = d.fft_filter(st, sr, "lowpass", 6500, order=3)
    return d.fft_filter(st, sr, "highpass", 150)


# ---------------------------------------------------------------- library spec
# name -> list of variant recipes (callables taking rng, sr)
LIBRARY: dict[str, list] = {
    "whoosh": [lambda r, sr: whoosh(r, sr, 0.55, 0.25, 260, 2400, 650, 1.4, 1),
               lambda r, sr: whoosh(r, sr, 0.6, 0.26, 320, 2400, 800, 1.6, -1, 0.45),
               lambda r, sr: whoosh(r, sr, 0.5, 0.23, 220, 1900, 500, 1.1, 1, 0.7)],
    "whoosh_long": [lambda r, sr: whoosh(r, sr, 1.1, 0.55, 200, 2200, 400, 1.2, -1, 0.6),
                    lambda r, sr: whoosh(r, sr, 1.0, 0.45, 260, 2300, 500, 1.4, 1, 0.55)],
    "swoosh_reverse": [lambda r, sr: swoosh_reverse(r, sr, 0.8, 7000),
                       lambda r, sr: swoosh_reverse(r, sr, 0.6, 5000)],
    "impact": [lambda r, sr: impact(r, sr, 1.5, (62, 34), (150, 48), 1.0, 1.3),
               lambda r, sr: impact(r, sr, 1.4, (55, 31), (180, 55), 0.7, 1.1)],
    "sub_drop": [lambda r, sr: sub_drop(r, sr, 1.2, 95, 30),
                 lambda r, sr: sub_drop(r, sr, 1.0, 120, 36)],
    "riser": [lambda r, sr: riser(r, sr, 1.5, 19, 110)],
    "riser_long": [lambda r, sr: riser(r, sr, 3.0, 24, 98)],
    "pop": [lambda r, sr: pop(r, sr, 1100, 380), lambda r, sr: pop(r, sr, 900, 300, 0.12),
            lambda r, sr: pop(r, sr, 1350, 480, 0.1)],
    "tick": [lambda r, sr: tick(r, sr, 2000, 1700), lambda r, sr: tick(r, sr, 2400, 2100),
             lambda r, sr: tick(r, sr, 1700, 1450)],
    "glitch": [lambda r, sr: glitch(r, sr, 0.4), lambda r, sr: glitch(r, sr, 0.3),
               lambda r, sr: glitch(r, sr, 0.35)],
    "ding": [lambda r, sr: ding(r, sr, 1318.5, 1.2), lambda r, sr: ding(r, sr, 1568.0, 1.1)],
    "shimmer": [lambda r, sr: shimmer(r, sr, 1.4)],
    "swish_text": [lambda r, sr: swish_text(r, sr, 0.09, 3500, 8000), lambda r, sr: swish_text(r, sr, 0.08, 4000, 9000),
                   lambda r, sr: swish_text(r, sr, 0.1, 3000, 7000), lambda r, sr: swish_text(r, sr, 0.085, 3800, 8500)],
    "shutter": [lambda r, sr: shutter(r, sr)],
    "notification": [lambda r, sr: notification(r, sr, (987.8, 1318.5)),
                     lambda r, sr: notification(r, sr, (1174.7, 1568.0), 0.09)],
    "heartbeat": [lambda r, sr: heartbeat(r, sr, 70, 2)],
    "typing": [lambda r, sr: typing(r, sr, 12)],
    "sparkle": [lambda r, sr: sparkle(r, sr, 1.3, 28)],
    "braam": [lambda r, sr: braam(r, sr, 2.8, 65.41)],
    "zap": [lambda r, sr: zap(r, sr, 0.35, 2600, 160), lambda r, sr: zap(r, sr, 0.3, 2000, 220),
            lambda r, sr: zap(r, sr, 0.42, 3200, 120)],
}


# dB of transient limiting before peak normalisation (more body / perceived loudness for spiky hits)
PUNCH = {"impact": 7, "braam": 4, "heartbeat": 4, "typing": 8, "shutter": 5, "zap": 3, "glitch": 2, "pop": 2}


def finish(x: np.ndarray, sr: int, peak_db: float = PEAK_DB, tail_db: float = -60, fade_out: float = 0.015,
           limit_db: float = 0.0) -> np.ndarray:
    """DC/rumble already removed by the recipes' high-pass; trim lead/tail, fade, peak-normalise."""
    x = _st(np.asarray(x, float))
    x = x - x.mean(0)
    if limit_db > 0:
        x = d.limit(x / np.abs(x).max(), sr, ceiling=d.db(-limit_db), release=0.04, lookahead=0.0015, block=16)
    a = np.abs(x).max(1)
    pk = a.max()
    if pk <= 0:
        return x
    # trim the tail only: the onset stays at t=0 so timed cues (whoosh peak at
    # 0.25 s, riser ending at its nominal length) line up with the pipeline
    e = np.nonzero(a > pk * d.db(tail_db))[0][-1] + 1
    x = x[:e]
    x = d.fade(x, sr, 0.0005, fade_out)
    return x * (d.db(peak_db) / np.abs(x).max())


def render(name: str, variant: int = 0, sr: int = SR, seed: int | None = None) -> np.ndarray:
    recipes = LIBRARY[name]
    rng = np.random.default_rng(seed if seed is not None else _seed(name, variant))
    return finish(recipes[variant % len(recipes)](rng, sr), sr, limit_db=PUNCH.get(name, 0.0))


def _seed(name: str, variant: int) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(name)) * 31 + variant


def file_names(name: str) -> list[str]:
    k = len(LIBRARY[name])
    return [f"{name}.wav"] if k == 1 else [f"{name}_{i + 1}.wav" for i in range(k)]


def render_library(folder: Path, sr: int = SR) -> list[Path]:
    import soundfile as sf
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for name in LIBRARY:
        for i, fn in enumerate(file_names(name)):
            p = folder / fn
            sf.write(p, render(name, i, sr).astype(np.float32), sr, subtype="PCM_16")
            out.append(p)
    return out


if __name__ == "__main__":  # pragma: no cover
    from .config import ROOT
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "assets" / "sfx"
    for p in render_library(target):
        print(p, p.stat().st_size)
