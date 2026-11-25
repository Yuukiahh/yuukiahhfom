"""Music engine v2: procedural, original background music for short-form explainers.

A small "producer in numpy":
  * key / mode / chord progression per mood, diatonic voice-led chords
  * drum kit (kick, snare, clap, closed/open hats, toms, crash) with velocity,
    swing and humanised timing
  * bass, sidechain-pumped pads, melodic lead hook (seeded variations), arps
  * arrangement driven by a per-bar energy curve: energetic hook from t=0,
    build (snare roll + riser + filter sweep) into a drop at ~25 % (or at the
    first major marker), breakdown, final build, final drop and a resolving
    end hit on the tonic with a reverb tail
  * markers (scene cuts): reverse swells end exactly on them, drum fills and
    crashes land on them when they sit on the beat grid
  * send reverb (FFT convolution), ping-pong delay, glue compressor,
    loudness normalisation and a look-ahead limiter

Deterministic for a given seed. ~1-2 s for 90 s of 44.1 kHz stereo on one core.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import audio_dsp as d

PC = {n: i for i, n in enumerate(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}
MODES = {"minor": [0, 2, 3, 5, 7, 8, 10], "major": [0, 2, 4, 5, 7, 9, 11], "dorian": [0, 2, 3, 5, 7, 9, 10]}

# Lead motifs: (eighth-note step within a 2-bar phrase, diatonic degree above the chord root, length in 8ths)
MOTIFS = {
    "arc": [(0, 4, 2), (2, 2, 1), (3, 4, 1), (4, 7, 2), (6, 6, 1), (7, 4, 1), (8, 4, 2), (10, 2, 2), (12, 1, 1),
            (13, 2, 1), (14, 0, 2)],
    "climb": [(0, 0, 1), (1, 2, 1), (2, 4, 1), (3, 7, 3), (6, 6, 1), (7, 4, 1), (8, 2, 3), (11, 4, 1), (12, 2, 2),
              (14, 1, 2)],
    "cine": [(0, 4, 3), (3, 2, 1), (4, 4, 4), (8, 7, 3), (11, 6, 1), (12, 4, 4)],
    "call": [(0, 7, 2), (2, 4, 1), (3, 2, 1), (4, 4, 3), (8, 7, 2), (10, 9, 1), (11, 7, 1), (12, 4, 4)],
    "bounce": [(0, 0, 1), (1, 4, 1), (2, 7, 1), (3, 4, 1), (5, 2, 1), (6, 4, 2), (8, 0, 1), (9, 4, 1), (10, 7, 1),
               (11, 9, 1), (12, 7, 2), (14, 4, 2)],
    "lazy": [(1, 4, 2), (3, 2, 1), (4, 1, 3), (9, 4, 1), (10, 6, 2), (12, 4, 3)],
    "riddle": [(0, 0, 1), (1, 1, 1), (2, 2, 2), (4, 4, 1), (5, 3, 1), (6, 2, 2), (8, 0, 1), (9, 1, 1), (10, 2, 1),
               (11, 4, 1), (12, 5, 3), (15, 4, 1)],
}

# 16-step drum grids per feel: kick, snare/clap, closed-hat velocities
FEELS = {
    "four": {"kick": [0, 4, 8, 12], "snare": [4, 12], "hat": [.35, .2, .8, .2] * 4, "open": [2, 6, 10, 14]},
    "half": {"kick": [0, 10], "snare": [8], "hat": [.7, .25, .5, .25] * 4, "open": [14]},
    "break": {"kick": [0, 7, 10], "snare": [4, 12], "hat": [.7, .3, .55, .3] * 4, "open": []},
    "bounce": {"kick": [0, 6, 8, 11], "snare": [4, 12], "hat": [.6, .3, .8, .3] * 4, "open": [14]},
    "epic": {"kick": [0, 3, 8, 11], "snare": [8], "hat": [.5, .0, .35, .0] * 4, "open": [], "toms": [12, 13, 14, 15]},
}

MOODS = {
    # documentary / space wonder
    "cosmic": dict(key="A", mode="minor", progs=[[0, 5, 2, 6], [0, 3, 5, 4]], bpm=90, feel="half", lead="bell",
                   pad="saw", bass="sub", arp="bell", motifs=["arc", "cine", "call"], sc=0.35, verb=(3.0, 0.34),
                   bright=0.55, clap=False),
    # lofi-ish mystery: dark plucks, ticking hats
    "mystery": dict(key="D", mode="minor", progs=[[0, 5, 3, 4], [0, 6, 5, 4]], bpm=84, feel="half", lead="pluck",
                    pad="dark", bass="sub", arp="pluck", motifs=["riddle", "cine", "lazy"], sc=0.28, verb=(2.6, 0.3),
                    bright=0.32, clap=False),
    # tech pulse: four-on-the-floor, offbeat saw bass, heavy pumping
    "pulse": dict(key="E", mode="minor", progs=[[0, 5, 2, 6], [0, 6, 5, 6]], bpm=122, feel="four", lead="pluck",
                  pad="saw", bass="saw", arp="pluck", motifs=["climb", "bounce", "arc"], sc=0.55, verb=(1.8, 0.24),
                  bright=0.75, clap=True),
    # trailer-ish epic: half-time, toms, brass lead, strings
    "epic": dict(key="C", mode="minor", progs=[[0, 5, 2, 6], [0, 5, 6, 4]], bpm=92, feel="epic", lead="brass",
                 pad="strings", bass="sub", arp="pluck", motifs=["cine", "call", "arc"], sc=0.3, verb=(3.2, 0.36),
                 bright=0.5, clap=False),
    "playful": dict(key="C", mode="major", progs=[[0, 5, 3, 4], [0, 3, 4, 3]], bpm=116, feel="bounce", lead="marimba",
                    pad="soft", bass="bounce", arp=None, motifs=["bounce", "climb", "call"], sc=0.3, verb=(1.4, 0.2),
                    bright=0.7, clap=True),
    "lofi": dict(key="F", mode="major", progs=[[1, 4, 0, 5], [3, 2, 1, 4]], bpm=80, feel="break", lead="bell",
                 pad="epiano", bass="round", arp=None, motifs=["lazy", "arc", "riddle"], sc=0.22, verb=(1.7, 0.22),
                 bright=0.3, clap=False, swing=0.6, sevenths=True, vinyl=True),
    "inspire": dict(key="G", mode="major", progs=[[0, 4, 5, 3], [5, 3, 0, 4]], bpm=104, feel="four", lead="piano",
                    pad="saw", bass="pulse8", arp="piano", motifs=["climb", "call", "arc"], sc=0.45, verb=(2.4, 0.3),
                    bright=0.65, clap=True, add9=True),
}


@dataclass
class Bar:
    t: float              # start time (s), may be negative for a pickup bar
    energy: float
    kind: str             # intro | build | drop | breakdown | end | tail
    chord: int            # index into the progression
    fill: bool = False
    roll: bool = False    # snare roll + riser into the next bar
    crash: bool = False


@dataclass
class Plan:
    bpm: float
    bar: float
    bars: list[Bar] = field(default_factory=list)
    drop: float = 0.0
    end_hit: float | None = None
    swells: list[float] = field(default_factory=list)   # reverse swells ending at these times
    hits: list[float] = field(default_factory=list)     # crash times off the grid


# ---------------------------------------------------------------- arrangement
def _energy_at(curve, t: float) -> float:
    if callable(curve):
        return float(curve(t))
    pts = sorted(curve)
    return float(np.interp(t, [p[0] for p in pts], [p[1] for p in pts]))


def plan(duration: float, bpm: float, prog_len: int, markers=None, energy_curve=None) -> Plan:
    D = duration
    markers = sorted(m for m in (markers or []) if 0 < m < D)
    bar = 240 / bpm
    origin = 0.0
    # the drop: first major marker in the 2nd..50th percent of the video, else ~25 %
    drop_m = next((m for m in markers if m >= max(2.0, 0.12 * D) and m <= 0.5 * D), None)
    if drop_m is not None:
        k = max(1, round(drop_m / bar))
        if abs(drop_m / k - bar) / bar <= 0.08:     # nudge the tempo so the drop lands on a downbeat from t=0
            bar = drop_m / k
            bpm = 240 / bar
        else:                                       # otherwise start with a pickup bar
            k = math.ceil(drop_m / bar)
            origin = drop_m - k * bar
        drop_t = drop_m
    else:
        k = max(1, round(0.25 * D / bar)) if D > 2.5 * bar else 1
        drop_t = k * bar
    drop_bar = k
    nb = max(1, math.ceil((D - origin) / bar))
    tail_need = max(1.2, min(2.6, 0.75 * bar))
    end_bar = None
    for b in range(nb - 1, drop_bar, -1):
        if origin + b * bar <= D - tail_need:
            end_bar = b
            break
    p = Plan(bpm=bpm, bar=bar, drop=drop_t)
    # default section map (bar indices)
    last = end_bar if end_bar is not None else nb
    span = last - drop_bar
    bd0 = bd1 = None
    if span >= 6:                         # room for a breakdown, a final build and a final drop
        bd0 = drop_bar + max(2, int(round(span * 0.45)))
        bd1 = min(last - 2, bd0 + max(2, int(round(span * 0.22))))  # bd1-1 becomes the build bar
        if bd1 - bd0 < 2:
            bd0 = bd1 - 2
    for b in range(nb):
        t = origin + b * bar
        if end_bar is not None and b == end_bar:
            kind, e = "end", 1.0
        elif end_bar is not None and b > end_bar:
            kind, e = "tail", 0.0
        elif b < drop_bar:
            kind, e = ("build" if b == drop_bar - 1 and drop_bar >= 2 else "intro"), 0.62
        elif bd0 is not None and bd0 <= b < bd1:
            kind, e = "breakdown", 0.3
        else:
            kind, e = "drop", 1.0
        chord = (b - drop_bar) % prog_len
        p.bars.append(Bar(t=t, energy=e, kind=kind, chord=chord))
    if end_bar is not None:
        p.bars[end_bar].chord = 0
        p.end_hit = origin + end_bar * bar
    if bd1 is not None and bd1 - 1 >= 0:
        p.bars[bd1 - 1].kind = "build"
        p.bars[bd1 - 1].energy = 0.55
    if energy_curve is not None:
        for bb in p.bars:
            if bb.kind not in ("end", "tail"):
                bb.energy = float(np.clip(_energy_at(energy_curve, bb.t + bar / 2), 0, 1))
                bb.kind = "drop" if bb.energy >= 0.8 else ("breakdown" if bb.energy < 0.4 else "intro")
    # builds: a bar followed by a big energy jump gets a roll; the jumped-into bar a crash
    for i in range(len(p.bars) - 1):
        a, b = p.bars[i], p.bars[i + 1]
        if b.kind in ("drop", "end") and b.energy - a.energy >= 0.3:
            a.roll = True
            a.kind = "build" if a.kind != "breakdown" else a.kind
            b.crash = True
        if p.bars[i].kind == "build":
            p.bars[i].roll = True
    if p.bars and p.bars[0].t >= 0:
        p.bars[0].crash = True
    # markers -> swells + fills / crashes
    s16 = bar / 16
    lastm = -9.0
    for m in markers:
        if m - lastm < 1.5 or m < 1.0 or (p.end_hit is not None and m > p.end_hit + 0.05):
            continue
        lastm = m
        p.swells.append(m)
        if abs(m - drop_t) < 1e-6:
            continue
        rel = (m - origin) / s16
        q = round(rel)
        if abs(rel - q) * s16 < 0.09:           # on the grid -> fill into it
            bi, step = divmod(q, 16)
            if step == 0 and 0 < bi < len(p.bars):
                p.bars[bi - 1].fill = True
                p.bars[bi].crash = True
            else:
                p.hits.append(origin + q * s16)
        else:
            p.hits.append(m)
    return p


# ---------------------------------------------------------------- engine
class Engine:
    def __init__(self, sr: int, rng: np.random.Generator, n: int):
        self.sr, self.rng, self.n = sr, rng, n
        z = lambda: np.zeros((n, 2), np.float32)  # noqa: E731
        self.drums, self.bass, self.music, self.fx = z(), z(), z(), z()
        self.verb_send, self.delay_send = z(), z()
        self.cache: dict = {}
        self.kicks: list[float] = []

    # -- placement
    def put(self, bus: np.ndarray, x: np.ndarray, t: float, gain: float = 1.0, send: float = 0.0,
            dly: float = 0.0):
        s = int(round(t * self.sr))
        if x.ndim == 1:
            x = np.stack([x, x], 1)
        a, b = max(0, s), min(self.n, s + len(x))
        if b <= a:
            return
        seg = x[a - s:b - s] * gain
        bus[a:b] += seg
        if send:
            self.verb_send[a:b] += seg * send
        if dly:
            self.delay_send[a:b] += seg * dly

    def cached(self, key, fn):
        v = self.cache.get(key)
        if v is None:
            v = self.cache[key] = np.asarray(fn(), np.float32)
        return v

    def panned(self, key, fn, p: float):
        """Cached equal-power panned stereo copy of a cached mono sound."""
        return self.cached(("pan", key, round(p, 3)), lambda: d.pan(fn(), p))

    # -- drums (one-shots, rendered once per call)
    def kick(self, style: str):
        def mk():
            sr = self.sr
            L = 0.5 if style == "boom" else 0.38
            m = int(L * sr)
            t = np.arange(m) / sr
            f = 48 + (150 if style != "boom" else 110) * np.exp(-t * (32 if style != "boom" else 22))
            body = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t * (6.5 if style == "boom" else 9))
            body = np.tanh(1.6 * body)
            click = d.fft_filter(self.rng.standard_normal(m), sr, "bandpass", 3500, q=0.8) * np.exp(-t * 400) * 0.35
            x = (body + click) * d.adsr(m, sr, 0.0005, 0, 1, 0.03)
            return x * 0.8
        return self.cached(("kick", style), mk)

    def snare(self, style: str):
        def mk():
            sr = self.sr
            m = int(0.32 * sr)
            t = np.arange(m) / sr
            nz = self.rng.standard_normal(m)
            if style == "clap":
                env = np.zeros(m)
                for k, o in enumerate((0.0, 0.009, 0.019)):
                    s = int(o * sr)
                    env[s:] += np.exp(-(t[: m - s]) * (160 if k < 2 else 22)) * (0.8 if k < 2 else 1)
                x = d.fft_filter(nz, sr, "bandpass", 1500, q=0.9) * env
                x += d.fft_filter(nz, sr, "highpass", 5000) * np.exp(-t * 30) * 0.25
            else:
                tone = np.sin(2 * np.pi * (185 + 40 * np.exp(-t * 60)) * t) * np.exp(-t * 22) * 0.8
                noise = d.fft_filter(nz, sr, "bandpass", 3200, q=0.6) * np.exp(-t * (16 if style == "lofi" else 20))
                x = tone + noise * 0.9
                if style == "lofi":
                    x = d.fft_filter(x, sr, "lowpass", 4500)
            x = x * d.adsr(m, sr, 0.0005, 0, 1, 0.05)
            return x / np.abs(x).max() * 1.0
        return self.cached(("snare", style), mk)

    def hat(self, open_: bool, lofi: bool):
        def mk():
            sr = self.sr
            m = int((0.35 if open_ else 0.07) * sr)
            t = np.arange(m) / sr
            nz = self.rng.standard_normal(m)
            # metallic: square-wave cluster + noise, high-passed
            metal = sum(np.sign(np.sin(2 * np.pi * f * t)) for f in (317, 451, 563, 789, 1031, 1279)) / 6
            x = d.fft_filter(0.6 * nz + 0.5 * metal, sr, "highpass", 7000 if not lofi else 5500, order=3)
            x = d.fft_filter(x, sr, "lowpass", 13000 if not lofi else 8000)
            x *= np.exp(-t * (9 if open_ else 70)) * d.adsr(m, sr, 0.0005, 0, 1, 0.01)
            return x / np.abs(x).max() * 0.7
        return self.cached(("hat", open_, lofi), mk)

    def crash(self):
        def mk():
            sr = self.sr
            m = int(2.2 * sr)
            t = np.arange(m) / sr
            x = d.fft_filter(np.stack([self.rng.standard_normal(m), self.rng.standard_normal(m)], 1), sr,
                             "highpass", 4000, order=2)
            x = d.fft_filter(x, sr, "lowpass", 12000)
            x *= (np.exp(-t * 2.2) * d.adsr(m, sr, 0.001, 0, 1, 0.3))[:, None]
            return x / np.abs(x).max() * 0.5
        return self.cached("crash", mk)

    def tom(self, f0: float):
        def mk():
            sr = self.sr
            m = int(0.5 * sr)
            t = np.arange(m) / sr
            f = f0 * (1 + 0.5 * np.exp(-t * 25))
            x = np.tanh(1.5 * np.sin(2 * np.pi * np.cumsum(f) / sr)) * np.exp(-t * 7)
            x += d.fft_filter(self.rng.standard_normal(m), sr, "bandpass", 900, q=1) * np.exp(-t * 40) * 0.3
            return x * d.adsr(m, sr, 0.001, 0, 1, 0.05) * 0.7
        return self.cached(("tom", f0), mk)

    def swell(self, length: float):
        """Reverse-cymbal swell, loudest at its end."""
        def mk():
            sr = self.sr
            m = int(length * sr)
            t = np.arange(m) / sr
            x = np.stack([self.rng.standard_normal(m), self.rng.standard_normal(m)], 1)
            x = d.tv_filter(x, sr, "bandpass", [(0, 1500), (length, 9000)], q=0.7)
            x *= ((t / length) ** 3)[:, None]
            x[-int(0.006 * sr):] *= np.linspace(1, 0, int(0.006 * sr))[:, None]
            return x / np.abs(x).max() * 0.25
        return self.cached(("swell", round(length, 3)), mk)

    def riser(self, length: float):
        def mk():
            sr = self.sr
            m = int(length * sr)
            t = np.arange(m) / sr
            x = np.stack([self.rng.standard_normal(m), self.rng.standard_normal(m)], 1)
            x = d.tv_filter(x, sr, "bandpass", [(0, 400), (length, 7000)], q=1.2)
            x *= ((t / length) ** 2)[:, None]
            x[-int(0.01 * sr):] *= np.linspace(1, 0, int(0.01 * sr))[:, None]
            return x / np.abs(x).max() * 0.2
        return self.cached(("riser", round(length, 3)), mk)

    def subdrop(self):
        def mk():
            sr = self.sr
            m = int(1.1 * sr)
            t = np.arange(m) / sr
            f = 30 + 60 * np.exp(-t * 3)
            x = np.sin(2 * np.pi * np.cumsum(f) / sr)
            x = (x + 0.3 * np.tanh(3 * x)) * np.exp(-t * 2.5) * d.adsr(m, sr, 0.003, 0, 1, 0.3)
            return x * 0.6
        return self.cached("subdrop", mk)

    # -- tonal instruments (cached per pitch/length)
    def tone(self, kind: str, midi: float, length: float, vel: float = 1.0):
        key = (kind, round(midi, 2), round(length, 3))
        return self.cached(key, lambda: self._tone(kind, midi, length)) * vel

    def _tone(self, kind: str, midi: float, length: float) -> np.ndarray:
        sr = self.sr
        f = 440 * 2 ** ((midi - 69) / 12)
        rel = {"bell": 0.9, "pluck": 0.25, "marimba": 0.35, "brass": 0.35, "piano": 0.8, "epiano": 0.5}.get(kind, 0.3)
        m = int((length + rel) * sr)
        t = np.arange(m) / sr
        ph0 = self.rng.random() * 2 * np.pi
        if kind == "bell":
            idx = 1.6 * np.exp(-t * 7) + 0.2
            x = np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * 2 * f * t) + ph0) * np.exp(-t * 2.2)
            x += 0.25 * np.sin(2 * np.pi * 3.01 * f * t) * np.exp(-t * 5)
        elif kind == "pluck":
            ph = f * t + self.rng.random()
            sw = d.saw(ph, f / sr) + d.saw(ph * 1.006, f * 1.006 / sr)
            bright = d.fft_filter(sw, sr, "lowpass", min(6000, f * 6), order=2) * np.exp(-t * 14)
            body = d.fft_filter(sw, sr, "lowpass", min(2000, f * 2.2), order=2) * np.exp(-t * 4.5)
            x = 0.5 * bright + 0.5 * body + 0.4 * np.sin(2 * np.pi * f * t) * np.exp(-t * 5)
        elif kind == "marimba":
            x = sum(np.sin(2 * np.pi * f * r * t + ph0) * g * np.exp(-t * 7 / rs)
                    for r, g, rs in ((1, 1, 1), (3.93, 0.2, 0.45), (9.2, 0.05, 0.25)))
        elif kind == "brass":
            vib = 1 + 0.004 * np.sin(2 * np.pi * 5.2 * t) * np.clip((t - 0.25) * 3, 0, 1)
            ph = np.cumsum(f * vib) / sr
            x = sum(d.saw(ph * r + self.rng.random(), f * r / sr) for r in (1.0, 1.004, 0.996)) / 3
            x = d.fft_filter(x, sr, "lowpass", min(2200, f * 5), order=2)
            x = np.tanh(1.5 * x) * d.adsr(m, sr, 0.06, 0.25, 0.75, rel)
        elif kind == "piano":
            B = 0.0004
            x = sum(np.sin(2 * np.pi * f * k * np.sqrt(1 + B * k * k) * t + ph0 * k) / k ** 1.3
                    * np.exp(-t * (1.6 + 1.2 * k)) for k in range(1, 8))
            x += d.fft_filter(self.rng.standard_normal(m), sr, "bandpass", 2500, q=0.7) * np.exp(-t * 150) * 0.05
        elif kind == "epiano":
            idx = 1.2 * np.exp(-t * 5) + 0.15
            x = np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * f * t) + ph0) * np.exp(-t * 1.4)
            x *= 1 + 0.18 * np.sin(2 * np.pi * 4.5 * t)  # tremolo
        else:
            x = np.sin(2 * np.pi * f * t)
        # natural decays above; this only adds a 2 ms attack and a release that ends the note
        env = np.ones(m)
        env[: int(0.002 * sr)] = np.linspace(0, 1, int(0.002 * sr))
        hold_end = int(length * sr)
        env[hold_end:] = np.linspace(1, 0, m - hold_end) ** 2
        return x * env

    def pad(self, kind: str, midis: tuple, length: float, cut0: float, cut1: float) -> np.ndarray:
        """Chord pad: raw oscillators cached per chord, filtered copies cached per cutoff."""
        sr = self.sr
        m = int((length + 0.6) * sr)

        def raw():
            t = np.arange(m) / sr
            out = np.zeros((m, 2))
            det = (-9, 0, 8) if kind != "dark" else (-6, 6)
            for i, mi in enumerate(midis):
                f = 440 * 2 ** ((mi - 69) / 12)
                for j, c in enumerate(det):
                    ff = f * 2 ** (c / 1200)
                    ph = ff * t + self.rng.random()
                    v = d.saw(ph, ff / sr) if kind != "soft" else np.sin(2 * np.pi * ph) + 0.2 * np.sin(4 * np.pi * ph)
                    p = ((j / max(1, len(det) - 1)) - 0.5) * 1.4 * (0.6 + 0.4 * (i % 2))
                    out += d.pan(v, float(np.clip(p, -1, 1)))
            out /= len(det) * len(midis) ** 0.5
            att = {"strings": 0.25, "dark": 0.3, "soft": 0.05}.get(kind, 0.08)
            return out * d.adsr(m, sr, att, 0, 1, 0.55)[:, None]

        def mk():
            if kind == "epiano":
                out = np.zeros((m, 2))
                for i, mi in enumerate(midis):
                    v = self._tone("epiano", mi, length)[:m]
                    out[: len(v)] += d.pan(v, (i / max(1, len(midis) - 1) - 0.5) * 0.8)
                return out * 0.5
            r = self.cached(("padraw", kind, midis, round(length, 3)), raw)
            if cut0 == cut1:
                return d.fft_chain(r, sr, [("lowpass", cut0), ("highpass", 140)])
            out = d.tv_filter(r, sr, "lowpass", [(0, cut0), (length, cut1), (length + 1, cut1)])
            return d.fft_chain(out, sr, [("highpass", 140)])
        return self.cached(("pad", kind, midis, round(length, 3), int(cut0), int(cut1)), mk)


def _midi(key_pc: int, mode: list[int], degree: int, base: int) -> int:
    o, dgr = divmod(degree, 7)
    return base + key_pc + mode[dgr] + 12 * o


def _place_range(m: int, lo: int) -> int:
    return lo + (m - lo) % 12


def _duck_curve(n: int, sr: int, times: list[float], depth: float, release: float) -> np.ndarray:
    """Sidechain gain curve: dips to 1-depth at each kick, recovers over `release`."""
    g = np.ones(n)
    L = int(release * sr)
    tt = np.arange(L) / sr
    shape = 1 - depth * np.clip(1 - tt / release, 0, 1) ** 2
    att = int(0.004 * sr)
    shape[:att] = np.minimum(shape[:att], 1 - depth * np.linspace(0, 1, att))
    for t in times:
        s = int(t * sr)
        if s >= n or s + L <= 0:
            continue
        a, b = max(0, s), min(n, s + L)
        g[a:b] = np.minimum(g[a:b], shape[a - s:b - s])
    return g


def render(duration: float, mood: str = "cosmic", sr: int = 44100, seed: int | None = None,
           energy_curve=None, markers=None, target_lufs: float = -14.0, ceiling: float = 0.79,
           stems: bool = False):
    """Returns (stereo float array of exactly int(duration*sr) samples, info dict)."""
    M = MOODS.get(mood, MOODS["cosmic"])
    rng = np.random.default_rng(seed)
    n_out = max(1, int(duration * sr))
    if duration < 0.05:
        return np.zeros((n_out, 2)), {"bpm": M["bpm"], "mood": mood}
    bpm = M["bpm"] * (1 + rng.uniform(-0.03, 0.03))
    prog = M["progs"][int(rng.integers(len(M["progs"])))]
    key_pc = (PC[M["key"]] + int(rng.integers(-2, 3))) % 12
    mode = MODES[M["mode"]]
    motif = list(MOTIFS[M["motifs"][int(rng.integers(len(M["motifs"])))]])
    # seeded motif variation: nudge 2 weak notes by a diatonic step
    for _ in range(2):
        i = int(rng.integers(len(motif)))
        st, dg, ln = motif[i]
        if st % 2 == 1:
            motif[i] = (st, dg + int(rng.choice([-1, 1])), ln)

    P = plan(duration, bpm, len(prog), markers, energy_curve)
    bar, beat = P.bar, P.bar / 4
    s16 = bar / 16
    tail_extra = 1.0
    n = int((duration + tail_extra) * sr)
    E = Engine(sr, rng, n)
    feel = FEELS[M["feel"]]
    swing = M.get("swing", 0.0)
    lofi = mood == "lofi"
    kstyle = "boom" if mood in ("epic", "cosmic", "lofi", "mystery") else "punch"
    sstyle = "clap" if M["clap"] else ("lofi" if lofi else "snare")

    def st_time(b: Bar, step: int) -> float:
        t = b.t + step * s16
        if step % 2 == 1 and swing:
            t += (swing - 0.5) * 2 * s16 * 0.5
        return t + rng.normal(0, 0.002)

    def chord_midis(ci: int) -> tuple:
        root_deg = prog[ci]
        degs = [root_deg, root_deg + 2, root_deg + 4]
        if M.get("sevenths"):
            degs.append(root_deg + 6)
        if M.get("add9"):
            degs.append(root_deg + 8)
        return tuple(sorted(_place_range(_midi(key_pc, mode, dg, 48), 52) for dg in degs))

    def root_midi(ci: int, lo: int = 33) -> int:
        return _place_range(_midi(key_pc, mode, prog[ci], 24), lo)

    tail_t = P.end_hit
    drop_idx = min(range(len(P.bars)), key=lambda i: abs(P.bars[i].t - P.drop))
    # ---------------------------------------------------------------- per bar
    for bi, b in enumerate(P.bars):
        e = b.energy
        if b.kind == "tail":
            continue
        ci = b.chord
        chord = chord_midis(ci)
        # ---- END HIT: resolve on the tonic and let it ring
        if b.kind == "end":
            E.put(E.drums, E.kick(kstyle), b.t, 1.0)
            E.put(E.drums, E.crash(), b.t, 1.0, send=0.25)
            E.kicks.append(b.t)
            ring = max(0.8, duration - b.t)
            cut = 900 + 2600 * M["bright"]
            E.put(E.music, E.pad(M["pad"], chord, ring, cut, cut * 0.5), b.t, 0.55, send=0.5)
            sub = E.cached(("endsub", root_midi(0)), lambda: _sub_note(sr, root_midi(0), min(ring, 2.0)))
            E.put(E.bass, sub, b.t, 0.9)
            top = _midi(key_pc, mode, prog[0] + 7, 60)
            E.put(E.music, d.pan(E.tone(M["lead"], top, min(ring, 1.2)), 0.0), b.t, 0.35, send=0.5, dly=0.2)
            continue
        # ---- drums
        drums_on = e >= 0.35
        if drums_on:
            ks = feel["kick"] if e >= 0.5 else feel["kick"][:1]
            if b.kind == "intro" and M["feel"] == "four" and e < 0.7:
                ks = [0, 8]
            for s in ks:
                if b.fill and s >= 12:
                    continue
                t = b.t + s * s16
                E.put(E.drums, E.kick(kstyle), t, 1.0 if s % 4 == 0 else 0.85)
                E.kicks.append(t)
            if e >= 0.7 and not b.roll:
                for s in feel["snare"]:
                    if b.fill and s >= 12:
                        continue
                    E.put(E.drums, E.snare(sstyle), st_time(b, s), 0.95, send=0.12)
            hat_scale = 0.75 if e < 0.7 else 1.0
            for s, v in enumerate(feel["hat"]):
                if v <= 0 or (b.fill and s >= 12):
                    continue
                if e < 0.7 and s % 2 == 1:
                    continue
                vel = v * hat_scale * rng.uniform(0.85, 1.1)
                E.put(E.drums, E.panned("hatc", lambda: E.hat(False, lofi), 0.25 if s % 2 else -0.1), st_time(b, s), vel)
            if e >= 0.8:
                for s in feel["open"]:
                    E.put(E.drums, E.panned("hato", lambda: E.hat(True, lofi), -0.3), st_time(b, s), 0.55)
            if "toms" in feel and e >= 0.8 and (bi % 2 == 1):
                for j, s in enumerate(feel["toms"]):
                    E.put(E.drums, d.pan(E.tom([110, 98, 82, 73][j]), -0.4 + 0.25 * j), b.t + s * s16, 0.8,
                          send=0.15)
        if b.fill and drums_on:  # 16th snare fill on the last beat, crescendo
            for j, s in enumerate(range(12, 16)):
                E.put(E.drums, E.snare(sstyle), b.t + s * s16, 0.45 + 0.15 * j, send=0.1)
            if "toms" in feel:
                E.put(E.drums, E.tom(82), b.t + 14 * s16, 0.8)
        if b.roll:  # snare roll: 8ths -> 16ths -> 32nds, rising
            pos = []
            pos += [b.t + k * 2 * s16 for k in range(4)]
            pos += [b.t + (8 + k) * s16 for k in range(4)]
            pos += [b.t + (12 + k * 0.5) * s16 for k in range(8)]
            for j, t in enumerate(pos):
                v = 0.25 + 0.75 * (j / len(pos)) ** 1.5
                E.put(E.drums, E.snare(sstyle), t, v * 0.8, send=0.1)
            E.put(E.fx, E.riser(bar), b.t, 1.0, send=0.2)
        if b.crash and b.t >= -0.01:
            E.put(E.drums, E.crash(), max(0.0, b.t), 0.9, send=0.25)
            if b.kind == "drop" and b.energy >= 0.8:
                E.put(E.bass, E.subdrop(), b.t, 0.7)
        # ---- bass
        if e >= 0.5 or (b.kind == "intro" and e >= 0.45):
            rm = root_midi(ci)
            bk = M["bass"]
            if bk == "saw":
                pat = [(s, 1.7) for s in (2, 6, 10, 14)] + ([(0, 1.0)] if e < 0.8 else [])
            elif bk == "bounce":
                pat = [(0, 2), (3, 1), (6, 2), (8, 2), (11, 1), (14, 2)]
            elif bk == "round":
                pat = [(0, 5), (7, 2), (10, 5)]
            elif bk == "pulse8":
                pat = [(s, 1.6) for s in range(0, 16, 2)]
            else:
                pat = [(0, 7), (10, 5)] if M["feel"] == "half" else [(0, 7), (8, 7)]
            for j, (s, ln) in enumerate(pat):
                if b.roll and s >= 8:
                    break
                mi = rm + (12 if bk == "bounce" and j in (2, 4) else 0) + (7 if bk == "bounce" and j == 5 else 0)
                key = ("bass", bk, mi, round(ln * s16, 3))
                note = E.cached(key, lambda mi=mi, ln=ln: _bass_note(sr, bk, mi, ln * s16, rng))
                E.put(E.bass, note, st_time(b, s), 0.9 if s % 4 == 0 else 0.75)
        # ---- pad / chords
        base_cut = round(500 + 3500 * M["bright"] * (0.35 + 0.65 * e), -2)
        c0, c1 = base_cut, base_cut
        if b.roll:
            c1 = min(9000, base_cut * 2.5)
        c0, c1 = round(c0, -2), round(c1, -2)
        if M["pad"] == "epiano":  # lofi comping: chord on 1 and the "and" of 2
            E.put(E.music, E.pad("epiano", chord, beat * 1.4, 0, 0), st_time(b, 0), 0.5, send=0.3)
            E.put(E.music, E.pad("epiano", chord, beat * 1.2, 0, 0), st_time(b, 6), 0.35, send=0.3)
            E.put(E.music, E.pad("dark", tuple(m - 12 for m in chord[:2]), bar, 700, 700), b.t, 0.25, send=0.3)
        else:
            lvl = 0.42 if e >= 0.5 else 0.55
            E.put(E.music, E.pad(M["pad"], chord, bar, c0, c1), b.t, lvl, send=0.35)
        # ---- arp
        if M["arp"] and e >= 0.75 and not b.roll:
            tones = list(chord) + [c + 12 for c in chord]
            order = tones + tones[-2:0:-1]
            for s in range(16):
                mi = order[s % len(order)] + 12
                E.put(E.music, E.panned(("arp", mi), lambda mi=mi: E.tone(M["arp"], mi, s16 * 0.9), 0.5 if s % 2 else -0.5),
                      st_time(b, s), 0.2 * (1.0 if s % 4 == 0 else 0.75), send=0.25, dly=0.35)
        # ---- lead hook
        lead_on = (b.kind in ("intro", "drop") and e >= 0.45) or (b.kind == "breakdown")
        if lead_on and not (b.roll and b.kind != "intro"):
            half = (bi - drop_idx) % 2  # 2-bar phrase aligned to the drop
            for st, dg, ln in motif:
                if st // 8 != half:
                    continue
                if b.kind == "breakdown" and st % 2 == 1:
                    continue
                s = (st % 8) * 2
                if b.roll and s >= 8:
                    continue
                mi = _midi(key_pc, mode, prog[ci] + dg, 60)
                mi = _place_range(mi, 62) if mi < 62 or mi > 86 else mi
                lvl = 0.5 if b.kind != "breakdown" else 0.4
                E.put(E.music, E.panned(("lead", mi, ln), lambda mi=mi, ln=ln: E.tone(M["lead"], mi, ln * 2 * s16 * 0.92), 0.05),
                      st_time(b, s), lvl,
                      send=0.3, dly=0.3)

    # swells & off-grid hits for markers
    for m in P.swells:
        L = min(1.2, max(0.4, m - 0.05))
        E.put(E.fx, E.swell(L), m - L, 0.8, send=0.2)
    for h in P.hits:
        E.put(E.drums, E.crash(), h, 0.6, send=0.2)
    if mood == "lofi" or M.get("vinyl"):
        _vinyl(E, duration)

    # ---------------------------------------------------------------- mix
    depth, rel = M["sc"], min(0.9 * beat, 0.32)
    duck = _duck_curve(n, sr, E.kicks, depth, rel).astype(np.float32)[:, None]
    duck_b = _duck_curve(n, sr, E.kicks, min(0.8, depth * 1.4), min(0.6 * beat, 0.2)).astype(np.float32)[:, None]
    dl = d.delay(E.delay_send, sr, beat * 0.75, 0.35, 4)
    E.verb_send += dl * 0.4
    ir = d.make_ir(sr, M["verb"][0], rng, predelay=0.025, damp_hz=4500)
    wet = d.convolve(E.verb_send, ir, hp=300, lp=9000, sr=sr) * (M["verb"][1] * 2.2)
    bass = E.bass
    # auto-balance drums and bass against the music bus (which also feeds reverb/delay)
    # to the mood's relative levels in dB (active-part RMS)
    bal = M.get("bal", (0.0, -4.0, -2.0))     # drums, bass, music
    music = (E.music + dl * 0.5) * duck
    bass = bass * duck_b
    r_m = _active_rms(music)

    def gain(stem, rel_db):
        r = _active_rms(stem)
        if r <= 0 or r_m <= 0:
            return 1.0
        return r_m * d.db(rel_db - bal[2]) / r

    mix = E.drums * gain(E.drums, bal[0]) + bass * gain(bass, bal[1]) + music + wet * (0.5 + 0.5 * duck) + E.fx * 0.8
    # master EQ in one FFT: rumble/DC high-pass, gentle low-shelf trim, "air" shelf (lofi: dusty top instead)
    eq = [("highpass", 28, 0.707, 2), ("lshelf", 120, M.get("low_db", -1.5))]
    eq += [("lowpass", 9000, 0.707, 1)] if lofi else [("hshelf", 4000, M.get("air_db", 3.0))]
    mix = d.fft_chain(mix, sr, eq)
    mix = mix[:n_out]
    mix -= mix.mean(0)
    # glue + loudness + limiter
    pk = np.abs(mix).max()
    if pk <= 0:
        return mix, {"bpm": P.bpm}
    mix *= 0.5 / pk
    mix = d.compress(mix, sr, threshold_db=-20, ratio=2.0, attack=0.015, release=0.2, knee_db=6)
    if n_out > int(1.0 * sr):
        loud = d.lufs(mix, sr, fast=True)
        mix *= d.db(np.clip(target_lufs + 0.2 - loud, -30, 30))
    else:
        mix *= 0.7 / max(1e-9, np.abs(mix).max())
    mix = d.limit(mix, sr, ceiling=ceiling, release=0.06, true_peak=True)
    # ending: the last stretch fades to silence (end hit rings into it)
    fo = min(1.0 if tail_t is None else max(0.3, duration - tail_t), duration * 0.3)
    nf = int(fo * sr)
    if nf > 1:
        mix[-nf:] *= (np.linspace(1, 0, nf) ** (2 if tail_t is not None else 1.5))[:, None]
    fi = min(int(0.008 * sr), len(mix))
    mix[:fi] *= np.linspace(0, 1, fi)[:, None]
    info = {"bpm": round(P.bpm, 2), "mood": mood, "key": list(PC)[key_pc], "mode": M["mode"], "prog": prog,
            "drop": round(P.drop, 3), "end_hit": None if P.end_hit is None else round(P.end_hit, 3),
            "sections": [(round(b.t, 3), b.kind, round(b.energy, 2)) for b in P.bars],
            "swells": P.swells, "seed": seed}
    if stems:
        info["stems"] = {"drums": E.drums[:n_out], "bass": bass[:n_out], "music": music[:n_out],
                         "reverb": wet[:n_out], "fx": E.fx[:n_out]}
    return mix, info


def _active_rms(x: np.ndarray) -> float:
    """RMS over the parts that actually play (100 ms blocks within 30 dB of the loudest)."""
    if not len(x) or not np.any(x):
        return 0.0
    e = d._block_env(x, 4410, "rms")
    if e.max() <= 0:
        return 0.0
    act = e[e > e.max() * 0.03]
    return float(np.sqrt((act ** 2).mean()))


def _sub_note(sr: int, midi: int, length: float) -> np.ndarray:
    f = 440 * 2 ** ((midi - 69) / 12)
    m = int(length * sr)
    t = np.arange(m) / sr
    x = np.sin(2 * np.pi * f * t)
    x = (x + 0.35 * np.tanh(2.5 * x)) * np.exp(-t * 1.2)
    return x * d.adsr(m, sr, 0.005, 0, 1, min(0.5, length * 0.6)) * 0.55


def _bass_note(sr: int, kind: str, midi: int, length: float, rng) -> np.ndarray:
    f = 440 * 2 ** ((midi - 69) / 12)
    rel = 0.06
    m = int((length + rel) * sr)
    t = np.arange(m) / sr
    if kind == "saw":
        ph = f * t + rng.random()
        x = d.fft_filter(d.saw(ph, f / sr) + d.saw(ph * 1.005, f / sr), sr, "lowpass", 900, order=2) * 0.45
        x += np.sin(2 * np.pi * f * t) * 0.6
        x *= d.adsr(m, sr, 0.003, 0.12, 0.8, rel)
    elif kind == "bounce":
        x = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t) * np.exp(-t * 10)
        x *= d.adsr(m, sr, 0.003, 0.15, 0.55, rel)
    elif kind == "round":
        fg = f * (1 - 0.03 * np.exp(-t * 30))
        x = np.sin(2 * np.pi * np.cumsum(fg) / sr)
        x = (x + 0.2 * np.tanh(3 * x)) * d.adsr(m, sr, 0.008, 0.3, 0.7, rel)
    elif kind == "pulse8":
        x = np.tanh(1.8 * np.sin(2 * np.pi * f * t)) * 0.8 + 0.2 * d.fft_filter(d.saw(f * t, f / sr), sr, "lowpass", 700)
        x *= d.adsr(m, sr, 0.003, 0.1, 0.7, rel)
    else:  # sub: sine + saturation harmonics so it translates to phones
        x = np.sin(2 * np.pi * f * t)
        x = (x + 0.35 * np.tanh(2.5 * x)) * d.adsr(m, sr, 0.006, 0.4, 0.8, rel + 0.04)
    return x * 0.6


def _vinyl(E: Engine, duration: float):
    sr, rng = E.sr, E.rng
    k = int(duration * 7)
    click = d.fft_filter(rng.standard_normal(int(0.004 * sr)), sr, "bandpass", 2500, q=0.7)
    for t in rng.uniform(0, duration, k):
        E.put(E.fx, d.pan(click * rng.uniform(0.01, 0.05), rng.uniform(-0.6, 0.6)), t)
    m = int(min(duration, 4.0) * sr)
    hiss = d.fft_filter(rng.standard_normal((m, 2)), sr, "bandpass", 4000, q=0.4) * 0.004
    reps = int(np.ceil(E.n / m))
    E.fx += np.tile(hiss, (reps, 1))[: E.n]
