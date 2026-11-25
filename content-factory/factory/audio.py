"""Background music and sound effects.

* Music: original procedural score (factory/music_gen.py) or, if present, a licensed
  track from assets/music/ or <data_dir>/music/ (see pick_library_track / fit_track).
* SFX: the committed, original synthesised library in assets/sfx/ (built by
  factory/sfx_design.py), with the small numpy synths below as fallback.

Everything synthesised here is generated from scratch -> no third-party copyright,
no Content-ID claims. Moods map to niches (space -> 'cosmic', tech -> 'pulse', ...).
"""
from __future__ import annotations

import json
import random
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from . import audio_dsp, music_gen, sfx_design
from .config import ROOT

SR = 44100
rng = np.random.default_rng()

NOTE = {n: i for i, n in enumerate(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}


def hz(note: str, octave: int) -> float:
    return 440.0 * 2 ** ((NOTE[note] - 9) / 12 + (octave - 4))


MOODS = music_gen.MOODS

NICHE_MOOD = {"space": "cosmic", "mystery": "mystery", "ai": "pulse", "technology": "pulse", "future": "epic",
              "facts": "playful", "psychology": "mystery", "gaming": "pulse", "history": "epic",
              "science": "inspire", "motivation": "inspire", "money": "pulse", "lifehacks": "lofi",
              "nature": "cosmic", "learning": "lofi"}


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


def music(duration: float, mood: str = "cosmic", out: Path | None = None, seed: int | None = None,
          energy_curve=None, markers: list[float] | None = None) -> np.ndarray:
    """Original procedural background score (music engine v2, see factory/music_gen.py).

    duration: seconds (output has exactly int(duration * SR) samples, stereo float, peak <= 0.79)
    mood:     cosmic | mystery | pulse | epic | playful | lofi | inspire (unknown -> cosmic)
    seed:     same seed -> identical audio; None -> a fresh variation every call
    energy_curve: optional [(t, energy 0..1), ...] or callable t -> energy, overrides the default arrangement
    markers:  optional key moments (s), e.g. scene cuts: the drop lands on the first major one,
              reverse swells / drum fills end exactly on them
    """
    global LAST_MUSIC_INFO
    mix, LAST_MUSIC_INFO = music_gen.render(duration, mood, SR, seed=seed, energy_curve=energy_curve, markers=markers)
    mix = mix.astype(np.float32)
    if out:
        sf.write(out, mix, SR)
    return mix


LAST_MUSIC_INFO: dict = {}


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


SFX_SR = sfx_design.SR
ASSETS_SFX = ROOT / "assets" / "sfx"
ASSETS_MUSIC = ROOT / "assets" / "music"

# fallback synths (used only when assets/sfx is missing a sound); arrays at SR
SFX = {"whoosh": sfx_whoosh, "impact": sfx_impact, "riser": sfx_riser, "pop": sfx_pop,
       "glitch": sfx_glitch, "ding": sfx_ding}
for _name in sfx_design.LIBRARY:
    SFX.setdefault(_name, (lambda n: (lambda: sfx_design.render(n, 0, SR)))(_name))

# alternative event names -> library names
ALIASES = {"camera_shutter": "shutter", "click": "tick", "transition_zap": "zap", "magic_sparkle": "sparkle",
           "deep_braam": "braam", "trailer_hit": "braam", "swoosh": "whoosh", "reverse": "swoosh_reverse",
           "riser_short": "riser", "riser_end": "zap", "boom": "impact", "bell": "ding"}

# default level trims (dB) applied on top of the event gain: the library is peak-normalised,
# this keeps a sensible loudness ladder (hits loudest, UI ticks and text swishes subtle)
LEVEL_DB = {"impact": 0.0, "braam": -1.0, "sub_drop": -1.0, "heartbeat": 0.0, "whoosh": -2.0, "whoosh_long": -2.0,
            "swoosh_reverse": -2.0, "riser": -2.0, "riser_long": -2.0, "zap": -3.0, "glitch": -4.0, "pop": -3.0,
            "ding": -4.0, "shimmer": -4.0, "sparkle": -3.0, "notification": -4.0, "shutter": -3.0, "typing": -3.0,
            "tick": -6.0, "swish_text": -3.0}


class SfxLibrary(dict):
    """name -> Path (first variant), plus .variants: name -> [Path, ...]."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.variants: dict[str, list[Path]] = {}


def _scan_sfx(folder: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[tuple[int, Path]]] = {}
    if not folder.is_dir():
        return {}
    for p in folder.glob("*.wav"):
        m = re.match(r"^(.+?)_(\d+)$", p.stem)
        name, idx = (m.group(1), int(m.group(2))) if m else (p.stem, 0)
        groups.setdefault(name, []).append((idx, p))
    return {k: [p for _, p in sorted(v)] for k, v in groups.items()}


def build_sfx_library(folder: Path, assets: Path | None = None) -> SfxLibrary:
    """Prefer the committed library in assets/sfx; synthesise anything missing into `folder`."""
    lib = SfxLibrary()
    for name, paths in _scan_sfx(ASSETS_SFX if assets is None else assets).items():
        lib[name], lib.variants[name] = paths[0], paths
    missing = [n for n in SFX if n not in lib]
    if missing:
        folder.mkdir(parents=True, exist_ok=True)
    for name in missing:
        p = folder / f"{name}.wav"
        if not p.exists():
            sf.write(p, np.asarray(SFX[name](), np.float32), SR)
        lib[name], lib.variants[name] = p, [p]
    return lib


_SFX_CACHE: dict[tuple[str, float], np.ndarray] = {}


def _load_sfx(path: Path) -> np.ndarray:
    key = (str(path), path.stat().st_mtime if path.exists() else 0.0)
    a = _SFX_CACHE.get(key)
    if a is None:
        a, sr = sf.read(path, dtype="float32", always_2d=True)
        if a.shape[1] == 1:
            a = np.repeat(a, 2, 1)
        a = a[:, :2]
        if sr != SFX_SR:
            a = audio_dsp.resample(a.astype(np.float64), sr, SFX_SR).astype(np.float32)
        _SFX_CACHE[key] = a
    return a


def sfx_track(events: list[tuple[float, str, float]], duration: float, lib: dict[str, Path], out: Path):
    """events: (time, name, gain) -> one stereo wav (48 kHz).

    Repeated events rotate through the variants of a sound (deterministic: the k-th
    occurrence of a name uses variant k mod n), so consecutive whooshes differ."""
    variants = getattr(lib, "variants", {}) or {}
    mix = np.zeros((int((duration + 2) * SFX_SR), 2), np.float32)
    count: dict[str, int] = {}
    for t, name, gain in sorted(events, key=lambda e: e[0]):
        name = name if name in lib else ALIASES.get(name, name)
        if name not in lib:
            continue
        paths = variants.get(name) or [lib[name]]
        k = count.get(name, 0)
        count[name] = k + 1
        a = _load_sfx(Path(paths[k % len(paths)]))
        g = gain * 10 ** (LEVEL_DB.get(name, 0.0) / 20)
        s = int(round(t * SFX_SR))
        if s < 0:  # event starts before 0: keep the part that falls inside the track
            a, s = a[-s:], 0
        e = min(len(mix), s + len(a))
        if e > s:
            mix[s:e] += a[: e - s] * g
    mix = mix[: int(duration * SFX_SR)]
    peak = float(np.abs(mix).max()) if len(mix) else 0.0
    if peak > 0.95:  # soft-limit instead of scaling everything down because of one overlap
        mix = audio_dsp.limit(mix, SFX_SR, ceiling=0.95, release=0.05).astype(np.float32)
    sf.write(out, mix, SFX_SR)
    return out


# ---------------------------------------------------------------- licensed music library
AUDIO_EXT = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".opus", ".aac"}
_BLOCKED_LICENSES = re.compile(r"(\bNC\b|non[- ]?commercial|all rights reserved)", re.I)


def music_dirs() -> list[Path]:
    dirs = [ASSETS_MUSIC]
    try:
        from .config import cfg
        dirs.append(cfg().path("general.data_dir") / "music")
    except Exception:  # pragma: no cover - config not available
        pass
    return dirs


def _probe_duration(p: Path) -> float:
    try:
        return float(sf.info(p).duration)
    except Exception:
        pass
    if shutil.which("ffprobe"):
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                           capture_output=True, text=True)
        try:
            return float(r.stdout.strip())
        except ValueError:
            return 0.0
    return 0.0


def library_tracks(dirs: list[Path] | None = None) -> list[tuple[Path, dict]]:
    """All audio files with a valid sidecar JSON {title, artist, license, source_url, moods}."""
    out = []
    for d in dirs or music_dirs():
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in AUDIO_EXT:
                continue
            side = p.with_suffix(".json")
            if not side.exists():
                continue
            try:
                meta = json.loads(side.read_text("utf-8"))
            except (OSError, ValueError):
                continue
            if not all(meta.get(k) for k in ("title", "artist", "license", "source_url")):
                continue
            if _BLOCKED_LICENSES.search(str(meta["license"])):
                continue  # non-commercial licences are not safe for monetised channels
            out.append((p, meta))
    return out


def pick_library_track(mood: str, duration: float, seed: int | None = None,
                       dirs: list[Path] | None = None) -> Path | None:
    """A licensed track whose sidecar lists `mood` (or "any"); prefers tracks long enough
    to cover `duration` without looping. None if the library is empty -> procedural score."""
    cands = []
    for p, meta in library_tracks(dirs):
        moods = [str(m).lower() for m in (meta.get("moods") or [])]
        if mood.lower() in moods or "any" in moods:
            cands.append(p)
    if not cands:
        return None
    long_enough = [p for p in cands if _probe_duration(p) >= duration]
    pool = long_enough or cands
    return random.Random(seed).choice(pool)


def fit_track(path: Path, duration: float, out: Path, fade_out: float = 2.0, crossfade: float = 2.0,
              lufs: float = -14.0) -> Path:
    """Trim (or loop with crossfades) a track to `duration`, fade in/out, loudness-normalise. Uses ffmpeg."""
    path, out = Path(path), Path(out)
    src = _probe_duration(path)
    if src <= 0:
        raise ValueError(f"cannot read duration of {path}")
    xf = max(0.05, min(crossfade, src / 3))
    reps = 1 if src >= duration else int(np.ceil((duration - xf) / (src - xf))) + 1
    fo = min(fade_out, duration / 3)
    cmd = ["ffmpeg", "-y", "-v", "error"]
    for _ in range(reps):
        cmd += ["-i", str(path)]
    fc = []
    if reps == 1:
        last = "[0:a]"
    else:
        prev = "[0:a]"
        for i in range(1, reps):
            lab = f"[x{i}]"
            fc.append(f"{prev}[{i}:a]acrossfade=d={xf:.3f}:c1=tri:c2=tri{lab}")
            prev = lab
        last = prev
    fc.append(f"{last}atrim=0:{duration:.3f},asetpts=N/SR/TB,afade=t=in:d=0.05,"
              f"afade=t=out:st={max(0.0, duration - fo):.3f}:d={fo:.3f},"
              f"loudnorm=I={lufs}:TP=-1.5:LRA=11,aresample=48000,aformat=channel_layouts=stereo[out]")
    cmd += ["-filter_complex", ";".join(fc), "-map", "[out]", "-c:a", "pcm_s16le", str(out)]
    subprocess.run(cmd, check=True)
    return out
