"""Custom transitions rendered in numpy/PIL between two scene clips.

Each transition takes the last T seconds of clip A and the first T seconds of clip B and
writes a transition clip. Effects are designed for short-form retention ("pattern interrupts"):
zoom_punch (radial blur punch-in + flash), whip (directional motion blur pan), glitch (RGB split
+ slice displacement), flash (bloom crossfade), spin (rotational zoom blur), luma (noise dissolve),
dip (quick dip through black with scale). Every effect has a matching SFX name.
"""
from __future__ import annotations

import math
import subprocess

import numpy as np
from PIL import Image

INTERMEDIATE = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p"]

SFX_FOR = {"zoom_punch": "impact", "whip_left": "whoosh", "whip_right": "whoosh", "whip_up": "whoosh",
           "glitch": "glitch", "flash": "shimmer", "spin": "whoosh", "luma": "sparkle", "dip": "sub_drop"}
SETS = {
    "cinematic": ["zoom_punch", "whip_left", "flash", "luma", "whip_up", "spin", "dip", "whip_right"],
    "energetic": ["zoom_punch", "whip_left", "glitch", "whip_right", "spin", "zoom_punch", "whip_up", "glitch"],
    "documentary": ["flash", "luma", "dip", "whip_left"],
}


def _read_frames(path: str, w: int, h: int, start: float | None = None, from_end: float | None = None,
                 dur: float = 0.4, fps: int = 60) -> list[np.ndarray]:
    cmd = ["ffmpeg", "-v", "error"]
    if from_end is not None:
        cmd += ["-sseof", f"-{from_end:.3f}"]
    elif start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", path, "-t", f"{dur:.3f}", "-r", str(fps), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = len(raw) // (w * h * 3)
    return [np.frombuffer(raw, np.uint8, w * h * 3, i * w * h * 3).reshape(h, w, 3) for i in range(n)]


def _ease(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def _zoom(img: Image.Image, z: float, angle: float = 0.0) -> Image.Image:
    w, h = img.size
    if angle:
        img = img.rotate(angle, resample=Image.BILINEAR)
    cw, ch = w / z, h / z
    box = ((w - cw) / 2, (h - ch) / 2, (w + cw) / 2, (h + ch) / 2)
    return img.resize((w, h), Image.BILINEAR, box=box)


def _radial(frame: np.ndarray, z: float, strength: float, angle: float = 0.0, ang_blur: float = 0.0) -> np.ndarray:
    img = Image.fromarray(frame)
    n = 1 + int(round(6 * min(1, strength)))
    acc = np.zeros(frame.shape, np.float32)
    for j in range(n):
        acc += np.asarray(_zoom(img, z * (1 + 0.018 * strength * j), angle + ang_blur * j), np.float32)
    return acc / n


def _box_blur(f: np.ndarray, length: int, axis: int) -> np.ndarray:
    if length < 2:
        return f.astype(np.float32)
    c = np.cumsum(np.pad(f.astype(np.float32), [(length, length) if a == axis else (0, 0) for a in range(3)], mode="edge"),
                  axis=axis)
    hi = np.take(c, np.arange(length + length // 2, length + length // 2 + f.shape[axis]), axis=axis)
    lo = np.take(c, np.arange(length // 2, length // 2 + f.shape[axis]), axis=axis)
    return (hi - lo) / length


def _shift_blur(frame: np.ndarray, dx: float, dy: float, blur: float) -> np.ndarray:
    moved = np.roll(frame, (int(dy), int(dx)), axis=(0, 1))
    return _box_blur(moved, int(220 * min(1.0, blur)), 1 if dx else 0)


def render(kind: str, a: list[np.ndarray], b: list[np.ndarray], seed: int = 0) -> list[np.ndarray]:
    n = min(len(a), len(b))
    h, w = a[0].shape[:2]
    rng = np.random.default_rng(seed)
    out = []
    mask = None
    if kind == "luma":
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        mask = np.sqrt(((xx - w / 2) / h) ** 2 + ((yy - h * 0.45) / h) ** 2)
        mask /= mask.max()
    for i in range(n):
        p = (i + 0.5) / n
        e = _ease(p)
        A, B = a[i], b[i]
        if kind == "zoom_punch":
            if p < 0.5:
                f = _radial(A, 1 + 0.6 * e * 2 * 0.5, strength=p * 2)
            else:
                f = _radial(B, 1 + 0.45 * (1 - p) * 2, strength=(1 - p) * 2)
            f = f + 255 * 0.55 * max(0, 1 - abs(p - 0.5) * 6) ** 2
        elif kind.startswith("whip"):
            axis = {"whip_left": (-1, 0), "whip_right": (1, 0), "whip_up": (0, -1)}[kind]
            span = w if axis[0] else h
            off = e * span
            speed = math.sin(p * math.pi)
            fa = _shift_blur(A, axis[0] * off, axis[1] * off, speed)
            fb = _shift_blur(B, axis[0] * (off - span), axis[1] * (off - span), speed)
            m = np.zeros((h, w, 1), np.float32)
            if axis[0]:
                edge = int(w - off) if axis[0] < 0 else int(off)
                if axis[0] < 0:
                    m[:, edge:] = 1
                else:
                    m[:, :edge] = 1
            else:
                edge = int(h - off)
                m[edge:, :] = 1
            f = fa * (1 - m) + fb * m
        elif kind == "glitch":
            src = (A if p < 0.5 else B).astype(np.float32)
            k = 1 - abs(p - 0.5) * 2
            f = src.copy()
            sh = int(30 * k)
            f[..., 0] = np.roll(src[..., 0], sh, axis=1)
            f[..., 2] = np.roll(src[..., 2], -sh, axis=1)
            for _ in range(int(14 * k)):
                y0 = int(rng.integers(0, h - 40))
                hh = int(rng.integers(8, 90))
                f[y0:y0 + hh] = np.roll(f[y0:y0 + hh], int(rng.integers(-120, 120) * k), axis=1)
            f[::4] *= 1 - 0.25 * k
        elif kind == "flash":
            f = A.astype(np.float32) * (1 - e) + B.astype(np.float32) * e
            f = f + 255 * 0.85 * max(0, 1 - abs(p - 0.5) * 3.5) ** 2
        elif kind == "spin":
            k = p * 2 if p < 0.5 else (1 - p) * 2
            ang = (-12 if p < 0.5 else 12) * k
            zz = math.cos(math.radians(abs(ang))) + (h / w) * math.sin(math.radians(abs(ang))) + 0.12 * k
            f = _radial(A if p < 0.5 else B, zz, strength=k, angle=ang, ang_blur=(-1.2 if p < 0.5 else 1.2) * k)
        elif kind == "luma":
            thr = e * 1.08
            m = np.clip((thr - mask) / 0.04, 0, 1)[..., None]
            ring = np.exp(-((mask - thr) / 0.012) ** 2)[..., None] * np.array([120, 200, 255], np.float32) * 1.4
            f = A.astype(np.float32) * (1 - m) + B.astype(np.float32) * m + ring * (0.2 < p < 0.95)
        else:  # dip
            if p < 0.5:
                f = np.asarray(_zoom(Image.fromarray(A), 1 + 0.15 * p), np.float32) * (1 - p * 2) ** 1.5
            else:
                f = np.asarray(_zoom(Image.fromarray(B), 1 + 0.15 * (1 - p)), np.float32) * ((p - 0.5) * 2) ** 0.8
        out.append(np.clip(f, 0, 255).astype(np.uint8))
    return out


def make(spec: dict) -> str:
    """spec: clip_a, clip_b, kind, dur, fps, w, h, out, seed"""
    w, h, fps, d = spec["w"], spec["h"], spec["fps"], spec["dur"]
    a = _read_frames(spec["clip_a"], w, h, from_end=d, dur=d, fps=fps)
    b = _read_frames(spec["clip_b"], w, h, start=0.0, dur=d, fps=fps)
    n = int(round(d * fps))
    a, b = (a + [a[-1]] * n)[:n], (b + [b[-1]] * n)[:n]
    frames = render(spec["kind"], a, b, spec.get("seed", 0))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", *INTERMEDIATE, spec["out"]]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("transition encode failed")
    return spec["out"]
