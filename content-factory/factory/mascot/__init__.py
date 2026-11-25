"""Mascot "Nova" runtime: packed Blender loops -> RGBA overlay track with lip-sync.

Assets: assets/mascot/<action>.webm (VP9 + alpha, 512x512, 60 fps), built from the Blender
renders with `python -m factory mascot build` (render) / `pack()` (encode).
Actions: intro, wave, point, wow, happy, think, idle_closed / idle_mid / idle_open (lip-sync set).
"""
from __future__ import annotations

import math
import re
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFilter

from ..config import ROOT, cfg

ASSET_DIR = ROOT / "assets" / "mascot"
ONE_SHOT = {"intro", "wave", "point", "wow", "happy"}
LOOP = {"think", "idle_closed", "idle_mid", "idle_open", "talk"}
ACTIONS = ["intro", "wave", "point", "wow", "happy", "think", "idle_closed", "idle_mid", "idle_open"]


def pack(frames_dir: Path, out_dir: Path = ASSET_DIR, crf: int = 30) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for d in sorted(p for p in Path(frames_dir).iterdir() if p.is_dir()):
        out = out_dir / f"{d.name}.webm"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", "60", "-i", str(d / "%04d.png"),
                        "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", str(crf),
                        "-row-mt", "1", "-auto-alt-ref", "0", str(out)], check=True)
        made.append(out)
    return made


def available() -> bool:
    return all((ASSET_DIR / f"{a}.webm").exists() for a in ("intro", "wave", "wow", "idle_closed"))


@lru_cache(maxsize=16)
def load(action: str, size: int) -> np.ndarray:
    path = ASSET_DIR / f"{action}.webm"
    p = subprocess.run(["ffmpeg", "-v", "error", "-c:v", "libvpx-vp9", "-i", str(path), "-vf", f"scale={size}:{size}",
                        "-f", "rawvideo", "-pix_fmt", "rgba", "-"], capture_output=True, check=True)
    n = len(p.stdout) // (size * size * 4)
    return np.frombuffer(p.stdout, np.uint8)[: n * size * size * 4].reshape(n, size, size, 4)


def plan_cues(script: dict, words: list[dict], spans: list[tuple[float, float]], total: float,
              stats: list[dict]) -> list[dict]:
    """Retention-oriented appearances: intro+wave on the hook, reactions on stats/questions, CTA at the end."""
    mode = cfg().get("mascot.mode", "moments")
    cues: list[dict] = [{"start": 0.0, "end": 1.15, "action": "intro"}, {"start": 1.15, "end": 2.6, "action": "wave"}]
    for st in stats:
        cues.append({"start": st["start"], "end": min(st["start"] + 1.6, st["end"] + 0.3), "action": "wow"})
    q = 0
    for i, sc in enumerate(script.get("scenes", [])):
        if sc.get("mascot") in ACTIONS + ["talk"]:
            s0 = spans[i][0] if i < len(spans) else 0
            cues.append({"start": s0, "end": s0 + 1.8, "action": sc["mascot"]})
        elif q < 2 and i > 0 and "?" in sc.get("narration", "") and not sc.get("stat"):
            s0 = spans[i][0]
            cues.append({"start": s0, "end": s0 + 2.0, "action": "think"})
            q += 1
    cues.append({"start": max(0, total - 2.2), "end": total, "action": "point"})
    cues.sort(key=lambda c: c["start"])
    merged: list[dict] = []
    for c in cues:
        if merged and c["start"] < merged[-1]["end"] + 0.35:
            if c["action"] in ("wow", "point") and merged[-1]["action"] not in ("intro",):
                merged[-1]["end"] = max(merged[-1]["start"] + 0.6, c["start"] - 0.05)
                merged.append(c)
            continue
        merged.append(c)
    if mode == "always":
        filled, t = [], 0.0
        for c in merged:
            if c["start"] - t > 0.6:
                filled.append({"start": t, "end": c["start"], "action": "talk"})
            filled.append(c)
            t = c["end"]
        if total - t > 0.6:
            filled.append({"start": t, "end": total, "action": "talk"})
        merged = filled
    return [c for c in merged if c["end"] - c["start"] > 0.3]


def _voice_env(voice_wav: Path, fps: int, n: int) -> np.ndarray:
    a, sr = sf.read(voice_wav, dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    hop = sr / fps
    env = np.array([np.sqrt(np.mean(seg ** 2) + 1e-9) if len(seg := a[int(i * hop): int(i * hop + hop)]) else 0.0
                    for i in range(n)])
    ref = np.percentile(env[env > 1e-3], 90) if (env > 1e-3).any() else 1
    env = np.clip(env / ref, 0, 1.2)
    return np.convolve(env, np.ones(3) / 3, mode="same")


def _shadow(size: int) -> np.ndarray:
    im = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(im)
    d.ellipse([size * 0.28, size * 0.86, size * 0.72, size * 0.93], fill=120)
    return np.asarray(im.filter(ImageFilter.GaussianBlur(size * 0.03)), np.float32) / 255


def render_track(cues: list[dict], voice_wav: Path, total: float, fps: int, out: Path, size: int = 440) -> Path | None:
    if not cues or not available():
        return None
    n = int(math.ceil(total * fps))
    env = _voice_env(voice_wav, fps, n)
    sh = _shadow(size)
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{size}x{size}", "-r", str(fps),
           "-i", "-", "-c:v", "qtrle", "-pix_fmt", "argb", str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    empty = np.zeros((size, size, 4), np.uint8).tobytes()
    ci = 0
    for i in range(n):
        t = i / fps
        while ci < len(cues) and t >= cues[ci]["end"]:
            ci += 1
        c = cues[ci] if ci < len(cues) and cues[ci]["start"] <= t < cues[ci]["end"] else None
        if not c:
            p.stdin.write(empty)
            continue
        act = c["action"]
        lt = t - c["start"]
        if act in ("talk", "idle_closed", "idle_mid", "idle_open"):
            lvl = env[i]
            act = "idle_open" if lvl > 0.55 else "idle_mid" if lvl > 0.22 else "idle_closed"
        frames = load(act, size)
        k = int(lt * fps)
        k = k % len(frames) if act in LOOP else min(k, len(frames) - 1)
        rgba = frames[k].astype(np.float32)
        # pop in / out (back-out ease)
        d_in, d_out = lt / 0.2, (c["end"] - t) / 0.16
        sc = 1.0
        if d_in < 1:
            x = d_in
            sc = 1 + 2.7 * (x - 1) ** 3 + 1.7 * (x - 1) ** 2
        elif d_out < 1:
            sc = max(0.0, d_out) ** 0.7
        if sc <= 0.02:
            p.stdin.write(empty)
            continue
        alpha = rgba[..., 3:4] / 255
        base = np.zeros_like(rgba)
        base[..., 3] = sh * 255 * 0.8
        comp = base.copy()
        comp[..., :3] = rgba[..., :3] * alpha
        comp[..., 3:4] = np.maximum(alpha * 255, base[..., 3:4] * (1 - alpha))
        comp[..., :3] = comp[..., :3]  # premultiplied colour over black shadow
        img = Image.fromarray(np.clip(comp, 0, 255).astype(np.uint8), "RGBA")
        if abs(sc - 1) > 1e-3:
            s = max(1, int(size * sc))
            small = img.resize((s, s), Image.BILINEAR)
            img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            img.paste(small, ((size - s) // 2, size - s - int(size * 0.02)))
        arr = np.asarray(img).copy()
        a = arr[..., 3:4].astype(np.float32) / 255
        arr[..., :3] = np.where(a > 0, np.clip(arr[..., :3] / np.maximum(a, 1e-3), 0, 255), 0).astype(np.uint8)
        p.stdin.write(arr.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("mascot track encode failed")
    return out
