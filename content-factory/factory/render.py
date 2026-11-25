"""Render engine.

1. Every scene is rendered to its own intermediate clip (parallel processes):
   procedural/image layer -> camera move (eased push/pan/orbit + impact shake)
   -> grade (contrast curve, vignette, film grain) -> ffmpeg pipe.
   Stock videos are graded/animated by ffmpeg filters instead.
2. Clips are joined with xfade transitions.
3. ASS subtitles, progress bar, audio (voice + ducked music + SFX, -14 LUFS).
GPU: h264_nvenc is used automatically when available (RTX 5070 Ti).
"""
from __future__ import annotations

import functools
import json
import math
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from .config import cfg

XFADE_SETS = {
    "cinematic": ["fade", "smoothup", "zoomin", "fadeblack", "circleopen", "smoothleft", "hblur", "dissolve"],
    "energetic": ["smoothleft", "zoomin", "slideup", "radial", "smoothright", "wipeup", "pixelize"],
    "documentary": ["fade", "dissolve", "fadeblack", "smoothup"],
}

CAMERA_MOVES = ["push_in", "pull_out", "pan_up", "pan_down", "pan_left", "pan_right", "orbit", "drift"]


# ------------------------------------------------------------------ encoder
@functools.lru_cache(maxsize=1)
def video_encoder() -> list[str]:
    pref = cfg().get("render.encoder", "auto")
    if pref in ("auto", "h264_nvenc"):
        try:
            p = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=black:s=256x256:d=0.2",
                                "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True, timeout=20)
            if p.returncode == 0:
                return ["-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", "19",
                        "-b:v", cfg().get("render.video_bitrate", "12M"), "-maxrate", "20M", "-bufsize", "30M",
                        "-profile:v", "high", "-spatial_aq", "1"]
        except Exception:
            pass
    return ["-c:v", "libx264", "-preset", "medium", "-crf", str(cfg().get("render.crf", 18)), "-profile:v", "high",
            "-maxrate", "20M", "-bufsize", "30M"]


INTERMEDIATE = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p"]


# ------------------------------------------------------------------ camera / grade
def _ease(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def camera(move: str, u: float, t: float, intensity: float = 1.0) -> tuple[float, float, float, float]:
    """returns (scale s [window size relative to output], dx, dy [fraction of free space -1..1], rot deg)."""
    e = _ease(u)
    k = intensity
    if move == "push_in":
        return 1.12 - 0.17 * e * k, 0, 0, 0
    if move == "pull_out":
        return 0.95 + 0.17 * e * k, 0, 0, 0
    if move == "pan_up":
        return 1.0, 0, (0.9 - 1.8 * e) * k, 0
    if move == "pan_down":
        return 1.0, 0, (-0.9 + 1.8 * e) * k, 0
    if move == "pan_left":
        return 1.0, (0.9 - 1.8 * e) * k, 0, 0
    if move == "pan_right":
        return 1.0, (-0.9 + 1.8 * e) * k, 0, 0
    if move == "orbit":
        return 1.04 - 0.06 * e, 0.3 * math.sin(u * math.pi), 0, (-2.0 + 4.0 * e) * k
    # drift
    return 1.06 - 0.05 * e, 0.4 * math.sin(t * 0.5), 0.3 * math.cos(t * 0.4), 0.6 * math.sin(t * 0.3)


def _contrast_lut(strength: float = 0.18) -> np.ndarray:
    x = np.linspace(0, 1, 256)
    s = x + strength * (x - 0.5) * (1 - np.abs(2 * x - 1)) * 2  # gentle S-curve
    s = np.clip(s, 0, 1) ** 0.98
    return (s * 255).astype(np.uint8)


def _vignette(w: int, h: int, amount: float = 0.42) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 * 0.9 + ((yy - h / 2) / (h / 2)) ** 2 * 0.75)
    v = 1 - amount * np.clip(d - 0.35, 0, 1) ** 1.6
    return (v * 256).astype(np.uint16)[..., None]


def render_scene_clip(spec: dict) -> str:
    from .visuals.media import MediaImage
    from .visuals.procedural import make_scene

    w, h, fps = spec["w"], spec["h"], spec["fps"]
    dur = spec["duration"]
    out = spec["out"]
    media = spec.get("media")
    if media and media.get("type") == "video":
        return _render_video_media(spec)
    if media and media.get("type") == "image":
        scene = MediaImage(w, h, spec["seed"], path=media["path"])
    else:
        scene = make_scene(spec["kind"], w, h, spec["seed"], spec.get("params"))
    lut = _contrast_lut(spec.get("contrast", 0.18))
    vig = _vignette(w, h, spec.get("vignette", 0.42))
    rng = np.random.default_rng(spec["seed"])
    grains = [rng.normal(0, spec.get("grain", 5.0), (h // 2, w // 2)).astype(np.int16) for _ in range(6)]
    grains = [np.kron(g, np.ones((2, 2), np.int16))[:h, :w, None] for g in grains]
    static = None if scene.dynamic else np.clip(scene.render(0), 0, 255).astype(np.uint8)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", *INTERMEDIATE, out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = int(round(dur * fps))
    move = spec.get("camera", "push_in")
    shakes = spec.get("shakes", [])
    t0 = spec.get("time_offset", 0.0)
    W, H = scene.W, scene.H
    for i in range(n):
        t = i / fps
        frame = static if static is not None else np.clip(scene.render(t + t0), 0, 255).astype(np.uint8)
        s, fx, fy, rot = camera(move, t / max(dur, 1e-3), t, spec.get("intensity", 1.0))
        win_w, win_h = w * s, h * s
        free_x, free_y = max(0.0, (W - win_w) / 2), max(0.0, (H - win_h) / 2)
        cx, cy = W / 2 + fx * free_x, H / 2 + fy * free_y
        for st in shakes:
            dt = t - st
            if 0 <= dt < 0.6:
                amp = 14 * math.exp(-dt * 7)
                cx += amp * math.sin(dt * 71)
                cy += amp * math.cos(dt * 53)
        a = math.radians(rot)
        ca, sa = math.cos(a) * s, math.sin(a) * s
        # output pixel (x,y) -> input = c + R*s*(x - w/2, y - h/2)
        data = (ca, -sa, cx - ca * w / 2 + sa * h / 2, sa, ca, cy - sa * w / 2 - ca * h / 2)
        img = Image.fromarray(frame).transform((w, h), Image.AFFINE, data, resample=Image.BILINEAR)
        arr = lut[np.asarray(img)]
        arr = ((arr.astype(np.uint16) * vig) >> 8).astype(np.int16) + grains[i % 6]
        # fade-in from black on the very first scene
        if spec.get("fade_in") and t < 0.35:
            arr = (arr * (t / 0.35)).astype(np.int16)
        proc.stdin.write(np.clip(arr, 0, 255).astype(np.uint8).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"scene render failed: {out}")
    return out


def _render_video_media(spec: dict) -> str:
    w, h, fps, dur = spec["w"], spec["h"], spec["fps"], spec["duration"]
    pan = spec.get("camera", "push_in")
    sw, sh = int(w * 1.12) // 2 * 2, int(h * 1.12) // 2 * 2
    if pan in ("pan_left", "pan_right"):
        xexpr = f"(iw-ow)*({'1-' if pan == 'pan_left' else ''}t/{dur})"
    else:
        xexpr = "(iw-ow)/2"
    yexpr = "(ih-oh)/2" if pan not in ("pan_up", "pan_down") else f"(ih-oh)*({'1-' if pan == 'pan_up' else ''}t/{dur})"
    vf = (f"scale={sw}:{sh}:force_original_aspect_ratio=increase,crop={sw}:{sh},"
          f"crop={w}:{h}:'{xexpr}':'{yexpr}',fps={fps},"
          "eq=contrast=1.08:saturation=1.12:brightness=-0.02,vignette=PI/5,noise=alls=6:allf=t,format=yuv420p")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-stream_loop", "-1", "-i", spec["media"]["path"], "-t", f"{dur:.3f}",
           "-an", "-vf", vf, *INTERMEDIATE, spec["out"]]
    subprocess.run(cmd, check=True)
    return spec["out"]


# ------------------------------------------------------------------ assembly
def ffprobe_duration(path: str | Path) -> float:
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
                       capture_output=True, text=True)
    return float(json.loads(p.stdout)["format"]["duration"])


def render_scenes(specs: list[dict], workers: int | None = None) -> list[str]:
    workers = workers or max(1, min(len(specs), (os.cpu_count() or 4) - 1))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(render_scene_clip, specs))


def assemble(clips: list[str], durations: list[float], transitions: list[str], trans_dur: float,
             voice_wav: Path, music_wav: Path, sfx_wav: Path, ass_file: Path, out: Path, total: float,
             progress_color: str = "#FFD60A") -> Path:
    w, h = cfg().get("render.width"), cfg().get("render.height")
    inputs: list[str] = []
    for c in clips:
        inputs += ["-i", c]
    vi = len(clips)
    inputs += ["-i", str(voice_wav), "-i", str(music_wav), "-i", str(sfx_wav)]
    fc = []
    last = "[0:v]"
    offset = 0.0
    for i in range(1, len(clips)):
        offset += durations[i - 1] - trans_dur
        lab = f"[x{i}]"
        fc.append(f"{last}[{i}:v]xfade=transition={transitions[(i - 1) % len(transitions)]}:duration={trans_dur}:offset={offset:.3f}{lab}")
        last = lab
    ass = str(ass_file).replace("\\", "/").replace(":", r"\:")
    pc = progress_color.lstrip("#")
    fc.append(f"{last}trim=0:{total:.3f},setpts=PTS-STARTPTS,ass='{ass}',"
              f"drawbox=x=0:y=ih-12:w=iw:h=12:color=black@0.35:t=fill[vsub]")
    fc.append(f"color=c=0x{pc}:s={w}x12:r={cfg().get('render.fps', 30)}:d={total:.3f}[bar]")
    fc.append(f"[vsub][bar]overlay=x='-W+W*t/{total:.3f}':y=H-12:eval=frame:shortest=1,format=yuv420p[vout]")
    mv = cfg().get("render.music_volume_db", -20)
    fc.append(f"[{vi}:a]aformat=sample_rates=48000:channel_layouts=stereo,asplit=2[voice][key]")
    fc.append(f"[{vi + 1}:a]aformat=sample_rates=48000:channel_layouts=stereo,volume={mv}dB[mus]")
    fc.append("[mus][key]sidechaincompress=threshold=0.03:ratio=6:attack=15:release=350[duck]")
    fc.append(f"[{vi + 2}:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=-6dB[sfx]")
    fc.append(f"[voice][duck][sfx]amix=inputs=3:normalize=0,atrim=0:{total:.3f},"
              f"loudnorm=I={cfg().get('render.loudness_lufs', -14)}:TP=-1.5:LRA=11,aresample=48000[aout]")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(fc),
           "-map", "[vout]", "-map", "[aout]", *video_encoder(), "-r", str(cfg().get("render.fps", 30)),
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", cfg().get("render.audio_bitrate", "192k"),
           "-movflags", "+faststart", str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg assemble failed: " + p.stderr[-2000:])
    return out


def thumbnail(video: Path, at: float, title: str, out: Path) -> Path:
    from PIL import ImageDraw, ImageFilter, ImageFont

    tmp = out.with_suffix(".raw.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{at:.2f}", "-i", str(video), "-frames:v", "1", str(tmp)], check=True)
    img = Image.open(tmp).convert("RGB")
    d = ImageDraw.Draw(img)
    font = _font(118)
    words = title.upper().split()
    lines, cur = [], ""
    for wd in words:
        if len(cur + " " + wd) > 14 and cur:
            lines.append(cur)
            cur = wd
        else:
            cur = (cur + " " + wd).strip()
    lines.append(cur)
    y = int(img.height * 0.12)
    shadow = Image.new("L", img.size, 0)
    sd = ImageDraw.Draw(shadow)
    for i, ln in enumerate(lines[:4]):
        tw = d.textlength(ln, font=font)
        sd.text(((img.width - tw) / 2, y + i * 135), ln, font=font, fill=255)
    img.paste((0, 0, 0), mask=shadow.filter(ImageFilter.GaussianBlur(14)))
    for i, ln in enumerate(lines[:4]):
        tw = d.textlength(ln, font=font)
        d.text(((img.width - tw) / 2, y + i * 135), ln, font=font, fill=(255, 214, 10) if i == 0 else (255, 255, 255),
               stroke_width=6, stroke_fill=(0, 0, 0))
    img.save(out, quality=92)
    tmp.unlink(missing_ok=True)
    return out


def _font(size: int):
    from PIL import ImageFont

    for name in ["InterDisplay-Black.ttf", "Inter-Black.ttf", "C:/Windows/Fonts/seguibl.ttf", "C:/Windows/Fonts/arialbd.ttf",
                 "/usr/share/fonts/opentype/inter/InterDisplay-Black.otf", "/usr/share/fonts/opentype/inter/Inter-Black.otf",
                 "DejaVuSans-Bold.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()
