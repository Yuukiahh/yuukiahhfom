"""Render engine v2 (1080x1920 @ 60 fps).

1. Scenes render to intermediate clips in parallel processes:
   3D/procedural/image layer -> camera v2 (shot splits every ~2.5 s = pattern interrupts,
   eased moves, handheld micro-motion, zoom bumps on emphasised words, impact shakes)
   -> grade (S-curve, bloom on highlights, vignette, animated grain).
2. Custom transitions (factory.transitions) are rendered between neighbouring clips.
3. Final pass: trimmed bodies + transitions are concatenated, ASS captions burned in
   (bundled fonts), mascot overlay track, progress bar, audio mix (polished voice,
   side-chained music, SFX) at -14 LUFS. NVENC on the RTX 5070 Ti when available.
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
from PIL import Image, ImageFilter

from .config import cfg
from .transitions import SETS as TRANSITION_SETS

XFADE_SETS = TRANSITION_SETS  # backwards compatible name (dashboard presets)
CAMERA_MOVES = ["push_in", "pull_out", "pan_up", "pan_down", "pan_left", "pan_right", "orbit", "drift", "dolly_tilt"]


# ------------------------------------------------------------------ encoder
@functools.lru_cache(maxsize=1)
def video_encoder() -> list[str]:
    pref = cfg().get("render.encoder", "auto")
    if pref in ("auto", "h264_nvenc"):
        try:
            p = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=black:s=256x256:d=0.2",
                                "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True, timeout=20)
            if p.returncode == 0:
                return ["-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", "18",
                        "-b:v", cfg().get("render.video_bitrate", "16M"), "-maxrate", "28M", "-bufsize", "40M",
                        "-profile:v", "high", "-spatial_aq", "1", "-temporal_aq", "1"]
        except Exception:
            pass
    return ["-c:v", "libx264", "-preset", cfg().get("render.x264_preset", "medium"), "-crf", str(cfg().get("render.crf", 18)),
            "-profile:v", "high", "-maxrate", "28M", "-bufsize", "40M"]


INTERMEDIATE = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p"]


# ------------------------------------------------------------------ camera v2
def _ease(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def _move(move: str, u: float, t: float, k: float = 1.0) -> tuple[float, float, float, float]:
    """(window scale s, dx, dy in [-1,1] of free space, rotation deg)"""
    e = _ease(u)
    if move == "push_in":
        return 1.1 - 0.16 * e * k, 0, 0, 0
    if move == "pull_out":
        return 0.95 + 0.15 * e * k, 0, 0, 0
    if move == "pan_up":
        return 1.0, 0, (0.85 - 1.7 * e) * k, 0
    if move == "pan_down":
        return 1.0, 0, (-0.85 + 1.7 * e) * k, 0
    if move == "pan_left":
        return 1.0, (0.85 - 1.7 * e) * k, 0, 0
    if move == "pan_right":
        return 1.0, (-0.85 + 1.7 * e) * k, 0, 0
    if move == "orbit":
        return 1.04 - 0.07 * e, 0.35 * math.sin(u * math.pi), 0, (-2.5 + 5.0 * e) * k
    if move == "dolly_tilt":
        return 1.08 - 0.12 * e, 0, (0.5 - 1.0 * e) * k, (1.5 - 3 * e) * k
    return 1.06 - 0.05 * e, 0.4 * math.sin(t * 0.5), 0.3 * math.cos(t * 0.4), 0.6 * math.sin(t * 0.3)


def camera_at(spec: dict, t: float) -> tuple[float, float, float, float]:
    shots = spec.get("shots") or [(0.0, spec.get("camera", "push_in"))]
    dur = spec["duration"]
    idx = max(i for i, (s, _) in enumerate(shots) if s <= t + 1e-6)
    s0, move = shots[idx]
    s1 = shots[idx + 1][0] if idx + 1 < len(shots) else dur
    s, dx, dy, rot = _move(move, (t - s0) / max(0.1, s1 - s0), t, spec.get("intensity", 1.0))
    if idx % 2 == 1:  # punch-in on alternate shots (jump-cut feel)
        s *= 0.9
    # handheld micro motion
    hh = spec.get("handheld", 0.6)
    dx += hh * 0.06 * (math.sin(t * 1.3 + 1) + 0.5 * math.sin(t * 2.9))
    dy += hh * 0.05 * (math.sin(t * 1.1 + 2) + 0.5 * math.sin(t * 3.3 + 1))
    rot += hh * 0.35 * math.sin(t * 0.9 + 3)
    # zoom bumps on emphasis words
    for b in spec.get("bumps", []):
        d = t - b
        if 0 <= d < 0.35:
            s *= 1 - 0.045 * math.sin(min(1, d / 0.35) * math.pi) * (1 - d / 0.35)
    return s, dx, dy, rot


# ------------------------------------------------------------------ grade
def _contrast_lut(strength: float = 0.18) -> np.ndarray:
    x = np.linspace(0, 1, 256)
    s = x + strength * (x - 0.5) * (1 - np.abs(2 * x - 1)) * 2
    return (np.clip(s, 0, 1) ** 0.98 * 255).astype(np.uint8)


def _vignette(w: int, h: int, amount: float = 0.42) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 * 0.9 + ((yy - h / 2) / (h / 2)) ** 2 * 0.75)
    return ((1 - amount * np.clip(d - 0.35, 0, 1) ** 1.6) * 256).astype(np.uint16)[..., None]


def _bloom(arr: np.ndarray, strength: float = 0.2, thresh: int = 205) -> np.ndarray:
    h, w = arr.shape[:2]
    small = Image.fromarray(arr).resize((w // 6, h // 6), Image.BILINEAR)
    s = np.asarray(small, np.int16) - thresh
    s = np.clip(s, 0, 255).astype(np.uint8)
    b = Image.fromarray(s).filter(ImageFilter.GaussianBlur(6)).resize((w, h), Image.BILINEAR)
    return np.asarray(b, np.int16) * strength


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
    grains = [rng.normal(0, spec.get("grain", 4.0), (h // 2, w // 2)).astype(np.int16) for _ in range(8)]
    grains = [np.kron(g, np.ones((2, 2), np.int16))[:h, :w, None] for g in grains]
    static = None if scene.dynamic else np.clip(scene.render(0), 0, 255).astype(np.uint8)
    # auto exposure: lift shadows of very dark scenes (measured once -> no flicker)
    probe = static if static is not None else np.clip(scene.render(dur / 2 + spec.get("time_offset", 0.0)), 0, 255).astype(np.uint8)
    luma = float(probe[::8, ::8].mean())
    target = float(spec.get("target_luma", 34.0))
    if 1.0 < luma < target:
        gamma = max(0.55, min(1.0, math.log(target / 255) / math.log(luma / 255)))
        lift = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
        lut = lift[lut]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", *INTERMEDIATE, out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = int(round(dur * fps))
    t0 = spec.get("time_offset", 0.0)
    W, H = scene.W, scene.H
    bloom = spec.get("bloom", 0.18)
    for i in range(n):
        t = i / fps
        frame = static if static is not None else np.clip(scene.render(t + t0), 0, 255).astype(np.uint8)
        s, fx, fy, rot = camera_at(spec, t)
        win_w, win_h = w * s, h * s
        free_x, free_y = max(0.0, (W - win_w) / 2), max(0.0, (H - win_h) / 2)
        cx, cy = W / 2 + fx * free_x, H / 2 + fy * free_y
        for st in spec.get("shakes", []):
            d = t - st
            if 0 <= d < 0.6:
                amp = 16 * math.exp(-d * 7)
                cx += amp * math.sin(d * 71)
                cy += amp * math.cos(d * 53)
        a = math.radians(rot)
        ca, sa = math.cos(a) * s, math.sin(a) * s
        data = (ca, -sa, cx - ca * w / 2 + sa * h / 2, sa, ca, cy - sa * w / 2 - ca * h / 2)
        img = Image.fromarray(frame).transform((w, h), Image.AFFINE, data, resample=Image.BILINEAR)
        arr = lut[np.asarray(img)]
        out_f = ((arr.astype(np.uint16) * vig) >> 8).astype(np.int16) + grains[i % 8]
        if bloom:
            out_f = out_f + _bloom(arr, bloom)
        if spec.get("fade_in") and t < 0.18:
            out_f = (out_f * (t / 0.18)).astype(np.int16)
        proc.stdin.write(np.clip(out_f, 0, 255).astype(np.uint8).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"scene render failed: {out}")
    return out


def _render_video_media(spec: dict) -> str:
    w, h, fps, dur = spec["w"], spec["h"], spec["fps"], spec["duration"]
    pan = spec.get("camera", "push_in")
    sw, sh = int(w * 1.12) // 2 * 2, int(h * 1.12) // 2 * 2
    xexpr = f"(iw-ow)*({'1-' if pan == 'pan_left' else ''}t/{dur})" if pan in ("pan_left", "pan_right") else "(iw-ow)/2"
    yexpr = f"(ih-oh)*({'1-' if pan == 'pan_up' else ''}t/{dur})" if pan in ("pan_up", "pan_down") else "(ih-oh)/2"
    vf = (f"scale={sw}:{sh}:force_original_aspect_ratio=increase,crop={sw}:{sh},"
          f"crop={w}:{h}:'{xexpr}':'{yexpr}',fps={fps},"
          "eq=contrast=1.08:saturation=1.12:brightness=-0.02,vignette=PI/5,noise=alls=5:allf=t,format=yuv420p")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-stream_loop", "-1", "-i", spec["media"]["path"], "-t", f"{dur:.3f}",
                    "-an", "-vf", vf, *INTERMEDIATE, spec["out"]], check=True)
    return spec["out"]


# ------------------------------------------------------------------ assembly
def ffprobe_duration(path: str | Path) -> float:
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
                       capture_output=True, text=True)
    return float(json.loads(p.stdout)["format"]["duration"])


def _pool_size(n: int) -> int:
    return max(1, min(n, int(cfg().get("render.workers", 0) or (os.cpu_count() or 4) - 1)))


def render_scenes(specs: list[dict], workers: int | None = None) -> list[str]:
    with ProcessPoolExecutor(max_workers=workers or _pool_size(len(specs))) as ex:
        return list(ex.map(render_scene_clip, specs))


def render_transitions(tspecs: list[dict]) -> list[str]:
    from .transitions import make
    if not tspecs:
        return []
    with ProcessPoolExecutor(max_workers=_pool_size(len(tspecs))) as ex:
        return list(ex.map(make, tspecs))


VOICE_CHAIN = ("highpass=f=75,lowpass=f=15500,deesser=i=0.35,"
               "equalizer=f=180:t=q:w=1.2:g=1.5,equalizer=f=3200:t=q:w=1.4:g=2.5,equalizer=f=9000:t=q:w=1.5:g=1.5,"
               "acompressor=threshold=-20dB:ratio=3.2:attack=4:release=90:makeup=3,"
               "aecho=0.8:0.4:28:0.06")


def assemble(clips: list[str], durations: list[float], transitions: list[str], trans_dur: float,
             voice_wav: Path, music_wav: Path, sfx_wav: Path, ass_file: Path, out: Path, total: float,
             progress_color: str = "#FFD60A", trans_clips: list[str] | None = None,
             mascot_track: Path | None = None) -> Path:
    from .subtitles import ass_filter_arg

    w, h = cfg().get("render.width"), cfg().get("render.height")
    fps = cfg().get("render.fps", 60)
    inputs: list[str] = []
    for c in clips:
        inputs += ["-i", c]
    fc = []
    if trans_clips:
        # bodies: clip i without the parts consumed by neighbouring transitions
        for t in trans_clips:
            inputs += ["-i", t]
        segs = []
        nt = len(trans_clips)
        for i, d in enumerate(durations):
            a = trans_dur if i > 0 else 0.0
            b = d - (trans_dur if i < len(durations) - 1 else 0.0)
            fc.append(f"[{i}:v]trim=start={a:.4f}:end={b:.4f},setpts=PTS-STARTPTS,fps={fps},format=yuv420p[b{i}]")
            segs.append(f"[b{i}]")
            if i < nt:
                fc.append(f"[{len(clips) + i}:v]fps={fps},format=yuv420p[t{i}]")
                segs.append(f"[t{i}]")
        fc.append("".join(segs) + f"concat=n={len(segs)}:v=1:a=0[vcat]")
        last = "[vcat]"
    else:
        last, offset = "[0:v]", 0.0
        for i in range(1, len(clips)):
            offset += durations[i - 1] - trans_dur
            fc.append(f"{last}[{i}:v]xfade=transition=fade:duration={trans_dur}:offset={offset:.3f}[x{i}]")
            last = f"[x{i}]"
    vi = len(inputs) // 2
    inputs += ["-i", str(voice_wav), "-i", str(music_wav), "-i", str(sfx_wav)]
    mi = None
    if mascot_track:
        inputs += ["-i", str(mascot_track)]
        mi = vi + 3
    pc = progress_color.lstrip("#")
    fc.append(f"{last}trim=0:{total:.3f},setpts=PTS-STARTPTS,{ass_filter_arg(ass_file)}[vsub]")
    cur = "[vsub]"
    if mi is not None:
        mx = int(w * cfg().get("mascot.x", 0.015))
        my = int(h * cfg().get("mascot.y", 0.655))
        fc.append(f"[{mi}:v]format=rgba[mas]")
        fc.append(f"{cur}[mas]overlay=x={mx}:y={my}:eof_action=pass:format=auto[vmas]")
        cur = "[vmas]"
    fc.append(f"{cur}drawbox=x=0:y=ih-10:w=iw:h=10:color=black@0.35:t=fill[vbar]")
    fc.append(f"color=c=0x{pc}:s={w}x10:r={fps}:d={total:.3f}[bar]")
    fc.append(f"[vbar][bar]overlay=x='-W+W*t/{total:.3f}':y=H-10:eval=frame:shortest=1,format=yuv420p[vout]")
    mv = cfg().get("render.music_volume_db", -20)
    fc.append(f"[{vi}:a]aformat=sample_rates=48000:channel_layouts=stereo,{VOICE_CHAIN},asplit=2[voice][key]")
    fc.append(f"[{vi + 1}:a]aformat=sample_rates=48000:channel_layouts=stereo,volume={mv}dB[mus]")
    fc.append("[mus][key]sidechaincompress=threshold=0.025:ratio=7:attack=12:release=320[duck]")
    fc.append(f"[{vi + 2}:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=-5dB[sfx]")
    fc.append(f"[voice][duck][sfx]amix=inputs=3:normalize=0,atrim=0:{total:.3f},"
              f"loudnorm=I={cfg().get('render.loudness_lufs', -14)}:TP=-1.5:LRA=9,aresample=48000,alimiter=limit=0.82:attack=2:release=60:level=disabled[aout]")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(fc),
           "-map", "[vout]", "-map", "[aout]", *video_encoder(), "-r", str(fps),
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", cfg().get("render.audio_bitrate", "192k"),
           "-movflags", "+faststart", str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg assemble failed: " + p.stderr[-2500:])
    return out


def thumbnail(video: Path, at: float, title: str, out: Path) -> Path:
    from PIL import ImageDraw

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

    from .subtitles import FONT_FILE
    for name in [str(FONT_FILE), "C:/Windows/Fonts/seguibl.ttf", "C:/Windows/Fonts/arialbd.ttf", "DejaVuSans-Bold.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()
