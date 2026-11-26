"""Thin wrappers around ffmpeg / ffprobe."""

from __future__ import annotations

import functools
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


class MediaError(RuntimeError):
    pass


def ffmpeg_bin() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise MediaError("ffmpeg not found on PATH (install: winget install Gyan.FFmpeg)")
    return path


def ffprobe_bin() -> str:
    path = shutil.which("ffprobe")
    if not path:
        raise MediaError("ffprobe not found on PATH (it ships with ffmpeg)")
    return path


def run(cmd: list[str], *, timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout
    )


def probe(path: Path) -> dict[str, Any]:
    proc = run([ffprobe_bin(), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)])
    if proc.returncode != 0:
        raise MediaError(f"ffprobe failed for {path}: {proc.stderr.strip()[-400:]}")
    return json.loads(proc.stdout or "{}")


def stream(info: dict, kind: str) -> dict | None:
    for s in info.get("streams", []):
        if s.get("codec_type") == kind:
            return s
    return None


def fps_of(video_stream: dict | None) -> float:
    if not video_stream:
        return 0.0
    raw = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0/1"
    num, _, den = raw.partition("/")
    try:
        return float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0


def stream_duration(s: dict | None, fallback: float) -> float:
    if not s:
        return 0.0
    try:
        return float(s.get("duration"))
    except (TypeError, ValueError):
        return fallback


def filter_log(path: Path, vf: str | None = None, af: str | None = None, timeout: float = 600) -> str:
    """Run ffmpeg with an analysis filter and return its stderr log."""
    cmd = [ffmpeg_bin(), "-hide_banner", "-nostats", "-i", str(path)]
    if vf:
        cmd += ["-vf", vf]
    else:
        cmd += ["-vn"]
    if af:
        cmd += ["-af", af]
    else:
        cmd += ["-an"]
    cmd += ["-f", "null", "-"]
    proc = run(cmd, timeout=timeout)
    if proc.returncode != 0:
        raise MediaError(f"ffmpeg analysis failed: {proc.stderr.strip()[-400:]}")
    return proc.stderr


def black_segments(path: Path) -> list[tuple[float, float]]:
    log = filter_log(path, vf="blackdetect=d=0.08:pic_th=0.98:pix_th=0.10")
    return [(float(a), float(b)) for a, b in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", log)]


def freeze_segments(path: Path) -> list[tuple[float, float]]:
    log = filter_log(path, vf="freezedetect=n=-60dB:d=2")
    starts = [float(x) for x in re.findall(r"freeze_start: ([\d.]+)", log)]
    ends = [float(x) for x in re.findall(r"freeze_end: ([\d.]+)", log)]
    return list(zip(starts, ends + [float("inf")] * (len(starts) - len(ends))))


def silence_segments(path: Path, noise_db: float = -40, min_dur: float = 1.2) -> list[tuple[float, float]]:
    log = filter_log(path, af=f"silencedetect=noise={noise_db}dB:d={min_dur}")
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", log)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", log)]
    return list(zip(starts, ends + [float("inf")] * (len(starts) - len(ends))))


def loudness(path: Path, target: float = -14.0, tp: float = -1.0) -> dict[str, float]:
    """First loudnorm pass: measured integrated loudness, true peak, LRA and threshold."""
    log = filter_log(path, af=f"loudnorm=I={target}:TP={tp}:LRA=11:print_format=json")
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", log, re.S)
    if not match:
        raise MediaError("could not read loudnorm measurement")
    data = json.loads(match.group(0))
    return {k: float(v) for k, v in data.items() if k.startswith(("input_", "target_offset"))}


@functools.lru_cache(maxsize=1)
def nvenc_available() -> bool:
    """True when h264_nvenc is compiled in AND an NVIDIA GPU actually accepts a test encode."""
    try:
        enc = run([ffmpeg_bin(), "-hide_banner", "-encoders"], timeout=30)
        if "h264_nvenc" not in enc.stdout:
            return False
        test = run([
            ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
            "color=c=black:s=256x256:d=0.2", "-c:v", "h264_nvenc", "-f", "null", "-",
        ], timeout=60)
        return test.returncode == 0
    except (MediaError, subprocess.TimeoutExpired, OSError):
        return False


def video_encoder_args(preference: str = "auto") -> list[str]:
    use_nvenc = preference == "h264_nvenc" or (preference == "auto" and nvenc_available())
    if use_nvenc:
        return ["-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", "19",
                "-b:v", "0", "-profile:v", "high", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-profile:v", "high", "-pix_fmt", "yuv420p"]
