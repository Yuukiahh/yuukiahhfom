"""Automatic repair of the problems qa.check marks as fixable.

One ffmpeg pass: trims black frames at the edges, jump-cuts dead air out, forces 1080x1920
(padding, never cropping, so faces and captions stay intact), resyncs audio to video,
normalises loudness (two-pass loudnorm) and re-encodes H.264/AAC with faststart.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import media
from .qa import QAReport

KEEP_PAUSE = 0.25   # seconds of each silence kept on both sides, so cuts still breathe


def keep_segments(duration: float, report: QAReport) -> list[tuple[float, float]]:
    start, end = 0.0, duration
    cuts: list[tuple[float, float]] = []
    for issue in report.issues:
        if issue.code == "BLACK_START":
            start = max(start, float(issue.data["end"]) + 0.02)
        elif issue.code == "BLACK_END":
            end = min(end, float(issue.data["start"]) - 0.02)
        elif issue.code == "DEAD_AIR":
            for s, e in issue.data.get("segments", []):
                if e - s > 2 * KEEP_PAUSE + 0.2:
                    cuts.append((s + KEEP_PAUSE, e - KEEP_PAUSE))
    segments: list[tuple[float, float]] = []
    cursor = start
    for s, e in sorted(cuts):
        if s <= cursor or s >= end:
            continue
        segments.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < end:
        segments.append((cursor, end))
    return [(round(a, 3), round(b, 3)) for a, b in segments if b - a > 0.15]


def build_command(src: Path, dst: Path, report: QAReport, *, has_audio: bool, target_lufs: float,
                  true_peak: float, encoder: str, loud: dict | None) -> list[str]:
    segments = keep_segments(report.duration, report)
    parts, labels = [], []
    for i, (s, e) in enumerate(segments):
        parts.append(f"[0:v]trim=start={s}:end={e},setpts=PTS-STARTPTS[v{i}]")
        labels.append(f"[v{i}]")
        if has_audio:
            parts.append(f"[0:a]atrim=start={s}:end={e},asetpts=PTS-STARTPTS[a{i}]")
            labels.append(f"[a{i}]")
    n = len(segments)
    a_flag = 1 if has_audio else 0
    parts.append(f"{''.join(labels)}concat=n={n}:v=1:a={a_flag}[vc]" + ("[ac]" if has_audio else ""))
    parts.append(
        "[vc]scale=1080:1920:force_original_aspect_ratio=decrease:flags=lanczos,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,format=yuv420p[vout]"
    )
    if has_audio:
        norm = f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=11"
        if loud and {"input_i", "input_tp", "input_lra", "input_thresh"} <= loud.keys():
            norm += (f":measured_I={loud['input_i']}:measured_TP={loud['input_tp']}"
                     f":measured_LRA={loud['input_lra']}:measured_thresh={loud['input_thresh']}"
                     f":offset={loud.get('target_offset', 0)}:linear=true")
        # Pad/trim audio to exactly the video length: fixes drift without `-shortest`
        # (apad + -shortest fails with ENOSPC on ffmpeg 6.x).
        total = round(sum(e - s for s, e in segments), 3)
        parts.append(f"[ac]{norm},aresample=48000:async=1:first_pts=0,"
                     f"apad=whole_dur={total},atrim=0:{total}[aout]")

    cmd = [media.ffmpeg_bin(), "-hide_banner", "-y", "-i", str(src),
           "-filter_complex", ";".join(parts), "-map", "[vout]"]
    if has_audio:
        cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
    cmd += media.video_encoder_args(encoder)
    cmd += ["-movflags", "+faststart", str(dst)]
    return cmd


def repair(path: Path, report: QAReport, *, target_lufs: float = -14.0, true_peak: float = -1.0,
           encoder: str = "auto") -> Path:
    """Repair ``path`` in place (a .orig.mp4 backup is kept next to it) and return the path."""
    path = Path(path)
    info = media.probe(path)
    has_audio = media.stream(info, "audio") is not None
    loud = None
    if has_audio:
        try:
            loud = media.loudness(path, target_lufs, true_peak)
        except media.MediaError:
            loud = None
    tmp = path.with_suffix(".fixing.mp4")
    cmd = build_command(path, tmp, report, has_audio=has_audio, target_lufs=target_lufs,
                        true_peak=true_peak, encoder=encoder, loud=loud)
    proc = media.run(cmd, timeout=1800)
    if proc.returncode != 0 and "nvenc" in " ".join(cmd):
        # GPU busy or driver hiccup: fall back to the CPU encoder once.
        cmd = build_command(path, tmp, report, has_audio=has_audio, target_lufs=target_lufs,
                            true_peak=true_peak, encoder="libx264", loud=loud)
        proc = media.run(cmd, timeout=3600)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise media.MediaError(f"repair failed for {path.name}: {proc.stderr.strip()[-600:]}")
    backup = path.with_suffix(".orig.mp4")
    if not backup.exists():
        os.replace(path, backup)
    os.replace(tmp, path)
    return path
