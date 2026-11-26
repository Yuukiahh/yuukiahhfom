"""Contact sheets for visual review (caption placement, faces, hook frames).

Claude looks at these PNGs with its image reader: it catches what ffmpeg cannot, such as
misspelled or badly placed captions, cut-off faces and weak opening frames.
"""

from __future__ import annotations

from pathlib import Path

from . import media


def contact_sheet(clip: Path, out_png: Path, *, cols: int = 4, rows: int = 3) -> Path:
    info = media.probe(clip)
    duration = float(info.get("format", {}).get("duration") or 0) or 1.0
    n = cols * rows
    out_png.parent.mkdir(parents=True, exist_ok=True)
    vf = (f"fps={n}/{duration:.3f},scale=270:480:force_original_aspect_ratio=decrease,"
          f"pad=270:480:(ow-iw)/2:(oh-ih)/2,drawtext=text='%{{pts\\:hms}}':x=6:y=6:fontsize=18:"
          f"fontcolor=yellow:box=1:boxcolor=black@0.6,tile={cols}x{rows}")
    proc = media.run([media.ffmpeg_bin(), "-hide_banner", "-y", "-i", str(clip), "-vf", vf,
                      "-frames:v", "1", "-update", "1", str(out_png)], timeout=300)
    if proc.returncode != 0:
        # drawtext needs a font; retry without timestamps rather than fail the review.
        vf = vf.split(",drawtext")[0] + f",tile={cols}x{rows}"
        proc = media.run([media.ffmpeg_bin(), "-hide_banner", "-y", "-i", str(clip), "-vf", vf,
                          "-frames:v", "1", "-update", "1", str(out_png)], timeout=300)
    if proc.returncode != 0:
        raise media.MediaError(f"contact sheet failed: {proc.stderr.strip()[-300:]}")
    return out_png


def hook_strip(clip: Path, out_png: Path) -> Path:
    """Frames at 0, 0.5, ... 2.5 seconds: is the hook visible and readable immediately?"""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    vf = "trim=0:3,fps=2,scale=270:480:force_original_aspect_ratio=decrease,pad=270:480:(ow-iw)/2:(oh-ih)/2,tile=6x1"
    proc = media.run([media.ffmpeg_bin(), "-hide_banner", "-y", "-i", str(clip), "-vf", vf,
                      "-frames:v", "1", "-update", "1", str(out_png)], timeout=300)
    if proc.returncode != 0:
        raise media.MediaError(f"hook strip failed: {proc.stderr.strip()[-300:]}")
    return out_png
