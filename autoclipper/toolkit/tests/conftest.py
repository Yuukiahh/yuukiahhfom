import subprocess
from pathlib import Path

import pytest


def make_clip(path: Path, *, size="720x1280", seconds=15, black_start=0.6, gap=(6, 9), volume=0.03,
              audio=True) -> Path:
    """Synthetic clip with known defects: wrong size, black opening, dead air, too quiet."""
    vf = f"drawbox=enable='lt(t,{black_start})':color=black:t=fill" if black_start else "null"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", f"testsrc2=s={size}:r=30:d={seconds}"]
    if audio:
        expr = f"sin(2*PI*440*t)*{volume}"
        if gap:
            expr += f"*(lt(t,{gap[0]})+gt(t,{gap[1]}))"
        cmd += ["-f", "lavfi", "-i", f"aevalsrc='{expr}':s=48000:d={seconds}"]
    cmd += ["-vf", vf, "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [str(path)]
    subprocess.run(cmd, check=True)
    return path


@pytest.fixture
def clip_factory(tmp_path):
    def factory(name="clip.mp4", **kw):
        return make_clip(tmp_path / name, **kw)
    return factory


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOCLIPPER_HOME", str(tmp_path / "home"))
    from autoclipper.config import Settings
    s = Settings(output_root=str(tmp_path / "AI Clips"), encoder="libx264",
                 ollama_url="http://127.0.0.1:9", min_virality=55, min_clips=2)
    return s
