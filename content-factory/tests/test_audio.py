"""Audio: music engine v2, SFX library, licensed-music helpers. Kept fast (< 10 s)."""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from factory import audio, audio_dsp, music_gen, sfx_design  # noqa: E402

REQUIRED_SFX = {"whoosh", "whoosh_long", "swoosh_reverse", "impact", "sub_drop", "riser", "riser_long", "pop", "tick",
                "glitch", "ding", "shimmer", "swish_text", "shutter", "notification", "heartbeat", "typing", "sparkle",
                "braam", "zap"}


# ---------------------------------------------------------------- music
@pytest.mark.parametrize("mood", sorted(music_gen.MOODS))
def test_music_moods_clean(mood):
    x = audio.music(5.0, mood, seed=11)
    assert x.shape == (int(5.0 * audio.SR), 2)
    assert np.isfinite(x).all()
    assert np.abs(x).max() <= 0.81
    assert audio_dsp.true_peak_db(x) < -1.0
    assert np.abs(x.mean(0)).max() < 1e-3          # no DC offset
    assert np.abs(x[-int(0.02 * audio.SR):]).max() < 0.01   # ends in silence (resolved ending)
    assert np.sqrt((x[: audio.SR] ** 2).mean()) > 0.02      # hook from the first second


def test_music_backward_compatible_signature(tmp_path):
    out = tmp_path / "m.wav"
    x = audio.music(2.0, "nonexistent-mood", out)
    assert out.exists() and len(x) == 2 * audio.SR
    assert sf.info(out).samplerate == audio.SR


def test_music_deterministic_and_seeded():
    a = audio.music(2.5, "pulse", seed=5)
    b = audio.music(2.5, "pulse", seed=5)
    c = audio.music(2.5, "pulse", seed=6)
    assert np.array_equal(a, b)
    assert not np.allclose(a, c)


def test_music_arrangement_loudness_and_speed():
    t = time.time()
    x, info = music_gen.render(30.0, "epic", 44100, seed=3)
    assert time.time() - t < 3.0                    # ~1.3 s here; 90 s renders in ~2.5 s
    kinds = [k for _, k, _ in info["sections"]]
    assert kinds[0] == "intro" and "build" in kinds and "drop" in kinds and "end" in kinds
    assert 0.15 * 30 <= info["drop"] <= 0.4 * 30
    assert abs(audio_dsp.lufs(x, 44100) - (-14)) < 1.5
    # the breakdown is quieter than the drops around it
    sr = 44100
    secs = info["sections"]
    rms = lambda a, b: np.sqrt((x[int(a * sr):int(b * sr)] ** 2).mean())  # noqa: E731
    bd = [(t0, secs[i + 1][0]) for i, (t0, k, _) in enumerate(secs[:-1]) if k == "breakdown"]
    dr = [(t0, secs[i + 1][0]) for i, (t0, k, _) in enumerate(secs[:-1]) if k == "drop"]
    if bd:
        assert rms(*bd[0]) < 0.8 * np.mean([rms(*s) for s in dr])


def test_music_markers_drive_drop_swells_and_fills():
    markers = [3.0, 8.0, 14.0, 20.0]
    p = music_gen.plan(30.0, 120.0, 4, markers=markers)
    assert p.drop == 8.0
    assert any(abs(b.t - 8.0) < 1e-6 and b.kind == "drop" for b in p.bars)   # drop on a downbeat at the marker
    assert p.bars[[round(b.t, 6) for b in p.bars].index(8.0) - 1].roll       # build into it
    assert set(p.swells) == set(markers)
    # 14.0 lies on the (tempo-nudged) grid -> fill into it or an on-grid crash
    assert any(b.fill for b in p.bars) or p.hits
    x = audio.music(10.0, "pulse", seed=1, markers=[2.5, 4.0, 7.0])
    assert len(x) == 10 * audio.SR


def test_music_energy_curve():
    x, info = music_gen.render(12.0, "lofi", 44100, seed=2, energy_curve=[(0, 0.2), (6, 0.2), (6.5, 1.0), (12, 1.0)])
    kinds = [k for _, k, _ in info["sections"]]
    assert kinds[0] == "breakdown" and "drop" in kinds
    assert np.abs(x).max() <= 0.81


def test_music_tiny_durations():
    for d in (0.01, 0.3, 1.0):
        assert len(audio.music(d, "playful", seed=1)) == int(d * audio.SR)


# ---------------------------------------------------------------- SFX
def test_sfx_assets_committed_and_clean():
    files = sorted(audio.ASSETS_SFX.glob("*.wav"))
    names = set(audio._scan_sfx(audio.ASSETS_SFX))
    assert REQUIRED_SFX <= names
    assert sum(f.stat().st_size for f in files) < 6 * 1024 * 1024
    assert (audio.ASSETS_SFX / "LICENSE.md").exists()
    for f in files:
        x, sr = sf.read(f, always_2d=True)
        assert sr == 48000 and x.shape[1] == 2, f.name
        pk = 20 * np.log10(np.abs(x).max())
        assert -12.0 < pk <= -0.99, (f.name, pk)
        assert audio_dsp.true_peak_db(x) <= -0.95, f.name
        assert np.abs(x[-3:]).max() < 0.01, f.name            # faded out
    assert len(audio._scan_sfx(audio.ASSETS_SFX)["whoosh"]) >= 3


def test_build_sfx_library_prefers_assets(tmp_path):
    lib = audio.build_sfx_library(tmp_path / "sfx")
    assert REQUIRED_SFX <= set(lib)
    assert {"whoosh", "impact", "riser", "pop", "glitch", "ding"} <= set(lib)
    assert all(Path(p).parent == audio.ASSETS_SFX for p in lib.values())
    assert len(lib.variants["whoosh"]) >= 3
    assert not (tmp_path / "sfx").exists() or not any((tmp_path / "sfx").iterdir())


def test_build_sfx_library_fallback_when_assets_missing(tmp_path):
    lib = audio.build_sfx_library(tmp_path / "sfx", assets=tmp_path / "nothing-here")
    assert set(lib) == set(audio.SFX)
    for p in lib.values():
        assert p.exists() and p.parent == tmp_path / "sfx"


def test_sfx_track_rotates_variants_and_aliases(tmp_path):
    lib = audio.build_sfx_library(tmp_path / "sfx")
    out = tmp_path / "s.wav"
    ev = [(0.5, "whoosh", 1.0), (2.0, "whoosh", 1.0), (3.5, "camera_shutter", 0.5), (4.0, "unknown", 1.0),
          (-0.1, "pop", 0.5), (4.5, "riser_end", 0.5)]
    audio.sfx_track(ev, 6.0, lib, out)
    x, sr = sf.read(out, always_2d=True)
    assert sr == audio.SFX_SR and len(x) == 6 * sr
    a, b = x[int(0.5 * sr):int(1.0 * sr)], x[int(2.0 * sr):int(2.5 * sr)]
    assert not np.allclose(a, b, atol=1e-3)          # two different whoosh variants
    assert np.abs(x[int(3.5 * sr):int(3.8 * sr)]).max() > 0.05    # alias resolved
    assert np.abs(x).max() <= 0.951


def test_sfx_design_deterministic():
    a = sfx_design.render("zap", 1)
    b = sfx_design.render("zap", 1)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------- licensed library
def _track(folder: Path, name: str, seconds: float, meta: dict | None):
    t = np.arange(int(seconds * 44100)) / 44100
    sf.write(folder / f"{name}.wav", np.stack([np.sin(2 * np.pi * 220 * t)] * 2, 1) * 0.3, 44100)
    if meta is not None:
        (folder / f"{name}.json").write_text(json.dumps(meta), "utf-8")


def test_pick_library_track(tmp_path):
    base = {"title": "T", "artist": "A", "source_url": "https://example.org"}
    _track(tmp_path, "nc", 10, {**base, "license": "CC BY-NC 4.0", "moods": ["cosmic"]})
    _track(tmp_path, "nometa", 10, None)
    assert audio.pick_library_track("cosmic", 5, dirs=[tmp_path]) is None
    _track(tmp_path, "short", 3, {**base, "license": "CC0", "moods": ["cosmic"]})
    _track(tmp_path, "long", 12, {**base, "license": "CC BY 4.0", "moods": ["cosmic", "epic"]})
    _track(tmp_path, "other", 12, {**base, "license": "CC0", "moods": ["playful"]})
    assert audio.pick_library_track("cosmic", 10, dirs=[tmp_path]).name == "long.wav"
    assert audio.pick_library_track("cosmic", 30, dirs=[tmp_path]).name in ("long.wav", "short.wav")
    assert audio.pick_library_track("lofi", 5, dirs=[tmp_path]) is None
    assert audio.pick_library_track("cosmic", 5, dirs=[tmp_path / "missing"]) is None


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg missing")
def test_fit_track_loops_and_trims(tmp_path):
    _track(tmp_path, "loop", 3, None)
    out = audio.fit_track(tmp_path / "loop.wav", 8.0, tmp_path / "fit.wav")
    x, sr = sf.read(out, always_2d=True)
    assert abs(len(x) / sr - 8.0) < 0.05 and x.shape[1] == 2
    assert np.abs(x[-200:]).max() < 0.01
    out2 = audio.fit_track(tmp_path / "loop.wav", 2.0, tmp_path / "fit2.wav")
    assert abs(sf.info(out2).duration - 2.0) < 0.05
