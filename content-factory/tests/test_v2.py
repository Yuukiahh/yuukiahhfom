"""Tests for v2 features: German text normalisation, transitions, camera, captions."""
import numpy as np

from factory import subtitles, textnorm, transitions
from factory.render import camera_at


def test_german_numbers_and_units():
    s = textnorm.spoken_text("Im Jahr 1977 startete Voyager 1 – heute 25,9 Mrd. km entfernt.", "de")
    assert "neunzehnhundertsiebenundsiebzig" in s and "fünfundzwanzig Komma neun Milliarden Kilometer" in s
    assert "Mrd" not in s and "km" not in s
    assert textnorm.spoken_text("Am 18. November", "de") == "Am achtzehnten November"
    assert "zehn hoch fünfundzwanzig" in textnorm.spoken_text("10²⁵ Jahre", "de")
    assert "vier Millionen mal" in textnorm.spoken_text("4.000.000×", "de")
    assert textnorm.spoken_text("99% KI", "de") == "neunundneunzig Prozent K I"


def test_spoken_words_keep_one_to_one_mapping():
    disp = "Das kostet 5 Mio. Euro und ~49 Jahre.".split()
    assert len(textnorm.spoken_words(disp, "de")) == len(disp)


def test_transitions_render_all_kinds():
    a = [np.full((64, 36, 3), 40, np.uint8)] * 6
    b = [np.full((64, 36, 3), 200, np.uint8)] * 6
    for kind in transitions.SFX_FOR:
        out = transitions.render(kind, a, b)
        assert len(out) == 6 and out[0].shape == (64, 36, 3)
        assert out[-1].mean() > out[0].mean()  # ends on clip B


def test_camera_shots_and_bumps_are_smooth():
    spec = {"duration": 6.0, "shots": [(0.0, "push_in"), (3.0, "pan_left")], "bumps": [1.0]}
    vals = [camera_at(spec, t / 60)[0] for t in range(0, 170)]
    jumps = np.abs(np.diff(vals))
    assert jumps.max() < 0.02  # no jitter inside a shot (shot cut at 3.0 s is intentional)


def test_caption_groups_and_pill(tmp_path):
    words = [{"text": w, "start": i * 0.3, "end": i * 0.3 + 0.28, "scene": 0}
             for i, w in enumerate("Stell dir vor, du fällst in ein SCHWARZES Loch.".split())]
    out = subtitles.build_ass(words, tmp_path / "s.ass", ["schwarzes"], [], "HOOK",
                              cta={"text": "Folge für mehr!", "start": 2, "end": 3})
    txt = out.read_text()
    assert "\\p1" in txt and "FOLGE FÜR MEHR!" in txt
    actives = [l for l in txt.splitlines() if l.startswith("Dialogue: 4,")]
    assert len(actives) == 9
