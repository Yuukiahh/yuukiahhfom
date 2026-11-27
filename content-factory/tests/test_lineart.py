"""Minimalist line-art motivation scenes: registry, shapes, determinism, motion, themes, params."""
import numpy as np
import pytest

from factory.visuals import lineart
from factory.visuals.procedural import KINDS, make_scene

KINDS_NEW = ["la_walk", "la_stairs", "la_clock", "la_wall", "la_calendar", "la_crowd",
             "la_fall_rise", "la_door", "la_mountain", "la_money", "la_phone", "la_mirror"]
THEMES = ["mono", "gold"]


def luma(f):
    f = np.clip(f, 0, 255)
    return float((f[..., 0] * 0.299 + f[..., 1] * 0.587 + f[..., 2] * 0.114).mean())


def test_kinds_registered():
    assert sorted(lineart.KINDS_LINEART) == sorted(KINDS_NEW)
    for k in KINDS_NEW:
        assert KINDS[k] is lineart.KINDS_LINEART[k]
        assert issubclass(KINDS[k], lineart.LineArt)


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("kind", KINDS_NEW)
def test_scene_renders_finite(kind, theme):
    s = make_scene(kind, 108, 192, 3, {"theme": theme, "label": "focus"})
    assert s.dynamic
    for t in (0.0, 1.0, 2.7, 6.1, 40.0):
        f = s.render(t)
        assert f.shape == (s.H, s.W, 3)
        assert f.dtype == np.float32
        assert np.isfinite(f).all()
        assert f.min() >= 0 and f.max() <= 255.5


@pytest.mark.parametrize("kind", KINDS_NEW)
def test_deterministic_and_animated(kind):
    for theme in THEMES:
        a = make_scene(kind, 216, 384, 5, {"theme": theme})
        b = make_scene(kind, 216, 384, 5, {"theme": theme})
        fa, fb = a.render(2.25), b.render(2.25)
        assert np.array_equal(fa, fb)
        a.render(9.0)  # no hidden per-frame state
        assert np.array_equal(a.render(2.25), fa)
        # something visibly moves within half a second (slow, calm genre -> modest threshold)
        for t in (1.0, 3.0):
            assert np.abs(a.render(t + 0.5) - a.render(t)).max() > 40


@pytest.mark.parametrize("kind", KINDS_NEW)
def test_themes_look(kind):
    mono = make_scene(kind, 270, 480, 2, {"theme": "mono"}).render(3.0)
    gold = make_scene(kind, 270, 480, 2, {"theme": "gold"}).render(3.0)
    # mono: black background, neutral grey/white lines that are bright and crisp
    assert np.allclose(mono[..., 0], mono[..., 1]) and np.allclose(mono[..., 1], mono[..., 2])
    assert np.median(mono) < 3 and mono.max() > 200
    assert luma(mono) < 20
    # gold: warm (red > blue) dark backdrop, gold highlights
    assert gold[..., 0].mean() > gold[..., 2].mean() + 3
    bright = gold[gold.max(-1) > 200]
    assert len(bright) > 0 and (bright[:, 0] >= bright[:, 2]).all()
    assert 8 < luma(gold) < 50


def test_gold_timer_counts_time():
    s = make_scene("la_walk", 270, 480, 1, {"theme": "gold", "t0": 5})
    assert s.show_timer
    a, b = s.render(1.2), s.render(2.2)  # 0:06 -> 0:07
    box = (slice(int(s.Y(0.09)), int(s.Y(0.2))), slice(int(s.X(0.62)), int(s.X(0.93))))
    assert np.abs(a[box] - b[box]).max() > 50
    assert not make_scene("la_walk", 108, 192, 1, {"theme": "mono"}).show_timer
    assert not make_scene("la_walk", 108, 192, 1, {"theme": "gold", "timer": False}).show_timer


def test_label_chip_and_params():
    plain = make_scene("la_walk", 216, 384, 1, {}).render(1.0)
    lab = make_scene("la_walk", 216, 384, 1, {"label": "keep going"}).render(1.0)
    assert np.abs(plain - lab).max() > 50
    assert lineart.clean_word("keep going!") == "KEEP GOING!"
    assert lineart.parse_theme("GOLDEN") == "gold" and lineart.parse_theme("???") == "mono"
    assert lineart.parse_labels("tomorrow, next week|next year") == ["TOMORROW", "NEXT WEEK", "NEXT YEAR"]
    cal = make_scene("la_calendar", 108, 192, 1, {"label": "tomorrow,next week,next year"})
    assert cal.words == ["TOMORROW", "NEXT WEEK", "NEXT YEAR"] and cal.npages == 3
    assert make_scene("la_calendar", 108, 192, 1, {}).npages == 7
    fast = make_scene("la_clock", 216, 384, 1, {"mode": "fast"})
    stop = make_scene("la_clock", 216, 384, 1, {"mode": "stop"})
    assert fast.fast and not stop.fast
    # stopped clock: the hands stay (almost) still
    assert np.abs(stop.render(1.0) - stop.render(1.6)).mean() < np.abs(fast.render(1.0) - fast.render(1.6)).mean()
    sit = make_scene("la_crowd", 108, 192, 1, {"pose": "sit"})
    assert sit.sit and not make_scene("la_crowd", 108, 192, 1, {}).sit


@pytest.mark.parametrize("bad", [None, "abc", -5, 2.0, float("nan"), [], {"x": 1}])
def test_params_are_tolerant(bad):
    for kind in KINDS_NEW:
        s = make_scene(kind, 108, 192, 1, {"theme": bad, "label": bad, "t0": bad, "timer": bad, "mode": bad,
                                           "pose": bad, "labels": bad})
        assert np.isfinite(s.render(1.0)).all()
        assert np.isfinite(s.render(float("nan"))).all()


def test_figure_poses_are_sane():
    h = 100.0
    for pose in (lineart.pose_stand(0), lineart.pose_walk(1.0), lineart.pose_run(2.0), lineart.pose_push(0),
                 lineart.pose_kneel(0), lineart.pose_fallen(0), lineart.pose_sit(0), lineart.pose_victory(0),
                 lineart.pose_lookup(0), lineart.pose_stand_front(0), lineart.pose_hunched(0), lineart.pose_proud(0)):
        A = lineart.place(pose, h, 50.0, 200.0)
        pts = np.array([v for v in A.values() if isinstance(v, tuple)])
        assert np.isfinite(pts).all()
        assert pts[:, 1].max() <= 200.0 + 1e-6          # nothing below the ground line
        assert A["head"][1] < A["hip"][1]                 # head above the hips
    # walk stride bookkeeping is continuous (no foot sliding jumps)
    g = lineart.Gait(lineart.pose_walk, h)
    offs = [g.offset(p) for p in np.linspace(0, 12, 600)]
    assert np.all(np.diff(offs) > -1e-6) and np.abs(np.diff(offs)).max() < 2.0


def test_catalog_for_scriptwriter():
    assert set(lineart.CATALOG) == set(KINDS_NEW)
    assert all(isinstance(v, str) and len(v) > 20 for v in lineart.CATALOG.values())
