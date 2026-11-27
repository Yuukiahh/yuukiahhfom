"""Minimalist line-art motivation scenes: registry, shapes, determinism, motion, themes, params."""
import numpy as np
import pytest

from factory.visuals import lineart
from factory.visuals.procedural import KINDS, make_scene

KINDS_NEW = ["la_walk", "la_stairs", "la_clock", "la_wall", "la_calendar", "la_crowd",
             "la_fall_rise", "la_door", "la_mountain", "la_money", "la_phone", "la_mirror"]
THEMES = ["mono", "gold"]
KINDS_STORY = ["la_split", "la_life", "la_typo", "la_fork", "la_hourglass", "la_seed", "la_domino",
               "la_boulder", "la_gym", "la_book"]
ALL_KINDS = KINDS_NEW + KINDS_STORY
ALL_THEMES = ["mono", "gold", "paper", "blueprint", "neon", "chalk"]


def luma(f):
    f = np.clip(f, 0, 255)
    return float((f[..., 0] * 0.299 + f[..., 1] * 0.587 + f[..., 2] * 0.114).mean())


def test_kinds_registered():
    assert sorted(lineart.KINDS_LINEART) == sorted(ALL_KINDS)
    for k in ALL_KINDS:
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
    box = (slice(int(s.Y(0.15)), int(s.Y(0.27))), slice(int(s.X(0.62)), int(s.X(0.93))))
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
    for kind in ALL_KINDS:
        s = make_scene(kind, 108, 192, 1, {"theme": bad, "label": bad, "t0": bad, "timer": bad, "mode": bad,
                                           "pose": bad, "labels": bad, "action": bad, "left": bad, "right": bad,
                                           "stage": bad, "text": bad, "fruit": bad, "figure": bad})
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
    assert set(lineart.CATALOG) == set(ALL_KINDS)
    assert all(isinstance(v, str) and len(v) > 20 for v in lineart.CATALOG.values())


# ------------------------------------------------------------------ themes x story scenes
def test_theme_list_and_aliases():
    assert lineart.THEMES == ALL_THEMES
    assert set(lineart.PALETTES) == set(ALL_THEMES)
    assert lineart.parse_theme("Chalkboard") == "chalk" and lineart.parse_theme("ink") == "paper"
    assert lineart.parse_theme("cyber") == "neon" and lineart.parse_theme("blue") == "blueprint"
    assert lineart.parse_theme("minimal_gold") == "gold" and lineart.parse_theme(["neon"]) == "mono"


@pytest.mark.parametrize("theme", ALL_THEMES)
@pytest.mark.parametrize("kind", KINDS_STORY)
def test_story_scene_renders_finite(kind, theme):
    s = make_scene(kind, 108, 192, 3, {"theme": theme, "label": "focus"})
    assert s.dynamic
    for t in (0.0, 0.7, 1.5, 2.7, 6.1, 40.0):
        f = s.render(t)
        assert f.shape == (s.H, s.W, 3)
        assert f.dtype == np.float32
        assert np.isfinite(f).all()
        assert f.min() >= 0 and f.max() <= 255.5


@pytest.mark.parametrize("theme", ["paper", "blueprint", "neon", "chalk"])
@pytest.mark.parametrize("kind", KINDS_NEW)
def test_existing_scenes_in_new_themes(kind, theme):
    s = make_scene(kind, 108, 192, 3, {"theme": theme, "label": "focus"})
    for t in (0.0, 2.7, 40.0):
        f = s.render(t)
        assert f.shape == (s.H, s.W, 3) and np.isfinite(f).all()
        assert f.min() >= 0 and f.max() <= 255.5


@pytest.mark.parametrize("kind", KINDS_STORY)
def test_story_deterministic_and_energetic(kind):
    for theme in ALL_THEMES:
        a = make_scene(kind, 216, 384, 5, {"theme": theme})
        b = make_scene(kind, 216, 384, 5, {"theme": theme})
        fa, fb = a.render(1.75), b.render(1.75)
        assert np.array_equal(fa, fb)
        a.render(9.0)  # no hidden per-frame state
        assert np.array_equal(a.render(1.75), fa)
        # fast genre: clear motion within a quarter / half second
        for t in (0.5, 1.5, 3.0):
            assert np.abs(a.render(t + 0.5) - a.render(t)).max() > 40, (theme, t)
        assert np.abs(a.render(1.25) - a.render(1.0)).mean() > 0.02


@pytest.mark.parametrize("kind", KINDS_NEW + KINDS_STORY)
def test_new_themes_look(kind):
    fr = {th: make_scene(kind, 270, 480, 2, {"theme": th}).render(2.0) for th in ("paper", "blueprint", "neon", "chalk")}
    paper = fr["paper"]
    assert luma(paper) > 180                                   # bright paper
    assert np.median(paper[..., 0]) > np.median(paper[..., 2]) + 10   # warm cream
    assert paper.min() < 90                                    # dark ink strokes
    bp = fr["blueprint"]
    assert np.median(bp[..., 2]) > np.median(bp[..., 0]) + 30 and 20 < luma(bp) < 90  # deep blue
    assert bp.min(-1).max() > 180                              # near-white lines
    neon = fr["neon"]
    assert luma(neon) < 45
    assert (neon[..., 0] - neon[..., 1]).max() > 120 or (neon[..., 2] - neon[..., 0]).max() > 120  # saturated glow
    ch = fr["chalk"]
    assert np.median(ch[..., 1]) > np.median(ch[..., 2]) and 20 < luma(ch) < 80   # green-grey board
    assert ch.min(-1).max() > 170                              # white chalk


def test_paper_lines_boil_and_textures_are_static():
    s = make_scene("la_walk", 216, 384, 1, {"theme": "paper"})
    # the jitter field changes 8x per second: same boil step -> same jitter phases
    assert lineart.PALETTES["paper"]["wob"] > 0 and s._boil.shape[1] == 4
    m = make_scene("la_hourglass", 216, 384, 1, {"theme": "paper"})
    # the background texture is precomputed once (no per-frame noise)
    assert m.tex.size == (m.W, m.H)
    corner = (slice(0, 20), slice(0, 20))
    assert np.array_equal(m.render(1.0)[corner], m.render(1.3)[corner])
    assert not make_scene("la_split", 108, 192, 1, {"theme": "paper"}).show_timer
    assert make_scene("la_split", 108, 192, 1, {"theme": "gold"}).show_timer


def test_story_params():
    sp = {a: make_scene("la_split", 216, 384, 1, {"action": a}) for a in lineart.SPLIT_ACTIONS}
    assert all(s.action == a for a, s in sp.items())
    assert make_scene("la_split", 108, 192, 1, {"action": "???"}).action == "phone"
    frames = [s.render(1.5) for s in sp.values()]
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            assert np.abs(frames[i] - frames[j]).max() > 50
    s = make_scene("la_split", 108, 192, 1, {"left": "broke", "right": "wealthy"})
    assert (s.left, s.right) == ("BROKE", "WEALTHY")
    assert make_scene("la_split", 108, 192, 1, {}).left == "POOR"
    assert make_scene("la_life", 108, 192, 1, {"stage": 3}).stage == 3
    assert make_scene("la_life", 108, 192, 1, {"stage": 9}).stage == 4
    assert make_scene("la_life", 108, 192, 1, {}).stage is None
    a = make_scene("la_life", 216, 384, 1, {"stage": 0}).render(2.0)
    b = make_scene("la_life", 216, 384, 1, {"stage": 4}).render(2.0)
    assert np.abs(a - b).max() > 50
    ty = make_scene("la_typo", 216, 384, 1, {"text": "nobody is coming"})
    assert ty.lines == ["NOBODY", "IS", "COMING"]
    assert make_scene("la_typo", 108, 192, 1, {}).lines == ["NO", "EXCUSES"]
    assert len(make_scene("la_typo", 108, 192, 1, {"text": "a b c d e f g"}).lines) == 3
    blank = make_scene("la_typo", 216, 384, 1, {"text": "discipline"})
    assert np.abs(blank.render(0.0) - blank.render(1.5)).max() > 100  # letters write themselves in
    fk = make_scene("la_fork", 108, 192, 1, {"left": "comfort", "right": "growth"})
    assert (fk.left, fk.right) == ("COMFORT", "GROWTH")
    assert make_scene("la_boulder", 108, 192, 1, {"mode": "top"}).top
    assert not make_scene("la_boulder", 108, 192, 1, {}).top
    assert make_scene("la_gym", 108, 192, 1, {"mode": "pushup"}).pushup
    assert make_scene("la_seed", 108, 192, 1, {"fruit": "apple"}).fruit == "fruit"
    assert make_scene("la_seed", 108, 192, 1, {}).fruit == "coin"


def test_caption_zones_stay_clear():
    """Main action lives in the middle band: top 12 % and bottom 20 % of the visible frame stay calm."""
    for kind in KINDS_STORY:
        s = make_scene(kind, 270, 480, 4, {"theme": "mono"})
        f = s.render(2.0)
        x0, y0 = (s.W - 270) // 2, (s.H - 480) // 2
        vis = f[y0:y0 + 480, x0:x0 + 270, 0]
        top, bot = vis[:int(0.12 * 480)], vis[int(0.8 * 480):]
        assert (top > 90).mean() < 0.004, kind
        assert (bot > 90).mean() < 0.006, kind


@pytest.mark.parametrize("theme", ALL_THEMES)
def test_render_speed_full_hd(theme):
    import time
    s = make_scene("la_split", 1080, 1920, 1, {"theme": theme, "action": "spend"})
    s.render(0.5)
    t0 = time.perf_counter()
    for t in (1.0, 1.3):
        s.render(t)
    assert (time.perf_counter() - t0) / 2 < 0.6  # generous CI bound (~0.1 s locally)
