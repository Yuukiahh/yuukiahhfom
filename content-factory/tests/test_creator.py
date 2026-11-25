"""Creator-economy / social / AI-video background scenes: registry, shapes, brightness, determinism."""
import numpy as np
import pytest

from factory.visuals.procedural import KINDS, make_scene
from factory.visuals import creator

KINDS_NEW = ["phone_feed", "edit_timeline", "money_rain", "analytics_dash", "ai_studio", "creator_desk"]
ALIASES = {"social_feed": "phone_feed", "short_video": "phone_feed", "video_editing": "edit_timeline",
           "clipping": "edit_timeline", "money": "money_rain", "coins": "money_rain", "analytics": "analytics_dash",
           "dashboard": "analytics_dash", "ai_generator": "ai_studio", "ai_video": "ai_studio",
           "creator_setup": "creator_desk", "ring_light": "creator_desk", "ugc": "creator_desk"}
PARAMS = {"accent": "#FF3D7F", "count": 250000, "intensity": 0.8, "trend": "volatile"}


def luma(f):
    f = np.clip(f, 0, 255)
    return float((f[..., 0] * 0.299 + f[..., 1] * 0.587 + f[..., 2] * 0.114).mean())


def test_kinds_registered():
    for k in KINDS_NEW:
        assert KINDS[k] is creator.KINDS_CREATOR[k]
    for alias, kind in ALIASES.items():
        assert KINDS[alias] is KINDS[kind]
    for alias, cls in creator.KINDS_CREATOR.items():
        assert KINDS[alias] is cls


@pytest.mark.parametrize("kind", KINDS_NEW)
def test_scene_renders_finite(kind):
    s = make_scene(kind, 108, 192, 3, PARAMS)
    assert s.dynamic
    for t in (0.0, 1.7, 40.0):
        f = s.render(t)
        assert f.shape == (s.H, s.W, 3)
        assert f.dtype == np.float32
        assert np.isfinite(f).all()
        assert f.min() >= -1e-3 and f.max() < 700
    assert f.max() > 5


@pytest.mark.parametrize("kind", KINDS_NEW)
def test_scene_is_deterministic_and_animated(kind):
    a = make_scene(kind, 108, 192, 5)
    b = make_scene(kind, 108, 192, 5)
    fa, fb = a.render(2.25), b.render(2.25)
    assert np.allclose(fa, fb)
    a.render(9.0)  # no hidden per-frame state
    assert np.allclose(a.render(2.25), fa)
    assert np.abs(a.render(3.0) - fa).mean() > 0.5  # frames change over time


@pytest.mark.parametrize("kind", KINDS_NEW)
def test_not_too_dark(kind):
    s = make_scene(kind, 270, 480, 2, {"accent": "#3FA9FF"})
    for t in (0.5, 6.0):
        assert luma(s.render(t)) >= 35


@pytest.mark.parametrize("bad", [None, "abc", -5, 2.0])
def test_params_are_tolerant(bad):
    for kind in ("phone_feed", "money_rain", "analytics_dash"):
        s = make_scene(kind, 108, 192, 1, {"count": bad, "intensity": bad, "trend": bad, "accent": bad})
        assert np.isfinite(s.render(1.0)).all()


def test_like_counter_climbs_toward_count():
    s = make_scene("phone_feed", 108, 192, 1, {"count": 50000})
    assert s.count == 50000
    assert creator.fmt_count(1_234_567) == "1.2M"
    assert creator.fmt_count(48_200) == "48.2K"
    assert creator.fmt_count(950) == "950"


def test_money_intensity_scales_rain():
    lo = make_scene("money_rain", 108, 192, 1, {"intensity": 0.0})
    hi = make_scene("money_rain", 108, 192, 1, {"intensity": 1.0})
    assert hi.nc > lo.nc and hi.nn > lo.nn


@pytest.mark.parametrize("trend", ["up", "down", "volatile"])
def test_analytics_trend(trend):
    s = make_scene("analytics_dash", 108, 192, 1, {"trend": trend})
    drift = float(s.series[-1] - s.series[0])
    if trend == "up":
        assert drift > 0
    elif trend == "down":
        assert drift < 0


def test_edit_timeline_ripple_closes_gaps():
    s = make_scene("edit_timeline", 108, 192, 1)
    c, g = s.cut_c[0], s.cut_g[0]
    after = c + g + 0.5
    e0 = s.ripple(s.cut_tc[0] - 0.1)
    e1 = s.ripple(s.cut_tc[0] + 2.0)
    assert np.isclose(s.disp(np.array([after]), e0)[0], after)
    assert np.isclose(s.disp(np.array([after]), e1)[0], after - g)
