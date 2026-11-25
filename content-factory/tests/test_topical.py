"""Topical (news / business / tech / politics) background scenes: registry, shapes, determinism."""
import numpy as np
import pytest

from factory.visuals import topical
from factory.visuals.procedural import KINDS, make_scene

KINDS_NEW = ["gpu_chip", "stock_chart", "ticker_wall", "corporate", "newsroom", "radar", "capitol", "datacenter",
             "calendar", "courtroom"]
ALIASES = {"gpu": "gpu_chip", "semiconductor": "gpu_chip", "stocks": "stock_chart", "market": "stock_chart",
           "wall_street": "ticker_wall", "skyline": "corporate", "ceo": "corporate", "news": "newsroom",
           "breaking": "newsroom", "spy": "radar", "surveillance": "radar", "government": "capitol",
           "white_house": "capitol", "data_center": "datacenter", "ai_compute": "datacenter", "deadline": "calendar",
           "date": "calendar", "court": "courtroom", "lawsuit": "courtroom"}


def test_kinds_registered():
    for k in KINDS_NEW:
        assert KINDS[k] is topical.KINDS_TOPICAL[k]
    for alias, kind in ALIASES.items():
        assert KINDS[alias] is KINDS[kind]
    for alias, cls in topical.KINDS_TOPICAL.items():
        assert KINDS[alias] is cls


@pytest.mark.parametrize("kind", KINDS_NEW)
def test_scene_renders_finite(kind):
    s = make_scene(kind, 108, 192, 3, {"accent": "#76B900", "trend": "down", "highlight": 9})
    assert s.dynamic
    for t in (0.0, 1.7, 40.0):
        f = s.render(t)
        assert f.shape == (s.H, s.W, 3)
        assert f.dtype == np.float32
        assert np.isfinite(f).all()
        assert f.min() >= -1e-3 and f.max() < 700
    assert f.max() > 5


@pytest.mark.parametrize("kind", KINDS_NEW)
def test_scene_is_deterministic_in_t(kind):
    a = make_scene(kind, 108, 192, 5)
    b = make_scene(kind, 108, 192, 5)
    fa, fb = a.render(2.25), b.render(2.25)
    assert np.allclose(fa, fb)
    a.render(9.0)  # no hidden per-frame state
    assert np.allclose(a.render(2.25), fa)


@pytest.mark.parametrize("trend", ["up", "down", "volatile"])
def test_stock_trend_direction(trend):
    s = make_scene("stock_chart", 108, 192, 1, {"trend": trend})
    assert s.trend == trend
    drift = float(s.c[-1] - s.c[0])
    if trend == "up":
        assert drift > 0
    elif trend == "down":
        assert drift < 0


def test_accent_parsing():
    assert np.allclose(topical.hex_rgb("#76B900"), [0x76, 0xB9, 0x00])
    assert np.allclose(topical.hex_rgb("fff"), [255, 255, 255])
    assert np.allclose(topical.hex_rgb("not-a-colour", "#000000"), [0, 0, 0])


def test_ticker_symbols_are_generic():
    rng = np.random.default_rng(0)
    for _ in range(500):
        assert topical.fake_symbol(rng) not in topical._FAMOUS


def test_world_mask_has_continents():
    m = topical.world_mask(180, 70)
    assert 0.2 < m.mean() < 0.45
    u, v = topical.lonlat_uv(20, 5)  # central Africa is land
    assert m[int(v * 70), int(u * 180)]
    u, v = topical.lonlat_uv(-30, 0)  # mid Atlantic is water
    assert not m[int(v * 70), int(u * 180)]


def test_full_size_luma():
    s = make_scene("newsroom", 1080, 1920, 2)
    f = np.clip(s.render(1.0), 0, 255)
    assert (f[..., 0] * 0.299 + f[..., 1] * 0.587 + f[..., 2] * 0.114).mean() >= 25
