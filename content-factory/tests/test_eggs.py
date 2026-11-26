"""Egg-heist / pet-tycoon game background scenes: registry, shapes, brightness, determinism, motion, params."""
import numpy as np
import pytest

from factory.visuals import creator, eggs
from factory.visuals.procedural import KINDS, make_scene

KINDS_NEW = ["egg_nest", "egg_heist", "egg_hatch", "pet_base"]


def luma(f):
    f = np.clip(f, 0, 255)
    return float((f[..., 0] * 0.299 + f[..., 1] * 0.587 + f[..., 2] * 0.114).mean())


def test_kinds_registered():
    for k in KINDS_NEW:
        assert KINDS[k] is eggs.KINDS_EGGS[k]
    for alias, cls in eggs.KINDS_EGGS.items():
        assert KINDS[alias] is cls
    # never hijacks an existing creator alias
    for alias, cls in creator.KINDS_CREATOR.items():
        assert KINDS[alias] is cls


@pytest.mark.parametrize("kind", KINDS_NEW)
def test_scene_renders_finite(kind):
    s = make_scene(kind, 108, 192, 3, {"accent": "#FF5FA2"})
    assert s.dynamic
    for t in (0.0, 1.7, 2.7, 40.0):
        f = s.render(t)
        assert f.shape == (s.H, s.W, 3)
        assert f.dtype == np.float32
        assert np.isfinite(f).all()
        assert f.min() >= -1e-3 and f.max() < 700


@pytest.mark.parametrize("kind", KINDS_NEW)
def test_scene_is_deterministic_and_animated(kind):
    a = make_scene(kind, 108, 192, 5)
    b = make_scene(kind, 108, 192, 5)
    fa, fb = a.render(2.25), b.render(2.25)
    assert np.allclose(fa, fb)
    a.render(9.0)  # no hidden per-frame state
    assert np.allclose(a.render(2.25), fa)
    for t in (0.5, 2.0, 3.5):  # lots of motion: half a second later the frame differs noticeably
        assert np.abs(a.render(t + 0.5) - a.render(t)).mean() > 3.0


@pytest.mark.parametrize("kind", KINDS_NEW)
def test_bright_and_playful(kind):
    s = make_scene(kind, 270, 480, 2, {"accent": "#3FA9FF"})
    for t in (0.5, 2.0, 3.5):
        f = s.render(t)
        assert luma(f) >= 45
        hsv_sat = (f.max(-1) - f.min(-1)) / np.maximum(f.max(-1), 1)
        assert float(hsv_sat.mean()) > 0.12  # colourful, not grey


@pytest.mark.parametrize("bad", [None, "abc", -5, 2.0, float("nan")])
def test_params_are_tolerant(bad):
    for kind in KINDS_NEW:
        s = make_scene(kind, 108, 192, 1, {"rarity": bad, "speed": bad, "income": bad, "accent": bad})
        assert np.isfinite(s.render(1.0)).all()


@pytest.mark.parametrize("rarity", ["common", "rare", "epic", "legendary"])
def test_rarity_param(rarity):
    n = make_scene("egg_nest", 108, 192, 1, {"rarity": rarity})
    assert n.hero == rarity and n.nests[0][3] == rarity
    assert sorted(r for *_, r in n.nests[:4]) == sorted(eggs.RARITIES)  # every rarity on the island
    h = make_scene("egg_hatch", 108, 192, 1, {"rarity": rarity})
    assert h.rarity == rarity
    assert np.allclose(h.g, eggs.EGG_PAL[rarity][3])
    assert eggs.parse_rarity("GOLD") == "legendary" and eggs.parse_rarity("purple") == "epic"
    assert eggs.parse_rarity("???", "rare") == "rare"


def test_hatch_rarity_changes_colours():
    a = make_scene("egg_hatch", 108, 192, 1, {"rarity": "rare"}).render(3.5)
    b = make_scene("egg_hatch", 108, 192, 1, {"rarity": "legendary"}).render(3.5)
    assert a[..., 2].mean() > b[..., 2].mean()  # blue vs gold


def test_hatch_loops_cleanly():
    s = make_scene("egg_hatch", 108, 192, 4)
    T = s.T
    # same phase in the next cycle: same egg / burst state (pet may differ, camera drifts slowly)
    assert np.abs(s.render(0.3) - s.render(0.3 + T)).mean() < np.abs(s.render(0.3) - s.render(3.3)).mean()
    # the burst flashes
    assert luma(s.render(s.TB + 0.02)) > luma(s.render(s.TB - 0.3))


def test_heist_speed_param():
    slow = make_scene("egg_heist", 108, 192, 1, {"speed": 0.0})
    fast = make_scene("egg_heist", 108, 192, 1, {"speed": 1.0})
    assert slow.speed == 1.0 and fast.speed == 1.0  # never used as a time multiplier
    assert fast.v > slow.v and fast.freq > slow.freq and len(fast.sl) > len(slow.sl)
    assert np.abs(slow.render(1.5) - slow.render(2.0)).mean() > 3.0  # even speed 0 keeps running


def test_pet_base_income_param_and_counter():
    lo = make_scene("pet_base", 108, 192, 1, {"income": 0.0})
    hi = make_scene("pet_base", 108, 192, 1, {"income": 1.0})
    assert (hi.pop_p < lo.pop_p).all()
    assert len(lo.pets) == lo.NPET == 9


def test_motion_helpers():
    y, sx, sy = eggs.hop(0.0, 1.0, 1.0)
    assert y == 0.0
    y, sx, sy = eggs.hop(0.31, 1.0, 1.0)
    assert y > 0.9 and sy >= 1.0  # airborne, stretched
    y, sx, sy = eggs.hop(0.81, 1.0, 1.0)
    assert y == 0.0 and sx > 1.0 > sy  # landing squash
    assert eggs.elastic(0) == 0 and eggs.elastic(1) == 1
    assert eggs.back_out(1.0) == pytest.approx(1.0)
    assert eggs.hsh(3, 1) == eggs.hsh(3, 1) and 0 <= eggs.hsh(7) < 1


def test_sprites():
    rng = np.random.default_rng(0)
    for r in eggs.RARITIES:
        e = eggs.egg_sprite(64, r, rng)
        a = np.asarray(e)
        assert e.mode == "RGBA" and a[..., 3].max() == 255 and a[0, 0, 3] == 0
    for k in range(7):
        p = eggs.pet_sprite(48, (255, 150, 190), k, rng, blink=k % 2 == 1)
        assert p.mode == "RGBA" and np.asarray(p)[..., 3].max() == 255
