"""True-3D scene smoke tests (small resolution, fast)."""
import numpy as np
import pytest

from factory.visuals import three_d
from factory.visuals.procedural import KINDS, make_scene

KINDS_NEW = ["galaxy3d", "warp3d", "globe3d", "dna3d", "neural3d", "terrain3d", "atom3d", "datacity3d", "ringplanet3d"]


def test_kinds_registered():
    for k in KINDS_NEW:
        assert k in KINDS and KINDS[k] is three_d.KINDS_3D[k]
    for alias, kind in three_d.KINDS_3D.items():
        assert KINDS[alias] is kind


@pytest.mark.parametrize("kind", KINDS_NEW)
def test_scene_renders_finite(kind):
    s = make_scene(kind, 108, 192, 3)
    assert s.dynamic
    for t in (0.0, 1.7, 40.0):
        f = s.render(t)
        assert f.shape == (s.H, s.W, 3)
        assert f.dtype == np.float32
        assert np.isfinite(f).all()
        assert f.min() >= -1e-3 and f.max() < 600  # compositor clips; tone curve keeps light near 0..255
    assert f.max() > 5  # something visible


@pytest.mark.parametrize("kind", KINDS_NEW)
def test_scene_is_deterministic_in_t(kind):
    a = make_scene(kind, 108, 192, 5)
    b = make_scene(kind, 108, 192, 5)
    fa, fb = a.render(2.25), b.render(2.25)
    assert np.allclose(fa, fb)
    # rendering other times in between must not change the result (no hidden per-frame state)
    a.render(9.0)
    assert np.allclose(a.render(2.25), fa)


def test_camera_projection_center():
    r = three_d.Renderer(120, 214)
    r.begin(three_d.Camera((0, 0, -5), (0, 0, 0), fov=50))
    x, y, z, inv = r.project(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    assert abs(x[0] - 60) < 1e-3 and abs(y[0] - 107) < 1e-3 and abs(z[0] - 5) < 1e-5
    assert x[1] != x[0]  # off-axis point moves horizontally
