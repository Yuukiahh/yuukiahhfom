import numpy as np

from factory.overlays import RENDER, OverlayLayer
from factory.scriptwriter import _valid_overlay

SAMPLES = {
    "steps": {"n": 2, "total": 5, "title": "Write the script", "detail": "Hook in 1.5 seconds"},
    "notify": {"items": [{"app": "Creator Fund", "title": "Payout sent", "text": "1.2M views", "time": "now"}]},
    "calc": {"terms": [{"value": "1,000,000", "label": "views"}, {"value": "$0.40", "label": "per 1K"}], "ops": ["×"], "result": "$400"},
    "prompt": {"tool": "Image model", "text": "Cinematic close-up of a card"},
    "stack": {"title": "Tools", "items": [{"name": "Kokoro", "role": "voice", "price": "FREE"}]},
    "meter": {"label": "Followers", "value": 6500, "max": 10000},
}


def test_new_types_render_and_validate():
    for k, d in SAMPLES.items():
        assert _valid_overlay({"type": k, "data": d}), k
        for t in (0.0, 0.5, 2.0):
            im = RENDER[k](d, t, 3.0, 1080)
            assert im.mode == "RGBA" and im.width <= 1080


def test_invalid_new_types_rejected():
    assert not _valid_overlay({"type": "meter", "data": {"value": 1, "max": 0}})
    assert not _valid_overlay({"type": "calc", "data": {"terms": [], "result": "$1"}})


def test_chips_and_tall_cards_stay_above_captions():
    W, H = 1080, 1920
    ov = {"type": "stack", "data": {"title": "T", "items": [{"name": f"Tool {i}", "role": "r"} for i in range(4)]}, "chips": ["a", "b", "c"]}
    L = OverlayLayer({"overlay": ov, "duration": 3.0, "overlay_start": 0.0, "overlay_y": 0.36}, W, H)
    f = np.zeros((H, W, 3), np.uint8)
    L.apply(f, 2.0)
    rows = np.where(f.max(axis=(1, 2)) > 0)[0]
    assert rows.max() < H * 0.575
