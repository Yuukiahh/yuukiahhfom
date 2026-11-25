"""Overlay (motion graphics) layer tests."""
import numpy as np

from factory import overlays, scriptwriter

SAMPLES = {
    "headline": {"kicker": "BREAKING", "text": "Board reinstates CEO", "source": "CNBC", "date": "Nov 23"},
    "timeline": {"events": [{"date": "Nov 12", "label": "Resigns"}, {"date": "Nov 23", "label": "Back"}], "active": 1},
    "bars": {"title": "Stock", "unit": "%", "items": [{"label": "A", "value": 31}, {"label": "B", "value": -8}]},
    "counter": {"value": 31, "prefix": "+", "suffix": "%", "label": "jump"},
    "quote": {"text": "Target of a foreign intelligence effort.", "source": "Board"},
    "person": {"name": "Jane Doe", "role": "CEO", "status": "BACK", "status_color": "green"},
    "checklist": {"title": "Facts", "items": ["one", "two"]},
    "compare": {"left": {"label": "Before", "value": "1"}, "right": {"label": "After", "value": "2"}},
}


def test_all_overlay_types_render_and_composite():
    frame = np.full((1920, 1080, 3), 20, np.uint8)
    for kind, data in SAMPLES.items():
        assert scriptwriter._valid_overlay({"type": kind, "data": data}), kind
        layer = overlays.OverlayLayer({"overlay": {"type": kind, "data": data}, "duration": 3.0,
                                       "tag": "TOPIC", "source_label": "CNBC", "lang": "en"}, 1080, 1920)
        out = layer.apply(frame.copy(), 1.5)
        assert out.shape == frame.shape and out.mean() > frame.mean()


def test_invalid_overlays_rejected():
    assert not scriptwriter._valid_overlay({"type": "bars", "data": {"items": [{"label": "x", "value": "lots"}]}})
    assert not scriptwriter._valid_overlay({"type": "nope", "data": {}})
    assert not scriptwriter._valid_overlay(None)


def test_overlay_hidden_outside_window():
    frame = np.zeros((1920, 1080, 3), np.uint8)
    layer = overlays.OverlayLayer({"overlay": {"type": "counter", "data": SAMPLES["counter"]}, "duration": 3.0,
                                   "overlay_start": 1.0, "overlay_end": 2.0}, 1080, 1920)
    assert layer.apply(frame.copy(), 0.5).max() == 0
    assert layer.apply(frame.copy(), 1.6).max() > 0
