"""Cinematic finishing pass per frame (applied after grade, before overlays).

light leaks (drifting accent/warm glows, screen-blended), anamorphic streaks on highlights,
halation, foreground bokeh particles with parallax, subtle chromatic aberration, and a
rack-focus depth-of-field that softens the background while an overlay card is on screen.
Everything heavy runs at 1/4-1/8 resolution and is upsampled -> ~25-40 ms per frame.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _hex(h: str | None, default=(124, 58, 237)):
    try:
        h = str(h).lstrip("#")
        return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32)
    except Exception:
        return np.array(default, np.float32)


def _up(a: np.ndarray, w: int, h: int) -> np.ndarray:
    if a.ndim == 2:
        return np.asarray(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).resize((w, h), Image.BILINEAR), np.float32)
    return np.asarray(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).resize((w, h), Image.BILINEAR), np.float32)


class CineFX:
    def __init__(self, spec: dict, w: int, h: int, seed: int):
        self.w, self.h = w, h
        self.rng = np.random.default_rng(seed + 777)
        self.accent = _hex((spec.get("params") or {}).get("accent") or spec.get("accent"))
        self.strength = float(spec.get("cine_strength", 1.0))
        self.dof = spec.get("overlay") is not None and spec.get("dof", True)
        self.ov0 = float(spec.get("overlay_start", 0.35))
        self.ov1 = float(spec.get("overlay_end", spec["duration"]))
        # light-leak blobs (low res)
        self.lw, self.lh = w // 8, h // 8
        yy, xx = np.mgrid[0:self.lh, 0:self.lw].astype(np.float32)
        self.yy, self.xx = yy / self.lh, xx / self.lw
        self.leaks = [(self.rng.uniform(-0.2, 1.2), self.rng.uniform(-0.1, 1.1), self.rng.uniform(0.25, 0.45),
                       self.rng.uniform(0, 6.28), col) for col in (self.accent, np.array([255, 150, 70], np.float32))]
        # bokeh particles (quarter res)
        self.bw, self.bh = w // 4, h // 4
        n = 16
        self.bok = np.stack([self.rng.uniform(0, 1, n), self.rng.uniform(0, 1, n), self.rng.uniform(0.3, 1.0, n),
                             self.rng.uniform(0, 6.28, n)], 1)
        spr = Image.new("L", (64, 64), 0)
        ImageDraw.Draw(spr).ellipse([6, 6, 58, 58], fill=255)
        self.sprite = np.asarray(spr.filter(ImageFilter.GaussianBlur(5)), np.float32) / 255
        base = Image.fromarray((self.sprite * 255).astype(np.uint8))
        self.sprites = [np.asarray(base.resize((int(18 + 40 * d),) * 2), np.float32) / 255 for d in self.bok[:, 2]]

    def _leaks(self, t: float) -> np.ndarray:
        acc = np.zeros((self.lh, self.lw, 3), np.float32)
        for cx, cy, r, ph, col in self.leaks:
            x = cx + 0.18 * math.sin(t * 0.35 + ph)
            y = cy + 0.12 * math.cos(t * 0.27 + ph)
            inten = 0.5 + 0.5 * math.sin(t * 0.6 + ph * 2)
            m = np.exp(-(((self.xx - x) / r) ** 2 + ((self.yy - y) / (r * 1.4)) ** 2) * 2.2)
            acc += m[..., None] * col * (0.18 + 0.22 * inten)
        return acc

    def _bokeh(self, t: float, cam_dx: float, cam_dy: float) -> np.ndarray:
        canvas = np.zeros((self.bh, self.bw), np.float32)
        for (x0, y0, depth, ph), spr in zip(self.bok, self.sprites):
            size = spr.shape[0]
            x = ((x0 + 0.02 * t * depth - cam_dx * 0.08 * depth) % 1.0) * self.bw
            y = ((y0 - 0.015 * t * depth - cam_dy * 0.08 * depth) % 1.0) * self.bh
            a = (0.10 + 0.12 * (0.5 + 0.5 * math.sin(t * 1.3 + ph))) * depth
            xi, yi = int(x - size / 2), int(y - size / 2)
            xa, ya, xb, yb = max(0, xi), max(0, yi), min(self.bw, xi + size), min(self.bh, yi + size)
            if xb > xa and yb > ya:
                canvas[ya:yb, xa:xb] += spr[ya - yi:yb - yi, xa - xi:xb - xi] * a
        return canvas

    def apply(self, f: np.ndarray, t: float, cam_dx: float = 0.0, cam_dy: float = 0.0) -> np.ndarray:
        """f: float32/int HxWx3 -> uint8 HxWx3. All full-res work happens in PIL's C routines."""
        from PIL import ImageChops

        k = self.strength
        h, w = self.h, self.w
        img = Image.fromarray(np.clip(f, 0, 255).astype(np.uint8))
        small = img.reduce(8)  # fast box downsample (w/8 x h/8)
        sm = np.asarray(small, np.float32)
        # depth of field (rack focus while an overlay card is visible)
        if self.dof:
            b = min(1.0, max(0.0, (t - self.ov0 + 0.1) / 0.3)) * min(1.0, max(0.0, (self.ov1 - t) / 0.3))
            if b > 0.01:
                bl = img.reduce(4).filter(ImageFilter.GaussianBlur(3)).resize((w, h), Image.BILINEAR)
                img = Image.blend(img, bl, 0.55 * b)
        mean = float(sm.mean())
        adapt = float(np.clip((150 - mean) / 90, 0.15, 1.0))
        low = self._leaks(t) * k * adapt
        hi = np.clip(sm.max(axis=2) - 225, 0, 30) / 30
        if hi.any():
            L = 41
            c = np.cumsum(np.pad(hi, ((0, 0), (L, L)), mode="constant"), axis=1)
            streak = (c[:, L + L // 2: L + L // 2 + hi.shape[1]] - c[:, L // 2: L // 2 + hi.shape[1]]) / L * 2.2
            low = low + streak[..., None] * np.array([110, 190, 255], np.float32) * 0.35 * k * adapt
        lw, lh = small.size
        glow = Image.fromarray(np.clip(low[:lh, :lw], 0, 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
        img = ImageChops.screen(img, glow)
        bo = self._bokeh(t, cam_dx, cam_dy)
        if bo.any():
            col = (self.accent * 0.5 + 255 * 0.5) * k * adapt
            bimg = Image.fromarray(np.clip(np.clip(bo, 0, 1)[..., None] * col, 0, 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
            img = ImageChops.add(img, bimg)
        r, g, b_ = img.split()
        img = Image.merge("RGB", (ImageChops.offset(r, 2, 0), g, ImageChops.offset(b_, -2, 0)))
        return np.array(img)  # writable copy (overlays draw into it)
