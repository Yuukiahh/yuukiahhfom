"""Collectible trading-card scene: floating 3D holographic cards (generic designs, no IP).

kind "holo_cards" (aliases: cards, trading_cards, tcg, card_pack, collectibles, pack_opening)
params: accent (hex), mode "float" | "fan" (cards fan out in an arc) | "burst" (pack-opening burst)
"""
from __future__ import annotations

import colorsys
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .procedural import KINDS, Scene, colorize, fbm, glow


def _hex(h: str, default=(199, 125, 255)):
    try:
        h = h.lstrip("#")
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return default


def _card_texture(w: int, h: int, rng: np.random.Generator, accent, rarity: int) -> Image.Image:
    """Generic holo card: foil gradient, border, abstract emblem (star / heart / rainbow arc), rarity pips."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    hue0 = rng.random()
    band = (xx / w * 0.8 + yy / h * 0.6 + hue0) % 1.0
    rgb = np.zeros((h, w, 3), np.float32)
    for i, hv in enumerate(np.linspace(0, 1, 7)):
        r, g, b = colorsys.hsv_to_rgb(hv, 0.62, 1.0)
        m = np.exp(-((band - hv) * 7) ** 2)
        rgb += m[..., None] * np.array([r, g, b]) * 255
    rgb = rgb / max(1e-3, rgb.max()) * 205
    sparkle = (rng.random((h, w)) > 0.995).astype(np.float32)
    rgb += sparkle[..., None] * 255
    im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(im)
    r = int(w * 0.07)
    # inner art window (pastel)
    pad = int(w * 0.09)
    d.rounded_rectangle([pad, pad, w - pad, int(h * 0.62)], r // 2, fill=(*[int(c * 0.35 + 150) for c in accent], 235))
    cx, cy, s = w / 2, h * 0.36, w * 0.22
    kind = rng.integers(0, 3)
    if kind == 0:  # star
        pts = [(cx + s * (1 if k % 2 == 0 else 0.45) * math.sin(k * math.pi / 5),
                cy - s * (1 if k % 2 == 0 else 0.45) * math.cos(k * math.pi / 5)) for k in range(10)]
        d.polygon(pts, fill=(255, 250, 230, 255))
    elif kind == 1:  # heart
        d.ellipse([cx - s, cy - s * 0.7, cx, cy + s * 0.2], fill=(255, 120, 190, 255))
        d.ellipse([cx, cy - s * 0.7, cx + s, cy + s * 0.2], fill=(255, 120, 190, 255))
        d.polygon([(cx - s * 0.97, cy - s * 0.1), (cx + s * 0.97, cy - s * 0.1), (cx, cy + s * 1.0)], fill=(255, 120, 190, 255))
    else:  # rainbow arc
        for k, col in enumerate([(255, 99, 132), (255, 170, 64), (255, 230, 90), (90, 220, 140), (90, 160, 255), (170, 110, 255)]):
            rr = s * (1.2 - k * 0.12)
            d.arc([cx - rr, cy - rr * 0.6, cx + rr, cy + rr * 1.4], 180, 360, fill=(*col, 255), width=max(3, int(s * 0.1)))
    # text-less info bars + rarity pips
    for k in range(3):
        y = int(h * 0.68) + k * int(h * 0.07)
        d.rounded_rectangle([pad, y, w - pad - (k * w * 0.12), y + int(h * 0.035)], 6, fill=(255, 255, 255, 140))
    for k in range(rarity):
        x = w - pad - k * int(w * 0.09) - int(w * 0.06)
        d.ellipse([x, int(h * 0.9), x + int(w * 0.055), int(h * 0.9) + int(w * 0.055)], fill=(255, 215, 90, 255))
    # border + rounded alpha
    d.rounded_rectangle([0, 0, w - 1, h - 1], r, outline=(255, 236, 170, 255), width=max(4, w // 40))
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], r, fill=255)
    im.putalpha(mask)
    return im


def _quad_coeffs(dst, src):
    """Perspective coefficients mapping output (dst) quad -> input (src) rectangle (PIL PERSPECTIVE)."""
    A, B = [], []
    for (x, y), (u, v) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); B.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); B.append(v)
    return np.linalg.solve(np.array(A, np.float64), np.array(B, np.float64)).tolist()


class HoloCards(Scene):
    dynamic = True

    def build(self):
        W, H, rng = self.W, self.H, self.rng
        self.accent = _hex(str(self.p.get("accent", "#C77DFF")))
        self.mode = self.p.get("mode", "float")
        a = np.array(self.accent, np.float32)
        n = fbm(W, H, rng, 5, 2)
        yy = np.linspace(0, 1, H)[:, None]
        base = colorize(np.clip(n * 0.55 + (1 - yy) * 0.35, 0, 1),
                        [(0, (12, 6, 28)), (0.5, tuple(int(c * 0.28) for c in a)), (0.85, tuple(int(c * 0.6) for c in a)), (1, (255, 220, 250))])
        # pastel bokeh
        pil = Image.new("RGB", (W, H))
        d = ImageDraw.Draw(pil)
        for _ in range(70):
            x, y, r = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(8, 60)
            col = tuple(int(c) for c in (np.array(colorsys.hsv_to_rgb(rng.random(), 0.35, 1.0)) * 255 * rng.uniform(0.25, 0.6)))
            d.ellipse([x - r, y - r, x + r, y + r], fill=col)
        self.bg = base + np.asarray(pil.filter(ImageFilter.GaussianBlur(10)), np.float32) * 0.8
        self.cw, self.ch = int(W * 0.34), int(W * 0.34 * 1.4)
        self.cards = [_card_texture(self.cw, self.ch, rng, self.accent, int(rng.integers(1, 6))) for _ in range(7)]
        self.phase = rng.random((7, 4)) * 6.28
        self.sparks = rng.random((260, 4)).astype(np.float32)

    def _place(self, i: int, t: float):
        n = len(self.cards)
        W, H = self.W, self.H
        p = self.phase[i]
        if self.mode == "fan":
            ang = (i - (n - 1) / 2) * 0.22 + 0.05 * math.sin(t * 0.8 + p[0])
            cx, cy, z = W / 2 + math.sin(ang) * W * 0.32, H * 0.5 - math.cos(ang) * H * 0.05 + abs(i - 3) * H * 0.02, 1.0 + abs(i - 3) * 0.08
            rot_y, rot_z = 0.25 * math.sin(t * 0.7 + p[1]), ang
        elif self.mode == "burst":
            k = min(1.0, t / 1.2)
            e = 1 - (1 - k) ** 3
            dirx, diry = math.cos(p[0]), math.sin(p[0]) * 0.8
            cx, cy = W / 2 + dirx * W * 0.38 * e, H * 0.5 + diry * H * 0.25 * e
            z = 0.9 + (p[2] / 6.28) * 0.8
            rot_y, rot_z = t * (0.8 + p[3] / 6.28) + p[1], 0.3 * math.sin(t + p[2])
        else:  # float
            ox = ((i * 0.37 + 0.13) % 1.0)
            cx = W * (0.12 + 0.76 * ox) + W * 0.04 * math.sin(t * 0.5 + p[0])
            cy = H * (0.15 + 0.7 * ((i * 0.61 + 0.3) % 1.0)) + H * 0.03 * math.sin(t * 0.6 + p[1]) - t * H * 0.012
            z = 0.8 + 0.6 * ((i * 0.53) % 1.0)
            rot_y, rot_z = 0.9 * math.sin(t * 0.6 + p[2]), 0.25 * math.sin(t * 0.4 + p[3])
        return cx, cy, z, rot_y, rot_z

    def render(self, t):
        W, H = self.W, self.H
        out = self.bg.copy()
        order = sorted(range(len(self.cards)), key=lambda i: self._place(i, t)[2])
        for i in order:
            cx, cy, z, ry, rz = self._place(i, t)
            w2, h2 = self.cw * z / 2, self.ch * z / 2
            sx = max(0.08, abs(math.cos(ry)))
            persp = 0.18 * math.sin(ry)
            corners = [(-w2 * sx, -h2 * (1 + persp)), (w2 * sx, -h2 * (1 - persp)), (w2 * sx, h2 * (1 - persp)), (-w2 * sx, h2 * (1 + persp))]
            cr, sr = math.cos(rz), math.sin(rz)
            dst = [(cx + x * cr - y * sr, cy + x * sr + y * cr) for x, y in corners]
            xs, ys = [p[0] for p in dst], [p[1] for p in dst]
            x0, y0, x1, y1 = int(min(xs)) - 2, int(min(ys)) - 2, int(max(xs)) + 2, int(max(ys)) + 2
            if x1 <= 0 or y1 <= 0 or x0 >= W or y0 >= H:
                continue
            local = [(x - x0, y - y0) for x, y in dst]
            src = [(0, 0), (self.cw, 0), (self.cw, self.ch), (0, self.ch)]
            warped = np.asarray(self.cards[i].transform((x1 - x0, y1 - y0), Image.PERSPECTIVE, _quad_coeffs(local, src),
                                                        Image.BILINEAR), np.float32)
            shine = 0.8 + 0.3 * (0.5 + 0.5 * math.sin(ry * 2 + t))
            rgb, al = warped[..., :3] * shine, warped[..., 3:4] / 255
            # clip to frame
            ax0, ay0, ax1, ay1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            sl = (slice(ay0 - y0, ay1 - y0), slice(ax0 - x0, ax1 - x0))
            # cheap drop shadow (offset, darkened)
            sx0, sy0 = ax0 + 16, ay0 + 22
            sx1, sy1 = min(W, ax1 + 16), min(H, ay1 + 22)
            if sx1 > sx0 and sy1 > sy0:
                sa = al[sl][: sy1 - sy0, : sx1 - sx0] * 0.45
                out[sy0:sy1, sx0:sx1] *= 1 - sa
            reg = out[ay0:ay1, ax0:ax1]
            out[ay0:ay1, ax0:ax1] = reg * (1 - al[sl]) + rgb[sl] * al[sl]
        # floating sparkles with glow (computed at 1/4 resolution)
        sp = self.sparks
        hs, ws = H // 4, W // 4
        xs = ((sp[:, 0] + t * 0.02 * (sp[:, 2] - 0.5)) % 1) * ws
        ys = ((sp[:, 1] - t * 0.03 * (0.5 + sp[:, 3])) % 1) * hs
        tw = (np.sin(t * 6 * sp[:, 2] + sp[:, 3] * 20) * 0.5 + 0.5) ** 3
        spark = np.zeros((hs, ws), np.float32)
        np.add.at(spark, (ys.astype(int).clip(0, hs - 1), xs.astype(int).clip(0, ws - 1)), tw * 255)
        sp_img = Image.fromarray(np.clip(spark, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2)).resize((W, H), Image.BILINEAR)
        out += np.asarray(sp_img, np.float32)[..., None] * np.array([1.6, 1.5, 1.3], np.float32)
        return out


for _k in ("holo_cards", "cards", "trading_cards", "tcg", "card_pack", "collectibles", "pack_opening"):
    KINDS[_k] = HoloCards
