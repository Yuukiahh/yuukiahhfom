"""Original procedural visuals (no licences needed – everything is generated).

Every scene class renders frames at layer resolution (output * OVERSCAN) so the
camera can push/pan without upscaling. Static parts are precomputed once; only
cheap per-frame work (texture rotation, scrolling, pulses) happens in render(t).
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OVERSCAN = 1.12


# ------------------------------------------------------------------ helpers
def fbm(w: int, h: int, rng: np.random.Generator, octaves: int = 6, base: int = 3, persistence: float = 0.55) -> np.ndarray:
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        gw, gh = base * 2 ** o + 1, int((base * 2 ** o) * h / w) + 1
        grid = rng.random((gh, gw)).astype(np.float32)
        layer = np.asarray(Image.fromarray(grid).resize((w, h), Image.BICUBIC), np.float32)
        out += layer * amp
        total += amp
        amp *= persistence
    out /= total
    out -= out.min()
    return out / max(1e-6, out.max())


def colorize(v: np.ndarray, stops: list[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    xs = [s[0] for s in stops]
    out = np.zeros(v.shape + (3,), np.float32)
    for c in range(3):
        out[..., c] = np.interp(v, xs, [s[1][c] for s in stops])
    return out


def glow(img: np.ndarray, radius: float, strength: float) -> np.ndarray:
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    blurred = np.asarray(pil.filter(ImageFilter.GaussianBlur(radius)), np.float32)
    return img + blurred * strength


def screen(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 255 - (255 - a) * (255 - b) / 255


def starfield(w: int, h: int, rng: np.random.Generator, density: float = 0.0009, big: float = 0.00004) -> np.ndarray:
    img = np.zeros((h, w, 3), np.float32)
    n = int(w * h * density)
    ys, xs = rng.integers(0, h, n), rng.integers(0, w, n)
    b = rng.power(3, n) * 255
    tint = rng.choice(np.array([[1, 1, 1], [0.8, 0.9, 1], [1, 0.9, 0.75], [0.75, 0.85, 1]]), n)
    img[ys, xs] = tint * b[:, None]
    img = glow(img, 1.2, 0.8)
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(pil)
    for _ in range(int(w * h * big)):
        x, y = rng.integers(0, w), rng.integers(0, h)
        r = rng.uniform(1.5, 3.2)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255))
        d.line([x - r * 6, y, x + r * 6, y], fill=(200, 220, 255), width=1)
        d.line([x, y - r * 6, x, y + r * 6], fill=(200, 220, 255), width=1)
    return glow(np.asarray(pil, np.float32), 3, 0.9)


PALETTES = {
    "violet": [(0, (2, 2, 10)), (0.45, (25, 10, 60)), (0.7, (120, 40, 160)), (0.88, (255, 120, 180)), (1, (255, 230, 240))],
    "teal":   [(0, (1, 4, 10)), (0.45, (5, 30, 50)), (0.7, (20, 120, 140)), (0.88, (120, 230, 220)), (1, (240, 255, 255))],
    "ember":  [(0, (5, 2, 2)), (0.45, (50, 10, 5)), (0.7, (170, 50, 20)), (0.88, (255, 160, 60)), (1, (255, 240, 200))],
    "ice":    [(0, (2, 3, 8)), (0.45, (10, 20, 60)), (0.7, (40, 90, 200)), (0.88, (150, 200, 255)), (1, (240, 250, 255))],
}


class Scene:
    dynamic = False

    def __init__(self, w: int, h: int, seed: int = 0, **params):
        self.W, self.H = int(w * OVERSCAN) // 2 * 2, int(h * OVERSCAN) // 2 * 2
        self.ow, self.oh = w, h
        self.rng = np.random.default_rng(seed)
        self.p = params
        self.build()

    def build(self):
        pass

    def render(self, t: float) -> np.ndarray:  # float32 HxWx3 0..255
        return self.base


# ------------------------------------------------------------------ space
class Nebula(Scene):
    def build(self):
        pal = PALETTES.get(self.p.get("palette", "violet"), PALETTES["violet"])
        n = fbm(self.W, self.H, self.rng, 7, 2)
        n2 = fbm(self.W, self.H, self.rng, 5, 3)
        yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        cx, cy = self.W * self.rng.uniform(0.35, 0.65), self.H * self.rng.uniform(0.3, 0.6)
        core = np.exp(-(((xx - cx) / (self.W * 0.55)) ** 2 + ((yy - cy) / (self.H * 0.4)) ** 2))
        v = np.clip((n * 0.65 + n2 * 0.35) * (0.6 + 0.8 * core), 0, 1) ** 1.35
        neb = colorize(v, pal)
        self.base = screen(neb, starfield(self.W, self.H, self.rng, 0.0006, 0.00002))
        self.near = starfield(self.W, self.H, self.rng, 0.00012, 0.00003)

    dynamic = True

    def render(self, t):
        shift = int(t * self.W * 0.018)
        near = np.roll(self.near, shift, axis=1)
        return screen(self.base, near)


class BlackHole(Scene):
    """Lensed starfield + Interstellar-style accretion disk with Keplerian rotation."""
    dynamic = True

    def build(self):
        W, H = self.W, self.H
        rng = self.rng
        bg_scene = Nebula(self.ow, self.oh, int(rng.integers(1 << 30)), palette=self.p.get("palette", "ember"))
        bg = bg_scene.base * 0.55
        self.cx, self.cy = W * 0.5, H * self.p.get("cy", 0.44)
        rs = W * self.p.get("size", 0.15)
        self.rs = rs
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        dx, dy = xx - self.cx, yy - self.cy
        r = np.sqrt(dx * dx + dy * dy) + 1e-3
        theta_e = rs * 1.9
        scale = (r - theta_e ** 2 / r) / r
        sx = np.clip(self.cx + dx * scale, 0, W - 1).astype(np.int32)
        sy = np.clip(self.cy + dy * scale, 0, H - 1).astype(np.int32)
        lensed = bg[sy, sx]
        shadow = np.clip((r - rs) / (rs * 0.06), 0, 1)
        self.base = lensed * shadow[..., None]
        # --- disk geometry
        ratio = self.p.get("tilt", 0.2)
        R = np.sqrt(dx ** 2 + (dy / ratio) ** 2)
        phi = np.arctan2(dy / ratio, dx)
        r_in, r_out = rs * 1.45, rs * 4.3
        disk_mask = ((R > r_in) & (R < r_out)).astype(np.float32)
        edge = np.clip((R - r_in) / (rs * 0.25), 0, 1) * np.clip((r_out - R) / (rs * 1.4), 0, 1)
        front = (dy > 0) | (r > rs * 1.02)
        self.disk_w = disk_mask * edge * front
        # lensed far side: arc above (and faint below) the shadow
        arc_r = (r - rs * 1.0) / (rs * 0.9)
        arc = np.clip(1 - np.abs(arc_r - 0.45) / 0.55, 0, 1) ** 1.5 * (r > rs)
        updown = np.where(dy < 0, 1.0, 0.35)
        sidefade = np.clip(np.abs(dy) / (r + 1e-3), 0, 1) ** 0.6
        self.arc_w = arc * updown * sidefade * (1 - np.clip(self.disk_w * 1.5, 0, 1))
        self.arc_R = r_in + np.clip(arc_r, 0, 1) * (r_out - r_in) * 0.6
        ring = np.exp(-((r - rs * 1.035) / (rs * 0.014)) ** 2) * 0.8 + np.exp(-((r - rs * 1.05) / (rs * 0.08)) ** 2) * 0.15
        self.ring = ring
        # texture lookup tables
        self.nR, self.nP = 256, 1024
        Ridx = np.clip((R - r_in) / (r_out - r_in), 0, 0.999)
        self.Ri = (Ridx * self.nR).astype(np.int32)
        self.Pi = (((phi + np.pi) / (2 * np.pi)) * self.nP).astype(np.int32) % self.nP
        aR = np.clip((self.arc_R - r_in) / (r_out - r_in), 0, 0.999)
        self.aRi = (aR * self.nR).astype(np.int32)
        self.aPi = (((np.arctan2(dy, dx) + np.pi) / (2 * np.pi)) * self.nP).astype(np.int32) % self.nP
        # disk texture: temperature gradient x angular streaks
        rr = np.linspace(0, 1, self.nR)[:, None]
        streak = fbm(self.nP, self.nR, rng, 6, 4)
        streak = np.asarray(Image.fromarray(streak).resize((self.nP, self.nR)), np.float32)
        temp = colorize(np.clip(1 - rr[:, 0], 0, 1)[:, None].repeat(self.nP, 1),
                        [(0, (40, 5, 0)), (0.35, (200, 60, 10)), (0.7, (255, 170, 70)), (0.92, (255, 235, 200)), (1, (255, 255, 255))])
        bright = (0.35 + 0.9 * streak ** 1.4) * np.exp(-rr * 1.6)
        ph = np.linspace(-np.pi, np.pi, self.nP)[None, :]
        doppler = 0.55 + 0.75 * (0.5 + 0.5 * np.cos(ph - np.pi))  # left side approaching -> brighter
        self.tex = temp * (bright * doppler)[..., None] * 1.8
        self.omega = (1.0 / (0.3 + np.linspace(0, 1, self.nR)) ** 1.5) * 0.12 * self.nP / (2 * np.pi)
        self.shadow = shadow

    def render(self, t):
        shift = (self.omega * t).astype(np.int32)  # per radius
        P = (self.Pi + shift[self.Ri]) % self.nP
        disk = self.tex[self.Ri, P] * self.disk_w[..., None]
        aP = (self.aPi + shift[self.aRi]) % self.nP
        arc = self.tex[self.aRi, aP] * self.arc_w[..., None] * 1.25
        ring = self.ring[..., None] * np.array([255, 220, 180], np.float32) * 1.1
        img = self.base + disk + arc + ring
        return img


class Planet(Scene):
    def build(self):
        W, H, rng = self.W, self.H, self.rng
        stars = starfield(W, H, rng, 0.0007)
        kind = self.p.get("kind", "gas")
        rad = W * self.p.get("size", 0.42)
        cx, cy = W * self.p.get("cx", 0.5), H * self.p.get("cy", 0.55)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        nx, ny = (xx - cx) / rad, (yy - cy) / rad
        d2 = nx ** 2 + ny ** 2
        inside = d2 < 1
        nz = np.sqrt(np.clip(1 - d2, 0, 1))
        lon = np.arctan2(nx, nz)
        lat = np.arcsin(np.clip(ny, -1, 1))
        tex_w, tex_h = 1024, 512
        if kind == "gas":
            bands = fbm(tex_w, tex_h, rng, 6, 2)
            v = (np.sin(np.linspace(0, 18, tex_h))[:, None] * 0.35 + bands * 0.8)
            v = (v - v.min()) / (v.max() - v.min())
            tex = colorize(v, [(0, (90, 50, 30)), (0.4, (200, 140, 90)), (0.7, (240, 210, 170)), (1, (160, 90, 60))])
        else:
            h = fbm(tex_w, tex_h, rng, 7, 3)
            tex = colorize(h, [(0, (10, 30, 90)), (0.52, (20, 70, 150)), (0.55, (200, 190, 140)), (0.62, (40, 120, 50)),
                               (0.8, (90, 80, 60)), (1, (250, 250, 250))])
        self.tex = tex
        self.u0 = ((lon / np.pi + 1) / 2 * (tex_w / 2)).astype(np.float32)
        self.v = np.clip(((lat / (np.pi / 2) + 1) / 2 * (tex_h - 1)), 0, tex_h - 1).astype(np.int32)
        light = np.array([-0.6, -0.35, 0.72])
        light /= np.linalg.norm(light)
        shade = np.clip(nx * light[0] + ny * light[1] + nz * light[2], 0, 1) ** 0.8
        self.shade = (0.04 + 0.96 * shade) * inside
        r = np.sqrt(d2)
        atm = np.exp(-((r - 1) / 0.035) ** 2) * np.clip((nx * light[0] + ny * light[1]) / (r + 1e-3) + 0.35, 0, 1) ** 1.5
        self.atm = atm[..., None] * np.array(self.p.get("atmo", (120, 180, 255)), np.float32)
        self.stars = stars * (~inside)[..., None]
        self.tex_w = tex_w

    dynamic = True

    def render(self, t):
        u = ((self.u0 + t * 6) % self.tex_w).astype(np.int32)
        surf = self.tex[self.v, u] * self.shade[..., None]
        return self.stars + surf + self.atm


class Wormhole(Scene):
    dynamic = True

    def build(self):
        W, H = self.W, self.H
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        dx, dy = xx - W / 2, yy - H * 0.47
        r = np.sqrt(dx * dx + dy * dy) + 1
        self.depth = (W * 0.35) / r
        self.ang = (np.arctan2(dy, dx) / (2 * np.pi) + 0.5)
        self.fade = np.clip(1 - self.depth / 6, 0, 1) ** 1.2 * np.clip(r / (W * 0.05), 0, 1)
        n = fbm(512, 512, self.rng, 6, 4)
        wx = np.sin(np.linspace(0, np.pi, 512))[None, :] ** 2
        n = n * wx + np.roll(n, 256, axis=1) * (1 - wx)
        wy = np.sin(np.linspace(0, np.pi, 512))[:, None] ** 2
        n = n * wy + np.roll(n, 256, axis=0) * (1 - wy)
        n = (n - n.min()) / (n.max() - n.min())
        pal = PALETTES.get(self.p.get("palette", "ice"), PALETTES["ice"])
        self.tex = colorize(n ** 1.3, pal) * 1.4
        self.core = np.exp(-(r / (W * 0.06)) ** 2)[..., None] * np.array([230, 240, 255], np.float32)

    def render(self, t):
        u = ((self.ang + self.depth * 0.04 + t * 0.03) * 512).astype(np.int32) % 512
        v = ((self.depth * 0.35 + t * 0.9) * 512).astype(np.int32) % 512
        return self.tex[v, u] * self.fade[..., None] + self.core


# ------------------------------------------------------------------ tech / AI
class SynthGrid(Scene):
    dynamic = True

    def build(self):
        W, H = self.W, self.H
        self.horizon = int(H * 0.46)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        below = yy > self.horizon
        dy = np.maximum(yy - self.horizon, 1)
        self.z = (H * 0.25) / dy
        self.x = (xx - W / 2) / dy * 0.9
        self.below = below
        sky = colorize(np.clip(1 - yy / self.horizon, 0, 1), [(0, (140, 20, 120)), (0.25, (50, 10, 70)), (1, (3, 2, 12))])
        stars = starfield(W, H, self.rng, 0.0004)
        sky = screen(sky, stars) * (~below)[..., None]
        # sun
        sun_r = W * 0.22
        d = np.sqrt((xx - W / 2) ** 2 + (yy - self.horizon + sun_r * 0.35) ** 2)
        sun = (d < sun_r) & (yy < self.horizon)
        stripes = ((yy - (self.horizon - sun_r)) / sun_r * 9) % 1 > np.clip((yy - (self.horizon - sun_r * 0.6)) / (sun_r * 0.6), 0, 0.6)
        sungrad = colorize(np.clip((yy - (self.horizon - sun_r * 1.35)) / (sun_r * 1.3), 0, 1),
                           [(0, (255, 230, 90)), (0.6, (255, 90, 120)), (1, (220, 30, 160))])
        sky = np.where((sun & stripes)[..., None], sungrad, sky)
        self.sky = glow(sky, 25, 0.6)
        self.fog = np.exp(-dy / (H * 0.06))[..., None] * below[..., None] * np.array([255, 60, 200], np.float32) * 0.5

    def render(self, t):
        gz = np.abs(((self.z + t * 1.2) % 1) - 0.5)
        gx = np.abs(((self.x * 4) % 1) - 0.5)
        width = 0.03 / np.maximum(self.z, 0.3) ** 0.2
        line = np.clip(1 - np.minimum(gz / (width * self.z ** 0.5 + 0.01), gx / (width * 2 + 0.01) * self.z ** -0.3) , 0, 1)
        line *= self.below * np.clip(1.4 - self.z * 0.08, 0, 1)
        col = line[..., None] * np.array([80, 220, 255], np.float32) * 1.3
        floor = np.array([12, 2, 25], np.float32) * self.below[..., None]
        return self.sky + floor + col + self.fog


class Circuit(Scene):
    dynamic = True

    def build(self):
        W, H, rng = self.W, self.H, self.rng
        pil = Image.new("RGB", (W, H), (2, 6, 12))
        d = ImageDraw.Draw(pil)
        step = W // 28
        self.paths = []
        accent = tuple(self.p.get("color", (0, 220, 255)))
        for _ in range(140):
            x, y = int(rng.integers(0, W // step)) * step, int(rng.integers(0, H // step)) * step
            pts = [(x, y)]
            for _ in range(int(rng.integers(3, 9))):
                if rng.random() < 0.5:
                    x += int(rng.integers(-6, 7)) * step
                else:
                    y += int(rng.integers(-6, 7)) * step
                pts.append((x, y))
            dim = tuple(int(c * 0.35) for c in accent)
            d.line(pts, fill=dim, width=3)
            for px, py in (pts[0], pts[-1]):
                d.ellipse([px - 7, py - 7, px + 7, py + 7], outline=accent, width=3)
            self.paths.append(pts)
        # central chip
        cw = W * 0.34
        x0, y0 = W / 2 - cw / 2, H * 0.45 - cw / 2
        d.rounded_rectangle([x0, y0, x0 + cw, y0 + cw], radius=24, fill=(8, 18, 30), outline=accent, width=5)
        for i in range(12):
            for side in (0, 1):
                xx = x0 + cw * (i + 0.5) / 12
                d.line([(xx, y0 - 30 if side == 0 else y0 + cw), (xx, y0 if side == 0 else y0 + cw + 30)], fill=accent, width=4)
        self.base = glow(np.asarray(pil, np.float32), 6, 1.2)
        self.accent = accent
        self.offsets = rng.random(len(self.paths))

    def render(self, t):
        pil = Image.new("RGB", (self.W, self.H))
        d = ImageDraw.Draw(pil)
        for pts, off in zip(self.paths, self.offsets):
            segs = list(zip(pts[:-1], pts[1:]))
            lens = [abs(b[0] - a[0]) + abs(b[1] - a[1]) for a, b in segs]
            total = sum(lens) or 1
            pos = ((t * 0.35 + off) % 1) * total
            for (a, b), ln in zip(segs, lens):
                if pos <= ln and ln > 0:
                    f = pos / ln
                    x, y = a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
                    d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(255, 255, 255))
                    break
                pos -= ln
        pulses = glow(np.asarray(pil, np.float32), 8, 2.5) * (np.array(self.accent) / 255 * 0.7 + 0.3)
        return self.base + pulses


class NeuralNet(Scene):
    dynamic = True

    def build(self):
        W, H, rng = self.W, self.H, self.rng
        n = 70
        self.nodes = np.stack([rng.uniform(0.08, 0.92, n) * W, rng.uniform(0.15, 0.85, n) * H], 1)
        pil = Image.new("RGB", (W, H), (3, 3, 14))
        d = ImageDraw.Draw(pil)
        self.edges = []
        for i in range(n):
            dist = np.linalg.norm(self.nodes - self.nodes[i], axis=1)
            for j in np.argsort(dist)[1:4]:
                if i < j:
                    self.edges.append((i, j))
                    d.line([tuple(self.nodes[i]), tuple(self.nodes[j])], fill=(40, 50, 120), width=2)
        for x, y in self.nodes:
            r = rng.uniform(5, 12)
            d.ellipse([x - r, y - r, x + r, y + r], fill=(120, 140, 255))
        bg = colorize(fbm(W, H, rng, 5, 2) ** 2, [(0, (2, 2, 10)), (1, (40, 15, 80))])
        self.base = glow(np.asarray(pil, np.float32), 5, 1.4) + bg
        self.phase = rng.random(len(self.edges))

    def render(self, t):
        pil = Image.new("RGB", (self.W, self.H))
        d = ImageDraw.Draw(pil)
        for (i, j), ph in zip(self.edges, self.phase):
            f = (t * 0.6 + ph) % 1.0
            if f < 0.75:
                f /= 0.75
                a, b = self.nodes[i], self.nodes[j]
                x, y = a + (b - a) * f
                d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(180, 120, 255))
        return self.base + glow(np.asarray(pil, np.float32), 7, 2.2)


# ------------------------------------------------------------------ mood
class Fog(Scene):
    dynamic = True

    def build(self):
        W, H, rng = self.W, self.H, self.rng
        pal = [(0, (2, 4, 6)), (0.5, (12, 28, 32)), (0.8, (50, 90, 95)), (1, (170, 200, 200))]
        if self.p.get("palette") == "red":
            pal = [(0, (4, 1, 1)), (0.5, (35, 8, 8)), (0.8, (100, 30, 25)), (1, (220, 160, 140))]
        self.fog1 = colorize(fbm(W * 2, H, rng, 6, 2) ** 1.5, pal)
        self.fog2 = colorize(fbm(W * 2, H, rng, 5, 3) ** 2, pal)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        ang = np.arctan2(xx - W * 0.7, yy + H * 0.2)
        ray_noise = np.interp(ang, np.linspace(ang.min(), ang.max(), 64), rng.random(64) ** 3)
        rays = (np.sin(ang * 23) * 0.5 + 0.5) ** 4 * ray_noise * np.exp(-yy / (H * 0.7))
        self.rays = glow(rays[..., None] * np.array([120, 160, 160], np.float32), 30, 0.8) * 0.55
        # silhouette tree line / horizon
        pil = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(pil)
        x = 0
        while x < W:
            hgt = rng.uniform(0.12, 0.3) * H
            w_ = rng.uniform(0.02, 0.05) * W
            d.polygon([(x - w_, H), (x + w_ / 2, H - hgt), (x + w_ * 2, H)], fill=255)
            x += w_ * rng.uniform(0.7, 1.4)
        self.sil = np.asarray(pil.filter(ImageFilter.GaussianBlur(2)), np.float32)[..., None] / 255

    def render(self, t):
        o1 = int(t * self.W * 0.03) % self.W  # fog arrays are 2W wide
        o2 = int(t * self.W * 0.06) % self.W
        f = self.fog1[:, o1:o1 + self.W] * 0.95 + self.fog2[:, o2:o2 + self.W] * 0.6
        img = f + self.rays
        return img * (1 - self.sil * 0.92)


class Abstract(Scene):
    """Flowing gradient blobs – neutral background for facts/psychology."""
    dynamic = True

    def build(self):
        W, H = self.W, self.H
        self.small = (W // 6, H // 6)
        yy, xx = np.mgrid[0:self.small[1], 0:self.small[0]].astype(np.float32)
        self.xx, self.yy = xx / self.small[0], yy / self.small[1]
        self.cols = np.array(self.p.get("colors", [(255, 60, 120), (60, 120, 255), (30, 230, 200), (120, 40, 255)]), np.float32)

    def render(self, t):
        acc = np.zeros(self.xx.shape + (3,), np.float32)
        for i, c in enumerate(self.cols):
            cx = 0.5 + 0.35 * math.sin(t * 0.3 + i * 1.7)
            cy = 0.5 + 0.35 * math.cos(t * 0.23 + i * 2.3)
            w = np.exp(-(((self.xx - cx) * 1.8) ** 2 + ((self.yy - cy) * 1.0) ** 2) * 3)
            acc += w[..., None] * c
        acc = acc * 0.75 + 8
        img = Image.fromarray(np.clip(acc, 0, 255).astype(np.uint8)).resize((self.W, self.H), Image.BICUBIC)
        return np.asarray(img, np.float32)


class Probe(Scene):
    """Lone deep-space probe (high-gain dish, booms) drifting against stars, distant Sun glint."""
    dynamic = True

    def build(self):
        W, H, rng = self.W, self.H, self.rng
        self.bg = Nebula(self.ow, self.oh, int(rng.integers(1 << 30)), palette=self.p.get("palette", "ice")).base * 0.6
        S = int(W * 0.62)
        pil = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        d = ImageDraw.Draw(pil)
        c = S / 2
        # booms
        d.line([(c, c), (S * 0.05, c + S * 0.18)], fill=(150, 150, 160, 255), width=6)
        d.line([(c, c), (S * 0.97, c - S * 0.08)], fill=(140, 140, 150, 255), width=4)
        d.rectangle([S * 0.02, c + S * 0.15, S * 0.12, c + S * 0.22], fill=(90, 80, 70, 255))   # RTG
        # bus
        d.polygon([(c - S * .09, c + S * .02), (c + S * .09, c + S * .02), (c + S * .07, c + S * .12), (c - S * .07, c + S * .12)],
                  fill=(185, 160, 90, 255))
        # dish (ellipse with shading rings)
        for i in range(14):
            k = 1 - i / 14
            col = int(150 + 100 * i / 14)
            d.ellipse([c - S * .3 * k, c - S * .2 - S * .12 * k, c + S * .3 * k, c - S * .2 + S * .12 * k], fill=(col, col, col + 5, 255))
        # concave bowl: darker inner ellipse offset down + bright rim
        d.ellipse([c - S * .24, c - S * .2 - S * .07, c + S * .24, c - S * .2 + S * .1], fill=(105, 108, 118, 255))
        d.ellipse([c - S * .16, c - S * .2 - S * .03, c + S * .16, c - S * .2 + S * .08], fill=(80, 82, 92, 255))
        d.arc([c - S * .3, c - S * .32, c + S * .3, c - S * .08], 180, 360, fill=(250, 250, 255, 255), width=4)
        d.line([(c, c - S * .2), (c, c - S * .36)], fill=(200, 200, 210, 255), width=4)
        d.line([(c, c + S * .02), (c, c - S * .2)], fill=(120, 120, 130, 255), width=5)
        self.sprite = pil
        self.S = S
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        sun = np.exp(-(((xx - W * 0.85) ** 2 + (yy - H * 0.12) ** 2) / (W * 0.02) ** 2))
        self.sun = glow(sun[..., None] * np.array([255, 240, 220], np.float32) * 2.5, 40, 1.5)

    def render(self, t):
        img = Image.fromarray(np.clip(self.bg + self.sun, 0, 255).astype(np.uint8))
        spr = self.sprite.rotate(-8 + 3 * math.sin(t * 0.25), resample=Image.BICUBIC)
        x = int(self.W * 0.5 - self.S / 2 + self.W * 0.03 * math.sin(t * 0.2))
        y = int(self.H * 0.46 - self.S / 2 + self.H * 0.01 * math.cos(t * 0.17))
        img.paste(spr, (x, y), spr)
        return np.asarray(img, np.float32)


class Fusion(Scene):
    """Glowing plasma core inside a toroidal magnetic cage."""
    dynamic = True

    def build(self):
        W, H, rng = self.W, self.H, self.rng
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.dx, self.dy = (xx - W / 2) / W, (yy - H * 0.45) / W
        self.r = np.sqrt(self.dx ** 2 + self.dy ** 2)
        self.ang = np.arctan2(self.dy, self.dx)
        n = fbm(512, 512, rng, 6, 4)
        self.tex = n
        self.cols = [(0, (0, 0, 0)), (0.4, (60, 10, 120)), (0.7, (255, 60, 200)), (0.9, (255, 200, 255)), (1, (255, 255, 255))]
        pil = Image.new("RGB", (W, H))
        d = ImageDraw.Draw(pil)
        for i in range(18):
            a = i / 18 * math.pi
            rx, ry = W * 0.42, W * 0.42 * abs(math.cos(a)) + 6
            d.ellipse([W / 2 - rx, H * 0.45 - ry, W / 2 + rx, H * 0.45 + ry], outline=(40, 120, 255), width=3)
        self.cage = glow(np.asarray(pil, np.float32) * 0.55, 6, 1.0)
        self.bg = Nebula(self.ow, self.oh, int(rng.integers(1 << 30)), palette="violet").base * 0.55

    def render(self, t):
        u = ((self.ang / (2 * np.pi) + 0.5 + t * 0.08) * 511).astype(np.int32) % 512
        v = ((self.r * 3 - t * 0.25) * 511).astype(np.int32) % 512
        core = np.clip(1 - self.r / 0.3, 0, 1) ** 1.5
        val = np.clip(self.tex[v, u] * 0.6 + core * 0.9 + 0.08 * math.sin(t * 6), 0, 1) * np.clip(1 - self.r / 0.36, 0, 1)
        return screen(self.bg, colorize(val, self.cols)) + self.cage


class DNA(Scene):
    """Rotating double helix with base-pair rungs."""
    dynamic = True

    def build(self):
        self.bg = colorize(fbm(self.W, self.H, self.rng, 5, 2) ** 2, [(0, (2, 6, 12)), (1, (10, 50, 70))])

    def render(self, t):
        W, H = self.W, self.H
        pil = Image.new("RGB", (W, H))
        d = ImageDraw.Draw(pil)
        n = 46
        amp = W * 0.22
        back1, back2 = [], []
        for i in range(n * 3):
            y = H * 0.05 + i * H * 0.9 / (n * 3)
            ph = i / 3 * 0.42 + t * 1.2
            back1.append((W / 2 + amp * math.sin(ph), y))
            back2.append((W / 2 + amp * math.sin(ph + math.pi), y))
        d.line(back1, fill=(90, 200, 240), width=9, joint="curve")
        d.line(back2, fill=(240, 110, 180), width=9, joint="curve")
        for i in range(n):
            y = H * 0.05 + i * H * 0.9 / n
            ph = i * 0.42 + t * 1.2
            x1, x2 = W / 2 + amp * math.sin(ph), W / 2 + amp * math.sin(ph + math.pi)
            z1 = math.cos(ph)
            d.line([(x1, y), (x2, y)], fill=(40, 120, 140) if i % 2 else (150, 60, 110), width=4)
            for x, z, col in ((x1, z1, (120, 230, 255)), (x2, -z1, (255, 130, 200))):
                r = 11 + 6 * z
                b = 0.55 + 0.45 * z
                d.ellipse([x - r, y - r, x + r, y + r], fill=tuple(int(c * b) for c in col))
        return self.bg + glow(np.asarray(pil, np.float32), 8, 1.3)


KINDS = {
    "nebula": Nebula, "space": Nebula, "stars": Nebula, "galaxy": Nebula,
    "blackhole": BlackHole, "black_hole": BlackHole,
    "planet": Planet, "earth": Planet,
    "wormhole": Wormhole, "tunnel": Wormhole, "time": Wormhole,
    "grid": SynthGrid, "future": SynthGrid, "synthwave": SynthGrid,
    "circuit": Circuit, "chip": Circuit, "technology": Circuit,
    "neural": NeuralNet, "ai": NeuralNet, "brain": NeuralNet,
    "fog": Fog, "mystery": Fog, "forest": Fog,
    "abstract": Abstract, "facts": Abstract, "psychology": Abstract,
    "probe": Probe, "spacecraft": Probe, "satellite": Probe, "voyager": Probe,
    "fusion": Fusion, "plasma": Fusion, "energy": Fusion, "reactor": Fusion,
    "dna": DNA, "gene": DNA, "biology": DNA, "medicine": DNA,
}


def make_scene(kind: str, w: int, h: int, seed: int, params: dict | None = None) -> Scene:
    cls = KINDS.get(kind, Nebula)
    params = dict(params or {})
    if kind == "earth":
        params.setdefault("kind", "rocky")
    return cls(w, h, seed, **params)


# True-3D scenes register themselves into KINDS on import (kept in a separate module).
from . import three_d as _three_d  # noqa: E402,F401
# Topical (news / business / tech / politics) backgrounds, same self-registration pattern; imported
# after three_d because topical.py builds on its renderer.
from . import topical as _topical  # noqa: E402,F401
