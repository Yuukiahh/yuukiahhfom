"""Creator-economy / social-media / AI-video-making 3D backgrounds.

Same architecture as :mod:`.topical`: real geometry seen through a moving perspective
:class:`~.three_d.Camera` (true parallax); opaque surfaces are flat-shaded PIL polygons in
painter's order; UI screens are PIL textures drawn per frame and perspective-warped onto 3D quads
(and into a quarter-res *emission* layer so screens glow); light (LEDs, sparkles, bokeh, rims) is
additive splatting with bloom.

All randomness is drawn in ``setup()``; ``frame(t)`` is a pure function of ``t``.
Every scene takes ``accent`` ("#RRGGBB"). No real app names, logos, currency designs or
claims: UI is generic shapes, the only text is short generic labels and random numbers.
Registered into ``procedural.KINDS`` at the bottom of this module (``KINDS_CREATOR``).
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import procedural as _procedural  # noqa: F401  (load the registry + renderer chain first)
from .three_d import TAU, Camera, _norm, lerp_col, periodic_noise, rot_x, rot_y, rot_z
from .topical import Topical, font, perspective_coeffs, tint, vgrad


# ======================================================================== helpers
def ease(x):
    x = min(1.0, max(0.0, float(x)))
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = min(1.0, max(0.0, float(x)))
    return 1 - (1 - x) ** 3


def rounded_mask(w, h, r) -> Image.Image:
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], radius=r, fill=255)
    return m


def heart_xy(n=40) -> np.ndarray:
    """Unit heart outline (n, 2), y up, roughly in [-1, 1]."""
    s = np.linspace(0, TAU, n, endpoint=False)
    x = 16 * np.sin(s) ** 3
    y = 13 * np.cos(s) - 5 * np.cos(2 * s) - 2 * np.cos(3 * s) - np.cos(4 * s)
    return np.stack([x / 17.0, (y + 2.5) / 17.0], 1).astype(np.float32)


def fmt_count(v) -> str:
    v = float(v)
    if v >= 1e6:
        return f"{v / 1e6:.1f}M"
    if v >= 1e4:
        return f"{v / 1e3:.1f}K"
    return f"{int(v):,}"


def rounded_rect_xy(hw, hh, r, per=8) -> np.ndarray:
    """Rounded-rectangle outline (n, 2) counter-clockwise, centred at 0."""
    pts = []
    for cx, cy, a0 in ((hw - r, hh - r, 0), (-hw + r, hh - r, 90), (-hw + r, -hh + r, 180), (hw - r, -hh + r, 270)):
        for k in range(per + 1):
            a = math.radians(a0 + 90 * k / per)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return np.array(pts, np.float64)


def _blob_layer(W, H, rng, n, cols, rmin, rmax, blur):
    im = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(im)
    for _ in range(n):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        r = rng.uniform(rmin, rmax)
        d.ellipse([x - r, y - r, x + r, y + r], fill=tint(cols[int(rng.integers(0, len(cols)))]))
    return im.filter(ImageFilter.GaussianBlur(blur))


def clip_art(style: int, W: int, H: int, rng, acc) -> Image.Image:
    """Abstract colourful 'video frame' content (RGB). styles: 0 creator talking to camera, 1 sunset
    landscape, 2 neon abstract, 3 night city, 4 beach / ocean, 5 product on a pedestal."""
    acc = np.asarray(acc, np.float32)
    style %= 6
    if style == 0:  # creator talking (head + shoulders) in a lit room
        pal = [(255, 170, 120), (120, 160, 255), (255, 120, 190), tuple(acc)]
        bg = np.asarray(Image.fromarray(vgrad(W, H, [(0, (60, 50, 80)), (1, (120, 80, 90))])), np.float32)
        bk = np.asarray(_blob_layer(W, H, rng, 26, pal, W * 0.04, W * 0.12, W * 0.03), np.float32)
        im = Image.fromarray(np.clip(bg * 0.7 + bk * 0.45, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(im)
        cx, cy = W * 0.5, H * 0.52
        skin = tuple(int(v) for v in rng.choice(np.array([[236, 188, 160], [200, 150, 110], [150, 100, 70], [250, 210, 180]]), 1)[0])
        shirt = tint(lerp_col(acc, rng.uniform(60, 220, 3), 0.5))
        d.ellipse([cx - W * 0.5, cy + W * 0.3, cx + W * 0.5, cy + W * 1.4], fill=shirt)
        d.rectangle([cx - W * 0.075, cy + W * 0.12, cx + W * 0.075, cy + W * 0.36], fill=tuple(int(c * 0.85) for c in skin))
        d.pieslice([cx - W * 0.12, cy + W * 0.22, cx + W * 0.12, cy + W * 0.42], 0, 180, fill=tuple(int(c * 0.85) for c in skin))
        d.ellipse([cx - W * 0.2, cy - W * 0.27, cx + W * 0.2, cy + W * 0.25], fill=skin)
        hair = tuple(int(v) for v in rng.choice(np.array([[40, 28, 22], [90, 60, 30], [20, 20, 24], [120, 70, 40]]), 1)[0])
        d.chord([cx - W * 0.22, cy - W * 0.32, cx + W * 0.22, cy + W * 0.12], 180, 360, fill=hair)
        for ex in (-0.075, 0.075):
            d.ellipse([cx + W * (ex - 0.02), cy - W * 0.03, cx + W * (ex + 0.02), cy + W * 0.01], fill=(40, 30, 30))
        d.arc([cx - W * 0.08, cy + W * 0.02, cx + W * 0.08, cy + W * 0.14], 20, 160, fill=(150, 70, 70), width=max(2, W // 90))
        im = im.filter(ImageFilter.GaussianBlur(W / 400))
    elif style == 1:  # sunset landscape
        sky = vgrad(W, H, [(0, (60, 40, 120)), (0.35, (230, 110, 120)), (0.55, (255, 190, 110)), (0.6, (255, 220, 160)), (1, (40, 30, 60))])
        im = Image.fromarray(sky)
        d = ImageDraw.Draw(im)
        sy = H * 0.5
        for r, c in ((W * 0.3, (255, 200, 150)), (W * 0.16, (255, 236, 200))):
            d.ellipse([W * 0.55 - r, sy - r, W * 0.55 + r, sy + r], fill=c)
        im = im.filter(ImageFilter.GaussianBlur(W / 60))
        d = ImageDraw.Draw(im)
        d.ellipse([W * 0.55 - W * 0.14, sy - W * 0.14, W * 0.55 + W * 0.14, sy + W * 0.14], fill=(255, 240, 210))
        for layer, (base, col) in enumerate(((0.52, (150, 70, 110)), (0.58, (100, 45, 90)), (0.66, (60, 28, 64)), (0.76, (30, 16, 40)))):
            n = periodic_noise((64,), rng, 2.0)
            xs = np.linspace(0, W, 64)
            ys = H * base - n * H * (0.12 - layer * 0.02)
            d.polygon([(0, H)] + list(zip(xs.tolist(), ys.tolist())) + [(W, H)], fill=col)
    elif style == 2:  # neon abstract
        im = Image.fromarray(vgrad(W, H, [(0, (20, 10, 40)), (1, (40, 10, 50))]))
        cols = [(255, 60, 160), (60, 200, 255), (160, 80, 255), tuple(acc), (255, 200, 80)]
        b = np.asarray(_blob_layer(W, H, rng, 14, cols, W * 0.15, W * 0.4, W * 0.08), np.float32)
        arr = np.asarray(im, np.float32) + b * 1.1
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(im)
        for _ in range(7):
            cx, cy, r = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(W * 0.08, W * 0.3)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=tint(np.array(cols[int(rng.integers(0, 5))]) * 0.5 + 127), width=max(2, W // 80))
        im = im.filter(ImageFilter.GaussianBlur(W / 300))
    elif style == 3:  # night city
        im = Image.fromarray(vgrad(W, H, [(0, (14, 14, 50)), (0.6, (70, 40, 110)), (1, (20, 14, 30))]))
        d = ImageDraw.Draw(im)
        x = -10.0
        while x < W:
            bw = rng.uniform(W * 0.1, W * 0.22)
            top = H * rng.uniform(0.25, 0.6)
            d.rectangle([x, top, x + bw, H], fill=(22, 20, 40))
            for wy in np.arange(top + 8, H, H * 0.018):
                for wx in np.arange(x + 5, x + bw - 6, W * 0.035):
                    if rng.random() < 0.45:
                        d.rectangle([wx, wy, wx + W * 0.018, wy + H * 0.008], fill=tint(lerp_col((255, 210, 140), acc, rng.random() * 0.6)))
            x += bw + rng.uniform(2, 12)
        im = im.filter(ImageFilter.GaussianBlur(W / 500))
    elif style == 4:  # beach / ocean
        im = Image.fromarray(vgrad(W, H, [(0, (90, 180, 255)), (0.42, (200, 235, 255)), (0.43, (20, 140, 190)), (0.62, (60, 200, 210)),
                                          (0.66, (240, 225, 190)), (1, (220, 190, 150))]))
        d = ImageDraw.Draw(im)
        for k in range(9):
            y = H * (0.45 + 0.02 * k)
            d.line([(0, y), (W, y + H * 0.01)], fill=(200, 240, 250), width=2)
        d.ellipse([W * 0.7, H * 0.08, W * 0.9, H * 0.08 + W * 0.2], fill=(255, 250, 230))
        im = im.filter(ImageFilter.GaussianBlur(W / 250))
    else:  # product on a pedestal
        im = Image.fromarray(vgrad(W, H, [(0, tint(acc * 0.5 + 60)), (1, tint(acc * 0.25 + 20))]))
        d = ImageDraw.Draw(im)
        cx = W * 0.5
        d.ellipse([cx - W * 0.35, H * 0.62, cx + W * 0.35, H * 0.7], fill=(235, 235, 240))
        d.rectangle([cx - W * 0.35, H * 0.66, cx + W * 0.35, H], fill=(210, 210, 220))
        d.rounded_rectangle([cx - W * 0.14, H * 0.38, cx + W * 0.14, H * 0.66], radius=int(W * 0.05), fill=tint(255 - acc * 0.5))
        d.rectangle([cx - W * 0.06, H * 0.33, cx + W * 0.06, H * 0.39], fill=(60, 60, 70))
        im = im.filter(ImageFilter.GaussianBlur(W / 400))
    return im


def gen_art(rng, S: int, style: int) -> np.ndarray:
    """Colourful abstract 'generated artwork' (S, S, 3) float32."""
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) / S
    n1 = periodic_noise((S // 4, S // 4), rng, 2.6)
    n1 = np.asarray(Image.fromarray((n1 * 255).astype(np.uint8)).resize((S, S), Image.BICUBIC), np.float32) / 255
    hues = rng.uniform(0, 1, 4)

    def hsv(h, s, v):
        h = (h % 1.0) * 6
        i = np.floor(h)
        f = h - i
        p, q, t_ = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
        i = i.astype(int) % 6 if isinstance(i, np.ndarray) else int(i) % 6
        r = np.choose(i, [v, q, p, p, t_, v])
        g = np.choose(i, [t_, v, v, q, p, p])
        b = np.choose(i, [p, p, t_, v, v, q])
        return np.stack([r, g, b], -1)

    st = style % 4
    if st == 0:  # dreamy landscape: gradient sky, sun, noise mountains
        sky = hsv(hues[0] + yy * 0.25, 0.55 + 0.2 * yy, 0.95 - 0.2 * yy)
        sun = np.exp(-(((xx - rng.uniform(0.3, 0.7)) ** 2 + (yy - 0.38) ** 2) / 0.012))[..., None]
        img = sky + sun * 0.8
        for k, base in enumerate((0.55, 0.68, 0.82)):
            ridge = base - 0.12 * n1[int(S * (0.2 + 0.2 * k)) % S][None, :] + 0.02 * k
            m = (yy > ridge)[..., None]
            col = hsv(np.full((S, S), hues[1] + 0.07 * k), np.full((S, S), 0.6), np.full((S, S), 0.55 - 0.13 * k))
            img = np.where(m, col, img)
    elif st == 1:  # swirling fluid
        ang = np.arctan2(yy - 0.5, xx - 0.5)
        rad = np.hypot(xx - 0.5, yy - 0.5)
        v = np.sin(ang * 3 + rad * 14 + n1 * 6) * 0.5 + 0.5
        img = hsv(hues[0] + v * 0.35 + n1 * 0.2, np.full((S, S), 0.75), 0.55 + 0.45 * v)
    elif st == 2:  # cosmic orb
        bg = hsv(hues[0] + n1 * 0.3, np.full((S, S), 0.7), 0.2 + 0.4 * n1)
        rad = np.hypot(xx - 0.5, yy - 0.52)
        orb = np.clip(1 - rad / 0.3, 0, 1)[..., None]
        oc = hsv(hues[1] + n1 * 0.2, np.full((S, S), 0.6), np.clip(0.6 + 0.4 * (0.5 - (xx + yy) * 0.5) + 0.3 * n1, 0, 1))
        img = bg * (1 - orb ** 0.3) + oc * orb ** 0.3 + np.exp(-((rad - 0.3) / 0.02) ** 2)[..., None] * 0.6
    else:  # geometric shapes / poster
        img = hsv(hues[0] + 0.1 * yy, np.full((S, S), 0.5), np.full((S, S), 0.85))
        for k in range(5):
            cx, cy, r = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8), rng.uniform(0.1, 0.3)
            m = (np.hypot(xx - cx, yy - cy) < r)[..., None]
            img = np.where(m, hsv(np.full((S, S), hues[k % 4] + 0.5 * k), np.full((S, S), 0.75), np.full((S, S), 0.95 - 0.1 * k)), img)
        img = img * (0.85 + 0.15 * n1[..., None])
    return np.clip(img * 255, 0, 255).astype(np.float32)


def sparkle(R, x, y, size, col, bloom=0.8):
    """Four-point star sparkle in screen space (additive)."""
    x, y = np.atleast_1d(np.asarray(x, np.float32)), np.atleast_1d(np.asarray(y, np.float32))
    size = np.broadcast_to(np.asarray(size, np.float32), x.shape)
    col = np.broadcast_to(np.asarray(col, np.float32), (len(x), 3))
    zero = col * 0.0
    for dx, dy in ((1, 0), (0, 1)):
        R.lines(x - dx * size, y - dy * size, x, y, zero, col, 1.2 * R.k, bloom=bloom)
        R.lines(x, y, x + dx * size, y + dy * size, col, zero, 1.2 * R.k, bloom=bloom)
    R.points(x, y, col * 6, np.full(len(x), 2.0 * R.k), bloom=bloom)


# ======================================================================== base class
class Creator(Topical):
    default_accent = "#FF3D7F"

    def init(self):
        pass

    # ---------------------------------------------------------------- drawing helpers
    def sp(self, P):
        """World polygon (n,3) -> screen point list (near clipped) or None."""
        return self.screen_poly(self.to_cam(np.asarray(P, np.float64)))

    def poly(self, d, P, col, outline=None, width=1):
        pts = self.sp(P)
        if pts and len(pts) >= 3:
            d.polygon(pts, fill=None if col is None else tint(col), outline=None if outline is None else tint(outline), width=width)
        return pts

    def begin_emit(self):
        R = self.R
        self._emit = Image.new("RGB", (R.Ws, R.Hs))

    def warp(self, canvas, tex, P4, emit=0.0, min_z=0.15):
        """Perspective-map RGBA texture ``tex`` onto world quad P4 (TL, TR, BR, BL). emit > 0 also draws it
        into the quarter-res emission layer (screen glow)."""
        x, y, z, _ = self.R.project(np.asarray(P4, np.float32))
        if (z < min_z).any():
            return False
        W, H = canvas.size
        x0, y0 = max(int(math.floor(x.min())), 0), max(int(math.floor(y.min())), 0)
        x1, y1 = min(int(math.ceil(x.max())) + 1, W), min(int(math.ceil(y.max())) + 1, H)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return False
        tw, th = tex.size
        co = perspective_coeffs([(x[i] - x0, y[i] - y0) for i in range(4)], [(0, 0), (tw, 0), (tw, th), (0, th)])
        if co is None:
            return False
        warped = tex.transform((x1 - x0, y1 - y0), Image.PERSPECTIVE, co, Image.BILINEAR)
        canvas.paste(warped, (x0, y0), warped)
        if emit > 0:
            ds = self.R.ds
            small = tex.reduce(4) if min(tw, th) > 64 else tex
            sw, sh = small.size
            X0, Y0 = max(int(math.floor(x.min() / ds)), 0), max(int(math.floor(y.min() / ds)), 0)
            X1, Y1 = min(int(math.ceil(x.max() / ds)) + 1, self.R.Ws), min(int(math.ceil(y.max() / ds)) + 1, self.R.Hs)
            if X1 - X0 >= 2 and Y1 - Y0 >= 2:
                co = perspective_coeffs([(x[i] / ds - X0, y[i] / ds - Y0) for i in range(4)], [(0, 0), (sw, 0), (sw, sh), (0, sh)])
                if co is not None:
                    wv = small.transform((X1 - X0, Y1 - Y0), Image.PERSPECTIVE, co, Image.BILINEAR)
                    a = wv.getchannel("A")
                    if emit < 1:
                        a = a.point([int(v * emit) for v in range(256)])
                    self._emit.paste(wv.convert("RGB"), (X0, Y0), a)
        return True

    def finish_emit(self, canvas, soft=None, gain=1.0, canvas_bloom=0.12, **kw):
        e = np.asarray(self._emit, np.float32) * gain
        soft = e if soft is None else soft + e
        return self.fin(canvas, soft, canvas_bloom=canvas_bloom, **kw)

    def pool(self, soft, P, radius, col, squash=1.0):
        """Add a big smooth gaussian light pool (world point P, world radius) into a quarter-res soft
        buffer (Hs, Ws, 3); squash < 1 flattens it vertically (light pool on a floor)."""
        R = self.R
        x, y, z, inv = R.project(np.asarray(P, np.float32)[None, :])
        if z[0] < 0.2:
            return
        ds = R.ds
        r = max(2.0, float(radius * inv[0] / ds))
        cx, cy = float(x[0] / ds), float(y[0] / ds)
        x0, x1 = int(max(0, cx - 3 * r)), int(min(R.Ws, cx + 3 * r + 1))
        y0, y1 = int(max(0, cy - 3 * r * squash)), int(min(R.Hs, cy + 3 * r * squash + 1))
        if x1 <= x0 or y1 <= y0:
            return
        gx = np.exp(-((np.arange(x0, x1, dtype=np.float32) - cx) / r) ** 2)
        gy = np.exp(-((np.arange(y0, y1, dtype=np.float32) - cy) / (r * squash)) ** 2)
        soft[y0:y1, x0:x1] += (gy[:, None] * gx[None, :])[..., None] * np.asarray(col, np.float32)

    def billboard(self, centre, xy, scale):
        """2D shape xy (n,2) -> camera-facing world points around centre."""
        cam = self.R.cam
        return (np.asarray(centre, np.float32)[None, :] + (xy[:, :1] * cam.R[0][None, :] + xy[:, 1:2] * cam.R[1][None, :]) * scale).astype(np.float32)

    def cylinder(self, d, A, B, r, base, light, cpos, n=12, spec=0.4, cap=True, amb=0.35):
        """Shaded cylinder A->B (painter strips facing the camera + end cap)."""
        A, B = np.asarray(A, np.float64), np.asarray(B, np.float64)
        ax = _norm(B - A)
        u = _norm(np.cross(ax, [0.3, 1.0, 0.2] if abs(ax[1]) < 0.9 else [1.0, 0.0, 0.3]))
        v = np.cross(ax, u)
        th = np.linspace(0, TAU, n + 1)
        ring = [np.cos(a) * u + np.sin(a) * v for a in th]
        mid = (A + B) / 2
        strips = []
        for k in range(n):
            nrm = _norm(ring[k] + ring[k + 1])
            if float(nrm @ (cpos - mid)) <= 0:
                continue
            P = [A + ring[k] * r, A + ring[k + 1] * r, B + ring[k + 1] * r, B + ring[k] * r]
            strips.append((P, nrm))
        for P, nrm in strips:
            self.poly(d, P, self.shade(base, nrm, light, cpos - mid, amb, 0.75, spec, 18))
        if cap:
            end, sgn = (B, 1) if float(ax @ (cpos - B)) > 0 else (A, -1)
            if float(ax @ (cpos - end)) * sgn > 0:
                self.poly(d, [end + ring[k] * r for k in range(n)], self.shade(base, ax * sgn, light, cpos - end, amb, 0.75, spec * 0.5, 18))


# ======================================================================== 1. phone feed
class PhoneFeed(Creator):
    """Modern smartphone floating at an angle in 3D; its screen plays a generic vertical short-video
    feed (abstract colourful clips swiping up, right-side heart / comment / bookmark / share icons,
    a like counter ticking up toward ``count``, double-tap heart pops, progress bar, spinning
    music disc); neon hearts float out of the phone; soft bokeh room behind; slow camera drift."""
    default_accent = "#FF3D7F"
    TW, TH = 420, 920
    CLIP_T = 4.0

    def init(self):
        rng = self.rng
        a = self.acc
        cnt = self.p.get("count")
        try:
            self.count = max(10.0, float(cnt)) if cnt is not None else float(rng.integers(80, 900)) * 1000
        except (TypeError, ValueError):
            self.count = float(rng.integers(80, 900)) * 1000
        TW, TH = self.TW, self.TH
        styles = rng.permutation(6)
        self.clips = [clip_art(int(s), TW, TH + 140, rng, a) for s in styles]
        self.clip_hue = [np.asarray(c.reduce(16), np.float32).reshape(-1, 3).mean(0) for c in self.clips]
        self.round = rounded_mask(TW, TH, 46)
        sh = np.zeros((TH, TW), np.float32)
        yy = np.linspace(0, 1, TH, dtype=np.float32)[:, None]
        xx = np.linspace(0, 1, TW, dtype=np.float32)[None, :]
        sh += np.clip((yy - 0.62) / 0.38, 0, 1) ** 1.3 * 190
        sh += np.clip((0.12 - yy) / 0.12, 0, 1) * 110
        sh += np.clip((xx - 0.72) / 0.28, 0, 1) ** 2 * 60 * (yy > 0.4)
        self.shade_mask = Image.fromarray(np.clip(sh, 0, 255).astype(np.uint8), "L")
        self.f_cnt = font(21, "Inter-Bold.otf")
        self.f_big = font(24, "Inter-Bold.otf")
        self.cmt = [float(rng.integers(800, 9000)) for _ in range(6)]
        self.shr = [float(rng.integers(300, 6000)) for _ in range(6)]
        self.caption = [[rng.uniform(0.25, 0.7), rng.uniform(0.3, 0.75)] for _ in range(6)]
        self.heart = heart_xy(40)
        # phone geometry (local, +z = screen side)
        self.hw, self.hh, self.th = 0.52, 1.10, 0.085
        self.outline = rounded_rect_xy(self.hw, self.hh, 0.13, 6)
        # floating hearts
        nh = 14
        self.fh_x = rng.uniform(0.35, 1.25, nh)
        self.fh_z = rng.uniform(-0.2, 0.6, nh)
        self.fh_ph = rng.random(nh)
        self.fh_s = rng.uniform(0.06, 0.14, nh)
        self.fh_c = np.array([[255, 60, 110], [255, 120, 170], [255, 90, 90]], np.float32)[rng.integers(0, 3, nh)] * 0.6 + a * 0.4
        self.fh_c[::3] = a
        # bokeh room
        nb = 70
        self.bok = np.stack([rng.uniform(-9, 9, nb), rng.uniform(-8, 8, nb), rng.uniform(-16, -6, nb)], 1).astype(np.float32)
        pal = np.array([[255, 190, 120], [255, 140, 170], [120, 180, 255], a], np.float32)
        self.bok_c = pal[rng.integers(0, 4, nb)] * (0.25 + 0.75 * rng.random(nb) ** 2)[:, None]
        self.bok_r = rng.uniform(0.25, 0.7, nb).astype(np.float32)
        self.bok_ph = rng.random(nb).astype(np.float32)
        self.bg = Image.fromarray(vgrad(self.W, self.H, [(0, (40, 28, 46)), (0.45, (58, 38, 56)), (0.75, (36, 26, 42)), (1, (22, 18, 30))]))

    # ------------------------------------------------------------ screen texture
    def screen_tex(self, t):
        TW, TH = self.TW, self.TH
        a = self.acc
        T = self.CLIP_T
        k = int(t // T)
        lt = t - k * T
        n = len(self.clips)
        img = Image.new("RGB", (TW, TH))
        sw = ease_out(lt / 0.55) if k > 0 else 1.0
        off = int(round((1 - sw) * TH))
        oy = int(70 + 60 * math.sin(lt * 0.5 + k))
        img.paste(self.clips[k % n].crop((0, oy, TW, oy + TH)), (0, off))
        if off > 0:
            img.paste(self.clips[(k - 1) % n].crop((0, 70, TW, 70 + TH)), (0, off - TH))
        img.paste((0, 0, 0), (0, 0, TW, TH), self.shade_mask)
        d = ImageDraw.Draw(img, "RGBA")
        W_ = (255, 255, 255, 255)
        # status bar
        d.text((34, 18), "9:41", fill=W_, font=self.f_cnt)
        for i in range(4):
            d.rectangle([TW - 112 + i * 8, 36 - i * 4, TW - 107 + i * 8, 40], fill=W_)
        d.rounded_rectangle([TW - 74, 22, TW - 36, 40], radius=5, outline=W_, width=2)
        d.rectangle([TW - 71, 25, TW - 47, 37], fill=W_)
        d.rounded_rectangle([TW // 2 - 58, 14, TW // 2 + 58, 46], radius=16, fill=(0, 0, 0, 255))
        # top tabs (bars, no words)
        d.rounded_rectangle([TW // 2 - 92, 70, TW // 2 - 22, 80], radius=5, fill=(255, 255, 255, 140))
        d.rounded_rectangle([TW // 2 + 14, 70, TW // 2 + 92, 80], radius=5, fill=W_)
        d.rectangle([TW // 2 + 38, 90, TW // 2 + 68, 93], fill=W_)
        # right column
        cx = TW - 46
        y_av = int(TH * 0.47)
        hue = tint(self.clip_hue[k % n] * 0.6 + a * 0.4)
        d.ellipse([cx - 27, y_av - 27, cx + 27, y_av + 27], fill=hue, outline=W_, width=3)
        d.ellipse([cx - 11, y_av - 17, cx + 11, y_av + 5], fill=(255, 255, 255, 200))
        d.chord([cx - 19, y_av + 4, cx + 19, y_av + 40], 180, 360, fill=(255, 255, 255, 200))
        d.ellipse([cx - 11, y_av + 18, cx + 11, y_av + 40], fill=tint(a))
        d.line([(cx - 5, y_av + 29), (cx + 5, y_av + 29)], fill=W_, width=3)
        d.line([(cx, y_av + 24), (cx, y_av + 34)], fill=W_, width=3)
        # heart (liked at 1.1 s into each clip, with a pop)
        y_h = int(TH * 0.585)
        liked = lt > 1.1
        pop = 1.0 + (0.45 * math.sin(min(1.0, (lt - 1.1) / 0.35) * math.pi) if liked and lt < 1.45 else 0.0)
        hs = 25 * pop
        pts = [(cx + float(x) * hs, y_h - float(y) * hs) for x, y in self.heart]
        d.polygon(pts, fill=(255, 45, 85, 255) if liked else (255, 255, 255, 235))
        likes = self.count * (1 - math.exp(-t / 7.0)) * (0.985 + 0.015 * math.tanh(t)) + 3 * t
        tx = fmt_count(likes)
        w_ = d.textlength(tx, font=self.f_cnt)
        d.text((cx - w_ / 2, y_h + 26), tx, fill=W_, font=self.f_cnt)
        # comment bubble
        y_c = int(TH * 0.685)
        d.ellipse([cx - 24, y_c - 21, cx + 24, y_c + 19], fill=(255, 255, 255, 235))
        d.polygon([(cx - 16, y_c + 10), (cx - 22, y_c + 27), (cx - 2, y_c + 16)], fill=(255, 255, 255, 235))
        for j in (-11, 0, 11):
            d.ellipse([cx + j - 3, y_c - 3, cx + j + 3, y_c + 3], fill=(40, 40, 50, 255))
        tx = fmt_count(self.cmt[k % n] + t * 2)
        d.text((cx - d.textlength(tx, font=self.f_cnt) / 2, y_c + 28), tx, fill=W_, font=self.f_cnt)
        # bookmark
        y_b = int(TH * 0.78)
        d.polygon([(cx - 17, y_b - 22), (cx + 17, y_b - 22), (cx + 17, y_b + 22), (cx, y_b + 9), (cx - 17, y_b + 22)],
                  fill=(255, 205, 60, 255) if lt > 2.4 else (255, 255, 255, 235))
        tx = fmt_count(self.shr[(k + 2) % n] * 1.7)
        d.text((cx - d.textlength(tx, font=self.f_cnt) / 2, y_b + 26), tx, fill=W_, font=self.f_cnt)
        # share arrow
        y_s = int(TH * 0.87)
        d.polygon([(cx + 22, y_s - 4), (cx + 2, y_s - 24), (cx + 2, y_s - 13), (cx - 22, y_s - 8), (cx - 20, y_s + 18),
                   (cx - 8, y_s + 4), (cx + 2, y_s + 3), (cx + 2, y_s + 15)], fill=(255, 255, 255, 235))
        tx = fmt_count(self.shr[k % n])
        d.text((cx - d.textlength(tx, font=self.f_cnt) / 2, y_s + 24), tx, fill=W_, font=self.f_cnt)
        # spinning music disc
        y_m = int(TH * 0.955)
        ang = t * 2.0
        d.ellipse([cx - 25, y_m - 25, cx + 25, y_m + 25], fill=(30, 30, 36, 255))
        d.ellipse([cx - 13, y_m - 13, cx + 13, y_m + 13], fill=hue)
        d.line([(cx + 17 * math.cos(ang), y_m + 17 * math.sin(ang)), (cx + 23 * math.cos(ang), y_m + 23 * math.sin(ang))], fill=(200, 200, 210, 255), width=3)
        # bottom-left: handle bar, caption bars, music ticker
        d.ellipse([26, TH - 172, 50, TH - 148], fill=hue)
        d.rounded_rectangle([58, TH - 168, 58 + 150, TH - 152], radius=8, fill=W_)
        d.rounded_rectangle([216, TH - 166, 248, TH - 154], radius=6, outline=W_, width=2)
        c1, c2 = self.caption[k % n]
        d.rounded_rectangle([26, TH - 132, 26 + int(300 * c1) + 40, TH - 120], radius=6, fill=(255, 255, 255, 215))
        d.rounded_rectangle([26, TH - 110, 26 + int(260 * c2) + 30, TH - 98], radius=6, fill=(255, 255, 255, 170))
        mx = 56 - (t * 40) % 60
        d.polygon([(28, TH - 64), (40, TH - 70), (40, TH - 56), (28, TH - 50)], fill=W_)
        for j in range(4):
            d.rounded_rectangle([int(mx + 60 * j) + 10, TH - 66, int(mx + 60 * j) + 54, TH - 58], radius=4, fill=(255, 255, 255, 190))
        d.rectangle([0, TH - 80, 22, TH - 40], fill=(0, 0, 0, 0))
        # progress bar
        prog = lt / T
        yb = TH - 30
        d.rectangle([0, yb, TW, yb + 4], fill=(255, 255, 255, 70))
        d.rectangle([0, yb, int(TW * prog), yb + 4], fill=(255, 255, 255, 255))
        d.ellipse([int(TW * prog) - 7, yb - 5, int(TW * prog) + 7, yb + 9], fill=W_)
        # home indicator
        d.rounded_rectangle([TW // 2 - 70, TH - 14, TW // 2 + 70, TH - 8], radius=3, fill=W_)
        # double-tap heart pop in the middle
        if 1.9 < lt < 2.9:
            u = (lt - 1.9) / 1.0
            s_ = 95 * (0.6 + 0.6 * ease_out(u / 0.3)) * (1 + 0.25 * u)
            al = int(255 * (1 - ease(max(0.0, u - 0.45) / 0.55)))
            hc = (TW * 0.45, TH * 0.42 - 40 * u)
            ang = -0.25
            ca, sa = math.cos(ang), math.sin(ang)
            pts = [(hc[0] + (x * ca - y * sa) * s_, hc[1] - (x * sa + y * ca) * s_) for x, y in self.heart]
            d.polygon(pts, fill=(255, 40, 80, al))
        # small hearts drifting up from the like button after liking
        if liked:
            for j in range(5):
                u = (lt - 1.1 - j * 0.12) / 1.4
                if 0 < u < 1:
                    s_ = 9 + 4 * j % 3
                    hx = cx - 30 * u * (1 + j % 2) + 8 * math.sin(u * 7 + j)
                    hy = y_h - 20 - 220 * u
                    al = int(230 * (1 - u))
                    d.polygon([(hx + float(x) * s_, hy - float(y) * s_) for x, y in self.heart[::2]], fill=(255, 60 + 30 * j, 110, al))
        img.putalpha(self.round)
        return img

    def frame(self, t):
        R, a = self.R, self.acc
        cpos = np.array([0.45 * math.sin(t * 0.21), 0.2 + 0.22 * math.sin(t * 0.17), 4.9 - 0.3 * math.sin(t * 0.13)])
        R.begin(Camera(cpos, (0.05, 0.2, 0.0), fov=40, roll=0.025 * math.sin(t * 0.15)))
        self.begin_emit()
        canvas = self.bg.copy()
        d = ImageDraw.Draw(canvas)
        # bokeh room
        tw = 0.8 + 0.2 * np.sin(t * 0.7 + self.bok_ph * TAU)
        bx, by, bz, binv = R.project(self.bok)
        okb = bz > 0.5
        br = self.bok_r[okb] * binv[okb]
        R.points(bx[okb], by[okb], self.bok_c[okb] * (tw[okb] * br * br * 0.6)[:, None], br, bloom=0.0)
        # blurred window-light strips far behind
        for k in range(4):
            x0 = -5.5 + k * 1.2
            A = np.array([[x0, -6, -14], [x0 + 0.5, -6, -14]], np.float32)
            B = np.array([[x0 + 1.5, 8, -14], [x0 + 2.0, 8, -14]], np.float32)
            R.lines3d(A, B, np.array([255, 200, 160], np.float32) * 0.55, width=40, bloom=0.0)
        # phone pose
        ay = -0.42 + 0.14 * math.sin(t * 0.37)
        ax = 0.12 + 0.05 * math.sin(t * 0.29 + 1)
        az = 0.08 + 0.05 * math.sin(t * 0.23)
        c = np.array([0.0, 0.05 * math.sin(t * 0.8), 0.0])
        M = rot_y(ay) @ rot_x(ax) @ rot_z(az)
        hw, hh, th = self.hw, self.hh, self.th
        ol = self.outline
        front = np.column_stack([ol, np.full(len(ol), th / 2)]) @ M.T + c
        back = np.column_stack([ol, np.full(len(ol), -th / 2)]) @ M.T + c
        light = _norm([0.6, 0.7, 0.8])
        nrm_f = M @ np.array([0, 0, 1.0])
        metal = np.array([150, 150, 165], np.float32) * 0.7 + a * 0.15
        # shadow-ish glow under the phone (soft, behind)
        sx, sy, _, _ = R.project(np.array([[0.1, -0.3, -0.6]], np.float32))
        R.soft(np.repeat(sx, 2), np.repeat(sy, 2), np.stack([a * 1400, np.array([255, 150, 200], np.float32) * 600]))
        # sides
        n = len(ol)
        faces = []
        for i in range(n):
            j = (i + 1) % n
            e = ol[j] - ol[i]
            on = np.array([e[1], -e[0], 0.0])
            on = M @ (on / (np.linalg.norm(on) + 1e-9))
            mid = (front[i] + front[j] + back[i] + back[j]) / 4
            if float(on @ (cpos - mid)) <= 0:
                continue
            faces.append((float(np.linalg.norm(mid - cpos)), [front[i], front[j], back[j], back[i]], on, mid))
        faces.sort(key=lambda f: -f[0])
        for _, P, on, mid in faces:
            self.poly(d, P, self.shade(metal, on, light, cpos - mid, 0.45, 0.6, 0.9, 30))
        # side buttons
        if float((M @ np.array([1.0, 0, 0])) @ (cpos - c)) > 0:
            for y0, y1 in ((0.35, 0.62),):
                P = np.array([[hw + 0.012, y0, th * 0.25], [hw + 0.012, y1, th * 0.25], [hw + 0.012, y1, -th * 0.25], [hw + 0.012, y0, -th * 0.25]]) @ M.T + c
                self.poly(d, P, metal * 0.8)
        # front glass + screen
        self.poly(d, front, (8, 8, 12))
        si = 0.035
        sw_, sh_ = hw - si, hh - si
        z_ = th / 2 + 0.002
        P4 = np.array([[-sw_, sh_, z_], [sw_, sh_, z_], [sw_, -sh_, z_], [-sw_, -sh_, z_]]) @ M.T + c
        tex = self.screen_tex(t)
        self.warp(canvas, tex, P4, emit=1.0)
        # edge highlight on the glass rim
        fA = front.astype(np.float32)
        fB = np.roll(fA, -1, 0)
        e2 = np.roll(ol, -1, 0) - ol
        on2 = np.column_stack([e2[:, 1], -e2[:, 0], np.zeros(n)])
        on2 = on2 / (np.linalg.norm(on2, axis=1, keepdims=True) + 1e-9) @ M.T
        key = np.clip(on2 @ _norm([0.5, 0.8, 0.3]), 0, 1) ** 2
        rim = np.clip(on2 @ _norm([-0.8, -0.2, -0.3]), 0, 1) ** 2
        cols = (np.array([255, 250, 245], np.float32)[None, :] * key[:, None] * 2.2 + a[None, :] * rim[:, None] * 2.5 + 8).astype(np.float32)
        R.lines3d(fA, fB, cols, width=1.6, bloom=0.8)
        # glass glare: soft diagonal band
        g0 = np.array([[-sw_, 0.2, z_ + 0.01], [-sw_, 0.55, z_ + 0.01]]) @ M.T + c
        g1 = np.array([[sw_, 0.75, z_ + 0.01], [sw_, 1.0, z_ + 0.01]]) @ M.T + c
        R.lines3d(g0.astype(np.float32), g1.astype(np.float32), np.array([255, 255, 255], np.float32) * 0.35, width=22, bloom=0.0)
        # floating neon hearts
        cam_dir = 1.0
        for i in range(len(self.fh_x)):
            u = (t / 3.6 + self.fh_ph[i]) % 1.0
            pos = np.array([self.fh_x[i] + 0.15 * math.sin(u * 6 + i) - 0.2 * u, -0.1 + 1.9 * u, self.fh_z[i]]) @ M.T + c
            fade = math.sin(math.pi * u) ** 1.5 * cam_dir
            s_ = self.fh_s[i] * (0.6 + 0.6 * ease_out(u * 3))
            P = self.billboard(pos, self.heart, s_)
            col = self.fh_c[i] * (2.2 * fade)
            R.lines3d(P, np.roll(P, -1, 0), col, width=2.0, bloom=0.9)
            x, y, z, inv = R.project(pos[None, :].astype(np.float32))
            R.points(x, y, (self.fh_c[i] * 60 * fade)[None, :], np.array([s_ * inv[0] * 0.6]), bloom=0.0)
        # screen light spill onto the room
        hue = self.clip_hue[int(t // self.CLIP_T) % len(self.clips)]
        x, y, _, _ = R.project(c[None, :].astype(np.float32))
        R.soft(np.repeat(x, 2), np.repeat(y, 2), np.stack([hue * 900, a * 500]))
        return self.finish_emit(canvas, None, gain=0.45, canvas_bloom=0.08, bloom=(0.6, 0.55), radii=(1.4, 9.0))


# ======================================================================== 2. edit timeline
class EditTimeline(Creator):
    """Video-editor workspace in 3D: multi-track timeline (coloured video clips with thumbnail frames,
    audio clips with waveforms) receding in perspective below a big floating preview monitor; a glowing
    playhead sweeps right, a razor cut flashes, the cut piece lifts away and the clips ripple-slide
    together; the camera dollies along with the playhead."""
    default_accent = "#4D8DFF"
    V = 0.55  # playhead speed (world units / s)
    CUT_T, CUT_0 = 3.2, 1.0

    def init(self):
        rng = self.rng
        a = self.acc
        # tracks (far = video on top of the screen, near = audio)
        self.tracks = [("v", -2.35), ("v", -1.7), ("a", -1.05), ("a", -0.45)]
        self.lane_w = 0.27
        span = (-12.0, self.V * 300 + 120)
        vid_pal = np.array([a, [140, 90, 255], [40, 200, 210], [255, 110, 150], [255, 170, 70], [90, 150, 255]], np.float32)
        self.clips = []
        for ti, (kind, z) in enumerate(self.tracks):
            x = span[0] + rng.uniform(0, 1.0)
            cl = []
            while x < span[1]:
                if ti == 0 and rng.random() < 0.45:  # overlay track has gaps
                    x += rng.uniform(1.0, 3.0)
                ln = rng.uniform(1.4, 3.4) if kind == "v" else rng.uniform(2.5, 6.0)
                if kind == "v":
                    col = vid_pal[int(rng.integers(0, len(vid_pal)))] * rng.uniform(0.75, 1.0)
                else:
                    col = np.array([60, 200, 120], np.float32) * rng.uniform(0.7, 1.0) if ti == 2 else np.array([70, 170, 230], np.float32) * rng.uniform(0.7, 1.0)
                cl.append([x, x + ln, col, int(rng.integers(0, 6))])
                x += ln + 0.03
            self.clips.append(cl)
        # waveform amplitude per source x (0.04 sampling)
        self.wd = 0.045
        nwav = int((span[1] - span[0]) / self.wd) + 2
        self.w0 = span[0]
        self.wav = []
        for _ in range(2):
            env = periodic_noise((nwav,), rng, 1.6)
            fine = rng.random(nwav)
            self.wav.append((0.25 + 0.75 * env) * (0.35 + 0.65 * fine))
        # cut schedule (ripple delete on tracks 1 and 2: main video + its audio)
        self.cuts = []
        shift = 0.0
        for k in range(100):
            tc = self.CUT_0 + k * self.CUT_T
            g = float(rng.uniform(0.6, 1.1))
            c = self.V * tc + shift  # playhead at tc (display) -> source coordinate
            self.cuts.append((tc, c, g))
            shift += g
        self.cut_tc = np.array([c[0] for c in self.cuts])
        self.cut_c = np.array([c[1] for c in self.cuts])
        self.cut_g = np.array([c[2] for c in self.cuts])
        self.spark_v = rng.normal(0, 1, (40, 3)) * [1.2, 1.6, 0.8] + [0, 1.2, 0.3]
        # monitor clips
        self.mon = [clip_art(s, 512, 330, rng, a) for s in range(6)]
        self.MW, self.MH = 480, 270
        self.mon_round = rounded_mask(self.MW, self.MH + 56, 10)
        self.f_tc = font(26, "Inter-Bold.otf")
        nb = 60
        self.bok = np.stack([rng.uniform(-10, 10, nb), rng.uniform(-3, 7, nb), rng.uniform(-14, -7, nb)], 1).astype(np.float32)
        self.bok_c = lerp_col(a, (255, 200, 150), rng.random(nb) * 0.6) * (0.2 + 0.6 * rng.random(nb) ** 2)[:, None]
        self.bg = Image.fromarray(vgrad(self.W, self.H, [(0, (20, 24, 40)), (0.4, (34, 38, 60)), (1, (16, 18, 28))]))

    def ripple(self, t):
        """Per cut: ease of the ripple slide (0..1)."""
        return np.clip((t - self.cut_tc - 0.55) / 0.6, 0, 1) ** 2 * (3 - 2 * np.clip((t - self.cut_tc - 0.55) / 0.6, 0, 1))

    def disp(self, s, e):
        """Display x of source positions s (array) for ripple eases e."""
        s = np.asarray(s, np.float64)
        thr = self.cut_c + self.cut_g
        return s - ((s[..., None] >= thr - 1e-9) * (self.cut_g * e)).sum(-1)

    def monitor_tex(self, t, clip_style, flash):
        MW, MH = self.MW, self.MH
        img = Image.new("RGB", (MW, MH + 56), (22, 24, 32))
        src = self.mon[clip_style % len(self.mon)]
        ox = int(16 + 14 * math.sin(t * 0.4))
        oy = int(30 + 20 * math.sin(t * 0.3))
        img.paste(src.crop((ox, oy, ox + MW, oy + MH)), (0, 0))
        if flash > 0.01:
            img.paste((255, 255, 255), (0, 0, MW, MH), Image.new("L", (MW, MH), int(200 * flash)))
        d = ImageDraw.Draw(img, "RGBA")
        # safe-frame guides + REC-style corner marks
        for (x0, y0, sx, sy) in ((14, 14, 1, 1), (MW - 14, 14, -1, 1), (14, MH - 14, 1, -1), (MW - 14, MH - 14, -1, -1)):
            d.line([(x0, y0), (x0 + 26 * sx, y0)], fill=(255, 255, 255, 180), width=2)
            d.line([(x0, y0), (x0, y0 + 26 * sy)], fill=(255, 255, 255, 180), width=2)
        # transport bar
        y = MH + 10
        d.polygon([(MW / 2 - 9, y + 6), (MW / 2 + 11, y + 17), (MW / 2 - 9, y + 28)], fill=(240, 240, 250, 255))
        for dx in (-60, 50):
            d.polygon([(MW / 2 + dx, y + 8), (MW / 2 + dx + 14, y + 17), (MW / 2 + dx, y + 26)], fill=(170, 170, 190, 255))
        secs = t * 1.0 + 12
        tc = f"00:{int(secs // 60):02d}:{int(secs % 60):02d}:{int((secs * 30) % 30):02d}"
        d.text((14, y + 4), tc, fill=tuple(int(v) for v in self.acc) + (255,), font=self.f_tc)
        d.rectangle([MW - 150, y + 15, MW - 16, y + 19], fill=(80, 84, 100, 255))
        d.rectangle([MW - 150, y + 15, MW - 150 + int(134 * ((t * 0.07) % 1)), y + 19], fill=tuple(int(v) for v in self.acc) + (255,))
        img.putalpha(self.mon_round)
        return img

    def frame(self, t):
        R, a = self.R, self.acc
        xp = self.V * t
        sway = math.sin(t * 0.21)
        cpos = np.array([xp + 0.55 * sway + 0.2, 3.5 + 0.15 * math.sin(t * 0.17), 1.9 + 0.25 * math.sin(t * 0.13)])
        tgt = np.array([xp + 0.15 * sway, 0.75, -2.5])
        R.begin(Camera(cpos, tgt, fov=60, roll=0.015 * math.sin(t * 0.19)))
        self.begin_emit()
        canvas = self.bg.copy()
        d = ImageDraw.Draw(canvas)
        light = _norm([-0.3, 1.0, 0.5])
        xl, xr = xp - 6.0, xp + 7.0
        # desk / timeline bed
        self.poly(d, [(xl - 4, -0.02, 0.4), (xr + 4, -0.02, 0.4), (xr + 4, -0.02, -3.2), (xl - 4, -0.02, -3.2)], (24, 27, 40))
        for ti, (kind, z) in enumerate(self.tracks):
            lw = self.lane_w + 0.03
            self.poly(d, [(xl - 4, 0.0, z + lw), (xr + 4, 0.0, z + lw), (xr + 4, 0.0, z - lw), (xl - 4, 0.0, z - lw)], (34, 38, 56) if kind == "v" else (30, 36, 48))
        # ruler strip + tool bar with buttons
        self.poly(d, [(xl - 4, 0.0, -2.75), (xr + 4, 0.0, -2.75), (xr + 4, 0.0, -3.05), (xl - 4, 0.0, -3.05)], (40, 44, 64))
        self.poly(d, [(xl - 4, 0.0, -3.1), (xr + 4, 0.0, -3.1), (xr + 4, 0.0, -3.55), (xl - 4, 0.0, -3.55)], (30, 33, 50))
        bx0 = math.floor((xp - 2.6) / 0.42) * 0.42
        for k in range(14):
            bx = bx0 + k * 0.42
            hot = int(round(bx / 0.42)) % 7 == 3
            self.poly(d, [(bx, 0.005, -3.17), (bx + 0.3, 0.005, -3.17), (bx + 0.3, 0.005, -3.48), (bx, 0.005, -3.48)], a * 0.8 + 20 if hot else (52, 58, 82))
        # ---- clips (painter: far tracks first, inside a track by distance)
        e = self.ripple(t)
        active = self.cut_tc <= t
        bounds = np.concatenate([self.cut_c[active], self.cut_c[active] + self.cut_g[active]])
        flash_mon = 0.0
        for ti, (kind, z) in enumerate(self.tracks):
            linked = ti in (1, 2)
            hgt = 0.075 if kind == "v" else 0.05
            pieces = []
            for (s0, s1, col, sty) in self.clips[ti]:
                if linked:
                    cuts = sorted(b for b in bounds if s0 < b < s1)
                    edges = [s0] + cuts + [s1]
                else:
                    edges = [s0, s1]
                for p0, p1 in zip(edges[:-1], edges[1:]):
                    lift = 0.0
                    fade = 1.0
                    if linked:
                        k = np.nonzero(active & (self.cut_c <= p0 + 1e-6) & (self.cut_c + self.cut_g >= p1 - 1e-6))[0]
                        if len(k):
                            age = t - self.cut_tc[k[0]]
                            if age > 1.0:
                                continue
                            lift = ease_out(max(0.0, age - 0.2) / 0.8) * 0.9
                            fade = 1.0 - ease(max(0.0, age - 0.5) / 0.5)
                    if linked:
                        x0, x1 = (float(v) for v in self.disp(np.array([p0, p1]), e))
                    else:
                        x0, x1 = p0, p1
                    if x1 < xl or x0 > xr:
                        continue
                    pieces.append((x0 + 0.012, x1 - 0.012, col, sty, lift, fade))
            pieces.sort(key=lambda p: -abs(0.5 * (p[0] + p[1]) - cpos[0]))
            for (x0, x1, col, sty, lift, fade) in pieces:
                zw = self.lane_w
                y0, y1 = lift, lift + hgt
                selc = (lerp_col(col, a * 0.5 + 120, 0.45) if lift > 0 else col) * (0.35 + 0.65 * fade)
                for quad, nrm in self.box_faces((x0, x1, y0, y1, z - zw, z + zw), cpos):
                    base = selc * (1.0 if nrm[1] > 0 else 0.55)
                    self.poly(d, quad, self.shade(base, nrm, light, cpos - np.mean(quad, 0), 0.55, 0.5, 0.25, 20))
                if kind == "v" and x1 - x0 > 0.2:
                    # thumbnail frames on the top face
                    fw = 0.42
                    nfr = max(1, int((x1 - x0 - 0.04) / fw))
                    for f_ in range(nfr):
                        fx0 = x0 + 0.03 + f_ * fw
                        fx1 = min(fx0 + fw - 0.03, x1 - 0.03)
                        if fx1 <= fx0:
                            continue
                        tcol = lerp_col(selc * 0.45, (220, 220, 235), 0.15 + 0.12 * ((f_ + sty) % 3))
                        self.poly(d, [(fx0, y1 + 0.001, z - zw + 0.05), (fx1, y1 + 0.001, z - zw + 0.05), (fx1, y1 + 0.001, z + zw - 0.12),
                                      (fx0, y1 + 0.001, z + zw - 0.12)], tcol)
                        hz = z - zw + 0.05 + (zw * 2 - 0.17) * 0.55
                        self.poly(d, [(fx0, y1 + 0.002, hz), (fx1, y1 + 0.002, hz - 0.04), (fx1, y1 + 0.002, z + zw - 0.12), (fx0, y1 + 0.002, z + zw - 0.12)],
                                  lerp_col(selc * 0.3, (30, 30, 40), 0.5))
                if lift > 0:  # selection outline glow
                    P = np.array([(x0, y1, z - zw), (x1, y1, z - zw), (x1, y1, z + zw), (x0, y1, z + zw)], np.float32)
                    R.lines3d(P, np.roll(P, -1, 0), (a * 1.5 + 60) * fade, width=2.0, bloom=1.0)
            if kind == "a":
                # waveforms (additive), on top of the clips
                wi = ti - 2
                xs = np.arange(math.floor(xl / self.wd) * self.wd, xr, self.wd)
                if linked:
                    # map display -> source for the waveform lookup: invert by fixed-point (few cuts in view)
                    src = xs.copy()
                    for _ in range(3):
                        src = xs + (src - self.disp(src, e))
                else:
                    src = xs
                inside = np.zeros(len(xs), bool)
                for (s0, s1, col, sty) in self.clips[ti]:
                    inside |= (src > s0 + 0.04) & (src < s1 - 0.04)
                if linked:
                    rem = ((src[:, None] > self.cut_c) & (src[:, None] < self.cut_c + self.cut_g) & active & (t - self.cut_tc > 0.25)).any(1)
                    inside &= ~rem
                idx = np.clip(((src - self.w0) / self.wd).astype(int), 0, len(self.wav[wi]) - 1)
                amp = self.wav[wi][idx] * self.lane_w * 0.85
                xs, amp = xs[inside], amp[inside]
                yv = hgt + 0.003
                A = np.stack([xs, np.full(len(xs), yv), z - amp], 1).astype(np.float32)
                B = np.stack([xs, np.full(len(xs), yv), z + amp], 1).astype(np.float32)
                wc = np.array([150, 255, 190], np.float32) if wi == 0 else np.array([150, 210, 255], np.float32)
                played = (xs < xp)[:, None]
                R.lines3d(A, B, np.where(played, wc * 0.9, wc * 0.5), width=1.3, bloom=0.35)
        # ruler ticks
        tx = np.arange(math.floor(xl * 4) / 4, xr, 0.25)
        major = (np.round(tx * 4) % 4 == 0)
        A = np.stack([tx, np.full(len(tx), 0.002), np.full(len(tx), -3.03)], 1).astype(np.float32)
        B = np.stack([tx, np.full(len(tx), 0.002), np.where(major, -2.8, -2.95)], 1).astype(np.float32)
        R.lines3d(A, B, np.where(major[:, None], np.float32(150), np.float32(70)) * np.ones((1, 3), np.float32), width=1.0, bloom=0.2)
        # ---- razor flashes
        for k in np.nonzero(active & (t - self.cut_tc < 1.2))[0]:
            age = t - self.cut_tc[k]
            for j, (sx_, dt) in enumerate(((self.cut_c[k], 0.0), (self.cut_c[k] + self.cut_g[k], 0.12))):
                ag = age - dt
                if ag < 0:
                    continue
                fl = math.exp(-ag * 6.0)
                xd = float(self.disp(np.array([sx_]), e)[0]) if j == 0 else float(self.disp(np.array([sx_ - 1e-6]), e)[0])
                A = np.array([[xd, 0.0, -1.7 - 0.36], [xd, 0.0, -1.05 - 0.36]], np.float32)
                B = np.array([[xd, 0.0, -1.7 + 0.36], [xd, 0.0, -1.05 + 0.36]], np.float32)
                R.lines3d(A, B, np.array([255, 250, 240], np.float32) * 9 * fl, width=2.5, bloom=1.6)
                R.lines3d(np.array([[xd, 0, -1.4]], np.float32), np.array([[xd, 1.4, -1.4]], np.float32), (a * 0.6 + 120) * 3.5 * fl, width=8, bloom=0.0)
                sp = np.array([xd, 0.05, -1.7]) + self.spark_v * (ag * 0.6) + np.array([0, -1.0, 0]) * ag ** 2
                R.points3d(sp.astype(np.float32), np.array([255, 230, 180], np.float32) * 40 * fl, 0.0, 1.2, bloom=0.8)
                px, py, _, _ = R.project(np.array([[xd, 0.1, -1.4]], np.float32))
                R.soft(np.repeat(px, 2), np.repeat(py, 2), np.stack([np.full(3, 255.0) * 1500 * fl, a * 2500 * fl]))
                flash_mon = max(flash_mon, fl * 0.7)
        # ---- playhead
        ph_top = np.array([[xp, 0.0, -3.05]], np.float32)
        ph_bot = np.array([[xp, 0.0, 0.35]], np.float32)
        R.lines3d(ph_top, ph_bot, a * 4 + 60, width=2.2, bloom=1.4)
        R.lines3d(np.array([[xp, 0.0, -3.0]], np.float32), np.array([[xp, 0.7, -3.0]], np.float32), a * 2.5 + 40, width=1.8, bloom=1.0)
        self.poly(d, [(xp - 0.12, 0.004, -3.08), (xp + 0.12, 0.004, -3.08), (xp + 0.12, 0.004, -2.92), (xp, 0.004, -2.82), (xp - 0.12, 0.004, -2.92)], a * 0.9 + 30)
        # light curtain along the playhead
        x, y, _, _ = R.project(np.array([[xp, 0.02, -2.9], [xp, 0.02, 0.3]], np.float32))
        xs_, ys_, _, s_, st = R.sample_segments(x[:1], y[:1], x[1:], y[1:], 4.0, 400)
        R.soft(xs_, ys_, a[None, :] * (0.8 * st)[:, None])
        # ---- preview monitor (floating, with bezel)
        mx, my, mz = xp + 0.15 * sway, 2.15, -4.1
        mw, mh = 2.5, 2.5 * (self.MH + 56) / self.MW
        sty = 0
        for (s0, s1, col, st_) in self.clips[1]:
            xs0, xs1 = (float(v) for v in self.disp(np.array([s0, s1]), e))
            if xs0 <= xp < xs1:
                sty = st_
                break
        bez = 0.06
        for quad, nrm in self.box_faces((mx - mw / 2 - bez, mx + mw / 2 + bez, my - mh / 2 - bez, my + mh / 2 + bez, mz - 0.12, mz), cpos):
            self.poly(d, quad, self.shade(np.array([40, 44, 58]), nrm, light, cpos - np.mean(quad, 0), 0.5, 0.5, 0.4, 20))
        P4 = [(mx - mw / 2, my + mh / 2, mz + 0.005), (mx + mw / 2, my + mh / 2, mz + 0.005), (mx + mw / 2, my - mh / 2, mz + 0.005), (mx - mw / 2, my - mh / 2, mz + 0.005)]
        self.warp(canvas, self.monitor_tex(t, sty, flash_mon), P4, emit=0.55)
        # side panels (media bin thumbnails) left/right of the monitor
        for side in (-1, 1):
            px0 = mx + side * (mw / 2 + 0.25)
            px1 = px0 + side * 1.4
            self.poly(d, [(px0, my + mh / 2, mz + 0.3), (px1, my + mh / 2, mz + 0.9), (px1, my - mh / 2 - 0.6, mz + 0.9), (px0, my - mh / 2 - 0.6, mz + 0.3)], (30, 34, 50))
            for r_ in range(4):
                for c_ in range(2):
                    u0, u1 = 0.1 + c_ * 0.45, 0.45 + c_ * 0.45
                    v0, v1 = 0.06 + r_ * 0.22, 0.22 + r_ * 0.22
                    yy0, yy1 = my + mh / 2 - v0 * (mh + 0.6), my + mh / 2 - v1 * (mh + 0.6)
                    q = [(px0 + (px1 - px0) * u, yy, mz + 0.3 + 0.6 * u) for (u, yy) in ((u0, yy0), (u1, yy0), (u1, yy1), (u0, yy1))]
                    cc = self.clips[1][(r_ * 2 + c_ + (side > 0) * 8) % len(self.clips[1])][2]
                    self.poly(d, q, cc * 0.55 + 20)
        # monitor glow + bokeh
        x, y, _, _ = R.project(np.array([[mx, my, mz + 0.2]], np.float32))
        R.soft(np.repeat(x, 2), np.repeat(y, 2), np.stack([a * 1200, np.array([255, 255, 255], np.float32) * 300]))
        R.points3d(self.bok + np.array([xp, 0, 0], np.float32) * 0.85, self.bok_c * 60, 0.15, 0.0, bloom=0.0)
        return self.finish_emit(canvas, None, gain=0.5, canvas_bloom=0.12, bloom=(0.6, 0.55), radii=(1.4, 9.0))


# ======================================================================== 3. money rain
class MoneyRain(Creator):
    """Gold coins (embossed generic '$' faces) and generic banknotes (guilloche lines, '$' medallion,
    no real currency design) tumbling down through warm light with motion-streak glow and sparkle
    glints, depth of field, heaps and stacks of coins at the bottom; slow push-in.
    intensity 0..1 = amount / speed of the rain."""
    default_accent = "#FFC24A"
    CT = 128
    NW, NH = 320, 140

    def _coin_tex(self):
        S = self.CT
        im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) / S - 0.5
        r = np.hypot(xx, yy)
        g = np.clip(1.0 - (xx + yy) * 0.9, 0.3, 1.6)
        gold = np.array([255, 196, 80], np.float32)
        col = gold[None, None, :] * (0.65 + 0.35 * g[..., None])
        col = np.where((r > 0.40)[..., None] & (r < 0.5)[..., None], gold * (0.95 - (xx - yy)[..., None] * 0.6), col)
        col = np.where((np.abs(r - 0.37) < 0.012)[..., None], gold * 0.6, col)
        alpha = np.clip((0.5 - r) * S * 1.5, 0, 1) * 255
        arr = np.dstack([np.clip(col, 0, 255), alpha]).astype(np.uint8)
        im = Image.fromarray(arr, "RGBA")
        d = ImageDraw.Draw(im)
        f = font(int(S * 0.55))
        l, t_, rr, b = f.getbbox("$")
        x0, y0 = (S - (rr - l)) / 2 - l, (S - (b - t_)) / 2 - t_
        d.text((x0 + 3, y0 + 3), "$", fill=(150, 95, 20, 255), font=f)
        d.text((x0 - 1, y0 - 1), "$", fill=(255, 236, 160, 255), font=f)
        d.text((x0 + 1, y0 + 1), "$", fill=(225, 160, 50, 255), font=f)
        for k in range(48):
            ang = k / 48 * TAU
            d.ellipse([S / 2 + 0.445 * S * math.cos(ang) - 1.5, S / 2 + 0.445 * S * math.sin(ang) - 1.5,
                       S / 2 + 0.445 * S * math.cos(ang) + 1.5, S / 2 + 0.445 * S * math.sin(ang) + 1.5], fill=(255, 230, 150, 255))
        return im

    def _note_tex(self):
        NW, NH = self.NW, self.NH
        base = np.array([150, 205, 160], np.float32) * 0.8 + self.acc * 0.05
        im = Image.new("RGBA", (NW, NH), tint(base) + (255,))
        d = ImageDraw.Draw(im)
        d.rectangle([6, 6, NW - 7, NH - 7], outline=tint(base * 0.55), width=3)
        d.rectangle([12, 12, NW - 13, NH - 13], outline=tint(base * 0.7), width=1)
        for k in range(26):  # guilloche waves
            ph = k * 0.24
            xs = np.linspace(14, NW - 14, 120)
            ys = NH / 2 + (NH * 0.36) * np.sin(xs / NW * TAU * 2 + ph) * np.cos(xs / NW * TAU * 0.5 + ph * 0.3)
            d.line(list(zip(xs.tolist(), ys.tolist())), fill=tint(base * (0.75 + 0.1 * (k % 2))), width=1)
        cx, cy, rr = NW * 0.5, NH * 0.5, NH * 0.33
        for k in range(36):  # rosette
            ang = k / 36 * math.pi
            d.ellipse([cx - rr, cy - rr * 0.45, cx + rr, cy + rr * 0.45], outline=None)
            pts = [(cx + rr * math.cos(s) * math.cos(ang) - rr * 0.45 * math.sin(s) * math.sin(ang),
                    cy + rr * math.cos(s) * math.sin(ang) + rr * 0.45 * math.sin(s) * math.cos(ang)) for s in np.linspace(0, TAU, 40)]
            d.line(pts, fill=tint(base * 0.62), width=1)
        d.ellipse([cx - rr * 0.62, cy - rr * 0.62, cx + rr * 0.62, cy + rr * 0.62], fill=tint(base * 1.08), outline=tint(base * 0.5), width=2)
        f = font(int(NH * 0.36))
        l, t_, r_, b = f.getbbox("$")
        d.text((cx - (r_ - l) / 2 - l, cy - (b - t_) / 2 - t_), "$", fill=tint(base * 0.4), font=f)
        f2 = font(int(NH * 0.2))
        for (x, y) in ((22, 18), (NW - 44, NH - 46)):
            d.text((x, y), "$", fill=tint(base * 0.45), font=f2)
        for x in (NW * 0.18, NW * 0.82):
            d.ellipse([x - 16, cy - 16, x + 16, cy + 16], outline=tint(base * 0.55), width=2)
        return im

    def init(self):
        rng = self.rng
        a = self.acc
        try:
            it = float(self.p.get("intensity", 0.6))
        except (TypeError, ValueError):
            it = 0.6
        self.it = it = float(np.clip(it, 0.0, 1.0))
        coin = self._coin_tex()
        backc = coin.copy()
        self.coin_lv = []  # [blur][light]
        for bl in (0, 2.5, 6):
            src = coin if bl == 0 else coin.filter(ImageFilter.GaussianBlur(bl))
            lv = []
            for s in (0.45, 0.7, 0.95, 1.25):
                arr = np.asarray(src, np.float32).copy()
                arr[..., :3] = np.clip(arr[..., :3] * s, 0, 255)
                lv.append(Image.fromarray(arr.astype(np.uint8), "RGBA"))
            self.coin_lv.append(lv)
        del backc
        note = self._note_tex()
        self.nstrip = 4
        self.note_lv = []
        for bl in (0, 2.5, 6):
            src = note if bl == 0 else note.filter(ImageFilter.GaussianBlur(bl))
            lv = []
            for s in (0.5, 0.75, 1.0, 1.2):
                arr = np.asarray(src, np.float32).copy()
                arr[..., :3] = np.clip(arr[..., :3] * s, 0, 255)
                im = Image.fromarray(arr.astype(np.uint8), "RGBA")
                sw = self.NW // self.nstrip
                lv.append([im.crop((i * sw, 0, (i + 1) * sw, self.NH)) for i in range(self.nstrip)])
            self.note_lv.append(lv)
        # falling coins
        nc = int(22 + 38 * it)
        self.nc = nc
        self.c_pos = np.stack([rng.uniform(-3.2, 3.2, nc), rng.uniform(0, 1, nc), rng.uniform(-7, 2.5, nc)], 1)
        self.c_v = rng.uniform(0.8, 1.4, nc) * (0.75 + 0.6 * it)
        self.c_ax = np.array([_norm(v) for v in rng.normal(0, 1, (nc, 3))])
        self.c_w = rng.uniform(1.5, 4.5, nc) * rng.choice([-1, 1], nc)
        self.c_ph = rng.uniform(0, TAU, nc)
        self.c_r = rng.uniform(0.17, 0.24, nc)
        # banknotes
        nn = int(5 + 11 * it)
        self.nn = nn
        self.n_pos = np.stack([rng.uniform(-3.0, 3.0, nn), rng.uniform(0, 1, nn), rng.uniform(-6, 1.8, nn)], 1)
        self.n_v = rng.uniform(0.45, 0.75, nn) * (0.75 + 0.6 * it)
        self.n_ph = rng.uniform(0, TAU, (nn, 4))
        self.n_w = rng.uniform(0.6, 1.4, (nn, 3))
        self.span_y = (-4.2, 5.4)
        # heaps at the bottom: stacks + loose coins on a mound
        self.floor = -3.35
        st = []
        for k in range(11):
            x = rng.uniform(-2.6, 2.6)
            z = rng.uniform(-2.0, 1.5)
            st.append((x, z, int(rng.integers(4, 13)), rng.uniform(0.2, 0.24)))
        self.stacks = sorted(st, key=lambda s: s[1])
        nl = 70
        lx = rng.normal(0, 1.4, nl)
        lz = rng.normal(-0.3, 1.0, nl)
        hy = 0.55 * np.exp(-(lx ** 2 + (lz + 0.3) ** 2) / 2.2)
        self.loose = np.stack([lx, self.floor + hy + rng.uniform(0, 0.08, nl), lz], 1)
        self.loose_n = np.array([_norm([rng.normal(0, 0.45), 1.0, rng.normal(0, 0.45)]) for _ in range(nl)])
        self.loose_r = rng.uniform(0.17, 0.23, nl)
        # bokeh + dust
        nb = 80
        self.bok = np.stack([rng.uniform(-9, 9, nb), rng.uniform(-6, 8, nb), rng.uniform(-18, -8, nb)], 1).astype(np.float32)
        self.bok_c = lerp_col((255, 190, 90), a, rng.random(nb) * 0.4) * (0.2 + 0.8 * rng.random(nb) ** 2)[:, None]
        nd = 350
        self.dust = np.stack([rng.uniform(-4, 4, nd), rng.uniform(-4, 5, nd), rng.uniform(-6, 3, nd)], 1).astype(np.float32)
        self.dust_ph = rng.random(nd).astype(np.float32)
        self.bg = Image.fromarray(vgrad(self.W, self.H, [(0, (50, 30, 14)), (0.4, (78, 48, 20)), (0.75, (54, 32, 16)), (1, (30, 18, 10))]))

    def _disc(self, c, nrm, r, steps=22):
        nrm = _norm(nrm)
        u = _norm(np.cross(nrm, [0.0, 0.0, 1.0] if abs(nrm[2]) < 0.9 else [1.0, 0.0, 0.0]))
        v = np.cross(nrm, u)
        th = np.linspace(0, TAU, steps, endpoint=False)
        return c + r * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * v), u, v

    def _draw_coin(self, canvas, d, c, M, r, light, cpos, blur, glints):
        n = M @ np.array([0, 0, 1.0])
        view = cpos - c
        if float(n @ view) < 0:
            n = -n
            M = M @ np.diag([-1.0, 1.0, -1.0])
        th = 0.035
        rim_c = c - n * th
        disc, _, _ = self._disc(rim_c, n, r, 20)
        lam = abs(float(n @ light))
        self.poly(d, disc, np.array([150, 100, 30]) * (0.45 + 0.5 * lam))
        lv = int(np.clip(lam * 3.2 + 0.2, 0, 3))
        P4 = np.array([[-r, r, 0], [r, r, 0], [r, -r, 0], [-r, -r, 0]]) @ M.T + c
        self.warp(canvas, self.coin_lv[blur][lv], P4)
        h = _norm(_norm(light) + _norm(view))
        s = max(0.0, float(n @ h)) ** 40
        if s > 0.05:
            glints.append((c, s))

    def frame(self, t):
        R, a = self.R, self.acc
        cpos = np.array([0.6 * math.sin(t * 0.13), 0.1 + 0.25 * math.sin(t * 0.11), 7.6 - 0.9 * (1 - math.exp(-t / 12.0))])
        R.begin(Camera(cpos, (0.0, -0.4, -1.0), fov=58, roll=0.02 * math.sin(t * 0.12)))
        canvas = self.bg.copy()
        d = ImageDraw.Draw(canvas)
        light = _norm([0.35, 0.8, 0.6])
        warm = np.array([255, 205, 120], np.float32)
        # light rays from the top + bokeh
        for k in range(7):
            x0 = -4 + k * 1.3
            A = np.array([[x0, 7, -9]], np.float32)
            B = np.array([[x0 * 0.4 + 0.8, -4, -2]], np.float32)
            R.lines3d(A, B, warm * (0.55 + 0.2 * math.sin(t * 0.4 + k * 1.7)), width=60, bloom=0.0)
        R.points3d(self.bok, self.bok_c * 70, 0.2, 0.0, bloom=0.0)
        # floor (dark, warm reflective)
        F = self.floor
        self.poly(d, [(-12, F, -14), (12, F, -14), (12, F, 8), (-12, F, 8)], (44, 28, 14))
        # collect drawables with depth
        items = []
        y0, y1 = self.span_y
        span = y1 - y0
        for i in range(self.nc):
            y = y1 - ((self.c_pos[i, 1] * span + self.c_v[i] * t) % span)
            c = np.array([self.c_pos[i, 0] + 0.15 * math.sin(t * 0.7 + i), y, self.c_pos[i, 2]])
            items.append((float(np.linalg.norm(c - cpos)), "coin", i, c))
        for i in range(self.nn):
            y = y1 + 0.6 - ((self.n_pos[i, 1] * (span + 1.2) + self.n_v[i] * t) % (span + 1.2))
            c = np.array([self.n_pos[i, 0] + 0.5 * math.sin(t * 0.5 * self.n_w[i, 0] + self.n_ph[i, 0]), y, self.n_pos[i, 2]])
            items.append((float(np.linalg.norm(c - cpos)), "note", i, c))
        for k, (x, z, n, r) in enumerate(self.stacks):
            c = np.array([x, F, z])
            items.append((float(np.linalg.norm(c - cpos)), "stack", k, c))
        items.append((float(np.linalg.norm(np.array([0, F, -0.3]) - cpos)) + 0.5, "heap", 0, None))
        items.sort(key=lambda it: -it[0])
        glints = []
        focus = float(np.linalg.norm(cpos - np.array([0, -0.5, 0])))
        for dist, kind, i, c in items:
            blur = min(2, int(abs(dist - focus) / 2.6))
            if kind == "coin":
                ang = self.c_ph[i] + self.c_w[i] * t
                ax_ = self.c_ax[i]
                K = np.array([[0, -ax_[2], ax_[1]], [ax_[2], 0, -ax_[0]], [-ax_[1], ax_[0], 0]])
                M = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K
                if c[1] < F + 0.2:
                    continue
                self._draw_coin(canvas, d, c, M, self.c_r[i], light, cpos, blur, glints)
                # motion streak above the coin
                A = (c + np.array([0, 0.15, 0]))[None, :].astype(np.float32)
                B = (c + np.array([0, 0.15 + 0.9 * self.c_v[i] / 1.4, 0]))[None, :].astype(np.float32)
                R.lines3d(A, B, warm * 1.4, warm * 0.0, width=3.0, bloom=0.4)
            elif kind == "note":
                ph = self.n_ph[i]
                w = self.n_w[i]
                ry = ph[1] + t * 0.6 * w[1]
                rx = 0.9 * math.sin(ph[2] + t * 0.9 * w[2]) + 0.3
                rz = 0.5 * math.sin(ph[3] + t * 0.7 * w[0])
                M = rot_y(ry) @ rot_x(rx) @ rot_z(rz)
                L, Hh = 0.95, 0.95 * self.NH / self.NW
                xs = np.linspace(-L / 2, L / 2, self.nstrip + 1)
                bend = 0.12 * np.sin(xs * 3.2 + t * 3.0 * w[0] + ph[0])
                nn_ = M @ np.array([0, 0, 1.0])
                front = float(nn_ @ (cpos - c)) > 0
                for s in range(self.nstrip):
                    P4 = np.array([[xs[s], Hh / 2, bend[s]], [xs[s + 1], Hh / 2, bend[s + 1]], [xs[s + 1], -Hh / 2, bend[s + 1]], [xs[s], -Hh / 2, bend[s]]]) @ M.T + c
                    tn = _norm(M @ np.array([-(bend[s + 1] - bend[s]), 0, xs[s + 1] - xs[s]]))
                    lam = abs(float(tn @ light))
                    lv = int(np.clip(lam * 3.4 + 0.1, 0, 3))
                    sidx = s if front else self.nstrip - 1 - s
                    tex = self.note_lv[blur][lv][sidx]
                    if not front:
                        P4 = P4[[1, 0, 3, 2]]
                    self.warp(canvas, tex, P4)
            elif kind == "stack":
                x, z, n, r = self.stacks[i]
                for k in range(n):
                    yc = F + 0.045 * k
                    disc, _, _ = self._disc(np.array([x + 0.01 * math.sin(k * 2.1 + i), yc, z]), [0, 1, 0], r, 22)
                    self.poly(d, disc, np.array([150, 95, 28]) * (0.7 + 0.1 * (k % 2)))
                    top, _, _ = self._disc(np.array([x + 0.01 * math.sin(k * 2.1 + i), yc + 0.035, z]), [0, 1, 0], r, 22)
                    self.poly(d, top, np.array([230, 165, 60]) * (0.9 + 0.08 * (k % 2)))
                ct = np.array([x + 0.01 * math.sin((n - 1) * 2.1 + i), F + 0.045 * (n - 1) + 0.036, z])
                P4 = np.array([[-r, 0, -r], [r, 0, -r], [r, 0, r], [-r, 0, r]]) + ct
                self.warp(canvas, self.coin_lv[blur][3], P4)
                h = _norm(light + _norm(cpos - ct))
                glints.append((ct, max(0.0, h[1]) ** 30 * (0.4 + 0.6 * math.sin(t * 1.3 + i * 1.7) ** 2)))
            else:  # heap of loose coins (sorted far -> near)
                order = np.argsort(-np.linalg.norm(self.loose - cpos, axis=1))
                for j in order:
                    disc, _, _ = self._disc(self.loose[j], self.loose_n[j], self.loose_r[j], 18)
                    lam = max(0.0, float(self.loose_n[j] @ light))
                    self.poly(d, disc - self.loose_n[j] * 0.03, np.array([140, 90, 26]))
                    self.poly(d, disc, np.array([255, 190, 80]) * (0.55 + 0.5 * lam))
                    if j % 3 == 0:
                        glints.append((self.loose[j] + self.loose_n[j] * 0.04, 0.5 * max(0.0, math.sin(t * 1.7 + j * 2.3)) ** 6))
        # glints (sparkles)
        if glints:
            G = np.array([g[0] for g in glints], np.float32)
            s = np.array([g[1] for g in glints], np.float32)
            x, y, z, inv = R.project(G)
            ok = z > 0.3
            sparkle(R, x[ok], y[ok], (14 + 26 * s[ok]) * self.k, (np.array([255, 245, 220], np.float32)[None, :] * (s[ok] * 6 * (0.6 + self.it))[:, None]), bloom=1.0)
        # dust motes + warm glow at the heap
        dp = self.dust + np.stack([0.2 * np.sin(t * 0.3 + self.dust_ph * TAU), 0.2 * np.sin(t * 0.2 + self.dust_ph * 9), 0 * self.dust_ph], 1).astype(np.float32)
        tw = 0.5 + 0.5 * np.sin(t * 1.3 + self.dust_ph * 30)
        R.points3d(dp, warm[None, :] * (1.0 + 2.0 * tw)[:, None], 0.0, 0.8, bloom=0.5)
        x, y, _, _ = R.project(np.array([[0, F + 0.3, -0.3], [0, 4, -8]], np.float32))
        R.soft(x, y, np.stack([warm * (1800 + 1200 * self.it), warm * 2500]))
        return self.fin(canvas, None, canvas_bloom=0.2, bloom=(0.6, 0.55), radii=(1.4, 9.0))


# ======================================================================== 4. analytics dashboard
class AnalyticsDash(Creator):
    """Creator analytics dashboard of glassy panels floating at different depths in perspective:
    follower counter rolling up, KPI tiles (VIEWS / WATCH TIME / RPM / SHARES) counting up with
    sparklines, a big glowing line graph growing (trend = up / down / volatile) with a pulsing head,
    a donut chart sweeping in and a bar chart; generic random numbers; slow camera drift."""
    default_accent = "#36E0A6"
    PX = 230  # texture px per world unit

    def init(self):
        rng = self.rng
        n = 600
        drift = {"up": 0.03, "down": -0.03, "volatile": 0.0}[self.trend]
        vol = 0.11 if self.trend == "volatile" else 0.05
        r = rng.normal(drift, vol, n + 40)
        if self.trend == "volatile":
            r += 0.12 * np.sin(np.arange(n + 40) * 0.11)
        k = np.exp(-np.linspace(-2.5, 2.5, 15) ** 2)
        self.series = np.cumsum(np.convolve(r, k / k.sum(), mode="valid")[:n]).astype(np.float32)
        self.min_span = float(np.median([np.ptp(self.series[i:i + 140]) for i in range(0, n - 140, 20)]))
        self.views = float(rng.uniform(0.4, 9.0)) * 1e6
        self.foll = float(rng.uniform(40, 900)) * 1e3
        self.watch = float(rng.uniform(2, 60)) * 1e3
        self.rpm = float(rng.uniform(1.5, 9.5))
        self.shares = float(rng.uniform(5, 90)) * 1e3
        self.pct = float(rng.uniform(4, 38)) * (-1 if self.trend == "down" else 1)
        self.donut = rng.dirichlet([3, 2, 1.5, 1]).astype(np.float32)
        self.bars = rng.uniform(0.25, 1.0, 7).astype(np.float32)
        if self.trend == "up":
            self.bars = np.sort(self.bars)
        elif self.trend == "down":
            self.bars = np.sort(self.bars)[::-1].copy()
        self.spark = [np.cumsum(rng.normal({"up": 0.15, "down": -0.15}.get(self.trend, 0.0), 0.25 if self.trend != "volatile" else 0.5, 24)) for _ in range(4)]
        self.f_lab = font(26, "Inter-Bold.otf")
        self.f_num = font(58)
        self.f_big = font(92)
        self.f_sm = font(22, "Inter-Bold.otf")
        # panel layout (centre x, y, z, width, height)
        self.panels = {
            "head": (0.0, 3.05, 0.5, 3.9, 1.15),
            "k0": (-0.99, 1.55, 0.25, 1.9, 1.3), "k1": (0.99, 1.55, 0.25, 1.9, 1.3),
            "k2": (-0.99, 0.15, 0.35, 1.9, 1.3), "k3": (0.99, 0.15, 0.35, 1.9, 1.3),
            "graph": (0.0, -1.85, 0.0, 3.9, 2.4),
            "donut": (-0.99, -4.15, 0.45, 1.9, 1.9), "bars": (0.99, -4.15, 0.45, 1.9, 1.9),
        }
        self.masks = {}
        self.glass = {}
        for key, (_, _, _, w, h) in self.panels.items():
            Wt, Ht = int(w * self.PX), int(h * self.PX)
            self.masks[key] = rounded_mask(Wt, Ht, 26)
            yy = np.linspace(0, 1, Ht, dtype=np.float32)[:, None]
            xx = np.linspace(0, 1, Wt, dtype=np.float32)[None, :]
            sheen = np.clip(1 - yy / 0.55, 0, 1) ** 2 * 34 + np.clip(1 - np.abs(xx - yy * 0.4 - 0.15) / 0.12, 0, 1) * 10
            arr = np.zeros((Ht, Wt, 4), np.float32)
            arr[..., 0] = 34 + sheen
            arr[..., 1] = 44 + sheen
            arr[..., 2] = 78 + sheen
            arr[..., 3] = 200
            im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")
            ImageDraw.Draw(im).rounded_rectangle([1, 1, Wt - 2, Ht - 2], radius=26, outline=(180, 210, 255, 120), width=2)
            self.glass[key] = im
        nb = 70
        self.bok = np.stack([rng.uniform(-10, 10, nb), rng.uniform(-9, 9, nb), rng.uniform(-18, -7, nb)], 1).astype(np.float32)
        self.bok_c = lerp_col(self.acc, (120, 150, 255), rng.random(nb)) * (0.2 + 0.8 * rng.random(nb) ** 2)[:, None]
        self.orbs = [(-2.5, 2.0, -6, self.acc), (2.8, -2.5, -7, np.array([110, 90, 255], np.float32)), (0.5, -6, -5, np.array([60, 160, 255], np.float32))]
        bg = vgrad(self.W, self.H, [(0, (16, 22, 44)), (0.5, (26, 30, 62)), (1, (12, 16, 34))])
        self.bg = Image.fromarray(bg)

    def _glass(self, key):
        im = self.glass[key].copy()
        return im, ImageDraw.Draw(im, "RGBA"), im.size[0], im.size[1]

    def frame(self, t):
        R, a = self.R, self.acc
        A_ = tuple(int(v) for v in a) + (255,)
        W_ = (240, 246, 255, 255)
        G_ = (160, 175, 210, 255)
        cpos = np.array([1.6 * math.sin(t * 0.12 + 0.4), -0.4 + 0.6 * math.sin(t * 0.09), 10.2 - 0.4 * math.sin(t * 0.1)])
        R.begin(Camera(cpos, (0.1 * math.sin(t * 0.12), -0.6, 0.0), fov=52, roll=0.02 * math.sin(t * 0.14)))
        self.begin_emit()
        canvas = self.bg.copy()
        # background orbs, floor grid, bokeh
        for (x, y, z, c) in self.orbs:
            px, py, pz, _ = R.project(np.array([[x, y, z]], np.float32))
            R.points(np.repeat(px, 2), np.repeat(py, 2), np.stack([c * 2500, c * 1500]), np.array([90.0, 40.0]) * self.k, bloom=0.0)
        gx = np.arange(-10, 10.1, 1.0, dtype=np.float32)
        A = np.stack([gx, np.full_like(gx, -6.5), np.full_like(gx, -16)], 1)
        B = np.stack([gx, np.full_like(gx, -6.5), np.full_like(gx, 4)], 1)
        gz = np.arange(-16, 4.1, 1.0, dtype=np.float32)
        A2 = np.stack([np.full_like(gz, -10), np.full_like(gz, -6.5), gz], 1)
        B2 = np.stack([np.full_like(gz, 10), np.full_like(gz, -6.5), gz], 1)
        R.lines3d(np.concatenate([A, A2]), np.concatenate([B, B2]), a * 0.35 + 8, width=1.0, bloom=0.4)
        R.points3d(self.bok, self.bok_c * 70, 0.15, 0.0, bloom=0.0)
        grow = 1 - math.exp(-t / 2.5)
        texs = {}
        self._gpts = None
        # ---- header: follower counter
        im, d, Wt, Ht = self._glass("head")
        d.ellipse([28, Ht / 2 - 70, 168, Ht / 2 + 70], fill=A_)
        d.ellipse([70, Ht / 2 - 50, 126, Ht / 2 + 6], fill=(255, 255, 255, 220))
        d.chord([52, Ht / 2 + 4, 144, Ht / 2 + 90], 180, 360, fill=(255, 255, 255, 220))
        d.text((196, 26), "FOLLOWERS", fill=G_, font=self.f_lab)
        fv = self.foll * grow + t * 37
        d.text((192, 58), f"{int(fv):,}", fill=W_, font=self.f_big)
        up = self.pct >= 0
        badge = f"{'+' if up else ''}{self.pct * grow:.1f}%"
        bw = d.textlength(badge, font=self.f_lab) + 60
        bc = A_ if up else (255, 80, 100, 255)
        d.rounded_rectangle([Wt - bw - 30, 34, Wt - 30, 80], radius=22, fill=bc[:3] + (60,), outline=bc, width=2)
        tri = [(Wt - bw - 8, 66), (Wt - bw + 6, 48), (Wt - bw + 20, 66)] if up else [(Wt - bw - 8, 48), (Wt - bw + 6, 66), (Wt - bw + 20, 48)]
        d.polygon(tri, fill=bc)
        d.text((Wt - bw + 26, 40), badge, fill=bc, font=self.f_lab)
        texs["head"] = im
        # ---- KPI tiles
        kpis = [("VIEWS", fmt_count(self.views * grow + t * 211)), ("WATCH TIME", f"{int(self.watch * grow + t * 3):,}H"),
                ("RPM", f"${self.rpm * grow:.2f}"), ("SHARES", fmt_count(self.shares * grow + t * 9))]
        for i, (lab, val) in enumerate(kpis):
            key = f"k{i}"
            im, d, Wt, Ht = self._glass(key)
            d.text((28, 24), lab, fill=G_, font=self.f_lab)
            d.text((26, 62), val, fill=W_, font=self.f_num)
            sp = self.spark[i]
            m = max(2, int(len(sp) * min(1.0, grow * 1.2)))
            sv = sp[:m]
            lo, hi = float(sp.min()), float(sp.max()) + 1e-6
            xs = 28 + np.arange(m) / (len(sp) - 1) * (Wt - 56)
            ys = Ht - 26 - (sv - lo) / (hi - lo) * (Ht * 0.3)
            pts = list(zip(xs.tolist(), ys.tolist()))
            d.polygon(pts + [(xs[-1], Ht - 20), (xs[0], Ht - 20)], fill=A_[:3] + (50,))
            d.line(pts, fill=A_, width=4)
            d.ellipse([xs[-1] - 6, ys[-1] - 6, xs[-1] + 6, ys[-1] + 6], fill=W_)
            texs[key] = im
        # ---- line graph
        im, d, Wt, Ht = self._glass("graph")
        d.text((30, 24), "VIEWS", fill=G_, font=self.f_lab)
        d.text((Wt - 150, 24), "28D", fill=A_, font=self.f_lab)
        x0g, x1g, y0g, y1g = 30, Wt - 30, 80, Ht - 30
        for k in range(5):
            y = y0g + (y1g - y0g) * k / 4
            d.line([(x0g, y), (x1g, y)], fill=(255, 255, 255, 30), width=1)
        N = 140
        head = 55 + t * 8.0
        i1 = int(head)
        i0 = max(0, i1 - N)
        seg = self.series[i0:i1 + 2].astype(np.float64)
        frac = head - i1
        if len(seg) >= 3:
            vals = np.concatenate([seg[:-1], [seg[-2] + (seg[-1] - seg[-2]) * frac]])
            lo, hi = float(vals.min()), float(vals.max())
            if hi - lo < self.min_span * 0.8:
                mid = 0.5 * (hi + lo) + (0.25 if self.trend == "up" else -0.25 if self.trend == "down" else 0) * self.min_span
                lo, hi = mid - self.min_span * 0.4, mid + self.min_span * 0.4
            pad = (hi - lo) * 0.12 + 1e-3
            lo, hi = lo - pad, hi + pad
            nn = len(vals)
            xs = x0g + (np.arange(nn) + (N - (i1 - i0)) * 0) / N * (x1g - x0g)
            ys = y1g - (vals - lo) / (hi - lo) * (y1g - y0g)
            pts = list(zip(xs.tolist(), ys.tolist()))
            d.polygon(pts + [(xs[-1], y1g), (xs[0], y1g)], fill=A_[:3] + (45,))
            d.line(pts, fill=A_, width=5)
            self._gpts = (xs / Wt, ys / Ht)
        texs["graph"] = im
        # ---- donut
        im, d, Wt, Ht = self._glass("donut")
        d.text((28, 22), "TRAFFIC", fill=G_, font=self.f_lab)
        cx, cy, rr = Wt / 2, Ht / 2 + 22, Wt * 0.3
        sweep = 360 * ease_out(t / 3.0)
        cols = [A_, (110, 120, 255, 255), (255, 120, 170, 255), (255, 200, 90, 255)]
        start = -90.0 + t * 4
        for v, c in zip(self.donut, cols):
            ext = float(v) * sweep
            if ext > 0.5:
                d.pieslice([cx - rr, cy - rr, cx + rr, cy + rr], start, start + ext - 2, fill=c)
            start += float(v) * sweep
        d.ellipse([cx - rr * 0.62, cy - rr * 0.62, cx + rr * 0.62, cy + rr * 0.62], fill=(30, 40, 72, 255))
        pc = f"{int(self.donut[0] * 100 * ease_out(t / 3.0))}%"
        d.text((cx - d.textlength(pc, font=self.f_num) / 2, cy - 36), pc, fill=W_, font=self.f_num)
        texs["donut"] = im
        # ---- bars
        im, d, Wt, Ht = self._glass("bars")
        d.text((28, 22), "REVENUE", fill=G_, font=self.f_lab)
        nb = len(self.bars)
        bw = (Wt - 60) / nb
        for k in range(nb):
            g = ease_out((t - 0.3 - k * 0.15) / 1.2) * (0.92 + 0.08 * math.sin(t * 1.3 + k))
            hb = (Ht - 110) * float(self.bars[k]) * g
            x = 30 + k * bw
            c = A_ if k == nb - 1 else (120, 140, 230, 230)
            d.rounded_rectangle([x + 6, Ht - 30 - hb, x + bw - 6, Ht - 30], radius=8, fill=c)
        texs["bars"] = im
        # ---- place panels (far first)
        order = sorted(self.panels.keys(), key=lambda k: -np.linalg.norm(np.array(self.panels[k][:3]) - cpos))
        for key in order:
            x, y, z, w, h = self.panels[key]
            ry = -0.08 * x
            M = rot_y(ry)
            c = np.array([x, y, z])
            P4 = np.array([[-w / 2, h / 2, 0], [w / 2, h / 2, 0], [w / 2, -h / 2, 0], [-w / 2, -h / 2, 0]]) @ M.T + c
            tex = texs[key]
            tex.putalpha(Image.fromarray(np.minimum(np.asarray(tex.getchannel("A")), np.asarray(self.masks[key]))))
            self.warp(canvas, tex, P4, emit=0.8)
            Pf = P4.astype(np.float32)
            R.lines3d(Pf, np.roll(Pf, -1, 0), a * 0.9 + 20, width=1.4, bloom=0.8)
            if key == "graph" and self._gpts is not None:
                u, v = self._gpts
                L = np.stack([(u - 0.5) * w, (0.5 - v) * h, np.full(len(u), 0.03)], 1) @ M.T + c
                L = L.astype(np.float32)
                R.lines3d(L[:-1], L[1:], a * 2.2, width=3.0, bloom=1.5)
                pulse = 0.5 + 0.5 * math.sin(t * 5)
                hx, hy, _, _ = R.project(L[-1:])
                R.points(np.repeat(hx, 3), np.repeat(hy, 3), np.stack([np.full(3, 255.0) * 5, a * 20, a * 260 * (0.6 + 0.4 * pulse)]),
                         np.array([1.5, 5.0, 16.0]) * self.k, bloom=1.0)
                ph = (t * 0.8) % 1.0
                th = np.linspace(0, TAU, 60, endpoint=False)
                ring = L[-1][None, :] + (np.stack([np.cos(th), np.sin(th), np.zeros(60)], 1) @ M.T * (0.05 + 0.25 * ph)).astype(np.float32)
                R.lines3d(ring, np.roll(ring, -1, 0), a * 2.0 * (1 - ph), width=1.4, bloom=0.8)
        return self.finish_emit(canvas, None, gain=0.35, canvas_bloom=0.1, bloom=(0.6, 0.55), radii=(1.4, 9.0))


# ======================================================================== 5. AI studio
class AIStudio(Creator):
    """AI generation interface floating in 3D: prompt box with abstract word-bars being typed and a
    blinking cursor, a pulsing generate button with a sparkle icon and loading dots, and a 2x3 grid of
    image tiles that materialise one after another from noise (diffusion-style: noise -> blur ->
    sharp colourful artwork) with scan lines and sparkle particles; slow camera drift; loops."""
    default_accent = "#A46BFF"
    TS = 224
    CYCLE = 13.0
    T_TYPE, T_PRESS, T_GEN = 1.5, 1.65, 1.9
    OFFSET = 1.2  # seconds into the first cycle at t = 0

    def init(self):
        rng = self.rng
        S = self.TS
        self.sets = []
        for s in range(3):
            arts = []
            for i in range(6):
                art = gen_art(rng, S, (i + s) % 4)
                b1 = np.asarray(Image.fromarray(art.astype(np.uint8)).filter(ImageFilter.GaussianBlur(14)), np.float32)
                b2 = np.asarray(Image.fromarray(art.astype(np.uint8)).filter(ImageFilter.GaussianBlur(5)), np.float32)
                arts.append((art, b1, b2))
            self.sets.append(arts)
        self.noise = [rng.uniform(0, 255, (S, S, 3)).astype(np.float32) for _ in range(5)]
        cn = np.asarray(Image.fromarray(rng.uniform(0, 255, (S // 8, S // 8, 3)).astype(np.uint8)).resize((S, S), Image.BICUBIC), np.float32)
        self.cnoise = cn
        self.tile_mask = rounded_mask(S, S, 18)
        yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) / S
        self.diag = (xx + yy) * 0.5
        ph = Image.fromarray(np.clip(np.dstack([np.full((S, S), 0.0)] * 3) + (np.array([36, 28, 64], np.float32) + self.acc * 0.08)
                                     * (1.1 - 0.3 * yy[..., None]), 0, 255).astype(np.uint8))
        dp = ImageDraw.Draw(ph)
        ic = tint(self.acc * 0.35 + 70)
        dp.rounded_rectangle([S * 0.33, S * 0.36, S * 0.67, S * 0.64], radius=10, outline=ic, width=4)
        dp.polygon([(S * 0.36, S * 0.61), (S * 0.46, S * 0.48), (S * 0.53, S * 0.56), (S * 0.57, S * 0.52), (S * 0.64, S * 0.61)], fill=ic)
        dp.ellipse([S * 0.55, S * 0.4, S * 0.61, S * 0.46], fill=ic)
        self.placeholder = np.asarray(ph, np.float32)
        self.words = []
        for c in range(3):
            ws = []
            x = 0
            line = 0
            while line < 3:
                wl = int(rng.integers(40, 130))
                if x + wl > 760:
                    x, line = 0, line + 1
                    continue
                ws.append((line, x, wl))
                x += wl + 18
            self.words.append(ws)
        self.PW, self.PH = 900, 330
        self.p_mask = rounded_mask(self.PW, self.PH, 30)
        self.f_sm = font(24, "Inter-Bold.otf")
        # grid
        self.tw = 1.62
        self.gap = 0.14
        self.tiles = []
        for i in range(6):
            r, c = divmod(i, 2)
            self.tiles.append(((c - 0.5) * (self.tw + self.gap), 1.95 - r * (self.tw + self.gap) - self.tw / 2))
        nsp = 120
        self.spk = np.stack([rng.uniform(-2.6, 2.6, nsp), rng.uniform(-5, 4.5, nsp), rng.uniform(-1.5, 1.2, nsp)], 1).astype(np.float32)
        self.spk_ph = rng.random(nsp).astype(np.float32)
        self.spk_s = rng.uniform(0.5, 1.0, nsp).astype(np.float32)
        nb = 60
        self.bok = np.stack([rng.uniform(-10, 10, nb), rng.uniform(-9, 9, nb), rng.uniform(-16, -7, nb)], 1).astype(np.float32)
        self.bok_c = lerp_col(self.acc, (90, 160, 255), rng.random(nb)) * (0.2 + 0.8 * rng.random(nb) ** 2)[:, None]
        self.bg = Image.fromarray(vgrad(self.W, self.H, [(0, (26, 16, 50)), (0.5, (40, 22, 70)), (1, (14, 10, 30))]))

    def tile_state(self, i, lt):
        t0 = self.T_GEN + i * 1.2
        return float(np.clip((lt - t0) / 2.2, 0, 1)), t0

    def tile_tex(self, i, lt, cyc, t):
        S = self.TS
        p, _ = self.tile_state(i, lt)
        fade_out = ease((lt - (self.CYCLE - 0.9)) / 0.9)
        art, b1, b2 = self.sets[cyc % 3][i]
        if p <= 0:
            band = np.exp(-((self.diag - ((t * 0.45 + i * 0.17) % 1.6 - 0.3)) / 0.09) ** 2)[..., None] * (self.acc * 0.18 + 22)
            img = self.placeholder + band
        else:
            nz = self.noise[int(t * 20) % 5] * 0.6 + self.cnoise * 0.4
            if p < 0.4:
                u = p / 0.4
                img = nz * (1 - u) + b1 * u
            elif p < 0.75:
                u = (p - 0.4) / 0.35
                img = b1 * (1 - u) + b2 * u + nz * 0.15 * (1 - u)
            else:
                u = (p - 0.75) / 0.25
                img = b2 * (1 - u) + art * u
            if p < 1:  # scan line sweeping down
                yy = np.linspace(0, 1, S, dtype=np.float32)[:, None, None]
                img = img + np.exp(-((yy - p) / 0.025) ** 2) * (self.acc * 0.35 + 50)
        if fade_out > 0:
            img = img * (1 - fade_out) + self.placeholder * fade_out
        im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        if 0 < p < 1:
            d = ImageDraw.Draw(im)
            d.text((14, S - 38), f"{int(p * 100)}%", fill=(255, 255, 255), font=self.f_sm)
        im.putalpha(self.tile_mask)
        return im

    def prompt_tex(self, lt, cyc, t):
        PW, PH = self.PW, self.PH
        a = self.acc
        im = Image.new("RGBA", (PW, PH), (30, 22, 56, 225))
        d = ImageDraw.Draw(im, "RGBA")
        d.rounded_rectangle([1, 1, PW - 2, PH - 2], radius=30, outline=(200, 180, 255, 140), width=2)
        # input field
        d.rounded_rectangle([30, 30, PW - 30, 210], radius=20, fill=(14, 10, 30, 230), outline=tint(a) + (200,), width=2)
        # sparkle icon at the left
        cx, cy = 66, 70
        s = 18
        d.polygon([(cx, cy - s), (cx + 5, cy - 5), (cx + s, cy), (cx + 5, cy + 5), (cx, cy + s), (cx - 5, cy + 5), (cx - s, cy), (cx - 5, cy - 5)], fill=tint(a * 0.6 + 100))
        ws = self.words[cyc % 3]
        typed = ease(lt / self.T_TYPE) * len(ws)
        nfull = int(typed)
        endx, endy = 100, 58
        for k, (line, x, wl) in enumerate(ws):
            if k > nfull:
                break
            w_ = wl if k < nfull else wl * (typed - nfull)
            if w_ < 2:
                continue
            y = 58 + line * 48
            d.rounded_rectangle([100 + x, y, 100 + x + w_, y + 24], radius=10, fill=(225, 220, 245, 235) if k % 5 else tint(a * 0.5 + 120) + (235,))
            endx, endy = 100 + x + w_ + 8, y
        if (t * 2.2) % 1.0 < 0.55 or lt < self.T_TYPE:
            d.rectangle([endx, endy - 4, endx + 4, endy + 28], fill=(255, 255, 255, 255))
        # chips row
        for k, w_ in enumerate((120, 96, 140)):
            x = 30 + sum((120, 96, 140)[:k]) + 16 * k
            d.rounded_rectangle([x, 236, x + w_, 286], radius=25, fill=(60, 48, 100, 220))
            d.rounded_rectangle([x + 22, 256, x + w_ - 22, 266], radius=5, fill=(190, 180, 230, 200))
        # generate button
        press = math.exp(-((lt - self.T_PRESS) / 0.12) ** 2)
        pulse = 0.5 + 0.5 * math.sin(t * 4.0)
        bx0, by0, bx1, by1 = PW - 250, 228, PW - 30, 296
        bc = a * (0.85 + 0.25 * pulse) + 255 * press * 0.5
        d.rounded_rectangle([bx0, by0, bx1, by1], radius=34, fill=tint(bc) + (255,))
        cx, cy = bx0 + 52, (by0 + by1) / 2
        s = 16
        d.polygon([(cx, cy - s), (cx + 4, cy - 4), (cx + s, cy), (cx + 4, cy + 4), (cx, cy + s), (cx - 4, cy + 4), (cx - s, cy), (cx - 4, cy - 4)], fill=(255, 255, 255, 255))
        generating = self.T_PRESS < lt < self.T_GEN + 5 * 1.2 + 2.2
        if generating:
            for k in range(3):
                yy = cy - 6 * max(0.0, math.sin(t * 8 - k * 0.8))
                d.ellipse([cx + 50 + k * 30 - 7, yy - 7, cx + 50 + k * 30 + 7, yy + 7], fill=(255, 255, 255, 255))
        else:
            d.rounded_rectangle([cx + 40, cy - 6, cx + 130, cy + 6], radius=6, fill=(255, 255, 255, 230))
        im.putalpha(Image.fromarray(np.minimum(np.asarray(im.getchannel("A")), np.asarray(self.p_mask))))
        return im, press, pulse, generating

    def frame(self, t):
        R, a = self.R, self.acc
        tc = t + self.OFFSET
        cyc = int(tc // self.CYCLE)
        lt = tc - cyc * self.CYCLE
        cpos = np.array([1.3 * math.sin(t * 0.11 + 0.5), 0.3 + 0.5 * math.sin(t * 0.09), 9.2 - 0.4 * math.sin(t * 0.13)])
        R.begin(Camera(cpos, (0.1 * math.sin(t * 0.11), 0.2, 0.0), fov=54, roll=0.02 * math.sin(t * 0.12)))
        self.begin_emit()
        canvas = self.bg.copy()
        # aurora / orbs + bokeh
        for (x, y, z, c) in ((-3, 3, -7, a), (3, -3, -8, np.array([60, 140, 255], np.float32)), (0, -7, -6, np.array([255, 90, 190], np.float32))):
            px, py, _, _ = R.project(np.array([[x, y, z]], np.float32))
            R.points(np.repeat(px, 2), np.repeat(py, 2), np.stack([c * 3000, c * 1500]), np.array([100.0, 45.0]) * self.k, bloom=0.0)
        R.points3d(self.bok, self.bok_c * 70, 0.15, 0.0, bloom=0.0)
        # back plate behind the whole UI (dark glass)
        d = ImageDraw.Draw(canvas)
        # prompt panel
        tex, press, pulse, generating = self.prompt_tex(lt, cyc, t)
        pw, ph = 3.5, 3.5 * self.PH / self.PW
        c = np.array([0.0, 3.0, 0.6])
        M = rot_y(0.05) @ rot_x(-0.05)
        P4 = np.array([[-pw / 2, ph / 2, 0], [pw / 2, ph / 2, 0], [pw / 2, -ph / 2, 0], [-pw / 2, -ph / 2, 0]]) @ M.T + c
        # tiles (behind the prompt panel in depth: z = 0)
        glows = []
        for i in range(6):
            x, y = self.tiles[i]
            p, t0 = self.tile_state(i, lt)
            pop = 0.25 * ease_out((lt - t0 - 2.2) / 0.5) * (1 - ease((lt - t0 - 3.2) / 1.0)) if p >= 1 else 0.0
            z = 0.0 + pop + 0.15 * (p > 0) * (p < 1)
            hw_ = self.tw / 2
            Mt = rot_y(0.04 * x) @ rot_x(-0.02 * y)
            ct = np.array([x, y, z])
            Q = np.array([[-hw_, hw_, 0], [hw_, hw_, 0], [hw_, -hw_, 0], [-hw_, -hw_, 0]]) @ Mt.T + ct
            self.warp(canvas, self.tile_tex(i, lt, cyc, t), Q, emit=0.9)
            Qf = Q.astype(np.float32)
            if 0 < p < 1:
                g = 1.5 + 0.8 * math.sin(t * 6)
                R.lines3d(Qf, np.roll(Qf, -1, 0), a * g + 40, width=2.0, bloom=1.2)
                glows.append((ct, 1.0))
                # scan line light
                yl = hw_ - 2 * hw_ * p
                L = (np.array([[-hw_, yl, 0.01], [hw_, yl, 0.01]]) @ Mt.T + ct).astype(np.float32)
                R.lines3d(L[:1], L[1:], a * 0.8 + 50, width=2.0, bloom=1.0)
            else:
                R.lines3d(Qf, np.roll(Qf, -1, 0), (a * 0.45 + 30) * (1.0 if p >= 1 else 0.5), width=1.0, bloom=0.4)
            if p >= 1 and lt - t0 - 2.2 < 0.8:
                glows.append((ct, 1 - (lt - t0 - 2.2) / 0.8))
        self.warp(canvas, tex, P4, emit=0.6)
        Pf = P4.astype(np.float32)
        R.lines3d(Pf, np.roll(Pf, -1, 0), a * 1.0 + 30, width=1.4, bloom=0.8)
        # button glow (button centre in texture px -> world)
        bu, bv = (self.PW - 140) / self.PW, 262 / self.PH
        bpos = (np.array([[(bu - 0.5) * pw, (0.5 - bv) * ph, 0.03]]) @ M.T + c).astype(np.float32)
        bx, by, _, _ = R.project(bpos)
        R.points(np.repeat(bx, 2), np.repeat(by, 2), np.stack([a * (300 + 500 * pulse + 3000 * press), a * (800 + 6000 * press)]),
                 np.array([20.0, 60.0]) * self.k, bloom=0.5)
        # sparkles: ambient rising + bursts around active tiles
        u = (t * 0.06 * self.spk_s + self.spk_ph) % 1.0
        P = self.spk + np.stack([0.15 * np.sin(t * 0.5 + self.spk_ph * TAU), u * 3.0 - 1.5, 0 * u], 1).astype(np.float32)
        x, y, z, inv = R.project(P)
        tw = np.maximum(0, np.sin(t * 2.0 + self.spk_ph * 40)) ** 4 * np.sin(np.pi * u)
        m = (tw > 0.05) & (z > 0.5)
        sparkle(R, x[m], y[m], (6 + 16 * tw[m]) * self.k, (lerp_col(a, (255, 255, 255), 0.5) * (2.5 * tw[m])[:, None]), bloom=0.8)
        for (ct, g) in glows:
            x, y, _, _ = R.project(ct[None, :].astype(np.float32))
            R.soft(np.repeat(x, 2), np.repeat(y, 2), np.stack([a * 1500 * g, np.array([255, 255, 255], np.float32) * 300 * g]))
            for k in range(6):
                ang = k / 6 * TAU + t * 0.8
                rr = 0.95 + 0.12 * math.sin(t * 3 + k)
                sp = ct + np.array([rr * math.cos(ang), rr * math.sin(ang), 0.1])
                sx, sy, _, _ = R.project(sp[None, :].astype(np.float32))
                tw_ = max(0.0, math.sin(t * 5 + k * 1.3)) ** 2 * g
                sparkle(R, sx, sy, (8 + 14 * tw_) * self.k, lerp_col(a, (255, 255, 255), 0.6) * 3 * tw_, bloom=0.8)
        return self.finish_emit(canvas, None, gain=0.4, canvas_bloom=0.1, bloom=(0.6, 0.55), radii=(1.4, 9.0))


# ======================================================================== 6. creator desk
class CreatorDesk(Creator):
    """Content-creator filming setup: big glowing ring light with a phone mounted in its centre
    (screen shows a recording view with a pulsing red record dot), desk with a monitor running an
    editing timeline, keyboard, microphone on a boom arm, hexagon acoustic panels and an accent LED
    strip on the wall, a neon play-icon sign, shelf with a plant; warm key light from the ring light
    and cool accent rim light; slow orbit camera."""
    default_accent = "#7A5CFF"

    def init(self):
        rng = self.rng
        a = self.acc
        self.f_rec = font(30, "Inter-Bold.otf")
        self.PW, self.PH = 220, 440
        self.ph_mask = rounded_mask(self.PW, self.PH, 26)
        self.person = clip_art(0, self.PW, self.PH, rng, a)
        self.MW, self.MH = 520, 300
        mon = Image.new("RGB", (self.MW, self.MH), (24, 26, 36))
        dm = ImageDraw.Draw(mon)
        art = clip_art(1, 300, 170, rng, a)
        mon.paste(art, (110, 16))
        cols = [a, (140, 90, 255), (40, 200, 210), (255, 110, 150), (255, 170, 70)]
        for row in range(3):
            x = 12
            while x < self.MW - 12:
                w = int(rng.integers(40, 130))
                c = cols[int(rng.integers(0, len(cols)))] if row < 2 else (60, 200, 120)
                dm.rectangle([x, 200 + row * 30, min(x + w, self.MW - 12), 222 + row * 30], fill=tint(np.asarray(c) * 0.8))
                x += w + 3
        for k in range(6):
            dm.rectangle([10, 16 + k * 28, 96, 38 + k * 28], fill=tint(np.asarray(cols[k % 5]) * 0.5 + 20))
            dm.rectangle([424, 16 + k * 28, 510, 38 + k * 28], fill=(44, 48, 64))
        self.mon_base = mon
        # hexagon acoustic panels on the back wall
        hx = []
        for r in range(4):
            for c in range(5):
                x = -1.5 + c * 0.42 + (r % 2) * 0.21
                y = 1.55 + r * 0.36
                if rng.random() < 0.8:
                    hx.append((x, y, rng.random() < 0.3))
        self.hexes = hx
        nb = 40
        self.bok = np.stack([rng.uniform(-6, 6, nb), rng.uniform(0, 4, nb), rng.uniform(-6, -3, nb)], 1).astype(np.float32)
        self.bok_c = lerp_col(a, (255, 200, 150), rng.random(nb)) * (0.2 + 0.6 * rng.random(nb) ** 2)[:, None]
        nd = 200
        self.dust = np.stack([rng.uniform(-2, 2, nd), rng.uniform(0.2, 2.8, nd), rng.uniform(-1.4, 1.5, nd)], 1).astype(np.float32)
        self.dust_ph = rng.random(nd).astype(np.float32)
        self.leaves = [(rng.uniform(-0.12, 0.12), rng.uniform(0.0, 0.35), rng.uniform(0.07, 0.12), rng.uniform(-0.8, 0.8)) for _ in range(14)]
        self.bg = Image.fromarray(vgrad(self.W, self.H, [(0, (30, 24, 44)), (1, (20, 16, 28))]))

    def phone_tex(self, t):
        PW, PH = self.PW, self.PH
        im = self.person.copy()
        d = ImageDraw.Draw(im, "RGBA")
        for x in (PW / 3, 2 * PW / 3):
            d.line([(x, 40), (x, PH - 70)], fill=(255, 255, 255, 60), width=1)
        for y in (PH / 3, 2 * PH / 3):
            d.line([(0, y), (PW, y)], fill=(255, 255, 255, 60), width=1)
        d.rectangle([0, 0, PW, 36], fill=(0, 0, 0, 120))
        on = 0.5 + 0.5 * math.sin(t * 5)
        d.ellipse([PW / 2 - 56, 10, PW / 2 - 40, 26], fill=(255, 40, 50, int(120 + 135 * on)))
        secs = int(t) + 7
        d.text((PW / 2 - 34, 4), f"00:{secs // 60:02d}:{secs % 60:02d}", fill=(255, 255, 255, 255), font=font(20, "Inter-Bold.otf"))
        d.rectangle([0, PH - 76, PW, PH], fill=(0, 0, 0, 110))
        cx, cy = PW / 2, PH - 38
        d.ellipse([cx - 26, cy - 26, cx + 26, cy + 26], outline=(255, 255, 255, 255), width=4)
        d.rounded_rectangle([cx - 11, cy - 11, cx + 11, cy + 11], radius=4, fill=(255, 40, 50, 255))
        im = im.convert("RGBA")
        im.putalpha(self.ph_mask)
        return im

    def monitor_tex(self, t):
        im = self.mon_base.copy()
        d = ImageDraw.Draw(im)
        x = 12 + ((t * 40) % (self.MW - 24))
        d.line([(x, 194), (x, 290)], fill=tint(self.acc * 0.5 + 127), width=3)
        d.polygon([(x - 7, 188), (x + 7, 188), (x, 196)], fill=tint(self.acc * 0.5 + 127))
        im = im.convert("RGBA")
        return im

    def frame(self, t):
        R, a = self.R, self.acc
        az = 0.32 * math.sin(t * 0.12) + 0.12
        tgt = np.array([0.5, 1.12, -0.6])
        dist = 3.9 - 0.2 * math.sin(t * 0.1)
        cpos = tgt + np.array([dist * math.sin(az), 0.3 + 0.1 * math.sin(t * 0.15), dist * math.cos(az)])
        R.begin(Camera(cpos, tgt, fov=58, roll=0.0))
        self.begin_emit()
        canvas = self.bg.copy()
        d = ImageDraw.Draw(canvas)
        ring_c = np.array([-0.2, 1.42, 0.1])
        ringR = 0.4
        Lr = ring_c + np.array([0, 0, 0.3])
        cool = a * 0.8 + 30
        warm = np.array([255, 232, 205], np.float32)

        def lit(base, nrm, P):
            ctr = np.mean(np.asarray(P, np.float64), 0)
            ld = Lr - ctr
            dd = float(np.linalg.norm(ld))
            ld = ld / dd
            n = np.asarray(nrm, np.float64)
            key = max(0.0, float(n @ ld)) * 1.6 / (1 + 0.25 * dd * dd)
            rim = max(0.0, float(n @ _norm([0.3, 0.4, -1.0]))) * 0.9 + max(0.0, float(n @ _norm([0.9, 0.2, -0.4]))) * 0.3
            v = _norm(cpos - ctr)
            h = _norm(ld + v)
            spec = max(0.0, float(n @ h)) ** 24 * 0.5
            b = np.asarray(base, np.float32)
            return b * (0.3 + key) * (warm / 255) + cool * rim * (b / 255 * 0.6 + 0.25) + warm * spec * 0.4

        def box(b, base, top=None):
            for quad, n in self.box_faces(b, cpos):
                self.poly(d, quad, lit(top if (top is not None and n[1] > 0) else base, n, quad))

        # room: floor, back wall, left wall
        self.poly(d, [(-4, 0, -1.6), (4, 0, -1.6), (4, 0, 4), (-4, 0, 4)], (52, 40, 40))
        for k in range(-8, 9):  # floor boards
            self.poly(d, [(k * 0.45, 0.001, -1.6), (k * 0.45 + 0.012, 0.001, -1.6), (k * 0.45 + 0.012, 0.001, 4), (k * 0.45, 0.001, 4)], (40, 30, 30))
        rug = rounded_rect_xy(0.9, 0.6, 0.25, 5)
        rug_p = np.column_stack([rug[:, 0] + 0.55, np.full(len(rug), 0.004), rug[:, 1] + 0.35])
        self.poly(d, rug_p, a * 0.22 + 34)
        self.poly(d, np.column_stack([rug[:, 0] * 0.85 + 0.55, np.full(len(rug), 0.005), rug[:, 1] * 0.8 + 0.35]), None, outline=a * 0.3 + 60, width=2)
        self.poly(d, [(-4, 0, -1.6), (4, 0, -1.6), (4, 3.4, -1.6), (-4, 3.4, -1.6)], (50, 44, 70))
        # smooth light pools from the ring light on wall and floor (additive, very soft)
        soft = np.zeros((R.Hs, R.Ws, 3), np.float32)
        amber = np.array([255, 180, 110], np.float32)
        self.pool(soft, [ring_c[0] + 0.2, 1.4, -1.58], 0.8, amber * 0.16)
        self.pool(soft, [ring_c[0] + 0.5, 0.0, 0.4], 0.9, amber * 0.14, squash=0.4)
        self.pool(soft, [1.5, 2.5, -1.58], 0.7, cool * 0.12)
        self.pool(soft, [0.8, 0.95, -1.58], 1.2, cool * 0.1, squash=0.45)
        # wall wash from the LED strip (soft gradient polygons)
        for k in range(6):
            yb = 0.8 + k * 0.12
            col = lerp_col((46, 40, 62), cool, 0.35 * (1 - k / 6))
            self.poly(d, [(-1.9, yb, -1.595), (2.4, yb, -1.595), (2.4, yb + 0.13, -1.595), (-1.9, yb + 0.13, -1.595)], col)
        # hexagon panels
        for (x, y, hi) in self.hexes:
            P = [(x + 0.19 * math.cos(k * TAU / 6 + math.pi / 6), y + 0.19 * math.sin(k * TAU / 6 + math.pi / 6), -1.58) for k in range(6)]
            base = a * 0.55 + 20 if hi else np.array([56, 50, 72])
            self.poly(d, P, lit(base, (0, 0, 1), P), outline=(30, 26, 40))
        # neon play-icon sign (dark backing)
        sx, sy, sz = 1.35, 2.45, -1.57
        # shelf + plant + small boxes
        box((0.6, 2.0, 1.25, 1.29, -1.6, -1.35), np.array([120, 84, 56]))
        box((0.75, 0.95, 1.29, 1.5, -1.55, -1.4), np.array([200, 80, 90]))
        box((0.98, 1.1, 1.29, 1.44, -1.55, -1.4), np.array([80, 140, 220]))
        box((1.6, 1.8, 1.29, 1.42, -1.55, -1.4), np.array([230, 220, 210]))
        # desk
        wood = np.array([150, 110, 80])
        box((-0.15, 1.75, 0.0, 0.72, -1.45, -1.38), np.array([60, 50, 50]))
        for lx in (-0.1, 1.68):
            box((lx, lx + 0.05, 0.0, 0.72, -1.4, -0.65), np.array([50, 50, 56]))
        box((-0.2, 1.8, 0.72, 0.77, -1.5, -0.55), wood, wood * 1.1)
        # plant pot on the desk + leaves
        box((1.42, 1.62, 0.77, 0.95, -1.35, -1.15), np.array([220, 210, 200]))
        for (lx, ly, ls, ang) in self.leaves:
            c0 = np.array([1.52 + lx, 0.98 + ly, -1.25])
            P = [c0, c0 + np.array([ls * math.cos(ang + 1.2), ls * 1.4, 0.02]), c0 + np.array([ls * math.cos(ang) * 1.6, ls * 2.2, 0.04]),
                 c0 + np.array([ls * math.cos(ang - 1.2), ls * 1.2, 0.02])]
            self.poly(d, P, lit(np.array([60, 160, 80]) * (0.8 + 0.3 * math.sin(ang * 3)), (0, 0.3, 1), P))
        # monitor: stand + body + screen
        box((0.78, 0.86, 0.77, 1.05, -1.32, -1.26), np.array([70, 70, 80]))
        box((0.6, 1.04, 0.77, 0.79, -1.36, -1.2), np.array([70, 70, 80]))
        mx0, mx1, my0, my1, mz = 0.22, 1.42, 0.98, 1.67, -1.22
        box((mx0, mx1, my0, my1, mz - 0.05, mz), np.array([34, 34, 42]))
        self.warp(canvas, self.monitor_tex(t), [(mx0 + 0.03, my1 - 0.03, mz + 0.003), (mx1 - 0.03, my1 - 0.03, mz + 0.003), (mx1 - 0.03, my0 + 0.03, mz + 0.003),
                                                (mx0 + 0.03, my0 + 0.03, mz + 0.003)], emit=1.0)
        # keyboard + mouse
        box((0.4, 1.15, 0.77, 0.8, -0.95, -0.75), np.array([46, 46, 54]), np.array([70, 70, 84]))
        box((1.3, 1.4, 0.77, 0.8, -0.9, -0.75), np.array([46, 46, 54]), np.array([70, 70, 84]))
        # mic on a boom arm
        arm = [np.array([1.72, 0.77, -1.4]), np.array([1.72, 1.32, -1.35]), np.array([1.25, 1.42, -0.85]), np.array([1.08, 1.2, -0.72])]
        for p0, p1 in zip(arm[:-1], arm[1:]):
            self.cylinder(d, p0, p1, 0.015, np.array([60, 60, 66]), _norm(Lr - p0), cpos, n=8, cap=False)
        mic0, mic1 = np.array([1.08, 1.24, -0.72]), np.array([1.02, 1.02, -0.66])
        self.cylinder(d, mic0 + (mic0 - mic1) * 0.15, mic1, 0.055, np.array([40, 40, 46]), _norm(Lr - mic0), cpos, n=14, spec=0.6)
        # grille rings
        for k in range(5):
            p = mic0 + (mic1 - mic0) * (0.1 + 0.18 * k)
            th = np.linspace(0, TAU, 20, endpoint=False)
            ax_ = _norm(mic1 - mic0)
            u = _norm(np.cross(ax_, [1, 0, 0]))
            v = np.cross(ax_, u)
            Pr = p + 0.057 * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * v)
            R.lines3d(Pr.astype(np.float32), np.roll(Pr, -1, 0).astype(np.float32), np.array([120, 120, 130], np.float32) * 0.3, width=1.0, bloom=0.0)
        # ring light: stand + tripod, ring body, phone in the middle
        base_c = np.array([ring_c[0], 0.0, ring_c[2]])
        for k in range(3):
            ang = k * TAU / 3 + 0.5
            foot = base_c + np.array([0.35 * math.cos(ang), 0.0, 0.35 * math.sin(ang)])
            self.cylinder(d, foot, base_c + np.array([0, 0.45, 0]), 0.012, np.array([40, 40, 44]), _norm(Lr - foot), cpos, n=6, cap=False)
        self.cylinder(d, base_c + np.array([0, 0.4, 0]), ring_c - np.array([0, ringR + 0.02, 0]), 0.018, np.array([50, 50, 56]), _norm([0.3, 0.5, 0.8]), cpos, n=8)
        # phone + holder in the middle of the ring
        self.cylinder(d, ring_c - np.array([0, ringR - 0.02, 0]), ring_c - np.array([0, 0.17, 0]), 0.01, np.array([50, 50, 56]), _norm([0.3, 0.5, 0.8]), cpos, n=6, cap=False)
        pw_, ph_ = 0.085, 0.17
        pc = ring_c + np.array([0, 0, 0.01])
        box((pc[0] - pw_ - 0.006, pc[0] + pw_ + 0.006, pc[1] - ph_ - 0.006, pc[1] + ph_ + 0.006, pc[2] - 0.02, pc[2]), np.array([30, 30, 36]))
        self.warp(canvas, self.phone_tex(t), [(pc[0] - pw_, pc[1] + ph_, pc[2] + 0.002), (pc[0] + pw_, pc[1] + ph_, pc[2] + 0.002),
                                              (pc[0] + pw_, pc[1] - ph_, pc[2] + 0.002), (pc[0] - pw_, pc[1] - ph_, pc[2] + 0.002)], emit=1.0)
        # ring body: annulus quads (opaque), facing +z
        th = np.linspace(0, TAU, 49)
        ro, ri = ringR + 0.045, ringR - 0.045
        for k in range(48):
            a0, a1 = th[k], th[k + 1]
            P = [ring_c + np.array([ro * math.cos(a0), ro * math.sin(a0), 0]), ring_c + np.array([ro * math.cos(a1), ro * math.sin(a1), 0]),
                 ring_c + np.array([ri * math.cos(a1), ri * math.sin(a1), 0]), ring_c + np.array([ri * math.cos(a0), ri * math.sin(a0), 0])]
            self.poly(d, P, (236, 232, 226))
        # ---- light
        on = 0.97 + 0.03 * math.sin(t * 0.7)
        Pring = np.stack([ring_c[0] + ringR * np.cos(th[:-1]), ring_c[1] + ringR * np.sin(th[:-1]), np.full(48, ring_c[2] + 0.01)], 1).astype(np.float32)
        R.lines3d(Pring, np.roll(Pring, -1, 0), warm * 9 * on, width=5.0, bloom=2.5)
        R.lines3d(Pring, np.roll(Pring, -1, 0), warm * 4 * on, width=2.0, bloom=0.0)
        rx, ry, _, rinv = R.project(ring_c[None, :].astype(np.float32))
        R.soft(np.repeat(rx, 3), np.repeat(ry, 3), np.stack([warm * 2200, warm * 1500, warm * 900]))
        # LED strip on the wall behind the desk + neon play sign
        led = np.array([[-1.9, 0.86, -1.59], [2.4, 0.86, -1.59]], np.float32)
        breathe = 0.85 + 0.15 * math.sin(t * 1.1)
        R.lines3d(led[:1], led[1:], cool * 5 * breathe, width=3.0, bloom=2.0)
        R.lines3d(np.array([[-2.19, 0.05, -1.5]], np.float32), np.array([[-2.19, 0.05, 3.0]], np.float32), cool * 3, width=2.0, bloom=1.5)
        sign = []
        w_, h_, r_ = 0.36, 0.25, 0.07
        rr = rounded_rect_xy(w_, h_, r_, 5)
        sign = np.column_stack([rr[:, 0] + sx, rr[:, 1] + sy, np.full(len(rr), sz)]).astype(np.float32)
        red = np.array([255, 50, 80], np.float32) * 0.6 + a * 0.4
        flick = 1.0 if (t * 0.37) % 1.0 > 0.04 else 0.4
        R.lines3d(sign, np.roll(sign, -1, 0), red * 4 * flick, width=2.5, bloom=1.8)
        tri = np.array([[sx - 0.09, sy + 0.13, sz], [sx + 0.14, sy, sz], [sx - 0.09, sy - 0.13, sz]], np.float32)
        R.lines3d(tri, np.roll(tri, -1, 0), np.array([255, 255, 255], np.float32) * 3 * flick + red, width=2.5, bloom=1.8)
        x, y, _, _ = R.project(np.array([[sx, sy, sz]], np.float32))
        R.soft(np.repeat(x, 2), np.repeat(y, 2), np.stack([red * 1600 * flick, red * 800 * flick]))
        # monitor glow + desk lamp-ish spill
        x, y, _, _ = R.project(np.array([[0.82, 1.3, -1.1]], np.float32))
        R.soft(x, y, (a * 0.5 + 100)[None, :] * 900)
        # dust in the ring-light beam + bokeh
        dp = self.dust + np.stack([0.1 * np.sin(t * 0.3 + self.dust_ph * TAU), 0.08 * np.sin(t * 0.2 + self.dust_ph * 9), 0.1 * np.cos(t * 0.25 + self.dust_ph * 5)], 1).astype(np.float32)
        tw = 0.5 + 0.5 * np.sin(t * 1.5 + self.dust_ph * 30)
        R.points3d(dp, warm[None, :] * (0.6 + 1.2 * tw)[:, None], 0.0, 0.7, bloom=0.4)
        R.points3d(self.bok, self.bok_c * 40, 0.1, 0.0, bloom=0.0)
        return self.finish_emit(canvas, soft, gain=0.5, canvas_bloom=0.15, bloom=(0.6, 0.55), radii=(1.4, 9.0))


KINDS_CREATOR = {
    "phone_feed": PhoneFeed, "social_feed": PhoneFeed, "short_video": PhoneFeed, "smartphone": PhoneFeed,
    "social_media": PhoneFeed, "reels": PhoneFeed, "scrolling": PhoneFeed,
    "edit_timeline": EditTimeline, "video_editing": EditTimeline, "editing": EditTimeline, "clipping": EditTimeline,
    "video_editor": EditTimeline, "timeline_edit": EditTimeline,
    "money_rain": MoneyRain, "money": MoneyRain, "coins": MoneyRain, "cash": MoneyRain, "income": MoneyRain,
    "monetization": MoneyRain, "payout": MoneyRain,
    "analytics_dash": AnalyticsDash, "analytics": AnalyticsDash, "dashboard": AnalyticsDash, "creator_stats": AnalyticsDash,
    "growth": AnalyticsDash, "metrics": AnalyticsDash,
    "ai_studio": AIStudio, "ai_generator": AIStudio, "ai_tools": AIStudio, "ai_video": AIStudio, "ai_art": AIStudio,
    "image_generator": AIStudio, "prompt": AIStudio,
    "creator_desk": CreatorDesk, "creator_setup": CreatorDesk, "ring_light": CreatorDesk, "ugc": CreatorDesk,
    "filming": CreatorDesk, "content_creator": CreatorDesk, "youtuber_setup": CreatorDesk,
}

_procedural.KINDS.update(KINDS_CREATOR)
