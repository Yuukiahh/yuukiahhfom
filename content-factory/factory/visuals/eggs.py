"""Egg-stealing / pet-hatching tycoon game 3D backgrounds (cute low-poly toy style).

Generic, ORIGINAL visuals for videos about egg-heist tycoon games: floating biome islands with guarded
nests, a blocky runner sprinting home with a stolen egg, an egg hatching into a pet, and a tycoon base
full of pets earning coins. No game / platform names, logos, avatar likenesses or copied assets: the
characters are plain cubes with dot eyes, the pets are original round blobs.

Same architecture as :mod:`.creator`: real geometry seen through a moving perspective
:class:`~.three_d.Camera`; opaque surfaces are flat-shaded PIL polygons sorted in painter's order
(one global draw list per frame, so boxes, meshes and sprites interleave correctly); smooth glossy
eggs and pets are pre-shaded RGBA sprites (numpy, in ``setup``) that are squashed / stretched /
rotated per frame; light (glows, god rays, sparkles, lasers, speed lines) is additive splatting
with bloom. All randomness is drawn in ``setup()``; ``frame(t)`` is a pure function of ``t``.
Every scene takes ``accent`` ("#RRGGBB"). Registered into ``procedural.KINDS`` (``KINDS_EGGS``).
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import procedural as _procedural  # noqa: F401  (load the registry + renderer chain first)
from .creator import Creator, ease, ease_out, heart_xy, sparkle
from .three_d import TAU, Camera, _norm, lerp_col, rot_x, rot_y, rot_z
from .topical import font, tint, vgrad

# ======================================================================== palette / params
RARITIES = ("common", "rare", "epic", "legendary")
RARITY_ALIASES = {
    "common": "common", "white": "common", "basic": "common", "normal": "common", "uncommon": "common",
    "rare": "rare", "blue": "rare",
    "epic": "epic", "purple": "epic", "violet": "epic",
    "legendary": "legendary", "legend": "legendary", "gold": "legendary", "golden": "legendary",
    "mythic": "legendary", "mythical": "legendary", "secret": "legendary",
}
# base shell colour, speckle colour, band colour (or None), glow colour
EGG_PAL = {
    "common": ((246, 240, 226), (186, 140, 96), None, (255, 250, 230)),
    "rare": ((84, 170, 255), (236, 250, 255), (36, 96, 220), (80, 175, 255)),
    "epic": ((172, 92, 250), (255, 150, 236), (255, 214, 120), (200, 90, 255)),
    "legendary": ((255, 200, 52), (255, 250, 214), (255, 128, 30), (255, 196, 50)),
}
PET_COLS = [(255, 150, 190), (120, 228, 186), (120, 190, 255), (192, 152, 255), (255, 182, 120), (255, 226, 108),
            (255, 124, 116), (86, 212, 222), (176, 232, 110), (255, 160, 230), (150, 170, 255), (250, 250, 250)]


def parse_rarity(value, default="legendary") -> str:
    return RARITY_ALIASES.get(str(value).strip().lower(), default) if value is not None else default


def parse_unit(value, default) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return float(np.clip(v, 0.0, 1.0))


# ======================================================================== motion helpers
def hop(t, period, height, air=0.62, phase=0.0):
    """Bouncing hop. Returns (y offset, scale_x, scale_y): stretched in the air, squashed on landing."""
    p = ((t / period) + phase) % 1.0
    if p < air:
        q = p / air
        y = height * 4 * q * (1 - q)
        st = 0.10 * abs(1 - 2 * q) * (1 if height > 0 else 0)
        return y, 1 - st * 0.6, 1 + st
    s = math.sin(math.pi * (p - air) / (1 - air))
    return 0.0, 1 + 0.16 * s, 1 - 0.20 * s


def elastic(x):
    x = min(1.0, max(0.0, float(x)))
    if x <= 0 or x >= 1:
        return x
    return 2 ** (-9 * x) * math.sin((x * 10 - 0.75) * TAU / 3) + 1


def back_out(x, s=2.2):
    x = min(1.0, max(0.0, float(x))) - 1
    return 1 + x * x * ((s + 1) * x + s)


def hsh(i, k=0):
    """Deterministic pseudo-random 0..1 from integers (for infinite procedural tiling)."""
    v = math.sin(i * 127.1 + k * 311.7 + 74.7) * 43758.5453
    return v - math.floor(v)


# ======================================================================== sprites (setup time)
def _sphere_normals(u, v, a):
    """Normals of an egg/ellipsoid front surface given normalised sprite coords."""
    f = (u / a) ** 2 + v ** 2
    nz = np.sqrt(np.clip(1 - f, 0, 1))
    n = np.stack([u / a, v, nz + 0.05], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6
    return f, n


def _light(n, albedo, rim_col, spec=1.0, metal=0.0, L=(-0.5, 0.62, 0.62)):
    L = _norm(L).astype(np.float32)
    Hh = _norm(np.array(L) + np.array([0, 0, 1.0])).astype(np.float32)
    diff = np.clip((n * L).sum(-1), 0, 1)[..., None]
    nh = np.clip((n * Hh).sum(-1), 0, 1)[..., None]
    fres = (1 - np.clip(n[..., 2:3], 0, 1)) ** 2.5
    bounce = np.clip(-n[..., 1:2], 0, 1) * 0.18
    col = albedo * (0.50 + 0.62 * diff + bounce)
    col += np.asarray(rim_col, np.float32) * fres * 0.55
    sc = np.array([255, 255, 255], np.float32) * (1 - metal) + albedo * metal * 1.2
    col += sc * (nh ** 110 * 1.25 + nh ** 14 * 0.13) * spec
    return col


def egg_sprite(Hpx: int, rarity: str, rng) -> Image.Image:
    """Glossy speckled egg (RGBA), height Hpx, lit from the upper left."""
    H = max(8, int(Hpx))
    W = max(6, int(H * 0.80))
    base, speck, band, glow = (np.asarray(c, np.float32) if c is not None else None for c in EGG_PAL[rarity])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    R_ = H / 2 * 0.97
    u = (xx + 0.5 - W / 2) / R_
    v = -(yy + 0.5 - H / 2) / R_
    a = 0.74 * (1 - 0.14 * v)
    f, n = _sphere_normals(u, v, a)
    alpha = np.clip((1 - f) * R_ * 0.5, 0, 1)
    alb = np.broadcast_to(base, (H, W, 3)).copy()
    # zig-zag band around the shell
    if band is not None:
        rad = a * np.sqrt(np.clip(1 - v ** 2, 1e-4, 1))
        lon = np.arcsin(np.clip(u / rad, -1, 1))
        for v0, amp, w, k in ((0.08, 0.07, 0.05, 5.0), (-0.32, 0.05, 0.03, 7.0)):
            tri = np.abs(((lon * k / math.pi) % 2) - 1) * 2 - 1
            m = np.clip((w - np.abs(v - v0 - amp * tri)) * R_ * 0.6, 0, 1)[..., None]
            alb = alb * (1 - m) + band * m
    # speckles / spots mapped onto the sphere (foreshortened)
    n_sp = {"common": 34, "rare": 16, "epic": 18, "legendary": 12}[rarity]
    for _ in range(n_sp):
        lon0, lat0 = rng.uniform(-1.25, 1.25), rng.uniform(-0.95, 0.95)
        if band is not None and abs(lat0 - 0.08) < 0.14:
            lat0 = lat0 + 0.3 * np.sign(lat0 - 0.08 + 1e-3)
        py = math.sin(lat0)
        px = math.sin(lon0) * math.cos(lat0)
        a0 = 0.74 * (1 - 0.14 * py)
        cu, cv = a0 * px, py
        r = rng.uniform(0.035, 0.085) if rarity == "common" else rng.uniform(0.06, 0.11)
        rx, ry = r * max(0.25, math.cos(lon0)), r * max(0.3, math.cos(lat0))
        x0, x1 = int(max(0, (cu - rx * 1.6) * R_ + W / 2)), int(min(W, (cu + rx * 1.6) * R_ + W / 2 + 2))
        y0, y1 = int(max(0, H / 2 - (cv + ry * 1.6) * R_)), int(min(H, H / 2 - (cv - ry * 1.6) * R_ + 2))
        if x1 <= x0 or y1 <= y0:
            continue
        du = (u[y0:y1, x0:x1] - cu) / rx
        dv = (v[y0:y1, x0:x1] - cv) / ry
        if rarity == "legendary":  # diamonds
            dist = np.abs(du) + np.abs(dv)
        elif rarity == "common":  # irregular blots
            ang = np.arctan2(dv, du)
            dist = np.hypot(du, dv) * (1 + 0.25 * np.sin(ang * 3 + rng.uniform(0, TAU)))
        else:
            dist = np.hypot(du, dv)
        m = np.clip((1 - dist) * r * R_ * 0.8, 0, 1)[..., None]
        sc = speck * rng.uniform(0.85, 1.05)
        alb[y0:y1, x0:x1] = alb[y0:y1, x0:x1] * (1 - m) + sc * m
    metal = 0.55 if rarity == "legendary" else 0.0
    col = _light(n, alb, glow * 0.8 + 50, spec=1.0, metal=metal)
    # soft occlusion at the very bottom (sits in a nest / on a pedestal)
    col *= (1 - 0.25 * np.clip((-v - 0.55) / 0.45, 0, 1))[..., None]
    out = np.dstack([np.clip(col, 0, 255), alpha * 255]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def pet_sprite(S: int, body, kind: int, rng, blink=False, accent=(255, 90, 160)) -> Image.Image:
    """Original cute round blob creature (RGBA, ~S x 1.15S): ears / horns / sprout / wings by ``kind``,
    big glossy eyes, blush, tiny feet. blink=True draws happy closed eyes."""
    ss = 2
    W = max(8, int(S)) * ss
    H = int(W * 1.15)
    body = np.asarray(body, np.float32)
    dark = tint(body * 0.72)
    inner = (255, 168, 190)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx, cy = W / 2, H * 0.60
    rx, ry = 0.40 * W, 0.355 * W
    k = kind % 7
    # ---- behind-body features
    if k == 0:  # round ears
        for s in (-1, 1):
            ex, ey, r = cx + s * 0.25 * W, cy - 0.27 * W, 0.115 * W
            d.ellipse([ex - r, ey - r, ex + r, ey + r], fill=dark)
            r2 = r * 0.55
            d.ellipse([ex - r2, ey - r2, ex + r2, ey + r2], fill=inner)
    elif k == 1:  # pointy ears
        for s in (-1, 1):
            p = [(cx + s * 0.36 * W, cy - 0.08 * W), (cx + s * 0.10 * W, cy - 0.28 * W), (cx + s * 0.35 * W, cy - 0.47 * W)]
            d.polygon(p, fill=dark)
            q = [(cx + s * 0.31 * W, cy - 0.15 * W), (cx + s * 0.18 * W, cy - 0.27 * W), (cx + s * 0.32 * W, cy - 0.39 * W)]
            d.polygon(q, fill=inner)
    elif k == 2:  # long floppy ears
        for s in (-1, 1):
            ear = Image.new("RGBA", (int(0.18 * W), int(0.5 * W)), (0, 0, 0, 0))
            de = ImageDraw.Draw(ear)
            de.ellipse([0, 0, ear.width - 1, ear.height - 1], fill=dark)
            de.ellipse([ear.width * 0.28, ear.height * 0.12, ear.width * 0.72, ear.height * 0.85], fill=inner)
            ear = ear.rotate(-s * 14, expand=True, resample=Image.BICUBIC)
            img.alpha_composite(ear, (int(cx + s * 0.15 * W - ear.width / 2), int(cy - 0.72 * W)))
    elif k == 3:  # antennae with glowing balls
        for s in (-1, 1):
            d.line([(cx + s * 0.08 * W, cy - 0.28 * W), (cx + s * 0.2 * W, cy - 0.56 * W)], fill=dark, width=int(0.025 * W))
            r = 0.055 * W
            bx, by = cx + s * 0.2 * W, cy - 0.58 * W
            d.ellipse([bx - r, by - r, bx + r, by + r], fill=tint(np.asarray(accent, np.float32) * 0.6 + 100))
            d.ellipse([bx - r * 0.4 - r * 0.25, by - r * 0.4 - r * 0.25, bx - r * 0.4 + r * 0.25, by - r * 0.4 + r * 0.25], fill=(255, 255, 255))
    elif k == 4:  # little horns
        for s in (-1, 1):
            p = [(cx + s * 0.12 * W, cy - 0.27 * W), (cx + s * 0.27 * W, cy - 0.23 * W), (cx + s * 0.25 * W, cy - 0.50 * W)]
            d.polygon(p, fill=(255, 238, 200))
    elif k == 5:  # leaf sprout
        d.line([(cx, cy - 0.33 * W), (cx + 0.02 * W, cy - 0.5 * W)], fill=(60, 150, 60), width=int(0.025 * W))
        for s in (-1, 1):
            lf = [(cx + 0.02 * W, cy - 0.49 * W), (cx + s * 0.13 * W, cy - 0.60 * W), (cx + s * 0.22 * W, cy - 0.52 * W),
                  (cx + s * 0.12 * W, cy - 0.46 * W)]
            d.polygon(lf, fill=(110, 210, 90))
    else:  # wings + tiny round ears
        for s in (-1, 1):
            for j in range(3):
                wx, wy = cx + s * (0.40 + 0.05 * j) * W, cy - (0.10 + 0.07 * j) * W
                d.ellipse([wx - 0.13 * W, wy - 0.07 * W, wx + 0.13 * W, wy + 0.07 * W], fill=(255, 255, 255) if j % 2 == 0 else (236, 240, 255))
            ex, ey, r = cx + s * 0.22 * W, cy - 0.29 * W, 0.08 * W
            d.ellipse([ex - r, ey - r, ex + r, ey + r], fill=dark)
    # feet
    for s in (-1, 1):
        fx, fy = cx + s * 0.18 * W, cy + 0.32 * W
        d.ellipse([fx - 0.10 * W, fy - 0.06 * W, fx + 0.10 * W, fy + 0.06 * W], fill=dark)
    # ---- shaded body (numpy, inside its bounding box only)
    bx0, bx1 = int(max(0, cx - rx * 1.02)), int(min(W, cx + rx * 1.02 + 2))
    by0, by1 = int(max(0, cy - ry * 1.02)), int(min(H, cy + ry * 1.12 + 2))
    yy, xx = np.mgrid[by0:by1, bx0:bx1].astype(np.float32)
    u = (xx + 0.5 - cx) / rx
    v = -(yy + 0.5 - cy) / ry
    a = 1.0 + 0.06 * (-v)  # slightly bottom-heavy blob
    f, n = _sphere_normals(u, v, a)
    alpha = np.clip((1 - f) * rx * 0.25, 0, 1)
    alb = np.broadcast_to(body, u.shape + (3,)).copy()
    bel = np.clip((1 - ((u / 0.62) ** 2 + ((v + 0.42) / 0.45) ** 2)) * 12, 0, 1)[..., None]
    alb = alb * (1 - bel * 0.85) + np.minimum(body * 0.35 + 175, 255) * bel * 0.85
    col = _light(n, alb, np.minimum(body + 80, 255), spec=0.55)
    bl = np.dstack([np.clip(col, 0, 255), alpha * 255]).astype(np.uint8)
    img.alpha_composite(Image.fromarray(bl, "RGBA"), (bx0, by0))
    d = ImageDraw.Draw(img)
    # ---- face
    ey = cy - 0.04 * W
    for s in (-1, 1):
        ex = cx + s * 0.145 * W
        if blink:
            d.arc([ex - 0.07 * W, ey - 0.04 * W, ex + 0.07 * W, ey + 0.08 * W], 200, 340, fill=(40, 30, 50), width=int(0.024 * W))
        else:
            d.ellipse([ex - 0.078 * W, ey - 0.098 * W, ex + 0.078 * W, ey + 0.098 * W], fill=(34, 26, 46))
            d.ellipse([ex - 0.05 * W, ey + 0.01 * W, ex + 0.05 * W, ey + 0.09 * W], fill=(70, 60, 110))
            r = 0.034 * W
            hx, hy = ex + 0.022 * W, ey - 0.04 * W
            d.ellipse([hx - r, hy - r, hx + r, hy + r], fill=(255, 255, 255))
            r = 0.015 * W
            hx, hy = ex - 0.03 * W, ey + 0.04 * W
            d.ellipse([hx - r, hy - r, hx + r, hy + r], fill=(255, 255, 255))
    cheeks = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dc = ImageDraw.Draw(cheeks)
    for s in (-1, 1):
        chx, chy = cx + s * 0.26 * W, cy + 0.07 * W
        dc.ellipse([chx - 0.06 * W, chy - 0.035 * W, chx + 0.06 * W, chy + 0.035 * W], fill=(255, 110, 150, 150))
    img.alpha_composite(cheeks)
    my = cy + 0.07 * W
    d.chord([cx - 0.055 * W, my - 0.045 * W, cx + 0.055 * W, my + 0.06 * W], 0, 180, fill=(120, 36, 60))
    d.chord([cx - 0.03 * W, my + 0.01 * W, cx + 0.03 * W, my + 0.055 * W], 0, 180, fill=(255, 120, 140))
    return img.resize((W // ss, H // ss), Image.LANCZOS)


def coin_sprite(S: int) -> Image.Image:
    S = max(6, int(S))
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) / S - 0.5
    r = np.hypot(xx, yy)
    gold = np.array([255, 200, 60], np.float32)
    g = np.clip(1.15 - (xx + yy) * 1.1, 0.5, 1.6)[..., None]
    col = gold * (0.62 + 0.38 * g)
    col = np.where(((r > 0.40) & (r < 0.5))[..., None], gold * (1.05 - (xx - yy)[..., None] * 0.8), col)
    col = np.where((np.abs(r - 0.37) < 0.018)[..., None], gold * 0.62, col)
    alpha = np.clip((0.5 - r) * S * 1.5, 0, 1) * 255
    im = Image.fromarray(np.dstack([np.clip(col, 0, 255), alpha]).astype(np.uint8), "RGBA")
    d = ImageDraw.Draw(im)
    f = font(max(6, int(S * 0.52)))
    l, t_, rr, b = f.getbbox("$")
    x0, y0 = (S - (rr - l)) / 2 - l, (S - (b - t_)) / 2 - t_
    d.text((x0 + S * 0.02, y0 + S * 0.02), "$", fill=(170, 110, 20, 255), font=f)
    d.text((x0, y0), "$", fill=(255, 240, 170, 255), font=f)
    return im


def text_sprite(s: str, size: int, fill, stroke=(20, 60, 20), sw=None) -> Image.Image:
    size = max(6, int(size))
    f = font(size)
    sw = max(1, int(size * 0.12)) if sw is None else sw
    l, t_, r, b = f.getbbox(s, stroke_width=sw)
    im = Image.new("RGBA", (r - l + 4, b - t_ + 4), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((2 - l, 2 - t_), s, font=f, fill=tuple(fill), stroke_width=sw, stroke_fill=tuple(stroke))
    return im


def cloud_puffs(rng, n=7):
    """Cartoon cloud as puffs (dx, dy, r) in cloud-width units (width ~1, flat bottom at dy=0)."""
    out = []
    for k in range(n):
        x = -0.36 + 0.72 * k / (n - 1) + rng.uniform(-0.05, 0.05)
        r = (0.13 + 0.13 * (1 - abs(x) / 0.45)) * rng.uniform(0.8, 1.2)
        out.append((x, -r * rng.uniform(0.35, 0.8), r))
    return out


def star_xy(n=5, inner=0.45) -> np.ndarray:
    a = np.arange(2 * n) * math.pi / n - math.pi / 2
    r = np.where(np.arange(2 * n) % 2 == 0, 1.0, inner)
    return np.stack([np.cos(a) * r, np.sin(a) * r], 1)


# ======================================================================== mesh builders (local coords)
def box_corners(c, h, M=None) -> np.ndarray:
    """(8,3) corners of a box centred at c with half sizes h, transformed by 3x3 M."""
    s = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float64)
    P = s * np.asarray(h, np.float64)
    if M is not None:
        P = P @ np.asarray(M).T
    return P + np.asarray(c, np.float64)


BOX_FACES = np.array([[4, 6, 7, 5], [0, 1, 3, 2], [2, 3, 7, 6], [0, 4, 5, 1], [1, 5, 7, 3], [0, 2, 6, 4]])  # +x -x +y -y +z -z


def lathe(profile, n=14):
    """Surface of revolution around +y: profile [(r, y), ...] bottom->top. Returns quads (m,4,3), axis refs (m,3)."""
    th = np.linspace(0, TAU, n + 1)
    Q, ref = [], []
    for j in range(len(profile) - 1):
        (r0, y0), (r1, y1) = profile[j], profile[j + 1]
        for i in range(n):
            a0, a1 = th[i], th[i + 1]
            Q.append([(r0 * math.cos(a0), y0, r0 * math.sin(a0)), (r0 * math.cos(a1), y0, r0 * math.sin(a1)),
                      (r1 * math.cos(a1), y1, r1 * math.sin(a1)), (r1 * math.cos(a0), y1, r1 * math.sin(a0))])
            ref.append((0, (y0 + y1) / 2, 0))
    return np.array(Q, np.float64), np.array(ref, np.float64)


def ico_blob(rng, n_lat=5, n_lon=8, jitter=0.12):
    """Low-poly lumpy sphere: quads (m,4,3) around the origin (unit radius)."""
    lat = np.linspace(-math.pi / 2, math.pi / 2, n_lat + 1)
    lon = np.linspace(0, TAU, n_lon + 1)
    R_ = 1 + rng.uniform(-jitter, jitter, (n_lat + 1, n_lon + 1))
    R_[:, -1] = R_[:, 0]
    R_[0, :] = R_[0, 0]
    R_[-1, :] = R_[-1, 0]

    def P(i, j):
        return (R_[i, j] * math.cos(lat[i]) * math.cos(lon[j]), R_[i, j] * math.sin(lat[i]), R_[i, j] * math.cos(lat[i]) * math.sin(lon[j]))

    Q = [[P(i, j), P(i, j + 1), P(i + 1, j + 1), P(i + 1, j)] for i in range(n_lat) for j in range(n_lon)]
    return np.array(Q, np.float64)


def torus(R0, r, n=16, m=6):
    """Torus around +y. Returns quads (n*m,4,3) and per-face tube-centre refs."""
    th = np.linspace(0, TAU, n + 1)
    ph = np.linspace(0, TAU, m + 1)
    Q, ref = [], []
    for i in range(n):
        for j in range(m):
            pts = []
            for a, b in ((th[i], ph[j]), (th[i + 1], ph[j]), (th[i + 1], ph[j + 1]), (th[i], ph[j + 1])):
                rr = R0 + r * math.cos(b)
                pts.append((rr * math.cos(a), r * math.sin(b), rr * math.sin(a)))
            am = (th[i] + th[i + 1]) / 2
            Q.append(pts)
            ref.append((R0 * math.cos(am), 0, R0 * math.sin(am)))
    return np.array(Q, np.float64), np.array(ref, np.float64)


# ======================================================================== blocky character
def char_parts(pos, yaw, t_run, freq, scale=1.0, mode="carry", lean=0.18, bounce=0.13, squash=0.06, cols=None):
    """Generic blocky toy character (cube head, box body). Returns list of (corners (8,3), colour (3,) or (6,3)).
    mode: 'carry' (arms raised holding something overhead), 'run' (arms swinging), 'idle'."""
    cols = cols or {}
    skin = np.asarray(cols.get("skin", (246, 206, 168)), np.float32)
    shirt = np.asarray(cols.get("shirt", (255, 90, 80)), np.float32)
    pants = np.asarray(cols.get("pants", (60, 70, 120)), np.float32)
    shoe = np.asarray(cols.get("shoe", (40, 40, 50)), np.float32)
    hat = cols.get("hat", (40, 180, 255))
    eye = np.asarray(cols.get("eye", (30, 30, 40)), np.float32)
    ph = t_run * freq * TAU
    sw = math.sin(ph)
    if mode == "idle":
        sw *= 0.15
    by = bounce * abs(math.cos(ph)) if mode != "idle" else 0.03 * math.sin(ph)
    sq = 1 + squash * math.cos(2 * ph)
    S = np.diag([scale / math.sqrt(sq), scale * sq, scale / math.sqrt(sq)])
    M = rot_y(yaw) @ rot_x(lean) @ S
    base = np.asarray(pos, np.float64) + np.array([0, by * scale, 0])
    parts = []

    def add(c_local, h, Rl, col):
        parts.append((box_corners(base + M @ np.asarray(c_local, np.float64), h, M @ Rl), col))

    I = np.eye(3)
    # legs (pivot at hips)
    for s, a in ((-1, sw * 0.95), (1, -sw * 0.95)):
        Rl = rot_x(a)
        hip = np.array([s * 0.15, 0.74, 0])
        add(hip + Rl @ np.array([0, -0.30, 0]), (0.13, 0.30, 0.14), Rl, pants)
        add(hip + Rl @ np.array([0, -0.66, 0.03]), (0.135, 0.07, 0.17), Rl, shoe)
    add((0, 1.08, 0), (0.33, 0.36, 0.18), I, shirt)
    add((0, 0.77, 0), (0.335, 0.06, 0.185), I, pants * 0.8)  # belt
    # arms
    for s in (-1, 1):
        if mode == "carry":
            a = -2.75 + 0.12 * math.sin(ph + (0 if s < 0 else 1.2))
            Rl = rot_z(s * 0.18) @ rot_x(a)
        else:
            a = (-sw if s < 0 else sw) * 1.0
            Rl = rot_z(-s * 0.08) @ rot_x(a)
        sh = np.array([s * 0.45, 1.36, 0])
        add(sh + Rl @ np.array([0, -0.28, 0]), (0.11, 0.30, 0.11), Rl, shirt * 0.92)
        add(sh + Rl @ np.array([0, -0.63, 0]), (0.10, 0.07, 0.10), Rl, skin)
    # head + face + cap
    add((0, 1.74, 0), (0.26, 0.26, 0.26), I, skin)
    for s in (-1, 1):
        add((s * 0.1, 1.77, 0.262), (0.04, 0.065, 0.006), I, eye)
    if hat is not None:
        hat = np.asarray(hat, np.float32)
        add((0, 2.04, 0), (0.275, 0.07, 0.275), I, hat)
        add((0, 1.985, 0.33), (0.26, 0.02, 0.12), I, hat * 0.85)
    return parts


# ======================================================================== base class
class Eggs(Creator):
    """Shared machinery: global painter's draw list (faces + sprites + callables), vectorised face
    shading with sky fill + distance haze, sprite placement with squash / stretch / rotation."""
    default_accent = "#FF5FA2"
    SKY_FILL = np.array([150, 190, 255], np.float32)

    def build(self):
        # 'speed' is a scene parameter here (chase intensity), not the Scene3D time multiplier
        self.speed_param = self.p.get("speed")
        self.p = {k: v for k, v in self.p.items() if k != "speed"}
        super().build()

    # ---------------------------------------------------------------- frame begin / flush
    def begin(self, cam, bg):
        self.R.begin(cam)
        self.cpos = np.asarray(cam.pos, np.float64)
        self.items = []
        self._soft = np.zeros((self.R.Hs, self.R.Ws, 3), np.float32)
        canvas = bg.copy()
        return canvas, ImageDraw.Draw(canvas)

    def flush(self, canvas, d):
        self.items.sort(key=lambda it: (it[0], it[1]))
        for it in self.items:
            kind = it[2]
            if kind == 0:
                d.polygon(it[3], fill=it[4])
            elif kind == 1:
                img, x0, y0 = it[3]
                canvas.paste(img, (x0, y0), img)
            else:
                it[3](canvas, d)
        self.items = []

    def push_fn(self, P, fn, layer=1, bias=0.0):
        z = float((np.asarray(P, np.float64) - self.cpos) @ self.R.cam.R[2])
        self.items.append((layer, -(z + bias), 2, fn))

    # ---------------------------------------------------------------- faces
    def push_faces(self, Q, cols, layer=1, ref=None, cull=True, amb=0.60, diff=0.52, spec=0.0, haze=None,
                   near=0.08, bias=0.0, emit=None):
        """World polygons Q (m,k,3) with base colours (m,3) -> shaded items in the draw list.
        ref: point(s) the normal must face away from (orientation); haze=(z0, z1, max, col)."""
        Q = np.asarray(Q, np.float64)
        if Q.ndim != 3 or len(Q) == 0:
            return
        m, k, _ = Q.shape
        n = np.cross(Q, np.roll(Q, -1, axis=1)).sum(1)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        cen = Q.mean(1)
        if ref is not None:
            flip = ((cen - np.asarray(ref, np.float64)) * n).sum(1) < 0
            n[flip] *= -1
        view = self.cpos - cen
        keep = ((n * view).sum(1) > 0) if cull else np.ones(m, bool)
        cam = self.R.cam
        C = ((Q.reshape(-1, 3) - cam.pos) @ cam.R.T).reshape(m, k, 3)
        z = C[..., 2]
        okall = (z >= near).all(1)
        keep &= (z >= near).any(1)
        if not keep.any():
            return
        cols = np.broadcast_to(np.asarray(cols, np.float32), (m, 3))
        lam = np.clip(n @ self.light, 0, 1)
        hemi = 0.5 + 0.5 * n[:, 1]
        col = cols * (amb + diff * lam)[:, None] + self.SKY_FILL[None, :] * (0.10 * hemi)[:, None]
        if spec:
            hv = view / np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-9) + self.light[None, :]
            hv /= np.linalg.norm(hv, axis=1, keepdims=True)
            col += 255 * spec * np.clip((n * hv).sum(1), 0, 1)[:, None] ** 24
        if emit is not None:
            col = col + np.asarray(emit, np.float32)
        zm = z.mean(1)
        if haze is not None:
            z0, z1, hm, hc = haze
            fz = (np.clip((zm - z0) / (z1 - z0), 0, 1) * hm)[:, None]
            col = col * (1 - fz) + np.asarray(hc, np.float32) * fz
        ci = np.clip(col, 0, 255).astype(np.int32).tolist()
        W2, H2, f = self.R.W * 0.5, self.R.H * 0.5, self.R.f
        zz = np.maximum(z, 1e-6)
        SX = (W2 + C[..., 0] * f / zz).tolist()
        SY = (H2 - C[..., 1] * f / zz).tolist()
        items = self.items
        for i in np.nonzero(keep)[0].tolist():
            pts = list(zip(SX[i], SY[i])) if okall[i] else self.screen_poly(C[i], near)
            if pts:
                items.append((layer, -(zm[i] + bias), 0, pts, tuple(ci[i])))

    def push_boxes(self, parts, layer=1, **kw):
        """parts: list of (corners (8,3), colour (3,) or per-face (6,3))."""
        if not parts:
            return
        Cs = np.stack([p[0] for p in parts])
        Q = Cs[:, BOX_FACES].reshape(-1, 4, 3)
        cols = np.concatenate([np.broadcast_to(np.asarray(p[1], np.float32), (6, 3)) for p in parts])
        ref = np.repeat(Cs.mean(1), 6, axis=0)
        self.push_faces(Q, cols, layer, ref=ref, **kw)

    # ---------------------------------------------------------------- sprites
    def place(self, img, x, y, h_px, sx=1.0, sy=1.0, ang=0.0, anchor=0.5, cache=False):
        """Resize/rotate sprite; (x, y) is the anchor point (anchor 0.5 = centre, 1.0 = bottom centre).
        cache=True quantises the size (2 % steps) and memoises the resize (static sprites like clouds).
        Returns (img, x0, y0) or None."""
        h = h_px * sy
        if cache and h >= 2:
            q = round(math.log(h) / 0.02)
            h = math.exp(q * 0.02)
        w = h / sy * img.width / img.height * sx
        if h < 2 or w < 2 or h > 6 * self.H:
            return None
        size = (max(1, int(round(w))), max(1, int(round(h))))
        if cache:
            rc = self.__dict__.setdefault("_rcache", {})
            key = (id(img), size)
            im = rc.get(key)
            if im is None:
                if len(rc) > 400:
                    rc.clear()
                im = rc[key] = img.resize(size, Image.BILINEAR, reducing_gap=2.0)
        else:
            im = img.resize(size, Image.BILINEAR, reducing_gap=2.0)
        off = (anchor - 0.5) * h
        if abs(ang) > 0.2:
            im = im.rotate(ang, resample=Image.BILINEAR, expand=True)
            a = math.radians(ang)
            cx, cy = x - off * math.sin(a), y - off * math.cos(a)
        else:
            cx, cy = x, y - off
        x0, y0 = int(round(cx - im.width / 2)), int(round(cy - im.height / 2))
        if x0 > self.W or y0 > self.H or x0 + im.width < 0 or y0 + im.height < 0:
            return None
        return im, x0, y0

    def push_sprite(self, img, P, h_world, sx=1.0, sy=1.0, ang=0.0, anchor=1.0, layer=1, bias=0.0, cache=False):
        """World anchored sprite (anchor 1.0 = bottom centre at P). Returns (x, y, px-per-unit) or None."""
        x, y, z, inv = self.R.project(np.asarray(P, np.float32)[None, :])
        if z[0] < 0.2:
            return None
        pl = self.place(img, float(x[0]), float(y[0]), h_world * float(inv[0]), sx, sy, ang, anchor, cache)
        if pl is not None:
            self.items.append((layer, -(float(z[0]) + bias), 1, pl))
        return float(x[0]), float(y[0]), float(inv[0])

    def rays_soft(self, cx, cy, angs, L, half, col, gain=1.0):
        """Soft god-ray wedges from screen point (cx, cy) into the quarter-res soft buffer."""
        ds = self.R.ds
        m = Image.new("L", (self.R.Ws, self.R.Hs), 0)
        d = ImageDraw.Draw(m)
        L = np.broadcast_to(np.asarray(L, np.float64), np.shape(angs))
        for a_, l_ in zip(np.asarray(angs).tolist(), L.tolist()):
            p0 = (cx / ds, cy / ds)
            p1 = ((cx + math.cos(a_ - half) * l_) / ds, (cy + math.sin(a_ - half) * l_) / ds)
            p2 = ((cx + math.cos(a_ + half) * l_) / ds, (cy + math.sin(a_ + half) * l_) / ds)
            d.polygon([p0, p1, p2], fill=255)
        m = m.filter(ImageFilter.GaussianBlur(2.5 * self.k + 0.5))
        arr = np.asarray(m, np.float32) / 255
        # fade with distance from the centre
        yy, xx = self._rgrid()
        fall = np.exp(-(((xx - cx / ds) ** 2 + (yy - cy / ds) ** 2) / (float(np.max(L)) / ds * 0.6) ** 2))
        self._soft += (arr * fall)[..., None] * (np.asarray(col, np.float32) * gain)

    def _rgrid(self):
        g = getattr(self, "_rg", None)
        if g is None:
            g = self._rg = np.mgrid[0:self.R.Hs, 0:self.R.Ws].astype(np.float32)
        return g

    def push_cloud(self, puffs, P, width, layer=-1, shade=(214, 226, 250), col=(255, 255, 255), bias=0.0):
        """Flat cartoon cloud (puff circles + soft underside tone) as one draw-list item."""
        x, y, z, inv = self.R.project(np.asarray(P, np.float32)[None, :])
        if z[0] < 0.3:
            return
        s = width * float(inv[0])
        X, Y = float(x[0]), float(y[0])
        if X + s < 0 or X - s > self.W or Y - s > self.H or Y + s < 0:
            return
        es = [[X + (dx - r) * s, Y + (dy - r) * s, X + (dx + r) * s, Y + (dy + r) * s] for dx, dy, r in puffs]
        sh = [[e[0], e[1] + 0.06 * s, e[2], e[3] + 0.06 * s] for e in es]
        base = [X - 0.38 * s, Y - 0.08 * s, X + 0.38 * s, Y + 0.06 * s]

        def fn(cv, d):
            for e in sh:
                d.ellipse(e, fill=shade)
            d.rounded_rectangle(base, radius=0.07 * s, fill=shade)
            for e in es:
                d.ellipse(e, fill=col)

        self.items.append((layer, -(float(z[0]) + bias), 2, fn))

    def shadow(self, P, rx, col, layer=1, bias=0.05, squash=0.32):
        x, y, z, inv = self.R.project(np.asarray(P, np.float32)[None, :])
        if z[0] < 0.2:
            return
        r = rx * float(inv[0])
        X, Y = float(x[0]), float(y[0])
        c = tint(col)
        e = [X - r, Y - r * squash, X + r, Y + r * squash]
        self.items.append((layer, -(float(z[0]) + bias), 2, lambda cv, d: d.ellipse(e, fill=c)))

    # ---------------------------------------------------------------- shared props
    def tree_items(self, x, z, ground, s, kind, sway, cols, haze=None):
        """Low-poly tree: box trunk + stacked cones (pine) or lumpy blob crown (round)."""
        trunk = np.array([150, 96, 56], np.float32)
        parts = [(box_corners((x, ground + 0.45 * s, z), (0.11 * s, 0.45 * s, 0.11 * s)), trunk)]
        self.push_boxes(parts, haze=haze)
        if kind == 0:
            for j, (r, y0, hgt) in enumerate(((0.85, 0.7, 1.0), (0.66, 1.25, 0.9), (0.45, 1.75, 0.8))):
                Q, ref = self.cone_cache[j]
                M = rot_z(sway * (0.4 + 0.3 * j))
                sc = np.array([r * s, hgt * s, r * s])
                Qw = (Q * sc) @ M.T + np.array([x, ground + y0 * s, z])
                rw = (ref * sc) @ M.T + np.array([x, ground + y0 * s, z])
                self.push_faces(Qw, np.asarray(cols[j % len(cols)], np.float32), ref=rw, haze=haze)
        else:
            M = rot_z(sway * 0.5) * 1.0
            c = np.array([x, ground + 1.45 * s, z])
            Qw = (self.blob_q * np.array([0.8, 0.72, 0.8]) * s) @ M.T + c
            self.push_faces(Qw, np.asarray(cols[0], np.float32), ref=c, haze=haze)

    def init_props(self, rng):
        self.cone_cache = []
        for _ in range(3):
            Q, ref = lathe([(1.0, 0.0), (0.001, 1.0)], 9)
            self.cone_cache.append((Q, ref * 0 + np.array([0, 0.3, 0])))
        self.blob_q = ico_blob(rng, 4, 8, 0.12)


# ======================================================================== 1. egg nest island
class EggNest(Eggs):
    """Floating low-poly sky island biome (grass, rocks, pine + round trees, hanging rocks) with
    straw nests holding big glossy speckled eggs of every rarity (common white, rare blue, epic
    purple, legendary gold with glow, god-rays and orbiting sparkles). Eggs wobble and hop with
    squash & stretch; a guard spotlight sweeps over them and a red laser scans the island; rarity
    sparkles float up; clouds and distant islands drift; the camera swings around in an orbit arc.
    params.rarity common|rare|epic|legendary = hero egg (closest, biggest)."""
    default_accent = "#4FC3FF"

    def init(self):
        rng = self.rng
        a = self.acc
        self.light = _norm([-0.45, 0.8, 0.55])
        self.hero = parse_rarity(self.p.get("rarity"), "legendary")
        others = [r for r in RARITIES if r != self.hero]
        # nests: (x, z, egg height, rarity)
        spots = [(0.0, 1.75, 1.45, self.hero), (-2.25, 0.25, 1.0, others[0]), (2.3, 0.35, 1.0, others[1]),
                 (-0.95, -1.75, 0.95, others[2]), (1.35, -1.95, 0.9, "common" if self.hero != "common" else "rare")]
        self.nests = spots
        k = self.k
        self.egg_spr = {}
        for r in RARITIES:
            self.egg_spr[r] = egg_sprite(330 * k, r, rng)
        self.egg_hero = egg_sprite(480 * k, self.hero, rng)
        self.nest_q, self.nest_ref = torus(1.0, 0.33, 16, 6)
        straw = np.array([[214, 164, 82], [188, 136, 62], [234, 190, 110], [170, 118, 52]], np.float32)
        self.nest_c = straw[rng.integers(0, 4, len(self.nest_q))]
        self.straws = rng.uniform(0, 1, (len(spots), 14, 3))
        self.ph = rng.uniform(0, TAU, len(spots))
        self.hop_p = rng.uniform(1.7, 2.6, len(spots))
        # island top: rings of jittered vertices
        ns = 22
        radii = [0.0, 1.5, 2.9, 3.9, 4.5]
        th = np.linspace(0, TAU, ns + 1)[:-1]
        jit = 1 + rng.uniform(-0.07, 0.07, (len(radii), ns))
        hy = rng.uniform(-0.05, 0.12, (len(radii), ns))
        hy[0] = 0.05
        ring = [[(radii[j] * jit[j, i] * math.cos(th[i]), hy[j, i] * (1 if j < 4 else 0.3), radii[j] * jit[j, i] * math.sin(th[i]))
                 for i in range(ns)] for j in range(len(radii))]
        Q, C = [], []
        greens = np.array([[120, 214, 92], [104, 198, 80], [138, 224, 100], [96, 188, 78]], np.float32)
        for j in range(len(radii) - 1):
            for i in range(ns):
                i1 = (i + 1) % ns
                Q.append([ring[j][i], ring[j + 1][i], ring[j + 1][i1], ring[j][i1]])
                C.append(greens[rng.integers(0, 4)] * (1.04 - 0.05 * j))
        # sandy path from the hero nest towards the edge
        self.top_q = np.array(Q, np.float64)
        self.top_c = np.array(C, np.float32)
        # skirt (grass lip + dirt / stone strata tapering to a point)
        prof = [(4.5, 0.0), (4.45, -0.32), (4.0, -1.1), (3.2, -2.1), (2.1, -3.1), (1.0, -4.0), (0.05, -4.9)]
        strata = [(92, 176, 70), (176, 120, 74), (150, 98, 64), (162, 150, 140), (130, 88, 60), (120, 112, 110)]
        Q, C = [], []
        jr = 1 + rng.uniform(-0.08, 0.08, (len(prof), ns))
        jr[0] = jit[-1]
        for j in range(len(prof) - 1):
            for i in range(ns):
                i1 = (i + 1) % ns
                (r0, y0), (r1, y1) = prof[j], prof[j + 1]
                P = [(r0 * jr[j, i] * math.cos(th[i]), y0 + (hy[-1, i] * 0.3 if j == 0 else 0), r0 * jr[j, i] * math.sin(th[i])),
                     (r1 * jr[j + 1, i] * math.cos(th[i]), y1, r1 * jr[j + 1, i] * math.sin(th[i])),
                     (r1 * jr[j + 1, i1] * math.cos(th[i1]), y1, r1 * jr[j + 1, i1] * math.sin(th[i1])),
                     (r0 * jr[j, i1] * math.cos(th[i1]), y0 + (hy[-1, i1] * 0.3 if j == 0 else 0), r0 * jr[j, i1] * math.sin(th[i1]))]
                Q.append(P)
                C.append(np.asarray(strata[j], np.float32) * rng.uniform(0.9, 1.08))
        self.skirt_q = np.array(Q, np.float64)
        self.skirt_c = np.array(C, np.float32)
        self.init_props(rng)
        self.trees = [(-3.2, -1.2, 1.05, 0), (3.35, -1.0, 0.95, 1), (-0.1, -3.55, 1.2, 0), (2.7, -2.85, 0.9, 0),
                      (-2.7, -2.8, 0.85, 1), (-3.7, 1.0, 0.7, 0), (3.6, 1.5, 0.65, 1)]
        self.tree_ph = rng.uniform(0, TAU, len(self.trees))
        # rocks
        self.rocks = []
        for (x, z, s) in ((-1.4, 2.6, 0.32), (1.7, 2.3, 0.26), (3.0, -0.2, 0.36), (-3.2, 2.2, 0.3), (0.9, -3.4, 0.4), (-1.9, -3.4, 0.28)):
            self.rocks.append((x, z, s, ico_blob(rng, 3, 6, 0.25), np.array([168, 164, 176], np.float32) * rng.uniform(0.9, 1.1)))
        # flowers
        nf = 36
        ang = rng.uniform(0, TAU, nf)
        rr = np.sqrt(rng.uniform(0.05, 1, nf)) * 4.0
        self.flowers = np.stack([np.cos(ang) * rr, np.full(nf, 0.06), np.sin(ang) * rr], 1)
        fc = np.array([[255, 110, 150], [255, 230, 90], [255, 255, 255], [190, 130, 255], [255, 150, 70]], np.float32)
        self.flower_c = fc[rng.integers(0, len(fc), nf)]
        # hanging rocks under the island
        self.hang = [(rng.uniform(-2.5, 2.5), rng.uniform(-5.6, -4.6), rng.uniform(-2, 2), rng.uniform(0.25, 0.5), ico_blob(rng, 3, 6, 0.3))
                     for _ in range(4)]
        # distant islands + clouds
        self.far = [(rng.uniform(-1, 1) * 22, rng.uniform(-6, 3), -rng.uniform(22, 34), rng.uniform(1.4, 2.6)) for _ in range(5)]
        self.far_q = ico_blob(rng, 3, 7, 0.2)
        self.cl_puffs = [cloud_puffs(rng, int(rng.integers(5, 9))) for _ in range(4)]
        ncl = 10
        self.clouds = np.stack([rng.uniform(-30, 30, ncl), rng.uniform(-2, 9, ncl), rng.uniform(-38, -12, ncl)], 1)
        nsea = 9
        self.sea = np.stack([np.linspace(-15, 15, nsea) + rng.uniform(-1.5, 1.5, nsea), rng.uniform(-10.5, -8.5, nsea),
                             rng.uniform(-10, 0.5, nsea), rng.uniform(3.0, 4.6, nsea)], 1)
        self.wf_ph = rng.random(8)
        self.cl_s = rng.uniform(2.5, 5.0, ncl)
        self.cl_v = rng.uniform(0.25, 0.6, ncl)
        # floating rarity sparkles
        npart = 70
        self.sp_pos = np.stack([rng.uniform(-4, 4, npart), rng.uniform(0, 1, npart), rng.uniform(-3.5, 3.5, npart)], 1)
        self.sp_c = np.array([EGG_PAL[r][3] for r in RARITIES], np.float32)[rng.integers(0, 4, npart)]
        self.sp_ph = rng.random(npart)
        self.orb = rng.uniform(0, TAU, 9)
        top = lerp_col((70, 160, 255), a, 0.25)
        self.bg = Image.fromarray(vgrad(self.W, self.H, [(0, tint(top)), (0.42, (130, 204, 255)), (0.72, (255, 196, 230)), (1, (255, 178, 206))]))

    def frame(self, t):
        R, a = self.R, self.acc
        az = 0.42 * math.sin(t * 0.21) + 0.05 * math.sin(t * 0.53)
        dist = 10.6 - 0.6 * math.sin(t * 0.17)
        el = 0.36 + 0.05 * math.sin(t * 0.29)
        tgt = np.array([0.0, 0.35 + 0.12 * math.sin(t * 0.4), 0.2])
        cpos = tgt + dist * np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)])
        cam = Camera(cpos, tgt + np.array([0, -0.55, 0]), fov=58, roll=0.035 * math.sin(t * 0.3))
        canvas, d = self.begin(cam, self.bg)
        bob = 0.18 * math.sin(t * 0.8)  # the whole island floats
        haze = (12, 40, 0.55, (190, 225, 255))
        # ---- background: clouds + distant islands (layer -1)
        for i in range(len(self.clouds)):
            x0, y0, z0 = self.clouds[i]
            x = ((x0 + self.cl_v[i] * t + 30) % 60) - 30
            self.push_cloud(self.cl_puffs[i % 4], (x, y0, z0), self.cl_s[i], layer=-1)
        for i in range(len(self.sea)):  # cloud sea under the island
            x0, y0, z0, s_ = self.sea[i]
            x = ((x0 + 0.35 * t + 16) % 32) - 16
            self.push_cloud(self.cl_puffs[(i + 1) % 4], (x, y0 + 0.2 * math.sin(t * 0.4 + i), z0), s_,
                            layer=-1 if z0 < -1.5 else 2, shade=(250, 214, 236))
        for j, (x, y, z, s) in enumerate(self.far):
            c = np.array([x + 0.3 * math.sin(t * 0.1 + j), y + 0.3 * math.sin(t * 0.5 + j * 2), z])
            Qg = self.far_q * np.array([s, 0.25 * s, s]) + c
            self.push_faces(Qg, (110, 214, 100), layer=-1, ref=c - [0, 2, 0], haze=(10, 40, 0.4, (190, 225, 255)))
            Qd = self.far_q * np.array([s * 0.85, 0.9 * s, s * 0.85]) + c - [0, 0.7 * s, 0]
            self.push_faces(Qd, (190, 130, 96), layer=-1, ref=c - [0, 0.7 * s, 0], haze=(10, 40, 0.4, (190, 225, 255)))
        # ---- island
        off = np.array([0, bob, 0])
        self.push_faces(self.top_q + off, self.top_c, layer=0, ref=np.array([0, -50, 0]), haze=haze)
        self.push_faces(self.skirt_q + off, self.skirt_c, layer=0, ref=np.array([0, -1.5 + bob, 0]), haze=haze)
        for (x, y, z, s, q) in self.hang:
            c = np.array([x, y + bob + 0.15 * math.sin(t * 0.9 + x), z])
            self.push_faces(q * s + c, (150, 120, 100), layer=0, ref=c, haze=haze)
        sh_col = (70, 150, 60)
        for j, (x, z, s, kind) in enumerate(self.trees):
            sway = 0.05 * math.sin(t * 1.3 + self.tree_ph[j])
            cols = [(70, 170, 90), (86, 190, 100), (104, 206, 112)] if kind == 0 else [(110, 200, 90)]
            if kind == 1:
                cols = [tuple(lerp_col((110, 205, 90), (255, 160, 200), 0.35 * (j % 2)))]
            self.shadow((x, bob + 0.02, z), 0.7 * s, sh_col, bias=2.0)
            self.tree_items(x, z, bob, s, kind, sway, cols, haze=haze)
        for (x, z, s, q, c) in self.rocks:
            cc = np.array([x, bob + s * 0.35, z])
            self.push_faces(q * np.array([s, s * 0.7, s]) + cc, c, ref=cc, haze=haze)
        # flowers (tiny canvas dots)
        fx, fy, fz, finv = R.project((self.flowers + off).astype(np.float32))
        for i in range(len(fx)):
            if fz[i] < 0.3:
                continue
            r = 0.07 * finv[i]
            e = [fx[i] - r, fy[i] - r * 0.6, fx[i] + r, fy[i] + r * 0.6]
            c = tint(self.flower_c[i])
            self.items.append((0, -(float(fz[i]) - 0.3), 2, (lambda e_, c_: (lambda cv, dd: dd.ellipse(e_, fill=c_)))(e, c)))
        # ---- nests + eggs
        glints = []
        for j, (x, z, eh, rar) in enumerate(self.nests):
            gpos = np.array([x, bob, z])
            nr = 0.40 * eh
            Qn = self.nest_q * np.array([nr, nr * 0.9, nr]) + gpos + [0, 0.16 * eh, 0]
            refn = self.nest_ref * nr + gpos + [0, 0.16 * eh, 0]
            self.shadow(gpos + [0, 0.02, 0], nr * 1.5, sh_col, bias=1.0)
            self.push_faces(Qn, self.nest_c, ref=refn, haze=haze)
            th_ = np.linspace(0, TAU, 13)[:-1]
            disc = np.stack([np.cos(th_) * nr * 1.05, np.full(12, 0.13 * eh), np.sin(th_) * nr * 1.05], 1) + gpos
            self.push_faces(disc[None], (150, 104, 46), ref=gpos - [0, 5, 0], bias=0.6 * eh, amb=0.8, diff=0.2)
            # straw strands
            for s_ in self.straws[j][:10]:
                ang = s_[0] * TAU
                p0 = gpos + np.array([math.cos(ang) * nr * 1.25, 0.16 * eh + 0.2 * nr * (s_[1] - 0.3), math.sin(ang) * nr * 1.25])
                p1 = p0 + np.array([math.cos(ang + 1.2) * 0.35 * eh, 0.12 * eh * s_[2], math.sin(ang + 1.2) * 0.35 * eh])
                xs, ys, zs, inv_ = R.project(np.array([p0, p1], np.float32))
                if (zs > 0.3).all():
                    L = [(float(xs[0]), float(ys[0])), (float(xs[1]), float(ys[1]))]
                    w_ = max(1, int(0.03 * inv_[0]))
                    self.items.append((1, -float(zs.mean()) + 0.2, 2, (lambda L_, w__: (lambda cv, dd: dd.line(L_, fill=(236, 196, 120), width=w__)))(L, w_)))
            # egg motion: wobble bursts + hops with squash & stretch
            ph = self.ph[j]
            burst = max(0.0, math.sin(t * 0.9 + ph)) ** 2
            wob = (7 + 9 * burst) * math.sin(t * (7.5 + j) + ph)
            hy_, sx, sy = hop(t, self.hop_p[j], 0.22 * eh * (1.4 if j == 0 else 1.0), phase=ph / TAU)
            spr = self.egg_hero if j == 0 else self.egg_spr[rar]
            P = gpos + np.array([0, 0.02 * eh + hy_, 0])
            res = self.push_sprite(spr, P, eh, sx, sy, wob, anchor=1.0, bias=-0.05)
            if res is None:
                continue
            ex, ey, inv = res
            ecx, ecy = ex, ey - eh * 0.55 * inv * sy
            g = np.asarray(EGG_PAL[rar][3], np.float32)
            power = {"common": 0.25, "rare": 0.55, "epic": 0.8, "legendary": 1.3}[rar] * (1.25 if j == 0 else 0.8)
            pulse = 0.75 + 0.25 * math.sin(t * 3 + ph)
            R.soft(np.array([ecx]), np.array([ecy]), (g * 2600 * power * pulse * (eh ** 2))[None, :])
            glints.append((ecx - eh * 0.17 * inv, ecy - eh * 0.2 * inv, inv * eh))
            if rar == "legendary":
                nr_ = 16
                ang0 = t * 0.35
                angs = ang0 + np.arange(nr_) * TAU / nr_
                L = eh * inv * (1.7 + 0.4 * np.sin(np.arange(nr_) * 2.3 + t * 2.0))
                self.rays_soft(ecx, ecy, angs, L, 0.07, g, 0.9 * (0.8 + 0.2 * math.sin(t * 3)))
            if rar in ("legendary", "epic"):  # orbiting sparkles
                no = 6 if rar == "legendary" else 4
                ang = t * (1.6 if rar == "legendary" else 1.1) + np.arange(no) * TAU / no + self.orb[j]
                O = np.stack([np.cos(ang) * 0.75 * eh, 0.55 * eh + 0.25 * eh * np.sin(ang * 2 + t), np.sin(ang) * 0.75 * eh], 1) + P
                ox, oy, oz, oinv = R.project(O.astype(np.float32))
                tw = 0.6 + 0.4 * np.sin(t * 6 + np.arange(no) * 1.7)
                sparkle(R, ox, oy, 0.09 * oinv * tw, (g * 0.5 + 128) * 1.6 * tw[:, None])
        # ---- guard spotlight (lamp post) + red scanning laser
        post = np.array([-1.7, bob, -3.05])
        self.push_boxes([(box_corners(post + [0, 1.5, 0], (0.1, 1.5, 0.1)), (110, 110, 135)),
                         (box_corners(post + [0, 0.12, 0], (0.3, 0.12, 0.3)), (90, 90, 115)),
                         (box_corners(post + [0, 3.05, 0], (0.3, 0.2, 0.3), rot_y(t * 0.9)), (70, 72, 96)),
                         (box_corners(post + [0, 3.36, 0], (0.12, 0.11, 0.12)), (255, 60, 70))])
        src = post + np.array([0.0, 3.0, 0.0])
        sir = post + np.array([0, 3.36, 0])
        sp_ = 0.5 + 0.5 * math.sin(t * 9)
        R.points3d(sir[None, :].astype(np.float32), (np.array([255, 40, 50], np.float32) * 60 * sp_)[None, :], 0.2, 0.0, bloom=2.0)
        tx = 1.6 * math.sin(t * 0.7) + 0.6 * math.sin(t * 1.9)
        tz = 0.6 + 1.5 * math.sin(t * 0.53 + 1.0)
        spot = np.array([tx, bob + 0.05, tz])
        warm = np.array([255, 240, 170], np.float32)
        ring_ = np.stack([np.cos(np.linspace(0, TAU, 18)) * 1.1, np.zeros(18), np.sin(np.linspace(0, TAU, 18)) * 1.1], 1) + spot
        R.lines3d(np.repeat(src[None, :], 18, 0).astype(np.float32), ring_.astype(np.float32), warm * 0.32, warm * 0.10, width=10, bloom=0.0)
        self.pool(self._soft, spot, 1.2, warm * 0.42, squash=0.45)
        R.points3d(src[None, :].astype(np.float32), (warm * 30)[None, :], 0.12, 0.0, bloom=1.2)
        tur = np.array([3.9, bob, -2.1])
        self.push_boxes([(box_corners(tur + [0, 0.35, 0], (0.22, 0.35, 0.22)), (80, 80, 100)),
                         (box_corners(tur + [0, 0.8, 0], (0.16, 0.12, 0.16), rot_y(-t)), (230, 60, 70))])
        lsrc = tur + np.array([0, 0.8, 0])
        sweep = math.sin(t * 1.1)
        hit = np.array([-0.5 + 3.2 * sweep, bob + 0.04, 0.4 + 1.6 * math.cos(t * 0.8)])
        red = np.array([255, 40, 60], np.float32)
        R.lines3d(lsrc[None, :].astype(np.float32), hit[None, :].astype(np.float32), red * 2.6, red * 1.8, width=4.0, bloom=1.2)
        R.points3d(hit[None, :].astype(np.float32), (red * 14)[None, :], 0.05, 0.0, bloom=1.5)
        self.pool(self._soft, hit, 0.4, red * 0.9, squash=0.5)
        lx, ly, lz, _ = R.project(np.array([lsrc, hit], np.float32))
        if (lz > 0.3).all():
            L_ = [(float(lx[0]), float(ly[0])), (float(lx[1]), float(ly[1]))]
            lw = max(1, int(3.5 * self.k))
            self.items.append((3, 0.0, 2, lambda cv, dd: dd.line(L_, fill=(255, 70, 90), width=lw)))
        # ---- waterfall off the rim (front right)
        thw = -0.24
        rim = np.array([4.5 * math.sin(thw), bob + 0.02, 4.5 * math.cos(thw)])
        outv = np.array([math.sin(thw), 0, math.cos(thw)])
        side = np.array([math.cos(thw), 0, -math.sin(thw)]) * 0.32
        bot = rim + outv * 0.5 + np.array([0, -7.5, 0])
        Qw = np.array([[rim - side, rim + side, bot + side * 1.5, bot - side * 1.5]])
        self.push_faces(Qw, (120, 200, 255), layer=1, cull=False, amb=1.0, diff=0.0, bias=-0.5)
        pts = np.array([rim + side * s_ + (bot + side * 1.5 * s_ - rim - side * s_) * u_ for s_ in np.linspace(-0.7, 0.7, 4)
                        for u_ in (0.0, 1.0)], np.float32)
        wx, wy, wz, _ = R.project(pts)
        if (wz > 0.3).all():
            segs = []
            for j_ in range(4):
                xa, ya, xb, yb = float(wx[2 * j_]), float(wy[2 * j_]), float(wx[2 * j_ + 1]), float(wy[2 * j_ + 1])
                for m_ in range(3):
                    u0 = (t * 0.9 + self.wf_ph[j_] + m_ / 3) % 1.0
                    u1 = min(1.0, u0 + 0.12)
                    segs.append([(xa + (xb - xa) * u0, ya + (yb - ya) * u0), (xa + (xb - xa) * u1, ya + (yb - ya) * u1)])
            ww = max(1, int(4 * self.k))
            zw = float(wz.mean())
            self.items.append((1, -(zw - 0.6), 2, lambda cv, dd: [dd.line(sg, fill=(235, 250, 255), width=ww) for sg in segs]))
        self.flush(canvas, d)
        # ---- egg glints (specular sparkle on each egg)
        for (gx, gy, s) in glints:
            tw = 0.6 + 0.4 * math.sin(t * 2.3 + gx * 0.01)
            sparkle(R, np.array([gx]), np.array([gy]), 0.16 * s * tw, np.array([[255, 255, 255]], np.float32) * 1.2)
        # ---- rising rarity sparkles
        u = (self.sp_pos[:, 1] + t * 0.12 + self.sp_ph) % 1.0
        Psp = np.stack([self.sp_pos[:, 0] + 0.2 * np.sin(t + self.sp_ph * 9), bob + 0.2 + u * 3.5, self.sp_pos[:, 2]], 1).astype(np.float32)
        fade = np.sin(u * math.pi) * (0.55 + 0.45 * np.sin(t * 5 + self.sp_ph * 20))
        x, y, z, inv = R.project(Psp)
        big = fade > 0.75
        sparkle(R, x[big], y[big], 0.10 * inv[big], self.sp_c[big] * 1.4)
        R.points(x, y, self.sp_c * (fade * 9)[:, None], 1.4 * self.k, bloom=0.6)
        return self.fin(canvas, self._soft, canvas_bloom=0.10, bloom=(0.55, 0.5), radii=(1.4, 8.0))

# ======================================================================== 2. egg heist chase
class EggHeist(Eggs):
    """Speed chase across a plank sky-bridge: a generic blocky runner holds a glowing stolen egg over
    its head and sprints toward a colourful base (beacon of light, bouncing arrow) while dark blocky
    guards with glowing red eyes chase behind; bouncing run cycle with squash & stretch, dust puffs at
    every footstep, speed lines, motion streaks, sparkle trail, red alarm beacons + pulsing red alarm
    vignette; the camera tracks alongside with run-shake. params.speed 0..1 (how frantic the chase is),
    params.rarity = colour of the carried egg."""
    default_accent = "#FF8A1E"

    def init(self):
        rng = self.rng
        a = self.acc
        k = self.k
        self.light = _norm([-0.35, 0.85, 0.45])
        self.sp = sp = parse_unit(self.speed_param, 0.7)
        self.v = 4.2 + 5.0 * sp
        self.freq = 1.55 + 1.25 * sp
        self.rarity = parse_rarity(self.p.get("rarity"), "legendary")
        self.egg = egg_sprite(260 * k, self.rarity, rng)
        self.egg_col = np.asarray(EGG_PAL[self.rarity][3], np.float32)
        self.runner_cols = {"shirt": tuple(a), "pants": (56, 86, 190), "skin": (250, 212, 170), "hat": (255, 255, 255),
                            "shoe": (255, 255, 255)}
        self.guard_cols = {"shirt": (58, 62, 92), "pants": (38, 40, 60), "skin": (78, 82, 110), "hat": (34, 34, 52),
                           "shoe": (26, 26, 36), "eye": (255, 70, 80)}
        self.cl_puffs = [cloud_puffs(rng, int(rng.integers(5, 9))) for _ in range(5)]
        self.isl_q = ico_blob(rng, 3, 7, 0.2)
        self.init_props(rng)
        self.alert = text_sprite("!", int(90 * k), (255, 255, 255), stroke=(230, 40, 60), sw=max(2, int(12 * k)))
        nl = int(22 + 40 * sp)
        self.sl = np.stack([rng.uniform(-0.6, 0.6, nl), rng.random(nl), rng.uniform(0.6, 1.4, nl), rng.uniform(0.12, 0.3, nl)], 1)
        self.trail_ph = rng.random(24)
        # alarm vignette (quarter res)
        yy, xx = np.mgrid[0:self.R.Hs, 0:self.R.Ws].astype(np.float32)
        r = np.sqrt(((xx / self.R.Ws - 0.5) / 0.5) ** 2 + ((yy / self.R.Hs - 0.5) / 0.5) ** 2)
        self.vig = (np.clip((r - 0.62) / 0.5, 0, 1) ** 1.4)[..., None]
        top = lerp_col((60, 150, 255), a, 0.15)
        self.bg = Image.fromarray(vgrad(self.W, self.H, [(0, tint(top)), (0.5, (140, 205, 255)), (0.8, (255, 205, 225)), (1, (255, 190, 200))]))

    def render(self, t):  # 'speed' is the chase intensity here, never a time multiplier
        return self.frame(float(t))

    def xr(self, t):
        return self.v * t

    def frame(self, t):
        R, a = self.R, self.acc
        sp = self.sp
        xr = self.xr(t)
        ph = t * self.freq * TAU
        shake = (0.035 + 0.05 * sp) * math.sin(ph * 2)
        cpos = np.array([xr - 8.4 + 0.5 * math.sin(t * 0.5), 5.4 + shake + 0.3 * math.sin(t * 0.37), 5.0 + 0.6 * math.sin(t * 0.29)])
        tgt = np.array([xr + 2.2, 0.6, -0.9])
        cam = Camera(cpos, tgt, fov=60, roll=0.05 * math.sin(t * 0.7) + 0.02 * math.sin(ph))
        canvas, d = self.begin(cam, self.bg)
        haze = (14, 45, 0.6, (200, 225, 255))
        # ---- clouds (world tiled cells) + floating islands
        c0 = int(math.floor((xr - 25) / 7.0))
        for c in range(c0, c0 + 15):
            for lay in range(2):
                h0, h1, h2 = hsh(c, 10 + lay), hsh(c, 20 + lay), hsh(c, 30 + lay)
                P = (c * 7.0 + 4 * h0, -6.5 - 3 * h1 + (lay * 11 if h2 > 0.6 else 0), -6 - 26 * h2 if lay else 3 - 9 * h2)
                self.push_cloud(self.cl_puffs[(c + lay) % 5], P, 3.0 + 3.0 * h1, layer=-1,
                                shade=(250, 214, 236) if P[1] < 0 else (214, 226, 250))
        i0 = int(math.floor((xr - 30) / 16.0))
        for c in range(i0, i0 + 6):
            h0, h1 = hsh(c, 41), hsh(c, 42)
            ctr = np.array([c * 16.0 + 6 * h0, -1.5 + 4 * h1 + 0.3 * math.sin(t * 0.6 + c), -12 - 10 * hsh(c, 43)])
            s_ = 1.6 + 1.2 * hsh(c, 44)
            self.push_faces(self.isl_q * np.array([s_, 0.22 * s_, s_]) + ctr, (110, 214, 100), layer=-1, ref=ctr - [0, 2, 0], haze=haze)
            self.push_faces(self.isl_q * np.array([s_ * 0.8, s_, s_ * 0.8]) + ctr - [0, 0.8 * s_, 0], (196, 136, 96), layer=-1,
                            ref=ctr - [0, 0.8 * s_, 0], haze=haze)
            Q, ref = self.cone_cache[0]
            for j_, (r_, y_, hh) in enumerate(((0.7, 0.3, 1.2), (0.5, 0.9, 1.0))):
                sc = np.array([r_ * s_ * 0.6, hh * s_ * 0.6, r_ * s_ * 0.6])
                o = ctr + [0.3 * s_, 0.2 * s_ + y_ * s_ * 0.6, 0]
                self.push_faces(Q * sc + o, (70 + 20 * j_, 180, 95), layer=-1, ref=ref * sc + o, haze=haze)
        # ---- the base ahead (stays on the horizon as the goal)
        self.draw_base(t, np.array([xr + 23.0, 0.0, -7.2]), haze)
        # ---- bridge: planks, beams, posts, ropes, alarm beacons
        parts = []
        p0 = int(math.floor((xr - 9) / 0.5))
        for i in range(p0, p0 + 68):
            hx = hsh(i, 1)
            col = np.array([206, 142, 84], np.float32) * (0.82 + 0.3 * hx)
            parts.append((box_corners((i * 0.5, -0.07 + 0.04 * (hsh(i, 2) - 0.5), 0), (0.215, 0.07, 1.1), rot_y((hx - 0.5) * 0.08)), col))
        b0 = int(math.floor((xr - 9) / 3.0))
        posts = []
        for i in range(b0, b0 + 13):
            for zs in (-1.15, 1.15):
                parts.append((box_corners((i * 3.0, 0.42, zs), (0.08, 0.5, 0.08)), (150, 96, 56)))
                posts.append((i, zs))
            parts.append((box_corners((i * 3.0 + 1.5, -0.24, 0), (1.52, 0.08, 0.95)), (120, 80, 50)))
        self.push_boxes(parts, haze=haze)
        # ropes (sagging lines between post tops)
        rp = []
        for i in range(b0, b0 + 12):
            for zs in (-1.15, 1.15):
                u = np.linspace(0, 1, 7)
                P = np.stack([i * 3.0 + 3.0 * u, 0.88 - 0.22 * np.sin(u * math.pi), np.full(7, zs)], 1)
                rp.append(P)
        RP = np.concatenate(rp).astype(np.float32)
        rx, ry, rz, rinv = R.project(RP)
        for j in range(len(rp)):
            sl = slice(j * 7, j * 7 + 7)
            if (rz[sl] < 0.3).any():
                continue
            pts = list(zip(rx[sl].tolist(), ry[sl].tolist()))
            w_ = max(1, int(0.05 * float(rinv[sl].mean())))
            self.items.append((1, -float(rz[sl].mean()) + 0.1, 2, (lambda p_, w__: (lambda cv, dd: dd.line(p_, fill=(170, 116, 70), width=w__, joint="curve")))(pts, w_)))
        # alarm beacons on every 2nd post (rotating red light)
        red = np.array([255, 40, 60], np.float32)
        pulse = (0.5 + 0.5 * math.sin(t * TAU * 1.6)) ** 2
        bl = []
        for (i, zs) in posts:
            if i % 2 or zs > 0:
                continue
            c = np.array([i * 3.0, 1.02, zs])
            bl.append((box_corners(c, (0.13, 0.1, 0.13)), (255, 70, 80)))
            ang = t * 7 + i
            R.points3d(c[None, :].astype(np.float32), (red * (20 + 40 * pulse))[None, :], 0.18, 0.0, bloom=1.6)
            beam = c + np.array([math.cos(ang) * 1.4, 0, math.sin(ang) * 1.4])
            R.lines3d(c[None, :].astype(np.float32), beam[None, :].astype(np.float32), red * 2.2, red * 0.0, width=14, bloom=0.0)
        self.push_boxes(bl, haze=haze)
        # ---- characters
        run_t = t
        rpos = np.array([xr, 0.0, 0.0])
        rparts = char_parts(rpos, math.pi / 2, run_t, self.freq, 1.0, "carry", lean=0.2, cols=self.runner_cols)
        self.push_boxes(rparts)
        self.shadow(rpos + [0, 0.01, 0], 0.55, (150, 100, 60), bias=0.6)
        by = 0.13 * abs(math.cos(ph))
        sq = 1 + 0.06 * math.cos(2 * ph)
        egg_p = rpos + np.array([0.22, 2.12 * sq + by, 0.0])
        _, esx, esy = hop(t, 1.0 / self.freq / 2, 0.0)
        wob = 9 * math.sin(ph + 0.6)
        res = self.push_sprite(self.egg, egg_p, 0.95, esx, esy, wob, anchor=1.0, bias=-0.3)
        guards = [(xr - 2.0 - 0.5 * math.sin(t * 0.7), -0.45, 0.0), (xr - 3.4 - 0.45 * math.sin(t * 0.9 + 1), 0.5, 0.37),
                  (xr - 4.9 - 0.6 * math.sin(t * 0.6 + 2), -0.15, 0.71)]
        eyes = []
        for gx, gz, gph in guards:
            gpos = np.array([gx, 0.0, gz])
            gp = char_parts(gpos, math.pi / 2, t + gph / self.freq, self.freq * 1.04, 1.12, "run", lean=0.28, cols=self.guard_cols)
            self.push_boxes(gp)
            self.shadow(gpos + [0, 0.01, 0], 0.6, (150, 100, 60), bias=0.6)
            M = rot_y(math.pi / 2)
            for s_ in (-1, 1):
                eyes.append(gpos + M @ np.array([s_ * 0.1, 1.77 * 1.12, 0.3]) + [0, 0.13 * 1.12 * abs(math.cos((t + gph / self.freq) * self.freq * 1.04 * TAU)), 0])
        # "!" alert over the lead guard
        g0 = guards[0]
        hy_, asx, asy = hop(t, 0.55, 0.25)
        self.push_sprite(self.alert, (g0[0], 2.75 + hy_, g0[1]), 0.62, asx, asy, 8 * math.sin(t * 5), anchor=1.0)
        # ---- dust puffs at every footstep (runner + guards)
        Ts = 0.5 / self.freq
        life = 0.5
        for who, (ox, oz, sc, phs) in enumerate([(0.0, 0.0, 1.0, 0.0)] + [(g[0] - xr, g[1], 1.1, g[2]) for g in guards]):
            kk = math.floor((t + phs / self.freq) / Ts)
            for j in range(4):
                ks = kk - j
                tk = ks * Ts - phs / self.freq
                age = t - tk
                if age < 0 or age > life:
                    continue
                xk = self.xr(tk) + ox
                side = 0.17 if ks % 2 else -0.17
                q = age / life
                for m_ in range(3):
                    off = np.array([-0.25 * m_ - 0.6 * q, 0.08 + 0.25 * q + 0.07 * m_, oz + side + (m_ - 1) * 0.12])
                    P = np.array([xk, 0, 0]) + off
                    x_, y_, z_, inv_ = R.project(P[None, :].astype(np.float32))
                    if z_[0] < 0.3:
                        continue
                    rr = (0.17 + 0.3 * q) * (1 - 0.25 * m_) * math.sin(math.pi * min(1.0, q * 1.15)) ** 0.6 * sc * float(inv_[0])
                    e = [float(x_[0]) - rr, float(y_[0]) - rr, float(x_[0]) + rr, float(y_[0]) + rr]
                    cc = tint(lerp_col((255, 250, 240), (235, 214, 196), q))
                    self.items.append((1, -float(z_[0]) + 0.05, 2, (lambda e_, c_: (lambda cv, dd: dd.ellipse(e_, fill=c_)))(e, cc)))
        self.flush(canvas, d)
        # ---- light: guard eyes, egg glow + sparkle trail, speed lines, motion streaks, alarm vignette
        E = np.array(eyes, np.float32)
        R.points3d(E, (red * 16)[None, :].repeat(len(E), 0), 0.05, 0.0, bloom=1.8)
        g = self.egg_col
        if res is not None:
            ex, ey, inv = res
            ecy = ey - 0.5 * 0.95 * inv
            R.soft(np.array([ex]), np.array([ecy]), (g * 3000 * (0.8 + 0.2 * math.sin(t * 6)))[None, :])
            self.rays_soft(ex, ecy, t * 0.8 + np.arange(10) * TAU / 10, 1.3 * inv, 0.08, g, 0.6)
        nt = 20
        dt = np.arange(1, nt + 1) * 0.028
        tp = t - dt
        TP = np.stack([self.xr(tp) + 0.28 - 0.1 * np.sin(self.trail_ph[:nt] * 30 + tp * 9), 2.75 + 0.18 * np.sin(self.trail_ph[:nt] * 17 + tp * 7),
                       0.2 * np.sin(self.trail_ph[:nt] * 11 + tp * 5)], 1).astype(np.float32)
        x, y, z, inv = R.project(TP)
        fade = (1 - dt / dt[-1])
        sparkle(R, x[::3], y[::3], 0.12 * inv[::3] * fade[::3], (g * 0.6 + 100)[None, :] * (1.4 * fade[::3])[:, None])
        R.points(x, y, (g * 6)[None, :] * fade[:, None], 2.0 * self.k, bloom=0.8)
        # motion direction on screen
        mx, my, _, _ = R.project(np.array([[xr, 1.2, 0], [xr + 1, 1.2, 0]], np.float32))
        dx, dy = float(mx[1] - mx[0]), float(my[1] - my[0])
        ln = math.hypot(dx, dy) + 1e-6
        dx, dy = dx / ln, dy / ln
        W, H = self.W, self.H
        span = math.hypot(W, H)
        sl = self.sl
        u = ((sl[:, 1] + t * sl[:, 2] * (1.2 + 1.6 * sp)) % 1.0) * 1.4 - 0.7
        off = sl[:, 0] * span
        cx_, cy_ = W / 2 - dx * u * span - dy * off, H / 2 - dy * u * span + dx * off
        Lh = sl[:, 3] * span * (0.6 + 0.6 * sp)
        cen_d = np.hypot((cx_ - W / 2) / W, (cy_ - H * 0.45) / H)
        inten = np.clip((cen_d - 0.12) / 0.25, 0, 1)[:, None] * (1.6 + 1.6 * sp)
        white = np.array([255, 255, 255], np.float32)
        R.lines(cx_, cy_, cx_ + dx * Lh, cy_ + dy * Lh, white * inten * 0.0, white * inten, width=2.6, bloom=0.2)
        # motion streaks behind the runner
        SP = np.array([[xr, 1.1, 0], [xr, 1.75, 0], [xr, 0.5, 0], [xr + 0.28, 2.8, 0]], np.float32)
        x, y, _, inv = R.project(SP)
        cols = np.array([a, (250, 212, 170), (56, 86, 190), g], np.float32)
        Ls = inv * (1.2 + 1.4 * sp)
        for o in (-0.25, 0.0, 0.25):
            R.lines(x + dy * o * inv * 0.5, y - dx * o * inv * 0.5, x - dx * Ls + dy * o * inv * 0.5, y - dy * Ls - dx * o * inv * 0.5,
                    cols * 0.9, cols * 0.0, width=7.0, bloom=0.3)
        self._soft += self.vig * (red * (0.2 + 0.6 * pulse))
        return self.fin(canvas, self._soft, canvas_bloom=0.08, bloom=(0.55, 0.5), radii=(1.4, 8.0))

    def draw_base(self, t, B, haze):
        """Colourful blocky base: platform, walls, two towers with pointy roofs, flag, glowing pad, beacon beam."""
        a = self.acc
        R = self.R
        parts = [(box_corners(B + [0, -0.4, 0], (4.0, 0.4, 3.0)), (110, 214, 100)),
                 (box_corners(B + [0, 1.2, -1.6], (2.6, 1.2, 0.6)), tuple(a * 0.85 + 30)),
                 (box_corners(B + [-2.6, 1.8, -1.0], (0.8, 1.8, 0.8)), (240, 240, 250)),
                 (box_corners(B + [2.6, 1.8, -1.0], (0.8, 1.8, 0.8)), (240, 240, 250)),
                 (box_corners(B + [0, 2.6, -1.6], (0.9, 0.5, 0.65)), (255, 220, 90))]
        for k_ in range(5):
            parts.append((box_corners(B + [-2.0 + k_, 2.55, -1.6], (0.22, 0.15, 0.62)), tuple(a * 0.85 + 30)))
        self.push_boxes(parts, haze=haze)
        Q, ref = self.cone_cache[1]
        for sx in (-2.6, 2.6):
            sc = np.array([1.1, 1.4, 1.1])
            o = B + [sx, 3.6, -1.0]
            self.push_faces(Q * sc + o, (255, 90, 110), ref=ref * sc + o, haze=haze)
        # flag
        fp = B + [2.6, 5.0, -1.0]
        fl = np.array([fp + [0, 1.2, 0], fp + [1.2, 1.05 + 0.12 * math.sin(t * 6), 0.15 * math.sin(t * 5)],
                       fp + [1.2, 0.45 + 0.12 * math.sin(t * 6 + 1), 0.15 * math.sin(t * 5 + 1)], fp + [0, 0.5, 0]])
        self.push_faces(fl[None], (255, 230, 60), cull=False, haze=haze)
        self.push_boxes([(box_corners(fp + [0, 0.6, 0], (0.05, 0.6, 0.05)), (90, 90, 100))], haze=haze)
        # glowing pad + beacon beam + bouncing arrow
        pad = B + [0, 0.02, 1.0]
        g = np.array([90, 255, 160], np.float32)
        self.pool(self._soft, pad, 1.6, g * 0.7, squash=0.4)
        top = pad + [0, 14, 0]
        for w_, c_ in ((90, 0.5), (40, 1.0)):
            R.lines3d(pad[None, :].astype(np.float32), top[None, :].astype(np.float32), g * c_, g * 0.0, width=w_, bloom=0.0)
        hy_, sx_, sy_ = hop(t, 0.8, 0.6)
        ap = pad + [0, 3.2 + hy_, 0]
        arr = np.array([[-0.5, 0.6], [0.5, 0.6], [0.5, 0.0], [0.95, 0.0], [0.0, -0.9], [-0.95, 0.0], [-0.5, 0.0]]) * 0.9
        x, y, z, inv = R.project(ap[None, :].astype(np.float32))
        if z[0] > 0.3:
            s = float(inv[0])
            pts = [(float(x[0]) + px * s * sx_, float(y[0]) - py * s * sy_) for px, py in arr]
            out = [(float(x[0]) + px * s * sx_ * 1.18, float(y[0]) - (py - 0.02) * s * sy_ * 1.15) for px, py in arr]
            self.items.append((1, -float(z[0]), 2, lambda cv, dd: (dd.polygon(out, fill=(255, 255, 255)), dd.polygon(pts, fill=(60, 230, 120)))))


# ======================================================================== 3. egg hatch
class EggHatch(Eggs):
    """Close-up hatch on a glowing pedestal, looping every 4.5 s: the egg drops in (squash on landing),
    shakes harder and harder, cracks spread across the shell and light leaks out, it swells and BURSTS
    (white flash, shockwave ring, spinning shell fragments, confetti, star burst) revealing a cute
    original round pet (ears / horns / sprout, big glossy eyes) that pops in elastically and bounces
    happily on a rotating rarity glow ring with floating hearts, then poofs away for the next egg.
    Sunburst backdrop, pedestal rim lights, camera sway + burst shake.
    params.rarity common|rare|epic|legendary = colour of egg, glow, rays and confetti."""
    default_accent = "#FF5FA2"
    T = 4.5
    TB = 2.65  # burst time inside the cycle

    def init(self):
        rng = self.rng
        a = self.acc
        k = self.k
        self.light = _norm([-0.4, 0.8, 0.6])
        self.rarity = parse_rarity(self.p.get("rarity"), "legendary")
        self.g = np.asarray(EGG_PAL[self.rarity][3], np.float32)
        self.eh = 2.3
        self.egg = egg_sprite(820 * k, self.rarity, rng)
        W_, H_ = self.egg.size
        R_ = H_ / 2 * 0.97
        # cracks: a zig-zag ring around the middle (grows from the centre out) + branches
        main = []
        n = 16
        for i in range(n + 1):
            u = -0.78 + 1.56 * i / n
            v = 0.12 + (0.075 if i % 2 else -0.075) + rng.uniform(-0.02, 0.02)
            main.append((u, v))
        cr = []
        mid = n // 2
        cr.append(main[mid::-1])
        cr.append(main[mid:])
        for _ in range(5):
            j = int(rng.integers(2, n - 1))
            u, v = main[j]
            br = [(u, v)]
            dv = rng.choice([-1, 1]) * rng.uniform(0.08, 0.13)
            for _ in range(3):
                u += rng.uniform(-0.09, 0.09)
                v += dv
                br.append((u, v))
            cr.append(br)
        self.cracks = []
        for line in cr:
            pts = []
            for u, v in line:
                aa = 0.74 * (1 - 0.14 * v) * math.sqrt(max(0.0, 1 - v * v)) * 0.96
                u = max(-aa, min(aa, u))
                pts.append((W_ / 2 + u * R_, H_ / 2 - v * R_))
            self.cracks.append(pts)
        self.crack_w = max(2, int(10 * k))
        # pets (one per cycle, rotating)
        self.pets = []
        kinds = rng.permutation(7)
        for i in range(4):
            body = lerp_col(PET_COLS[int(rng.integers(0, len(PET_COLS)))], self.g, 0.3 if self.rarity != "common" else 0.0)
            kd = int(kinds[i])
            self.pets.append((pet_sprite(520 * k, body, kd, rng, False, a), pet_sprite(520 * k, body, kd, rng, True, a)))
        # shell fragments: jagged polygons (local 2D, world units)
        nfr = 18
        self.fr = []
        shell = np.asarray(EGG_PAL[self.rarity][0], np.float32)
        for i in range(nfr):
            ang = i / nfr * TAU + rng.uniform(-0.2, 0.2)
            m = int(rng.integers(3, 6))
            th = np.sort(rng.uniform(0, TAU, m))
            rr = rng.uniform(0.18, 0.36, m)
            poly = np.stack([np.cos(th) * rr, np.sin(th) * rr], 1)
            p0 = np.array([math.cos(ang) * 0.7, 1.8 + math.sin(ang) * 0.9, rng.uniform(-0.3, 0.4)])
            v0 = np.array([math.cos(ang) * rng.uniform(2.5, 4.5), rng.uniform(3.5, 7.0) + math.sin(ang) * 2, rng.uniform(0.5, 2.5)])
            self.fr.append((poly, p0, v0, rng.uniform(-9, 9), rng.uniform(4, 10), shell * rng.uniform(0.85, 1.05)))
        # confetti
        nc = 130
        ang = rng.uniform(0, TAU, nc)
        sp = rng.uniform(2.0, 7.0, nc)
        self.cf_v = np.stack([np.cos(ang) * sp * 0.9, rng.uniform(3, 9, nc), rng.uniform(-1.0, 3.5, nc)], 1)
        self.cf_p = np.stack([rng.uniform(-0.3, 0.3, nc), rng.uniform(1.3, 2.0, nc), rng.uniform(-0.2, 0.2, nc)], 1)
        pal = np.array([self.g, (255, 90, 140), (90, 200, 255), (255, 230, 80), (120, 240, 140), (255, 255, 255), a], np.float32)
        self.cf_c = pal[rng.integers(0, len(pal), nc)]
        self.cf_c[: nc // 3] = self.g
        self.cf_ph = rng.uniform(0, TAU, (nc, 3))
        self.cf_s = rng.uniform(0.06, 0.11, nc)
        nst = 22
        d3 = rng.normal(0, 1, (nst, 3))
        d3[:, 2] = np.abs(d3[:, 2]) * 0.5
        self.st_d = d3 / np.linalg.norm(d3, axis=1, keepdims=True) * rng.uniform(2.0, 4.0, (nst, 1))
        self.st_ph = rng.random(nst)
        self.star = star_xy(5, 0.46)
        self.heart = heart_xy(24)
        self.hearts_ph = rng.random(6)
        # pedestal geometry
        self.ped_q, self.ped_ref = lathe([(1.5, 0.0), (1.5, 0.18), (1.25, 0.28), (1.2, 0.5), (1.32, 0.56), (1.32, 0.62)], 18)
        nq = len(self.ped_q) // 18
        cols = []
        for j in range(nq):
            base = [(255, 255, 255), (250, 246, 255), (255, 224, 120), (250, 246, 255), (255, 214, 90)][j % 5]
            cols += [base] * 18
        self.ped_c = np.array(cols, np.float32)
        th = np.linspace(0, TAU, 19)[:-1]
        self.ped_top = np.stack([np.cos(th) * 1.32, np.full(18, 0.62), np.sin(th) * 1.32], 1)
        self.floor_rings = []
        for r0, r1, c in ((7.5, 5.0, 0.92), (5.0, 3.2, 1.0), (3.2, 1.6, 0.9)):
            ring = []
            for i in range(24):
                a0, a1 = i / 24 * TAU, (i + 1) / 24 * TAU
                ring.append([(r0 * math.cos(a0), 0, r0 * math.sin(a0)), (r0 * math.cos(a1), 0, r0 * math.sin(a1)),
                             (r1 * math.cos(a1), 0, r1 * math.sin(a1)), (r1 * math.cos(a0), 0, r1 * math.sin(a0))])
            self.floor_rings.append((np.array(ring, np.float64), c))
        # backdrop: saturated rarity / accent gradient + rotating sunburst (quarter res soft)
        deep = {"common": (90, 150, 255), "rare": (60, 70, 220), "epic": (120, 40, 200), "legendary": (240, 90, 60)}[self.rarity]
        deep = lerp_col(deep, a, 0.25)
        self.bg = Image.fromarray(vgrad(self.W, self.H, [(0, tint(deep * 0.75)), (0.4, tint(lerp_col(deep, self.g, 0.45))),
                                                          (0.62, tint(lerp_col(self.g, (255, 255, 255), 0.2))), (1, tint(deep * 0.9))]))
        fl = lerp_col(self.g, (255, 255, 255), 0.25)
        self.floor_cols = [lerp_col(deep, (255, 255, 255), 0.35), fl, lerp_col(a, (255, 255, 255), 0.45)]
        self.top_col = lerp_col(deep, (255, 255, 255), 0.3)
        self.shc = tint(self.top_col * 0.95 * 0.72)
        yy, xx = self._rgrid()
        self.cx0, self.cy0 = self.R.Ws * 0.5, self.R.Hs * 0.36
        self.ang = np.arctan2(yy - self.cy0, xx - self.cx0)
        self.rad = np.hypot(xx - self.cx0, yy - self.cy0) / self.R.Ws

    def frame(self, t):
        R, a, g = self.R, self.acc, self.g
        T, TB = self.T, self.TB
        cyc = math.floor(t / T)
        u = t - cyc * T
        tau = u - TB
        flash = math.exp(-tau / 0.16) if tau >= 0 else 0.0
        boom = math.exp(-tau / 0.35) * math.sin(tau * 55) if tau >= 0 else 0.0
        cpos = np.array([1.3 * math.sin(t * 0.27), 2.9 + 0.25 * math.sin(t * 0.31), 7.4 - 0.5 * math.sin(t * 0.21)])
        cpos += np.array([0.12 * boom, 0.1 * boom, 0])
        cam = Camera(cpos, (0.0, 1.75 + 0.06 * boom, 0.0), fov=52, roll=0.03 * math.sin(t * 0.4) + 0.02 * boom)
        canvas, d = self.begin(cam, self.bg)
        # ---- sunburst backdrop (soft)
        burst_gain = 0.45 + 0.4 * (math.exp(-max(0.0, tau) / 1.2) if tau >= 0 else min(1.0, max(0.0, (u - 1.0) / 1.6)) * 0.6)
        st = np.cos(self.ang * 12 + t * 0.6)
        rays = np.clip((st - 0.25) / 0.5, 0, 1) * np.clip(1.25 - self.rad * 1.1, 0, 1)
        self._soft += rays[..., None] * (g * 0.3 + 70) * burst_gain * 0.45
        # ---- floor + pedestal
        for j, (Q, c) in enumerate(self.floor_rings):
            col = self.floor_cols[j]
            self.push_faces(Q, col, layer=0, ref=np.array([0, -10, 0]), amb=0.8, diff=0.25)
        self.push_faces(self.ped_q, self.ped_c, ref=self.ped_ref, spec=0.25)
        self.push_faces(self.ped_top[None], self.top_col, ref=np.array([0, -5, 0]), amb=0.9, diff=0.2)
        th = np.linspace(0, TAU, 49)
        ring = np.stack([np.cos(th) * 1.31, np.full(49, 0.62), np.sin(th) * 1.31], 1).astype(np.float32)
        rg = 0.7 + 0.3 * math.sin(t * 4) + 1.2 * flash
        R.lines3d(ring[:-1], ring[1:], g * 1.6 * rg, width=4.0, bloom=0.7)
        ring2 = np.stack([np.cos(th) * 1.5, np.full(49, 0.09), np.sin(th) * 1.5], 1).astype(np.float32)
        R.lines3d(ring2[:-1], ring2[1:], g * 1.2 * rg, width=3.0, bloom=0.8)
        self.pool(self._soft, np.array([0, 0.65, 0]), 1.5, g * (0.18 + 0.12 * rg), squash=0.4)
        top_y = 0.62
        # ---- egg (before burst)
        if u < TB:
            if u < 0.45:
                yo = 6.0 * (1 - (u / 0.45) ** 2)
                sx, sy, rot, xo = 0.92, 1.1, 0.0, 0.0
            else:
                sq = math.sin(math.pi * min(1.0, (u - 0.45) / 0.3)) * (1 if u < 0.75 else 0)
                sx, sy = 1 + 0.2 * sq, 1 - 0.22 * sq
                yo = 0.0
                inten = min(1.0, max(0.0, (u - 0.75) / (TB - 0.75))) ** 1.5
                rot = inten * 17 * math.sin(u * 37) + 3 * math.sin(u * 9)
                xo = inten * 0.07 * math.sin(u * 53)
                for h0, h1, hh in ((1.45, 1.72, 0.18), (2.0, 2.3, 0.3)):
                    if h0 <= u < h1:
                        q = (u - h0) / (h1 - h0)
                        yo = hh * 4 * q * (1 - q)
                        sx, sy = sx * (1 - 0.06 * (1 - abs(2 * q - 1))), sy * (1 + 0.1 * (1 - abs(2 * q - 1)))
                sw = max(0.0, (u - 2.42) / (TB - 2.42))
                sx *= 1 + 0.12 * sw
                sy *= 1 + 0.08 * sw
            crack = min(1.0, max(0.0, (u - 1.1) / 1.3))
            spr = self.egg
            if crack > 0:
                spr = spr.copy()
                dd = ImageDraw.Draw(spr)
                leak0 = min(1.0, max(0.0, (u - 1.6) / (TB - 1.6)))
                dark = tint(lerp_col(np.asarray(EGG_PAL[self.rarity][0], np.float32) * 0.35, (255, 255, 236), leak0 ** 0.7))
                for j, line in enumerate(self.cracks):
                    f = crack if j < 2 else max(0.0, (crack - 0.35) / 0.65)
                    nseg = (len(line) - 1) * f
                    if nseg <= 0:
                        continue
                    k_ = int(nseg)
                    pts = line[:k_ + 1]
                    if k_ < len(line) - 1:
                        fr = nseg - k_
                        (x0, y0), (x1, y1) = line[k_], line[k_ + 1]
                        pts = pts + [(x0 + (x1 - x0) * fr, y0 + (y1 - y0) * fr)]
                    if len(pts) >= 2:
                        dd.line(pts, fill=dark + (255,), width=self.crack_w if j < 2 else max(1, self.crack_w * 2 // 3), joint="curve")
            P = np.array([xo, top_y + yo, 0.0])
            res = self.push_sprite(spr, P, self.eh, sx, sy, rot, anchor=1.0, bias=-2.5)
            self.shadow((0, top_y + 0.005, 0), 0.5 * sx / (1 + yo * 0.3), self.shc, bias=-2.0, squash=0.28)
            if res is not None:
                ex, ey, inv = res
                hpx = self.eh * inv * sy
                ecx, ecy = ex - math.sin(math.radians(rot)) * hpx * 0.5, ey - math.cos(math.radians(rot)) * hpx * 0.5
                leak = min(1.0, max(0.0, (u - 1.6) / (TB - 1.6)))
                R.soft(np.array([ecx]), np.array([ecy]), (g * (1800 + 9000 * leak ** 2))[None, :])
                if leak > 0:  # light leaking through the cracks
                    W_, H_ = self.egg.size
                    s_ = hpx / H_
                    ca, sa = math.cos(math.radians(rot)), math.sin(math.radians(rot))
                    for j, line in enumerate(self.cracks[:2]):
                        L = np.array(line, np.float32)
                        nseg = max(1, int((len(L) - 1) * crack))
                        L = L[:nseg + 1]
                        dx = (L[:, 0] - W_ / 2) * s_ * sx / sy
                        dy = (L[:, 1] - H_ / 2) * s_
                        X = ecx + dx * ca + dy * sa
                        Y = ecy - dx * sa + dy * ca
                        c_ = (g * 0.6 + 100) * (2.0 + 5.0 * leak)
                        R.lines(X[:-1], Y[:-1], X[1:], Y[1:], c_, width=5.0, bloom=1.6)
                    if leak > 0.3:
                        self.rays_soft(ecx, ecy, t * 1.5 + np.arange(12) * TAU / 12, hpx * (0.8 + 0.8 * leak), 0.05, g, 0.6 * leak)
                if u < 0.45:  # falling streak
                    R.lines(np.array([ecx]), np.array([ecy - hpx * 0.6]), np.array([ecx]), np.array([ecy - hpx * 2.0]),
                            (g * 0.5 + 120) * 1.5, (g * 0.0), width=hpx * 0.5, bloom=0.0)
        else:
            # ---- shell fragments
            for (poly, p0, v0, w, sw_, col) in self.fr:
                if tau > 2.0:
                    break
                P = p0 + v0 * tau + np.array([0, -9.0, 0]) * 0.5 * tau * tau
                x, y, z, inv = R.project(P[None, :].astype(np.float32))
                if z[0] < 0.3:
                    continue
                ang = w * tau
                flip = math.cos(sw_ * tau)
                ca, sa = math.cos(ang), math.sin(ang)
                pts = [(float(x[0]) + (px * flip * ca - py * sa) * float(inv[0]), float(y[0]) - (px * flip * sa + py * ca) * float(inv[0]))
                       for px, py in poly]
                cc = tint(col * (0.75 + 0.35 * abs(flip))) if flip > 0 else tint(lerp_col(col, (255, 255, 250), 0.6))
                self.items.append((1, -float(z[0]), 2, (lambda p_, c_: (lambda cv, dd: dd.polygon(p_, fill=c_, outline=tint(np.array(c_) * 0.7))))(pts, cc)))
            # ---- pet
            pi = cyc % len(self.pets)
            if u < 4.15:
                pop = elastic(tau / 0.75)
                hy_, sx, sy = hop(max(0.0, tau - 0.55), 0.55, 0.45) if tau > 0.55 else (0.0, 1.0, 1.0)
                blink = ((tau % 1.6) > 1.3) and tau > 0.9
                wob = 7 * math.sin(tau * 6.5)
                spin = 0.0
            else:
                q = min(1.0, (u - 4.15) / 0.22)
                pop, hy_, sx, sy, blink, wob = 1 - ease(q), 0.0, 1.0, 1.0, True, 0.0
                spin = 360 * q
            if pop > 0.01:
                spr = self.pets[pi][1 if blink else 0]
                res = self.push_sprite(spr, (0, top_y + hy_, 0.0), 1.9 * pop, sx, sy, wob + spin, anchor=1.0, bias=-2.5)
                self.shadow((0, top_y + 0.005, 0), 0.55 * pop / (1 + hy_), self.shc, bias=-2.0)
                if res is not None:
                    px_, py_, inv = res
                    R.soft(np.array([px_]), np.array([py_ - 0.9 * inv]), (g * 220 * pop)[None, :])
                    # floating hearts
                    for j, hp in enumerate(self.hearts_ph):
                        hq = (tau * 0.6 + hp) % 1.0
                        if tau < 0.6:
                            continue
                        hx = px_ + (j - 2.5) * 0.35 * inv + math.sin(hq * 7 + j) * 0.1 * inv
                        hyy = py_ - (1.6 + 1.6 * hq) * inv
                        hs = 0.16 * inv * math.sin(math.pi * hq) * pop
                        pts = [(hx + x_ * hs, hyy - y_ * hs) for x_, y_ in self.heart.tolist()]
                        self.items.append((2, 0.0, 2, (lambda p_: (lambda cv, dd: dd.polygon(p_, fill=(255, 92, 140), outline=(255, 255, 255))))(pts)))
            # rotating rarity ring under the pet
            th2 = np.linspace(0, TAU, 25)[:-1] + t * 2.0
            gr = 1.05 + 0.08 * math.sin(t * 5)
            A = np.stack([np.cos(th2) * gr, np.full(24, top_y + 0.03), np.sin(th2) * gr], 1).astype(np.float32)
            B = np.stack([np.cos(th2 + 0.14) * gr, np.full(24, top_y + 0.03), np.sin(th2 + 0.14) * gr], 1).astype(np.float32)
            R.lines3d(A, B, g * 4.0 * (pop if u >= 4.15 else 1.0), width=6.0, bloom=1.4)
            # confetti
            if tau < 3.0:
                kd, gy = 2.2, -7.0
                vt = np.array([0, gy / kd, 0])
                e = (1 - math.exp(-kd * tau)) / kd
                P = self.cf_p + vt * tau + (self.cf_v - vt) * e
                P[:, 0] += 0.25 * np.sin(tau * 4 + self.cf_ph[:, 0])
                x, y, z, inv = R.project(P.astype(np.float32))
                for i in np.nonzero(z > 0.3)[0].tolist():
                    s_ = self.cf_s[i] * float(inv[i])
                    ang = tau * 8 + self.cf_ph[i, 1]
                    fl = math.cos(tau * 9 + self.cf_ph[i, 2])
                    ca, sa = math.cos(ang), math.sin(ang)
                    w2, h2 = s_ * fl, s_ * 0.55
                    X, Y = float(x[i]), float(y[i])
                    pts = [(X + w2 * ca - h2 * sa, Y + w2 * sa + h2 * ca), (X - w2 * ca - h2 * sa, Y - w2 * sa + h2 * ca),
                           (X - w2 * ca + h2 * sa, Y - w2 * sa - h2 * ca), (X + w2 * ca + h2 * sa, Y + w2 * sa - h2 * ca)]
                    c_ = tint(self.cf_c[i] * (0.7 + 0.3 * abs(fl)))
                    self.items.append((1, -float(z[i]), 0, pts, c_))
            # star burst + shockwave
            if tau < 1.6:
                e = 1 - math.exp(-3.2 * tau)
                P = np.array([0, 1.8, 0]) + self.st_d * e
                fade = math.exp(-tau * 1.6)
                x, y, z, inv = R.project(P.astype(np.float32))
                for i in range(len(P)):
                    if z[i] < 0.3:
                        continue
                    s_ = 0.17 * float(inv[i]) * (0.5 + 0.5 * fade)
                    ro = tau * 4 + self.st_ph[i] * TAU
                    ca, sa = math.cos(ro), math.sin(ro)
                    pts = [(float(x[i]) + (px * ca - py * sa) * s_, float(y[i]) + (px * sa + py * ca) * s_) for px, py in self.star.tolist()]
                    self.items.append((2, 0.0, 2, (lambda p_: (lambda cv, dd: dd.polygon(p_, fill=(255, 236, 90), outline=(255, 255, 255))))(pts)))
                sparkle(R, x, y, 0.3 * inv * fade, np.repeat(((g * 0.5 + 128) * 2.0 * fade)[None, :], len(x), 0))
                rr = 0.5 + 4.5 * (1 - math.exp(-3.5 * tau))
                ring = np.stack([np.cos(th) * rr, np.full(49, top_y), np.sin(th) * rr], 1).astype(np.float32)
                R.lines3d(ring[:-1], ring[1:], (g * 0.5 + 128) * 5 * fade, width=10.0 * fade + 2, bloom=1.0)
            if u >= 4.15:  # poof sparkles as the pet vanishes
                q = (u - 4.15) / 0.35
                ang = np.arange(10) * TAU / 10
                P = np.stack([np.cos(ang) * 1.2 * q, 1.4 + np.sin(ang) * 1.2 * q, np.zeros(10)], 1).astype(np.float32)
                x, y, z, inv = R.project(P)
                sparkle(R, x, y, 0.25 * inv * (1 - q), np.repeat(((g * 0.5 + 128) * 2.0 * (1 - q))[None, :], 10, 0))
        self.flush(canvas, d)
        # ambient floating sparkles
        ph = np.arange(30) * 0.618 % 1.0
        q = (ph + t * 0.15) % 1.0
        ang = np.arange(30) * 2.4
        P = np.stack([np.cos(ang) * (2.2 + ph * 2), 0.3 + q * 4.5, np.sin(ang) * (1.5 + ph * 2) - 1.0], 1).astype(np.float32)
        x, y, z, inv = R.project(P)
        tw = np.sin(q * math.pi) * (0.5 + 0.5 * np.sin(t * 4 + ang * 3))
        sparkle(R, x, y, 0.09 * inv * tw, ((g * 0.4 + 150) * 1.2)[None, :] * tw[:, None])
        if flash > 0.01:
            self._soft += flash * 70
        out = self.fin(canvas, self._soft, canvas_bloom=0.10, bloom=(0.55, 0.5), radii=(1.4, 8.0))
        if flash > 0.01:
            out += np.float32(95 * flash)
        return out


# ======================================================================== 4. pet tycoon base
def arrow_sprite(S: int, fill=(70, 235, 110)) -> Image.Image:
    S = max(8, int(S))
    ss = 2
    W = S * ss
    im = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    pts = [(0.5, 0.06), (0.92, 0.5), (0.68, 0.5), (0.68, 0.94), (0.32, 0.94), (0.32, 0.5), (0.08, 0.5)]
    P = [(x * W, y * W) for x, y in pts]
    d.polygon(P, fill=(255, 255, 255, 255))
    c = (0.5 * W, 0.55 * W)
    Pi = [(c[0] + (x - c[0]) * 0.8, c[1] + (y - c[1]) * 0.8) for x, y in P]
    d.polygon(Pi, fill=tuple(fill) + (255,))
    hi = [(c[0] + (x - c[0]) * 0.55, c[1] + (y - c[1]) * 0.55 - 0.05 * W) for x, y in P[:3]]
    d.polygon(hi, fill=tuple(int(min(255, v + 70)) for v in fill) + (255,))
    return im.resize((S, S), Image.LANCZOS)


class PetBase(Eggs):
    """Pet tycoon base on a checker-tiled platform: a 3x3 grid of glowing pedestals with different
    cute original blob pets bouncing out of phase, each popping floating "+$" income numbers and
    flinging spinning gold coins in arcs into a big counter board whose total ticks up (board bumps on
    every hit); a treadmill in front with a blocky character sprinting on the moving belt (speed lines,
    sweat drops, "+1" speed pops); green upgrade arrows pulse and bob; trees, hills and clouds around;
    the camera slowly cranes and drifts. params: income 0..1 (how fast money pops)."""
    default_accent = "#36D17A"
    NPET = 9

    def init(self):
        rng = self.rng
        a = self.acc
        k = self.k
        self.light = _norm([-0.4, 0.85, 0.5])
        self.inc = parse_unit(self.p.get("income", self.p.get("intensity")), 0.6)
        xs, zs = (-2.6, 0.0, 2.6), (-6.4, -3.7, -1.0)
        self.ped = [(x, z) for z in zs for x in xs]
        kinds = list(rng.permutation(7)) + list(rng.integers(0, 7, 2))
        cols = rng.permutation(len(PET_COLS))
        self.pets = []
        for i in range(self.NPET):
            body = PET_COLS[int(cols[i % len(cols)])]
            self.pets.append((pet_sprite(330 * k, body, int(kinds[i]), rng, False, a), pet_sprite(330 * k, body, int(kinds[i]), rng, True, a)))
        rar = [RARITIES[int(r)] for r in rng.integers(0, 4, self.NPET)]
        rar[4] = "legendary"
        self.ped_g = np.array([EGG_PAL[r][3] for r in rar], np.float32)
        self.ped_q, self.ped_ref = lathe([(0.85, 0.0), (0.85, 0.12), (0.72, 0.2), (0.7, 0.42), (0.8, 0.48), (0.8, 0.55)], 10)
        nq = len(self.ped_q) // 10
        pc = []
        for j in range(nq):
            pc += [[(250, 250, 255), (235, 236, 248), (255, 214, 90), (235, 236, 248), (255, 214, 90)][j % 5]] * 10
        self.ped_c = np.array(pc, np.float32)
        th = np.linspace(0, TAU, 11)[:-1]
        self.ped_top = np.stack([np.cos(th) * 0.8, np.full(10, 0.55), np.sin(th) * 0.8], 1)
        self.hop_p = rng.uniform(0.5, 0.75, self.NPET)
        self.hop_ph = rng.random(self.NPET)
        self.pop_p = rng.uniform(1.4, 2.2, self.NPET) / (0.6 + 0.8 * self.inc)
        self.pop_ph = rng.random(self.NPET)
        amts = [5, 12, 25, 50, 100, 250, 500]
        self.amt = [amts[int(rng.integers(0, len(amts)))] for _ in range(self.NPET)]
        self.amt[4] = 1000
        self.pop_spr = [text_sprite(f"+${v}", int(120 * k), (110, 255, 120), stroke=(20, 96, 40)) for v in self.amt]
        self.coin = coin_sprite(120 * k)
        self.coin_off = rng.uniform(-1, 1, (self.NPET, 2))
        self.plus1 = text_sprite("+1", int(110 * k), (110, 225, 255), stroke=(20, 60, 140))
        self.arrow = arrow_sprite(200 * k)
        self.arrow_on = [1, 5, 6]
        self.base_cash = float(rng.integers(12, 80)) * 1000
        self.f_num = font(max(8, int(120)), "InterDisplay-Black.otf")
        self.init_props(rng)
        self.trees = [(-6.2, -9.0, 1.3, 0), (6.0, -8.5, 1.2, 1), (-6.6, -4.0, 1.0, 1), (6.6, -3.5, 1.1, 0), (-6.0, 1.5, 0.9, 0),
                      (6.2, 1.8, 1.0, 1), (-3.5, -11.5, 1.4, 0), (3.2, -12.0, 1.3, 0), (0.0, -13.0, 1.5, 1), (-8.5, -7.0, 1.2, 0),
                      (8.8, -6.0, 1.2, 0)]
        self.tree_ph = rng.uniform(0, TAU, len(self.trees))
        self.hills = [(rng.uniform(-30, 30), rng.uniform(-45, -25), rng.uniform(5, 10)) for _ in range(6)]
        self.hill_q = ico_blob(rng, 3, 8, 0.15)
        self.cl_puffs = [cloud_puffs(rng, int(rng.integers(5, 9))) for _ in range(4)]
        ncl = 8
        self.clouds = np.stack([rng.uniform(-30, 30, ncl), rng.uniform(7, 15, ncl), rng.uniform(-40, -22, ncl)], 1)
        self.sweat = rng.random((6, 2))
        # platform tiles
        Q, C = [], []
        light_t = lerp_col(a, (255, 255, 255), 0.55)
        for ix in range(-9, 9):
            for iz in range(-18, 8):
                x0, z0 = ix * 0.5, iz * 0.5
                Q.append([(x0, 0, z0), (x0 + 0.5, 0, z0), (x0 + 0.5, 0, z0 + 0.5), (x0, 0, z0 + 0.5)])
                C.append(light_t if (ix + iz) % 2 else (250, 250, 252))
        self.tile_q = np.array(Q, np.float64)
        self.tile_c = np.array(C, np.float32)
        self.bg = Image.fromarray(vgrad(self.W, self.H, [(0, (70, 160, 255)), (0.35, (140, 210, 255)), (0.55, (200, 236, 255)), (1, (150, 220, 140))]))

    def counter_tex(self, value, bump):
        TW, TH = 720, 240
        im = Image.new("RGBA", (TW, TH), (30, 36, 70, 255))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([8, 8, TW - 9, TH - 9], radius=34, fill=(40, 48, 96), outline=(255, 214, 90), width=8)
        cx, cy, r = 120, TH // 2, 70
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 200, 60), outline=(200, 140, 30), width=6)
        f = self.f_num
        l, t_, rr, b = f.getbbox("$")
        d.text((cx - (rr - l) / 2 - l, cy - (b - t_) / 2 - t_), "$", font=f, fill=(255, 245, 190))
        s = f"{int(value):,}"
        l, t_, rr, b = f.getbbox(s)
        tw = rr - l
        sc = min(1.0, 470 / max(1, tw))
        txt = Image.new("RGBA", (tw + 20, b - t_ + 20), (0, 0, 0, 0))
        g = tint(lerp_col((120, 255, 140), (255, 255, 255), bump))
        ImageDraw.Draw(txt).text((10 - l, 10 - t_), s, font=f, fill=g, stroke_width=6, stroke_fill=(20, 70, 30))
        if sc < 1:
            txt = txt.resize((int(txt.width * sc), int(txt.height * sc)), Image.BILINEAR)
        im.alpha_composite(txt, (int(220 + (470 - txt.width) / 2), int(TH / 2 - txt.height / 2)))
        return im

    def frame(self, t):
        R, a = self.R, self.acc
        cpos = np.array([2.6 * math.sin(t * 0.14), 6.3 + 1.2 * math.sin(t * 0.23 + 0.6), 11.4 - 0.8 * math.sin(t * 0.17)])
        tgt = np.array([0.25 * math.sin(t * 0.2), 0.9, -3.6])
        cam = Camera(cpos, tgt, fov=58, roll=0.02 * math.sin(t * 0.3))
        canvas, d = self.begin(cam, self.bg)
        haze = (16, 50, 0.5, (200, 230, 255))
        # ---- surroundings
        for i in range(len(self.clouds)):
            x0, y0, z0 = self.clouds[i]
            self.push_cloud(self.cl_puffs[i % 4], (((x0 + 0.4 * t + 30) % 60) - 30, y0, z0), 5.0, layer=-1)
        for j, (x, z, s_) in enumerate(self.hills):
            c = np.array([x, -0.6 * s_, z])
            self.push_faces(self.hill_q * np.array([s_ * 1.4, s_, s_]) + c, (110, 200, 100), layer=-1, ref=c, haze=haze)
        g0 = np.array([[-60, -0.02, -60], [60, -0.02, -60], [60, -0.02, 30], [-60, -0.02, 30]], np.float64)
        self.push_faces(g0[None], (124, 214, 96), layer=-1, ref=np.array([0, -5, 0]), haze=haze, cull=False)
        # ---- platform
        self.push_faces(self.tile_q, self.tile_c, layer=0, ref=np.array([0, -5, 0]), amb=0.75, diff=0.3)
        slab = [(box_corners((0, -0.2, -2.5), (4.5, 0.2, 6.5)), (180, 186, 210))]
        self.push_boxes(slab, layer=0, haze=haze)
        for j, (x, z, s_, kind) in enumerate(self.trees):
            sway = 0.05 * math.sin(t * 1.2 + self.tree_ph[j])
            cols = [(70, 170, 90), (86, 190, 100), (104, 206, 112)] if kind == 0 else [(110, 205, 90)]
            self.tree_items(x, z, 0.0, s_, kind, sway, cols, haze=haze)
        # ---- pedestals + pets + income pops + coins
        board = np.array([0.0, 4.4, -9.2])
        bump = 0.0
        cash = self.base_cash
        for i, (x, z) in enumerate(self.ped):
            c = np.array([x, 0.0, z])
            self.push_faces(self.ped_q + c, self.ped_c, ref=self.ped_ref + c, spec=0.2)
            gcol = self.ped_g[i]
            self.push_faces((self.ped_top + c)[None], lerp_col(gcol, (255, 255, 255), 0.45), ref=c - [0, 5, 0], amb=0.9, diff=0.2)
            th = np.linspace(0, TAU, 25)
            ring = np.stack([np.cos(th) * 0.8, np.full(25, 0.56), np.sin(th) * 0.8], 1).astype(np.float32) + c.astype(np.float32)
            R.lines3d(ring[:-1], ring[1:], gcol * (1.2 + 0.6 * math.sin(t * 3 + i)), width=3.0, bloom=0.8)
            hy_, sx, sy = hop(t, self.hop_p[i], 0.35, phase=self.hop_ph[i])
            blink = ((t * 0.7 + self.hop_ph[i] * 3) % 2.3) > 2.15
            wob = 6 * math.sin(t * 4 + i)
            top = c + np.array([0, 0.55, 0])
            self.shadow(top + [0, 0.005, 0], 0.36 / (1 + hy_), lerp_col(gcol, (255, 255, 255), 0.45) * 0.72, bias=-0.4)
            res = self.push_sprite(self.pets[i][1 if blink else 0], top + [0, hy_, 0], 1.25, sx, sy, wob, anchor=1.0, bias=-0.8)
            if res is not None:
                R.soft(np.array([res[0]]), np.array([res[1] - 0.6 * res[2]]), (gcol * 500)[None, :])
            # pop + coin, both launched at the same moments
            P_ = self.pop_p[i]
            n = math.floor(t / P_ + self.pop_ph[i])
            age = t - (n - self.pop_ph[i]) * P_
            if age < 1.15:
                q = age / 1.15
                sc = back_out(min(1.0, age / 0.25)) * (1 - max(0.0, (q - 0.8) / 0.2))
                if sc > 0.02:
                    self.push_sprite(self.pop_spr[i], top + [0, 1.45 + 1.3 * ease_out(q), 0], 0.48 * sc, anchor=0.5, layer=2, cache=True)
            if age < 1.0:
                q = age / 1.0
                p0 = top + np.array([0, 1.0, 0])
                p2 = board + np.array([self.coin_off[i, 0] * 1.0, self.coin_off[i, 1] * 0.3, 0.25])
                p1 = (p0 + p2) / 2 + np.array([0, 3.0, 1.0])
                qq = q ** 1.25
                P = (1 - qq) ** 2 * p0 + 2 * (1 - qq) * qq * p1 + qq ** 2 * p2
                spin = abs(math.cos(age * 11 + i))
                self.push_sprite(self.coin, P, 0.42, max(0.08, spin), 1.0, 0, anchor=0.5, layer=2)
                x_, y_, _, inv_ = R.project(P[None, :].astype(np.float32))
                R.points(x_, y_, np.array([[255, 210, 90]], np.float32) * 12, 2.0 * self.k, bloom=0.8)
            arr_age = age - 1.0 if age >= 1.0 else age - 1.0 + P_
            bump = max(bump, math.exp(-max(0.0, arr_age) / 0.18))
            cash += self.amt[i] * max(0, math.floor((t - 1.0) / P_ + self.pop_ph[i]))
        # upgrade arrows
        for j_, i in enumerate(self.arrow_on):
            x, z = self.ped[i]
            pul = 1 + 0.12 * math.sin(t * 6 + j_)
            yb = 2.45 + 0.25 * math.sin(t * 3.2 + j_ * 2)
            res = self.push_sprite(self.arrow, (x + 0.75, yb, z), 0.75 * pul, anchor=0.5, layer=1, bias=-1.0, cache=True)
            if res is not None:
                R.soft(np.array([res[0]]), np.array([res[1]]), np.array([[80, 255, 120]], np.float32) * 900 * pul)
        # ---- counter board
        bs = 1 + 0.06 * bump
        bw, bh = 3.1 * bs, 1.05 * bs
        posts = [(box_corners(board + [sx_, -2.4, -0.2], (0.12, 2.2, 0.12)), (120, 120, 140)) for sx_ in (-2.4, 2.4)]
        posts.append((box_corners(board + [0, 0, -0.18], (bw + 0.15, bh + 0.15, 0.12)), (255, 196, 60)))
        self.push_boxes(posts, haze=haze)
        tex = self.counter_tex(cash, bump)
        P4 = [board + [-bw, bh, 0], board + [bw, bh, 0], board + [bw, -bh, 0], board + [-bw, -bh, 0]]
        zb = float((board - self.cpos) @ self.R.cam.R[2])
        self.items.append((1, -(zb - 0.3), 2, lambda cv, dd: self.warp(cv, tex, P4)))
        bx, by, _, binv = R.project(board[None, :].astype(np.float32))
        R.soft(bx, by, np.array([[255, 210, 90]], np.float32) * (1500 + 4000 * bump))
        # ---- treadmill + runner
        tz = 2.6
        tm = [(box_corners((0, 0.2, tz), (2.0, 0.2, 0.62)), (70, 74, 96)),
              (box_corners((0, 0.405, tz), (1.85, 0.01, 0.5)), (40, 42, 54)),
              (box_corners((0, 1.0, tz - 0.62), (1.7, 0.04, 0.04)), (200, 200, 215)),
              (box_corners((1.75, 0.95, tz - 0.62), (0.04, 0.55, 0.04)), (200, 200, 215)),
              (box_corners((-1.75, 0.95, tz - 0.62), (0.04, 0.55, 0.04)), (200, 200, 215)),
              (box_corners((2.2, 0.75, tz), (0.1, 0.75, 0.1)), (200, 200, 215)),
              (box_corners((2.2, 1.6, tz), (0.18, 0.28, 0.42), rot_z(0.5)), (60, 64, 90))]
        self.push_boxes(tm)
        v = 3.0 + 2.0 * self.inc
        st = []
        for kk in range(16):
            xx = ((kk * 0.25 - t * v) % 4.0) - 2.0
            if -1.82 < xx < 1.75:
                st.append([(xx, 0.418, tz - 0.48), (xx + 0.07, 0.418, tz - 0.48), (xx + 0.07, 0.418, tz + 0.48), (xx, 0.418, tz + 0.48)])
        if st:
            self.push_faces(np.array(st), (150, 156, 180), ref=np.array([0, -5, tz]), bias=-0.05, amb=0.9, diff=0.1)
        # console meter: bars that fill and reset
        lvl = (t * 0.6) % 1.0
        bars = []
        for kk in range(6):
            on = kk / 6 < lvl
            col = lerp_col((80, 255, 120), (255, 80, 80), kk / 5) if on else np.array([50, 54, 80], np.float32)
            y0 = 1.42 + kk * 0.07
            bars.append((box_corners((2.2 - 0.14 * math.sin(0.5), y0 + 0.14 * math.cos(0.5) * 0, tz), (0.02, 0.025, 0.3), rot_z(0.5)), col))
        self.push_boxes(bars, bias=-0.1)
        rp = np.array([-0.3, 0.42, tz])
        parts = char_parts(rp, math.pi / 2, t, 2.7, 0.82, "run", lean=0.22,
                           cols={"shirt": tuple(a), "pants": (60, 70, 150), "skin": (250, 212, 170), "hat": (255, 90, 110)})
        self.push_boxes(parts, bias=-0.2)
        # "+1" speed pops, sweat, speed lines
        age = (t % 0.9)
        q = age / 0.9
        self.push_sprite(self.plus1, rp + [0.2, 2.25 + 0.9 * ease_out(q), 0], 0.42 * back_out(min(1.0, age / 0.2)) * (1 - max(0.0, (q - 0.8) / 0.2)),
                         anchor=0.5, layer=2, cache=True)
        for j in range(len(self.sweat)):
            sq = (t * 1.6 + self.sweat[j, 0]) % 1.0
            P = rp + np.array([-0.3 - 1.1 * sq, 1.95 + 0.5 * sq - 1.4 * sq * sq, (self.sweat[j, 1] - 0.5) * 0.5])
            x_, y_, z_, inv_ = R.project(P[None, :].astype(np.float32))
            r_ = 0.06 * float(inv_[0])
            e = [float(x_[0]) - r_, float(y_[0]) - r_ * 1.3, float(x_[0]) + r_, float(y_[0]) + r_]
            self.items.append((2, 0.0, 2, (lambda e_: (lambda cv, dd: dd.ellipse(e_, fill=(140, 210, 255), outline=(255, 255, 255))))(e)))
        A = np.array([[-0.9, 0.7 + 0.25 * j, tz + 0.3] for j in range(5)], np.float32)
        A[:, 0] -= ((t * 4 + np.arange(5) * 0.37) % 1.0) * 0.8
        B = A - np.array([1.1, 0, 0], np.float32)
        R.lines3d(A, B, np.array([255, 255, 255], np.float32) * 2.2, np.zeros(3, np.float32), width=3.0, bloom=0.3)
        self.flush(canvas, d)
        # ambient sparkles around the legendary pedestal + coin glints
        x4, z4 = self.ped[4]
        ang = t * 1.4 + np.arange(6) * TAU / 6
        O = np.stack([x4 + np.cos(ang) * 0.95, 1.2 + 0.4 * np.sin(ang * 2 + t), z4 + np.sin(ang) * 0.95], 1).astype(np.float32)
        ox, oy, _, oinv = R.project(O)
        tw = 0.6 + 0.4 * np.sin(t * 6 + np.arange(6))
        sparkle(R, ox, oy, 0.1 * oinv * tw, np.array([[255, 220, 120]], np.float32) * 1.5 * tw[:, None])
        return self.fin(canvas, self._soft, canvas_bloom=0.08, bloom=(0.5, 0.45), radii=(1.4, 8.0))


KINDS_EGGS = {
    "egg_nest": EggNest, "egg_island": EggNest, "eggs": EggNest, "egg_nests": EggNest, "egg_biome": EggNest,
    "egg_heist": EggHeist, "egg_chase": EggHeist, "heist": EggHeist, "steal_egg": EggHeist, "chase": EggHeist,
    "egg_hatch": EggHatch, "hatch": EggHatch, "hatching": EggHatch, "egg_hatching": EggHatch, "pet_reveal": EggHatch,
    "pet_base": PetBase, "pet_tycoon": PetBase, "tycoon": PetBase, "pets": PetBase, "pet_income": PetBase,
    "treadmill": PetBase, "speed_training": PetBase,
}

_procedural.KINDS.update(KINDS_EGGS)
