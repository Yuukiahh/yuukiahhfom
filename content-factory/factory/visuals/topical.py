"""Topical (news / business / tech / politics) 3D backgrounds.

Each scene is real geometry seen through a moving perspective :class:`~.three_d.Camera`
(true parallax), rendered with the small numpy renderer from :mod:`.three_d`:

* opaque surfaces are flat-shaded polygons drawn with PIL in painter's order onto a
  per-frame *canvas* (cheap, ~µs per polygon);
* light (LEDs, glowing traces, windows, bokeh, sweeps) is additive splatting with bloom;
* haze / fog / big glows go straight into the renderer's quarter-res soft buffer.

All randomness is drawn in ``setup()``; ``frame(t)`` is a pure function of ``t``.
Every scene takes ``accent`` ("#RRGGBB") as its main tint and, where meaningful,
``trend`` ("up" / "down" / "volatile").  No real brand names, logos or readable words
(except day numbers on the calendar cards and generic random ticker symbols).
Registered into ``procedural.KINDS`` at the bottom of this module.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .three_d import TAU, Camera, Scene3D, _norm, lerp_col, orbit, periodic_noise, rot_x, rot_y, rot_z, smoothstep

FONT_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"


# ======================================================================== helpers
def hex_rgb(value, default="#3FA9FF") -> np.ndarray:
    """'#76B900' / '76b900' / (r, g, b) -> float32 (3,)."""
    v = default if value is None else value
    if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 3:
        return np.asarray(v, np.float32)
    s = str(v).strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    try:
        return np.array([int(s[i:i + 2], 16) for i in (0, 2, 4)], np.float32)
    except ValueError:
        return hex_rgb(default)


def font(size: int, name: str = "InterDisplay-Black.otf"):
    for cand in (FONT_DIR / name, FONT_DIR / "Inter-Bold.otf"):
        try:
            return ImageFont.truetype(str(cand), size)
        except OSError:
            continue
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default(size)


def tint(c) -> tuple:
    return tuple(int(v) for v in np.clip(c, 0, 255))


def vgrad(W, H, stops) -> np.ndarray:
    """Vertical gradient (H, W, 3) uint8 from [(pos, rgb), ...]."""
    yy = np.linspace(0, 1, H, dtype=np.float32)
    col = np.stack([np.interp(yy, [s[0] for s in stops], [s[1][c] for s in stops]) for c in range(3)], 1)
    return np.ascontiguousarray(np.broadcast_to(np.clip(col, 0, 255).astype(np.uint8)[:, None, :], (H, W, 3)))


# --- 5x7 LED font (ticker symbols / numbers); '#' = lit
_GLYPHS = {
    "A": ".###.#...##...########...##...##...#", "B": "####.#...#####.#...##...##...#####.",
    "C": ".####.#....#....#....#....#.....####", "D": "####.#...##...##...##...##...#####.",
    "E": "######....####.#....#....#....#####", "F": "######....####.#....#....#....#....",
    "G": ".####.#....#..###...##...##...#.####", "H": "#...##...########...##...##...##...#",
    "I": "#####..#....#....#....#....#..#####", "J": "..###...#....#....#....##..#.##...",
    "K": "#...##..#.###..#.#..#..#.#...##...#", "L": "#....#....#....#....#....#....#####",
    "M": "#...####.###.#.##...##...##...##...#", "N": "#...###..##.#.##..###...##...##...#",
    "O": ".###.#...##...##...##...##...#.###.", "P": "####.#...##...#####.#....#....#....",
    "Q": ".###.#...##...##...##.#.##..#..##.#", "R": "####.#...##...#####.#..#.#...##...#",
    "S": ".#####.....###......#....#....#####.", "T": "#####..#....#....#....#....#....#..",
    "U": "#...##...##...##...##...##...#.###.", "V": "#...##...##...##...#.#.#..#.#...#..",
    "W": "#...##...##...##.#.##.#.###.###...#", "X": "#...#.#.#...#...#.#.#...##...##...#",
    "Y": "#...#.#.#...#....#....#....#....#..", "Z": "#####....#...#...#...#...#....#####",
    "0": ".###.#..###.#.###..##...##...#.###.", "1": "..#..##....#....#....#....#..#####.",
    "2": ".###.#...#....#...#...#...#...#####", "3": "####.....#....#.###.....#....#####.",
    "4": "...#...##..#.#.#..#.#####...#....#.", "5": "######....####.....#....#....#####.",
    "6": ".###.#....#....####.#...##...#.###.", "7": "#####....#...#...#...#....#....#...",
    "8": ".###.#...##...#.###.#...##...#.###.", "9": ".###.#...##...#.####....#....#.###.",
    ".": "...............................##..", "%": "##...##..#...#...#...#...#..##...##",
    "+": ".......#....#..#####..#....#.......", "-": "...............#####...............",
    "^": "..#....#...###..###.#####.........", "v": ".........#####.###..###...#....#...",
    " ": "." * 35,
}
GLYPH = {k: np.array([c == "#" for c in (v + "." * 35)[:35]], bool).reshape(7, 5) for k, v in _GLYPHS.items()}


def led_text(s: str) -> np.ndarray:
    """String -> bool bitmap (7, 6*len) using the 5x7 font (1 px spacing)."""
    cols = []
    for ch in s:
        g = GLYPH.get(ch, GLYPH[" "])
        cols.append(np.concatenate([g, np.zeros((7, 1), bool)], 1))
    return np.concatenate(cols, 1) if cols else np.zeros((7, 0), bool)


# --- coarse world map (lon, lat) polygons, generic and low-res on purpose (dot-matrix look)
_LAND = [
    # North America
    [(-168, 66), (-162, 70), (-140, 70), (-125, 70), (-95, 72), (-82, 72), (-66, 62), (-60, 55), (-56, 52), (-66, 45),
     (-70, 42), (-76, 35), (-81, 31), (-80, 25), (-82, 27), (-84, 30), (-90, 29), (-97, 27), (-97, 22), (-92, 18),
     (-87, 21), (-88, 16), (-83, 10), (-78, 7), (-80, 8), (-83, 8), (-86, 12), (-92, 14), (-96, 16), (-105, 20),
     (-110, 24), (-112, 29), (-115, 30), (-117, 33), (-120, 35), (-124, 40), (-124, 48), (-130, 54), (-137, 58),
     (-150, 60), (-158, 57), (-165, 55), (-160, 59), (-165, 62)],
    [(-120, 72), (-95, 74), (-80, 74), (-75, 80), (-90, 82), (-110, 78), (-120, 75)],
    [(-80, 63), (-62, 66), (-70, 72), (-82, 72)],
    [(-73, 78), (-60, 82), (-30, 83), (-20, 80), (-20, 70), (-30, 68), (-42, 60), (-50, 64), (-55, 70), (-60, 76)],
    [(-85, 22), (-74, 20), (-77, 20)],
    # South America
    [(-80, 9), (-72, 12), (-62, 11), (-52, 5), (-50, 0), (-44, -2), (-35, -5), (-35, -9), (-39, -15), (-41, -22),
     (-48, -26), (-53, -34), (-58, -38), (-63, -41), (-65, -46), (-68, -51), (-70, -55), (-74, -52), (-75, -46),
     (-73, -38), (-71, -30), (-70, -18), (-76, -14), (-81, -6), (-80, -1), (-77, 4), (-78, 7)],
    # Eurasia
    [(-9, 37), (-9, 43), (-1.5, 43.5), (-1, 46), (-4.5, 48), (-1, 49.5), (2, 51), (4, 52), (8, 54), (8, 57), (10, 59),
     (6, 58), (5, 62), (14, 67), (20, 70), (28, 71), (33, 69), (41, 67), (44, 68), (55, 69), (60, 70), (70, 73),
     (80, 73), (100, 77), (112, 74), (130, 72), (140, 72), (160, 70), (180, 69), (180, 65), (170, 60), (163, 58),
     (156, 51), (155, 57), (143, 59), (137, 54), (141, 48), (135, 43), (130, 42), (129, 35), (126, 35), (125, 38),
     (122, 40), (119, 39), (121, 37), (119, 35), (122, 31), (121, 28), (117, 23), (108, 21), (106, 17), (109, 12),
     (105, 9), (100, 13), (100, 8), (103, 1.5), (98, 8), (98, 16), (94, 17), (92, 22), (88, 22), (80, 15), (80, 10),
     (77, 8), (73, 16), (72, 21), (68, 23), (57, 25), (52, 27), (48, 30), (50, 26), (56, 24), (59, 22), (55, 17),
     (45, 13), (43, 13), (39, 21), (35, 28), (33, 30), (34, 31), (36, 35), (36, 37), (30, 36), (26, 40), (23, 40),
     (22, 37), (20, 40), (19, 42), (13, 45), (16, 41), (18, 40), (16, 38), (12, 42), (10, 44), (7, 44), (3, 43),
     (0, 39), (-2, 37), (-5, 36)],
    [(-5, 50), (1, 51), (2, 53), (0, 54), (-2, 56), (-2, 58), (-5, 58.5), (-6, 56), (-5, 54), (-3, 54), (-4.5, 52)],
    [(-10, 52), (-6, 52), (-6, 55), (-8, 55), (-10, 54)],
    [(-24, 64), (-14, 64), (-14, 66), (-22, 66.5)],
    # Africa
    [(-17, 21), (-17, 15), (-15, 11), (-8, 4.5), (-2, 5), (5, 6), (9, 4), (9, 1), (13, -5), (12, -17), (15, -27),
     (18, -34), (20, -35), (26, -34), (32, -29), (35, -24), (35, -19), (40, -15), (40, -10), (39, -5), (43, 0),
     (51, 10), (51, 12), (44, 11), (43, 13), (39, 17), (37, 22), (35, 27), (32, 31), (25, 32), (20, 31), (20, 33),
     (10, 34), (10, 37), (3, 37), (-6, 36), (-10, 30), (-13, 27)],
    [(44, -25), (47, -25), (50, -15), (49, -12), (44, -16)],
    # Asia-Pacific islands
    [(130, 31), (132, 34), (136, 34), (140, 35), (141, 38), (142, 40), (141, 41.5), (140, 40), (139, 38), (137, 37),
     (133, 35.5), (130, 33)],
    [(140, 42), (145, 43), (145, 44), (142, 45.5), (140, 43)],
    [(95, 5), (98, 4), (104, -1), (106, -6), (102, -4), (97, 1)],
    [(109, 1), (111, -3), (116, -4), (119, 1), (117, 7), (115, 5), (110, 2)],
    [(105, -6), (114, -7), (114, -8.5), (106, -7)],
    [(131, -1), (138, -2), (147, -6), (150, -10), (143, -9), (138, -8), (132, -4)],
    [(120, 18), (122, 18), (124, 12), (126, 7), (122, 7), (120, 14)],
    [(119, -1), (125, 1), (122, -3), (121, -5)],
    # Australia / NZ
    [(114, -22), (114, -34), (118, -35), (124, -34), (131, -31), (135, -35), (138, -35), (140, -38), (146, -39),
     (150, -37), (153, -30), (153, -25), (146, -19), (145, -15), (142, -11), (141, -17), (136, -12), (130, -12),
     (126, -14), (122, -18)],
    [(145, -41), (148, -41), (147, -43.5)],
    [(172, -35), (178, -38), (175, -41), (171, -46), (167, -46), (172, -41)],
]
_WATER = [
    [(-95, 58), (-82, 55), (-78, 60), (-80, 64), (-88, 64), (-94, 60)],  # Hudson Bay
    [(28, 41), (31, 46.5), (37, 47), (41, 42), (36, 41.5)],  # Black Sea
    [(47, 45), (51, 47), (53, 41), (53, 37), (49, 38), (47, 43)],  # Caspian
    [(10, 54), (14, 54), (20, 55), (22, 60), (25, 65), (21, 65), (17, 61), (17, 57), (12, 56)],  # Baltic
]
LAT_TOP, LAT_BOT = 80.0, -58.0


def world_mask(w: int, h: int) -> np.ndarray:
    """Equirectangular land mask (h, w) bool, lon -180..180, lat LAT_TOP..LAT_BOT."""
    ss = 4
    img = Image.new("L", (w * ss, h * ss), 0)
    d = ImageDraw.Draw(img)

    def xy(poly):
        return [((lo + 180) / 360 * w * ss, (LAT_TOP - la) / (LAT_TOP - LAT_BOT) * h * ss) for lo, la in poly]

    for p in _LAND:
        d.polygon(xy(p), fill=255)
    for p in _WATER:
        d.polygon(xy(p), fill=0)
    return np.asarray(img.reduce(ss), np.float32) > 100


def lonlat_uv(lon, lat):
    return (np.asarray(lon) + 180) / 360, (LAT_TOP - np.asarray(lat)) / (LAT_TOP - LAT_BOT)


CITIES = [(-74, 40.7), (-118, 34), (-87.6, 41.9), (-46.6, -23.5), (-99, 19.4), (-0.1, 51.5), (2.3, 48.9), (13.4, 52.5),
          (37.6, 55.8), (28.9, 41), (31.2, 30), (55.3, 25.2), (77.2, 28.6), (72.9, 19), (116.4, 39.9), (121.5, 31.2),
          (139.7, 35.7), (126.9, 37.6), (103.8, 1.35), (151.2, -33.9), (18.4, -33.9), (3.4, 6.5), (36.8, -1.3),
          (-79.4, 43.7), (-122.4, 37.8), (114.2, 22.3), (100.5, 13.8), (-58.4, -34.6), (-77, -12), (44.4, 33.3)]


# ======================================================================== base class
class Topical(Scene3D):
    default_accent = "#3FA9FF"

    def setup(self):
        self.acc = hex_rgb(self.p.get("accent"), self.default_accent)
        tr = str(self.p.get("trend", "up")).lower()
        self.trend = "down" if tr in ("down", "bear", "bearish", "fall", "falling", "negative", "loss") else (
            "volatile" if tr in ("volatile", "mixed", "flat", "sideways", "choppy") else "up")
        self.init()

    def init(self):
        pass

    # ---------------------------------------------------------------- polygon helpers
    def to_cam(self, P):
        cam = self.R.cam
        return (np.asarray(P, np.float64) - cam.pos) @ cam.R.T

    def screen_poly(self, V, near=0.05):
        """Camera-space polygon (n,3) -> list of screen (x, y) after near clipping, or None."""
        W, H, f = self.R.W, self.R.H, self.R.f
        if (V[:, 2] >= near).all():
            return list(zip((W * 0.5 + V[:, 0] * f / V[:, 2]).tolist(), (H * 0.5 - V[:, 1] * f / V[:, 2]).tolist()))
        out = []
        n = len(V)
        for i in range(n):
            a, b = V[i], V[(i + 1) % n]
            ina, inb = a[2] >= near, b[2] >= near
            if ina:
                out.append(a)
            if ina != inb:
                out.append(a + (b - a) * ((near - a[2]) / (b[2] - a[2])))
        if len(out) < 3:
            return None
        return [(W * 0.5 + p[0] * f / p[2], H * 0.5 - p[1] * f / p[2]) for p in out]

    def quads(self, Q):
        """World quads (n, k, 3) -> per quad screen polygons (None when fully behind) + mean depth."""
        n, k, _ = Q.shape
        V = self.to_cam(Q.reshape(-1, 3)).reshape(n, k, 3)
        W, H, f = self.R.W, self.R.H, self.R.f
        z = np.maximum(V[..., 2], 1e-6)
        SX = (W * 0.5 + V[..., 0] * f / z).tolist()
        SY = (H * 0.5 - V[..., 1] * f / z).tolist()
        ok = (V[..., 2] >= 0.05)
        allok, anyok = ok.all(1), ok.any(1)
        polys = []
        for i in range(n):
            if allok[i]:
                polys.append(list(zip(SX[i], SY[i])))
            elif anyok[i]:
                polys.append(self.screen_poly(V[i]))
            else:
                polys.append(None)
        return polys, V[..., 2].mean(1)

    def box_faces(self, b, cam_pos):
        """Box b = (x0, x1, y0, y1, z0, z1) -> list of (quad (4,3), normal) for faces visible from cam_pos."""
        x0, x1, y0, y1, z0, z1 = b
        cx, cy, cz = cam_pos
        out = []
        if cy > y1:
            out.append(([(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)], (0, 1, 0)))
        if cy < y0:
            out.append(([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], (0, -1, 0)))
        if cz < z0:
            out.append(([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)], (0, 0, -1)))
        if cz > z1:
            out.append(([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], (0, 0, 1)))
        if cx < x0:
            out.append(([(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)], (-1, 0, 0)))
        if cx > x1:
            out.append(([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], (1, 0, 0)))
        return out

    @staticmethod
    def shade(base, normal, light, view, amb=0.35, diff=0.75, spec=0.0, power=24.0, spec_col=(255, 255, 255)):
        n = np.asarray(normal, np.float64)
        l = _norm(light)
        v = _norm(view)
        h = _norm(l + v)
        d = max(0.0, float(n @ l))
        s = max(0.0, float(n @ h)) ** power
        return np.asarray(base, np.float32) * (amb + diff * d) + np.asarray(spec_col, np.float32) * spec * s

    def fin(self, canvas=None, soft_extra=None, canvas_bloom=0.15, base=None, **kw):
        """R.finish with an opaque PIL canvas whose bloom contribution is controllable (the default
        canvas path of Renderer.finish blooms flat surfaces strongly, which washes out contrast)."""
        R = self.R
        if canvas is not None:
            arr = np.asarray(canvas, np.float32)
            if canvas_bloom > 0:
                small = arr[::R.ds, ::R.ds] * canvas_bloom
                soft_extra = small if soft_extra is None else soft_extra + small
            base = arr if base is None else base + arr
        if soft_extra is not None:
            soft_extra = soft_extra[:R.Hs, :R.Ws]
            if soft_extra.shape[:2] != (R.Hs, R.Ws):
                pad = np.zeros((R.Hs, R.Ws, 3), np.float32)
                pad[:soft_extra.shape[0], :soft_extra.shape[1]] = soft_extra
                soft_extra = pad
        return R.finish(base=base, soft_extra=soft_extra, **kw)

    def tilt_shift(self, canvas: Image.Image, focus_y: float, sharp: float, ramp: float, radius: float = 3.0):
        """Fake shallow depth of field on the opaque canvas: blur grows with |y - focus_y| (screen px)."""
        W, H = canvas.size
        small = canvas.reduce(4).filter(ImageFilter.GaussianBlur(radius * self.k))
        blur = small.resize((W, H), Image.BILINEAR)
        key = (round(focus_y / 8), round(sharp), round(ramp))
        if getattr(self, "_ts_key", None) != key:
            yy = np.arange(H, dtype=np.float32)
            m = np.clip((np.abs(yy - focus_y) - sharp) / ramp, 0, 1) ** 1.2 * 255
            self._ts_mask = Image.fromarray(np.broadcast_to(m.astype(np.uint8)[:, None], (H, W)).copy(), "L")
            self._ts_key = key
        return Image.composite(blur, canvas, self._ts_mask)


# ======================================================================== 1. GPU chip
class GPUChip(Topical):
    """Close-up of a generic GPU package: substrate with gold contacts and SMD parts, metal
    stiffener frame, memory stacks, big die with a grid of shader tiles lit by travelling light
    waves, glowing traces with data pulses; slow orbit + out-of-focus board (depth of field)."""
    default_accent = "#76B900"

    def init(self):
        rng = self.rng
        a = self.acc
        self.S = 2.7  # substrate half size
        self.die = (-1.0, 1.0, 0.0, 0.2, -0.85, 0.85)
        # shader tiles: two blocks of 8x3 GPC tiles with a cache band in the middle
        tiles = []
        nx = 8
        for blk, (zs, ze) in enumerate(((-0.79, -0.12), (0.12, 0.79))):
            for i in range(nx):
                for j in range(3):
                    x0 = -0.94 + i * 1.88 / nx
                    z0 = zs + j * (ze - zs) / 3
                    tiles.append((x0 + 0.014, x0 + 1.88 / nx - 0.014, z0 + 0.014, z0 + (ze - zs) / 3 - 0.014))
        self.tiles = np.array(tiles, np.float32)
        tc = np.stack([(self.tiles[:, 0] + self.tiles[:, 1]) / 2, (self.tiles[:, 2] + self.tiles[:, 3]) / 2], 1)
        self.tile_c = tc
        self.tile_r = np.sqrt(tc[:, 0] ** 2 + (tc[:, 1] * 1.15) ** 2)
        self.tile_ph = rng.random(len(tiles)).astype(np.float32)
        # core dots inside tiles (additive light): 4x3 per tile
        cx, cz = [], []
        for (x0, x1, z0, z1) in tiles:
            for u in (0.2, 0.4, 0.6, 0.8):
                for v in (0.25, 0.5, 0.75):
                    cx.append(x0 + (x1 - x0) * u)
                    cz.append(z0 + (z1 - z0) * v)
        self.core = np.stack([cx, np.full(len(cx), 0.203), cz], 1).astype(np.float32)
        self.core_tile = np.repeat(np.arange(len(tiles)), 12)
        # cache band: fine stripes (lines) across the middle of the die
        zz = np.linspace(-0.1, 0.1, 9)
        self.cacheA = np.stack([np.full(9, -0.94), np.full(9, 0.202), zz], 1).astype(np.float32)
        self.cacheB = np.stack([np.full(9, 0.94), np.full(9, 0.202), zz], 1).astype(np.float32)
        # memory stacks left/right of the die
        self.hbm = []
        for sx in (-1, 1):
            for zc in (-0.6, 0.0, 0.6):
                x0 = 1.15 if sx > 0 else -1.6
                self.hbm.append((x0, x0 + 0.45, 0.0, 0.18, zc - 0.26, zc + 0.26))
        fo, fi, fh = 2.15, 1.92, 0.07
        self.frame_bars = [(-fo, fo, 0, fh, -fo, -fi), (-fo, fo, 0, fh, fi, fo), (-fo, -fi, 0, fh, -fi, fi), (fi, fo, 0, fh, -fi, fi)]
        # gold pads: two rows on each side
        pads = []
        for side in range(4):
            for row in range(2):
                for k in range(30):
                    pads.append((side, -2.35 + k * 4.7 / 29 + row * 0.08, 2.36 + row * 0.15))
        self.pads = pads
        caps = []
        for k in range(14):
            caps.append((-1.65 + k * 3.3 / 13, 1.72))
            caps.append((-1.65 + k * 3.3 / 13, -1.72))
        self.caps = caps

        def place(side, u, v, y):
            return [(u, y, v), (u, y, -v), (v, y, u), (-v, y, u)][side]
        self.place = place
        # substrate traces: die/HBM edge -> stiffener and stiffener -> pads (canvas lines) + pulse points
        tA, tB = [], []
        for side in range(4):
            lim = 0.92 if side < 2 else 0.8
            start = 0.9 if side < 2 else 1.65
            for k in range(22):
                u = -lim + k * 2 * lim / 21
                tA.append(place(side, u, start, 0.003))
                tB.append(place(side, u * 1.9 if side < 2 else u * 2.0, 1.88, 0.003))
            for k in range(30):
                u = -2.35 + k * 4.7 / 29
                tA.append(place(side, u * 0.93, 2.18, 0.003))
                tB.append(place(side, u, 2.34, 0.003))
        self.tA, self.tB = np.array(tA, np.float32), np.array(tB, np.float32)
        self.t_ph = rng.random(len(tA)).astype(np.float32)
        self.t_on = rng.random(len(tA)) < 0.45
        # board-level manhattan routes (out of focus, drawn into the soft layer)
        routes = []
        for k in range(70):
            side = int(rng.integers(0, 4))
            u = rng.uniform(-2.6, 2.6)
            pts = [place(side, u, 2.8, -0.35)]
            x, _, z = pts[0]
            dvec = np.array(place(side, 0, 1, 0))[[0, 2]]
            for seg in range(int(rng.integers(2, 5))):
                ln = rng.uniform(0.8, 3.5)
                x, z = x + dvec[0] * ln, z + dvec[1] * ln
                pts.append((x, -0.35, z))
                ang = rng.choice([-1, 1]) * math.pi / 4
                c, s_ = math.cos(ang), math.sin(ang)
                dvec = np.array([dvec[0] * c - dvec[1] * s_, dvec[0] * s_ + dvec[1] * c])
            routes.append(np.array(pts, np.float32))
        self.routes = routes
        self.r_ph = rng.random(len(routes)).astype(np.float32)
        vias = rng.uniform(-9, 9, (500, 2))
        vias = vias[np.maximum(np.abs(vias[:, 0]), np.abs(vias[:, 1])) > 2.9]
        self.vias = np.stack([vias[:, 0], np.full(len(vias), -0.35), vias[:, 1]], 1).astype(np.float32)
        nb = 60
        self.bok = np.stack([rng.uniform(-7, 7, nb), rng.uniform(0.6, 4.5, nb), rng.uniform(-7, 7, nb)], 1).astype(np.float32)
        self.bok_c = lerp_col(a, (255, 220, 160), rng.random(nb) ** 2) * (0.3 + rng.random(nb))[:, None]
        # top edges (subdivided) used for moving metallic glints
        ea, eb = [], []

        def box_top_edges(b, n=12):
            x0, x1, _, y1, z0, z1 = b
            c = [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)]
            for k in range(4):
                A_, B_ = np.array(c[k]), np.array(c[(k + 1) % 4])
                L = max(2, int(np.linalg.norm(B_ - A_) * n))
                s_ = np.linspace(0, 1, L + 1)[:, None]
                P_ = A_ + (B_ - A_) * s_
                ea.append(P_[:-1])
                eb.append(P_[1:])
        box_top_edges((-2.15, 2.15, 0, 0.0701, -2.15, 2.15))
        box_top_edges((-1.92, 1.92, 0, 0.0701, -1.92, 1.92))
        box_top_edges((-1.0, 1.0, 0, 0.2005, -0.85, 0.85))
        for b in self.hbm:
            box_top_edges((b[0], b[1], 0, 0.1805, b[4], b[5]), 6)
        self.edges_A = np.concatenate(ea).astype(np.float32)
        self.edges_B = np.concatenate(eb).astype(np.float32)
        self.small_bg = Image.fromarray(vgrad(self.R.Ws, self.R.Hs, [(0, (10, 18, 16)), (0.45, (20, 34, 28)), (1, (8, 12, 10))]))

    def frame(self, t):
        R, a = self.R, self.acc
        azim = 0.6 + t * 0.07
        elev = 0.9 + 0.06 * math.sin(t * 0.21)
        dist = 9.6 - 0.6 * math.sin(t * 0.13)
        cpos = orbit((0, 0.0, 0), dist, azim, elev)
        R.begin(Camera(cpos, (0.0, -0.2, 0.0), fov=42, roll=0.04 * math.sin(t * 0.17)))
        light = np.array([math.cos(t * 0.25) * 0.7 - 0.3, 1.0, math.sin(t * 0.25) * 0.7 - 0.4])
        # ---- out-of-focus board, quarter resolution -> soft buffer (blurred + upsampled for free)
        small = self.small_bg.copy()
        ds = R.ds
        sd = ImageDraw.Draw(small)
        for i, pts in enumerate(self.routes):
            x, y, z, _ = R.project(pts)
            if (z < 0.1).any():
                continue
            sd.line(list(zip((x / ds).tolist(), (y / ds).tolist())), fill=tint(a * 0.35 + 6), width=1)
        S = self.S
        sub = np.array([(-S, -0.12, -S), (S, -0.12, -S), (S, -0.12, S), (-S, -0.12, S)], np.float32)
        x, y, _, _ = R.project(sub)
        sd.polygon(list(zip((x / ds).tolist(), (y / ds).tolist())), fill=(0, 0, 0))
        soft_bg = np.asarray(small, np.float32) * 0.75
        x, y, _, _ = R.project(self.vias)
        R.soft(x, y, np.array([60, 52, 30], np.float32) * 0.8)
        for i, pts in enumerate(self.routes):  # pulses travelling on the board routes
            seg = len(pts) - 1
            s = ((t * 0.22 + self.r_ph[i]) % 1.0) * seg
            j = min(int(s), seg - 1)
            p = pts[j] + (pts[j + 1] - pts[j]) * (s - j)
            R.points3d(p[None, :], (a * 0.5 + 110)[None, :] * 7, 0.0, 3.5, bloom=1.0)
        # ---- package, sharp, painter's order
        canvas = Image.new("RGB", (self.W, self.H))
        d = ImageDraw.Draw(canvas)
        polys = []

        def add_box(b, base, layer, spec=0.0, power=24.0, top=None, amb=0.35, diff=0.75, spec_col=(255, 255, 255)):
            for quad, n in self.box_faces(b, cpos):
                ctr = np.mean(quad, 0)
                c = self.shade(top if (top is not None and n[1] > 0) else base, n, light, cpos - ctr, amb, diff, spec, power, spec_col)
                polys.append((layer, -float(np.linalg.norm(cpos - ctr)), np.array(quad, np.float32), tint(c)))

        add_box((-S, S, -0.12, 0.0, -S, S), (12, 30, 27), 1, spec=0.06, power=8, top=(15, 38, 33))
        for side, u, v in self.pads:
            w2, l2 = 0.032, 0.05
            q = [self.place(side, u - w2, v - l2, 0.002), self.place(side, u + w2, v - l2, 0.002),
                 self.place(side, u + w2, v + l2, 0.002), self.place(side, u - w2, v + l2, 0.002)]
            polys.append((2, 0, np.array(q, np.float32), None))
        for (u, v) in self.caps:
            add_box((u - 0.07, u + 0.07, 0, 0.06, v - 0.04, v + 0.04), (110, 80, 55), 4, spec=0.4)
        for b in self.frame_bars:
            add_box(b, (52, 56, 64), 5, spec=0.55, power=22, amb=0.4, diff=0.6, spec_col=(230, 238, 255))
        for b in self.hbm:
            add_box(b, (24, 26, 32), 6, spec=0.55, power=50, top=(28, 30, 38))
        add_box(self.die, (22, 24, 32), 7, spec=0.7, power=70, top=(10, 12, 18))
        quads = np.stack([p[2] for p in polys])
        scr, _ = self.quads(quads)
        order = sorted(range(len(polys)), key=lambda i: (polys[i][0], polys[i][1]))
        gold_ph = t * 0.8
        for i in order:
            pts = scr[i]
            if pts is None:
                continue
            fill = polys[i][3]
            if fill is None:  # gold pad with a glint sweeping across as the light moves
                ctr = polys[i][2].mean(0)
                g = 0.6 + 0.6 * max(0.0, math.sin(ctr[0] * 0.9 + ctr[2] * 0.7 + gold_ph)) ** 6
                fill = tint(np.array([190, 140, 55]) * g + 20)
            d.polygon(pts, fill=fill)
        # substrate traces, then the tall parts are redrawn over them (cheap: few polygons)
        xa, ya, _, _ = R.project(self.tA)
        xb, yb, _, _ = R.project(self.tB)
        tc = tint(a * 0.42 + 8)
        for x0, y0, x1, y1 in zip(xa.tolist(), ya.tolist(), xb.tolist(), yb.tolist()):
            d.line([(x0, y0), (x1, y1)], fill=tc, width=max(1, int(1.6 * self.k)))
        for i in order:
            if polys[i][0] >= 4 and scr[i] is not None:
                d.polygon(scr[i], fill=polys[i][3])
        # shader tiles: travelling light waves
        wave_r = (t * 0.5) % 1.7
        w1 = np.exp(-((self.tile_r - wave_r) / 0.17) ** 2)
        w2 = np.exp(-(((self.tile_c[:, 0] * 0.8 - self.tile_c[:, 1] * 0.6) - ((t * 0.8) % 3.4 - 1.7)) / 0.22) ** 2)
        flick = 0.5 + 0.5 * np.sin(t * 2.5 + self.tile_ph * TAU)
        e = np.clip(0.12 + 0.9 * w1 + 0.6 * w2 + 0.15 * flick, 0, 1.6)
        T = self.tiles
        y = 0.201
        n = len(T)
        tq = np.stack([np.stack([T[:, 0], np.full(n, y), T[:, 2]], 1), np.stack([T[:, 1], np.full(n, y), T[:, 2]], 1),
                       np.stack([T[:, 1], np.full(n, y), T[:, 3]], 1), np.stack([T[:, 0], np.full(n, y), T[:, 3]], 1)], 1)
        tscr, _ = self.quads(tq)
        for i, pts in enumerate(tscr):
            if pts is None:
                continue
            c = np.array([12, 14, 20], np.float32) + a * (0.04 + 0.34 * e[i])
            d.polygon(pts, fill=tint(c), outline=tint(a * (0.18 + 0.35 * e[i]) + 12))
        # ---- additive light
        R.points3d(self.core, a[None, :] * (e[self.core_tile] * 7.0)[:, None] + 4, 0.0, 0.7, bloom=0.7)
        R.lines3d(self.cacheA, self.cacheB, a * 0.5 + 30, width=0.8, bloom=0.4)
        x0, x1, _, _, z0, z1 = self.die
        ring = np.array([(x0, 0.2, z0), (x1, 0.2, z0), (x1, 0.2, z1), (x0, 0.2, z1)], np.float32)
        R.lines3d(ring, np.roll(ring, -1, 0), a * 1.2 + 20, width=1.3, bloom=0.9)
        R.lines3d(self.tA, self.tB, a * 0.22 + 4, width=0.8, bloom=0.6)
        on = self.t_on
        tl = (t * 0.45 + self.t_ph[on]) % 1.0
        P = self.tA[on] + (self.tB[on] - self.tA[on]) * tl[:, None]
        R.points3d(P, np.broadcast_to(a * 0.6 + 90, P.shape) * 5, 0.0, 1.2, bloom=1.0)
        R.points3d(self.bok, self.bok_c * 45, 0.06, 0.0, bloom=0.0)
        # metallic glints sliding along the top edges of the stiffener, die and memory stacks
        gA, gB = self.edges_A, self.edges_B
        mid = (gA + gB) * 0.5
        gpos = (mid[:, 0] * math.cos(t * 0.35) + mid[:, 2] * math.sin(t * 0.35))
        g = np.exp(-((gpos - 2.2 * math.sin(t * 0.5)) / 0.5) ** 2)
        R.lines3d(gA, gB, (np.array([200, 215, 235], np.float32)[None, :] * (0.25 + 2.2 * g)[:, None]), width=1.1, bloom=0.7)
        cx, cy, _, _ = R.project(np.array([[0, 0.2, 0]], np.float32))
        R.soft(np.repeat(cx, 2), np.repeat(cy, 2), np.stack([a * 90, a * 40]))
        return self.fin(canvas, soft_bg, canvas_bloom=0.2, bloom=(0.6, 0.55), radii=(1.4, 8.0))


# ======================================================================== 2. stock chart
GREEN = np.array([40, 235, 130], np.float32)
RED = np.array([255, 55, 80], np.float32)


def number_sprites(rng, n, size, blur, seed_strs=None):
    """Blurred price / percent strings as float (h, w) alpha sprites (for bokeh-like background numbers)."""
    f = font(size, "Inter-Bold.otf")
    out = []
    for i in range(n):
        if seed_strs:
            txt = seed_strs[i % len(seed_strs)]
        elif rng.random() < 0.5:
            txt = f"{rng.uniform(10, 999):.2f}"
        else:
            txt = f"{'+' if rng.random() < 0.6 else '-'}{rng.uniform(0.1, 9.9):.1f}%"
        l, t_, r, b = f.getbbox(txt)
        im = Image.new("L", (r - l + 4 * blur + 4, b - t_ + 4 * blur + 4), 0)
        ImageDraw.Draw(im).text((2 * blur + 2 - l, 2 * blur + 2 - t_), txt, fill=255, font=f)
        if blur:
            im = im.filter(ImageFilter.GaussianBlur(blur))
        out.append((np.asarray(im, np.float32) / 255.0, txt))
    return out


def add_sprite(buf, spr, cx, cy, col):
    """Add an alpha sprite (h, w) tinted col into buf (H, W, 3) centred at (cx, cy) (clipped)."""
    h, w = spr.shape
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    H, W = buf.shape[:2]
    sx0, sy0 = max(0, -x0), max(0, -y0)
    ex, ey = min(w, W - x0), min(h, H - y0)
    if ex <= sx0 or ey <= sy0:
        return
    buf[y0 + sy0:y0 + ey, x0 + sx0:x0 + ex] += spr[sy0:ey, sx0:ex, None] * np.asarray(col, np.float32)


class StockChart(Topical):
    """3D candlestick chart receding in perspective; glowing price line drawing forward with a bright
    head, translucent light curtain under the line, reflective grid floor, blurred ticker numbers
    floating in depth; the camera dollies along with the newest candle. trend = up / down / volatile."""
    default_accent = "#2FD4FF"

    def init(self):
        rng = self.rng
        n = 900
        drift = {"up": 0.010, "down": -0.010, "volatile": 0.0}[self.trend]
        vol = 0.030 if self.trend == "volatile" else 0.018
        r = rng.normal(drift, vol, n)
        if self.trend == "volatile":
            r += 0.022 * np.sin(np.arange(n) * 0.35) * rng.random(n)
        logp = np.cumsum(r)
        close = logp
        open_ = np.concatenate([[0.0], close[:-1]]) + rng.normal(0, vol * 0.15, n)
        hi = np.maximum(open_, close) + np.abs(rng.normal(0, vol * 0.6, n))
        lo = np.minimum(open_, close) - np.abs(rng.normal(0, vol * 0.6, n))
        self.sc = 14.0  # world units per unit log-price
        self.o, self.c, self.h, self.l = (v.astype(np.float32) * self.sc for v in (open_, close, hi, lo))
        self.dx = 0.42
        k = np.exp(-np.linspace(-3, 3, 41) ** 2)
        k /= k.sum()
        pad = np.pad(self.c, 20, mode="edge")
        self.smooth = np.convolve(pad, k, mode="valid").astype(np.float32)
        self.i0 = 300
        self.cps = 1.6  # candles per second
        self.line_col = {"up": GREEN, "down": RED, "volatile": self.acc}[self.trend]
        nspr = 18
        self.spr = number_sprites(rng, nspr, 30, 3)
        ns = 60
        self.sp_idx = rng.integers(0, nspr, ns)
        self.sp_pos = np.stack([rng.uniform(-40, 160, ns), rng.uniform(-5, 7, ns), rng.uniform(-16, -4, ns)], 1).astype(np.float32)
        self.sp_col = np.where((rng.random(ns) < (0.7 if self.trend == "up" else 0.3 if self.trend == "down" else 0.5))[:, None],
                               GREEN, RED) * 0.5 + self.acc * 0.2
        nb = 90
        self.bok = np.stack([rng.uniform(-40, 160, nb), rng.uniform(-8, 10, nb), rng.uniform(-30, -6, nb)], 1).astype(np.float32)
        self.bok_c = lerp_col(self.acc, self.line_col, rng.random(nb)) * (0.3 + rng.random(nb) ** 2)[:, None]

    def _y(self, arr, x):
        i = np.clip(x, 0, len(arr) - 2)
        j = np.floor(i).astype(np.int64)
        f = i - j
        return arr[j] * (1 - f) + arr[j + 1] * f

    def frame(self, t):
        R, a = self.R, self.acc
        head = self.i0 + t * self.cps
        hx = head * self.dx
        ys = float(self._y(self.smooth, head - 6))
        ylag = float(self._y(self.smooth, head - 14))
        tgt = np.array([hx - 2.6, ys + 0.2, 0.0])
        sway = math.sin(t * 0.23)
        cpos = tgt + np.array([-5.4 + 0.5 * sway, 1.3 + 0.3 * math.sin(t * 0.17), 6.4 + 0.5 * sway])
        R.begin(Camera(cpos, tgt, fov=52, roll=0.03 * math.sin(t * 0.19)))
        W, H = self.W, self.H
        canvas = Image.new("RGB", (W, H))
        d = ImageDraw.Draw(canvas)
        # sky gradient + horizon glow (screen-aligned; fine since roll is small)
        fy = ylag - 3.0  # floor height follows the price smoothly
        _, hy, _, _ = R.project(np.array([[hx + 400.0, fy, 400.0]]))
        hy = float(np.clip(hy[0], -H, 2 * H))
        yy = np.arange(H, dtype=np.float32)
        top = np.clip((hy - yy) / max(hy, 1.0), 0, 1)
        col = np.where((yy < hy)[:, None], lerp_col(a * 0.25 + 14, (10, 14, 26), top ** 0.7),
                       lerp_col(a * 0.14 + 12, (6, 8, 16), np.clip((yy - hy) / (H * 0.6), 0, 1) ** 0.6))
        col += np.exp(-np.abs(yy - hy) / (H * 0.05))[:, None] * (a * 0.22 + self.line_col * 0.12)
        bgcol = np.clip(col, 0, 255).astype(np.uint8)
        soft = np.ascontiguousarray(np.broadcast_to(bgcol[::R.ds][:R.Hs, None, :].astype(np.float32), (R.Hs, R.Ws, 3))) * 0.9
        # floating blurred numbers (quarter res, into the soft buffer)
        x, y, z, inv = R.project(self.sp_pos + np.array([0, ys, 0], np.float32))
        for j in np.nonzero(z > 1)[0]:
            spr = self.spr[self.sp_idx[j]][0]
            add_sprite(soft, spr, x[j] / R.ds, y[j] / R.ds, self.sp_col[j] * min(1.4, 18 / z[j]) * 0.8)
        # floor grid (additive lines), x-lines every candle*4, z-lines every unit
        x0 = math.floor((hx - 40) / 2) * 2
        gx = np.arange(x0, hx + 30, 2.0, dtype=np.float32)
        A = np.stack([gx, np.full_like(gx, fy), np.full_like(gx, -18)], 1)
        B = np.stack([gx, np.full_like(gx, fy), np.full_like(gx, 6)], 1)
        gz = np.arange(-18, 6.1, 1.5, dtype=np.float32)
        A2 = np.stack([np.full_like(gz, hx - 40), np.full_like(gz, fy), gz], 1)
        B2 = np.stack([np.full_like(gz, hx + 30), np.full_like(gz, fy), gz], 1)
        GA, GB = np.concatenate([A, A2]), np.concatenate([B, B2])
        mid = (GA + GB) * 0.5
        fade = np.exp(-np.abs(mid[:, 0] - hx) / 22.0)[:, None]
        R.lines3d(GA, GB, (a * 0.30 + 6) * fade, width=1.0, bloom=0.5)
        # candles
        i_lo = max(1, int(head) - 70)
        i_hi = int(head)
        frac = head - i_hi
        idx = np.arange(i_lo, i_hi + 1)
        o, c, h_, l_ = self.o[idx].copy(), self.c[idx].copy(), self.h[idx].copy(), self.l[idx].copy()
        # live candle grows
        ease = frac * frac * (3 - 2 * frac)
        c[-1] = o[-1] + (c[-1] - o[-1]) * ease
        h_[-1] = max(o[-1], c[-1]) + (h_[-1] - max(o[-1], self.c[i_hi])) * ease
        l_[-1] = min(o[-1], c[-1]) - (min(o[-1], self.c[i_hi]) - l_[-1]) * ease
        xs = idx * self.dx
        up = c >= o
        light = np.array([-0.4, 1.0, -0.6])
        order = np.argsort(-np.hypot(xs - cpos[0], cpos[2]))  # far first
        hw = 0.14
        for k in order.tolist():
            xc = float(xs[k])
            dist = float(np.hypot(xc - cpos[0], cpos[2]))
            fogf = math.exp(-max(0.0, dist - 9.0) / 14.0)
            base = (GREEN if up[k] else RED) * (0.55 * fogf) + a * 0.05
            y0, y1 = float(min(o[k], c[k])), float(max(o[k], c[k]))
            y1 = max(y1, y0 + 0.04)
            # wick (canvas line through the body)
            wv = self.to_cam(np.array([[xc, l_[k], 0.0], [xc, h_[k], 0.0]]))
            if (wv[:, 2] > 0.1).all():
                pts = self.screen_poly(np.vstack([wv, wv[::-1]]))
                if pts:
                    d.line([pts[0], pts[1]], fill=tint(base * 1.1 + 10), width=max(1, int(R.f * 0.025 / wv[0, 2])))
            for quad, n in self.box_faces((xc - hw, xc + hw, y0, y1, -hw, hw), cpos):
                ctr = np.mean(quad, 0)
                col_ = self.shade(base, n, light, cpos - ctr, amb=0.55, diff=0.6, spec=0.35 * fogf, power=20)
                V = self.to_cam(np.array(quad))
                pts = self.screen_poly(V)
                if pts:
                    d.polygon(pts, fill=tint(col_))
        # candle glow (additive) at body centres + reflections below the floor
        ctr = np.stack([xs, (o + c) * 0.5, np.zeros_like(xs)], 1).astype(np.float32)
        ccol = np.where(up[:, None], GREEN, RED) * 0.9
        R.points3d(ctr, ccol * 6, 0.0, 6.0, bloom=0.0)
        refl = ctr.copy()
        refl[:, 1] = 2 * fy - refl[:, 1]
        x, y, z, _ = R.project(refl)
        R.soft(x, y, ccol * 6.0)
        # price line (tube-ish glow) through closes up to the live head
        lx = np.concatenate([xs, [xs[-1]]]).astype(np.float32)
        ly = np.concatenate([c, [c[-1]]]).astype(np.float32)
        Lp = np.stack([lx, ly + 0.0, np.full_like(lx, 0.2)], 1)[:-1]
        lc = self.line_col
        age = np.clip((xs[-1] - Lp[:, 0]) / (60 * self.dx), 0, 1)[:, None]
        cl = lc * (1.4 - 0.8 * age) + 30 * (1 - age)
        R.lines3d(Lp[:-1], Lp[1:], cl[:-1] * 1.5, cl[1:] * 1.5, width=3.2, bloom=1.6)
        R.lines3d(Lp[:-1], Lp[1:], cl[:-1] * 0.6 + 60, cl[1:] * 0.6 + 60, width=0.8, bloom=0.0)
        # light curtain under the line (soft buffer)
        hx_, hy_, _, _ = R.project(Lp)
        fx_, fyy, _, _ = R.project(np.stack([Lp[:, 0], np.full(len(Lp), fy), Lp[:, 2]], 1))
        xs_, ys_, seg, s, step = R.sample_segments(hx_, hy_, fx_, fyy, 4.0, 300)
        R.soft(xs_, ys_, (lc * 0.32)[None, :] * ((1 - s) ** 1.5 * step * (1 - age[seg, 0] * 0.7))[:, None])
        # head: bright dot, pulse ring, guide lines
        hp = Lp[-1]
        px, py, pz, pi = R.project(hp[None, :])
        R.points(np.repeat(px, 3), np.repeat(py, 3), np.stack([np.full(3, 255.0)] * 3, 1) * [[3.0], [1.2], [0.5]] + lc * [[1.0], [3.0], [6.0]],
                 np.array([1.5, 4.0, 14.0]) * self.k, bloom=1.0)
        ph = (t * 0.9) % 1.0
        th = np.linspace(0, TAU, 90, endpoint=False)
        ring = hp[None, :] + np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], 1) * (0.15 + ph * 0.9)
        R.lines3d(ring, np.roll(ring, -1, 0), lc * (1 - ph) * 2.2, width=1.4, bloom=0.8)
        dash = np.linspace(fy, hp[1], 30)
        DA = np.stack([np.full(29, hp[0]), dash[:-1], np.full(29, hp[2])], 1)
        DB = np.stack([np.full(29, hp[0]), dash[:-1] + (dash[1] - dash[0]) * 0.5, np.full(29, hp[2])], 1)
        R.lines3d(DA, DB, lc * 0.7, width=1.0, bloom=0.5)
        hl = np.array([[hp[0], hp[1], hp[2]], [hp[0] + 16, hp[1], hp[2]]], np.float32)
        R.lines3d(hl[:1], hl[1:], lc * 0.8, lc * 0.0, width=1.0, bloom=0.6)
        R.points3d(self.bok + np.array([0, ys, 0], np.float32), self.bok_c * 40, 0.08, 0.0, bloom=0.0)
        return self.fin(canvas, soft, canvas_bloom=0.25, bloom=(0.6, 0.6), radii=(1.4, 8.0))


# ======================================================================== 3. ticker wall
_FAMOUS = {"AAPL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "GOOG", "IBM", "AMD", "INTC", "NFLX", "BABA", "ORCL", "UBER",
           "COKE", "KO", "GE", "GM", "F", "T", "BA", "JPM", "GS", "MS", "WMT", "DIS", "SONY", "SAP", "ASML", "TSM", "BAC",
           "CVX", "XOM", "PEP", "MCD", "NKE", "V", "MA", "PYPL", "ADBE", "CRM", "QCOM", "AVGO", "MU", "ARM", "SHOP"}


def fake_symbol(rng) -> str:
    letters = "ABCDEFGHJKLMNPRSTVWXZ"
    while True:
        sym = "".join(letters[i] for i in rng.integers(0, len(letters), int(rng.integers(3, 5))))
        if sym not in _FAMOUS:
            return sym


class TickerWall(Topical):
    """Stock-exchange LED ticker boards: dot-matrix bands crawling with generic random symbols and
    green/red changes, seen at an angle while the camera pans; glossy floor reflections + bokeh."""
    default_accent = "#FFB020"

    def init(self):
        rng = self.rng
        self.pitch = 0.075
        self.nb = 10
        self.band_y = (np.arange(self.nb) - (self.nb - 1) / 2) * 0.78 + 0.6
        p_up = {"up": 0.72, "down": 0.28, "volatile": 0.5}[self.trend]
        self.bands = []
        for b in range(self.nb):
            parts, cols = [], []
            while sum(len(x) for x in parts) < 420:
                sym = fake_symbol(rng)
                upq = rng.random() < p_up
                chg = f"{'+' if upq else '-'}{rng.uniform(0.1, 9.9):.1f}%"
                px = f"{rng.uniform(5, 950):.2f}"
                seg = [(sym + " ", self.acc * 0.85 + 30), (px + " ", np.array([235, 235, 240], np.float32)),
                       (("^" if upq else "v") + chg, GREEN if upq else RED), ("   ", GREEN)]
                for txt, c in seg:
                    parts.append(txt)
                    cols.append(np.repeat(np.asarray(c, np.float32)[None, :], len(txt) * 6, 0))
            bmp = led_text("".join(parts))
            colmap = np.concatenate(cols)[:bmp.shape[1]]
            self.bands.append((bmp, colmap))
        self.scroll = rng.uniform(16, 26, self.nb)
        nbk = 80
        self.bok = np.stack([rng.uniform(-10, 60, nbk), rng.uniform(-5, 6, nbk), rng.uniform(4, 9, nbk)], 1).astype(np.float32)
        self.bok_c = lerp_col(self.acc, (120, 180, 255), rng.random(nbk)) * (0.3 + rng.random(nbk))[:, None]
        nfar = 300
        self.far = np.stack([rng.uniform(-20, 80, nfar), rng.uniform(-3, 7, nfar), rng.uniform(-14, -4, nfar)], 1).astype(np.float32)
        self.far_c = lerp_col((255, 170, 80), (90, 140, 255), rng.random(nfar)) * (0.2 + rng.random(nfar) ** 3)[:, None]
        self.floor_y = self.band_y[0] - 1.2

    def frame(self, t):
        R, a = self.R, self.acc
        pan = t * 0.55
        cpos = np.array([pan - 3.0, 0.9 + 0.15 * math.sin(t * 0.2), 7.2 + 0.3 * math.sin(t * 0.13)])
        tgt = np.array([pan + 3.6, 0.7, 0.0])
        R.begin(Camera(cpos, tgt, fov=58, roll=0.02 * math.sin(t * 0.15)))
        W, H = self.W, self.H
        canvas = Image.new("RGB", (W, H))
        d = ImageDraw.Draw(canvas)
        bg = vgrad(1, H, [(0, (6, 6, 16)), (0.45, (14, 12, 24)), (0.75, (10, 8, 16)), (1, (5, 5, 10))])[:, 0].astype(np.float32) + a * 0.07 + 3
        soft = np.ascontiguousarray(np.broadcast_to(bg[::R.ds][:R.Hs, None, :], (R.Hs, R.Ws, 3))).copy()
        # board backing panels (opaque)
        x_lo, x_hi = pan - 9.0, pan + 34.0
        p = self.pitch
        fy = self.floor_y
        for b in range(self.nb):
            y0 = self.band_y[b] - 0.06
            y1 = self.band_y[b] + 7 * p + 0.02
            V = self.to_cam(np.array([(x_lo, y0, -0.02), (x_hi, y0, -0.02), (x_hi, y1, -0.02), (x_lo, y1, -0.02)]))
            pts = self.screen_poly(V)
            if pts:
                d.polygon(pts, fill=(5, 5, 8), outline=(46, 44, 58))
        # wall above/below bands slightly lit
        V = self.to_cam(np.array([(x_lo, fy, 0.05), (x_hi, fy, 0.05), (x_hi, fy + 9.5, 0.05), (x_lo, fy + 9.5, 0.05)]))
        # LEDs
        c0 = int(math.floor(x_lo / p))
        ncol = int((x_hi - x_lo) / p)
        cols = c0 + np.arange(ncol)
        X = cols * p
        lit_P, lit_C = [], []
        dist = np.hypot(X - cpos[0], cpos[2])
        fade = np.clip(np.exp(-np.maximum(0, dist - 8) / 18.0), 0, 1).astype(np.float32)
        for b, (bmp, colmap) in enumerate(self.bands):
            L = bmp.shape[1]
            sh = int(t * self.scroll[b])
            src = (cols + sh) % L
            on = bmp[:, src]  # (7, ncol)
            r_, c_ = np.nonzero(on)
            y = self.band_y[b] + (6 - r_) * p
            lit_P.append(np.stack([X[c_], y, np.zeros_like(y)], 1))
            lit_C.append(colmap[src[c_]] * fade[c_, None])
        P = np.concatenate(lit_P).astype(np.float32)
        C = np.concatenate(lit_C).astype(np.float32)
        x, y, z, inv = R.project(P)
        r = np.clip(0.011 * inv, 0.5, 2.9)
        R.points(x, y, C * (6.0 + 6.0 * r * r)[:, None], r, bloom=0.45)
        # unlit LED grid: faint, only near rows (cheap moire-free hint via soft buffer)
        # reflections in the glossy floor
        Pr = P.copy()
        Pr[:, 1] = 2 * fy - Pr[:, 1]
        x, y, z, _ = R.project(Pr[::3])
        R.soft(x, y, C[::3] * 3.0)
        # floor sheen line + far bokeh lights (hall) + near bokeh
        R.points3d(self.far, self.far_c * 30, 0.06, 0.0, bloom=0.2)
        R.points3d(self.bok, self.bok_c * 90, 0.18, 0.0, bloom=0.0)
        fl = np.array([[x_lo, fy, 0.0], [x_hi, fy, 0.0]], np.float32)
        R.lines3d(fl[:1], fl[1:], a * 0.5, width=1.2, bloom=1.0)
        # accent light washing the wall from below
        xs = np.linspace(x_lo, x_hi, 60)
        x, y, _, _ = R.project(np.stack([xs, np.full(60, fy + 0.3), np.zeros(60)], 1))
        R.soft(x, y, a * 14)
        return self.fin(canvas, soft, canvas_bloom=0.0, bloom=(0.6, 0.35), radii=(1.2, 6.0))


# ======================================================================== 4. corporate skyline
class Corporate(Topical):
    """Night skyline of glass towers: lit window grids (occlusion-tested against an ID buffer),
    view-dependent glass reflections, one central headquarters tower outlined in the accent colour
    with a beacon, street lights and traffic, drifting low clouds; slow aerial push-in."""
    default_accent = "#3FA9FF"

    def init(self):
        rng = self.rng
        towers = []
        # central HQ
        towers.append((0.0, -6.0, 1.5, 1.5, 11.5))
        for gx in range(-8, 9):
            for gz in range(-12, 5):
                x, z = gx * 2.2, gz * 2.2 - 6.0
                if abs(x) < 2.6 and abs(z + 6.0) < 2.6:
                    continue
                if rng.random() < 0.18:
                    continue
                for sub in range(1 if rng.random() < 0.6 else 2):
                    w = rng.uniform(0.55, 0.95) * (1 if sub == 0 else 0.7)
                    dd = rng.uniform(0.55, 0.95) * (1 if sub == 0 else 0.7)
                    ox, oz = (0, 0) if sub == 0 else (rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4))
                    dcen = math.hypot(x, (z + 6.0) * 0.7)
                    hmax = 2.0 + 7.5 * math.exp(-dcen / 9.0)
                    h = float(np.clip(rng.gamma(2.0, 0.5) * hmax * 0.55, 0.6, 9.5))
                    towers.append((x + ox, z + oz, w, dd, h))
        self.towers = np.array(towers, np.float32)
        T = self.towers
        n = len(T)
        self.glass = lerp_col((10, 14, 28), (22, 26, 42), rng.random(n))
        # windows: per tower, per side face (0:+z, 1:-z, 2:+x, 3:-x)
        P, tid, fid, col, ph = [], [], [], [], []
        warm = np.array([255, 165, 70], np.float32)
        cool = np.array([140, 190, 255], np.float32)
        for i, (x, z, w, dd, h) in enumerate(T):
            fl = np.arange(0.22, h - 0.1, 0.24 if i else 0.2)
            for f, (nx, nz) in enumerate(((0, 1), (0, -1), (1, 0), (-1, 0))):
                half = w if nz else dd
                cs = np.arange(-half + 0.12, half - 0.05, 0.17 if i else 0.14)
                if not len(cs) or not len(fl):
                    continue
                U, Y = np.meshgrid(cs, fl)
                U, Y = U.ravel(), Y.ravel()
                if nz:
                    X, Z = x + U, np.full_like(U, z + nz * dd * 1.002)
                else:
                    X, Z = np.full_like(U, x + nx * w * 1.002), z + U
                lit = rng.random(len(U)) < (0.85 if i == 0 else 0.5)
                P.append(np.stack([X, Y, Z], 1)[lit])
                k = int(lit.sum())
                tid.append(np.full(k, i))
                fid.append(np.full(k, f))
                c = np.where((rng.random(k) < 0.75)[:, None], warm, cool) * (0.35 + 0.65 * rng.random(k) ** 2)[:, None]
                if i == 0:
                    c = lerp_col(self.acc, (255, 255, 255), rng.random(k) * 0.35) * (0.5 + 0.3 * rng.random(k))[:, None]
                col.append(c)
                ph.append(rng.random(k))
        self.win = np.concatenate(P).astype(np.float32)
        self.win_t = np.concatenate(tid)
        self.win_f = np.concatenate(fid)
        self.win_c = np.concatenate(col).astype(np.float32)
        self.win_ph = np.concatenate(ph).astype(np.float32)
        # street lights along avenues + traffic
        sl = []
        for gx in range(-9, 10):
            for zz in np.arange(-42, 12, 0.7):
                sl.append((gx * 2.2 + 1.1, 0.02, zz))
        for gz in range(-16, 6):
            for xx in np.arange(-20, 20, 0.7):
                sl.append((xx, 0.02, gz * 2.2 - 6.0 + 1.1))
        self.sl = np.array(sl, np.float32)
        ncar = 260
        self.car_ax = rng.integers(0, 2, ncar)
        self.car_lane = np.where(self.car_ax == 0, rng.integers(-9, 10, ncar) * 2.2 + 1.1, rng.integers(-16, 6, ncar) * 2.2 - 6.0 + 1.1)
        self.car_s0 = rng.uniform(0, 1, ncar)
        self.car_v = rng.uniform(0.01, 0.025, ncar) * rng.choice([-1, 1], ncar)
        self.make_sky(1200, bright=0.5)
        Hs, Ws = self.R.Hs, self.R.Ws
        cl = periodic_noise((128, 256), rng, beta=2.2)
        cl = np.asarray(Image.fromarray(cl).resize((Ws * 2, Hs), Image.BICUBIC), np.float32)
        self.cloud = np.clip(cl - 0.45, 0, 1) ** 1.3

    def frame(self, t):
        R, a = self.R, self.acc
        push = 1 - math.exp(-t / 14.0)
        cpos = np.array([4.0 * math.sin(t * 0.09 + 0.3), 13.0 - 2.0 * push + 0.2 * math.sin(t * 0.2), 16.0 - 7.0 * push])
        tgt = np.array([0.0, 4.0, -8.0])
        R.begin(Camera(cpos, tgt, fov=55, roll=0.015 * math.sin(t * 0.13)))
        W, H = self.W, self.H
        # sky (screen gradient keyed to the horizon) + clouds into the soft buffer
        _, hy, _, _ = R.project(np.array([[0.0, 0.0, -2000.0]]))
        hy = float(np.clip(hy[0], 1, 2 * H))
        yy = np.arange(H, dtype=np.float32)
        u = np.clip(yy / hy, 0, 1)
        sky = lerp_col((10, 14, 38), (70, 50, 96), u ** 2.5) + np.exp(-np.abs(yy - hy) / (H * 0.05))[:, None] * (a * 0.25 + np.array([70, 40, 50], np.float32))
        sky = np.where((yy > hy)[:, None], lerp_col((40, 30, 52), (14, 12, 24), np.clip((yy - hy) / (H * 0.25), 0, 1) ** 0.6), sky)
        key = (int(hy), tuple(a.tolist()))
        if getattr(self, '_sky_key', None) != key:
            self._sky_img = Image.fromarray(np.ascontiguousarray(np.broadcast_to(np.clip(sky, 0, 255).astype(np.uint8)[:, None, :], (H, W, 3))))
            self._sky_key = key
        canvas = self._sky_img.copy()
        d = ImageDraw.Draw(canvas)
        idb = Image.new("I", (W // 2, H // 2), 0)
        di = ImageDraw.Draw(idb)
        T = self.towers
        dist = np.hypot(T[:, 0] - cpos[0], T[:, 1] - cpos[2])
        order = np.argsort(-dist)
        n = len(T)
        x0s, x1s = T[:, 0] - T[:, 2], T[:, 0] + T[:, 2]
        z0s, z1s = T[:, 1] - T[:, 3], T[:, 1] + T[:, 3]
        hs = T[:, 4]
        zero = np.zeros(n, np.float32)
        CX = np.stack([x0s, x1s, x1s, x0s, x0s, x1s, x1s, x0s], 1)
        CY = np.stack([zero, zero, zero, zero, hs, hs, hs, hs], 1)
        CZ = np.stack([z0s, z0s, z1s, z1s, z0s, z0s, z1s, z1s], 1)
        Cw = np.stack([CX, CY, CZ], -1).reshape(-1, 3)
        V = self.to_cam(Cw).reshape(n, 8, 3)
        f = R.f
        zz = np.maximum(V[..., 2], 1e-3)
        SX = (W * 0.5 + V[..., 0] * f / zz).tolist()
        SY = (H * 0.5 - V[..., 1] * f / zz).tolist()
        okv = V[..., 2] > 0.05
        # faces: (vertex idx, normal) ; 0:-z 1:+z 2:-x 3:+x 4:top
        FACES = (((0, 1, 5, 4), (0, 0, -1)), ((3, 2, 6, 7), (0, 0, 1)), ((0, 3, 7, 4), (-1, 0, 0)), ((1, 2, 6, 5), (1, 0, 0)),
                 ((4, 5, 6, 7), (0, 1, 0)))
        vis = np.stack([cpos[2] < z0s, cpos[2] > z1s, cpos[0] < x0s, cpos[0] > x1s, cpos[1] > hs], 1)
        fog = np.exp(-np.maximum(0.0, dist - 12) / 30.0)[:, None]
        lightd = _norm([0.5, 0.6, 0.8])
        cols = []
        for fi, (_, nrm) in enumerate(FACES):
            nrm = np.array(nrm, np.float64)
            ctr = np.stack([T[:, 0] + nrm[0] * T[:, 2], hs * (0.5 if fi < 4 else 1.0), T[:, 1] + nrm[2] * T[:, 3]], 1)
            vd = cpos[None, :] - ctr
            vd /= np.linalg.norm(vd, axis=1, keepdims=True)
            fres = (1 - np.clip(vd @ nrm, 0, 1)) ** 3
            base = self.glass * (0.55 + 0.35 * max(0.0, float(nrm @ lightd)))
            c = base + np.array([28, 30, 56], np.float32) * fres[:, None] if fi < 4 else self.glass * 0.45
            c = c * fog + np.array([30, 26, 50], np.float32) * (1 - fog)
            cols.append(np.clip(c, 0, 255).astype(np.int64).tolist())
        visl = vis.tolist()
        allok = okv.all(1).tolist()
        for i in order.tolist():
            for fi, (idx, _) in enumerate(FACES):
                if not visl[i][fi]:
                    continue
                if allok[i]:
                    pts = [(SX[i][j], SY[i][j]) for j in idx]
                else:
                    pts = self.screen_poly(V[i, list(idx)])
                    if not pts:
                        continue
                d.polygon(pts, fill=tuple(cols[fi][i]))
                di.polygon([(px * 0.5, py * 0.5) for px, py in pts], fill=i + 1)
        ids = np.asarray(idb)
        # windows
        vis_face = np.zeros((len(T), 4), bool)
        vis_face[:, 0] = cpos[2] > T[:, 1] + T[:, 3]
        vis_face[:, 1] = cpos[2] < T[:, 1] - T[:, 3]
        vis_face[:, 2] = cpos[0] > T[:, 0] + T[:, 2]
        vis_face[:, 3] = cpos[0] < T[:, 0] - T[:, 2]
        m = vis_face[self.win_t, self.win_f]
        Pw = self.win[m]
        x, y, z, inv = R.project(Pw)
        xi = np.clip((x * 0.5).astype(np.int64), 0, W // 2 - 1)
        yi = np.clip((y * 0.5).astype(np.int64), 0, H // 2 - 1)
        own = ids[yi, xi] == self.win_t[m] + 1
        inside = (x >= 0) & (x < W) & (y >= 0) & (y < H) & (z > 0.1)
        k = own & inside
        fl = 0.75 + 0.25 * np.sin(t * 1.3 + self.win_ph[m][k] * 40)
        fog = np.exp(-np.maximum(0, z[k] - 10) / 30.0)
        c = self.win_c[m][k] * (fl * (0.35 + 0.65 * fog) * 22)[:, None]
        R.points(x[k], y[k], c, np.clip(0.03 * inv[k], 0.4, 1.6), bloom=0.5)
        # HQ tower: accent edges, crown rings, beacon
        x0, z0, w, dd, h = (float(v) for v in T[0])
        corners = [(x0 - w, z0 - dd), (x0 + w, z0 - dd), (x0 + w, z0 + dd), (x0 - w, z0 + dd)]
        segA, segB = [], []
        for cx, cz in corners:
            segA.append((cx, 0.0, cz))
            segB.append((cx, h, cz))
        for yy_ in (h, h - 0.35, h - 0.7):
            for k_ in range(4):
                segA.append((corners[k_][0], yy_, corners[k_][1]))
                segB.append((corners[(k_ + 1) % 4][0], yy_, corners[(k_ + 1) % 4][1]))
        xa, ya, za, _ = R.project(np.array(segA, np.float32))
        xb, yb, zb, _ = R.project(np.array(segB, np.float32))
        xs_, ys_, seg, s_, step = R.sample_segments(xa, ya, xb, yb, 1.0, 2000)
        xi = np.clip((xs_ * 0.5).astype(np.int64), 0, W // 2 - 1)
        yi = np.clip((ys_ * 0.5).astype(np.int64), 0, H // 2 - 1)
        idn = ids[yi, xi]
        # keep edge samples not hidden by another tower (check a 1px neighbourhood: edges sit on silhouettes)
        idn2 = ids[np.clip(yi + 1, 0, H // 2 - 1), np.clip(xi + 1, 0, W // 2 - 1)]
        idn3 = ids[np.clip(yi - 1, 0, H // 2 - 1), np.clip(xi - 1, 0, W // 2 - 1)]
        keep = (idn <= 1) | ((idn2 <= 1) & (idn3 <= 1))
        ec = (a * 2.0)[None, :] * (step * (0.5 + 0.5 * s_ * (seg < 4) + (seg >= 4)))[:, None]
        R.line_samples(xs_[keep], ys_[keep], ec[keep], 1.8 * self.k, np.ones(keep.sum(), np.float32), np.zeros(keep.sum(), np.float32), bloom=1.0)
        bl = 0.5 + 0.5 * math.sin(t * 3.0)
        R.points3d(np.array([[x0, h + 0.6, z0]], np.float32), np.array([[255, 255, 255]], np.float32) * (2 + 6 * bl) + a * 4, 0.0,
                   4.0, bloom=1.0)
        R.lines3d(np.array([[x0, h, z0]], np.float32), np.array([[x0, h + 0.6, z0]], np.float32), a * 1.2, width=1.0)
        cxp, cyp, _, _ = R.project(np.array([[x0, h * 0.6, z0]], np.float32))
        R.soft(np.repeat(cxp, 2), np.repeat(cyp, 2), np.stack([a * 260, a * 120]))
        # street lights & traffic (occluded by towers via the ID buffer)
        x, y, z, inv = R.project(self.sl)
        xi = np.clip((x * 0.5).astype(np.int64), 0, W // 2 - 1)
        yi = np.clip((y * 0.5).astype(np.int64), 0, H // 2 - 1)
        k = (ids[yi, xi] == 0) & (z > 0.1)
        fog = np.exp(-np.maximum(0, z[k] - 10) / 25.0)
        R.points(x[k], y[k], np.array([255, 160, 70], np.float32)[None, :] * (fog * 3.5)[:, None], None, bloom=0.8)
        s_ = (self.car_s0 + self.car_v * t) % 1.0
        along = np.where(self.car_ax == 0, -42 + s_ * 54, -20 + s_ * 40)
        Pc = np.where((self.car_ax == 0)[:, None], np.stack([self.car_lane, np.full_like(s_, 0.03), along], 1),
                      np.stack([along, np.full_like(s_, 0.03), self.car_lane], 1)).astype(np.float32)
        x, y, z, inv = R.project(Pc)
        xi = np.clip((x * 0.5).astype(np.int64), 0, W // 2 - 1)
        yi = np.clip((y * 0.5).astype(np.int64), 0, H // 2 - 1)
        k = (ids[yi, xi] == 0) & (z > 0.1)
        cc = np.where((self.car_v > 0)[:, None], np.array([255, 60, 50], np.float32), np.array([255, 240, 220], np.float32))
        R.points(x[k], y[k], cc[k] * 4, None, bloom=0.9)
        self.draw_sky(lambda dvec: dvec[:, 1] > 0.08)
        # drifting low clouds / fog (quarter res soft)
        o = int(t * 6) % self.R.Ws
        cl = self.cloud[:, o:o + R.Ws]
        hyq = hy / R.ds
        band = np.exp(-((np.arange(R.Hs, dtype=np.float32) - hyq * 1.05) / (R.Hs * 0.12)) ** 2)[:, None]
        soft = cl[..., None] * band[..., None] * np.array([70, 70, 100], np.float32) * 0.9
        return self.fin(canvas, soft, canvas_bloom=0.12, bloom=(0.6, 0.5), radii=(1.3, 7.0))


# ======================================================================== 5. newsroom
class Newsroom(Topical):
    """TV news studio: huge curved LED screen wall showing a world-map dot matrix with scanning light
    bars, red pulsing on-air shapes (no words), ceiling light rig with anamorphic lens flares, anchor
    desk with an accent light strip, glossy floor reflections; slow camera drift."""
    default_accent = "#2F8BFF"

    def init(self):
        rng = self.rng
        self.Rw = 8.0  # wall radius (centre at origin, camera inside looking -z)
        self.a0, self.a1 = -1.05, 1.05  # angular extent
        self.y0, self.y1 = 0.3, 6.3
        mw, mh = 200, 80
        land = world_mask(mw, mh)
        jj, ii = np.nonzero(np.ones((mh, mw), bool))
        u = (ii + 0.5) / mw
        v = (jj + 0.5) / mh
        ang = self.a0 + (self.a1 - self.a0) * u
        y = self.y1 - 0.4 - (self.y1 - self.y0 - 1.0) * v
        P = np.stack([self.Rw * np.sin(ang), y, -self.Rw * np.cos(ang)], 1).astype(np.float32)
        isl = land[jj, ii]
        self.dots = P
        self.dot_land = isl
        self.dot_v = v.astype(np.float32)
        self.dot_u = u.astype(np.float32)
        self.dot_ph = rng.random(len(P)).astype(np.float32)
        cidx = []
        for lo, la in CITIES:
            cu, cv = lonlat_uv(lo, la)
            cidx.append(int(min(mh - 1, int(cv * mh))) * mw + int(min(mw - 1, int(cu * mw))))
        self.city = np.array(cidx)
        # wall panels (quads) for the opaque screen body
        nA, nY = 14, 6
        quads = []
        for i in range(nA):
            for j in range(nY):
                a0 = self.a0 + (self.a1 - self.a0) * i / nA
                a1 = self.a0 + (self.a1 - self.a0) * (i + 1) / nA
                yb = self.y0 + (self.y1 - self.y0) * j / nY
                yt = self.y0 + (self.y1 - self.y0) * (j + 1) / nY
                quads.append([(self.Rw * math.sin(a0), yb, -self.Rw * math.cos(a0)), (self.Rw * math.sin(a1), yb, -self.Rw * math.cos(a1)),
                              (self.Rw * math.sin(a1), yt, -self.Rw * math.cos(a1)), (self.Rw * math.sin(a0), yt, -self.Rw * math.cos(a0))])
        self.panels = np.array(quads, np.float32)
        # ceiling light rig: spotlights along two arcs
        L = []
        for k in range(13):
            ang = -1.0 + 2.0 * k / 12
            L.append((6.0 * math.sin(ang), 7.4, -6.0 * math.cos(ang) + 1.0))
        for k in range(7):
            L.append((-4.5 + 1.5 * k, 7.9, 1.0))
        self.lights = np.array(L, np.float32)
        self.light_ph = rng.random(len(L)).astype(np.float32)
        # anchor desk: arc of quads
        desk = []
        for i in range(16):
            a0 = -0.75 + 1.5 * i / 16
            a1 = -0.75 + 1.5 * (i + 1) / 16
            r = 2.6
            cz = -1.2
            p0 = (r * math.sin(a0), 0.0, cz + r * math.cos(a0) * 0.45)
            p1 = (r * math.sin(a1), 0.0, cz + r * math.cos(a1) * 0.45)
            desk.append((p0, p1))
        self.desk = desk
        nb = 50
        self.bok = np.stack([rng.uniform(-8, 8, nb), rng.uniform(0.5, 7, nb), rng.uniform(1.5, 5, nb)], 1).astype(np.float32)
        self.bok_c = lerp_col(self.acc, (255, 255, 255), rng.random(nb) * 0.5) * (0.2 + 0.5 * rng.random(nb))[:, None]

    def frame(self, t):
        R, a = self.R, self.acc
        cx = 2.6 * math.sin(t * 0.11 + 0.6)
        cpos = np.array([cx, 1.9 + 0.2 * math.sin(t * 0.17), 5.0 + 0.5 * math.sin(t * 0.09)])
        tgt = np.array([-cx * 0.4, 3.4, -6.0])
        R.begin(Camera(cpos, tgt, fov=62, roll=0.015 * math.sin(t * 0.11)))
        W, H = self.W, self.H
        canvas = Image.new("RGB", (W, H), (12, 14, 26))
        d = ImageDraw.Draw(canvas)
        # floor plane (dark glossy) + back studio walls
        G = 30.0
        for quad, col in (([(-G, 0, -G), (G, 0, -G), (G, 0, G), (-G, 0, G)], (16, 18, 30)),):
            pts = self.screen_poly(self.to_cam(np.array(quad)))
            if pts:
                d.polygon(pts, fill=col)
        # screen wall panels
        polys, _ = self.quads(self.panels)
        for pts in polys:
            if pts:
                d.polygon(pts, fill=(12, 22, 44), outline=(30, 42, 72))
        # desk (front face + top with light strip)
        dA = np.array([p for p, _ in self.desk], np.float32)
        dB = np.array([q for _, q in self.desk], np.float32)
        h = 1.05
        dq = []
        for (p0, p1) in self.desk:
            dq.append([p0, p1, (p1[0], h, p1[2]), (p0[0], h, p0[2])])
        tq = []
        for (p0, p1) in self.desk:
            tq.append([(p0[0], h, p0[2]), (p1[0], h, p1[2]), (p1[0] * 0.8, h, p1[2] - 0.5), (p0[0] * 0.8, h, p0[2] - 0.5)])
        tpolys, _ = self.quads(np.array(tq, np.float32))
        fpolys, _ = self.quads(np.array(dq, np.float32))
        for pts in tpolys:
            if pts:
                d.polygon(pts, fill=(30, 34, 48))
        for i, pts in enumerate(fpolys):
            if pts:
                sh = 0.6 + 0.4 * math.cos((i - 7.5) / 8 * 1.5)
                d.polygon(pts, fill=tint(np.array([14, 16, 26]) * sh + 4))
        yq = np.linspace(0, 1, R.Hs, dtype=np.float32)[:, None, None]
        soft = np.broadcast_to((np.exp(-((yq - 0.4) / 0.25) ** 2) * (a * 0.05 + 4)), (R.Hs, R.Ws, 3)).astype(np.float32)
        # ---- map dots on the curved wall
        P = self.dots
        x, y, z, inv = R.project(P)
        land = self.dot_land
        bar1 = np.exp(-((self.dot_v - ((t * 0.18) % 1.4 - 0.2)) / 0.035) ** 2)
        bar2 = np.exp(-((self.dot_u - ((t * 0.11 + 0.5) % 1.4 - 0.2)) / 0.02) ** 2)
        tw = 0.75 + 0.25 * np.sin(t * 2.0 + self.dot_ph * TAU)
        e = np.where(land, 1.5 * tw + 2.0 * bar1 + 1.4 * bar2, 0.22 + 0.9 * bar1 + 0.6 * bar2)
        col = lerp_col(a, (220, 240, 255), np.clip(bar1 + bar2 * 0.6, 0, 1) * 0.7) * (e * 9)[:, None]
        R.points(x, y, col, np.clip(0.012 * inv, 0.5, 2.2), bloom=0.5)
        # reflection on the glossy floor
        Pr = P[land].copy()
        Pr[:, 1] = -Pr[:, 1]
        xr, yr, zr, _ = R.project(Pr[::2])
        dx_ = [p_[0] for q in tpolys if q for p_ in q]
        dy_ = [p_[1] for q in tpolys if q for p_ in q]
        if dx_:
            hide = (xr > min(dx_)) & (xr < max(dx_)) & (yr > min(dy_))
            xr, yr = xr[~hide], yr[~hide]
            cr = col[land][::2][~hide]
        else:
            cr = col[land][::2]
        R.soft(xr, yr, cr * 0.5)
        # city pings + red alert blips (live markers)
        pc = P[self.city]
        ph = (t * 0.7 + np.arange(len(pc)) * 0.37) % 1.0
        xc, yc, zc, ic = R.project(pc)
        R.points(xc, yc, np.broadcast_to(np.array([255, 255, 255], np.float32) * 4 + a * 6, (len(pc), 3)), 1.6 * self.k, bloom=1.0)
        th = np.linspace(0, TAU, 48, endpoint=False)
        for j in range(len(pc)):
            r = (0.05 + ph[j] * 0.35) * ic[j]
            R.points(xc[j] + np.cos(th) * r, yc[j] + np.sin(th) * r, np.broadcast_to(a * (1 - ph[j]) * 3.5, (48, 3)), None, bloom=0.6)
        # red pulsing on-air shapes: rounded rectangle outline + dot, upper-left of the wall, and a red tally light
        pulse = 0.55 + 0.45 * math.sin(t * 4.0)
        ang_c, yc0 = -0.30, 5.75
        w_, h_ = 0.55, 0.17
        pts3 = []
        for k in range(120):
            s_ = k / 120 * TAU
            ox = math.copysign(min(1.0, abs(math.cos(s_)) * 1.6), math.cos(s_)) * w_
            oy = math.copysign(min(1.0, abs(math.sin(s_)) * 1.6), math.sin(s_)) * h_
            ang = ang_c + ox / self.Rw
            pts3.append((self.Rw * 0.995 * math.sin(ang), yc0 + oy, -self.Rw * 0.995 * math.cos(ang)))
        x, y, z, _ = R.project(np.array(pts3, np.float32))
        red = np.array([255, 40, 50], np.float32)
        R.points(x, y, np.broadcast_to(red * 14 * pulse, (len(x), 3)), 1.2 * self.k, bloom=1.0)
        dotp = np.array([[self.Rw * 0.99 * math.sin(ang_c - 0.03), yc0, -self.Rw * 0.99 * math.cos(ang_c - 0.03)]], np.float32)
        x, y, z, _ = R.project(dotp)
        R.points(np.repeat(x, 2), np.repeat(y, 2), np.stack([red * 120 * pulse, red * 900 * pulse]), np.array([3.0, 12.0]) * self.k, bloom=1.0)
        # bars beside it (abstract lower-third shapes, no text)
        for k_, (ang_s, ln, yb) in enumerate(((-0.18, 0.42, 5.75), (-0.20, 0.3, 5.45))):
            seg = np.linspace(ang_s, ang_s + ln, 40)
            Pb = np.stack([self.Rw * 0.995 * np.sin(seg), np.full(40, yb), -self.Rw * 0.995 * np.cos(seg)], 1).astype(np.float32)
            R.lines3d(Pb[:-1], Pb[1:], (a * 1.2 + 40) if k_ == 0 else a * 0.7, width=5.0 if k_ == 0 else 3.0, bloom=0.8)
        # desk light strip
        R.lines3d(np.array([(p[0], 0.85, p[2] + 0.01) for p in dA], np.float32), np.array([(q[0], 0.85, q[2] + 0.01) for q in dB], np.float32),
                  a * 3.0, width=2.5, bloom=1.2)
        R.lines3d(np.array([(p[0], h, p[2]) for p in dA], np.float32), np.array([(q[0], h, q[2]) for q in dB], np.float32),
                  np.array([200, 210, 230], np.float32) * 0.8, width=1.2, bloom=0.5)
        # ceiling lights + anamorphic flares
        x, y, z, inv = R.project(self.lights)
        ok = (z > 0.2) & (x > -50) & (x < W + 50) & (y > -50) & (y < H + 50)
        flick = 0.85 + 0.15 * np.sin(t * 1.7 + self.light_ph * TAU)
        warm = np.array([255, 236, 210], np.float32)
        for j in np.nonzero(ok)[0]:
            b = float(flick[j])
            R.points(np.repeat(x[j], 3), np.repeat(y[j], 3), np.stack([warm * 300 * b, warm * 900 * b, (a * 0.7 + 60) * 2500 * b]),
                     np.array([1.5, 5.0, 16.0]) * self.k, bloom=1.0)
            L = W * 0.18
            xs = np.linspace(x[j] - L, x[j] + L, 120)
            fall = np.exp(-np.abs(xs - x[j]) / (L * 0.3))
            R.points(xs, np.full(120, y[j]), (a * 0.5 + 60)[None, :] * (fall * 22.0 * b)[:, None], None, bloom=0.3)
            # volumetric beam down to the floor
            fl_ = self.lights[j].copy()
            fl_[1] = 0.0
            fx, fy_, fz, _ = R.project(fl_[None, :])
            if fz[0] > 0.2:
                xs2, ys2, _, s2, st2 = R.sample_segments(np.array([x[j]]), np.array([y[j]]), fx, fy_, 3.0, 600)
                R.soft(xs2, ys2, (warm * 0.6 + a * 0.4)[None, :] * (0.5 * (1 - s2) ** 2 * st2 * b)[:, None])
        # lens ghosts along the line through the frame centre from the brightest light
        j = int(np.argmin(np.where(ok, np.abs(x - W / 2) + np.abs(y - H * 0.15), 1e9)))
        if ok[j]:
            for k_, (f_, r_) in enumerate(((0.6, 24), (1.25, 60), (1.6, 14))):
                gx = W / 2 + (W / 2 - x[j]) * (f_ - 1) * 1.0 + (x[j] - W / 2) * (1 - f_)
                gy = H / 2 + (y[j] - H / 2) * (1 - f_)
                R.points(np.array([gx]), np.array([gy]), (a * 0.6 + 40)[None, :] * 40, np.array([r_ * self.k]), bloom=0)
        R.points3d(self.bok, self.bok_c * 60, 0.12, 0.0, bloom=0.0)
        # wall glow + studio haze
        xw, yw, _, _ = R.project(np.array([[0, 3.3, -7.5]], np.float32))
        R.soft(np.repeat(xw, 3), np.repeat(yw, 3), np.stack([a * 900, a * 450, a * 250]))
        return self.fin(canvas, soft, canvas_bloom=0.1, bloom=(0.65, 0.6), radii=(1.4, 9.0))


# ======================================================================== 6. radar
class Radar(Topical):
    """Intelligence / surveillance: a world-map dot grid on a dark table, rotating radar sweep with a
    phosphor trail, range rings, ping rings, arcing connection lines with travelling packets, red
    alert blips and a CRT scanline; the camera slowly tilts and drifts."""
    default_accent = "#19E6B4"

    def init(self):
        rng = self.rng
        self.MW = 18.0
        self.MH = self.MW * (LAT_TOP - LAT_BOT) / 360.0
        mw, mh = 230, 88
        land = world_mask(mw, mh)
        jj, ii = np.nonzero(np.ones((mh, mw), bool))
        X = ((ii + 0.5) / mw - 0.5) * self.MW
        Z = ((jj + 0.5) / mh - 0.5) * self.MH
        self.P = np.stack([X, np.zeros_like(X), Z], 1).astype(np.float32)
        self.land = land[jj, ii]
        self.ph = rng.random(len(X)).astype(np.float32)
        ex = np.abs(X) / (self.MW / 2)
        ez = np.abs(Z) / (self.MH / 2)
        self.edge = (smoothstep(1.0, 0.82, ex) * smoothstep(1.0, 0.75, ez)).astype(np.float32)

        def cxz(lo, la):
            u, v = lonlat_uv(lo, la)
            return ((u - 0.5) * self.MW, (v - 0.5) * self.MH)
        self.cxz = cxz
        self.center = np.array(cxz(25, 28), np.float32)
        d = self.P[:, [0, 2]] - self.center
        self.ang = np.arctan2(d[:, 1], d[:, 0]).astype(np.float32)
        self.rad = np.hypot(d[:, 0], d[:, 1]).astype(np.float32)
        self.cities = np.array([cxz(lo, la) for lo, la in CITIES], np.float32)
        nc = len(self.cities)
        pairs = set()
        while len(pairs) < 16:
            i, j = rng.integers(0, nc, 2)
            if i != j:
                pairs.add((int(min(i, j)), int(max(i, j))))
        self.pairs = sorted(pairs)
        self.pair_ph = rng.random(len(self.pairs)).astype(np.float32)
        self.alerts = rng.choice(nc, 5, replace=False)
        self.ping_ph = rng.random(nc).astype(np.float32)
        self.th = np.linspace(0, TAU, 160, endpoint=False).astype(np.float32)

    def ring(self, cx, cz, r, y=0.0):
        return np.stack([cx + r * np.cos(self.th), np.full_like(self.th, y), cz + r * np.sin(self.th)], 1)

    def frame(self, t):
        R, a = self.R, self.acc
        c = self.center
        tilt = 0.98 + 0.1 * math.sin(t * 0.13)
        dist = 15.5 - 1.0 * math.sin(t * 0.1)
        azim = math.pi / 2 + 0.25 * math.sin(t * 0.07)
        tgt = np.array([c[0] + 0.8 * math.sin(t * 0.09), 0.0, c[1] + 0.2])
        cpos = tgt + dist * np.array([math.cos(tilt) * math.cos(azim), math.sin(tilt), math.cos(tilt) * math.sin(azim)])
        R.begin(Camera(cpos, tgt, fov=55, roll=0.02 * math.sin(t * 0.11)))
        Hs, Ws = R.Hs, R.Ws
        yq = np.linspace(0, 1, Hs, dtype=np.float32)[:, None, None]
        soft = np.broadcast_to(lerp_col(a * 0.11 + np.array([8, 12, 16], np.float32), np.array([4, 6, 10], np.float32), np.abs(yq[..., 0] - 0.45) * 1.6),
                               (Hs, Ws, 3)).astype(np.float32).copy()
        # sweep
        sw = (t * 1.1) % TAU
        dl = (sw - self.ang) % TAU
        trail = np.exp(-dl / 0.55) * (self.rad < 7.5)
        tw = 0.8 + 0.2 * np.sin(t * 1.5 + self.ph * TAU)
        e = np.where(self.land, 1.0 * tw + 1.7 * trail, 0.12 + 0.5 * trail) * self.edge
        col = lerp_col(a, (230, 255, 245), np.clip(trail * 0.8, 0, 1) * self.land) * (e * 8)[:, None]
        x, y, z, inv = R.project(self.P)
        R.points(x, y, col * 1.4, np.clip(0.016 * inv, 0.5, 2.4), bloom=0.6)
        # sweep beam: line + soft wedge
        L = 7.5
        th = sw - np.linspace(0, 0.5, 18)
        for k, tt in enumerate(th):
            tip = np.array([[c[0] + L * math.cos(tt), 0.0, c[1] + L * math.sin(tt)]], np.float32)
            base = np.array([[c[0], 0.0, c[1]]], np.float32)
            xa, ya, _, _ = R.project(base)
            xb, yb, _, _ = R.project(tip)
            xs, ys, _, s_, st = R.sample_segments(xa, ya, xb, yb, 4.0, 400)
            R.soft(xs, ys, (a * 0.5 + 20)[None, :] * (st * 1.2 * math.exp(-k / 4.0) * (1 - s_ * 0.5))[:, None])
        tip = np.array([[c[0] + L * math.cos(sw), 0.0, c[1] + L * math.sin(sw)]], np.float32)
        R.lines3d(np.array([[c[0], 0, c[1]]], np.float32), tip, a * 2.5 + 60, a * 0.6, width=1.6, bloom=1.0)
        # range rings + crosshair
        for r in (2.5, 5.0, 7.5):
            P = self.ring(c[0], c[1], r)
            R.lines3d(P, np.roll(P, -1, 0), a * 0.35, width=1.0, bloom=0.4)
        cr = np.array([[c[0] - 7.5, 0, c[1]], [c[0] + 7.5, 0, c[1]], [c[0], 0, c[1] - 7.5], [c[0], 0, c[1] + 7.5]], np.float32)
        R.lines3d(cr[[0, 2]], cr[[1, 3]], a * 0.3, width=0.9, bloom=0.3)
        # pings
        C = self.cities
        php = (t * 0.45 + self.ping_ph) % 1.0
        for j in range(len(C)):
            if j in self.alerts:
                continue
            P = self.ring(C[j, 0], C[j, 1], 0.05 + php[j] * 0.6)
            R.lines3d(P, np.roll(P, -1, 0), a * (1 - php[j]) * 1.5, width=1.0, bloom=0.6)
        Pc = np.stack([C[:, 0], np.zeros(len(C)), C[:, 1]], 1).astype(np.float32)
        R.points3d(Pc, np.broadcast_to(a * 3 + 200, Pc.shape) * 1.5, 0.0, 1.6, bloom=1.0)
        # red alerts
        red = np.array([255, 40, 50], np.float32)
        blink = 0.5 + 0.5 * math.sin(t * 6.0)
        for k, j in enumerate(self.alerts):
            pa = (t * 0.8 + k * 0.23) % 1.0
            for r0 in (0.0, 0.5):
                pp = (pa + r0) % 1.0
                P = self.ring(C[j, 0], C[j, 1], 0.08 + pp * 0.9)
                R.lines3d(P, np.roll(P, -1, 0), red * (1 - pp) * 2.5, width=1.4, bloom=0.8)
            R.points3d(Pc[j:j + 1], red[None, :] * (6 + 14 * blink), 0.0, 3.0, bloom=1.0)
            sq = np.array([[C[j, 0] - 0.25, 0, C[j, 1] - 0.25], [C[j, 0] + 0.25, 0, C[j, 1] - 0.25], [C[j, 0] + 0.25, 0, C[j, 1] + 0.25],
                           [C[j, 0] - 0.25, 0, C[j, 1] + 0.25]], np.float32)
            R.lines3d(sq, np.roll(sq, -1, 0), red * 1.2 * (0.6 + 0.4 * blink), width=1.0, bloom=0.5)
        # connection arcs with packets
        s = np.linspace(0, 1, 40, dtype=np.float32)[:, None]
        for k, (i, j) in enumerate(self.pairs):
            A, B = C[i], C[j]
            dd = float(np.hypot(*(B - A)))
            P = np.concatenate([A[None, :] + (B - A)[None, :] * s], 0)
            Y = 4 * s * (1 - s) * (0.25 + dd * 0.22)
            P3 = np.stack([P[:, 0], Y[:, 0], P[:, 1]], 1)
            R.lines3d(P3[:-1], P3[1:], a * 0.55 + np.array([30, 30, 60]), width=1.0, bloom=0.5)
            u = (t * 0.35 + self.pair_ph[k]) % 1.0
            iu = u * 39
            i0 = int(iu)
            pk = P3[i0] + (P3[min(i0 + 1, 39)] - P3[i0]) * (iu - i0)
            R.points3d(pk[None, :].astype(np.float32), (a * 0.5 + 140)[None, :] * 6, 0.0, 2.2, bloom=1.0)
        # CRT scanline sweeping down + faint line texture
        sy = ((t * 0.22) % 1.2 - 0.1) * Hs
        band = np.exp(-((np.arange(Hs, dtype=np.float32) - sy) / 2.5) ** 2)[:, None, None]
        soft += band * (a * 0.25 + 15)
        return self.fin(None, soft, bloom=(0.6, 0.55), radii=(1.3, 7.5))


# ======================================================================== 7. capitol
class Capitol(Topical):
    """Generic neoclassical government building at dusk: stepped podium, colonnaded portico with
    pediment, wings with lit windows, drum + ribbed dome + lantern; dramatic sunset sky with light
    rays behind the dome, foreground trees and lamp posts for parallax; subtle push-in."""
    default_accent = "#FFB45A"

    def init(self):
        rng = self.rng
        self.light = _norm([-0.3, 0.2, 1.0])
        # dome profile
        self.dome_c = np.array([0.0, 0.0, -2.2])
        nl, nt = 12, 20
        phi = np.linspace(0, math.pi / 2, nl + 1)
        th = np.linspace(-math.pi * 0.55, math.pi * 0.55, nt + 1)
        rr = 1.85 * np.cos(phi) ** 0.9
        yy = 6.1 + 2.1 * np.sin(phi)
        q, nrm = [], []
        for i in range(nl):
            for j in range(nt):
                ps = []
                for (a_, b_) in ((i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)):
                    ps.append((rr[a_] * math.sin(th[b_]), yy[a_], -2.2 + rr[a_] * math.cos(th[b_])))
                q.append(ps)
                pm = (phi[i] + phi[i + 1]) / 2
                tm = (th[j] + th[j + 1]) / 2
                nrm.append((math.cos(pm) * math.sin(tm), math.sin(pm), math.cos(pm) * math.cos(tm)))
        self.dome_q = np.array(q, np.float32)
        self.dome_n = np.array(nrm, np.float32)
        self.dome_rib = (np.arange(nl * nt) % nt) % 2
        # cylinders: (cx, cz, r, y0, y1, segments) -> strip quads facing +z
        def cyl(cx, cz, r, y0, y1, n=6, span=0.55):
            th_ = np.linspace(-math.pi * span, math.pi * span, n + 1)
            qs, ns = [], []
            for k in range(n):
                a0, a1 = th_[k], th_[k + 1]
                qs.append([(cx + r * math.sin(a0), y0, cz + r * math.cos(a0)), (cx + r * math.sin(a1), y0, cz + r * math.cos(a1)),
                           (cx + r * math.sin(a1), y1, cz + r * math.cos(a1)), (cx + r * math.sin(a0), y1, cz + r * math.cos(a0))])
                am = (a0 + a1) / 2
                ns.append((math.sin(am), 0.0, math.cos(am)))
            return qs, ns
        self.cyl = cyl
        dq, dn = cyl(0.0, -2.2, 2.05, 4.3, 6.15, 22)
        self.drum = (np.array(dq, np.float32), np.array(dn, np.float32))
        dcols = []
        for k in range(15):
            ang = -1.35 + 2.7 * k / 14
            qs, ns = cyl(2.2 * math.sin(ang), -2.2 + 2.2 * math.cos(ang), 0.1, 4.3, 5.95, 4)
            dcols.append((qs, ns))
        self.drum_cols = dcols
        lq, ln = cyl(0.0, -2.2, 0.38, 8.1, 8.9, 10)
        self.lantern = (np.array(lq, np.float32), np.array(ln, np.float32))
        cols = []
        for k in range(8):
            x = -2.45 + k * 4.9 / 7
            qs, ns = cyl(x, 0.55, 0.2, 1.35, 3.75, 6)
            cols.append((np.array(qs, np.float32), np.array(ns, np.float32)))
        self.cols = cols
        # windows on the wings (front face z = -0.4)
        wx = [x for x in np.arange(-8.6, 8.7, 0.62) if abs(x) > 3.2]
        W_ = []
        for x in wx:
            for y in (1.9, 2.75):
                W_.append((x, y, -0.39))
        self.win = np.array(W_, np.float32)
        self.win_c = np.where((rng.random(len(W_)) < 0.75)[:, None], np.array([255, 190, 110], np.float32), np.array([120, 110, 100], np.float32))
        # foreground trees (blobby silhouettes) & lamp posts
        trees = []
        for side in (-1, 1):
            for k in range(5):
                x = side * rng.uniform(3.6, 5.2)
                z = rng.uniform(7.0, 12.5)
                h = rng.uniform(2.5, 4.5)
                pts = []
                n = 22
                for j in range(n):
                    ang = j / n * TAU
                    r = (1.2 + 0.4 * rng.random()) * (1.0 if math.sin(ang) > -0.3 else 0.75)
                    pts.append((x + r * math.cos(ang) * 1.1, h + r * math.sin(ang) * 1.05, z))
                trees.append((np.array(pts, np.float32), np.array([[x, 0, z], [x, h, z]], np.float32)))
        self.trees = trees
        lamps = []
        for side in (-1, 1):
            for z in (2.0, 4.5, 7.0, 9.5, 12.0, 14.5):
                lamps.append((side * 3.3, z))
        self.lamps = lamps
        self.ray_ang = rng.uniform(0, TAU, 26)
        self.ray_w = rng.uniform(0.02, 0.08, 26)
        self.ray_b = rng.uniform(0.3, 1.0, 26)

    def frame(self, t):
        R, a = self.R, self.acc
        push = 3.5 * (1 - math.exp(-t / 12.0)) + t * 0.05
        cpos = np.array([1.8 * math.sin(t * 0.09 + 0.4), 1.7 + 0.15 * math.sin(t * 0.13), 17.0 - push])
        tgt = np.array([0.0, 3.3, -2.0])
        R.begin(Camera(cpos, tgt, fov=54, roll=0.0))
        W, H = self.W, self.H
        _, hy, _, _ = R.project(np.array([[0.0, 0.0, -3000.0]]))
        hy = float(hy[0])
        key = int(hy)
        if getattr(self, "_sky_key", None) != key:
            yy = np.arange(H, dtype=np.float32)
            u = np.clip(yy / max(hy, 1.0), 0, 1)
            stops_x = [0.0, 0.45, 0.75, 0.92, 1.0]
            stops = [(18, 26, 66), (52, 46, 110), (150, 80, 120), (245, 130, 80), (255, 190, 120)]
            sky = np.stack([np.interp(u, stops_x, [c_[i] for c_ in stops]) for i in range(3)], 1)
            sky = np.where((yy > hy)[:, None], lerp_col((92, 64, 70), (30, 24, 34), np.clip((yy - hy) / (H * 0.35), 0, 1) ** 0.7), sky)
            self._sky = Image.fromarray(np.ascontiguousarray(np.broadcast_to(np.clip(sky, 0, 255).astype(np.uint8)[:, None, :], (H, W, 3))))
            self._sky_key = key
        canvas = self._sky.copy()
        d = ImageDraw.Draw(canvas)
        Ld = self.light
        warm = np.array([255, 205, 160], np.float32)

        def shade(n, base, amb=0.32, diff=0.75):
            k = amb + diff * np.clip(np.asarray(n, np.float32) @ Ld, 0, 1)
            return np.asarray(base, np.float32) * np.atleast_1d(k)[:, None]

        def draw_q(Q, cols, outline=None):
            polys, _ = self.quads(np.asarray(Q, np.float32))
            for pts, c_ in zip(polys, cols):
                if pts:
                    d.polygon(pts, fill=tint(c_), outline=outline)

        def box(b, base, top_base=None):
            for quad, n in self.box_faces(b, cpos):
                c_ = shade(n, top_base if (top_base is not None and n[1] > 0) else base)[0]
                draw_q([quad], [c_])

        # sun glow + rays behind the dome (soft, under the opaque canvas -> only visible in the sky)
        sx, sy, _, _ = R.project(np.array([[1.2, 6.8, -40.0]], np.float32))
        sx, sy = float(sx[0]), float(sy[0])
        stone = np.array([104, 92, 86], np.float32)
        # dome group (back)
        draw_q(self.lantern[0], shade(self.lantern[1], stone * 1.1))
        lt = np.array([(0, 8.9, -2.2)], np.float32)
        dq, dn = self.dome_q, self.dome_n
        dc = shade(dn, np.array([100, 102, 118], np.float32)) * (1 - 0.12 * self.dome_rib)[:, None]
        draw_q(dq, dc)
        draw_q(self.drum[0], shade(self.drum[1], stone * 0.8))
        for qs, ns in self.drum_cols:
            draw_q(qs, shade(np.array(ns), stone * 1.25))
        box((-2.6, 2.6, 3.9, 4.35, -4.4, -0.2), stone * 0.9)  # drum base
        # wings + central block
        box((-9.0, 9.0, 1.2, 3.4, -4.0, -0.4), stone * 0.85, stone * 0.6)
        box((-9.2, 9.2, 3.4, 3.7, -4.1, -0.3), stone * 0.95, stone * 0.6)
        box((-3.1, 3.1, 1.2, 3.9, -4.0, 0.1), stone * 0.5, stone * 0.6)
        for xd in np.linspace(-2.1, 2.1, 4):  # lit doorways in the recess
            draw_q([[(xd - 0.25, 1.25, 0.11), (xd + 0.25, 1.25, 0.11), (xd + 0.25, 2.5, 0.11), (xd - 0.25, 2.5, 0.11)]], [np.array([110, 72, 40])])
        # podium + steps
        box((-9.4, 9.4, 0.0, 1.2, -4.4, -0.2), stone * 0.75, stone * 0.8)
        for k in range(6):
            box((-3.4 - k * 0.12, 3.4 + k * 0.12, 0.0, 1.2 - k * 0.2, 0.0, 1.0 + k * 0.38), stone * (0.8 + 0.03 * k), stone * 1.05)
        # portico columns, entablature, pediment
        for qs, ns in self.cols:
            draw_q(qs, shade(ns, stone * 1.3, 0.35, 0.8))
        box((-2.85, 2.85, 3.75, 4.25, -0.2, 0.85), stone * 1.05, stone * 0.7)
        ped = np.array([(-2.95, 4.25, 0.86), (2.95, 4.25, 0.86), (0.0, 5.25, 0.86)], np.float32)
        pv = self.to_cam(ped)
        pts = self.screen_poly(pv)
        if pts:
            d.polygon(pts, fill=tint(stone * 1.0), outline=tint(stone * 1.3))
        inner = np.array([(-2.4, 4.4, 0.87), (2.4, 4.4, 0.87), (0.0, 5.0, 0.87)], np.float32)
        pts = self.screen_poly(self.to_cam(inner))
        if pts:
            d.polygon(pts, fill=tint(stone * 0.75))
        # reflecting pool: mirrored sky gradient strip on the ground
        nb_ = 10
        for k in range(nb_):  # banded mirrored-sky gradient: bright (horizon) far, dark near
            z0_, z1_ = 3.4 + 7.6 * k / nb_, 3.4 + 7.6 * (k + 1) / nb_
            q_ = np.array([(-2.0, 0.02, z0_), (2.0, 0.02, z0_), (2.0, 0.02, z1_), (-2.0, 0.02, z1_)], np.float32)
            pts = self.screen_poly(self.to_cam(q_))
            if pts:
                f_ = 1 - k / nb_
                d.polygon(pts, fill=tint(lerp_col((30, 28, 70), (200, 110, 100), f_ ** 2.2)))
        # foreground lamp posts + trees (dark silhouettes, nearest last)
        for (x, z) in sorted(self.lamps, key=lambda p_: -abs(cpos[2] - p_[1])):
            pp = self.to_cam(np.array([(x - 0.035, 0, z), (x + 0.035, 0, z), (x + 0.035, 2.6, z), (x - 0.035, 2.6, z)]))
            pts = self.screen_poly(pp)
            if pts:
                d.polygon(pts, fill=(14, 12, 18))
        for poly, trunk in sorted(self.trees, key=lambda tr: -abs(cpos[2] - tr[1][0, 2])):
            pts = self.screen_poly(self.to_cam(np.vstack([trunk[0] + [-0.12, 0, 0], trunk[0] + [0.12, 0, 0], trunk[1] + [0.12, 0, 0], trunk[1] + [-0.12, 0, 0]])))
            if pts:
                d.polygon(pts, fill=(8, 8, 12))
            pts = self.screen_poly(self.to_cam(poly))
            if pts:
                d.polygon(pts, fill=(10, 10, 16))
        # ---- additive light
        x, y, z, inv = R.project(self.win)
        R.points(x, y, self.win_c * 14.0, np.clip(0.05 * inv, 0.8, 4.0), bloom=0.7)
        # uplights washing the facade
        ups = np.array([(xx, 0.3, 1.6) for xx in np.linspace(-2.8, 2.8, 8)] + [(xx, 1.3, 0.0) for xx in (-8, -6, -4, 4, 6, 8)], np.float32)
        x, y, _, _ = R.project(ups)
        R.soft(x, y - 60 * self.k, warm * 700)
        hx_ = np.linspace(-200, W + 200, 40)
        R.soft(hx_, np.full(40, hy + 10.0), (warm * 0.5 + a * 0.3) * 120)
        # dome rim light in accent + lantern glow
        xl, yl, _, _ = R.project(lt)
        R.points(np.repeat(xl, 2), np.repeat(yl, 2), np.stack([warm * 50, a * 700]), np.array([2.5, 22.0]) * self.k, bloom=1.0)
        rim_th = np.linspace(-math.pi * 0.5, math.pi * 0.5, 80)
        for ph_, yv in ((0.0, 6.12),):
            Pm = np.stack([1.9 * np.sin(rim_th), np.full(80, yv), -2.2 + 1.9 * np.cos(rim_th)], 1).astype(np.float32)
            R.lines3d(Pm[:-1], Pm[1:], a * 0.9, width=1.4, bloom=0.8)
        prof = np.linspace(0, math.pi / 2, 40)
        for side in (-1, 1):
            Pp = np.stack([side * 1.87 * np.cos(prof) ** 0.9, 6.1 + 2.1 * np.sin(prof), np.full(40, -2.2)], 1).astype(np.float32)
            R.lines3d(Pp[:-1], Pp[1:], a * 1.4 + 30, width=1.6, bloom=1.0)
        # god rays from the sun behind the dome (soft buffer, rotating very slowly)
        rays_x, rays_y, rays_c = [], [], []
        for k in range(len(self.ray_ang)):
            ang = self.ray_ang[k] + 0.02 * math.sin(t * 0.2 + k)
            Lr = H * 0.9
            n = 160
            sr = np.linspace(0, 1, n, dtype=np.float32)
            rays_x.append(sx + np.cos(ang) * sr * Lr)
            rays_y.append(sy + np.sin(ang) * sr * Lr)
            rays_c.append(np.outer((1 - sr) ** 2 * self.ray_b[k] * (0.7 + 0.3 * math.sin(t * 0.5 + k)), a * 0.5 + warm * 0.5) * 2.0)
        R.soft(np.concatenate(rays_x), np.concatenate(rays_y), np.concatenate(rays_c))
        R.soft(np.repeat(sx, 2), np.repeat(sy, 2), np.stack([warm * 1500, a * 900]))
        # lamp lights
        LP = np.array([(x, 2.65, z) for x, z in self.lamps], np.float32)
        R.points3d(LP, np.broadcast_to(warm * 400, LP.shape), 0.0, 3.0, bloom=1.0)
        R.points3d(LP, np.broadcast_to(warm * 4000, LP.shape), 0.0, 14.0, bloom=0.0)
        return self.fin(canvas, None, canvas_bloom=0.12, bloom=(0.6, 0.6), radii=(1.4, 9.0))


# ======================================================================== 8. datacenter
class DataCenter(Topical):
    """Server-hall aisle in true perspective: rack fronts with server units and blinking LEDs, ceiling
    light strips, cable trays, perforated floor with cold-blue reflections and haze; the camera glides
    down the aisle forever (the rack pattern is periodic, so it is seamless)."""
    default_accent = "#3FB8FF"

    def init(self):
        rng = self.rng
        self.pitch = 0.72  # rack width along the aisle
        self.NP = 40  # pattern period (racks)
        self.xr = 1.25  # rack front plane distance from the aisle centre
        self.hr = 2.35
        self.nu = 20  # server units per rack
        nl = 4
        # LED table per (side, rack mod NP, unit, led): local position + colour + blink
        side, rk, un, ld = np.meshgrid(np.arange(2), np.arange(self.NP), np.arange(self.nu), np.arange(nl), indexing="ij")
        side, rk, un, ld = side.ravel(), rk.ravel(), un.ravel(), ld.ravel()
        self.l_side, self.l_rk = side, rk
        self.l_y = (0.12 + (un + 0.5) * (self.hr - 0.2) / self.nu).astype(np.float32)
        self.l_u = (0.08 + ld * 0.05 + (ld >= 2) * 0.35).astype(np.float32)  # offset along the rack width
        pal = np.array([[60, 180, 255], [80, 255, 160], [60, 180, 255], [255, 170, 60]], np.float32)
        pick = rng.choice(4, len(side), p=[0.45, 0.3, 0.15, 0.10])
        self.l_c = lerp_col(pal[pick], self.acc, 0.35 * (pick == 0))
        self.l_f = rng.uniform(0.5, 6.0, len(side)).astype(np.float32)
        self.l_ph = rng.random(len(side)).astype(np.float32)
        self.l_on = rng.random(len(side)) < 0.8
        self.l_steady = rng.random(len(side)) < 0.45
        self.unit_kind = rng.integers(0, 3, (2, self.NP, self.nu))

    def frame(self, t):
        R, a = self.R, self.acc
        v = 1.3
        cz = -t * v
        sway = 0.08 * math.sin(t * 0.4)
        cpos = np.array([sway, 1.55 + 0.05 * math.sin(t * 0.7), cz])
        tgt = np.array([sway * 0.5 + 0.12 * math.sin(t * 0.13), 1.45, cz - 10.0])
        R.begin(Camera(cpos, tgt, fov=66, roll=0.01 * math.sin(t * 0.3)))
        W, H = self.W, self.H
        canvas = Image.new("RGB", (W, H), (4, 6, 12))
        d = ImageDraw.Draw(canvas)
        far = 24.0
        p = self.pitch
        xr, hr = self.xr, self.hr
        k0 = int(math.floor(-cz / p)) - 1
        ks = np.arange(k0, k0 + int(far / p) + 2)
        cold = np.array([150, 200, 255], np.float32)
        fogc = np.array([12, 20, 36], np.float32)
        # floor / ceiling / walls as big quads (painter: far parts first doesn't matter, they don't overlap)
        def quad(q, c):
            pts = self.screen_poly(self.to_cam(np.array(q)))
            if pts:
                d.polygon(pts, fill=tint(c))
        zf = cz - far
        quad([(-xr, 0, cz + 1), (xr, 0, cz + 1), (xr, 0, zf), (-xr, 0, zf)], (14, 18, 28))
        quad([(-xr - 0.5, 3.1, cz + 1), (xr + 0.5, 3.1, cz + 1), (xr + 0.5, 3.1, zf), (-xr - 0.5, 3.1, zf)], (8, 10, 16))
        for sd in (-1, 1):  # wall above racks
            quad([(sd * xr, hr, cz + 1), (sd * (xr + 0.5), 3.1, cz + 1), (sd * (xr + 0.5), 3.1, zf), (sd * xr, hr, zf)], (10, 12, 20))
        # rack fronts, far to near
        for k in ks[::-1].tolist():
            z0 = -k * p
            z1 = z0 - p + 0.03
            dist = cz - (z0 + z1) / 2
            if dist < -1.0:
                continue
            fog = math.exp(-max(0.0, dist) / 15.0)
            for sd in (-1, 1):
                body = np.array([9, 11, 17], np.float32) * fog + fogc * (1 - fog)
                q = [(sd * xr, 0.02, z0), (sd * xr, 0.02, z1), (sd * xr, hr, z1), (sd * xr, hr, z0)]
                V = self.to_cam(np.array(q))
                pts = self.screen_poly(V)
                if not pts:
                    continue
                d.polygon(pts, fill=tint(body), outline=tint(body * 1.8 + 6))
                if dist < 12:  # server unit seams
                    A = np.array([(sd * xr * 0.999, 0.12 + u * (hr - 0.2) / self.nu, z0 - 0.02) for u in range(1, self.nu)])
                    B = A.copy()
                    B[:, 2] = z1 + 0.02
                    VA, VB = self.to_cam(A), self.to_cam(B)
                    ok = (VA[:, 2] > 0.06) & (VB[:, 2] > 0.06)
                    f = R.f
                    lc = tint(body * 0.45)
                    for u in np.nonzero(ok)[0].tolist():
                        d.line([(W * 0.5 + VA[u, 0] * f / VA[u, 2], H * 0.5 - VA[u, 1] * f / VA[u, 2]),
                                (W * 0.5 + VB[u, 0] * f / VB[u, 2], H * 0.5 - VB[u, 1] * f / VB[u, 2])], fill=lc, width=1)
        # ---- LEDs (additive), on all racks in range
        kk = (ks % self.NP)
        sel = np.isin(self.l_rk, kk)
        rk = self.l_rk[sel]
        # map pattern index -> actual rack index k (unique within range since range < NP*... take nearest)
        kmap = np.full(self.NP, -1)
        for k in ks.tolist():
            if kmap[k % self.NP] < 0:
                kmap[k % self.NP] = k
        kr = kmap[rk]
        sd = np.where(self.l_side[sel] == 0, -1.0, 1.0)
        z = -kr * p - self.l_u[sel]
        P = np.stack([sd * (xr - 0.01), self.l_y[sel], z], 1).astype(np.float32)
        blink = np.where(self.l_steady[sel], 1.0, (np.sin(t * self.l_f[sel] * TAU * 0.5 + self.l_ph[sel] * TAU) > 0.2).astype(np.float32))
        on = self.l_on[sel] * blink
        dist = cz - z
        fog = np.exp(-np.maximum(0, dist) / 12.0)
        col = self.l_c[sel] * (on * fog * 14.0)[:, None]
        m = on > 0
        x, y, zz, inv = R.project(P[m])
        R.points(x, y, col[m] * (1 + np.clip(0.012 * inv, 0.4, 2.2) ** 2)[:, None], np.clip(0.012 * inv, 0.4, 2.2), bloom=0.7)
        # floor reflections of LEDs
        Pr = P[m].copy()
        Pr[:, 1] = -Pr[:, 1]
        xr_, yr_, _, _ = R.project(Pr)
        R.points(xr_, yr_, col[m] * 0.18, None, bloom=0.0)
        R.soft(xr_, yr_, col[m] * 0.45)
        # ceiling light strips (segmented) + reflections
        zs = -np.arange(k0, k0 + int(far / p) + 2) * p
        for sx in (-0.55, 0.55):
            A = np.stack([np.full(len(zs), sx), np.full(len(zs), 3.08), zs - 0.05], 1).astype(np.float32)
            B = A.copy()
            B[:, 2] = zs - p + 0.12
            fg = np.exp(-np.maximum(0, cz - zs) / 14.0)[:, None]
            R.lines3d(A, B, cold * 2.2 * fg, width=3.0, bloom=0.9)
            Ar, Br = A.copy(), B.copy()
            Ar[:, 1] *= -1
            Br[:, 1] *= -1
            R.lines3d(Ar, Br, cold * 0.35 * fg, width=4.0, bloom=0.8)
        # far end of the aisle: bright cold doorway
        quad([(-xr, 0, zf + 0.01), (xr, 0, zf + 0.01), (xr, 3.1, zf + 0.01), (-xr, 3.1, zf + 0.01)], cold * 0.32 + a * 0.05)
        # glowing accent door strips on every rack
        zs2 = -ks * p - p * 0.5
        dd = cz - zs2
        okz = dd > -0.5
        fg2 = np.exp(-np.maximum(0, dd[okz]) / 14.0)[:, None]
        for sd in (-1, 1):
            A = np.stack([np.full(okz.sum(), sd * (xr - 0.012)), np.full(okz.sum(), 0.25), zs2[okz]], 1).astype(np.float32)
            B = A.copy()
            B[:, 1] = hr - 0.15
            R.lines3d(A, B, a * 0.9 * fg2, width=1.2, bloom=0.8)
        # cable trays above racks (accent tint)
        for sd in (-1, 1):
            A = np.array([[sd * (xr - 0.15), 2.62, cz + 0.5]], np.float32)
            B = np.array([[sd * (xr - 0.15), 2.62, cz - far]], np.float32)
            R.lines3d(A, B, a * 0.9, a * 0.05, width=1.5, bloom=0.5)
        # floor tile seams
        zt = np.arange(math.floor(cz / 0.6) * 0.6, cz - far, -0.6)
        A = np.stack([np.full(len(zt), -xr), np.zeros(len(zt)), zt], 1).astype(np.float32)
        B = A.copy()
        B[:, 0] = xr
        fg = np.exp(-np.maximum(0, cz - zt) / 10.0)[:, None]
        R.lines3d(A, B, cold * 0.10 * fg, width=1.0, bloom=0.2)
        xs = np.arange(-xr + 0.6, xr - 0.1, 0.6)
        A = np.stack([xs, np.zeros(len(xs)), np.full(len(xs), cz + 0.3)], 1).astype(np.float32)
        B = A.copy()
        B[:, 2] = cz - far
        R.lines3d(A, B, cold * 0.10, cold * 0.0, width=1.0, bloom=0.2)
        # haze at the vanishing point
        vx, vy, _, _ = R.project(np.array([[0, 1.5, cz - far]], np.float32))
        R.soft(np.repeat(vx, 3), np.repeat(vy, 3), np.stack([cold * 600, a * 500, cold * 250]))
        yq = np.linspace(0, 1, R.Hs, dtype=np.float32)[:, None, None]
        soft = np.broadcast_to(np.exp(-((yq - 0.5) / 0.25) ** 2) * (a * 0.04 + cold * 0.04), (R.Hs, R.Ws, 3)).astype(np.float32)
        return self.fin(canvas, soft, canvas_bloom=0.1, bloom=(0.65, 0.6), radii=(1.3, 8.0))


# ======================================================================== 9. calendar
def perspective_coeffs(dst, src):
    """PIL PERSPECTIVE coefficients mapping output (dst) quad corners to input (src) corners."""
    A, B = [], []
    for (x, y), (u, v) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        B += [u, v]
    try:
        return np.linalg.solve(np.array(A, np.float64), np.array(B, np.float64)).tolist()
    except np.linalg.LinAlgError:
        return None


class Calendar(Topical):
    """Tear-off calendar day cards (numbers 1-31 only) tumbling and drifting through depth with
    depth-of-field (pre-blurred mip levels), a hero card showing the ``highlight`` day in the accent
    colour, and a slowly turning clock-face ring of glowing ticks behind; gentle camera drift."""
    default_accent = "#FF4D5E"
    CW, CH = 240, 290

    def _card(self, day, hi):
        CW, CH = self.CW, self.CH
        im = Image.new("RGBA", (CW, CH), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        paper = (242, 240, 234, 255) if not hi else (250, 250, 250, 255)
        d.rounded_rectangle([2, 2, CW - 3, CH - 3], radius=18, fill=paper)
        band = tuple(int(v) for v in (self.acc if hi else np.array([44, 48, 62]))) + (255,)
        d.rounded_rectangle([2, 2, CW - 3, 74], radius=18, fill=band)
        d.rectangle([2, 50, CW - 3, 74], fill=band)
        for hx in (CW * 0.3, CW * 0.7):
            d.ellipse([hx - 9, 22, hx + 9, 40], fill=(20, 20, 26, 255))
        f = font(170 if day < 10 else 150)
        txt = str(day)
        l, t_, r, b = f.getbbox(txt)
        col = tuple(int(v) for v in (self.acc * 0.85 if hi else np.array([34, 36, 46]))) + (255,)
        d.text(((CW - (r - l)) / 2 - l, 74 + (CH - 74 - (b - t_)) / 2 - t_), txt, fill=col, font=f)
        d.line([(16, CH - 30), (CW - 16, CH - 30)], fill=(210, 206, 198, 255), width=2)
        return im

    def init(self):
        rng = self.rng
        self.hi = int(self.p.get("highlight", self.p.get("day", 0)) or 0)
        if not 1 <= self.hi <= 31:
            self.hi = int(rng.integers(1, 32))
        self.tex = {}
        for day in range(1, 32):
            im = self._card(day, False)
            self.tex[day] = [im, im.filter(ImageFilter.GaussianBlur(3)), im.filter(ImageFilter.GaussianBlur(8))]
        h = self._card(self.hi, True)
        self.tex_hi = [h, h.filter(ImageFilter.GaussianBlur(3)), h.filter(ImageFilter.GaussianBlur(8))]
        back = Image.new("RGBA", (self.CW, self.CH), (0, 0, 0, 0))
        ImageDraw.Draw(back).rounded_rectangle([2, 2, self.CW - 3, self.CH - 3], radius=18, fill=(200, 196, 190, 255))
        self.back = [back, back.filter(ImageFilter.GaussianBlur(3)), back.filter(ImageFilter.GaussianBlur(8))]
        n = 18
        self.n = n
        self.cx = rng.uniform(-3.4, 3.4, n)
        self.cz = rng.uniform(-9.0, 0.3, n)
        self.cy0 = rng.uniform(0, 1, n)
        self.vy = rng.uniform(0.25, 0.5, n)
        self.rx0, self.ry0, self.rz0 = (rng.uniform(0, TAU, n) for _ in range(3))
        self.wx, self.wy = rng.uniform(0.2, 0.7, n), rng.uniform(0.15, 0.6, n)
        days = [d for d in range(1, 32) if d != self.hi]
        self.day = rng.choice(days, n)
        self.span = 14.0
        ticks = []
        for k in range(60):
            ang = k / 60 * TAU
            r0, r1 = (4.6, 5.3) if k % 5 == 0 else (4.95, 5.25)
            ticks.append(((0.8 * r0 * math.cos(ang), 0.8 * r0 * math.sin(ang)), (0.8 * r1 * math.cos(ang), 0.8 * r1 * math.sin(ang)), k % 5 == 0))
        self.ticks = ticks
        nb = 60
        self.bok = np.stack([rng.uniform(-7, 7, nb), rng.uniform(-6, 8, nb), rng.uniform(-16, -10, nb)], 1).astype(np.float32)
        self.bok_c = lerp_col(self.acc, (255, 220, 190), rng.random(nb)) * (0.2 + 0.6 * rng.random(nb))[:, None]
        self.bg = Image.fromarray(vgrad(self.W, self.H, [(0, (14, 12, 22)), (0.5, (30, 22, 36)), (1, (10, 10, 18))]))

    def _draw_card(self, canvas, Pc, n_world, tex3, light, sharp=False):
        """Pc: 4 world corners (TL, TR, BR, BL)."""
        R = self.R
        cam = R.cam
        x, y, z, _ = R.project(Pc)
        if (z < 0.3).any():
            return
        front = float(np.dot(n_world, cam.pos - Pc.mean(0))) > 0
        zc = float(z.mean())
        blur = abs(zc - 9.0) / 9.0
        lvl = 0 if (sharp or blur < 0.25) else (1 if blur < 0.6 else 2)
        img = tex3[lvl] if front else self.back[lvl]
        if not front:  # mirror so the back maps the right way round
            x, y = x[[1, 0, 3, 2]], y[[1, 0, 3, 2]]
        x0, y0 = int(math.floor(x.min())), int(math.floor(y.min()))
        x1, y1 = int(math.ceil(x.max())) + 1, int(math.ceil(y.max())) + 1
        if x1 < 0 or y1 < 0 or x0 > self.W or y0 > self.H or (x1 - x0) * (y1 - y0) > self.W * self.H or x1 - x0 < 2 or y1 - y0 < 2:
            return
        CW, CH = self.CW, self.CH
        co = perspective_coeffs([(x[i] - x0, y[i] - y0) for i in range(4)], [(0, 0), (CW, 0), (CW, CH), (0, CH)])
        if co is None:
            return
        warped = img.transform((x1 - x0, y1 - y0), Image.PERSPECTIVE, co, Image.BILINEAR)
        sh = float(np.clip(0.45 + 0.6 * abs(float(np.dot(n_world, light))), 0, 1)) * (0.55 + 0.45 * math.exp(-max(0.0, zc - 6) / 10.0))
        alpha = warped.getchannel("A")
        canvas.paste(warped, (x0, y0), alpha)
        if sh < 0.97:
            k_ = 1 - sh
            canvas.paste((8, 6, 14), (x0, y0, x1, y1), alpha.point([int(v * k_) for v in range(256)]))

    def frame(self, t):
        R, a = self.R, self.acc
        cpos = np.array([0.6 * math.sin(t * 0.13), 0.3 * math.sin(t * 0.11), 9.0 + 0.4 * math.sin(t * 0.09)])
        R.begin(Camera(cpos, (0.2 * math.sin(t * 0.1), 0.0, 0.0), fov=52, roll=0.03 * math.sin(t * 0.12)))
        canvas = self.bg.copy()
        light = _norm([0.3, 0.6, 0.8])
        cards = []
        for i in range(self.n):
            y = 7.0 - ((self.cy0[i] * self.span + self.vy[i] * t) % self.span)
            ang = (0.9 * math.sin(self.rx0[i] + self.wx[i] * t), 0.8 * math.sin(self.ry0[i] + self.wy[i] * t), (self.rz0[i] - math.pi) * 0.25 + 0.2 * math.sin(t * 0.3 + i))
            cards.append((np.array([self.cx[i], y, self.cz[i]]), ang, 0.55, self.tex[int(self.day[i])]))
        # hero card: lower third, swinging gently
        hero_ang = (0.12 * math.sin(t * 0.6), 0.35 * math.sin(t * 0.45), 0.04 * math.sin(t * 0.5))
        cards.append((np.array([0.0, -1.75, 2.6]), hero_ang, 1.0, self.tex_hi))
        order = sorted(range(len(cards)), key=lambda i: -float(np.linalg.norm(cards[i][0] - cpos)))
        hw = self.CW / self.CH
        for i in order:
            c, (ax, ay, az), sz, tex3 = cards[i]
            M = rot_y(ay) @ rot_x(ax) @ rot_z(az)
            loc = np.array([[-hw, 1, 0], [hw, 1, 0], [hw, -1, 0], [-hw, -1, 0]]) * sz
            Pc = (loc @ M.T + c).astype(np.float32)
            n_world = M @ np.array([0, 0, 1.0])
            self._draw_card(canvas, Pc, n_world, tex3, light, sharp=tex3 is self.tex_hi)
        # clock-face ring behind (additive)
        rot = t * 0.05
        A, B, C = [], [], []
        for (p0, p1, major) in self.ticks:
            c_, s_ = math.cos(rot), math.sin(rot)
            A.append((p0[0] * c_ - p0[1] * s_, p0[1] * c_ + p0[0] * s_ + 2.2, -12.0))
            B.append((p1[0] * c_ - p1[1] * s_, p1[1] * c_ + p1[0] * s_ + 2.2, -12.0))
            C.append(a * (1.6 if major else 0.7) + 25)
        R.lines3d(np.array(A, np.float32), np.array(B, np.float32), np.array(C, np.float32), width=2.2, bloom=0.8)
        th = np.linspace(0, TAU, 240, endpoint=False)
        ring = np.stack([4.5 * np.cos(th), 4.5 * np.sin(th) + 2.2, np.full(240, -12.0)], 1).astype(np.float32)
        R.lines3d(ring, np.roll(ring, -1, 0), a * 0.5, width=1.0, bloom=0.6)
        # sweeping hand + trail
        sw = -t * 0.6 + 1.2
        for k in range(14):
            ang = sw + k * 0.04
            tip = np.array([[3.5 * math.cos(ang), 3.5 * math.sin(ang) + 2.2, -12.0]], np.float32)
            R.lines3d(np.array([[0, 2.2, -12.0]], np.float32), tip, a * (2.0 if k == 0 else 0.5 * math.exp(-k / 4)), width=1.6, bloom=0.8)
        cx, cy, _, _ = R.project(np.array([[0, 2.2, -12.0]], np.float32))
        R.soft(np.repeat(cx, 3), np.repeat(cy, 3), np.stack([a * 900, a * 400, np.array([255, 230, 210], np.float32) * 100]))
        hx, hy, _, _ = R.project(np.array([[0, -1.75, 2.4]], np.float32))
        R.soft(np.repeat(hx, 2), np.repeat(hy, 2), np.stack([a * 700, a * 300]))
        R.points3d(self.bok, self.bok_c * 60, 0.12, 0.0, bloom=0.0)
        return self.fin(canvas, None, canvas_bloom=0.12, bloom=(0.6, 0.55), radii=(1.4, 9.0))


# ======================================================================== 10. courtroom
class Courtroom(Topical):
    """Courtroom: raised wooden judge's bench with panelling, a gavel resting on its sound block in
    the foreground, tall windows throwing warm light shafts through dusty air; slow push-in."""
    default_accent = "#FFB45A"

    def init(self):
        rng = self.rng
        nd = 500
        self.dust = np.stack([rng.uniform(-4, 4, nd), rng.uniform(0, 5, nd), rng.uniform(-5, 4, nd)], 1).astype(np.float32)
        self.dust_ph = rng.random(nd).astype(np.float32)
        self.bg = Image.fromarray(vgrad(self.W, self.H, [(0, (26, 16, 12)), (0.55, (48, 30, 20)), (1, (18, 12, 10))]))

    def frame(self, t):
        R, a = self.R, self.acc
        cpos = np.array([0.6 * math.sin(t * 0.1), 3.1 + 0.05 * math.sin(t * 0.2), 7.8 - 2.0 * (1 - math.exp(-t / 10.0))])
        tgt = np.array([0.0, 1.7, -2.0])
        R.begin(Camera(cpos, tgt, fov=55, roll=0.0))
        canvas = self.bg.copy()
        d = ImageDraw.Draw(canvas)
        light = _norm([-0.5, 0.6, 0.7])
        wood = np.array([96, 54, 30], np.float32)
        dark = np.array([46, 26, 16], np.float32)

        def q(pts, c, outline=None):
            sp = self.screen_poly(self.to_cam(np.array(pts, np.float64)))
            if sp:
                d.polygon(sp, fill=tint(c), outline=None if outline is None else tint(outline))

        def box(b, base, top=None):
            for quad, n in self.box_faces(b, cpos):
                c = self.shade(top if (top is not None and n[1] > 0) else base, n, light, cpos - np.mean(quad, 0), 0.45, 0.6, 0.15, 12)
                q(quad, c)

        # back wall panelling + tall windows
        q([(-8, 0, -5), (8, 0, -5), (8, 7, -5), (-8, 7, -5)], (40, 26, 18))
        for wx in (-2.3, 2.3):
            q([(wx - 0.8, 2.4, -4.98), (wx + 0.8, 2.4, -4.98), (wx + 0.8, 6.0, -4.98), (wx - 0.8, 6.0, -4.98)], (170, 128, 84))
            q([(wx - 0.04, 2.4, -4.97), (wx + 0.04, 2.4, -4.97), (wx + 0.04, 6.0, -4.97), (wx - 0.04, 6.0, -4.97)], (60, 40, 26))
            for yb in (3.6, 4.8):
                q([(wx - 0.8, yb - 0.04, -4.97), (wx + 0.8, yb - 0.04, -4.97), (wx + 0.8, yb + 0.04, -4.97), (wx - 0.8, yb + 0.04, -4.97)], (60, 40, 26))
        for px in np.arange(-7.5, 7.6, 1.5):
            q([(px - 0.6, 0.3, -4.99), (px + 0.6, 0.3, -4.99), (px + 0.6, 2.0, -4.99), (px - 0.6, 2.0, -4.99)], (52, 32, 20), (70, 44, 28))
        # floor
        q([(-8, 0, -5), (8, 0, -5), (8, 0, 8), (-8, 0, 8)], (34, 22, 16))
        # judge's bench: raised dais + front with panels + top
        box((-3.2, 3.2, 0.0, 0.35, -4.4, -2.2), dark * 1.2, wood * 0.8)
        box((-2.8, 2.8, 0.35, 2.05, -3.2, -2.5), wood, wood * 1.2)
        box((-2.95, 2.95, 2.05, 2.15, -3.3, -2.4), wood * 1.15, wood * 1.4)
        for px in np.linspace(-2.2, 2.2, 5):
            q([(px - 0.42, 0.6, -2.49), (px + 0.42, 0.6, -2.49), (px + 0.42, 1.8, -2.49), (px - 0.42, 1.8, -2.49)], wood * 0.75, wood * 1.25)
        # chair back behind the bench
        box((-0.55, 0.55, 2.15, 3.3, -4.0, -3.8), dark * 0.9)
        # foreground: sound block + gavel on a table edge
        box((-2.2, 2.2, 0.0, 0.9, 2.3, 4.6), dark * 1.1, wood * 0.75)
        box((-0.45, 0.45, 0.9, 1.02, 3.0, 3.7), wood * 1.1, wood * 1.35)
        cyl_c = np.array([-0.1, 1.2, 3.35])
        # handle
        q([(-0.1, 1.15, 3.35), (-0.1, 1.27, 3.35), (1.3, 1.08, 4.1), (1.3, 0.98, 4.1)], wood * 1.05)
        # gavel head: cylinder along x (approx: 8 side strips)
        th = np.linspace(0, TAU, 13)
        for k in range(12):
            a0, a1 = th[k], th[k + 1]
            n = np.array([0, math.sin((a0 + a1) / 2), math.cos((a0 + a1) / 2)])
            if float(n @ (cpos - cyl_c)) <= 0:
                continue
            r_ = 0.17
            pts = [(cyl_c[0] - 0.42, cyl_c[1] + r_ * math.sin(a0), cyl_c[2] + r_ * math.cos(a0)),
                   (cyl_c[0] + 0.42, cyl_c[1] + r_ * math.sin(a0), cyl_c[2] + r_ * math.cos(a0)),
                   (cyl_c[0] + 0.42, cyl_c[1] + r_ * math.sin(a1), cyl_c[2] + r_ * math.cos(a1)),
                   (cyl_c[0] - 0.42, cyl_c[1] + r_ * math.sin(a1), cyl_c[2] + r_ * math.cos(a1))]
            q(pts, self.shade(wood * 1.25, n, light, cpos - cyl_c, 0.4, 0.7, 0.5, 20))
        for ex in (-0.42, 0.42):
            if (cpos[0] - (cyl_c[0] + ex)) * ex > 0:
                q([(cyl_c[0] + ex, cyl_c[1] + 0.17 * math.sin(a_), cyl_c[2] + 0.17 * math.cos(a_)) for a_ in th[:-1]], wood * 0.9)
        # ---- light: window shafts, dust, accent glow
        warm = np.array([255, 214, 160], np.float32)
        for wx in (-2.3, 2.3):
            top = np.array([[wx, 5.2, -4.9]], np.float32)
            for k in range(10):
                end = np.array([[wx * 0.2 + (k - 4.5) * 0.18, 0.0, 1.0 + (k % 3) * 0.6]], np.float32)
                xa, ya, za, _ = R.project(top)
                xb, yb, zb, _ = R.project(end)
                if za[0] < 0.1 or zb[0] < 0.1:
                    continue
                xs, ys, _, s_, st = R.sample_segments(xa, ya, xb, yb, 4.0, 600)
                R.soft(xs, ys, warm[None, :] * (0.45 * (1 - s_) * st * (0.8 + 0.2 * math.sin(t * 0.4 + k)))[:, None])
            xw, yw, _, _ = R.project(np.array([[wx, 4.2, -4.9]], np.float32))
            R.soft(np.repeat(xw, 2), np.repeat(yw, 2), np.stack([warm * 900, warm * 300]))
        dp = self.dust + np.stack([0.15 * np.sin(t * 0.3 + self.dust_ph * TAU), -0.05 * t % 5.0 * 0 + 0.1 * np.sin(t * 0.2 + self.dust_ph * 9),
                                   0.1 * np.cos(t * 0.25 + self.dust_ph * 5)], 1).astype(np.float32)
        tw = 0.5 + 0.5 * np.sin(t * 1.5 + self.dust_ph * 30)
        R.points3d(dp, warm[None, :] * (1.5 + 2.0 * tw)[:, None], 0.0, 0.7, bloom=0.6)
        R.lines3d(np.array([[-2.95, 2.151, -2.4]], np.float32), np.array([[2.95, 2.151, -2.4]], np.float32), a * 1.5, width=1.5, bloom=1.0)
        gx, gy, _, _ = R.project(cyl_c[None, :].astype(np.float32))
        R.soft(np.repeat(gx, 2), np.repeat(gy, 2), np.stack([a * 300, warm * 150]))
        return self.fin(canvas, None, canvas_bloom=0.15, bloom=(0.6, 0.55), radii=(1.4, 9.0))


KINDS_TOPICAL = {
    "gpu_chip": GPUChip, "gpu": GPUChip, "graphics_card": GPUChip, "semiconductor": GPUChip, "nvidia_chip": GPUChip,
    "stock_chart": StockChart, "stocks": StockChart, "market": StockChart, "finance": StockChart,
    "trading": StockChart, "share_price": StockChart,
    "ticker_wall": TickerWall, "stock_exchange": TickerWall, "ticker": TickerWall, "wall_street": TickerWall,
    "exchange": TickerWall,
    "corporate": Corporate, "boardroom": Corporate, "company": Corporate, "office": Corporate, "skyline": Corporate,
    "business": Corporate, "headquarters": Corporate, "ceo": Corporate,
    "newsroom": Newsroom, "news": Newsroom, "breaking": Newsroom, "studio": Newsroom, "media": Newsroom,
    "report": Newsroom,
    "radar": Radar, "intelligence": Radar, "spy": Radar, "espionage": Radar, "security": Radar,
    "surveillance": Radar, "investigation": Radar,
    "capitol": Capitol, "government": Capitol, "politics": Capitol, "president": Capitol, "white_house": Capitol,
    "congress": Capitol, "law": Capitol,
    "datacenter": DataCenter, "data_center": DataCenter, "ai_compute": DataCenter, "cloud": DataCenter,
    "servers": DataCenter, "server_room": DataCenter,
    "calendar": Calendar, "date": Calendar, "time": Calendar, "days": Calendar, "deadline": Calendar,
    "timeline_bg": Calendar,
    "courtroom": Courtroom, "court": Courtroom, "lawsuit": Courtroom, "judge": Courtroom, "trial": Courtroom,
}

from . import procedural as _procedural  # noqa: E402

_procedural.KINDS.update(KINDS_TOPICAL)
