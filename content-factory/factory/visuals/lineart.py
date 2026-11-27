"""Minimalist line-art motivation scenes (the "discipline / get rich / keep going" shorts genre).

Pure black (or very dark warm) backgrounds, a faint drifting starfield (or warm dust), thin glowing
line drawings, a simple procedural human figure as the protagonist and one visual metaphor per
scene (stairs, clock, wall, calendar, mountain, door, coins, phone, mirror...). Everything is drawn
procedurally and is ORIGINAL: no fonts besides the bundled Inter, no copied artwork.

Rendering: every frame is drawn with PIL at 2x supersampling (``SS``) into a single-channel ("mono"
theme) or RGB ("gold" theme) canvas, box-downsampled to the layer size (clean anti-aliasing), then
a two-scale bloom (quarter / eighth resolution blur) and an optional quarter-res *soft light* layer
(door light, phone glow, flag halo...) are added. All randomness is drawn in ``setup()``;
``render(t)`` is a pure function of ``t``.

Themes (``params.theme``):
  * ``mono`` (default) - black background, #EDEDED strokes, tiny drifting stars, soft white glow,
    stroke ("stick") figure with a circle head.
  * ``gold`` - #0B0A0D background with a large warm radial glow behind the figure, filled gold
    (#F2B33D) pictogram figure, thin grey ground line, faint gold dust and a "TIME GONE" timer in the
    upper right (local scene time + ``params.t0``; ``params.timer`` false hides it).

  * ``paper`` - warm cream paper (#F3EBDD) with grain and fibres, black ink strokes with a hand-drawn
    "boiling line" jitter (re-drawn 8x per second), red (#D23B2E) highlights, ink stick figure.
  * ``blueprint`` - deep blue (#0E2A4F) with a fine light grid, white / cyan technical strokes, stick figure
    with joint rings.
  * ``neon`` - near-black purple (#0A0612), magenta lines and a cyan figure with a strong two-colour glow,
    hot pink accent.
  * ``chalk`` - green-grey chalkboard (#1E2A24) with smudges and eraser swipes, grainy white chalk strokes,
    yellow chalk accent.
The four textured themes draw "delta" colours (``|colour - bg|``) that are added to (or, for paper,
subtracted from) a precomputed full-res background texture, so occluders (black) show the texture.

Scenes 13-22 (``la_split``, ``la_life``, ``la_typo``, ``la_fork``, ``la_hourglass``, ``la_seed``,
``la_domino``, ``la_boulder``, ``la_gym``, ``la_book``) are faster "story" scenes (cut every 2-3 s) built on
:class:`LAStory` (shared props: coins, bulbs, books, bed, car, house...).

Common params: ``theme``, ``label`` (one short word, drawn as a subtle chip where meaningful),
``t0`` (timer offset, seconds), ``timer`` (bool). Scene specific params are documented per class.
Registered into ``procedural.KINDS`` (``KINDS_LINEART``).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from . import procedural as _procedural
from .procedural import Scene

TAU = math.tau
SS = 2  # supersampling factor
FONT_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"
SYS_FONT_DIR = Path("/usr/share/fonts/opentype/inter")

MONO = {"bg": (0, 0, 0), "fig": (237, 237, 237), "line": (226, 226, 226), "dim": (120, 120, 120),
        "faint": (62, 62, 62), "accent": (255, 255, 255), "text": (237, 237, 237), "soft": (210, 220, 235),
        "accent2": (170, 170, 170)}
GOLD = {"bg": (11, 10, 13), "fig": (242, 179, 61), "line": (176, 168, 158), "dim": (92, 86, 80),
        "faint": (46, 42, 40), "accent": (242, 179, 61), "text": (242, 179, 61), "soft": (255, 150, 50),
        "accent2": (236, 228, 214)}
# "delta" themes: colours are the FINAL on-screen colours; the canvas stores |colour - bg| and the frame is
# texture +/- canvas (``sign`` -1 = ink on bright paper). ``fig_w`` scales the stroke figure, ``wob`` is the
# boiling-line jitter amplitude (px at 1080 wide), ``bloom`` the two glow gains, ``softg`` the soft-light gain.
PAPER = {"bg": (243, 235, 221), "fig": (26, 23, 21), "line": (34, 31, 28), "dim": (128, 118, 104),
         "faint": (196, 186, 170), "accent": (210, 59, 46), "text": (30, 27, 24), "soft": (240, 200, 120),
         "accent2": (40, 90, 160), "sign": -1, "lw": 1.3, "fig_w": 1.15, "wob": 1.7, "bloom": (0.07, 0.0), "softg": 0.5,
         "grain": (0.80, 1.0), "dust": None, "chip": (243, 235, 221)}
BLUEPRINT = {"bg": (14, 42, 79), "fig": (242, 248, 255), "line": (214, 234, 250), "dim": (112, 158, 204),
             "faint": (52, 92, 138), "accent": (110, 228, 255), "text": (226, 242, 255), "soft": (100, 200, 255),
             "accent2": (255, 214, 120), "sign": 1, "fig_w": 1.15, "wob": 0.0, "bloom": (0.34, 0.3),
             "softg": 0.45, "grain": None, "dust": (150, 205, 245), "joints": True}
NEON = {"bg": (10, 6, 18), "fig": (96, 240, 255), "line": (255, 92, 226), "dim": (150, 64, 172),
        "faint": (62, 30, 84), "accent": (255, 46, 150), "text": (255, 226, 248), "soft": (255, 60, 190),
        "accent2": (96, 240, 255), "sign": 1, "lw": 1.15, "fig_w": 1.1, "wob": 0.0, "bloom": (1.0, 0.95), "softg": 0.75,
        "glow2": (80, 110, 255), "grain": None, "dust": (255, 120, 230)}
CHALK = {"bg": (30, 42, 36), "fig": (236, 238, 230), "line": (226, 230, 222), "dim": (138, 150, 140),
         "faint": (70, 84, 76), "accent": (246, 222, 104), "text": (236, 238, 230), "soft": (240, 230, 170),
         "accent2": (150, 205, 240), "sign": 1, "lw": 1.35, "fig_w": 1.15, "wob": 0.8, "bloom": (0.16, 0.12),
         "softg": 0.35, "grain": (0.42, 1.0), "dust": (220, 225, 215)}
PALETTES = {"mono": MONO, "gold": GOLD, "paper": PAPER, "blueprint": BLUEPRINT, "neon": NEON, "chalk": CHALK}
THEMES = ["mono", "gold", "paper", "blueprint", "neon", "chalk"]
THEME_ALIASES = {"mono": "mono", "white": "mono", "black": "mono", "default": "mono", "silver": "mono",
                 "line": "mono", "minimal": "mono", "gold": "gold", "golden": "gold", "warm": "gold",
                 "amber": "gold", "timer": "gold",
                 "paper": "paper", "ink": "paper", "sketch": "paper", "notebook": "paper", "cream": "paper",
                 "hand": "paper", "doodle": "paper",
                 "blueprint": "blueprint", "blue": "blueprint", "technical": "blueprint", "plan": "blueprint",
                 "engineer": "blueprint", "grid": "blueprint",
                 "neon": "neon", "cyber": "neon", "synth": "neon", "synthwave": "neon", "glow": "neon",
                 "pink": "neon", "night": "neon",
                 "chalk": "chalk", "chalkboard": "chalk", "board": "chalk", "school": "chalk",
                 "blackboard": "chalk", "green": "chalk"}


# ======================================================================== small helpers
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def ease(x):
    x = clamp(float(x))
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = clamp(float(x))
    return 1 - (1 - x) ** 3


def ease_in_out(x):
    x = clamp(float(x))
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def lerp(a, b, k):
    return a + (b - a) * k


def parse_theme(v) -> str:
    if v is None or isinstance(v, (bool, list, dict, tuple)):
        return "mono"
    return THEME_ALIASES.get(str(v).strip().lower().replace("minimal_", ""), "mono")


def parse_float(v, default):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def parse_bool(v, default):
    if v is None:
        return default
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off", "none"):
            return False
        return default
    if isinstance(v, (bool, int)):
        return bool(v)
    return default


def clean_word(v, maxlen=14) -> str:
    if v is None or isinstance(v, bool):
        return ""
    if isinstance(v, float) and not math.isfinite(v):
        return ""
    s = " ".join(str(v).replace("_", " ").split()).upper()
    s = "".join(c for c in s if c.isalnum() or c in " !?+-$%:.'&#")
    return s[:maxlen].strip()


def parse_labels(v) -> list[str]:
    if v is None:
        return []
    items = v if isinstance(v, (list, tuple)) else str(v).replace("|", ",").replace(";", ",").split(",")
    return [w for w in (clean_word(x) for x in items) if w][:12]


_FONTS: dict = {}


def font(px: float, weight: str = "Bold"):
    px = max(6, int(round(px)))
    key = (px, weight)
    if key in _FONTS:
        return _FONTS[key]
    cands = {"Bold": [FONT_DIR / "Inter-Bold.otf"],
             "Display": [FONT_DIR / "InterDisplay-ExtraBold.otf", FONT_DIR / "InterDisplay-Black.otf"],
             "Medium": [SYS_FONT_DIR / "Inter-Medium.otf", SYS_FONT_DIR / "Inter-SemiBold.otf", FONT_DIR / "Inter-Bold.otf"],
             "Light": [SYS_FONT_DIR / "InterDisplay-Light.otf", SYS_FONT_DIR / "Inter-Light.otf", FONT_DIR / "Inter-Bold.otf"],
             }.get(weight, [FONT_DIR / "Inter-Bold.otf"])
    f = None
    for c in cands + [FONT_DIR / "Inter-Bold.otf"]:
        try:
            f = ImageFont.truetype(str(c), px)
            break
        except OSError:
            continue
    if f is None:
        try:
            f = ImageFont.truetype("DejaVuSans-Bold.ttf", px)
        except OSError:
            f = ImageFont.load_default()
    _FONTS[key] = f
    return f


# ======================================================================== pen (2x supersampled drawing)
class Pen:
    """ImageDraw wrapper: takes layer (1x) coordinates and sizes, draws at ``SS`` x.

    ``ref`` (delta themes): colours are final on-screen colours, the canvas stores ``sign * (col - ref)``
    (0 / black = "background", used by occluders). ``wob`` > 0 adds a smooth boiling-line jitter field
    whose phases (``boil``) change a few times per second (hand-drawn look)."""

    def __init__(self, img: Image.Image, mono: bool, ref=None, sign=1, wob=0.0, boil=None, u=1.0, ws=1.0):
        self.img = img
        self.ws = ws                         # global stroke-width scale (chunkier ink / chalk)
        self.d = ImageDraw.Draw(img)
        self.mono = mono
        self.softs: list = []
        self.ref = ref
        self.sign = sign
        self.wob = float(wob) if boil is not None else 0.0
        self.bp = boil
        self.wf = 0.072 / max(0.2, u)        # spatial frequency of the jitter field (rad / px)
        self.seglen = 34.0 * max(0.2, u)     # subdivision length for wobbly strokes

    def c(self, col, k=1.0):
        k = max(0.0, float(k))
        if self.mono:
            if isinstance(col, (int, float)):
                return int(clamp(col * k, 0, 255))
            return int(clamp((0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]) * k, 0, 255))
        if isinstance(col, (int, float)):
            col = (col, col, col)
        if self.ref is not None:
            if col[0] == 0 and col[1] == 0 and col[2] == 0:
                return (0, 0, 0)
            r, sg = self.ref, self.sign
            return (int(clamp(sg * (col[0] - r[0]) * k, 0, 255)), int(clamp(sg * (col[1] - r[1]) * k, 0, 255)),
                    int(clamp(sg * (col[2] - r[2]) * k, 0, 255)))
        return tuple(int(clamp(v * k, 0, 255)) for v in col)

    # ------------------------------------------------------------ hand-drawn jitter
    def J(self, x, y):
        """Jittered point (identity unless the theme wobbles). Smooth in space -> joints stay connected."""
        if not self.wob:
            return (x, y)
        p1, p2, p3, p4 = self.bp
        f = self.wf
        A = self.wob
        return (x + A * (math.sin(x * f + y * 0.61 * f + p1) + 0.55 * math.sin(x * 2.3 * f - y * 1.7 * f + p2)),
                y + A * (math.sin(y * f - x * 0.73 * f + p3) + 0.55 * math.sin(y * 2.1 * f + x * 1.9 * f + p4)))

    def _wobble(self, pts):
        out = []
        L0 = self.seglen
        for i in range(len(pts) - 1):
            (x0, y0), (x1, y1) = pts[i], pts[i + 1]
            n = min(10, max(1, int(math.hypot(x1 - x0, y1 - y0) / L0)))
            for j in range(n):
                f = j / n
                out.append(self.J(x0 + (x1 - x0) * f, y0 + (y1 - y0) * f))
        out.append(self.J(*pts[-1]))
        return out

    def _w(self, w):
        return max(1, int(round(w * SS * self.ws)))

    def line(self, pts, col, w, k=1.0, caps=True):
        if len(pts) < 2 or k <= 0.003:
            return
        c = self.c(col, k)
        if self.wob:
            pts = self._wobble(pts)
        P = [(x * SS, y * SS) for x, y in pts]
        ww = self._w(w)
        self.d.line(P, fill=c, width=ww, joint="curve" if len(P) > 2 else None)
        if caps and ww >= 3:
            r = ww / 2 - 0.25
            for x, y in (P[0], P[-1]):
                self.d.ellipse([x - r, y - r, x + r, y + r], fill=c)

    def seg(self, a, b, col, w, k=1.0, caps=True):
        self.line([a, b], col, w, k, caps)

    def circle(self, c, r, col=None, w=1.0, k=1.0, fill=None, kf=1.0):
        c = self.J(*c)
        x, y = c[0] * SS, c[1] * SS
        R = max(0.5, r * SS)
        self.d.ellipse([x - R, y - R, x + R, y + R],
                       fill=None if fill is None else self.c(fill, kf),
                       outline=None if col is None or k <= 0.003 else self.c(col, k),
                       width=self._w(w) if col is not None else 0)

    def dot(self, c, r, col, k=1.0):
        if k <= 0.003:
            return
        c = self.J(*c)
        x, y = c[0] * SS, c[1] * SS
        R = max(0.6, r * SS)
        self.d.ellipse([x - R, y - R, x + R, y + R], fill=self.c(col, k))

    def _box(self, box):
        if not self.wob:
            return box
        x0, y0 = self.J(box[0], box[1])
        x1, y1 = self.J(box[2], box[3])
        return (x0, y0, x1, y1)

    def ellipse(self, box, col=None, w=1.0, k=1.0, fill=None, kf=1.0):
        x0, y0, x1, y1 = (v * SS for v in self._box(box))
        self.d.ellipse([x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)], fill=None if fill is None else self.c(fill, kf),
                       outline=None if col is None or k <= 0.003 else self.c(col, k), width=self._w(w) if col is not None else 0)

    def arc(self, c, r, a0, a1, col, w, k=1.0):
        """Arc from angle a0 to a1 (degrees, 0 = 3 o'clock, clockwise)."""
        if k <= 0.003 or a1 - a0 < 0.05:
            return
        c = self.J(*c)
        x, y, R = c[0] * SS, c[1] * SS, r * SS
        self.d.arc([x - R, y - R, x + R, y + R], a0, a1, fill=self.c(col, k), width=self._w(w))

    def poly(self, pts, col=None, w=1.0, k=1.0, fill=None, kf=1.0):
        if fill is not None:
            P = [(x * SS, y * SS) for x, y in (pts if not self.wob else [self.J(*q) for q in pts])]
            self.d.polygon(P, fill=self.c(fill, kf))
        if col is not None and k > 0.003:
            self.line(list(pts) + [pts[0]], col, w, k, caps=False)

    def rrect(self, box, r, col=None, w=1.0, k=1.0, fill=None, kf=1.0):
        x0, y0, x1, y1 = (v * SS for v in self._box(box))
        self.d.rounded_rectangle([x0, y0, max(x0 + 2, x1), max(y0 + 2, y1)], radius=max(0, r * SS),
                                 fill=None if fill is None else self.c(fill, kf),
                                 outline=None if col is None or k <= 0.003 else self.c(col, k),
                                 width=self._w(w) if col is not None else 0)

    def text(self, xy, s, px, col, k=1.0, anchor="mm", weight="Bold", track=0.0):
        """Text with letter spacing ``track`` (fraction of the font size)."""
        if not s or k <= 0.003:
            return
        f = font(px * SS, weight)
        c = self.c(col, k)
        xy = self.J(*xy)
        x, y = xy[0] * SS, xy[1] * SS
        if track <= 0:
            self.d.text((x, y), s, font=f, fill=c, anchor=anchor)
            return
        sp = track * px * SS
        widths = [f.getlength(ch) for ch in s]
        total = sum(widths) + sp * (len(s) - 1)
        h = anchor[0]
        x0 = x - total / 2 if h == "m" else x - total if h == "r" else x
        for ch, wch in zip(s, widths):
            self.d.text((x0, y), ch, font=f, fill=c, anchor="l" + anchor[1])
            x0 += wch + sp

    def text_width(self, s, px, weight="Bold", track=0.0):
        f = font(px * SS, weight)
        return (f.getlength(s) + track * px * SS * max(0, len(s) - 1)) / SS

    def fade_line(self, x0, x1, y, col, w, k=1.0, fade=0.22, n=28):
        """Horizontal line whose brightness fades out towards both ends."""
        xs = np.linspace(x0, x1, n + 1)
        for i in range(n):
            m = (i + 0.5) / n
            e = min(m, 1 - m) / max(1e-3, fade)
            self.seg((xs[i], y), (xs[i + 1] + 0.5, y), col, w, k * ease(e), caps=False)

    def dashed(self, pts, col, w, k=1.0, dash=10.0, gap=8.0, offset=0.0):
        pts = np.asarray(pts, np.float64)
        seglen = np.hypot(*(pts[1:] - pts[:-1]).T)
        cum = np.concatenate([[0], np.cumsum(seglen)])
        total = cum[-1]
        period = dash + gap
        s = -(offset % period)
        while s < total:
            a, b = max(0.0, s), min(total, s + dash)
            if b > a:
                self.seg(tuple(_along(pts, cum, a)), tuple(_along(pts, cum, b)), col, w, k)
            s += period

    def soft(self, x, y, r, col, k=1.0):
        """Additive soft light (rendered at quarter resolution)."""
        if k > 0.003 and r > 0:
            self.softs.append((x, y, r, col, k))


def _along(pts, cum, s):
    i = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(pts) - 2))
    seg = cum[i + 1] - cum[i]
    f = 0.0 if seg <= 1e-9 else (s - cum[i]) / seg
    return pts[i] + (pts[i + 1] - pts[i]) * f


def polyline_at(pts, cum, s):
    """Point + unit direction at arc length s along a polyline."""
    i = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(pts) - 2))
    seg = max(1e-9, cum[i + 1] - cum[i])
    f = clamp((s - cum[i]) / seg)
    d = (pts[i + 1] - pts[i]) / seg
    return pts[i] + (pts[i + 1] - pts[i]) * f, d


# ======================================================================== the figure
# Lengths in units of standing height h (feet at 0, head top ~0.985).
TORSO, NECK, HEAD_R = 0.31, 0.11, 0.072
UPPER_ARM, FOREARM = 0.16, 0.15
THIGH, SHIN, FOOT = 0.245, 0.245, 0.05

POSE_KEYS = ("lean", "head", "thL", "thR", "knL", "knR", "shL", "shR", "elL", "elR")


def P(**kw) -> dict:
    p = {k: 0.0 for k in POSE_KEYS}
    p["front"] = False
    p.update(kw)
    return p


def mix(a: dict, b: dict, k: float) -> dict:
    k = clamp(k)
    out = dict(a)
    for key in POSE_KEYS:
        out[key] = a.get(key, 0.0) + (b.get(key, 0.0) - a.get(key, 0.0)) * k
    for key in ("dy", "liftL", "liftR", "sway"):
        if key in a or key in b:
            out[key] = a.get(key, 0.0) + (b.get(key, 0.0) - a.get(key, 0.0)) * k
    out["front"] = b["front"] if k >= 0.5 else a["front"]
    out["hands"] = bool((a.get("hands") and k < 0.85) or (b.get("hands") and k > 0.15))
    return out


def pose_stand(t=0.0):
    b = math.sin(t * 1.6)
    return P(lean=0.015 + 0.008 * b, head=0.03 + 0.015 * b, thL=0.035, thR=-0.035, knL=0.03, knR=0.02,
             shL=0.07 + 0.012 * b, shR=-0.05 - 0.012 * b, elL=0.14, elR=0.1)


def pose_stand_front(t=0.0):
    b = math.sin(t * 1.6)
    return P(front=True, head=0.0, thL=0.1, thR=0.1, shL=0.36 + 0.015 * b, shR=0.36 + 0.015 * b,
             elL=-0.16, elR=-0.16, dy=0.003 * b)


def pose_lookup(t=0.0):
    p = pose_stand(t)
    p.update(head=-0.42, lean=-0.06, shL=0.05, shR=-0.08)
    return p


def pose_walk(ph):
    s, c = math.sin(ph), math.cos(ph)
    thR, thL = 0.36 * s, -0.36 * s
    knR = 0.07 + 0.62 * max(0.0, c) ** 2
    knL = 0.07 + 0.62 * max(0.0, -c) ** 2
    shR, shL = -0.30 * s, 0.30 * s
    return P(lean=0.06, head=0.04 + 0.02 * math.sin(2 * ph), thL=thL, thR=thR, knL=knL, knR=knR,
             shL=shL, shR=shR, elL=0.22 + 0.3 * max(0.0, s), elR=0.22 + 0.3 * max(0.0, -s))


def pose_run(ph):
    s = math.sin(ph)
    thR = 0.22 + 0.62 * s
    thL = 0.22 - 0.62 * s
    knR = 0.25 + 1.45 * max(0.0, math.cos(ph + 0.75)) ** 1.3
    knL = 0.25 + 1.45 * max(0.0, math.cos(ph + 0.75 + math.pi)) ** 1.3
    return P(lean=0.2, head=0.1, thL=thL, thR=thR, knL=knL, knR=knR, shL=0.75 * s, shR=-0.75 * s,
             elL=1.45 + 0.15 * s, elR=1.45 - 0.15 * s, dy=-0.028 * abs(math.sin(ph)))


def pose_push(t=0.0):
    e = 0.02 * math.sin(t * 9.0) * (0.5 + 0.5 * math.sin(t * 1.3))
    return P(lean=0.78 + e, head=-0.25, thR=0.42, knR=0.75, thL=-0.42 + e, knL=0.06,
             shR=1.38 + e, shL=1.5 + e, elR=0.42, elL=0.3)


def pose_kneel(t=0.0):
    b = math.sin(t * 1.2)
    return P(lean=0.42 + 0.02 * b, head=0.55 + 0.03 * b, thR=1.45, knR=1.48, thL=0.12, knL=1.7,
             shR=0.05, shL=0.1, elR=0.15, elL=0.2)


def pose_fallen(t=0.0):
    return P(lean=1.15, head=0.25, thR=0.55, knR=1.75, thL=0.4, knL=1.8, shR=0.32, shL=0.5,
             elR=-0.15, elL=-0.1, hands=True)


def pose_sit(t=0.0, hunch=0.35):
    b = math.sin(t * 1.3)
    return P(lean=hunch + 0.015 * b, head=0.45 + 0.02 * b, thR=1.5, knR=1.5, thL=1.42, knL=1.32,
             shR=0.55, shL=0.65, elR=0.85, elL=0.8)


def pose_victory(t=0.0):
    b = math.sin(t * 2.0)
    return P(front=True, head=-0.18, thL=0.17, thR=0.17, shL=2.55 + 0.05 * b, shR=2.55 + 0.05 * b,
             elL=0.12, elR=0.12)


def pose_proud(t=0.0):
    b = math.sin(t * 1.4)
    return P(lean=-0.035, head=-0.12 + 0.01 * b, thL=0.05, thR=-0.06, knL=0.02, knR=0.02,
             shL=0.12, shR=-0.12, elL=0.25, elR=0.25, dy=0.002 * b)


def pose_hunched(t=0.0):
    b = math.sin(t * 1.2)
    return P(lean=0.24 + 0.01 * b, head=0.48 + 0.02 * b, thL=0.04, thR=-0.02, knL=0.1, knR=0.08,
             shL=0.28, shR=0.18, elL=0.25, elR=0.3)


def _ik(hip, foot, a, b, facing):
    dx, dy = foot[0] - hip[0], foot[1] - hip[1]
    d = math.hypot(dx, dy)
    d = clamp(d, 1e-6, (a + b) * 0.999)
    base = math.atan2(dy, dx)
    cosA = clamp((a * a + d * d - b * b) / (2 * a * d), -1, 1)
    A = math.acos(cosA)
    # knee in front of the body: rotate towards the facing direction
    ang = base - A * facing
    knee = (hip[0] + a * math.cos(ang), hip[1] + a * math.sin(ang))
    ang2 = base + A * facing
    alt = (hip[0] + a * math.cos(ang2), hip[1] + a * math.sin(ang2))
    if (alt[0] - knee[0]) * facing > 0:
        knee = alt
    fx = hip[0] + dx / max(1e-9, math.hypot(dx, dy)) * d
    fy = hip[1] + dy / max(1e-9, math.hypot(dx, dy)) * d
    return knee, (fx, fy)


def fk(pose: dict, h: float, facing: int = 1, picto: bool = False) -> dict:
    """Joint positions relative to the hip (y down)."""
    f = 1 if facing >= 0 else -1
    J = {"hip": (0.0, 0.0)}
    T = TORSO * h
    if pose.get("front"):
        sway = pose.get("sway", 0.0) * h
        neck = (sway, -T)
        J["neck"] = neck
        J["head"] = (sway * 1.2 + pose["head"] * 0.0, -T - NECK * h * (1 - 0.15 * max(0.0, pose["head"])))
        sw = 0.062 * h if picto else 0.0
        hw = 0.036 * h if picto else 0.0
        sh_y = -T + 0.035 * h
        for side, sgn in (("L", -1), ("R", 1)):
            S = (sway * 0.95 + sgn * sw, sh_y)
            a, e = pose["sh" + side], pose["el" + side]
            E = (S[0] + sgn * UPPER_ARM * h * math.sin(a), S[1] + UPPER_ARM * h * math.cos(a))
            Hd = (E[0] + sgn * FOREARM * h * math.sin(a + e), E[1] + FOREARM * h * math.cos(a + e))
            J["sh" + side], J["el" + side], J["hand" + side] = S, E, Hd
            Hp = (sgn * hw, 0.0)
            th = pose["th" + side]
            lift = pose.get("lift" + side, 0.0)
            K = (Hp[0] + sgn * THIGH * h * math.sin(th), Hp[1] + THIGH * h * math.cos(th) * (1 - 0.35 * lift))
            Fp = (K[0] + sgn * SHIN * h * math.sin(th * 0.6), K[1] + SHIN * h * (1 - 1.1 * lift))
            J["hp" + side], J["knee" + side], J["foot" + side] = Hp, K, Fp
            J["toe" + side] = (Fp[0] + sgn * FOOT * h * 0.55, Fp[1])
        J["front"] = True
        return J
    lean, hd = pose["lean"], pose["head"]
    neck = (f * math.sin(lean) * T, -math.cos(lean) * T)
    J["neck"] = neck
    ah = lean + hd
    nl = NECK * h * (1.12 if picto else 1.0)
    J["head"] = (neck[0] + f * math.sin(ah) * nl, neck[1] - math.cos(ah) * nl)
    k = (T - 0.032 * h) / T
    S = (neck[0] * k, neck[1] * k)
    for side in ("L", "R"):
        a, e = pose["sh" + side], pose["el" + side]
        E = (S[0] + f * UPPER_ARM * h * math.sin(a), S[1] + UPPER_ARM * h * math.cos(a))
        Hd = (E[0] + f * FOREARM * h * math.sin(a + e), E[1] + FOREARM * h * math.cos(a + e))
        J["sh" + side], J["el" + side], J["hand" + side] = S, E, Hd
        th, kn = pose["th" + side], pose["kn" + side]
        ik = pose.get("ik" + side)
        if ik is not None:
            K, Fp = _ik((0.0, 0.0), ik, THIGH * h, SHIN * h, f)
            sa = math.atan2((Fp[0] - K[0]) * f, Fp[1] - K[1])
        else:
            K = (f * THIGH * h * math.sin(th), THIGH * h * math.cos(th))
            sa = th - kn
            Fp = (K[0] + f * SHIN * h * math.sin(sa), K[1] + SHIN * h * math.cos(sa))
        J["hp" + side], J["knee" + side], J["foot" + side] = (0.0, 0.0), K, Fp
        tw = FOOT * h
        J["toe" + side] = (Fp[0] + f * tw * math.cos(sa), Fp[1] - tw * math.sin(sa) * 0.6)
    J["front"] = False
    return J


def contact_offset(J: dict, pose: dict) -> float:
    """Largest y (lowest point) among the parts that can touch the ground."""
    keys = ["footL", "footR", "kneeL", "kneeR", "toeL", "toeR"]
    if pose.get("hands"):
        keys += ["handL", "handR"]
    return max(J[k][1] for k in keys)


def place(pose, h, x, ground, facing=1, picto=False) -> dict:
    """FK + put the figure on a flat ground line at y=ground. Returns absolute joints."""
    J = fk(pose, h, facing, picto)
    low = contact_offset(J, pose)
    if picto:
        low += 0.035 * h  # thick round caps reach below the joint
    else:
        low += 0.006 * h
    oy = ground - low + pose.get("dy", 0.0) * h
    return {k: ((v[0] + x, v[1] + oy) if isinstance(v, tuple) else v) for k, v in J.items()} | {"h": h, "facing": facing}


def seat_height(h, picto=False):
    """Seat height (px above the ground) at which a sitting figure's shins are vertical."""
    return SHIN * h * 0.96 + (0.03 * h if picto else 0.004 * h)


def seated(pose, h, hip_x, seat_y, ground, facing=1, picto=False, feet=(0.22, 0.17)) -> dict:
    """Sitting figure: hip on the seat at y=seat_y, feet planted on the ground via leg IK."""
    pose = dict(pose)
    hip_y = seat_y - (0.045 * h if picto else 0.01 * h)
    foot_y = ground - hip_y - (0.035 * h if picto else 0.006 * h)
    pose["ikR"] = (feet[0] * h * facing, foot_y)
    pose["ikL"] = (feet[1] * h * facing, foot_y)
    J = fk(pose, h, facing, picto)
    A = {k: ((v[0] + hip_x, v[1] + hip_y) if isinstance(v, tuple) else v) for k, v in J.items()}
    A.update(h=h, facing=facing)
    return A


def draw_figure(pen: Pen, A: dict, theme: dict, picto: bool, k: float = 1.0, col=None, far_k=None,
                occlude=True, w_scale=1.0, head_scale=1.0, joints=False):
    """Draw a placed figure (absolute joints from :func:`place`). ``head_scale`` > 1 for kids / babies,
    ``joints`` adds small technical joint rings (blueprint)."""
    h = A["h"]
    col = theme["fig"] if col is None else col
    front = A.get("front")
    if picto:
        lw = 0.074 * h * w_scale
        tw = 0.118 * h * w_scale
        fk_ = 0.62 if far_k is None else far_k
        gap = max(1.2, 0.022 * h)

        def limb(side, kk, under):
            pts_arm = [A["sh" + side], A["el" + side], A["hand" + side]]
            pts_leg = [A["hp" + side], A["knee" + side], A["foot" + side]]
            for pts, w in ((pts_leg, lw * 1.08), (pts_arm, lw)):
                if under and occlude:
                    pen.line(pts, theme["bg"] if not pen.mono else 0, w + 2 * gap, 1.0)
                pen.line(pts, col, w, k * kk)

        far, near = ("L", "R")
        if front:
            if occlude:  # dark halo around the whole silhouette (reads against lights / crowds)
                blk = 0 if pen.mono else (0, 0, 0)
                for side in ("L", "R"):
                    pen.line([A["sh" + side], A["el" + side], A["hand" + side]], blk, lw + 2 * gap, 1.0)
                    pen.line([A["hp" + side], A["knee" + side], A["foot" + side]], blk, lw * 1.08 + 2 * gap, 1.0)
                pen.line([A["neck"], A["hip"]], blk, tw + 2 * gap, 1.0)
            limb("L", 0.92, False)
            limb("R", 0.92, False)
        else:
            limb(far, fk_, False)
        # torso (shoulder to hip) + head disk
        top = A["neck"]
        hip = A["hip"]
        tt = (top[0] + (hip[0] - top[0]) * 0.1, top[1] + (hip[1] - top[1]) * 0.1)
        if occlude and not front:
            pen.line([tt, hip], 0 if pen.mono else (0, 0, 0), tw + 2 * gap, 1.0)
        pen.line([tt, hip], col, tw, k)
        hr = 0.085 * h * w_scale * head_scale
        if occlude:
            pen.circle(A["head"], hr + gap, fill=0 if pen.mono else (0, 0, 0))
        pen.circle(A["head"], hr, fill=col, kf=k)
        if not front:
            limb(near, 1.0, True)
        return
    # ---- mono stroke figure
    w = max(1.0, 0.0145 * h) * w_scale
    fk_ = 0.5 if far_k is None else far_k
    gap = max(1.0, w * 0.9)
    blk = 0 if pen.mono else (0, 0, 0)

    def stroke(pts, kk, trim=0.0):
        if occlude:
            u = pts
            if trim > 0:
                a, b = pts[0], pts[1]
                u = [(a[0] + (b[0] - a[0]) * trim, a[1] + (b[1] - a[1]) * trim)] + list(pts[1:])
            pen.line(u, blk, w + 2 * gap, 1.0)
        pen.line(pts, col, w, k * kk)

    def arm(side, kk):
        stroke([A["sh" + side], A["el" + side], A["hand" + side]], kk, 0.45)

    def leg(side, kk):
        stroke([A["hp" + side], A["knee" + side], A["foot" + side], A["toe" + side]], kk, 0.4)

    if front:
        leg("L", 1.0)
        leg("R", 1.0)
        stroke([A["hip"], A["neck"]], 1.0)
        arm("L", 1.0)
        arm("R", 1.0)
    else:
        arm("L", fk_)
        leg("L", fk_)
        stroke([A["hip"], A["neck"]], 1.0)
        leg("R", 1.0)
    hr = HEAD_R * h * head_scale
    if occlude:
        pen.circle(A["head"], hr + gap, fill=blk)
    pen.circle(A["head"], hr, col, w, k)
    if not front:
        arm("R", 1.0)
    if joints:
        jr = max(1.5, w * 1.25)
        for key in ("elL", "elR", "kneeL", "kneeR", "hip", "neck"):
            pen.circle(A[key], jr, col, max(0.8, w * 0.45), 0.8 * k, fill=blk)


class Gait:
    """Walk-cycle stride bookkeeping so feet stay planted while the world scrolls."""

    def __init__(self, pose_fn, h):
        self.fn, self.h = pose_fn, h
        # planted foot x (relative to hip) at the start / end of the right foot's stance
        self.a = self._fx(math.pi / 2)
        self.b = self._fx(3 * math.pi / 2)
        self.stride = self.a - self.b  # hip travel per half cycle

    def _fx(self, ph):
        J = fk(self.fn(ph), self.h, 1)
        return J["footR"][0]

    def offset(self, ph):
        """Hip travel (px) after phase ph (radians) - piecewise exact for the stance foot."""
        q = (ph - math.pi / 2) / math.pi
        j = math.floor(q)
        loc = math.pi / 2 + (q - j) * math.pi  # phase inside a right-foot-style stance
        fx = self._fx(loc)
        return j * self.stride + (self.a - fx)


# ======================================================================== base scene
def _smooth_noise(w, h, cell, rng):
    """Value noise in [0, 1]: a random grid with ``cell`` px spacing, bicubic-upscaled to (h, w)."""
    gw, gh = max(2, int(w / cell) + 2), max(2, int(h / cell) + 2)
    g = rng.random((gh, gw)).astype(np.float32)
    im = Image.fromarray(g).resize((int(gw * cell), int(gh * cell)), Image.BICUBIC)
    return np.clip(np.asarray(im, np.float32)[:h, :w], 0, 1)


class LineArt(Scene):
    dynamic = True
    focus = (0.5, 0.55)       # warm glow centre (visible-frame fractions), gold theme
    glow_size = 0.62          # glow radius as fraction of frame width
    star_drift = (-3.0, 0.0)  # px/s at depth 1 (scaled by u)
    label_at = (0.5, 0.715)   # default chip position (None -> scene draws it itself)

    # ---------------------------------------------------------------- setup
    def build(self):
        p = self.p
        self.theme_name = parse_theme(p.get("theme"))
        self.th = PALETTES[self.theme_name]
        name = self.theme_name
        self.gold = name == "gold"
        self.gray = name == "mono"            # single-channel render (mono only)
        self.delta = name not in ("mono", "gold")  # textured themes (paper / blueprint / neon / chalk)
        self.mono = not self.gold             # "neutral stroke style" (every theme except gold)
        self.picto = self.gold                # filled pictogram figure (gold only)
        self.mode = "L" if self.gray else "RGB"
        self.label = clean_word(p.get("label"))
        self.t0 = parse_float(p.get("t0"), 0.0)
        self.show_timer = parse_bool(p.get("timer"), self.gold)
        self.u = self.ow / 1080.0
        rng = self.rng
        W, H = self.W, self.H
        # starfield (mono, neon) / warm dust (gold, blueprint, chalk; paper draws none)
        self.stars = name in ("mono", "neon")
        if self.stars:
            n = int(360 * W * H / (1209 * 2150)) + 12
            self.st_x = rng.uniform(0, W, n)
            self.st_y = rng.uniform(0, H, n)
            self.st_d = rng.uniform(0.25, 1.0, n)
            self.st_r = (0.45 + 0.9 * rng.power(4, n)) * max(0.6, self.u)
            self.st_b = 35 + 175 * rng.power(3.2, n)
            self.st_f = rng.uniform(0.25, 1.1, n)
            self.st_p = rng.uniform(0, TAU, n)
        else:
            n = int(110 * W * H / (1209 * 2150)) + 8
            fx, fy = self.focus
            self.st_x = rng.uniform(0, W, n)
            self.st_y = np.clip(rng.normal(self.Y(fy), 0.24 * H, n), 0, H)
            self.st_d = rng.uniform(0.3, 1.0, n)
            self.st_r = (0.6 + 1.3 * rng.power(3, n)) * max(0.6, self.u)
            self.st_b = 0.12 + 0.5 * rng.power(2.5, n)
            self.st_f = rng.uniform(0.2, 0.7, n)
            self.st_p = rng.uniform(0, TAU, n)
        self.base = None
        self.tex = self.grain = None
        self.setup()
        if self.gold:
            self.base = self._gold_base()
        if self.delta:
            self.tex, self.grain = self._texture()
            self._boil = rng.uniform(0, TAU, (32, 4))
            g1, g2 = self.th["bloom"]
            self._soft_gain = self.th["softg"]
        else:
            g1, g2 = (0.85, 0.75) if self.gray else (0.5, 0.6)
            self._soft_gain = 0.42 if self.gray else 1.0
        self._glut1 = [int(min(255, v * g1)) for v in range(256)] * (1 if self.gray else 3)
        self._glut2 = [int(min(255, v * g2)) for v in range(256)] * (1 if self.gray else 3)
        if self.delta and "glow2" in self.th:  # neon: the wide glow is re-tinted (two-colour glow)
            c2 = self.th["glow2"]
            self._glut2c = [[int(min(255, v * g2 * c / 255.0)) for v in range(256)] for c in c2]

    def setup(self):
        pass

    def X(self, fx):
        return self.W / 2 + (fx - 0.5) * self.ow

    def Y(self, fy):
        return self.H / 2 + (fy - 0.5) * self.oh

    def _gold_base(self):
        """Quarter-res warm backdrop: dark warm base + large soft radial glow behind the focus."""
        W, H = self.W, self.H
        w4, h4 = (W + 3) // 4, (H + 3) // 4
        yy, xx = np.mgrid[0:h4, 0:w4].astype(np.float32)
        cx, cy = self.X(self.focus[0]) / 4, self.Y(self.focus[1]) / 4
        r = self.glow_size * self.ow / 4
        d2 = ((xx - cx) ** 2 + ((yy - cy) * 0.9) ** 2) / (r * r)
        g = np.exp(-d2 * 1.6) * 0.75 + np.exp(-d2 * 6.0) * 0.35
        bg = np.array(self.th["bg"], np.float32)
        warm = np.array([80, 43, 10], np.float32)
        img = bg[None, None, :] + g[..., None] * warm[None, None, :]
        img += self.rng.uniform(-0.5, 0.5, img.shape[:2] + (1,)).astype(np.float32)
        return Image.fromarray(np.clip(img + 0.5, 0, 255).astype(np.uint8))

    def _texture(self):
        """Full-res background texture (+ optional stroke grain multiplier) for the delta themes."""
        W, H, u, rng = self.W, self.H, self.u, self.rng
        name = self.theme_name
        bg = np.array(self.th["bg"], np.float32)
        yy = (np.arange(H, dtype=np.float32)[:, None] - self.Y(self.focus[1])) / (0.75 * self.oh)
        xx = (np.arange(W, dtype=np.float32)[None, :] - self.X(self.focus[0])) / (0.75 * self.ow)
        rad = np.clip(xx * xx + yy * yy, 0, 4)  # 0 at the focus, ~1 near the frame edges
        grain = None
        if name == "paper":
            mott = _smooth_noise(W, H, 150 * u + 8, rng) - 0.5
            fine = _smooth_noise(W, H, 3.0, rng) - 0.5
            fib = Image.new("L", (W, H), 0)
            d = ImageDraw.Draw(fib)
            for _ in range(int(1500 * W * H / (1209 * 2150))):
                x, y = rng.uniform(0, W), rng.uniform(0, H)
                a = rng.uniform(0, TAU)
                ln = rng.uniform(6, 34) * u
                bend = rng.uniform(-0.6, 0.6)
                pts = [(x + math.cos(a + bend * f) * ln * f, y + math.sin(a + bend * f) * ln * f) for f in (0, 0.5, 1)]
                d.line(pts, fill=int(rng.uniform(60, 255)), width=1)
            fib = np.asarray(fib.filter(ImageFilter.GaussianBlur(0.5)), np.float32) / 255.0
            spk = (rng.random((H, W)) > 0.9993).astype(np.float32)
            shade = 1.0 - 0.07 * rad
            img = bg[None, None, :] * shade[..., None] + (mott * 9 + fine * 5)[..., None]
            img -= (fib * 7 + spk * 40)[..., None] * np.array([1.0, 1.05, 1.15], np.float32)
            g = 0.80 + 0.20 * np.clip(_smooth_noise(W, H, 2.2, rng) * 1.25, 0, 1)
            grain = g
        elif name == "blueprint":
            img = bg[None, None, :] * (1.12 - 0.3 * np.clip(rad, 0, 1.4))[..., None]
            img += ((_smooth_noise(W, H, 120 * u + 8, rng) - 0.5) * 6)[..., None]
            cell = 36 * u
            gx = np.arange(W, dtype=np.float32) - self.X(0.5)
            gy = np.arange(H, dtype=np.float32) - self.Y(0.5)

            def lines(v, period, width):
                m = np.abs(((v / period) + 0.5) % 1.0 - 0.5) * period
                return np.clip(1.0 - m / max(0.6, width), 0, 1)
            minor = np.maximum(lines(gx, cell, 0.9 * u)[None, :], lines(gy, cell, 0.9 * u)[:, None])
            major = np.maximum(lines(gx, cell * 5, 1.4 * u)[None, :], lines(gy, cell * 5, 1.4 * u)[:, None])
            gk = (1.0 - 0.45 * np.clip(rad, 0, 1.5))[..., None]
            img += (minor[..., None] * np.array([16, 30, 42], np.float32) + major[..., None] * np.array([20, 34, 46], np.float32)) * gk
        elif name == "neon":
            fy = self.Y(self.focus[1])
            hy = (np.arange(H, dtype=np.float32)[:, None] - fy) / (0.45 * self.oh)
            hx = xx * 0.75 / 0.6
            haze = np.exp(-(hx * hx + hy * hy) * 1.4)
            low = np.exp(-(hx * hx * 0.5 + ((hy - 0.6) / 0.5) ** 2))
            img = bg[None, None, :] + haze[..., None] * np.array([34, 6, 44], np.float32) \
                + low[..., None] * np.array([0, 16, 30], np.float32)
            img *= (1.0 - 0.35 * np.clip(rad - 0.4, 0, 1.2))[..., None]
            img += rng.uniform(-1.2, 1.2, (H, W, 1)).astype(np.float32)
        else:  # chalk
            smud = _smooth_noise(W, H, 220 * u + 8, rng) - 0.5
            swipe = np.zeros((H, W), np.float32)
            for _ in range(7):  # old eraser swipes: wide soft horizontal bands with streaks
                cy, hh = rng.uniform(0, H), rng.uniform(40, 140) * u
                band = np.exp(-((np.arange(H, dtype=np.float32) - cy) / hh) ** 2)[:, None]
                x0, x1 = sorted(rng.uniform(-0.2 * W, 1.2 * W, 2))
                xs = np.arange(W, dtype=np.float32)
                span = np.clip(np.minimum(xs - x0, x1 - xs) / (60 * u + 1), 0, 1)[None, :]
                swipe += band * span * rng.uniform(0.4, 1.0)
            streak = np.asarray(Image.fromarray((rng.random((H // 3 + 1, W // 24 + 1)) * 255).astype(np.uint8))
                                .resize((W, H), Image.BILINEAR), np.float32)[:H, :W] / 255.0
            fine = rng.random((H, W)).astype(np.float32) - 0.5
            spk = (rng.random((H, W)) > 0.9985).astype(np.float32) * rng.random((H, W)).astype(np.float32)
            img = bg[None, None, :] * (1.0 - 0.28 * np.clip(rad - 0.3, 0, 1.2))[..., None]
            img += (smud * 9 + swipe * (4 + 6 * streak) + fine * 6 + spk * 30)[..., None]
            n1 = rng.random((H, W)).astype(np.float32)
            n2 = _smooth_noise(W, H, 2.5, rng)
            grain = np.clip(0.42 + 0.75 * (0.55 * n1 + 0.45 * n2) ** 1.4, 0.3, 1.0)
        tex = Image.fromarray(np.clip(img + 0.5, 0, 255).astype(np.uint8))
        if grain is not None:
            gi = Image.fromarray(np.clip(grain * 255 + 0.5, 0, 255).astype(np.uint8))
            grain = Image.merge("RGB", (gi, gi, gi))
        return tex, grain

    # ---------------------------------------------------------------- frame
    def render(self, t):
        t = parse_float(t, 0.0)
        W, H = self.W, self.H
        img = Image.new(self.mode, (W * SS, H * SS), 0)
        if self.delta:
            boil = tuple(self._boil[int(math.floor(t * 8.0)) % 32])
            pen = Pen(img, False, self.th["bg"], self.th["sign"], self.th["wob"] * self.u, boil, self.u,
                      self.th.get("lw", 1.0))
        else:
            pen = Pen(img, self.gray)
        self.draw_particles(pen, t)
        self.draw(pen, t)
        if self.label and self.label_at is not None:
            self.chip(pen, self.X(self.label_at[0]), self.Y(self.label_at[1]), self.label)
        if self.show_timer:
            self.draw_timer(pen, t)
        return self.compose(img, pen)

    def compose(self, img, pen):
        """Downsample (AA), add a two-scale bloom + soft lights (+ gold backdrop) built at quarter res."""
        W, H = self.W, self.H
        small = img.reduce(SS)
        if self.delta and self.grain is not None:
            small = ImageChops.multiply(small, self.grain)
        s4 = small.reduce(4)
        b1 = s4.filter(ImageFilter.GaussianBlur(2.2 * max(0.3, self.u)))
        b2 = s4.reduce(2).filter(ImageFilter.GaussianBlur(3.5 * max(0.3, self.u))).resize(s4.size, Image.BILINEAR)
        if self.delta and "glow2" in self.th:
            L = b2.convert("L")
            b2c = Image.merge("RGB", [L.point(lut) for lut in self._glut2c])
            g = ImageChops.add(ImageChops.add(b1.point(self._glut1), b2.point(self._glut2)), b2c)
        else:
            g = ImageChops.add(b1.point(self._glut1), b2.point(self._glut2))
        if pen.softs:
            g = ImageChops.add(g, self._soft_layer(pen.softs, s4.size))
        if self.base is not None:
            g = ImageChops.add(g, self.base)
        glow = g.resize((W, H), Image.BILINEAR, box=(0, 0, W / 4, H / 4))
        if self.gray:
            return np.asarray(ImageChops.add(small, glow).convert("RGB"), np.float32)
        if self.delta:
            if self.th["sign"] < 0:
                out = ImageChops.subtract(ImageChops.subtract(self.tex, small), glow)
            else:
                out = ImageChops.add(ImageChops.add(self.tex, small), glow)
            return np.asarray(out, np.float32)
        # colour: sum in 16 bit, then compress over-range pixels along their own hue (a plain per-channel
        # clip would turn saturated gold into lemon yellow)
        tot = np.asarray(small, np.uint16) + np.asarray(glow, np.uint16)
        m = np.maximum(np.maximum(tot[..., 0], tot[..., 1]), tot[..., 2])
        ys, xs = np.nonzero(m > 255)
        out = tot.astype(np.float32)
        if len(ys):
            out[ys, xs] *= (255.0 / m[ys, xs].astype(np.float32))[:, None]
        return out

    def _soft_layer(self, softs, size):
        q = 4
        w4, h4 = size
        ch = 1 if self.gray else 3
        acc = np.zeros((h4, w4, ch), np.float32)
        for x, y, r, col, k in softs:
            r4 = max(1.0, r / q)
            x4, y4 = x / q, y / q
            x0, x1 = int(max(0, x4 - 3 * r4)), int(min(w4, x4 + 3 * r4 + 1))
            y0, y1 = int(max(0, y4 - 3 * r4)), int(min(h4, y4 + 3 * r4 + 1))
            if x1 <= x0 or y1 <= y0:
                continue
            gx = np.exp(-((np.arange(x0, x1, dtype=np.float32) - x4) / r4) ** 2)
            gy = np.exp(-((np.arange(y0, y1, dtype=np.float32) - y4) / r4) ** 2)
            gg = gy[:, None] * gx[None, :] * k
            if self.gray:
                c = np.array([0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]], np.float32)
            elif self.delta:
                c = np.clip(self.th["sign"] * (np.asarray(col, np.float32) - np.asarray(self.th["bg"], np.float32)), 0, 255)
            else:
                c = np.asarray(col, np.float32)
            acc[y0:y1, x0:x1] += gg[..., None] * (c * self._soft_gain)[None, None, :]
        a = np.clip(acc + 0.5, 0, 255).astype(np.uint8)
        return Image.fromarray(a[..., 0] if self.gray else a)

    def draw_particles(self, pen, t):
        W, H = self.W, self.H
        u = self.u
        if self.stars:
            vx, vy = self.star_drift
            xs = (self.st_x + vx * u * self.st_d * t) % W
            ys = (self.st_y + vy * u * self.st_d * t) % H
            bs = self.st_b * (0.72 + 0.28 * np.sin(t * self.st_f * TAU * 0.5 + self.st_p))
            if self.gray:
                for x, y, r, b in zip(xs, ys, self.st_r, bs):
                    pen.dot((x, y), r, int(b))
            else:  # neon: two-colour stars
                cols = (self.th["fig"], self.th["line"], self.th["text"])
                for i, (x, y, r, b) in enumerate(zip(xs, ys, self.st_r, bs)):
                    pen.dot((x, y), r, cols[i % 3], b / 255.0)
        else:
            dust = self.th["accent"] if self.gold else self.th.get("dust")
            if dust is None:
                return
            vx, vy = self.star_drift
            fall = -1.0 if self.theme_name == "chalk" else 1.0  # chalk dust sinks, warm dust rises
            xs = (self.st_x + vx * 0.5 * u * self.st_d * t + 9 * u * np.sin(t * 0.3 + self.st_p)) % W
            ys = (self.st_y - fall * (7 + 5 * self.st_d) * u * t) % H
            bs = self.st_b * (0.55 + 0.45 * np.sin(t * self.st_f * TAU + self.st_p))
            gk = 1.0 if self.gold else 0.75
            for x, y, r, b in zip(xs, ys, self.st_r, bs):
                pen.dot((x, y), r, dust, max(0.0, b * gk))

    def draw(self, pen, t):
        pass

    # ---------------------------------------------------------------- shared decorations
    def chip(self, pen, x, y, word, k=1.0):
        """Subtle pill label: thin rounded outline + letter-spaced caps."""
        if not word or k <= 0.01:
            return
        u = self.u
        px = 25 * u
        tw = pen.text_width(word, px, "Bold", 0.32)
        ww, hh = tw + 52 * u, 54 * u
        col = self.th["text"]
        pen.rrect((x - ww / 2, y - hh / 2, x + ww / 2, y + hh / 2), hh / 2, self.th["dim"], 1.6 * u, k,
                  fill=0 if self.mono else (0, 0, 0))
        pen.text((x + 0.16 * px, y + 1 * u), word, px, col, 0.95 * k, "mm", "Bold", 0.32)

    def draw_timer(self, pen, t):
        u = self.u
        sec = max(0, int(math.floor(t + self.t0)))
        s = f"{sec // 60}:{sec % 60:02d}" if sec < 6000 else f"{sec // 3600}:{(sec // 60) % 60:02d}"
        x, y = self.X(0.885), self.Y(0.24)  # below the karaoke caption line
        pen.text((x, y), s, 92 * u, self.th["accent"], 1.0, "rs", "Display", 0.0)
        pen.text((x - 2 * u, y + 34 * u), "TIME GONE", 21 * u, self.th["line"], 0.75, "rm", "Bold", 0.28)

    def ground(self, pen, y, x0=None, x1=None, k=1.0, fade=0.22):
        x0 = self.X(-0.02) if x0 is None else x0
        x1 = self.X(1.02) if x1 is None else x1
        col = self.th["line"] if self.mono else (120, 116, 112)
        pen.fade_line(x0, x1, y, col, 1.7 * self.u, (0.8 if self.mono else 1.0) * k, fade)

    def figure(self, pen, A, k=1.0, **kw):
        if self.delta:
            kw["w_scale"] = kw.get("w_scale", 1.0) * self.th["fig_w"]
            kw.setdefault("joints", bool(self.th.get("joints")))
        draw_figure(pen, A, self.th, self.picto, k, **kw)


# ======================================================================== 1. walk
class LAWalk(LineArt):
    """Figure walks along a horizon; ground marks, distant hills and stars scroll with parallax."""
    focus = (0.47, 0.54)
    star_drift = (-6.0, 0.0)

    def setup(self):
        self.h = 0.17 * self.oh
        self.G = self.Y(0.63)
        self.period = 1.3
        self.gait = Gait(pose_walk, self.h)
        rng = self.rng
        self.tile = 2.0 * self.ow
        self.marks = [(rng.uniform(0, self.tile), rng.uniform(3, 16)) for _ in range(16)]
        self.marks2 = [(rng.uniform(0, self.tile), rng.uniform(6, 30)) for _ in range(9)]
        # distant hills: periodic sum of sines
        self.hill_k = [(int(k), rng.uniform(0, TAU), a) for k, a in ((3, 0.022), (5, 0.012), (11, 0.004))]

    def hill(self, x):
        """Distant ridge line: sharp-crested periodic peaks (|sin| folds)."""
        v = 0.0
        for kk, ph, a in self.hill_k:
            v += a * (1 - abs(math.sin(math.pi * kk * x / self.tile + ph))) ** 1.5
        return v * self.oh

    def draw(self, pen, t):
        u, G = self.u, self.G
        ph = TAU * t / self.period
        off = self.gait.offset(ph)
        # far hills (parallax 0.18)
        xs = np.linspace(self.X(-0.06), self.X(1.06), 160)
        pts = [(x, G - 0.004 * self.oh - self.hill(x + off * 0.18)) for x in xs]
        pen.line(pts, self.th["dim"], 1.3 * u, 0.6 if self.mono else 0.75)
        self.ground(pen, G)
        # ground marks moving with the world (parallax 1) and foreground ones (1.5)
        for wx, ln in self.marks:
            x = self.X(0) + ((wx - off) % self.tile) - 0.4 * self.ow
            e = ease(min(x - self.X(0.0), self.X(1.0) - x) / (0.25 * self.ow))
            pen.seg((x, G + 9 * u), (x + ln * u, G + 9 * u), self.th["dim"], 1.4 * u, 0.9 * e)
        for wx, ln in self.marks2:
            x = self.X(0) + ((wx - off * 1.5) % self.tile) - 0.4 * self.ow
            e = ease(min(x - self.X(0.0), self.X(1.0) - x) / (0.25 * self.ow))
            pen.seg((x, G + 26 * u), (x + ln * u, G + 26 * u), self.th["faint"], 1.4 * u, 1.3 * e)
        A = place(pose_walk(ph), self.h, self.X(0.46), G, 1, self.picto)
        self.figure(pen, A)


# ======================================================================== 2. stairs
class LAStairs(LineArt):
    """Endless line staircase rising to the upper right; steps draw themselves ahead of the climber."""
    focus = (0.47, 0.5)
    star_drift = (-4.0, 2.5)
    label_at = (0.3, 0.30)

    def setup(self):
        self.h = 0.15 * self.oh
        self.sw, self.sh = 0.27 * self.h, 0.16 * self.h
        self.P = 1.05  # seconds per step

    def surface(self, s):
        return -s * self.sh

    def draw(self, pen, t):
        u, h, sw, sh = self.u, self.h, self.sw, self.sh
        s = t / self.P
        n = math.floor(s)
        q = s - n
        # world: step i tread spans x in [i*sw, (i+1)*sw] at y = -i*sh  (y up negative)
        hipx = (s + 0.62) * sw
        camx, camy = hipx, -(s + 0.5) * sh
        ox, oy = self.X(0.47) - camx, self.Y(0.56) - camy

        def S(x, y):
            return (x + ox, y + oy)

        lc = self.th["line"]
        lo, hi = n - 16, n + 9
        # stringer (underside), dim
        end = s + 4.5
        a0, a1 = S(lo * sw, -lo * sh + sh + 0.12 * h), S(end * sw, -end * sh + sh + 0.12 * h)
        pen.seg(a0, a1, self.th["faint"], 1.3 * u, 1.2)
        for i in range(lo, hi):
            ahead = s + 4.5 - i
            grow = clamp(ahead)
            behind = clamp(1 - (s - i - 7) / 7)
            if grow <= 0 or behind <= 0:
                continue
            kk = behind * (0.5 + 0.5 * clamp(ahead))
            x0, y0 = i * sw, -i * sh
            # riser then tread; drawn progressively
            pts = [S(x0, y0 + sh), S(x0, y0), S(x0 + sw, y0)]
            if grow < 1:
                L1 = sh
                L = (sh + sw) * ease_out(grow)
                if L <= L1:
                    pts = [pts[0], (pts[0][0], pts[0][1] - L)]
                else:
                    pts = [pts[0], pts[1], (pts[1][0] + (L - L1), pts[1][1])]
                pen.dot(pts[-1], 2.6 * u, self.th["accent"], 0.9)
            pen.line(pts, lc, 2.0 * u, kk)
        # feet targets: foot A on even steps, B on odd steps
        def foot_world(step_idx):
            return ((step_idx + 0.5) * sw, -step_idx * sh - 0.002 * h)

        lead = "R" if n % 2 == 0 else "L"
        back = "L" if lead == "R" else "R"
        # body rises during the step (eased), forward progress linear
        rise = n + ease_in_out(clamp((q - 0.05) / 0.8))
        hip_y_world = -rise * sh - (THIGH + SHIN) * h * 0.9
        hip_x_world = (n + 0.25 + 0.75 * ease_in_out(q)) * sw + 0.0
        # swinging foot: from step n-1 to n+1
        qs = ease_in_out(clamp(q / 0.72))
        f0, f1 = foot_world(n - 1), foot_world(n + 1)
        sx = lerp(f0[0], f1[0], qs)
        sy = lerp(f0[1], f1[1], qs) - math.sin(math.pi * qs) * 0.85 * sh - (0.4 * sh if qs < 0.5 else 0)
        sy = min(sy, -(math.floor((sx) / sw)) * sh - 0.01 * h) if qs < 1 else sy
        plant = foot_world(n)
        swing = (sx, sy)
        pose = P(lean=0.17, head=0.12, shL=0, shR=0, elL=0.4, elR=0.4)
        arm = 0.32 * math.sin(math.pi * (q + (0 if lead == "R" else 1)))
        pose["sh" + lead] = -arm
        pose["sh" + back] = arm
        pose["el" + lead] = 0.35 + 0.3 * max(0.0, -arm)
        pose["el" + back] = 0.35 + 0.3 * max(0.0, arm)
        pose["ik" + back] = (swing[0] - hip_x_world, swing[1] - hip_y_world)
        pose["ik" + lead] = (plant[0] - hip_x_world, plant[1] - hip_y_world)
        J = fk(pose, h, 1, self.picto)
        hx, hy = S(hip_x_world, hip_y_world)
        A = {k: ((v[0] + hx, v[1] + hy) if isinstance(v, tuple) else v) for k, v in J.items()}
        A.update(h=h, facing=1)
        self.figure(pen, A)


# ======================================================================== 3. clock
class LAClock(LineArt):
    """Huge minimalist clock; hands spin fast (mode=fast) or freeze (mode=stop); rings rotate."""
    focus = (0.5, 0.47)
    glow_size = 0.5
    label_at = None

    def setup(self):
        self.fast = str(self.p.get("mode", "fast")).strip().lower() not in ("stop", "stopped", "freeze", "still", "slow")
        self.R = 0.39 * self.ow
        self.c = (self.X(0.5), self.Y(0.47))
        self.dash_ph = self.rng.uniform(0, 360)

    def draw(self, pen, t):
        u, R, (cx, cy) = self.u, self.R, self.c
        lc, dim = self.th["line"], self.th["dim"]
        spin = 1.0 if self.fast else 0.0
        # outer dashed ring + inner segmented ring, counter-rotating
        r1 = 1.12 * R
        rot1 = self.dash_ph + 7 * t * spin
        for i in range(48):
            a = rot1 + i * 7.5
            pen.arc((cx, cy), r1, a, a + 3.2, dim, 1.4 * u, 0.9)
        r2 = 0.74 * R
        rot2 = -11 * t * spin + 20
        for i in range(3):
            a = rot2 + i * 120
            pen.arc((cx, cy), r2, a, a + 78, self.th["faint"], 1.4 * u, 1.3)
        # face
        pen.circle((cx, cy), R, lc, 2.0 * u, 0.95)
        for i in range(60):
            a = TAU * i / 60
            major = i % 5 == 0
            r0 = R * (0.86 if major else 0.92)
            ca, sa = math.cos(a), math.sin(a)
            pen.seg((cx + ca * r0, cy + sa * r0), (cx + ca * R * 0.965, cy + sa * R * 0.965), lc if major else dim,
                    (2.0 if major else 1.3) * u, 1.0 if major else 0.85)
        # hands
        if self.fast:
            am = TAU * t / 3.2 - math.pi / 2
            ah = TAU * t / 38.0 - math.pi / 2 + 1.2
        else:
            jit = 0.012 * math.sin(t * 23) * (0.5 + 0.5 * math.sin(t * 1.7))
            am = TAU * 58 / 60 - math.pi / 2 + jit
            ah = TAU * (11 + 58 / 60) / 12 - math.pi / 2
        Lm, Lh = 0.8 * R, 0.52 * R
        if self.fast:  # motion-blur ghosts trailing the minute hand
            for j in range(1, 9):
                a = am - j * 0.05
                pen.seg((cx, cy), (cx + math.cos(a) * Lm, cy + math.sin(a) * Lm), lc, 1.6 * u, 0.32 * (1 - j / 9) ** 1.5)
        pen.seg((cx, cy), (cx + math.cos(ah) * Lh, cy + math.sin(ah) * Lh), lc, 4.0 * u, 1.0)
        pen.seg((cx, cy), (cx + math.cos(am) * Lm, cy + math.sin(am) * Lm), self.th["accent"], 2.4 * u, 1.0)
        pen.circle((cx, cy), 7 * u, self.th["accent"], 2.0 * u, 1.0, fill=0 if self.mono else (0, 0, 0))
        # small figure standing on a short line inside the lower half
        G = cy + 0.66 * R
        pen.fade_line(cx - 0.36 * R, cx + 0.36 * R, G, (130, 126, 122) if self.picto else lc, 1.5 * u, 0.9, 0.35)
        pose = pose_stand_front(t) if self.fast else pose_lookup(t)
        A = place(pose, 0.4 * R, cx, G, 1, self.picto)
        self.figure(pen, A)
        if self.label:
            self.chip(pen, cx, cy - 0.4 * R, self.label)


# ======================================================================== 4. wall
class LAWall(LineArt):
    """Figure pushes a tall monolith; cracks spread, it shatters into line fragments, re-forms (5 s loop)."""
    focus = (0.45, 0.5)
    label_at = None
    LOOP = 5.0

    def setup(self):
        rng = self.rng
        self.h = 0.16 * self.oh
        self.G = self.Y(0.66)
        x0, x1 = self.X(0.53), self.X(0.76)
        y0, y1 = self.Y(0.28), self.G
        self.box = (x0, y0, x1, y1)
        cols, rows = 3, 7
        vx = np.linspace(x0, x1, cols + 1)
        vy = np.linspace(y0, y1, rows + 1)
        V = np.zeros((rows + 1, cols + 1, 2))
        for r in range(rows + 1):
            for c in range(cols + 1):
                jx = 0 if c in (0, cols) else rng.uniform(-0.3, 0.3) * (x1 - x0) / cols
                jy = 0 if r in (0, rows) else rng.uniform(-0.32, 0.32) * (y1 - y0) / rows
                V[r, c] = (vx[c] + jx, vy[r] + jy)
        self.V = V
        pose = pose_push(0)
        J = fk(pose, self.h, 1, self.picto)
        self.fig_x = x0 - J["handR"][0] - 0.004 * self.ow
        A = place(pose, self.h, self.fig_x, self.G, 1, self.picto)
        self.contact = A["handR"]
        cx, cy = self.contact
        # crack edges = interior grid edges, ordered by distance from the hands
        edges = []
        for r in range(rows + 1):
            for c in range(cols + 1):
                for dr, dc in ((0, 1), (1, 0)):
                    r2, c2 = r + dr, c + dc
                    if r2 > rows or c2 > cols:
                        continue
                    if (dr == 0 and r in (0, rows)) or (dc == 0 and c in (0, cols)):
                        continue  # outer border
                    a, b = V[r, c], V[r2, c2]
                    da, db = math.hypot(a[0] - cx, a[1] - cy), math.hypot(b[0] - cx, b[1] - cy)
                    if db < da:
                        a, b, da, db = b, a, db, da
                    edges.append((da, db, tuple(a), tuple(b)))
        self.maxd = max(e[1] for e in edges)
        self.edges = edges
        self.frags = []
        for r in range(rows):
            for c in range(cols):
                poly = np.array([V[r, c], V[r, c + 1], V[r + 1, c + 1], V[r + 1, c]])
                cen = poly.mean(0)
                d = (cen - np.array([cx - 0.12 * self.ow, cy])) * np.array([1.0, 0.4])
                d /= np.linalg.norm(d) + 1e-6
                sp = rng.uniform(0.13, 0.26) * self.oh
                vel = d * sp + np.array([rng.uniform(0.02, 0.1), rng.uniform(-0.1, 0.0)]) * self.oh
                self.frags.append((poly - cen, cen, vel, rng.uniform(-1.4, 1.4)))
        self.sparks = [(rng.uniform(-2.0, 0.5), rng.uniform(0.12, 0.3) * self.oh, rng.uniform(0.015, 0.04) * self.oh)
                       for _ in range(12)]

    def draw(self, pen, t):
        u, G = self.u, self.G
        L = self.LOOP
        q = t % L
        x0, y0, x1, y1 = self.box
        lc = self.th["line"]
        self.ground(pen, G)
        crack = ease(clamp((q - 0.35) / 2.25))
        broke = q >= 2.6
        rebuild = ease(clamp((q - 4.3) / 0.6))
        # light beyond the wall when it breaks
        if broke:
            lk = clamp((q - 2.6) / 0.4) * (1 - rebuild)
            pen.soft(self.X(0.66), self.Y(0.5), 0.24 * self.ow, self.th["soft"], (0.55 if self.mono else 0.32) * lk)
        if not broke or rebuild > 0:
            k = 1.0 if not broke else rebuild
            shake = 0.0
            if not broke and q > 1.8:
                shake = 1.6 * u * math.sin(q * 61) * (q - 1.8) / 0.8
            pen.rrect((x0 + shake, y0, x1 + shake, y1), 0, lc, 2.2 * u, k, fill=0 if self.mono else (0, 0, 0))
            pen.seg((x0 + 14 * u + shake, y0 + 14 * u), (x0 + 14 * u + shake, y1 - 0.2 * (y1 - y0)), self.th["dim"], 1.3 * u, 0.6 * k)
            if self.label:
                pen.text(((x0 + x1) / 2 + shake, y0 + 0.2 * (y1 - y0)), self.label, 30 * u, self.th["dim"],
                         k * (1 - 0.6 * crack), "mm", "Bold", 0.35)
            if not broke:
                reach = crack * self.maxd * 1.05
                for da, db, a, b in self.edges:
                    if reach <= da:
                        continue
                    f = clamp((reach - da) / max(1e-6, db - da))
                    e = (a[0] + (b[0] - a[0]) * f + shake, a[1] + (b[1] - a[1]) * f)
                    pen.seg((a[0] + shake, a[1]), e, self.th["accent"] if self.mono else lc, 1.5 * u, 0.9)
        if broke and rebuild < 1:
            tau = q - 2.6
            fade = clamp(1 - tau / 1.5) ** 1.5
            for rel, cen, vel, om in self.frags:
                ee = 1 - math.exp(-tau * 1.4)
                pos = cen + vel * ee / 1.4
                ang = om * ee
                ca, sa = math.cos(ang), math.sin(ang)
                pts = [(pos[0] + x * ca - y * sa, pos[1] + x * sa + y * ca) for x, y in rel * (1 - 0.25 * ee)]
                pen.poly(pts, lc, 1.6 * u, fade)
            cx, cy = self.contact
            for a, sp, ln in self.sparks:
                r0 = sp * ease_out(tau / 0.9)
                kk = clamp(1 - tau / 0.9)
                d = (math.cos(a), math.sin(a))
                pen.seg((cx + d[0] * r0, cy + d[1] * r0), (cx + d[0] * (r0 + ln), cy + d[1] * (r0 + ln)),
                        self.th["accent"], 1.4 * u, kk)
        # figure: push -> stumble forward -> stand tall -> back to push
        if q < 2.6:
            pose, dx = pose_push(q), 0.0
        elif q < 3.3:
            k = ease((q - 2.6) / 0.7)
            pose = mix(pose_push(q), pose_walk(1.2), k)
            dx = 0.07 * self.h * ease_out((q - 2.6) / 0.7)
        elif q < 4.3:
            k = ease((q - 3.3) / 0.8)
            pose, dx = mix(pose_walk(1.2), pose_proud(q), k), 0.07 * self.h
        else:
            k = ease((q - 4.3) / 0.6)
            pose = mix(pose_proud(q), pose_push(q), k)
            dx = 0.07 * self.h * (1 - k)
        A = place(pose, self.h, self.fig_x + dx, G, 1, self.picto)
        self.figure(pen, A)


# ======================================================================== 5. calendar
WEEKDAYS = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]


class LACalendar(LineArt):
    """Calendar pages tear off and fly away above a still figure. params.label / labels = page words."""
    focus = (0.5, 0.48)
    label_at = None
    PAGE = 1.6

    def setup(self):
        words = parse_labels(self.p.get("labels")) or parse_labels(self.p.get("label"))
        self.words = words
        self.npages = len(words) if words else 7
        self.cw, self.chh = 0.46 * self.ow, 0.25 * self.oh
        self.cx, self.cy = self.X(0.5), self.Y(0.375)
        self.G = self.Y(0.685)
        u = self.u
        self.page_box = (self.cx - self.cw / 2 + 16 * u, self.cy - self.chh / 2 + 62 * u,
                         self.cx + self.cw / 2 - 16 * u, self.cy + self.chh / 2 - 16 * u)
        self.day0 = int(self.rng.integers(3, 20))
        self.sprites = [self._page_sprite(i) for i in range(self.npages)]

    def _page_sprite(self, i):
        x0, y0, x1, y1 = self.page_box
        w, h = int((x1 - x0) * SS) + 4, int((y1 - y0) * SS) + 4
        line = Image.new("L", (w, h), 0)
        fill = Image.new("L", (w, h), 0)
        u = self.u
        dl, df = ImageDraw.Draw(line), ImageDraw.Draw(fill)
        r = 10 * u * SS
        df.rounded_rectangle([1, 1, w - 2, h - 2], radius=r, fill=255)
        dl.rounded_rectangle([1, 1, w - 2, h - 2], radius=r, outline=170, width=max(1, int(1.6 * u * SS)))
        if self.words:
            word = self.words[i]
            px = 92 * u * SS
            f = font(px, "Display")
            while f.getlength(word) > w * 0.84 and px > 8:
                px *= 0.9
                f = font(px, "Display")
            dl.text((w / 2, h / 2 + 4 * u * SS), word, font=f, fill=255, anchor="mm")
        else:
            wd = WEEKDAYS[i % 7]
            f1 = font(34 * u * SS, "Bold")
            sp = 0.3 * 34 * u * SS
            tw = sum(f1.getlength(c) for c in wd) + sp * (len(wd) - 1)
            xx = w / 2 - tw / 2
            for ch in wd:
                dl.text((xx, h * 0.2), ch, font=f1, fill=200, anchor="lm")
                xx += f1.getlength(ch) + sp
            f2 = font(190 * u * SS, "Display")
            dl.text((w / 2, h * 0.6), str((self.day0 + i - 1) % 28 + 1), font=f2, fill=255, anchor="mm")
        return line, fill

    def _paste(self, pen, idx, angle, ox, oy, k, scale=1.0):
        line, fill = self.sprites[idx % self.npages]
        x0, y0, x1, y1 = self.page_box
        if abs(angle) > 0.01 or scale != 1.0:
            if scale != 1.0:
                sz = (max(2, int(line.width * scale)), max(2, int(line.height * scale)))
                line, fill = line.resize(sz, Image.BILINEAR), fill.resize(sz, Image.BILINEAR)
            line = line.rotate(angle, Image.BICUBIC, expand=True)
            fill = fill.rotate(angle, Image.BICUBIC, expand=True)
        cxp = ((x0 + x1) / 2 + ox) * SS
        cyp = ((y0 + y1) / 2 + oy) * SS
        pos = (int(cxp - line.width / 2), int(cyp - line.height / 2))
        black = 0 if self.mono else (0, 0, 0)
        pen.img.paste(black, pos, fill)
        m = line if k >= 0.999 else line.point(lambda v: int(v * k))
        pen.img.paste(pen.c(self.th["line"] if self.mono else self.th["accent"]), pos, m)

    def draw(self, pen, t):
        u = self.u
        cx, cy, cw, chh = self.cx, self.cy, self.cw, self.chh
        lc, dim = self.th["line"], self.th["dim"]
        # board, header and binder rings
        x0, y0, x1, y1 = cx - cw / 2, cy - chh / 2, cx + cw / 2, cy + chh / 2
        pen.rrect((x0, y0, x1, y1), 16 * u, lc, 2.0 * u, 0.9)
        pen.seg((x0, y0 + 46 * u), (x1, y0 + 46 * u), dim, 1.5 * u, 1.0)
        for fx in (0.28, 0.72):
            rx = x0 + fx * cw
            pen.rrect((rx - 7 * u, y0 - 22 * u, rx + 7 * u, y0 + 18 * u), 7 * u, lc, 2.0 * u, 1.0,
                      fill=0 if self.mono else (0, 0, 0))
        # stacked page edges under the top page (thickness cue)
        px0, py0, px1, py1 = self.page_box
        n = math.floor(t / self.PAGE)
        q = t / self.PAGE - n
        # page beneath (next) and top page
        self._paste(pen, n + 1, 0, 0, 0, 1.0)
        fl = ease_in_out(clamp((q - 0.25) / 0.75))
        if q < 0.25:
            lift = ease(q / 0.25)
            self._paste(pen, n, 2.5 * lift, -3 * u * lift, -8 * u * lift, 1.0)
        elif fl < 1:
            ang = 2.5 + 38 * fl
            dx = 0.42 * self.ow * fl ** 1.2
            dy = -0.2 * self.oh * fl + 0.06 * self.oh * math.sin(math.pi * fl)
            self._paste(pen, n, ang, dx, dy, (1 - fl) ** 1.4, 1 - 0.3 * fl)
        # still figure looking up at the calendar
        self.ground(pen, self.G, self.X(0.2), self.X(0.8), 1.0, 0.35)
        A = place(pose_stand_front(t), 0.12 * self.oh, cx, self.G, 1, self.picto)
        self.figure(pen, A)


# ======================================================================== 6. crowd
class LACrowd(LineArt):
    """Small figures walk past in both directions while the protagonist stands (or sits) still."""
    focus = (0.5, 0.55)
    label_at = (0.5, 0.405)

    def setup(self):
        rng = self.rng
        self.sit = str(self.p.get("pose", "stand")).strip().lower() in ("sit", "sitting", "bench", "box", "seated")
        self.lanes = [(self.Y(0.6), 0.09 * self.oh, 0.3 if self.mono else 0.42),
                      (self.Y(0.665), 0.118 * self.oh, 0.5 if self.mono else 0.62)]
        self.hero_h = 0.145 * self.oh
        self.crowd_col = None if self.mono else (150, 142, 132)
        self.walkers = []
        span = self.ow * 1.3
        for li, (gy, h, k) in enumerate(self.lanes):
            n = 7 if li == 0 else 6
            for j in range(n):
                per = rng.uniform(1.0, 1.25)
                self.walkers.append((li, -1 if rng.random() < 0.5 else 1, rng.uniform(0, span), per,
                                     rng.uniform(0, TAU), Gait(pose_walk, h)))
        self.span = span

    def draw(self, pen, t):
        u = self.u
        for li, (gy, h, k) in enumerate(self.lanes):
            self.ground(pen, gy, k=0.55 if li == 0 else 0.9)
            for wl, d, x0, per, ph0, gait in self.walkers:
                if wl != li:
                    continue
                ph = ph0 + TAU * t / per
                x = (x0 + d * gait.offset(ph) - gait.offset(ph0)) % self.span + self.X(0.5) - self.span / 2
                edge = ease(min(x - self.X(-0.05), self.X(1.05) - x) / (0.15 * self.ow))
                A = place(pose_walk(ph), h, x, gy, d, self.picto)
                self.figure(pen, A, k * edge, far_k=0.6, col=self.crowd_col)
        gy = self.lanes[1][0]
        hp = self.hero_h
        if self.sit:
            bx = self.X(0.49)
            by = gy - seat_height(hp, self.picto)
            pen.seg((bx - 0.1 * self.ow, by), (bx + 0.1 * self.ow, by), self.th["line"], 2.2 * u, 0.9)
            for sx in (-0.085, 0.085):
                pen.seg((bx + sx * self.ow, by), (bx + sx * self.ow, gy), self.th["dim"], 1.8 * u, 1.0)
            A = seated(pose_sit(t, 0.38), hp, bx - 0.02 * self.ow, by, gy, 1, self.picto)
        else:
            A = place(pose_stand_front(t), hp, self.X(0.5), gy + 0.004 * self.oh, 1, self.picto)
        self.figure(pen, A)


# ======================================================================== 7. fall & rise
class LAFallRise(LineArt):
    """Run, stumble, kneel, get back up and run on (6.5 s loop) with a dotted motion trail."""
    focus = (0.47, 0.56)
    LOOP = 6.5
    label_at = (0.5, 0.71)

    def setup(self):
        self.h = 0.16 * self.oh
        self.G = self.Y(0.645)
        self.vrun = 1.15 * self.h / 0.36  # px/s, matches the run cycle stride roughly
        ts = np.linspace(0, self.LOOP, 1301)
        v = np.array([self._speed(x) for x in ts])
        d = np.concatenate([[0], np.cumsum((v[1:] + v[:-1]) * 0.5 * np.diff(ts))])
        self._ts, self._d = ts, d
        self.D = d[-1]
        rng = self.rng
        self.tile = 2.0 * self.ow
        self.marks = [(rng.uniform(0, self.tile), rng.uniform(4, 18)) for _ in range(16)]

    def _speed(self, q):
        if q < 1.7:
            return self.vrun
        if q < 2.3:
            return self.vrun * (1 - ease_out((q - 1.7) / 0.6))
        if q < 4.4:
            return 0.0
        if q < 5.1:
            return self.vrun * ease((q - 4.4) / 0.7)
        return self.vrun

    def world(self, t):
        n = math.floor(t / self.LOOP)
        return n * self.D + float(np.interp(t - n * self.LOOP, self._ts, self._d))

    def pose_at(self, t):
        q = t % self.LOOP
        ph = TAU * t / 0.72
        if q < 1.7:
            return pose_run(ph)
        if q < 2.35:
            return mix(pose_run(ph), pose_fallen(q), ease((q - 1.7) / 0.6))
        if q < 2.95:
            return pose_fallen(q)
        if q < 3.5:
            return mix(pose_fallen(q), pose_kneel(q), ease((q - 2.95) / 0.55))
        if q < 3.9:
            return pose_kneel(q)
        if q < 4.6:
            return mix(pose_kneel(q), pose_stand(q), ease_in_out((q - 3.9) / 0.7))
        if q < 5.1:
            return mix(pose_stand(q), pose_run(ph), ease((q - 4.6) / 0.5))
        return pose_run(ph)

    def draw(self, pen, t):
        u, G, h = self.u, self.G, self.h
        wx = self.world(t)
        fx = self.X(0.47)
        self.ground(pen, G)
        for mx, ln in self.marks:
            x = self.X(0) + ((mx - wx) % self.tile) - 0.4 * self.ow
            e = ease(min(x - self.X(0.0), self.X(1.0) - x) / (0.25 * self.ow))
            pen.seg((x, G + 10 * u), (x + ln * u, G + 10 * u), self.th["dim"], 1.4 * u, 0.9 * e)
        # dotted motion trail of the head over the last ~1.3 s
        n = 26
        for j in range(n, 0, -1):
            tj = t - j * 0.05
            Aj = place(self.pose_at(tj), h, fx + (self.world(tj) - wx), G, 1, self.picto)
            hx, hy = Aj["head"]
            age = j / n
            pen.dot((hx, hy), (3.2 - 1.8 * age) * u, self.th["accent"] if self.mono else self.th["fig"],
                    0.75 * (1 - age) ** 1.4)
        A = place(self.pose_at(t), h, fx, G, 1, self.picto)
        self.figure(pen, A)


# ======================================================================== 8. door
class LADoor(LineArt):
    """A glowing doorway opens in the dark; the figure (seen from behind) walks towards the light."""
    focus = (0.5, 0.47)
    glow_size = 0.45
    label_at = None

    def setup(self):
        self.hz = self.Y(0.47)          # horizon / eye level
        self.door_b = self.Y(0.565)     # door threshold on the floor
        self.dw, self.dh = 0.2 * self.ow, 0.24 * self.oh
        self.dx = self.X(0.5)
        self.rays = np.linspace(-1, 1, 11)

    def draw(self, pen, t):
        u = self.u
        x0, x1 = self.dx - self.dw / 2, self.dx + self.dw / 2
        yb, yt = self.door_b, self.door_b - self.dh
        lc, acc = self.th["line"], self.th["accent"]
        appear = ease_out(t / 1.1)
        opn = ease_in_out((t - 0.6) / 2.0)
        breathe = 0.85 + 0.15 * math.sin(t * 1.4)
        light = opn * breathe
        # light on the floor: rays fanning towards the viewer
        for i, r in enumerate(self.rays):
            xa = self.dx + r * self.dw * 0.42
            xb = self.dx + r * self.ow * 1.05
            yb2 = self.Y(0.80)
            k = light * (0.55 - 0.25 * abs(r)) * (1.0 if i % 2 == 0 else 0.6)
            steps = 12
            for s in range(steps):
                f0, f1 = s / steps, (s + 1) / steps
                pa = (lerp(xa, xb, f0), lerp(yb, yb2, f0))
                pb = (lerp(xa, xb, f1), lerp(yb, yb2, f1))
                pen.seg(pa, pb, lc, 1.3 * u, k * (1 - f0) ** 1.6, caps=False)
        sk = 1.0 if self.mono else 0.45
        pen.soft(self.dx, yb - 0.55 * self.dh, 0.45 * self.dw, self.th["soft"], 0.95 * light * sk)
        pen.soft(self.dx, yb + 0.05 * self.oh, 0.55 * self.dw, self.th["soft"], 0.3 * light * sk)
        # floor line
        self.ground(pen, yb, self.X(0.0), self.X(1.0), 0.8 * appear, 0.4)
        # door interior glow lines (vertical light strands) + frame
        if light > 0:
            for i in range(1, 8):
                xx = x0 + i * self.dw / 8
                pen.seg((xx, yt + 10 * u), (xx, yb - 2 * u), acc, 1.1 * u, 0.22 * light)
        # frame drawn progressively (perimeter growth)
        per = [(x0, yb), (x0, yt), (x1, yt), (x1, yb)]
        L = [self.dh, self.dw, self.dh]
        tot = sum(L) * appear
        pts = [per[0]]
        for i in range(3):
            if tot <= 0:
                break
            f = clamp(tot / L[i])
            a, b = per[i], per[i + 1]
            pts.append((lerp(a[0], b[0], f), lerp(a[1], b[1], f)))
            tot -= L[i]
        pen.line(pts, acc, 2.6 * u, 1.0)
        pen.line([(p[0] + (-9 * u if p[0] < self.dx else 9 * u), p[1] + (0 if p[1] == yb else -9 * u)) for p in pts],
                 lc, 1.3 * u, 0.45 * appear)
        # door leaf swinging open (hinged on the left)
        if opn < 0.995:
            w = self.dw * (1 - opn)
            skew = 0.08 * self.dh * opn
            pen.poly([(x0, yb), (x0, yt), (x0 + w, yt + skew), (x0 + w, yb - skew)], lc, 1.6 * u, appear,
                     fill=0 if self.mono else (0, 0, 0))
        # spill rays around the frame
        for i in range(14):
            a = -math.pi * (i + 0.5) / 14
            ca, sa = math.cos(a), math.sin(a)
            rx, ry = self.dx + ca * self.dw * 0.62, yt + 0.45 * self.dh + sa * self.dh * 0.62
            ln = (26 + 14 * math.sin(t * 1.3 + i * 1.7)) * u
            pen.seg((rx, ry), (rx + ca * ln, ry + sa * ln), acc, 1.3 * u, 0.5 * light)
        if self.label:
            self.chip(pen, self.dx, yt - 60 * u, self.label, appear)
        # the figure, from behind, walking into depth
        z = 1.0 + 1.45 * (1 - math.exp(-max(0.0, t) / 5.0))
        gy = self.hz + (self.Y(0.735) - self.hz) / z
        hh = 0.2 * self.oh / z
        amp = 1 - ease((t - 9.0) / 2.0)  # slows to a stop in front of the light
        ph = TAU * t / 1.3
        s = math.sin(ph) * amp
        pose = mix(pose_stand_front(t),
                   P(front=True, thL=0.1, thR=0.1, shL=0.3 + 0.05 * s, shR=0.3 - 0.05 * s, elL=-0.14, elR=-0.14,
                     liftL=max(0.0, s) * 0.22, liftR=max(0.0, -s) * 0.22, sway=0.012 * s, dy=-0.006 * abs(s)), amp)
        A = place(pose, hh, self.dx + 0.02 * self.ow / z, gy, 1, self.picto)
        self.figure(pen, A, 1.0)


# ======================================================================== 9. mountain
class LAMountain(LineArt):
    """Line-drawn mountain; the figure ascends a zig-zag path to a glowing summit flag; slow zoom."""
    focus = (0.5, 0.42)
    label_at = None
    DUR = 11.0

    def setup(self):
        rng = self.rng
        X, Y = self.X, self.Y
        self.peak = (X(0.52), Y(0.305))
        self.ridge = [(X(-0.06), Y(0.665)), (X(0.1), Y(0.6)), (X(0.2), Y(0.575)), (X(0.3), Y(0.49)), (X(0.37), Y(0.455)),
                      (X(0.44), Y(0.37)), self.peak, (X(0.6), Y(0.36)), (X(0.66), Y(0.41)), (X(0.73), Y(0.43)),
                      (X(0.82), Y(0.53)), (X(0.92), Y(0.58)), (X(1.06), Y(0.65))]
        self.back = [(X(-0.06), Y(0.53)), (X(0.12), Y(0.43)), (X(0.22), Y(0.47)), (X(0.3), Y(0.44)), (X(0.4), Y(0.5)),
                     (X(0.66), Y(0.47)), (X(0.8), Y(0.38)), (X(0.9), Y(0.43)), (X(1.06), Y(0.47))]
        self.inner = [[(X(0.6), Y(0.36)), (X(0.615), Y(0.41)), (X(0.6), Y(0.45))],
                      [(X(0.37), Y(0.455)), (X(0.385), Y(0.5))],
                      [(X(0.73), Y(0.43)), (X(0.74), Y(0.48)), (X(0.76), Y(0.51))]]
        self.G = Y(0.665)
        # zig-zag path: switchbacks narrowing towards the summit
        pts = [(X(0.3), Y(0.665))]
        n = 7
        for i in range(1, n + 1):
            f = i / n
            y = lerp(Y(0.655), self.peak[1] + 0.012 * self.oh, f)
            half = lerp(0.2, 0.012, f ** 0.9) * self.ow
            x = self.peak[0] + (half if i % 2 else -half) * (1 if i < n else 0)
            pts.append((x + rng.uniform(-6, 6) * self.u * (i < n), y))
        self.path = np.array(pts)
        self.cum = np.concatenate([[0], np.cumsum(np.hypot(*(self.path[1:] - self.path[:-1]).T))])

    def tf(self, zoom, pts):
        cx, cy = self.X(0.5), self.Y(0.44)
        return [(cx + (x - cx) * zoom, cy + (y - cy) * zoom) for x, y in pts]

    def draw(self, pen, t):
        u = self.u
        zoom = 1.0 + 0.12 * ease_in_out(t / 14.0)
        T = lambda pts: self.tf(zoom, pts)  # noqa: E731
        lc, dim = self.th["line"], self.th["dim"]
        pen.line(T(self.back), dim, 1.4 * u, 0.75)
        fillc = 0 if self.mono else (0, 0, 0)
        pen.poly(T(self.ridge + [(self.X(1.06), self.Y(0.9)), (self.X(-0.06), self.Y(0.9))]), None, fill=fillc)
        for seg in self.inner:
            pen.line(T(seg), dim, 1.3 * u, 0.8)
        pen.line(T(self.ridge), lc, 2.1 * u, 1.0)
        self.ground(pen, self.tf(zoom, [(0, self.G)])[0][1], k=0.6)
        # path: dashed ahead, solid behind
        prog = ease_in_out(clamp(t / self.DUR)) * 0.985 + 0.015
        sdist = prog * self.cum[-1]
        path = T(self.path.tolist())
        P_ = np.array(path)
        cum = self.cum * zoom
        pen.dashed(P_, dim, 1.5 * u, 1.0, 7 * u, 9 * u)
        done = [tuple(p) for p, c in zip(P_, cum) if c < sdist * zoom]
        here, d = polyline_at(P_, cum, sdist * zoom)
        pen.line(done + [tuple(here)], self.th["accent"], 2.0 * u, 0.9)
        # summit flag with glow
        px, py = T([self.peak])[0]
        pole = 0.06 * self.oh * zoom
        top = (px, py - pole)
        pen.seg((px, py), top, lc, 2.0 * u, 1.0)
        # waving pennant: upper and lower edges converge to the tip, with a travelling wave
        fw, fh = 0.065 * self.ow * zoom, 0.03 * self.oh * zoom
        upper, lower = [], []
        for i in range(11):
            f = i / 10
            wv = 0.12 * fh * math.sin(t * 3.2 - f * 5.0) * f
            upper.append((px + f * fw, top[1] + 0.5 * fh * f + wv))
            lower.append((px + f * fw, top[1] + fh - 0.5 * fh * f + wv))
        pen.poly(upper + lower[::-1], self.th["accent"], 1.6 * u, 1.0, fill=self.th["accent"], kf=0.55 if self.mono else 0.8)
        pen.soft(px + 0.4 * fw, top[1] + 0.5 * fh, 0.05 * self.ow, self.th["soft"], 0.8)
        if self.label:
            pen.text((px + 0.075 * self.ow * zoom, top[1] + 0.013 * self.oh * zoom), self.label, 22 * u,
                     self.th["text"], 0.9, "lm", "Bold", 0.3)
        # climber
        hh = 0.05 * self.oh * zoom
        if t < self.DUR + 0.3:
            facing = 1 if d[0] >= 0 else -1
            pose = pose_walk(TAU * t / 1.1)
            pose["lean"] += 0.12
            A = place(pose, hh, here[0], here[1], facing, self.picto)
        else:
            k = ease((t - self.DUR - 0.3) / 0.8)
            pose = mix(pose_stand_front(t), pose_victory(t), k)
            A = place(pose, hh, here[0], here[1], 1, self.picto)
        self.figure(pen, A)


# ======================================================================== 10. money
class LAMoney(LineArt):
    """Coin stacks grow step by step (compounding) next to the figure; a growth curve draws itself."""
    focus = (0.55, 0.55)
    label_at = None

    def setup(self):
        self.G = self.Y(0.665)
        self.h = 0.15 * self.oh
        n = 7
        self.cw = 0.095 * self.ow
        self.gap = 0.125 * self.cw
        self.xs = [self.X(0.37) + i * (self.X(0.885) - self.X(0.37)) / (n - 1) for i in range(n)]
        self.counts = [int(round(2.0 * 1.55 ** i)) for i in range(n)]
        self.start = [0.15 + i * 0.6 for i in range(n)]
        self.dt = [min(0.12, 0.55 / c) for c in self.counts]
        self.done = [st + c * d for st, c, d in zip(self.start, self.counts, self.dt)]
        rng = self.rng
        self.spark = [(rng.uniform(-0.7, 0.7), rng.uniform(0.0, 1.0), rng.uniform(0, 2.4)) for _ in range(5)]
        # smooth exponential growth curve passing above the final stack tops
        eh = self.cw * 0.16
        xs = np.array(self.xs)
        tops = np.array([self.G - eh - (c - 1) * self.gap - eh for c in self.counts])
        lift = 0.035 * self.oh
        k = math.log(max(1e-3, (self.G - tops[-1] + lift)) / max(1e-3, self.G - tops[0] + lift)) / (xs[-1] - xs[0])
        a0 = self.G - tops[0] + lift
        cx = np.linspace(self.X(0.3), self.X(0.93), 140)
        cy = self.G - a0 * np.exp(k * (cx - xs[0]))
        self.curve = np.stack([cx, cy], 1)

    def draw(self, pen, t):
        u = self.u
        lc, acc = self.th["line"], self.th["accent"]
        self.ground(pen, self.G)
        blk = 0 if self.mono else (0, 0, 0)
        coin_col = lc if self.mono else acc
        ew, eh = self.cw / 2, self.cw * 0.16
        built_x = self.X(0.3)
        tops = []
        for i, (x, cnt) in enumerate(zip(self.xs, self.counts)):
            built = (t - self.start[i]) / self.dt[i]
            if built <= 0:
                continue
            nb = int(clamp(math.floor(built), 0, cnt))
            for c in range(min(cnt, nb + 1)):
                y = self.G - eh - c * self.gap
                k = 1.0
                if c == nb:  # the coin currently dropping in
                    f = built - nb
                    y -= (1 - ease_out(f)) * 0.05 * self.oh
                    k = ease(f * 1.5)
                pen.ellipse((x - ew, y - eh, x + ew, y + eh), coin_col, 1.7 * u, 0.95 * k, fill=blk)
            if nb > 0:
                ty = self.G - eh - (nb - 1) * self.gap
                pen.ellipse((x - ew * 0.62, ty - eh * 0.62, x + ew * 0.62, ty + eh * 0.62), coin_col, 1.1 * u, 0.45)
                tops.append((x, ty - eh, self.done[i]))
            built_x = max(built_x, x + (self.xs[1] - self.xs[0]) * 0.5 * clamp(built / cnt))
        # compound growth curve, drawn up to the stack being built
        C = self.curve
        m = C[:, 0] <= built_x
        if m.sum() >= 2:
            pts = [tuple(p) for p in C[m]]
            pen.line(pts, acc, 2.4 * u, 1.0)
            hx, hy = pts[-1]
            pen.dot((hx, hy), 5.5 * u, acc, 1.0)
            pen.soft(hx, hy, 0.03 * self.ow, self.th["soft"], 0.7)
        if self.label:
            self.chip(pen, self.X(0.62), self.Y(0.36), self.label, ease((t - 0.5) / 0.6))
        # "+" sparkles above completed stacks
        for x, y, tdone in tops:
            if t < tdone:
                continue
            for ox, oy, ph in self.spark[:3]:
                age = (t - tdone + ph) % 2.4
                if age > 1.1:
                    continue
                k = math.sin(math.pi * age / 1.1)
                sx, sy = x + ox * self.cw, y - (0.015 + 0.03 * oy) * self.oh - age * 0.025 * self.oh
                sz = (4 + 4 * k) * u
                pen.seg((sx - sz, sy), (sx + sz, sy), acc, 1.6 * u, k)
                pen.seg((sx, sy - sz), (sx, sy + sz), acc, 1.6 * u, k)
        pose = mix(pose_stand(t), pose_lookup(t), ease((t - 1.0) / 1.2))
        A = place(pose, self.h, self.X(0.2), self.G, 1, self.picto)
        self.figure(pen, A)


# ======================================================================== 11. phone
class LAPhone(LineArt):
    """Figure sits hunched over a glowing phone while a time ring around him drains."""
    focus = (0.5, 0.53)
    label_at = (0.5, 0.75)

    def setup(self):
        self.G = self.Y(0.665)
        self.h = 0.22 * self.oh
        self.c = (self.X(0.5), self.Y(0.525))
        self.R = 0.37 * self.ow
        rng = self.rng
        self.notes = [(rng.uniform(-1, 1), rng.uniform(0.5, 1.0), rng.uniform(0, 3.0)) for _ in range(9)]

    def draw(self, pen, t):
        u = self.u
        lc, acc, dim = self.th["line"], self.th["accent"], self.th["dim"]
        cx, cy = self.c
        R = self.R
        # time ring: dim track + draining bright arc + end dot
        pen.circle((cx, cy), R, self.th["faint"], 1.6 * u, 1.3)
        for i in range(12):
            a = TAU * i / 12
            pen.dot((cx + math.cos(a) * (R + 20 * u), cy + math.sin(a) * (R + 20 * u)), 2.0 * u, dim, 1.0)
        frac = clamp(0.93 - 0.045 * t, 0.06, 1)
        a0 = -90
        a1 = a0 + 360 * frac
        pen.arc((cx, cy), R, a0, a1, acc, 3.0 * u, 1.0)
        ea = math.radians(a1)
        ex, ey = cx + math.cos(ea) * R, cy + math.sin(ea) * R
        pen.dot((ex, ey), 6 * u, acc, 1.0)
        pen.soft(ex, ey, 0.03 * self.ow, self.th["soft"], 0.6)
        # box seat + ground
        h = self.h
        self.ground(pen, self.G, self.X(0.1), self.X(0.9), 1.0, 0.3)
        hipx = self.X(0.45)
        seat = self.G - seat_height(h, self.picto)
        pen.rrect((hipx - 0.15 * h, seat, hipx + 0.07 * h, self.G), 4 * u, lc, 1.8 * u, 0.75,
                  fill=0 if self.mono else (0, 0, 0))
        # hunched sitting figure, feet on the ground (IK), hands holding the phone up to the face
        b = math.sin(t * 1.1)
        pose = pose_sit(t, 0.55)
        pose.update(head=0.85 + 0.02 * b, shR=0.62, shL=0.72, elR=1.2, elL=1.1)
        A = seated(pose, h, hipx, seat, self.G, 1, self.picto, feet=(0.27, 0.22))
        self.figure(pen, A)
        # the phone (in front of the hands), tilted towards the face
        hx = (A["handR"][0] + A["handL"][0]) / 2 + (0.006 if self.mono else 0.022) * self.ow
        hy = (A["handR"][1] + A["handL"][1]) / 2 - 0.012 * self.oh
        pw, ph = 0.032 * self.ow, 0.032 * self.oh
        ang = math.radians(-55)
        ca, sa = math.cos(ang), math.sin(ang)

        def rot(px, py):
            return (hx + px * ca - py * sa, hy + px * sa + py * ca)

        corners = [rot(-pw / 2, -ph / 2), rot(pw / 2, -ph / 2), rot(pw / 2, ph / 2), rot(-pw / 2, ph / 2)]
        screen = acc if self.mono else (236, 228, 214)  # cold screen light reads against the gold figure
        pen.poly(corners, screen, 2.0 * u, 1.0, fill=0 if self.mono else (0, 0, 0))
        for j in range(4):  # scrolling content lines
            yy = ((j * 0.25 - t * 0.6) % 1.0) * 0.8 - 0.4
            k = math.sin(math.pi * (yy + 0.4) / 0.8)
            pen.seg(rot(-pw * 0.3, yy * ph), rot(pw * 0.3, yy * ph), lc, 1.4 * u, 0.7 * k)
        flick = 0.85 + 0.15 * math.sin(t * 7.3) * math.sin(t * 2.1)
        pen.soft(hx, hy - 0.01 * self.oh, 0.05 * self.ow, self.th["soft"] if self.mono else (200, 190, 175),
                 (1.0 if self.mono else 0.45) * flick)
        # notification dots rising from the phone
        for ox, sp, ph0 in self.notes:
            age = (t * 0.5 + ph0) % 1.5
            if age > 1.0:
                continue
            k = math.sin(math.pi * age)
            nx = hx + ox * 0.08 * self.ow + 0.03 * self.ow * age * ox
            ny = hy - 0.02 * self.oh - age * sp * 0.12 * self.oh
            pen.circle((nx, ny), 4.5 * u, acc, 1.5 * u, k)
            pen.dot((nx, ny), 1.8 * u, acc, k)


# ======================================================================== 12. mirror
class LAMirror(LineArt):
    """Figure faces a mirror that shows a brighter, upright future self fading in."""
    focus = (0.6, 0.5)
    label_at = None

    def setup(self):
        self.G = self.Y(0.665)
        self.h = 0.175 * self.oh
        self.mx = self.X(0.665)
        self.mw, self.mh = 0.31 * self.ow, 0.37 * self.oh

    def draw(self, pen, t):
        u = self.u
        lc, acc, dim = self.th["line"], self.th["accent"], self.th["dim"]
        G = self.G
        self.ground(pen, G)
        x0, x1 = self.mx - self.mw / 2, self.mx + self.mw / 2
        yb, yt = G - 0.012 * self.oh, G - self.mh
        r = self.mw / 2
        fade = ease_in_out((t - 0.3) / 2.6)
        # arch frame (double line), feet
        def arch(inset):
            a, b = x0 + inset, x1 - inset
            rr = r - inset
            pts = [(a, yb - inset * 0.3)]
            for i in range(25):
                ang = math.pi + math.pi * i / 24
                pts.append((self.mx + math.cos(ang) * rr, yt + r + math.sin(ang) * rr))
            pts.append((b, yb - inset * 0.3))
            return pts
        outer, inner = arch(0), arch(13 * u)
        pen.poly(outer, None, fill=0 if self.mono else (0, 0, 0))
        pen.soft(self.mx, yt + 0.55 * self.mh, 0.2 * self.ow, self.th["soft"], (0.35 if self.mono else 0.14) * fade)
        pen.line(outer + [outer[0]], lc, 2.2 * u, 1.0)
        pen.line(inner + [inner[0]], dim, 1.3 * u, 0.9)
        for fx in (-0.32, 0.32):
            pen.seg((self.mx + fx * self.mw, yb), (self.mx + fx * self.mw * 1.15, G), lc, 2.0 * u, 0.9)
        # glare strokes
        gx, gy = x0 + 0.2 * self.mw, yt + 0.3 * self.mh
        pen.seg((gx, gy + 0.05 * self.mh), (gx + 0.12 * self.mw, gy - 0.04 * self.mh), lc, 1.6 * u, 0.35)
        pen.seg((gx + 0.02 * self.mw, gy + 0.1 * self.mh), (gx + 0.09 * self.mw, gy + 0.05 * self.mh), lc, 1.6 * u, 0.25)
        # future self inside, facing left, upright and brighter
        fh = self.h * 0.95
        fy = yb - 0.03 * self.oh
        pen.fade_line(self.mx - 0.3 * self.mw, self.mx + 0.3 * self.mw, fy, dim, 1.3 * u, 0.9 * fade, 0.35)
        A2 = place(pose_proud(t), fh, self.mx + 0.02 * self.ow, fy, -1, self.picto)
        pen.soft(self.mx, fy - 0.55 * fh, 0.11 * self.ow, self.th["soft"], (0.55 if self.mono else 0.16) * fade)
        col2 = acc if self.mono else self.th["fig"]
        self.figure(pen, A2, fade, col=col2)
        if self.label:
            self.chip(pen, self.mx, yt - 0.045 * self.oh, self.label, max(0.25, fade))
        # present self, hunched, dimmer
        A = place(pose_hunched(t), self.h, self.X(0.25), G, 1, self.picto)
        self.figure(pen, A, 0.72 if self.mono else 0.62)


# ======================================================================== shared helpers for the story scenes
def map_A(A: dict, fn) -> dict:
    """Apply ``fn(x, y) -> (x, y)`` to every joint of a placed figure."""
    out = {k: (fn(*v) if isinstance(v, tuple) else v) for k, v in A.items()}
    return out


def rotate_A(A: dict, ang: float, pivot, to=None) -> dict:
    """Rotate a placed figure by ``ang`` radians (screen coords, y down) around ``pivot``; optionally move
    the pivot to ``to``."""
    ca, sa = math.cos(ang), math.sin(ang)
    px, py = pivot
    tx, ty = to if to is not None else pivot
    return map_A(A, lambda x, y: (tx + (x - px) * ca - (y - py) * sa, ty + (x - px) * sa + (y - py) * ca))


def arm_reach(A: dict, side: str, target, bend: int = -1):
    """Two-bone IK: put the hand of arm ``side`` on ``target`` (elbow bends back/down by default)."""
    h, f = A["h"], A.get("facing", 1)
    S = A["sh" + side]
    E, Hd = _ik(S, target, UPPER_ARM * h, FOREARM * h, bend * f)
    A["el" + side], A["hand" + side] = E, Hd


def leg_reach(A: dict, side: str, target):
    h, f = A["h"], A.get("facing", 1)
    K, F = _ik(A["hp" + side], target, THIGH * h, SHIN * h, f)
    A["knee" + side], A["foot" + side] = K, F
    A["toe" + side] = (F[0] + f * FOOT * h, F[1])


def pose_lie(t=0.0):
    """Lying on the back (use with rotate_A by -90 deg): arms along the body, one knee slightly up."""
    b = math.sin(t * 1.1)
    return P(lean=0.0, head=-0.05 + 0.02 * b, thL=0.05, thR=-0.02, knL=0.1, knR=0.05,
             shL=0.12, shR=0.05, elL=0.1, elR=0.1)


def pose_cheer(t=0.0):
    b = math.sin(t * 9.0)
    return P(front=True, head=-0.15, thL=0.2, thR=0.2, shL=2.45 + 0.18 * b, shR=2.45 - 0.18 * b,
             elL=0.25, elR=0.25, dy=-0.02 * abs(b))


def pose_think(t=0.0):
    b = math.sin(t * 1.8)
    return P(lean=0.02, head=-0.25 + 0.03 * b, thL=0.04, thR=-0.04, knL=0.03, knR=0.02,
             shL=0.1, shR=-0.35, elL=0.2, elR=2.3)


def fit_text_px(pen, s, max_w, px, weight="Display", track=0.0):
    w = pen.text_width(s, px, weight, track)
    return px if w <= max_w else px * max_w / max(1e-6, w)


class LAStory(LineArt):
    """Base for the faster "story" scenes: a small library of line-drawn props shared between them."""

    def col(self, key):
        return self.th[key]

    @property
    def acc(self):
        return self.th["accent"]

    @property
    def lc(self):
        return self.th["line"]

    def coin(self, pen, x, y, r, k=1.0, col=None, spin=0.0):
        """Coin seen from the front (``spin`` squashes it horizontally like a flipping coin)."""
        col = self.acc if col is None else col
        u = self.u
        sx = max(0.15, abs(math.cos(spin)))
        pen.ellipse((x - r * sx, y - r, x + r * sx, y + r), col, 1.8 * u, k, fill=0)
        if sx > 0.45:
            pen.ellipse((x - r * 0.62 * sx, y - r * 0.62, x + r * 0.62 * sx, y + r * 0.62), col, 1.1 * u, 0.7 * k)
            if r > 13 * u:
                pen.text((x, y + 0.5 * u), "$", r * 0.95 * sx + 1, col, k, "mm", "Bold")
            else:
                pen.seg((x, y - r * 0.4), (x, y + r * 0.4), col, 1.3 * u, k)

    def coin_stack(self, pen, x, base, n, w, k=1.0, col=None, drop=None):
        """Stack of ``n`` coins seen slightly from above; ``drop`` = (fraction) animates the top coin falling in."""
        col = self.acc if col is None else col
        u = self.u
        ew, eh = w / 2, w * 0.16
        gap = w * 0.13
        for c in range(n):
            y = base - eh - c * gap
            kk = k
            if drop is not None and c == n - 1:
                y -= (1 - ease_out(drop)) * 0.06 * self.oh
                kk = k * ease(drop * 2)
            pen.ellipse((x - ew, y - eh, x + ew, y + eh), col, 1.7 * u, kk, fill=0)
        if n:
            ty = base - eh - (n - 1) * gap
            pen.ellipse((x - ew * 0.6, ty - eh * 0.6, x + ew * 0.6, ty + eh * 0.6), col, 1.1 * u, 0.5 * k)
        return base - eh * 2 - max(0, n - 1) * gap

    def bulb(self, pen, x, y, s, k=1.0, rays=0.0, t=0.0, col=None):
        """Light bulb icon (``s`` = bulb radius) with optional rays."""
        col = self.acc if col is None else col
        u = self.u
        if k <= 0.01:
            return
        pen.circle((x, y), s, col, 2.0 * u, k, fill=0)
        pen.line([(x - 0.42 * s, y + 0.75 * s), (x - 0.38 * s, y + 1.25 * s), (x + 0.38 * s, y + 1.25 * s),
                  (x + 0.42 * s, y + 0.75 * s)], col, 1.8 * u, k)
        pen.seg((x - 0.33 * s, y + 1.45 * s), (x + 0.33 * s, y + 1.45 * s), col, 1.8 * u, k)
        pen.line([(x - 0.3 * s, y + 0.3 * s), (x - 0.12 * s, y - 0.25 * s), (x, y + 0.1 * s), (x + 0.12 * s, y - 0.25 * s),
                  (x + 0.3 * s, y + 0.3 * s)], col, 1.3 * u, 0.8 * k)
        if rays > 0.01:
            for i in range(8):
                a = -math.pi / 2 + (i - 3.5) * 0.42
                r0 = s * (1.45 + 0.12 * math.sin(t * 9 + i))
                r1 = r0 + s * 0.55 * rays
                pen.seg((x + math.cos(a) * r0, y + math.sin(a) * r0), (x + math.cos(a) * r1, y + math.sin(a) * r1),
                        col, 1.8 * u, k * rays)
            pen.soft(x, y, s * 2.4, self.th["soft"], 0.7 * k * rays)

    def book_open(self, pen, x, y, w, k=1.0, flip=None, col=None, tilt=0.0):
        """Open book seen from the front (spine at x, bottom at y); ``flip`` in [0, 1) turns a page."""
        col = self.lc if col is None else col
        u = self.u
        hgt = w * 0.62
        ca, sa = math.cos(tilt), math.sin(tilt)

        def R(px, py):
            return (x + px * ca - py * sa, y + px * sa + py * ca)
        for sgn in (-1, 1):
            pts = [R(0, 0), R(sgn * w * 0.5, -0.07 * w), R(sgn * w * 0.5, -hgt), R(0, -hgt + 0.07 * w)]
            pen.poly(pts, col, 1.8 * u, k, fill=0)
            for j in range(4):
                yy = -hgt * (0.25 + 0.17 * j)
                pen.seg(R(sgn * w * 0.08, yy + 0.02 * w), R(sgn * w * 0.4, yy - 0.01 * w * sgn * 0 - 0.0), self.th["dim"],
                        1.2 * u, 0.8 * k)
        if flip is not None and 0 < flip < 1:
            c = math.cos(math.pi * ease_in_out(flip))
            px = c * w * 0.5
            lift = math.sin(math.pi * flip) * 0.12 * w
            pts = [R(0, 0), R(px, -0.07 * w - lift), R(px, -hgt - lift), R(0, -hgt + 0.07 * w)]
            pen.poly(pts, col, 1.6 * u, k, fill=0)

    def book_closed(self, pen, x, y, w, hh, k=1.0, col=None):
        col = self.lc if col is None else col
        u = self.u
        pen.rrect((x - w / 2, y - hh, x + w / 2, y), 2 * u, col, 1.7 * u, k, fill=0)
        pen.seg((x - w / 2 + 5 * u, y - hh * 0.5), (x + w / 2 - 3 * u, y - hh * 0.5), self.th["dim"], 1.1 * u, k)

    def bag(self, pen, hand, s, k=1.0, col=None, sway=0.0):
        """Shopping bag hanging from a hand point."""
        col = self.lc if col is None else col
        u = self.u
        hx, hy = hand
        top = hy + 0.35 * s
        x0, x1 = hx - 0.5 * s + sway, hx + 0.5 * s + sway
        pen.line([(x0 + 0.25 * s, top), (hx, hy), (x1 - 0.25 * s, top)], col, 1.6 * u, k)
        pen.poly([(x0, top), (x1, top), (x1 + 0.06 * s, top + 1.1 * s), (x0 - 0.06 * s, top + 1.1 * s)], col, 1.9 * u, k,
                 fill=0)
        pen.text(((x0 + x1) / 2, top + 0.58 * s), "$", 0.5 * s, col, 0.8 * k, "mm", "Bold")

    def zzz(self, pen, x, y, t, k=1.0):
        u = self.u
        for i in range(3):
            age = (t * 0.7 + i / 3) % 1.0
            kk = math.sin(math.pi * age) * k
            px = (26 + 16 * i) * u
            pen.text((x + age * 70 * u + 8 * u * math.sin(t * 3 + i), y - age * 150 * u), "Z", px * (0.7 + 0.5 * age),
                     self.th["text"], kk, "mm", "Bold")

    def sun(self, pen, x, y, r, t, k=1.0, col=None, horizon=None):
        """Sun disk (clipped by a horizon line if given) with rotating rays."""
        col = self.acc if col is None else col
        u = self.u
        if horizon is None:
            pen.circle((x, y), r, col, 2.2 * u, k)
            a0, a1 = 0, TAU
        else:
            pts = []
            for i in range(25):
                a = math.pi + math.pi * i / 24
                pts.append((x + math.cos(a) * r, y + math.sin(a) * r))
            pen.line(pts, col, 2.2 * u, k)
            a0, a1 = math.pi, TAU
        n = 12
        for i in range(n):
            a = a0 + (a1 - a0) * (i + 0.5) / n + 0.15 * math.sin(t * 0.8)
            if horizon is not None and math.sin(a) > -0.08:
                continue
            ln = r * (0.35 + 0.12 * math.sin(t * 5 + i * 1.7))
            r0 = r * 1.22
            pen.seg((x + math.cos(a) * r0, y + math.sin(a) * r0), (x + math.cos(a) * (r0 + ln), y + math.sin(a) * (r0 + ln)),
                    col, 2.0 * u, k)
        pen.soft(x, y, r * 2.2, self.th["soft"], 0.6 * k)

    def cloud(self, pen, x, y, s, k=1.0, col=None, fill=True):
        """Bumpy cloud outline (union of circles)."""
        col = self.lc if col is None else col
        u = self.u
        blobs = [(-0.62, 0.12, 0.36), (-0.25, -0.12, 0.48), (0.2, -0.2, 0.52), (0.6, 0.08, 0.38), (0.0, 0.18, 0.4)]
        w = 2.0 * u * pen.ws
        for bx, by, br in blobs:
            pen.circle((x + bx * s, y + by * s), br * s + w * 0.5, col, 2.0 * u, k)
        if fill:
            for bx, by, br in blobs:
                pen.circle((x + bx * s, y + by * s), br * s - w * 0.6, fill=0)

    def house(self, pen, x, g, s, k=1.0, col=None):
        """Small house outline standing on the ground at y=g (s = width). Returns chimney top."""
        col = self.lc if col is None else col
        u = self.u
        x0, x1 = x - s / 2, x + s / 2
        wall = g - 0.62 * s
        roof = wall - 0.42 * s
        pen.poly([(x0, g), (x0, wall), (x, roof), (x1, wall), (x1, g)], col, 2.2 * u, k, fill=0)
        pen.seg((x0 - 0.08 * s, wall + 0.06 * s), (x, roof - 0.05 * s), col, 2.2 * u, k)
        pen.seg((x, roof - 0.05 * s), (x1 + 0.08 * s, wall + 0.06 * s), col, 2.2 * u, k)
        pen.rrect((x - 0.1 * s, g - 0.32 * s, x + 0.1 * s, g), 2 * u, col, 1.8 * u, k)
        pen.rrect((x0 + 0.12 * s, wall + 0.12 * s, x0 + 0.3 * s, wall + 0.28 * s), 2 * u, col, 1.6 * u, k)
        pen.rrect((x1 - 0.3 * s, wall + 0.12 * s, x1 - 0.12 * s, wall + 0.28 * s), 2 * u, col, 1.6 * u, k)
        cx = x0 + 0.72 * s
        cy = wall - 0.42 * s * 0.44 * 0.62
        pen.poly([(cx, cy + 0.02 * s), (cx, roof + 0.06 * s), (cx + 0.11 * s, roof + 0.06 * s), (cx + 0.11 * s, cy + 0.1 * s)],
                 col, 1.8 * u, k, fill=0)
        return (cx + 0.055 * s, roof + 0.06 * s)

    def car(self, pen, x, g, s, k=1.0, col=None, t=0.0):
        """Low flashy sports car profile (s = length), wheels on y=g."""
        col = self.lc if col is None else col
        u = self.u
        r = 0.105 * s
        wy = g - r
        x0, x1 = x - s / 2, x + s / 2
        body = [(x0 + 0.02 * s, wy), (x0, wy - 0.1 * s), (x0 + 0.05 * s, wy - 0.17 * s), (x0 + 0.3 * s, wy - 0.2 * s),
                (x0 + 0.45 * s, wy - 0.34 * s), (x0 + 0.68 * s, wy - 0.33 * s), (x0 + 0.84 * s, wy - 0.2 * s),
                (x1, wy - 0.15 * s), (x1, wy - 0.02 * s), (x1 - 0.02 * s, wy)]
        pen.poly(body, col, 2.2 * u, k, fill=0)
        pen.line([(x0 + 0.46 * s, wy - 0.3 * s), (x0 + 0.66 * s, wy - 0.3 * s), (x0 + 0.78 * s, wy - 0.2 * s),
                  (x0 + 0.36 * s, wy - 0.2 * s), (x0 + 0.46 * s, wy - 0.3 * s)], col, 1.5 * u, 0.8 * k)
        pen.seg((x0 + 0.08 * s, wy - 0.1 * s), (x0 + 0.92 * s, wy - 0.1 * s), self.th["dim"], 1.3 * u, k)
        for wx in (x0 + 0.2 * s, x1 - 0.2 * s):
            pen.circle((wx, wy), r, col, 2.2 * u, k, fill=0)
            pen.circle((wx, wy), r * 0.45, col, 1.4 * u, 0.8 * k)
            a = t * 6
            pen.seg((wx + math.cos(a) * r * 0.45, wy + math.sin(a) * r * 0.45),
                    (wx - math.cos(a) * r * 0.45, wy - math.sin(a) * r * 0.45), col, 1.2 * u, 0.7 * k)
        for i in range(3):  # shine sparkles
            sx = x0 + (0.3 + 0.2 * i) * s
            sy = wy - 0.26 * s - 0.08 * s * math.sin(i * 2.1)
            kk = max(0.0, math.sin(t * 4 + i * 2.2))
            sz = 0.035 * s * kk
            pen.seg((sx - sz, sy), (sx + sz, sy), self.acc, 1.4 * u, k * kk)
            pen.seg((sx, sy - sz), (sx, sy + sz), self.acc, 1.4 * u, k * kk)

    def bed(self, pen, x0, x1, top, g, k=1.0, col=None):
        col = self.lc if col is None else col
        u = self.u
        th = 0.028 * self.oh
        pen.rrect((x0, top, x1, top + th), 6 * u, col, 2.0 * u, k, fill=0)
        pen.seg((x0 + 8 * u, top + th), (x0 + 8 * u, g), col, 2.0 * u, k)
        pen.seg((x1 - 8 * u, top + th), (x1 - 8 * u, g), col, 2.0 * u, k)
        pen.rrect((x0 - 0.02 * self.ow, top - 0.07 * self.oh, x0, g), 5 * u, col, 2.0 * u, k, fill=0)
        pen.ellipse((x0 + 0.01 * self.ow, top - 0.022 * self.oh, x0 + 0.11 * self.ow, top + 0.004 * self.oh), col, 1.7 * u, k,
                    fill=0)

    def tv(self, pen, x, g, s, t, k=1.0, col=None):
        col = self.lc if col is None else col
        u = self.u
        x0, x1 = x - s / 2, x + s / 2
        y1 = g - 0.35 * s
        y0 = y1 - 0.62 * s
        pen.rrect((x0, y0, x1, y1), 8 * u, col, 2.2 * u, k, fill=0)
        pen.rrect((x0 + 0.06 * s, y0 + 0.07 * s, x1 - 0.06 * s, y1 - 0.07 * s), 4 * u, self.th["dim"], 1.4 * u, k)
        for i in range(5):  # flickering scan lines
            yy = y0 + 0.12 * s + ((i / 5 + t * 1.7) % 1.0) * 0.38 * s
            ln = 0.2 + 0.5 * abs(math.sin(t * 13 + i * 2.3))
            pen.seg((x - ln * 0.35 * s, yy), (x + ln * 0.35 * s, yy), col, 1.3 * u, 0.55 * k)
        pen.line([(x - 0.12 * s, g), (x, y1), (x + 0.12 * s, g)], col, 1.8 * u, k)
        fl = 0.75 + 0.25 * math.sin(t * 17) * math.sin(t * 5.3)
        pen.soft(x, (y0 + y1) / 2, 0.55 * s, self.th["soft"], 0.45 * k * fl)
        return (x, (y0 + y1) / 2)

    def desk(self, pen, x0, x1, top, g, k=1.0, col=None):
        col = self.lc if col is None else col
        u = self.u
        pen.seg((x0, top), (x1, top), col, 2.4 * u, k)
        pen.seg((x0 + 10 * u, top), (x0 + 10 * u, g), col, 2.0 * u, k)
        pen.seg((x1 - 10 * u, top), (x1 - 10 * u, g), col, 2.0 * u, k)

    def laptop(self, pen, x, top, s, t, k=1.0, facing=1, col=None):
        """Laptop on a desk at y=top, screen hinged at x and opened towards ``-facing``."""
        col = self.lc if col is None else col
        u = self.u
        pen.seg((x - facing * 0.05 * s, top - 2 * u), (x + facing * 0.85 * s, top - 2 * u), col, 2.4 * u, k)
        tip = (x - facing * 0.18 * s, top - 0.78 * s)
        pen.seg((x, top - 2 * u), tip, col, 2.4 * u, k)
        pen.soft((x + tip[0]) / 2 - facing * 0.15 * s, (top + tip[1]) / 2, 0.45 * s, self.th["soft"],
                 0.35 * k * (0.85 + 0.15 * math.sin(t * 7)))

    def chair(self, pen, hip_x, seat, g, facing, h, k=1.0, col=None):
        col = self.lc if col is None else col
        u = self.u
        f = facing
        back_x = hip_x - f * 0.12 * h
        front_x = hip_x + f * 0.2 * h
        pen.seg((back_x, seat), (front_x, seat), col, 2.2 * u, k)
        pen.seg((back_x, seat - 0.36 * h), (back_x, g), col, 2.2 * u, k)
        pen.seg((front_x, seat), (front_x, g), col, 2.0 * u, k)

    def sparkle(self, pen, x, y, sz, k=1.0, col=None):
        col = self.acc if col is None else col
        if k <= 0.01:
            return
        u = self.u
        pen.seg((x - sz, y), (x + sz, y), col, 1.7 * u, k)
        pen.seg((x, y - sz), (x, y + sz), col, 1.7 * u, k)

    def arrow(self, pen, a, b, col, w, k=1.0, head=None):
        u = self.u
        head = 18 * u if head is None else head
        pen.seg(a, b, col, w, k)
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        for s in (-1, 1):
            aa = ang + math.pi + s * 0.5
            pen.seg(b, (b[0] + math.cos(aa) * head, b[1] + math.sin(aa) * head), col, w, k)

    def big_label(self, pen, x, y, word, px, col, k=1.0, under=None):
        """Large caps label with letter spacing and an optional underline in ``under`` colour."""
        if not word or k <= 0.01:
            return
        u = self.u
        px = fit_text_px(pen, word, 0.42 * self.ow, px, "Display", 0.06)
        pen.text((x, y), word, px, col, k, "mm", "Display", 0.06)
        if under is not None:
            tw = pen.text_width(word, px, "Display", 0.06)
            pen.seg((x - tw / 2, y + 0.62 * px), (x + tw / 2, y + 0.62 * px), under, 3.2 * u, k)


# ======================================================================== 13. split screen (poor vs rich)
SPLIT_ACTIONS = ("phone", "spend", "sleep", "car", "complain", "learn")
SPLIT_ALIASES = {"phone": "phone", "scroll": "phone", "read": "phone", "reading": "phone", "book": "phone",
                 "spend": "spend", "shopping": "spend", "shop": "spend", "money": "spend", "save": "spend",
                 "saving": "spend", "invest": "spend", "sleep": "sleep", "wake": "sleep", "morning": "sleep",
                 "run": "sleep", "gym": "sleep", "car": "car", "house": "car", "asset": "car", "assets": "car",
                 "complain": "complain", "build": "complain", "excuses": "complain", "work": "complain",
                 "learn": "learn", "tv": "learn", "study": "learn", "laptop": "learn", "skills": "learn"}


class LASplit(LAStory):
    """Vertical split screen: LEFT (bad habit, dim) vs RIGHT (good habit, glowing). params.action picks
    the contrast, params.left / params.right the two headings (default POOR / RICH)."""
    focus = (0.5, 0.5)
    label_at = None

    def setup(self):
        p = self.p
        a = str(p.get("action", "phone") if isinstance(p.get("action"), str) else "phone").strip().lower()
        self.action = SPLIT_ALIASES.get(a, "phone")
        self.left = clean_word(p.get("left")) or "POOR"
        self.right = clean_word(p.get("right")) or "RICH"
        self.G = self.Y(0.625)
        self.h = 0.15 * self.oh
        self.xl, self.xr = self.X(0.25), self.X(0.75)
        rng = self.rng
        self.coins = [(rng.uniform(0, 1), rng.uniform(-0.5, 0.5), rng.uniform(0.6, 1.0)) for _ in range(7)]
        self.rain = [(rng.uniform(-1, 1), rng.uniform(0, 1)) for _ in range(14)]
        self.hatch_ph = rng.uniform(0, 1)
        dimc = np.array(self.th["fig"], np.float32) * 0.5 + np.array(self.th["dim"], np.float32) * 0.5
        self.lcol = tuple(int(v) for v in dimc)  # desaturated "bad side" colour
        self.lk = 0.85

    def draw(self, pen, t):
        u = self.u
        X, Y = self.X, self.Y
        lc, acc = self.lc, self.acc
        intro = ease_out(t / 0.35)
        # divider + VS badge
        top, bot = Y(0.19), Y(0.72)
        mid = X(0.5)
        yy = lerp(top + (bot - top) * 0.5, top, intro)
        yb = lerp(top + (bot - top) * 0.5, bot, intro)
        pen.seg((mid, yy), (mid, yb), self.th["dim"], 2.0 * u, 1.0)
        hy = Y(0.3) if self.show_timer else Y(0.255)  # gold: leave the timer corner free
        vy = hy
        pen.circle((mid, vy), 30 * u, lc, 2.0 * u, intro, fill=0)
        pen.text((mid, vy + 1 * u), "VS", 24 * u, self.th["text"], intro, "mm", "Display")
        # gloomy hatching on the left, light rays on the right
        for i in range(9):
            f = (i / 9 + 0.04 * t + self.hatch_ph) % 1.0
            x0 = X(0.02) + f * 0.46 * self.ow
            pen.seg((x0, Y(0.34)), (x0 - 0.06 * self.ow, Y(0.43)), self.th["faint"], 1.3 * u, 0.55 * intro)
        pen.soft(self.xr, Y(0.47), 0.2 * self.ow, self.th["soft"], 0.28)
        # headings
        pen.text((self.xl, hy), self.left, fit_text_px(pen, self.left, 0.34 * self.ow, 68 * u, "Display", 0.06),
                 self.lcol, 0.9 * intro, "mm", "Display", 0.06)
        rp = fit_text_px(pen, self.right, 0.34 * self.ow, 68 * u, "Display", 0.06)
        pen.text((self.xr, hy), self.right, rp, acc, intro, "mm", "Display", 0.06)
        tw = pen.text_width(self.right, rp, "Display", 0.06)
        sw = ease_out((t - 0.25) / 0.4)
        pen.seg((self.xr - tw / 2, hy + 0.6 * rp), (self.xr - tw / 2 + tw * sw, hy + 0.6 * rp), acc, 3.4 * u, sw)
        # grounds
        G = self.G
        pen.fade_line(X(0.03), X(0.47), G, self.th["dim"], 1.7 * u, 0.9, 0.18)
        pen.fade_line(X(0.53), X(0.97), G, lc, 1.7 * u, 0.9, 0.18)
        getattr(self, "_l_" + self.action)(pen, t)
        getattr(self, "_r_" + self.action)(pen, t)

    # ---------------------------------------------------------------- helpers
    def figL(self, pen, A, k=1.0, **kw):
        self.figure(pen, A, k * self.lk, col=self.lcol, **kw)

    def figR(self, pen, A, k=1.0, **kw):
        self.figure(pen, A, k, **kw)

    def _lying(self, pose, hip, h):
        A = place(pose, h, 0.0, 0.0, 1, self.picto)
        return rotate_A(A, -math.pi / 2, A["hip"], hip)

    # ---------------------------------------------------------------- phone / book
    def _l_phone(self, pen, t):
        u, h, G = self.u, self.h, self.G
        x0, x1 = self.xl - 0.2 * self.ow, self.xl + 0.19 * self.ow
        top = G - 0.07 * self.oh
        self.bed(pen, x0, x1, top, G, self.lk, self.lcol)
        hip = (self.xl + 0.02 * self.ow, top - 0.006 * self.oh)
        pose = pose_lie(t)
        pose.update(shL=1.75, shR=1.55, elL=0.95, elR=1.1, knR=0.9, thR=0.5)
        A = self._lying(pose, hip, h)
        # phone held above the face, thumb flicking
        fl = math.sin(t * 11) * 0.5 + 0.5
        ph = (A["head"][0] + 0.07 * h, A["head"][1] - 0.2 * h)
        arm_reach(A, "R", (ph[0] + 0.02 * h, ph[1] + 0.02 * h + 0.012 * h * fl), bend=1)
        arm_reach(A, "L", (ph[0] - 0.02 * h, ph[1] + 0.03 * h), bend=1)
        self.figL(pen, A)
        pw, phh = 0.06 * h, 0.12 * h
        pen.rrect((ph[0] - phh / 2, ph[1] - pw / 2 - 0.03 * h, ph[0] + phh / 2, ph[1] + pw / 2 - 0.03 * h), 4 * u,
                  self.th["text"], 2.2 * u, 1.0, fill=0)
        for j in range(3):
            yy = ph[1] - 0.03 * h + ((j / 3 + t * 2.0) % 1.0 - 0.5) * pw * 0.7
            pen.seg((ph[0] - phh * 0.3, yy), (ph[0] + phh * 0.3, yy), self.th["text"], 1.2 * u, 0.6)
        pen.soft(ph[0], ph[1] + 0.03 * h, 0.06 * self.ow, self.th["soft"], 0.55 * (0.8 + 0.2 * fl))
        # blanket over the legs
        hx, hy = A["hip"]
        kx, ky = A["kneeR"]
        fx = max(A["footL"][0], A["footR"][0])
        bl = [(hx - 0.05 * h, top + 0.03 * self.oh), (hx - 0.06 * h, hy - 0.04 * h), (kx, ky - 0.045 * h),
              (fx + 0.04 * h, hy - 0.035 * h), (fx + 0.07 * h, top + 0.03 * self.oh)]
        pen.poly(bl, self.lcol, 2.0 * u, self.lk, fill=0)
        # endless scroll icons
        for i in range(3):
            age = (t * 0.9 + i / 3) % 1.0
            pen.rrect((ph[0] - 0.02 * self.ow + 40 * u * i - 40 * u, ph[1] - 0.06 * self.oh - age * 0.1 * self.oh,
                       ph[0] + 0.02 * self.ow + 40 * u * i - 40 * u, ph[1] - 0.06 * self.oh - age * 0.1 * self.oh + 22 * u),
                      5 * u, self.lcol, 1.4 * u, math.sin(math.pi * age) * 0.8)

    def _r_phone(self, pen, t):
        u, h, G = self.u, self.h, self.G
        f = -1
        hipx = self.xr + 0.06 * self.ow
        seat = G - seat_height(h, self.picto)
        self.chair(pen, hipx, seat, G, f, h)
        pose = pose_sit(t, 0.12)
        pose.update(head=0.32 + 0.02 * math.sin(t * 2))
        A = seated(pose, h, hipx, seat, G, f, self.picto)
        bx, by = A["hip"][0] + f * 0.4 * h, A["hip"][1] - 0.2 * h
        arm_reach(A, "R", (bx + 0.1 * h, by - 0.02 * h))
        arm_reach(A, "L", (bx - 0.1 * h, by - 0.02 * h))
        self.figR(pen, A)
        q = (t / 0.8) % 1.0
        self.book_open(pen, bx, by, 0.32 * h, 1.0, q if q < 0.6 else None, tilt=0.2, col=self.acc)
        # ideas popping above the head
        hx, hy = A["head"]
        cyc = (t / 1.6) % 1.0
        pop = ease_out(cyc / 0.25) * (1 - ease((cyc - 0.8) / 0.2))
        self.bulb(pen, hx - 0.02 * self.ow, hy - 0.22 * h - 0.02 * h * cyc, 0.07 * h * (0.6 + 0.4 * pop), pop, pop, t)

    # ---------------------------------------------------------------- spend / save
    def _l_spend(self, pen, t):
        u, h, G = self.u, self.h, self.G
        ph = TAU * t / 0.8
        pose = pose_walk(ph)
        pose.update(shL=0.08 * math.sin(ph), shR=-0.08 * math.sin(ph), elL=0.05, elR=0.05)
        A = place(pose, h, self.xl + 0.04 * self.ow, G, -1, self.picto)
        self.figL(pen, A)
        for side, ss in (("L", 1.0), ("R", 0.85)):
            self.bag(pen, A["hand" + side], 0.2 * h * ss, self.lk, self.lcol, 0.01 * h * math.sin(ph + 1))
        # coins flying out of the pocket and away
        hx, hy = A["hip"]
        for i, (ph0, dx, sp) in enumerate(self.coins):
            age = (t * 1.3 * sp + ph0) % 1.0
            x = hx + (0.1 + 0.9 * age) * 0.2 * self.ow * (1 if i % 2 else 0.7)
            y = hy - 0.05 * h - math.sin(math.pi * min(1, age * 1.4)) * 0.18 * h + age * age * 0.25 * h + dx * 10 * u
            self.coin(pen, x, y, 10 * u, math.sin(math.pi * age) * 0.95, self.acc, spin=t * 9 + i)
        pen.text((hx + 0.12 * self.ow, hy - 0.42 * h), "-$", 44 * u, self.th["accent2"] if self.theme_name != "paper" else self.acc,
                 0.6 + 0.4 * abs(math.sin(t * 5)), "mm", "Display")

    def _r_spend(self, pen, t):
        u, h, G = self.u, self.h, self.G
        sx = self.xr - 0.05 * self.ow
        per = 0.42
        n = 3 + int(min(12, t / per))
        q = (t / per) % 1.0 if t / per < 12 else 1.0
        topy = self.coin_stack(pen, sx, G, n, 0.11 * self.ow, 1.0, self.acc, drop=q)
        A = place(pose_stand(t), h, self.xr + 0.12 * self.ow, G, -1, self.picto)
        hand = (sx + 0.03 * self.ow, min(topy - 0.08 * self.oh, A["sh" + "R"][1] + 0.1 * h))
        bob = 0.015 * h * math.sin(TAU * q)
        arm_reach(A, "R", (hand[0], hand[1] + bob))
        self.figR(pen, A)
        if q < 0.35:  # next coin in the fingers
            self.coin(pen, hand[0], hand[1] + 14 * u + bob, 9 * u, 1.0, spin=t * 5)
        for i in range(3):
            age = (t * 0.9 + i / 3) % 1.0
            self.sparkle(pen, sx + (i - 1) * 0.05 * self.ow, topy - 0.03 * self.oh - age * 0.06 * self.oh, 8 * u,
                         math.sin(math.pi * age))
        # rising arrow
        k = ease_out((t - 0.3) / 0.6)
        self.arrow(pen, (self.xr - 0.17 * self.ow, G - 0.03 * self.oh),
                   (self.xr - 0.17 * self.ow + 0.0, G - 0.03 * self.oh - 0.16 * self.oh * k), self.acc, 2.6 * u, k)

    # ---------------------------------------------------------------- sleep / sunrise run
    def _l_sleep(self, pen, t):
        u, h, G = self.u, self.h, self.G
        x0, x1 = self.xl - 0.2 * self.ow, self.xl + 0.19 * self.ow
        top = G - 0.07 * self.oh
        self.bed(pen, x0, x1, top, G, self.lk, self.lcol)
        hip = (self.xl + 0.02 * self.ow, top - 0.006 * self.oh)
        A = self._lying(pose_lie(t), hip, h)
        self.figL(pen, A)
        hx, hy = A["hip"]
        fx = max(A["footL"][0], A["footR"][0])
        br = 0.012 * h * math.sin(t * 2.2)
        sh = A["neck"]
        bl = [(sh[0] - 0.02 * h, top + 0.03 * self.oh), (sh[0] - 0.02 * h, sh[1] - 0.08 * h - br),
              (hx, hy - 0.1 * h - br), (fx + 0.06 * h, hy - 0.07 * h), (fx + 0.09 * h, top + 0.03 * self.oh)]
        pen.poly(bl, self.lcol, 2.0 * u, self.lk, fill=0)
        for j in range(3):
            xx = lerp(sh[0], fx, 0.25 + 0.25 * j)
            pen.seg((xx, hy - 0.06 * h), (xx + 0.04 * h, top + 0.01 * self.oh), self.th["faint"], 1.3 * u, 0.8)
        hd = A["head"]
        self.zzz(pen, hd[0] + 0.02 * self.ow, hd[1] - 0.12 * h, t, 0.9)
        # moon
        mx, my = self.xl - 0.12 * self.ow, self.Y(0.36)
        pen.circle((mx, my), 0.035 * self.ow, self.lcol, 2.0 * u, 0.8)
        pen.circle((mx + 0.018 * self.ow, my - 0.01 * self.ow), 0.03 * self.ow, fill=0)

    def _r_sleep(self, pen, t):
        u, h, G = self.u, self.h, self.G
        self.sun(pen, self.xr + 0.03 * self.ow, G, 0.09 * self.ow, t, 1.0, horizon=G)
        ph = TAU * t / 0.55
        A = place(pose_run(ph), h * 0.95, self.xr - 0.05 * self.ow, G, 1, self.picto)
        self.figR(pen, A)
        for i in range(6):  # speed lines + ground marks
            f = (i / 6 + t * 1.8) % 1.0
            x = self.X(0.97) - f * 0.44 * self.ow
            e = math.sin(math.pi * f)
            pen.seg((x, G + 12 * u), (x + 22 * u, G + 12 * u), self.th["dim"], 1.6 * u, e)
        hx, hy = A["neck"]
        for j in range(3):
            yy = hy + (j - 1) * 0.08 * h
            ln = (30 + 15 * math.sin(t * 9 + j)) * u
            pen.seg((hx - 0.25 * h - ln, yy), (hx - 0.25 * h, yy), self.acc, 1.8 * u, 0.7)

    # ---------------------------------------------------------------- car / house
    def _l_car(self, pen, t):
        u, h, G = self.u, self.h, self.G
        cx = self.xl + 0.05 * self.ow
        self.car(pen, cx, G, 0.34 * self.ow, self.lk, self.lcol, t)
        A = place(pose_proud(t), h * 0.9, self.xl - 0.17 * self.ow, G, 1, self.picto)
        self.figL(pen, A)
        # money draining from under the car
        for i, (ph0, dx, sp) in enumerate(self.coins):
            age = (t * 1.1 * sp + ph0) % 1.0
            x = cx + dx * 0.2 * self.ow
            y = G - 0.02 * self.oh + age * 0.09 * self.oh
            self.coin(pen, x, y, 8 * u, (1 - age) * 0.85, self.acc, spin=t * 6 + i)
        # value line going down
        k = ease_out((t - 0.2) / 0.8)
        pts = [(self.xl - 0.15 * self.ow + f * 0.3 * self.ow,
                self.Y(0.36) + (f ** 1.6) * 0.09 * self.oh + 0.008 * self.oh * math.sin(f * 17)) for f in np.linspace(0, k, 14)]
        if len(pts) > 1 and k > 0.05:
            pen.line(pts, self.th["accent2"] if self.theme_name != "paper" else self.acc, 2.4 * u, 0.85)
            self.arrow(pen, pts[-2], pts[-1], self.th["accent2"] if self.theme_name != "paper" else self.acc, 2.4 * u, 0.85,
                       14 * u)

    def _r_car(self, pen, t):
        u, h, G = self.u, self.h, self.G
        hx = self.xr + 0.06 * self.ow
        ch = self.house(pen, hx, G, 0.22 * self.ow, 1.0, self.lc)
        A = place(pose_proud(t), h * 0.9, self.xr - 0.15 * self.ow, G, 1, self.picto)
        self.figR(pen, A)
        # coins popping out of the chimney and piling up
        for i, (ph0, dx, sp) in enumerate(self.coins[:5]):
            age = (t * 0.9 + i / 5) % 1.0
            x = ch[0] + age * (0.1 + 0.05 * dx) * self.ow
            y = ch[1] - math.sin(math.pi * min(1.0, age * 1.2)) * 0.07 * self.oh + age * 0.02 * self.oh
            self.coin(pen, x, y, 11 * u, math.sin(math.pi * age), spin=t * 5 + i)
        n = 2 + int(min(8, t / 0.5))
        self.coin_stack(pen, self.xr + 0.25 * self.ow, G, n, 0.075 * self.ow, 1.0, drop=(t / 0.5) % 1.0 if t < 4 else None)
        k = ease_out((t - 0.2) / 0.8)
        pts = [(self.xr - 0.17 * self.ow + f * 0.32 * self.ow,
                self.Y(0.39) - (f ** 1.6) * 0.08 * self.oh) for f in np.linspace(0, k, 14)]
        if k > 0.05:
            pen.line(pts, self.acc, 2.6 * u, 1.0)
            self.arrow(pen, pts[-2], pts[-1], self.acc, 2.6 * u, 1.0, 15 * u)

    # ---------------------------------------------------------------- complain / build
    def _l_complain(self, pen, t):
        u, h, G = self.u, self.h, self.G
        A = place(pose_hunched(t), h, self.xl, G, 1, self.picto)
        arm_reach(A, "R", (A["neck"][0] + 0.2 * h, A["neck"][1] - 0.05 * h + 0.03 * h * math.sin(t * 10)))
        self.figL(pen, A)
        hx, hy = A["head"]
        cx, cy = hx - 0.02 * self.ow, hy - 0.42 * h
        self.cloud(pen, cx, cy, 0.09 * self.ow, 0.9, self.lcol)
        for i, (dx, ph0) in enumerate(self.rain[:10]):
            f = (t * 2.2 + ph0) % 1.0
            x = cx + dx * 0.07 * self.ow - f * 8 * u
            y = cy + 0.05 * self.ow + f * 0.3 * h
            pen.seg((x, y), (x - 5 * u, y + 16 * u), self.th["dim"], 1.4 * u, math.sin(math.pi * f))
        flash = max(0.0, math.sin(t * 3.1) * math.sin(t * 7.7))
        if flash > 0.45:
            pts = [(cx + 0.01 * self.ow, cy + 0.04 * self.ow), (cx - 0.01 * self.ow, cy + 0.09 * self.ow),
                   (cx + 0.012 * self.ow, cy + 0.09 * self.ow), (cx - 0.006 * self.ow, cy + 0.14 * self.ow)]
            pen.line(pts, self.th["accent2"], 2.4 * u, 1.0)
        # speech bubble with an angry scribble
        bx, by = hx + 0.13 * self.ow, hy - 0.08 * h
        pen.ellipse((bx - 0.075 * self.ow, by - 0.035 * self.ow, bx + 0.075 * self.ow, by + 0.035 * self.ow), self.lcol,
                    1.8 * u, 0.9, fill=0)
        pen.line([(bx - 0.06 * self.ow, by + 0.025 * self.ow), (bx - 0.085 * self.ow, by + 0.06 * self.ow),
                  (bx - 0.04 * self.ow, by + 0.032 * self.ow)], self.lcol, 1.8 * u, 0.9)
        pts = []
        for i in range(13):
            f = i / 12
            pts.append((bx - 0.05 * self.ow + f * 0.1 * self.ow, by + (0.018 if i % 2 else -0.018) * self.ow
                        * (0.6 + 0.4 * math.sin(t * 14 + i))))
        pen.line(pts, self.th["accent2"] if self.theme_name != "paper" else self.acc, 1.7 * u, 0.9)

    def _r_complain(self, pen, t):
        u, h, G = self.u, self.h, self.G
        bw = 0.085 * self.ow
        bh = 0.042 * self.oh
        tx = self.xr + 0.08 * self.ow
        per = 0.5
        n = int(min(7, t / per))
        q = (t / per) % 1.0 if t / per < 7 else 1.0
        for i in range(n):
            off = (0.012 * self.ow) * (1 if i % 2 else -1)
            pen.rrect((tx - bw / 2 + off, G - (i + 1) * bh, tx + bw / 2 + off, G - i * bh), 3 * u, self.lc, 2.0 * u, 1.0,
                      fill=0)
        A = place(pose_stand(t), h, self.xr - 0.13 * self.ow, G, 1, self.picto)
        if n < 7:  # carry the next block up to the top
            off = (0.012 * self.ow) * (1 if n % 2 else -1)
            start = (A["hip"][0] + 0.25 * h, A["hip"][1] - 0.15 * h)
            end = (tx + off, G - (n + 0.5) * bh)
            e = ease_in_out(q)
            bxp = lerp(start[0], end[0], e)
            byp = lerp(start[1], end[1], e) - math.sin(math.pi * e) * 0.07 * self.oh
            pen.rrect((bxp - bw / 2, byp - bh / 2, bxp + bw / 2, byp + bh / 2), 3 * u, self.acc, 2.4 * u, 1.0, fill=0)
            if e < 0.5:
                arm_reach(A, "R", (bxp - bw / 2, byp))
                arm_reach(A, "L", (bxp - bw / 2 + 4 * u, byp + bh / 2))
            if q > 0.9:
                for j in range(4):
                    a = j * 1.6
                    self.sparkle(pen, end[0] + math.cos(a) * bw * 0.7, end[1] + math.sin(a) * bh, 7 * u, (1 - q) * 9)
        self.figR(pen, A)

    # ---------------------------------------------------------------- tv / laptop
    def _l_learn(self, pen, t):
        u, h, G = self.u, self.h, self.G
        f = 1
        hipx = self.xl - 0.13 * self.ow
        seat = G - seat_height(h, self.picto)
        # couch
        pen.rrect((hipx - 0.2 * h, seat - 0.02 * self.oh, hipx + 0.25 * h, seat + 0.02 * self.oh), 10 * u, self.lcol, 2.0 * u,
                  self.lk, fill=0)
        pen.rrect((hipx - 0.3 * h, seat - 0.3 * h, hipx - 0.17 * h, G), 10 * u, self.lcol, 2.0 * u, self.lk, fill=0)
        pose = pose_sit(t, 0.0)
        pose.update(lean=-0.25, head=0.15, shR=0.3, elR=1.4, shL=0.2, elL=0.4)
        A = seated(pose, h, hipx, seat, G, f, self.picto)
        self.figL(pen, A)
        self.tv(pen, self.xl + 0.14 * self.ow, G, 0.18 * self.ow, t, self.lk, self.lcol)
        # snack bag
        hx, hy = A["handR"]
        pen.rrect((hx - 10 * u, hy - 30 * u, hx + 10 * u, hy + 2 * u), 3 * u, self.lcol, 1.5 * u, 0.8, fill=0)

    def _r_learn(self, pen, t):
        u, h, G = self.u, self.h, self.G
        f = -1
        hipx = self.xr + 0.15 * self.ow
        seat = G - seat_height(h, self.picto)
        self.chair(pen, hipx, seat, G, f, h)
        top = G - 0.5 * h
        self.desk(pen, self.xr - 0.16 * self.ow, self.xr + 0.07 * self.ow, top, G)
        pose = pose_sit(t, 0.22)
        pose.update(head=0.2)
        A = seated(pose, h, hipx, seat, G, f, self.picto)
        lx = self.xr - 0.06 * self.ow
        self.laptop(pen, lx, top, 0.3 * h, t, 1.0, facing=1)
        typing = 0.01 * h * math.sin(t * 22)
        arm_reach(A, "R", (lx + 0.14 * h, top - 0.03 * h + typing))
        arm_reach(A, "L", (lx + 0.2 * h, top - 0.03 * h - typing))
        self.figR(pen, A)
        hx, hy = A["head"]
        pulse = 0.5 + 0.5 * math.sin(t * 4)
        self.bulb(pen, hx - 0.01 * self.ow, hy - 0.3 * h, 0.075 * h, 1.0, 0.6 + 0.4 * pulse, t)
        for i in range(3):
            age = (t * 0.8 + i / 3) % 1.0
            self.sparkle(pen, hx + (i - 1) * 0.05 * self.ow, hy - 0.42 * h - age * 0.06 * self.oh, 7 * u,
                         math.sin(math.pi * age))


# ======================================================================== 14. stickman through life
LIFE_AGES = ("0", "10", "20", "40", "70")


class LALife(LAStory):
    """The figure through life: baby -> kid with backpack -> student with books -> worker at a desk ->
    old man with a cane, along a timeline with age labels; the camera pans from stage to stage.
    params.stage (0-4) focuses one stage."""
    focus = (0.5, 0.52)
    label_at = (0.5, 0.31)

    def setup(self):
        st = self.p.get("stage")
        try:
            st = int(float(st)) if st is not None and not isinstance(st, bool) else None
        except (TypeError, ValueError):
            st = None
        self.stage = None if st is None else int(clamp(st, 0, 4))
        self.G = self.Y(0.6)
        self.S = 0.56 * self.ow
        self.STEP = 1.15

    def cam(self, t):
        if self.stage is not None:
            return self.S * (self.stage - 0.7 * (1 - ease_out(t / 1.1))) + 0.02 * self.S * math.sin(t * 1.3)
        n = math.floor(t / self.STEP)
        q = t / self.STEP - n
        return self.S * ((n % 4) + ease_in_out(clamp(q / 0.5)))

    def _crawl(self, x, G, h, ph, f=1):
        s = math.sin(ph)
        hip = (x, G - 0.27 * h + 0.01 * h * math.sin(2 * ph))
        sh = (x + f * 0.36 * h, G - 0.3 * h + 0.01 * h * math.sin(2 * ph + 1))
        A = {"hip": hip, "neck": sh, "head": (sh[0] + f * 0.13 * h, sh[1] - 0.1 * h), "h": h, "facing": f,
             "front": False}
        for side, sg in (("L", 1), ("R", -1)):
            st = sg * s * 0.06 * h
            A["sh" + side] = sh
            A["hand" + side] = (sh[0] + f * (0.03 * h + st), G - max(0.0, sg * math.cos(ph)) * 0.04 * h)
            A["el" + side] = ((sh[0] + A["hand" + side][0]) / 2 - f * 0.03 * h, (sh[1] + A["hand" + side][1]) / 2)
            A["hp" + side] = hip
            A["knee" + side] = (hip[0] + f * (0.05 * h - st), G - 0.03 * h - max(0.0, -sg * math.cos(ph)) * 0.03 * h)
            A["foot" + side] = (hip[0] - f * (0.2 * h + st * 0.5), G - 0.06 * h)
            A["toe" + side] = (A["foot" + side][0] - f * 0.04 * h, A["foot" + side][1] - 0.02 * h)
        return A

    def draw(self, pen, t):
        u, G, S = self.u, self.G, self.S
        cam = self.cam(t)
        cx = self.X(0.5)

        def sx(i):
            return cx + i * S - cam
        lc, acc, dim = self.lc, self.acc, self.th["dim"]
        # timeline: dim full line + accent progress up to the camera centre
        x0 = max(self.X(-0.05), sx(-0.3))
        x1 = min(self.X(1.05), sx(4.5))
        pen.seg((x0, G), (x1, G), dim, 1.8 * u, 1.0)
        xp = min(cx, x1)
        if xp > x0:
            pen.seg((x0, G), (xp, G), acc, 3.0 * u, 1.0)
        if sx(4.5) < self.X(1.05):
            self.arrow(pen, (sx(4.5) - 10 * u, G), (sx(4.5) + 10 * u, G), dim, 1.8 * u, 1.0, 16 * u)
        for i in range(5):
            x = sx(i)
            if x < self.X(-0.3) or x > self.X(1.3):
                continue
            d = abs(x - cx) / S
            foc = 1 - ease(clamp(d / 0.6))
            if self.stage is not None:
                foc = max(foc * 0.6, 1.0 if i == self.stage else 0.0) if d < 0.4 else foc * 0.6
            k = 0.45 + 0.55 * foc
            pen.dot((x, G), (6 + 4 * foc) * u, acc if foc > 0.5 else dim, 1.0)
            if foc > 0.5:
                pen.soft(x, G - 0.07 * self.oh, 0.16 * self.ow, self.th["soft"], 0.3 * foc)
            pop = 1.0 + 0.25 * math.sin(math.pi * clamp((0.35 - d) / 0.35)) * foc
            pen.text((x, G + 0.055 * self.oh), LIFE_AGES[i], 78 * u * pop, acc if foc > 0.5 else dim,
                     0.55 + 0.45 * foc, "mm", "Display")
            pen.text((x, G + 0.093 * self.oh), "YEARS" if i else "BORN", 20 * u, self.th["text"], 0.35 + 0.4 * foc,
                     "mm", "Bold", 0.3)
            getattr(self, "_stage%d" % i)(pen, t, x, G, k)

    def _stage0(self, pen, t, x, G, k):
        h = 0.13 * self.oh
        A = self._crawl(x - 0.08 * h, G, h, TAU * t / 0.9)
        self.figure(pen, A, k, head_scale=1.9)
        # rattle
        hx, hy = A["handL"]
        pen.circle((x + 0.35 * h, G - 0.05 * h), 0.06 * h, self.acc, 1.6 * self.u, k)

    def _stage1(self, pen, t, x, G, k):
        h = 0.155 * self.oh
        ph = TAU * t / 0.7
        hop = abs(math.sin(ph)) * 0.04 * h
        A = place(pose_walk(ph), h, x, G - hop, 1, self.picto)
        neck, hip = A["neck"], A["hip"]
        bx, by = lerp(neck[0], hip[0], 0.35) - 0.1 * h, lerp(neck[1], hip[1], 0.35)
        pen.rrect((bx - 0.07 * h, by - 0.11 * h, bx + 0.05 * h, by + 0.11 * h), 5 * self.u, self.acc, 2.0 * self.u, k,
                  fill=0)
        pen.seg((bx - 0.05 * h, by + 0.02 * h), (bx + 0.03 * h, by + 0.02 * h), self.acc, 1.4 * self.u, k)
        self.figure(pen, A, k, head_scale=1.35)

    def _stage2(self, pen, t, x, G, k):
        h = 0.185 * self.oh
        ph = TAU * t / 0.85
        A = place(pose_walk(ph), h, x, G, 1, self.picto)
        bx, by = A["neck"][0] + 0.13 * h, A["neck"][1] + 0.17 * h
        arm_reach(A, "R", (bx - 0.04 * h, by + 0.03 * h))
        arm_reach(A, "L", (bx + 0.05 * h, by + 0.04 * h))
        self.figure(pen, A, k)
        for j, (w, c) in enumerate(((0.2, self.lc), (0.17, self.acc), (0.22, self.lc))):
            self.book_closed(pen, bx + (j - 1) * 0.01 * h, by - j * 0.045 * h, w * h, 0.045 * h, k, c)

    def _stage3(self, pen, t, x, G, k):
        h = 0.2 * self.oh
        u = self.u
        hipx = x - 0.12 * h
        seat = G - seat_height(h, self.picto)
        self.chair(pen, hipx, seat, G, 1, h, k)
        top = G - 0.5 * h
        self.desk(pen, x + 0.08 * h, x + 0.62 * h, top, G, k)
        # monitor + coffee
        mx = x + 0.42 * h
        pen.rrect((mx - 0.04 * h, top - 0.3 * h, mx + 0.04 * h, top - 0.05 * h), 3 * u, self.lc, 2.0 * u, k, fill=0)
        pen.seg((mx, top - 0.05 * h), (mx, top), self.lc, 2.0 * u, k)
        pen.soft(mx - 0.06 * h, top - 0.18 * h, 0.12 * h, self.th["soft"], 0.35 * k)
        pen.rrect((x + 0.53 * h, top - 0.07 * h, x + 0.59 * h, top), 2 * u, self.acc, 1.6 * u, k)
        for j in range(2):
            age = (t * 0.8 + j * 0.5) % 1.0
            pen.seg((x + 0.55 * h + j * 0.02 * h, top - 0.09 * h - age * 0.08 * h),
                    (x + 0.56 * h + j * 0.02 * h, top - 0.12 * h - age * 0.08 * h), self.th["dim"], 1.4 * u,
                    k * math.sin(math.pi * age))
        pose = pose_sit(t, 0.18)
        pose.update(head=0.12)
        A = seated(pose, h, hipx, seat, G, 1, self.picto)
        ty = 0.008 * h * math.sin(t * 20)
        arm_reach(A, "R", (x + 0.24 * h, top - 0.02 * h + ty))
        arm_reach(A, "L", (x + 0.3 * h, top - 0.02 * h - ty))
        pen.seg((x + 0.18 * h, top - 0.01 * h), (x + 0.34 * h, top - 0.01 * h), self.lc, 2.6 * u, k)
        self.figure(pen, A, k)

    def _stage4(self, pen, t, x, G, k):
        h = 0.175 * self.oh
        ph = TAU * t / 1.4
        pose = mix(pose_walk(ph), pose_hunched(t), 0.55)
        pose.update(lean=0.32, head=0.35)
        A = place(pose, h, x, G, 1, self.picto)
        tip = (A["hip"][0] + 0.32 * h + 0.02 * h * math.sin(ph), G)
        top = (tip[0] - 0.02 * h, G - 0.47 * h)
        arm_reach(A, "R", top)
        self.figure(pen, A, k)
        pen.line([tip, top, (top[0] - 0.06 * h, top[1] - 0.03 * h)], self.th["accent"], 2.4 * self.u, k)


# ======================================================================== 15. kinetic typography
def _back_out(x, s=1.9):
    x = clamp(x) - 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2


class LATypo(LAStory):
    """Kinetic typography for punch lines: params.text (1-3 words) slams in letter by letter (outline wipe,
    then fill), the last word in the accent colour with an underline swipe, sparks and a tiny figure."""
    focus = (0.5, 0.47)
    label_at = None

    def setup(self):
        p = self.p
        raw = p.get("text") if isinstance(p.get("text"), str) and p.get("text").strip() else p.get("label")
        words = [clean_word(w, 12) for w in str(raw if isinstance(raw, (str, int, float)) and not isinstance(raw, bool)
                                                 else "").split()]
        words = [w for w in words if w] or ["NO", "EXCUSES"]
        words = words[:6]
        if len(words) > 3:  # pack into three balanced lines
            lines, per = [], math.ceil(len(words) / 3)
            for i in range(0, len(words), per):
                lines.append(" ".join(words[i:i + per]))
        else:
            lines = words
        self.lines = lines
        self.show_fig = parse_bool(p.get("figure"), True)
        u = self.u
        ow = self.ow
        sizes = []
        for ln in lines:
            f = font(100, "Display")
            w100 = f.getlength(ln)
            sizes.append(min(0.24 * ow, 0.74 * ow * 100 / max(1.0, w100)))
        lead = 0.14
        tot = sum(s * (0.74 + lead) for s in sizes) - sizes[-1] * lead
        lim = 0.38 * self.oh
        if tot > lim:
            sizes = [s * lim / tot for s in sizes]
            tot = lim
        self.sizes = sizes
        y = self.Y(0.47) - tot / 2
        self.letters = []
        self.line_geo = []
        idx = 0
        rng = self.rng
        for li, (ln, px) in enumerate(zip(lines, sizes)):
            cap = 0.74 * px
            base = y + cap
            f = font(px * SS, "Display")
            width = f.getlength(ln) / SS
            x = self.X(0.5) - width / 2
            self.line_geo.append((self.X(0.5) - width / 2, self.X(0.5) + width / 2, base, cap, px))
            for ch in ln:
                adv = f.getlength(ch) / SS
                if ch != " ":
                    sw = max(2, int(0.035 * px * SS))
                    x0b, y0b, x1b, y1b = f.getbbox(ch, anchor="ls", stroke_width=sw)
                    wi, hi = int(x1b - x0b) + 4, int(y1b - y0b) + 4
                    fill = Image.new("L", (wi, hi), 0)
                    outl = Image.new("L", (wi, hi), 0)
                    ImageDraw.Draw(fill).text((-x0b + 2, -y0b + 2), ch, font=f, fill=255, anchor="ls")
                    ImageDraw.Draw(outl).text((-x0b + 2, -y0b + 2), ch, font=f, fill=255, anchor="ls", stroke_width=sw,
                                              stroke_fill=255)
                    ImageDraw.Draw(outl).text((-x0b + 2, -y0b + 2), ch, font=f, fill=0, anchor="ls")
                    start = 0.08 + idx * 0.05 + li * 0.1
                    sparks = [(rng.uniform(0, TAU), rng.uniform(0.5, 1.0)) for _ in range(5)]
                    # sprite top-left in layer coords (x, baseline) + bbox offset
                    self.letters.append(dict(fill=fill, outl=outl, x=x + (x0b - 2) / SS, y=base + (y0b - 2) / SS,
                                             cx=x + adv / 2, cy=base - cap / 2, line=li, start=start, sparks=sparks,
                                             w=wi / SS, h=hi / SS))
                    idx += 1
                x += adv
            y = base + lead * px
        self.t_end = max(L["start"] for L in self.letters) + 0.3
        self.amb = [(rng.uniform(0.05, 0.95), rng.uniform(0, 1), rng.uniform(0.6, 1.4), rng.uniform(0, TAU))
                    for _ in range(26)]

    def _paste(self, pen, mask, x, y, col, k):
        if k <= 0.01:
            return
        if pen.wob:
            jx, jy = pen.J(x, y)
            x, y = jx, jy
        m = mask if k >= 0.999 else mask.point([int(v * k) for v in range(256)])
        pen.img.paste(pen.c(col), (int(round(x * SS)), int(round(y * SS))), m)

    def draw(self, pen, t):
        u = self.u
        nl = len(self.lines)
        acc, tc = self.acc, self.th["text"]
        # ambient rising particles
        for fx, fy, sp, ph in self.amb:
            yy = self.Y(0.72) - ((fy + t * 0.08 * sp) % 1.0) * 0.5 * self.oh
            xx = self.X(fx) + 12 * u * math.sin(t * sp + ph)
            kk = math.sin(math.pi * ((fy + t * 0.08 * sp) % 1.0)) * 0.6
            pen.dot((xx, yy), 2.2 * u * sp, acc if ph > 3 else self.th["dim"], kk)
        last = nl - 1
        lx0, lx1, lbase, lcap, lpx = self.line_geo[last]
        pulse = 0.5 + 0.5 * math.sin(t * 3.2)
        if t > self.t_end - 0.2:
            pen.soft((lx0 + lx1) / 2, lbase - lcap / 2, 0.3 * self.ow, self.th["soft"], (0.18 + 0.1 * pulse))
        for L in self.letters:
            a = (t - L["start"]) / 0.32
            if a <= 0:
                continue
            col = acc if (L["line"] == last or nl == 1) else tc
            e = _back_out(a)
            sc = 1.0 + 0.55 * (1 - ease_out(a))
            dy = -(1 - e) * 0.05 * self.oh
            reveal = ease_out(clamp(a / 0.6))
            fk = ease(clamp((a - 0.4) / 0.5))
            fill, outl = L["fill"], L["outl"]
            x, y = L["x"], L["y"] + dy
            if sc != 1.0:
                sz = (max(2, int(outl.width * sc)), max(2, int(outl.height * sc)))
                fill, outl = fill.resize(sz, Image.BILINEAR), outl.resize(sz, Image.BILINEAR)
                x = L["cx"] - (L["cx"] - x) * sc
                y = (L["cy"] + dy) - (L["cy"] - L["y"]) * sc
            if reveal < 1:
                wcut = max(1, int(outl.width * reveal))
                outl = outl.crop((0, 0, wcut, outl.height))
                fill = fill.crop((0, 0, wcut, fill.height))
                tipx = x + wcut / SS
                pen.dot((tipx, y + outl.height / SS * 0.5), 4 * u, acc, 1.0)
            self._paste(pen, outl, x, y, col, min(1.0, a * 2.5))
            self._paste(pen, fill, x, y, col, fk * {"neon": 0.14, "gold": 0.45}.get(self.theme_name, 1.0))  # neon = tubes
            # landing sparks
            sa = (t - L["start"] - 0.15) / 0.4
            if 0 < sa < 1:
                for ang, sp in L["sparks"]:
                    r0 = (0.3 + 0.7 * ease_out(sa)) * 0.09 * self.oh * sp
                    ln = 14 * u * (1 - sa)
                    cx, cy = L["cx"], L["cy"]
                    pen.seg((cx + math.cos(ang) * r0, cy + math.sin(ang) * r0),
                            (cx + math.cos(ang) * (r0 + ln), cy + math.sin(ang) * (r0 + ln)), acc, 2.0 * u, 1 - sa)
        # underline swipe under the last line
        ua = (t - self.t_end - 0.05) / 0.32
        if ua > 0:
            e = ease_out(ua)
            uy = lbase + 0.16 * lpx
            pts = []
            for i in range(25):
                f = i / 24 * e
                pts.append((lerp(lx0 - 0.02 * self.ow, lx1 + 0.02 * self.ow, f), uy + math.sin(f * math.pi) * 0.04 * lpx
                            - f * 0.05 * lpx))
            pen.line(pts, acc, max(5.0, 0.07 * lpx) * u / max(0.2, u) * 0.9, 1.0)
            if e >= 0.999:  # glint running along the underline
                g = (t * 0.7) % 1.6
                if g < 1:
                    gx, gy = pts[int(g * 23)]
                    pen.soft(gx, gy, 0.04 * self.ow, self.th["soft"], 0.6 * math.sin(math.pi * g))
                    self.sparkle(pen, gx, gy - 0.1 * lpx, 9 * u, math.sin(math.pi * g), tc)
        # tiny figure standing on the first line
        if self.show_fig:
            fa = (t - self.t_end - 0.2) / 0.4
            if fa > 0:
                x0, x1, base, cap, px = self.line_geo[0]
                fx = x1 - 0.12 * (x1 - x0)
                h = 0.075 * self.oh * (0.4 + 0.6 * _back_out(fa))
                pose = mix(pose_stand_front(t), pose_cheer(t), clamp((fa - 1.0) / 0.5))
                A = place(pose, h, fx, base - cap - 3 * u, 1, self.picto)
                self.figure(pen, A, min(1.0, fa * 2))


# ======================================================================== 16. fork in the road
def _bezier(p0, p1, p2, n=24):
    out = []
    for i in range(n + 1):
        f = i / n
        out.append(((1 - f) ** 2 * p0[0] + 2 * (1 - f) * f * p1[0] + f * f * p2[0],
                    (1 - f) ** 2 * p0[1] + 2 * (1 - f) * f * p1[1] + f * f * p2[1]))
    return out


def _ribbon(center, w0, w1):
    """Left/right edges of a ribbon along a centreline, width lerped w0 -> w1."""
    C = np.asarray(center, np.float64)
    d = np.gradient(C, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    w = np.linspace(w0, w1, len(C))[:, None] / 2
    return [tuple(p) for p in C + nrm * w], [tuple(p) for p in C - nrm * w]


class LAFork(LAStory):
    """A road splits: left an easy wide road sloping down (others drift along it), right a steep narrow
    path up to a glowing light. The figure chooses the hard path. params.left / params.right = sign words."""
    focus = (0.6, 0.4)
    label_at = None

    def setup(self):
        X, Y = self.X, self.Y
        p = self.p
        self.left = clean_word(p.get("left")) or "EASY"
        self.right = clean_word(p.get("right")) or "HARD"
        self.J = (X(0.5), Y(0.53))
        self.light = (X(0.8), Y(0.235))
        self.easy_top = _bezier((X(0.44), Y(0.53)), (X(0.25), Y(0.535)), (X(-0.08), Y(0.6)), 30)
        self.branch = (X(0.17) + 0.64 * (X(0.44) - X(0.17)), Y(0.62))
        self.easy_bot = _bezier(self.branch, (X(0.2), Y(0.665)), (X(-0.08), Y(0.745)), 30)
        self.easy = [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b in zip(self.easy_top, self.easy_bot)]
        zz = [self.J, (X(0.63), Y(0.47)), (X(0.55), Y(0.415)), (X(0.71), Y(0.36)), (X(0.64), Y(0.315)),
              (X(0.77), Y(0.265))]
        path = [zz[0]]
        for a, b in zip(zz[:-1], zz[1:]):
            for i in range(1, 7):
                f = i / 6
                path.append((lerp(a[0], b[0], f), lerp(a[1], b[1], f)))
        self.hard = path
        self.main = [(X(0.5), Y(0.8)), self.J]
        # figure route: up the main road to the junction, then the hard path
        route = [(X(0.5), Y(0.64))] + path
        self.route = np.array(route)
        self.rcum = np.concatenate([[0], np.cumsum(np.hypot(*(self.route[1:] - self.route[:-1]).T))])
        self.walkers = [i / 3 for i in range(3)]

    def h_at(self, y):
        return self.oh * (0.045 + 0.085 * clamp((y - self.Y(0.24)) / (self.Y(0.64) - self.Y(0.24))))

    def draw(self, pen, t):
        u, X, Y = self.u, self.X, self.Y
        lc, acc, dim = self.lc, self.acc, self.th["dim"]
        intro = ease_out(t / 0.5)
        # main road (perspective trapezoid)
        J = self.J
        pen.seg((X(0.83), Y(0.78)), (J[0] + 0.06 * self.ow, J[1]), lc, 2.2 * u, intro)
        pen.seg((X(0.17), Y(0.78)), self.branch, lc, 2.2 * u, intro)
        for i in range(5):  # centre dashes flowing towards the viewer
            f = (i / 5 + t * 0.6) % 1.0
            y0, y1 = lerp(J[1], Y(0.78), f), lerp(J[1], Y(0.78), min(1, f + 0.07))
            pen.seg((X(0.5), y0), (X(0.5), y1), dim, (1.5 + 3 * f) * u, intro * math.sin(math.pi * f))
        # easy road: wide, smooth, sloping down to the left
        pen.line(self.easy_top, dim, 2.0 * u, intro)
        pen.line(self.easy_bot, dim, 2.0 * u, intro)
        for i in range(4):  # chevrons sliding downhill
            f = (i / 4 + t * 0.35) % 1.0
            j = int(f * (len(self.easy) - 2))
            (px, py), (qx, qy) = self.easy[j], self.easy[j + 1]
            ang = math.atan2(qy - py, qx - px)
            s = (10 + 22 * f) * u
            for sg in (-1, 1):
                aa = ang + math.pi + sg * 0.6
                pen.seg((px, py), (px + math.cos(aa) * s, py + math.sin(aa) * s), self.th["faint"], 1.8 * u,
                        math.sin(math.pi * f) * intro)
        # other people drifting down the easy road
        for ph0 in self.walkers:
            f = (ph0 + t * 0.16) % 1.0
            j = f * (len(self.easy) - 1)
            i0 = int(j)
            p0, p1 = self.easy[i0], self.easy[min(i0 + 1, len(self.easy) - 1)]
            x, y = lerp(p0[0], p1[0], j - i0), lerp(p0[1], p1[1], j - i0)
            hh = self.oh * (0.05 + 0.05 * f)
            A = place(pose_walk(TAU * t / 1.0 + ph0 * 7), hh, x, y, -1, self.picto)
            self.figure(pen, A, 0.55 * math.sin(math.pi * min(1, f * 1.15)) * intro, col=dim)
        # hard path: narrow, steep switchbacks with steps
        a, b = _ribbon(self.hard, 0.05 * self.ow, 0.012 * self.ow)
        pen.line(a, lc, 1.9 * u, intro)
        pen.line(b, lc, 1.9 * u, intro)
        for i in range(2, len(a) - 1, 2):
            pen.seg(a[i], b[i], dim, 1.3 * u, 0.8 * intro)
        # the light at the top
        lx, ly = self.light
        route_s = self.route_pos(t)[2]
        near = clamp(route_s / self.rcum[-1])
        glow = 0.75 + 0.25 * math.sin(t * 3.0) + 0.4 * near
        pen.circle((lx, ly), 0.032 * self.ow, acc, 2.6 * u, 1.0, fill=acc, kf=0.35)
        for i in range(14):
            ang = TAU * i / 14 + t * 0.4
            r0 = 0.05 * self.ow
            r1 = r0 + (0.03 + 0.012 * math.sin(t * 6 + i * 1.3)) * self.ow * glow
            pen.seg((lx + math.cos(ang) * r0, ly + math.sin(ang) * r0), (lx + math.cos(ang) * r1, ly + math.sin(ang) * r1),
                    acc, 1.8 * u, 0.9)
        pen.soft(lx, ly, 0.13 * self.ow, self.th["soft"], 0.7 * glow)
        # signpost at the junction
        px, py = X(0.41), J[1] - 0.006 * self.oh
        top = Y(0.395)
        pen.seg((px, py), (px, top), lc, 3.0 * u, intro)
        self._board(pen, px, top + 0.018 * self.oh, -1, self.left, dim, intro, -0.08)
        self._board(pen, px, top + 0.062 * self.oh, 1, self.right, acc, intro, -0.18)
        # the figure
        x, y, s, d = self.route_pos(t)
        hh = self.h_at(y)
        if s <= 1e-6 or d[1] < -0.9 and abs(d[0]) < 0.3 and s < self.rcum[1] + 1:
            # front (from behind) on the main road
            moving = s > 1e-6
            ph = TAU * t / 0.9
            sw = math.sin(ph) if moving else 0.0
            pose = P(front=True, thL=0.1, thR=0.1, shL=0.3 + 0.05 * sw, shR=0.3 - 0.05 * sw, elL=-0.14, elR=-0.14,
                     liftL=max(0.0, sw) * 0.22, liftR=max(0.0, -sw) * 0.22, sway=0.012 * sw, dy=-0.006 * abs(sw))
            if not moving:
                pose = pose_stand_front(t)
                pose["sway"] = 0.02 * math.sin(t * 3)
            A = place(pose, hh, x, y, 1, self.picto)
        else:
            facing = 1 if d[0] >= 0 else -1
            pose = pose_walk(TAU * t / 0.8)
            pose["lean"] += 0.18
            A = place(pose, hh, x, y, facing, self.picto)
        self.figure(pen, A)

    def route_pos(self, t):
        v = 0.21 * self.oh  # px / s
        s = clamp((t - 0.45) * v, 0, self.rcum[-1] - 1)
        if s >= self.rcum[1]:  # slower on the steep part
            s = self.rcum[1] + (s - self.rcum[1]) * 0.55
        here, d = polyline_at(self.route, self.rcum, s)
        return float(here[0]), float(here[1]), s, d

    def _board(self, pen, x, y, direction, word, col, k, tilt):
        u = self.u
        px = 27 * u
        tw = pen.text_width(word, px, "Display", 0.1)
        w, hh = tw + 46 * u, 46 * u
        x0 = x if direction > 0 else x - w
        x1 = x0 + w
        tip = 20 * u
        if direction > 0:
            pts = [(x0, y - hh / 2), (x1, y - hh / 2), (x1 + tip, y), (x1, y + hh / 2), (x0, y + hh / 2)]
        else:
            pts = [(x1, y - hh / 2), (x0, y - hh / 2), (x0 - tip, y), (x0, y + hh / 2), (x1, y + hh / 2)]
        ca, sa = math.cos(tilt * direction), math.sin(tilt * direction)
        R = [(x + (a - x) * ca - (b - y) * sa, y + (a - x) * sa + (b - y) * ca) for a, b in pts]
        pen.poly(R, col, 2.2 * u, k, fill=0)
        cx = (x0 + x1) / 2 + direction * 4 * u
        cxr = x + (cx - x) * ca
        cyr = y + (cx - x) * sa
        pen.text((cxr, cyr + 1 * u), word, px, col, k, "mm", "Display", 0.1)


# ======================================================================== 17. hourglass
class LAHourglass(LAStory):
    """A big hourglass: sand pours from the top bulb into a growing pile while a figure watches; when it
    runs out the glass flips and it starts again (time running out)."""
    focus = (0.58, 0.47)
    label_at = (0.58, 0.735)
    RUN, FLIP = 6.2, 0.6

    def setup(self):
        self.c = (self.X(0.6), self.Y(0.465))
        self.Hh = 0.19 * self.oh
        self.Wd = 0.2 * self.ow
        self.neck = 0.012 * self.ow
        self.G = self.Y(0.465) + self.Hh + 0.03 * self.oh
        rng = self.rng
        self.grains = [(rng.uniform(0, 1), rng.uniform(-1, 1)) for _ in range(26)]
        self.sand_k = 0.16 if self.theme_name == "neon" else 0.38
        self.prof = [self._half(f) for f in np.linspace(0, 1, 26)]

    def _half(self, f):
        """Half-width of the glass at fraction f in [0 = neck, 1 = cap]."""
        return self.neck + (self.Wd - self.neck) * math.sin(math.pi / 2 * min(1.0, f * 1.05)) ** 0.9

    def state(self, t):
        per = self.RUN + self.FLIP
        q = (t + 1.2) % per
        if q < self.RUN:
            return clamp(1 - q / self.RUN), 0.0
        return 0.0, ease_in_out((q - self.RUN) / self.FLIP)

    def draw(self, pen, t):
        u = self.u
        cx, cy = self.c
        Hh, Wd = self.Hh, self.Wd
        lc, acc, dim = self.lc, self.acc, self.th["dim"]
        frac, flip = self.state(t)
        ang = math.pi * flip
        ca, sa = math.cos(ang), math.sin(ang)

        def R(x, y):  # local (centre-relative) -> screen with flip rotation
            return (cx + x * ca - y * sa, cy + x * sa + y * ca)
        self.ground(pen, self.G, self.X(0.08), self.X(0.95), 1.0, 0.25)
        # glass outline
        for sgn in (-1, 1):
            pts = [R(sgn * self._half(abs(f)), -Hh * f) for f in np.linspace(1, -1, 51)]
            pen.line(pts, lc, 2.4 * u, 1.0)
        # caps + posts
        for sy in (-1, 1):
            y = sy * (Hh + 0.012 * self.oh)
            pen.poly([R(-Wd - 0.03 * self.ow, y - 0.012 * self.oh), R(Wd + 0.03 * self.ow, y - 0.012 * self.oh),
                      R(Wd + 0.03 * self.ow, y + 0.012 * self.oh), R(-Wd - 0.03 * self.ow, y + 0.012 * self.oh)],
                     lc, 2.2 * u, 1.0, fill=0)
        for sx in (-1, 1):
            pen.seg(R(sx * (Wd + 0.012 * self.ow), -Hh), R(sx * (Wd + 0.012 * self.ow), Hh), dim, 2.0 * u, 1.0)
        # sand (volume ~ area): top fill level and bottom pile height
        top_level = 0.9 * frac ** 0.65  # fraction of the bulb height still filled (top)
        pile = 0.85 * (1 - frac) ** 0.7
        if flip > 0:
            top_level, pile = 0.0, 0.85
        sand = acc
        if top_level > 0.01:
            lvl = top_level
            pts = []
            for f in np.linspace(0.0, lvl, 16):
                pts.append((self._half(f), -Hh * f))
            dip = 0.05 * Hh * min(1.0, lvl * 3)
            poly = [R(x, y) for x, y in pts] + [R(0, -Hh * lvl + dip)] + [R(-x, y) for x, y in pts[::-1]]
            pen.poly(poly, None, fill=sand, kf=self.sand_k)
            pen.line([R(-self._half(lvl), -Hh * lvl), R(0, -Hh * lvl + dip), R(self._half(lvl), -Hh * lvl)], sand,
                     2.0 * u, 1.0)
        if pile > 0.01:
            ph = Hh * pile
            pts = []
            for f in np.linspace(1.0, 1.0 - pile * 0.75, 14):
                pts.append((self._half(f), Hh * f))
            peak = Hh - ph
            poly = [R(-x, y) for x, y in pts] + [R(0, peak)] + [R(x, y) for x, y in pts[::-1]]
            pen.poly(poly, None, fill=sand, kf=self.sand_k)
            pen.line([R(-x, y) for x, y in pts[-1:]] + [R(0, peak)] + [R(x, y) for x, y in pts[-1:]], sand, 2.0 * u, 1.0)
        # stream + falling grains
        if frac > 0.003 and flip == 0:
            bottom = Hh - Hh * pile
            pen.seg(R(0, 0), R(0, bottom), sand, 1.6 * u, 0.8)
            for f0, dx in self.grains:
                f = (f0 + t * 2.2) % 1.0
                y = f * bottom
                pen.dot(R(dx * 2.5 * u * (0.3 + f), y), 2.2 * u, sand, 0.9)
            # splash at the pile top
            for j in range(3):
                a = (t * 3 + j / 3) % 1.0
                pen.dot(R((j - 1) * 12 * u * a, bottom - 10 * u * math.sin(math.pi * a)), 1.8 * u, sand, 1 - a)
        pen.soft(cx, cy + Hh * 0.6, 0.12 * self.ow, self.th["soft"], 0.3)
        # figure watching, anxious when it runs low
        hh = 0.15 * self.oh
        worry = 1 - frac if flip == 0 else 0.0
        pose = mix(pose_lookup(t), pose_think(t), ease(worry * 1.5))
        A = place(pose, hh, self.X(0.2), self.G, 1, self.picto)
        self.figure(pen, A)
        if worry > 0.6:  # sweat drop
            hx, hy = A["head"]
            a = (t * 1.5) % 1.0
            pen.dot((hx - 0.05 * hh, hy - 0.02 * hh + a * 0.1 * hh), 3.5 * u, self.th["text"], 1 - a)


# ======================================================================== 18. seed -> tree
class LASeed(LAStory):
    """Time-lapse growth: a seed in the soil sprouts, becomes a sapling, then a big tree carrying coins
    (params.fruit='fruit' for apples) while the figure waters it (patience, compounding)."""
    focus = (0.58, 0.5)
    label_at = (0.5, 0.715)
    GROW = 3.2

    def setup(self):
        rng = self.rng
        self.G = self.Y(0.64)
        self.root = (self.X(0.62), self.G)
        fr = self.p.get("fruit")
        self.fruit = "fruit" if isinstance(fr, str) and fr.strip().lower() in ("fruit", "apple", "apples", "fruits") \
            else "coin"
        # branches: (parent index, angle, length, depth, birth in [0,1])
        L0 = 0.075 * self.oh
        br = [(-1, -math.pi / 2, L0, 0, 0.12)]
        tips = []
        frontier = [0]
        while frontier:
            nxt = []
            for i in frontier:
                _, ang, ln, dep, b0 = br[i]
                if dep >= 5:
                    tips.append(i)
                    continue
                for s in (-1, 1):
                    a = ang + s * rng.uniform(0.38, 0.72) + rng.uniform(-0.08, 0.08)
                    a = clamp(a, -math.pi + 0.35, -0.35)
                    br.append((i, a, ln * rng.uniform(0.7, 0.82), dep + 1, b0 + 0.12 + rng.uniform(0, 0.05)))
                    nxt.append(len(br) - 1)
                if dep >= 2 and rng.random() < 0.5:
                    a = ang + rng.uniform(-0.15, 0.15)
                    br.append((i, a, ln * 0.6, dep + 1, b0 + 0.2))
                    nxt.append(len(br) - 1)
            frontier = nxt
        self.br = br
        self.tips = tips
        self.sway_ph = [rng.uniform(0, TAU) for _ in br]
        self.drops = [rng.uniform(0, 1) for _ in range(9)]
        self.leaf_off = [(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in tips]

    def grow(self, t):
        return clamp((t - 0.25) / self.GROW) ** 0.85

    def draw(self, pen, t):
        u = self.u
        lc, acc, dim = self.lc, self.acc, self.th["dim"]
        G = self.G
        rx, ry = self.root
        self.ground(pen, G, self.X(0.05), self.X(0.97), 1.0, 0.2)
        # soil mound + seed
        pen.arc((rx, G + 0.03 * self.oh), 0.09 * self.ow, 200, 340, dim, 1.8 * u, 1.0)
        g = self.grow(t)
        sk = 1 - ease(clamp(g / 0.08))
        pen.ellipse((rx - 9 * u, G + 12 * u, rx + 9 * u, G + 24 * u), acc, 2.0 * u, 0.4 + 0.6 * sk, fill=acc, kf=0.5)
        for j in range(3):  # roots
            ang = math.pi / 2 + (j - 1) * 0.7
            ln = 0.04 * self.oh * ease(clamp(g / 0.2))
            pen.seg((rx, G + 18 * u), (rx + math.cos(ang) * ln, G + 18 * u + math.sin(ang) * ln * 0.7), dim, 1.4 * u, 0.8)
        # branches
        pos = {}
        wtot = 0.0105 * self.ow
        for i, (par, ang, ln, dep, b0) in enumerate(self.br):
            gk = clamp((g - b0 + 0.12) / 0.3) if i else clamp(g / 0.4)
            if gk <= 0:
                continue
            sway = 0.025 * (dep + 1) * math.sin(t * 1.7 + self.sway_ph[i]) * g
            a = ang + sway
            start = pos[par][0] if par >= 0 else (rx, ry)
            L = ln * ease_out(gk) * (0.35 + 0.65 * g)
            end = (start[0] + math.cos(a) * L, start[1] + math.sin(a) * L)
            pos[i] = (end, a)
            w = max(1.4, wtot * (0.62 ** dep) * (0.3 + 0.7 * g))
            pen.seg(start, end, lc, w / max(0.3, u) * u, 1.0)
        # sprout leaves early on
        if 0.03 < g < 0.35 and 0 in pos:
            e, a = pos[0]
            sz = 0.03 * self.ow * ease(clamp((g - 0.03) / 0.1)) * (1 - ease(clamp((g - 0.25) / 0.1)))
            for s in (-1, 1):
                self._leaf(pen, e, -math.pi / 2 + s * 0.9, sz, 1.0)
        for i, (par, ang, ln, dep, b0) in enumerate(self.br):
            if dep == 3 and i in pos:
                lk = ease(clamp((g - 0.45) / 0.25))
                if lk > 0:
                    e, a = pos[i]
                    self._leaf(pen, e, a - 1.1, 0.024 * self.ow * lk, lk)
                    self._leaf(pen, e, a + 1.1, 0.02 * self.ow * lk, lk)
        # leaves + fruit / coins on the tips
        for j, i in enumerate(self.tips):
            if i not in pos:
                continue
            e, a = pos[i]
            lk = ease(clamp((g - 0.4) / 0.25))
            ox, oy = self.leaf_off[j]
            if lk > 0:
                self._leaf(pen, e, a - 0.6, 0.03 * self.ow * lk, lk)
                self._leaf(pen, e, a + 0.6, 0.025 * self.ow * lk, lk)
            fk = ease(clamp((g - 0.62 - 0.06 * abs(ox)) / 0.15))
            if fk > 0 and j % 3 == 0:
                fx, fy = e[0] + ox * 10 * u, e[1] + 0.022 * self.oh + 3 * u * math.sin(t * 3 + j)
                pen.seg(e, (fx, fy - 10 * u), dim, 1.2 * u, fk)
                if self.fruit == "coin":
                    self.coin(pen, fx, fy, 15 * u * fk, fk, spin=0.4 * math.sin(t * 2 + j))
                else:
                    pen.circle((fx, fy), 13 * u * fk, acc, 2.0 * u, fk, fill=acc, kf=0.35)
                    pen.seg((fx, fy - 13 * u * fk), (fx + 4 * u, fy - 20 * u * fk), lc, 1.4 * u, fk)
        if g > 0.75:
            pen.soft(rx, G - 0.25 * self.oh, 0.25 * self.ow, self.th["soft"], 0.25 * min(1.0, (g - 0.75) / 0.15))
            # a coin occasionally drops and bounces
            q = (t * 0.6) % 1.0
            if self.tips:
                e = pos.get(self.tips[len(self.tips) // 3], ((rx, G - 0.2 * self.oh), 0))[0]
                y = e[1] + (G - 12 * u - e[1]) * min(1.0, (q / 0.55) ** 2)
                if q > 0.55:
                    y -= abs(math.sin((q - 0.55) / 0.45 * math.pi * 2)) * 0.03 * self.oh * (1 - (q - 0.55) / 0.45)
                if self.fruit == "coin":
                    self.coin(pen, e[0] + 0.03 * self.ow * q, y, 12 * u, 1 - max(0.0, q - 0.85) / 0.15, spin=t * 7)
        # gardener with watering can
        hh = 0.15 * self.oh
        A = place(pose_stand(t), hh, self.X(0.25), G, 1, self.picto)
        tilt = 0.35 + 0.08 * math.sin(t * 3)
        hand = (A["sh" + "R"][0] + 0.25 * hh, A["shR"][1] + 0.12 * hh)
        arm_reach(A, "R", hand)
        arm_reach(A, "L", (hand[0] + 0.07 * hh, hand[1] + 0.02 * hh))
        self.figure(pen, A)
        ca, sa = math.cos(tilt), math.sin(tilt)

        def C(x, y):
            return (hand[0] + x * ca - y * sa, hand[1] + x * sa + y * ca)
        cw, chh = 0.1 * hh, 0.1 * hh
        body = [C(0, 0), C(cw * 1.4, 0), C(cw * 1.3, chh), C(0.1 * cw, chh)]
        pen.poly(body, acc, 2.0 * u, 1.0, fill=0)
        pen.line([C(-0.1 * cw, 0.2 * chh), C(-0.5 * cw, 0.5 * chh), C(-0.1 * cw, 0.8 * chh)], acc, 1.8 * u, 1.0)
        spout = C(cw * 2.4, -0.35 * chh)
        pen.seg(C(cw * 1.3, 0.55 * chh), spout, acc, 2.0 * u, 1.0)
        # water drops arcing to the soil
        tx, ty = rx - 0.03 * self.ow, G
        for f0 in self.drops:
            f = (f0 + t * 1.6) % 1.0
            x = lerp(spout[0], tx, f)
            y = lerp(spout[1], ty, f) - math.sin(math.pi * f) * 0.03 * self.oh + f * f * 0.02 * self.oh
            pen.seg((x, y), (x + 2 * u, y + 9 * u), self.th["accent2"], 2.0 * u, 0.9 * math.sin(math.pi * f))

    def _leaf(self, pen, base, ang, size, k):
        if size < 1.0:
            return
        u = self.u
        ca, sa = math.cos(ang), math.sin(ang)
        pts = []
        for i in range(9):
            f = i / 8
            pts.append((f, 0.35 * math.sin(math.pi * f)))
        for i in range(8, -1, -1):
            f = i / 8
            pts.append((f, -0.35 * math.sin(math.pi * f)))
        P_ = [(base[0] + (x * ca - y * sa) * size, base[1] + (x * sa + y * ca) * size) for x, y in pts]
        pen.poly(P_, self.th["line"], 1.6 * u, k, fill=0)


# ======================================================================== 19. domino chain
class LADomino(LAStory):
    """A chain of dominoes growing in size: the figure tips the tiny first one, they fall in sequence up
    to a giant one that crashes down (small habits -> big results)."""
    focus = (0.55, 0.52)
    label_at = (0.5, 0.715)
    N = 9

    def setup(self):
        self.G = self.Y(0.645)
        r = 1.3
        h0 = 0.025 * self.oh
        self.hs = [h0 * r ** i for i in range(self.N)]
        self.th_ = [0.2 * h for h in self.hs]
        xs = [self.X(0.15)]
        for i in range(self.N - 1):
            xs.append(xs[-1] + 0.42 * self.hs[i])
        shift = self.X(0.6) - xs[-1]
        self.xs = [x + shift for x in xs]
        # fall schedule: own fall phi_i(tau) = (pi/2) (tau/Tf_i)^2; the next one starts when the top hits it
        self.T0 = 0.55
        self.Tf = [0.32 * (h / h0) ** 0.3 for h in self.hs]
        self.starts = [self.T0]
        for i in range(self.N - 1):
            d = self.xs[i + 1] - (self.xs[i] + self.th_[i])
            th = math.asin(clamp(d / self.hs[i], 0, 1))
            self.starts.append(self.starts[-1] + self.Tf[i] * math.sqrt(th / (math.pi / 2)))
        self.crash = self.starts[-1] + self.Tf[-1]
        rng = self.rng
        self.dust = [(rng.uniform(-0.2, 1.1), rng.uniform(0.4, 1.0), rng.uniform(0.3, 1.0)) for _ in range(16)]

    def angles(self, t):
        free = []
        for i in range(self.N):
            tau = t - self.starts[i]
            free.append(0.0 if tau <= 0 else (math.pi / 2) * min(1.0, (tau / self.Tf[i]) ** 2))
        ang = list(free)
        for i in range(self.N - 2, -1, -1):  # resting on the next domino
            d = self.xs[i + 1] - (self.xs[i] + self.th_[i])
            h = self.hs[i]
            tn = math.tan(min(ang[i + 1], 1.5))
            th = ang[i]
            lim = th
            for _ in range(4):
                lim = math.asin(clamp((d + h * math.cos(lim) * tn) / h, 0, 1))
            ang[i] = min(ang[i], lim)
        return ang

    def draw(self, pen, t):
        u = self.u
        G = self.G
        lc, acc, dim = self.lc, self.acc, self.th["dim"]
        shake = 0.0
        if t > self.crash:
            shake = 7 * u * math.exp(-(t - self.crash) * 5) * math.sin((t - self.crash) * 60)
        self.ground(pen, G + shake * 0.3, self.X(-0.02), self.X(1.02), 1.0, 0.12)
        ang = self.angles(t)
        for i in range(self.N):
            h, w = self.hs[i], self.th_[i]
            px, py = self.xs[i] + w, G
            a = ang[i]
            ca, sa = math.cos(a), math.sin(a)

            def R(x, y):  # local: pivot at bottom-right, x left (negative), y up (negative)
                return (px + x * ca - y * sa + shake, py + x * sa + y * ca)
            pts = [R(-w, 0), R(-w, -h), R(0, -h), R(0, 0)]
            col = acc if i == self.N - 1 else lc
            pen.poly(pts, col, max(1.6, 1.4 + 0.25 * i) * u, 1.0, fill=0)
            # pips
            for fy in (0.3, 0.7):
                pen.dot(R(-w / 2, -h * fy), max(1.6, 0.13 * w), col, 0.9)
            pen.seg(R(-w * 0.85, -h / 2), R(-w * 0.15, -h / 2), col, 1.2 * u, 0.7)
            if i == self.N - 1 and t < self.crash:
                pen.soft(px - w / 2, G - h * 0.6, 0.12 * self.ow, self.th["soft"], 0.25)
        # impact: dust + shock lines
        if t > self.crash:
            q = t - self.crash
            gx = self.xs[-1] + self.th_[-1] + self.hs[-1] * 0.55
            for dx, sp, sz in self.dust:
                f = min(1.0, q / (1.1 * sp + 0.3))
                x = gx + (dx - 0.45) * self.hs[-1] * (0.6 + 1.2 * f)
                y = G - 8 * u - f * 0.05 * self.oh * sp
                pen.dot((x, y), (2.5 + 5 * sz * (1 - f)) * u, dim, (1 - f) * 0.9)
                pen.dot((x - (dx - 0.45) * 30 * u * f, y + 10 * u * f), (2 + 3 * sz) * u, dim, (1 - f) * 0.6)
            if q < 0.5:
                for j in range(6):
                    a = -math.pi / 2 + (j - 2.5) * 0.35
                    r0 = 0.06 * self.oh + q * 0.15 * self.oh
                    pen.seg((gx + math.cos(a) * r0, G - 0.03 * self.oh + math.sin(a) * r0),
                            (gx + math.cos(a) * (r0 + 0.03 * self.oh), G - 0.03 * self.oh + math.sin(a) * (r0 + 0.03 * self.oh)),
                            acc, 2.2 * u, 1 - q / 0.5)
        # the figure: walks up, taps the first one, then watches / cheers
        hh = 0.12 * self.oh
        fx = self.xs[0] - 0.075 * self.ow
        if t < self.T0:
            pose = mix(pose_stand(t), P(lean=0.3, head=0.25, thL=0.1, thR=-0.2, knL=0.05, knR=0.1, shL=0.1,
                                        shR=1.0, elL=0.2, elR=0.1), ease(t / self.T0))
        elif t < self.crash:
            pose = mix(P(lean=0.3, head=0.25, thL=0.1, thR=-0.2, knL=0.05, knR=0.1, shL=0.1, shR=1.0, elL=0.2, elR=0.1),
                       pose_lookup(t), ease((t - self.T0) / 0.5))
        else:
            pose = mix(pose_lookup(t), pose_cheer(t), ease((t - self.crash) / 0.3))
        A = place(pose, hh, fx, G, 1, self.picto)
        self.figure(pen, A)
        if self.T0 - 0.15 < t < self.T0 + 0.25:
            hx, hy = A["handR"]
            for j in range(3):
                a = -0.6 + j * 0.6
                pen.seg((hx + 8 * u * math.cos(a), hy + 8 * u * math.sin(a)),
                        (hx + 18 * u * math.cos(a), hy + 18 * u * math.sin(a)), acc, 1.6 * u, 0.9)


# ======================================================================== 20. boulder
class LABoulder(LAStory):
    """The figure pushes a huge boulder up a slope in surges, dust kicking up (struggle);
    params.mode='top' shows the boulder resting on the summit and the figure victorious."""
    focus = (0.55, 0.47)
    label_at = (0.5, 0.735)

    def setup(self):
        m = self.p.get("mode")
        self.top = isinstance(m, str) and m.strip().lower() in ("top", "summit", "win", "victory", "done", "peak")
        self.al = 0.36
        self.d = (math.cos(self.al), -math.sin(self.al))
        self.n = (-math.sin(self.al), -math.cos(self.al))
        self.O = (self.X(0.37), self.Y(0.655))
        self.R = 0.145 * self.ow
        self.h = 0.19 * self.oh
        rng = self.rng
        self.tile = 1.6 * self.ow
        self.marks = [(rng.uniform(0, self.tile), rng.uniform(0.3, 1.0), int(rng.integers(0, 3))) for _ in range(22)]
        self.cracks = []
        for _ in range(4):
            a = rng.uniform(0, TAU)
            r0 = rng.uniform(0.15, 0.5)
            pts = [(math.cos(a) * r0, math.sin(a) * r0)]
            for _ in range(3):
                a += rng.uniform(-0.6, 0.6)
                r0 = min(0.92, r0 + rng.uniform(0.12, 0.25))
                pts.append((math.cos(a) * r0, math.sin(a) * r0))
            self.cracks.append(pts)
        self.specks = [(rng.uniform(0, TAU), rng.uniform(0.2, 0.85)) for _ in range(9)]
        self.dust = [(rng.uniform(0, 1), rng.uniform(-1, 1)) for _ in range(10)]

    def S(self, s, off=0.0):
        """Screen point at slope coordinate s (px along the slope from O), ``off`` along the up-normal."""
        return (self.O[0] + self.d[0] * s + self.n[0] * off, self.O[1] + self.d[1] * s + self.n[1] * off)

    def boulder(self, pen, c, R, rot, k=1.0):
        u = self.u
        lc = self.lc
        pen.circle(c, R, lc, 2.6 * u, k, fill=0)
        ca, sa = math.cos(rot), math.sin(rot)
        for pts in self.cracks:
            P_ = [(c[0] + (x * ca - y * sa) * R, c[1] + (x * sa + y * ca) * R) for x, y in pts]
            pen.line(P_, self.th["dim"], 1.6 * u, k)
        for a, r in self.specks:
            pen.dot((c[0] + math.cos(a + rot) * r * R, c[1] + math.sin(a + rot) * r * R), 2.5 * u, self.th["dim"], k)
        # shading arcs on the lower-right
        for j in range(3):
            pen.arc(c, R * (0.86 - 0.08 * j), 10 + 8 * j, 80 - 6 * j, self.th["faint"], 1.6 * u, k)

    def draw(self, pen, t):
        if self.top:
            return self._draw_top(pen, t)
        u, h, R = self.u, self.h, self.R
        lc, acc, dim = self.lc, self.acc, self.th["dim"]
        v = 0.06 * self.oh
        c = v * (t + 0.24 * math.sin(2.7 * t))  # surges forward, slips back a little
        push = 0.5 + 0.5 * math.cos(2.7 * t)    # effort (1 at the strongest surge)
        # slope line and hatching below it
        pen.seg(self.S(-1.2 * self.ow), self.S(1.4 * self.ow), lc, 2.4 * u, 1.0)
        for sm, f, kind in self.marks:
            s = ((sm - c) % self.tile) - 0.6 * self.ow
            x, y = self.S(s)
            if kind == 0:  # grass tuft
                for j in (-1, 0, 1):
                    pen.seg((x + j * 4 * u, y), (x + j * 7 * u, y - 13 * u * f), dim, 1.4 * u, 0.9)
            elif kind == 1:  # pebble
                pen.ellipse((x - 7 * u * f, y - 7 * u * f, x + 7 * u * f, y), dim, 1.4 * u, 0.9)
            else:  # hatch below the surface
                p2 = (x - 0.05 * self.ow * f, y + 0.05 * self.oh * f)
                pen.seg((x, y + 6 * u), p2, self.th["faint"], 1.4 * u, 1.0)
        # boulder ahead of the figure
        sB = 0.5 * h + R * 0.72
        B = self.S(sB, R)
        self.boulder(pen, B, R, c / R)
        # figure: hips/feet on the slope via IK, hands on the boulder
        pose = pose_push(t)
        pose.update(lean=0.62 + 0.06 * push, head=-0.15)
        hip_s = -0.18 * h
        hx, _ = self.S(hip_s)
        hy = self.S(hip_s)[1] - 0.43 * h + 0.03 * h * push
        J = fk(pose, h, 1, self.picto)
        A = {k_: ((v_[0] + hx, v_[1] + hy) if isinstance(v_, tuple) else v_) for k_, v_ in J.items()}
        A.update(h=h, facing=1)
        step = math.sin(TAU * t / 0.9)
        back = self.S(hip_s - 0.3 * h + 0.03 * h * step)
        front = self.S(hip_s + 0.12 * h - 0.03 * h * step, 0.012 * h * max(0.0, step))
        leg_reach(A, "L", back)
        leg_reach(A, "R", front)
        for side, dy in (("R", -0.05), ("L", 0.06)):
            sh = A["sh" + side]
            vx, vy = B[0] - sh[0], B[1] + dy * h - sh[1]
            dist = math.hypot(vx, vy)
            tgt = (B[0] - vx / dist * R, B[1] + dy * h - vy / dist * R)
            arm_reach(A, side, tgt, bend=-1)
        self.figure(pen, A)
        # dust from the back foot + effort marks
        for f0, dx in self.dust:
            f = (f0 + t * 1.6) % 1.0
            x = back[0] - f * 0.07 * self.ow + dx * 6 * u
            y = back[1] - 4 * u - math.sin(math.pi * f) * 0.025 * self.oh
            pen.dot((x, y), (2 + 4 * f) * u, dim, (1 - f) * (0.4 + 0.6 * push))
        hx_, hy_ = A["head"]
        for j in range(3):
            a = -2.2 + j * 0.35
            r0 = 0.1 * h
            pen.seg((hx_ + math.cos(a) * r0, hy_ + math.sin(a) * r0),
                    (hx_ + math.cos(a) * (r0 + 0.05 * h * push), hy_ + math.sin(a) * (r0 + 0.05 * h * push)),
                    acc, 1.6 * u, push)
        # the summit far ahead
        fx, fy = self.X(0.88), self.Y(0.29)
        pen.seg((fx, fy), (fx, fy - 0.05 * self.oh), lc, 1.8 * u, 0.8)
        pen.poly([(fx, fy - 0.05 * self.oh), (fx + 0.045 * self.ow, fy - 0.038 * self.oh), (fx, fy - 0.026 * self.oh)],
                 acc, 1.6 * u, 0.9, fill=acc, kf=0.5)
        pen.soft(fx, fy - 0.03 * self.oh, 0.06 * self.ow, self.th["soft"], 0.5)

    def _draw_top(self, pen, t):
        u, h, R = self.u, self.h, self.R
        X, Y = self.X, self.Y
        lc, acc, dim = self.lc, self.acc, self.th["dim"]
        top_y = Y(0.6)
        self.sun(pen, X(0.42), top_y, 0.16 * self.ow, t, 0.9, horizon=top_y)
        hill = [(X(-0.05), Y(0.86)), (X(0.12), Y(0.74)), (X(0.3), top_y + 0.004 * self.oh), (X(0.86), top_y),
                (X(0.95), Y(0.66)), (X(1.06), Y(0.7))]
        pen.poly(hill + [(X(1.06), Y(0.95)), (X(-0.05), Y(0.95))], None, fill=0)
        pen.line(hill, lc, 2.4 * u, 1.0)
        for j in range(5):  # the long way up (dashed)
            f = j / 5
            pen.seg((lerp(X(-0.03), X(0.28), f), lerp(Y(0.84), top_y + 0.02 * self.oh, f)),
                    (lerp(X(-0.03), X(0.28), f + 0.1), lerp(Y(0.84), top_y + 0.02 * self.oh, f + 0.1)), dim, 1.6 * u, 0.8)
        settle = ease_out(t / 0.6)
        bx = X(0.71) - (1 - settle) * 0.05 * self.ow
        self.boulder(pen, (bx, top_y - R - 1.2 * u), R, (1 - settle) * -0.5)
        A = place(pose_cheer(t), h, X(0.42), top_y, 1, self.picto)
        self.figure(pen, A)
        for j in range(6):
            age = (t * 0.7 + j / 6) % 1.0
            ang = -math.pi / 2 + (j - 2.5) * 0.45
            r = (0.05 + 0.1 * age) * self.oh
            hx, hy = A["head"]
            self.sparkle(pen, hx + math.cos(ang) * r, hy + math.sin(ang) * r, 9 * u * (1 - age), math.sin(math.pi * age))


# ======================================================================== 21. gym
class LAGym(LAStory):
    """Discipline / body: the figure presses a barbell overhead rep after rep (params.mode='pushup' for
    push-ups); a row of rep dots fills up with a counter."""
    focus = (0.5, 0.5)
    label_at = None

    def setup(self):
        m = self.p.get("mode")
        self.pushup = isinstance(m, str) and m.strip().lower() in ("pushup", "push-up", "pushups", "push", "floor")
        self.per = 0.85 if self.pushup else 0.95
        self.G = self.Y(0.63)

    def counter(self, pen, t, reps):
        u = self.u
        n = 10
        y = self.Y(0.685)
        done = reps % (n + 1)
        for i in range(n):
            x = self.X(0.5) + (i - (n - 1) / 2) * 0.062 * self.ow
            if i < done:
                pop = 1 + 0.5 * (1 - ease_out(clamp((t - (math.floor(t / self.per)) * self.per) / 0.25))) * (i == done - 1)
                pen.circle((x, y), 11 * u * pop, self.acc, 2.0 * u, 1.0, fill=self.acc, kf=0.55)
            else:
                pen.circle((x, y), 9 * u, self.th["dim"], 1.8 * u, 1.0)
        rx, ry = self.X(0.82), self.Y(0.33)
        pen.text((rx, ry), str(reps), 96 * u, self.acc, 1.0, "mm", "Display")
        pen.text((rx, ry + 0.04 * self.oh), "REPS" if not self.label else self.label, 22 * u, self.th["text"], 0.8,
                 "mm", "Bold", 0.3)

    def draw(self, pen, t):
        u = self.u
        reps = int(max(0.0, t + 0.6) / self.per)
        q = ((t + 0.6) / self.per) % 1.0
        self.ground(pen, self.G, self.X(0.08), self.X(0.92), 1.0, 0.2)
        if self.pushup:
            self._pushup(pen, t, q)
        else:
            self._press(pen, t, q)
        self.counter(pen, t, reps)

    def _press(self, pen, t, q):
        u, G = self.u, self.G
        h = 0.2 * self.oh
        cx = self.X(0.45)
        if q < 0.12:
            dip, b = math.sin(math.pi * q / 0.12), 0.0
        elif q < 0.45:
            dip, b = 0.0, ease_out((q - 0.12) / 0.33)
        elif q < 0.6:
            dip, b = 0.0, 1.0
        else:
            dip, b = 0.0, 1 - ease_in_out((q - 0.6) / 0.4)
        pose = P(front=True, head=-0.05 - 0.1 * b, thL=0.2, thR=0.2, shL=0.4, shR=0.4, elL=0.5, elR=0.5,
                 liftL=0.28 * dip, liftR=0.28 * dip)
        A = place(pose, h, cx, G, 1, self.picto)
        sy = A["shL"][1]
        headtop = A["head"][1] - HEAD_R * h
        bar_y = lerp(sy - 0.02 * h, headtop - 0.13 * h, b)
        vel = math.sin(math.pi * clamp((q - 0.12) / 0.33)) if q < 0.45 else -math.sin(math.pi * clamp((q - 0.6) / 0.4))
        sag = 0.035 * h * vel
        gx = 0.21 * h
        arm_reach(A, "L", (cx - gx, bar_y), bend=-1)
        arm_reach(A, "R", (cx + gx, bar_y), bend=1)
        self.figure(pen, A)
        L = 0.46 * h
        pts = [(cx + f * L, bar_y + sag * (abs(f) ** 2)) for f in np.linspace(-1, 1, 11)]
        pen.line(pts, self.lc, 3.2 * u, 1.0)
        for sg in (-1, 1):
            for j, (off, hh) in enumerate(((0.33, 0.24), (0.38, 0.18))):
                x = cx + sg * off * h
                y = bar_y + sag * (off / 0.46) ** 2
                pen.rrect((x - 0.022 * h, y - hh * h / 2, x + 0.022 * h, y + hh * h / 2), 3 * u, self.acc, 2.2 * u, 1.0,
                          fill=0)
        if 0.4 < q < 0.65:  # effort lines + glow at lockout
            kk = math.sin(math.pi * (q - 0.4) / 0.25)
            for sg in (-1, 1):
                for j in range(3):
                    yy = bar_y - 0.02 * h + j * 0.04 * h
                    x0 = cx + sg * 0.52 * h
                    pen.seg((x0, yy), (x0 + sg * 0.06 * h, yy - 0.02 * h), self.acc, 1.8 * u, kk)
            pen.soft(cx, bar_y, 0.2 * self.ow, self.th["soft"], 0.35 * kk)
        hx, hy = A["head"]
        a = (t * 1.3) % 1.0
        pen.dot((hx + 0.08 * h + a * 0.05 * h, hy - 0.02 * h + a * a * 0.12 * h), 3.5 * u, self.th["accent2"], 1 - a)

    def _pushup(self, pen, t, q):
        u, G = self.u, self.G
        h = 0.27 * self.oh
        if q < 0.45:
            dn = ease_in_out(q / 0.45)
        else:
            dn = 1 - ease_in_out((q - 0.45) / 0.55)
        hs = lerp(0.29 * h, 0.12 * h, dn)  # shoulder height above the ground
        Lb = (THIGH + SHIN) * h + (TORSO - 0.03) * h
        beta = math.asin(clamp(hs / Lb, 0, 1))
        xf = self.X(0.17)
        foot = (xf, G - 0.02 * h)
        d = (math.cos(beta), -math.sin(beta))
        hip = (foot[0] + d[0] * (THIGH + SHIN) * h, foot[1] + d[1] * (THIGH + SHIN) * h)
        sh = (foot[0] + d[0] * Lb, foot[1] + d[1] * Lb)
        neck = (sh[0] + d[0] * 0.035 * h, sh[1] + d[1] * 0.035 * h)
        head = (neck[0] + 0.1 * h, neck[1] - 0.02 * h + 0.03 * h * (1 - dn))
        knee = ((foot[0] + hip[0]) / 2, (foot[1] + hip[1]) / 2)
        A = {"hip": hip, "neck": neck, "head": head, "h": h, "facing": 1, "front": False}
        for side, dx in (("L", -0.01), ("R", 0.01)):
            A["sh" + side] = sh
            A["hp" + side] = hip
            A["knee" + side] = knee
            A["foot" + side] = foot
            A["toe" + side] = (foot[0] - 0.02 * h, G)
            hand = (sh[0] + 0.03 * h + dx * h, G)
            E, Hd = _ik(sh, hand, UPPER_ARM * h, FOREARM * h, 1)
            A["el" + side], A["hand" + side] = E, Hd
        self.figure(pen, A)
        if dn > 0.8:
            pen.soft(sh[0], G - 0.02 * self.oh, 0.12 * self.ow, self.th["soft"], 0.3)
        a = (t * 1.2) % 1.0
        pen.dot((head[0] + 0.05 * h, head[1] + a * 0.1 * h), 3.5 * u, self.th["accent2"], 1 - a)


# ======================================================================== 22. book
class LABook(LAStory):
    """The figure reads in an armchair under a floor lamp; pages flip, little light bulbs (ideas) rise
    from the book and the stack of finished books grows (learning, knowledge)."""
    focus = (0.5, 0.5)
    label_at = (0.5, 0.715)

    def setup(self):
        self.G = self.Y(0.64)
        self.h = 0.2 * self.oh
        rng = self.rng
        self.ideas = [(rng.uniform(-1, 1), rng.uniform(0.7, 1.2)) for _ in range(5)]

    def draw(self, pen, t):
        u, h, G = self.u, self.h, self.G
        lc, acc, dim = self.lc, self.acc, self.th["dim"]
        self.ground(pen, G, self.X(0.04), self.X(0.96), 1.0, 0.2)
        # floor lamp with a light cone
        lx = self.X(0.74)
        top = self.Y(0.335)
        pen.seg((lx, G), (lx, top), lc, 2.4 * u, 1.0)
        pen.seg((lx - 0.05 * self.ow, G), (lx + 0.05 * self.ow, G), lc, 2.4 * u, 1.0)
        pen.line([(lx, top), (lx - 0.06 * self.ow, top - 0.015 * self.oh)], lc, 2.0 * u, 1.0)
        sx, sy = lx - 0.1 * self.ow, top - 0.012 * self.oh
        shade = [(sx - 0.05 * self.ow, sy + 0.035 * self.oh), (sx - 0.025 * self.ow, sy - 0.015 * self.oh),
                 (sx + 0.035 * self.ow, sy - 0.015 * self.oh), (sx + 0.06 * self.ow, sy + 0.035 * self.oh)]
        pen.poly(shade, acc, 2.2 * u, 1.0, fill=0)
        flick = 0.9 + 0.1 * math.sin(t * 13) * math.sin(t * 3.1)
        for j in range(7):
            f = j / 6
            a = (lerp(sx - 0.04 * self.ow, sx + 0.05 * self.ow, f), sy + 0.04 * self.oh)
            b = (lerp(self.X(0.28), self.X(0.66), f), G - 0.02 * self.oh)
            pts = [a, (lerp(a[0], b[0], 0.55), lerp(a[1], b[1], 0.55))]
            pen.line(pts, acc, 1.2 * u, 0.3 * flick)
        pen.soft(sx, sy + 0.08 * self.oh, 0.2 * self.ow, self.th["soft"], 0.45 * flick)
        # armchair + reader
        f = 1
        hipx = self.X(0.4)
        seat = G - seat_height(h, self.picto)
        pen.rrect((hipx - 0.2 * h, seat - 0.4 * h, hipx - 0.09 * h, G), 12 * u, lc, 2.2 * u, 1.0, fill=0)
        pen.rrect((hipx - 0.14 * h, seat - 0.01 * self.oh, hipx + 0.28 * h, seat + 0.025 * self.oh), 8 * u, lc, 2.2 * u,
                  1.0, fill=0)
        pen.seg((hipx + 0.24 * h, seat + 0.025 * self.oh), (hipx + 0.24 * h, G), lc, 2.0 * u, 1.0)
        pose = pose_sit(t, 0.12)
        pose.update(head=0.42 + 0.03 * math.sin(t * 1.5))
        A = seated(pose, h, hipx, seat, G, f, self.picto)
        bx, by = A["hip"][0] + 0.38 * h, A["hip"][1] - 0.2 * h
        arm_reach(A, "R", (bx - 0.12 * h, by - 0.03 * h))
        arm_reach(A, "L", (bx + 0.12 * h, by - 0.03 * h))
        self.figure(pen, A)
        q = (t / 0.55) % 1.0
        self.book_open(pen, bx, by, 0.36 * h, 1.0, q if q < 0.7 else None, tilt=-0.15)
        # ideas rising
        for i, (dx, sp) in enumerate(self.ideas):
            age = (t * 0.55 * sp + i / len(self.ideas)) % 1.0
            x = bx + dx * 0.06 * self.ow + 0.02 * self.ow * math.sin(t * 2 + i)
            y = by - 0.06 * self.oh - age * 0.22 * self.oh
            k = math.sin(math.pi * age)
            self.bulb(pen, x, y, (0.012 + 0.01 * age) * self.ow, k, k * 0.8, t)
        # growing stack of finished books
        n = 2 + int(min(6, t / 0.7))
        stx = self.X(0.18)
        for j in range(n):
            w = 0.12 * self.ow * (1 - 0.05 * (j % 3))
            off = (j % 2) * 6 * u - 3 * u
            self.book_closed(pen, stx + off, G - j * 0.022 * self.oh, w, 0.022 * self.oh, 1.0,
                             acc if j == n - 1 else lc)


# ======================================================================== registry
KINDS_LINEART = {
    "la_walk": LAWalk, "la_stairs": LAStairs, "la_clock": LAClock, "la_wall": LAWall,
    "la_calendar": LACalendar, "la_crowd": LACrowd, "la_fall_rise": LAFallRise, "la_door": LADoor,
    "la_mountain": LAMountain, "la_money": LAMoney, "la_phone": LAPhone, "la_mirror": LAMirror,
}
for _k, _c in (("la_split", "LASplit"), ("la_life", "LALife"), ("la_typo", "LATypo"), ("la_fork", "LAFork"),
               ("la_hourglass", "LAHourglass"), ("la_seed", "LASeed"), ("la_domino", "LADomino"),
               ("la_boulder", "LABoulder"), ("la_gym", "LAGym"), ("la_book", "LABook")):
    if _c in globals():
        KINDS_LINEART[_k] = globals()[_c]

# one-line catalogue for the script writer (key: what it shows -> what it means)
CATALOG = {
    "la_walk": "lone figure walking along a horizon under the stars (journey, starting, keep going, alone)",
    "la_stairs": "figure climbing an endless staircase that builds itself ahead (progress, discipline, small steps)",
    "la_clock": "huge clock with fast-spinning hands, tiny figure below (time, wasting time, life passing); "
                "params.mode='stop' freezes it (time stops, now)",
    "la_wall": "figure pushes a tall wall until it cracks and shatters (obstacles, fear, breaking through)",
    "la_calendar": "calendar pages tear off and fly away above a still figure (procrastination, someday, tomorrow); "
                   "params.label = comma list of page words, e.g. 'TOMORROW,NEXT WEEK,NEXT YEAR'",
    "la_crowd": "everyone walks past while the figure stands still (standing still, others moving on, comparison); "
                "params.pose='sit' sits him on a bench",
    "la_fall_rise": "figure runs, falls, kneels, gets back up and runs on (failure, getting back up, resilience)",
    "la_door": "a glowing doorway opens, the figure walks towards the light (opportunity, decision, new beginning)",
    "la_mountain": "figure climbs a zig-zag path up a mountain to a glowing summit flag (goals, success, long road)",
    "la_money": "coin stacks grow step by step with a compound growth curve (money, investing, compounding, wealth)",
    "la_phone": "figure hunched over a glowing phone while a time ring drains (distraction, scrolling, wasted time)",
    "la_mirror": "figure faces a mirror showing a brighter, upright future self (self-improvement, identity, future you)",
    "la_split": "split screen POOR vs RICH: two figures doing opposite habits side by side (comparison, poor vs rich "
                "mindset, bad vs good habits); params.action = phone (scrolls in bed vs reads) | spend (shopping bags, "
                "coins fly out vs coin stack grows) | sleep (oversleeps vs runs at sunrise) | car (flashy car drains money "
                "vs house makes money) | complain (storm cloud, complaining vs building a tower) | learn (watches TV vs "
                "works at a laptop, lightbulb); params.left / params.right = the two headings (default POOR / RICH)",
    "la_life": "stickman through life: baby -> kid with backpack -> student -> worker at a desk -> old man with a cane, "
               "ages 0/10/20/40/70 on a timeline, camera pans (life is short, time passes, regrets, the years go by); "
               "params.stage = 0-4 focuses one stage",
    "la_typo": "kinetic typography: params.text (1-3 words, e.g. 'NOBODY IS COMING') slams onto the screen letter by "
               "letter, last word in accent colour, underlined (harsh truths, punch lines, the key message)",
    "la_fork": "road splits into an easy wide road downhill (others take it) and a steep narrow path up to a light; "
               "the figure takes the hard path (choices, discipline, the hard road); params.left / params.right = sign "
               "words (default EASY / HARD)",
    "la_hourglass": "big hourglass with sand running out next to a worried figure, then it flips (time running out, "
                    "deadline, urgency, mortality)",
    "la_seed": "seed sprouts and grows into a big tree full of coins while the figure waters it (patience, compound "
               "growth, investing, consistency); params.fruit='fruit' for apples",
    "la_domino": "chain of dominoes growing in size, the figure tips the tiny first one and they topple up to a giant "
                 "one (small habits -> big results, momentum, chain reaction)",
    "la_boulder": "figure pushes a huge boulder uphill with dust and effort (struggle, grind, hard work, never give "
                  "up); params.mode='top' shows the boulder on the summit and the figure cheering (victory, made it)",
    "la_gym": "figure presses a barbell overhead rep after rep with a rep counter (discipline, body, gym, consistency); "
              "params.mode='pushup' for push-ups",
    "la_book": "figure reads in an armchair under a lamp, pages flip, small lightbulb ideas rise, the stack of finished "
               "books grows (learning, reading, knowledge, self-education)",
}

_procedural.KINDS.update(KINDS_LINEART)
