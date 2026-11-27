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
        "faint": (62, 62, 62), "accent": (255, 255, 255), "text": (237, 237, 237), "soft": (210, 220, 235)}
GOLD = {"bg": (11, 10, 13), "fig": (242, 179, 61), "line": (176, 168, 158), "dim": (92, 86, 80),
        "faint": (46, 42, 40), "accent": (242, 179, 61), "text": (242, 179, 61), "soft": (255, 150, 50)}
THEMES = {"mono": MONO, "gold": GOLD}
THEME_ALIASES = {"mono": "mono", "white": "mono", "black": "mono", "default": "mono", "silver": "mono",
                 "line": "mono", "minimal": "mono", "gold": "gold", "golden": "gold", "warm": "gold",
                 "amber": "gold", "timer": "gold"}


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
    return THEME_ALIASES.get(str(v).strip().lower(), "mono") if v is not None else "mono"


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
    """ImageDraw wrapper: takes layer (1x) coordinates and sizes, draws at ``SS`` x."""

    def __init__(self, img: Image.Image, mono: bool):
        self.img = img
        self.d = ImageDraw.Draw(img)
        self.mono = mono
        self.softs: list = []

    def c(self, col, k=1.0):
        k = max(0.0, float(k))
        if self.mono:
            if isinstance(col, (int, float)):
                return int(clamp(col * k, 0, 255))
            return int(clamp((0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]) * k, 0, 255))
        if isinstance(col, (int, float)):
            col = (col, col, col)
        return tuple(int(clamp(v * k, 0, 255)) for v in col)

    @staticmethod
    def _w(w):
        return max(1, int(round(w * SS)))

    def line(self, pts, col, w, k=1.0, caps=True):
        if len(pts) < 2 or k <= 0.003:
            return
        c = self.c(col, k)
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
        x, y = c[0] * SS, c[1] * SS
        R = max(0.5, r * SS)
        self.d.ellipse([x - R, y - R, x + R, y + R],
                       fill=None if fill is None else self.c(fill, kf),
                       outline=None if col is None or k <= 0.003 else self.c(col, k),
                       width=self._w(w) if col is not None else 0)

    def dot(self, c, r, col, k=1.0):
        if k <= 0.003:
            return
        x, y = c[0] * SS, c[1] * SS
        R = max(0.6, r * SS)
        self.d.ellipse([x - R, y - R, x + R, y + R], fill=self.c(col, k))

    def ellipse(self, box, col=None, w=1.0, k=1.0, fill=None, kf=1.0):
        x0, y0, x1, y1 = (v * SS for v in box)
        self.d.ellipse([x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)], fill=None if fill is None else self.c(fill, kf),
                       outline=None if col is None or k <= 0.003 else self.c(col, k), width=self._w(w) if col is not None else 0)

    def arc(self, c, r, a0, a1, col, w, k=1.0):
        """Arc from angle a0 to a1 (degrees, 0 = 3 o'clock, clockwise)."""
        if k <= 0.003 or a1 - a0 < 0.05:
            return
        x, y, R = c[0] * SS, c[1] * SS, r * SS
        self.d.arc([x - R, y - R, x + R, y + R], a0, a1, fill=self.c(col, k), width=self._w(w))

    def poly(self, pts, col=None, w=1.0, k=1.0, fill=None, kf=1.0):
        P = [(x * SS, y * SS) for x, y in pts]
        if fill is not None:
            self.d.polygon(P, fill=self.c(fill, kf))
        if col is not None and k > 0.003:
            self.line(list(pts) + [pts[0]], col, w, k, caps=False)

    def rrect(self, box, r, col=None, w=1.0, k=1.0, fill=None, kf=1.0):
        x0, y0, x1, y1 = (v * SS for v in box)
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
                occlude=True, w_scale=1.0):
    """Draw a placed figure (absolute joints from :func:`place`)."""
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
        hr = 0.085 * h * w_scale
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
    hr = HEAD_R * h
    if occlude:
        pen.circle(A["head"], hr + gap, fill=blk)
    pen.circle(A["head"], hr, col, w, k)
    if not front:
        arm("R", 1.0)


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
        self.th = THEMES[self.theme_name]
        self.mono = self.theme_name == "mono"
        self.picto = not self.mono
        self.mode = "L" if self.mono else "RGB"
        self.label = clean_word(p.get("label"))
        self.t0 = parse_float(p.get("t0"), 0.0)
        self.show_timer = parse_bool(p.get("timer"), not self.mono)
        self.u = self.ow / 1080.0
        rng = self.rng
        W, H = self.W, self.H
        # starfield (mono) / warm dust (gold)
        if self.mono:
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
        self.setup()
        if not self.mono:
            self.base = self._gold_base()
        g1, g2 = (0.85, 0.75) if self.mono else (0.5, 0.6)
        self._glut1 = [int(min(255, v * g1)) for v in range(256)] * (1 if self.mono else 3)
        self._glut2 = [int(min(255, v * g2)) for v in range(256)] * (1 if self.mono else 3)
        self._soft_gain = 0.42 if self.mono else 1.0

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

    # ---------------------------------------------------------------- frame
    def render(self, t):
        t = parse_float(t, 0.0)
        W, H = self.W, self.H
        img = Image.new(self.mode, (W * SS, H * SS), 0)
        pen = Pen(img, self.mono)
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
        s4 = small.reduce(4)
        b1 = s4.filter(ImageFilter.GaussianBlur(2.2 * max(0.3, self.u)))
        b2 = s4.reduce(2).filter(ImageFilter.GaussianBlur(3.5 * max(0.3, self.u))).resize(s4.size, Image.BILINEAR)
        g = ImageChops.add(b1.point(self._glut1), b2.point(self._glut2))
        if pen.softs:
            g = ImageChops.add(g, self._soft_layer(pen.softs, s4.size))
        if self.base is not None:
            g = ImageChops.add(g, self.base)
        glow = g.resize((W, H), Image.BILINEAR, box=(0, 0, W / 4, H / 4))
        if self.mono:
            return np.asarray(ImageChops.add(small, glow).convert("RGB"), np.float32)
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
        ch = 1 if self.mono else 3
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
            if self.mono:
                c = np.array([0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]], np.float32)
            else:
                c = np.asarray(col, np.float32)
            acc[y0:y1, x0:x1] += gg[..., None] * (c * self._soft_gain)[None, None, :]
        a = np.clip(acc + 0.5, 0, 255).astype(np.uint8)
        return Image.fromarray(a[..., 0] if self.mono else a)

    def draw_particles(self, pen, t):
        W, H = self.W, self.H
        u = self.u
        if self.mono:
            vx, vy = self.star_drift
            xs = (self.st_x + vx * u * self.st_d * t) % W
            ys = (self.st_y + vy * u * self.st_d * t) % H
            bs = self.st_b * (0.72 + 0.28 * np.sin(t * self.st_f * TAU * 0.5 + self.st_p))
            for x, y, r, b in zip(xs, ys, self.st_r, bs):
                pen.dot((x, y), r, int(b))
        else:
            vx, vy = self.star_drift
            xs = (self.st_x + vx * 0.5 * u * self.st_d * t + 9 * u * np.sin(t * 0.3 + self.st_p)) % W
            ys = (self.st_y - (7 + 5 * self.st_d) * u * t) % H
            bs = self.st_b * (0.55 + 0.45 * np.sin(t * self.st_f * TAU + self.st_p))
            gold = self.th["accent"]
            for x, y, r, b in zip(xs, ys, self.st_r, bs):
                pen.dot((x, y), r, gold, max(0.0, b))

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
        x, y = self.X(0.885), self.Y(0.175)
        pen.text((x, y), s, 92 * u, self.th["accent"], 1.0, "rs", "Display", 0.0)
        pen.text((x - 2 * u, y + 34 * u), "TIME GONE", 21 * u, self.th["line"], 0.75, "rm", "Bold", 0.28)

    def ground(self, pen, y, x0=None, x1=None, k=1.0, fade=0.22):
        x0 = self.X(-0.02) if x0 is None else x0
        x1 = self.X(1.02) if x1 is None else x1
        col = self.th["line"] if self.mono else (120, 116, 112)
        pen.fade_line(x0, x1, y, col, 1.7 * self.u, (0.8 if self.mono else 1.0) * k, fade)

    def figure(self, pen, A, k=1.0, **kw):
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


# ======================================================================== registry
KINDS_LINEART = {
    "la_walk": LAWalk, "la_stairs": LAStairs, "la_clock": LAClock, "la_wall": LAWall,
    "la_calendar": LACalendar, "la_crowd": LACrowd, "la_fall_rise": LAFallRise, "la_door": LADoor,
    "la_mountain": LAMountain, "la_money": LAMoney, "la_phone": LAPhone, "la_mirror": LAMirror,
}

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
}

_procedural.KINDS.update(KINDS_LINEART)
