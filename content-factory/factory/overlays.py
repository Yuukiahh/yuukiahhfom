"""Animated motion-graphics overlays that explain what is being said (infographics layer).

Each scene can carry one "overlay" (planned by the LLM from verified facts):
  headline   {"kicker": "BREAKING", "text": "...", "source": "CNBC", "date": "Nov 23"}
  timeline   {"events": [{"date": "Nov 12", "label": "Resigns"}, ...], "active": 1}
  bars       {"title": "...", "unit": "%", "items": [{"label": "Nov 23", "value": 31}, {"label": "Nov 25", "value": -8}]}
  counter    {"value": 31, "prefix": "+", "suffix": "%", "label": "stock jump", "decimals": 0}
  quote      {"text": "...", "source": "CNBC (sources familiar)"}
  person     {"name": "Jensen Huang", "role": "CEO, Nvidia", "status": "BACK AS CEO", "status_color": "green"}
  checklist  {"title": "...", "items": ["...", "..."]}
  compare    {"left": {"label": "...", "value": "..."}, "right": {"label": "...", "value": "..."}}
Plus automatic chrome: topic tag (top-left pill) and source lower-third per scene.
All drawing is PIL into a small RGBA box -> composited over the graded frame (stable, not
affected by camera moves). Layout stays inside the safe zone between hook text and captions.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .config import ROOT

FONT_DIR = ROOT / "assets" / "fonts"
GREEN, RED, ACCENT, ACCENT2, WHITE = (52, 211, 153), (248, 81, 104), (124, 58, 237), (56, 225, 255), (255, 255, 255)
CARD = (12, 14, 24, 150)
TYPES = ["headline", "timeline", "bars", "counter", "quote", "person", "checklist", "compare"]


@lru_cache(maxsize=32)
def F(size: int, weight: str = "black"):
    name = {"black": "InterDisplay-Black.otf", "bold": "Inter-Bold.otf", "xbold": "InterDisplay-ExtraBold.otf"}[weight]
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except OSError:
        return ImageFont.load_default()


def ease_out(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def back_out(x: float) -> float:
    x = min(1.0, max(0.0, x))
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


def _wrap(text: str, font, maxw: float) -> list[str]:
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if font.getlength(t) <= maxw or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _card(w: int, h: int, accent=ACCENT, radius: int = 34) -> Image.Image:
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius, fill=CARD, outline=(*accent, 160), width=3)
    d.rounded_rectangle([0, 0, 12, h - 1], 6, fill=(*accent, 255))
    return im


def _fade(im: Image.Image, a: float) -> Image.Image:
    if a >= 0.999:
        return im
    r, g, b, al = im.split()
    al = al.point(lambda v: int(v * max(0.0, a)))
    return Image.merge("RGBA", (r, g, b, al))


def _envelope(t: float, dur: float, fade_in: float = 0.25, fade_out: float = 0.25) -> float:
    return min(1.0, t / fade_in) * min(1.0, max(0.0, (dur - t) / fade_out))


# ------------------------------------------------------------------ overlay renderers (return RGBA box)
def r_headline(d: dict, t: float, dur: float, W: int) -> Image.Image:
    w = int(W * 0.88)
    fk, fh, fs = F(34, "bold"), F(62), F(30, "bold")
    lines = _wrap(d.get("text", ""), fh, w - 90)[:3]
    h = 150 + 74 * len(lines)
    im = _card(w, h, RED)
    dr = ImageDraw.Draw(im)
    kicker = (d.get("kicker") or "NEWS").upper()
    kw = fk.getlength(kicker) + 36
    dr.rounded_rectangle([40, 30, 40 + kw, 78], 12, fill=(*RED, 255))
    dr.text((58, 34), kicker, font=fk, fill=WHITE)
    meta = " · ".join(x for x in (d.get("source"), d.get("date")) if x)
    if meta:
        dr.text((60 + kw, 38), meta.upper(), font=fs, fill=(200, 205, 220))
    n_chars = sum(len(x) for x in lines)
    shown = int(n_chars * ease_out((t - 0.15) / 0.9))  # typewriter
    y = 100
    for ln in lines:
        part = ln[:max(0, shown)]
        shown -= len(ln)
        dr.text((40, y), part, font=fh, fill=WHITE)
        y += 74
    return im


def r_timeline(d: dict, t: float, dur: float, W: int) -> Image.Image:
    ev = d.get("events", [])[:4]
    w, h = int(W * 0.9), 330
    im = _card(w, h, ACCENT2)
    dr = ImageDraw.Draw(im)
    if not ev:
        return im
    x0, x1, y = 160, w - 160, 150
    prog = ease_out(t / max(0.6, min(dur * 0.6, 2.2)))
    dr.line([x0, y, x1, y], fill=(80, 90, 120, 255), width=8)
    dr.line([x0, y, x0 + (x1 - x0) * prog, y], fill=(*ACCENT2, 255), width=8)
    active = d.get("active", len(ev) - 1)
    for i, e in enumerate(ev):
        fx = i / max(1, len(ev) - 1) if len(ev) > 1 else 0.5
        cx = x0 + (x1 - x0) * fx
        reached = prog >= fx - 1e-3
        s = back_out((prog - fx) * 6 + 1) if reached else 0.0
        r = 22 * s * (1.35 if i == active else 1)
        col = ACCENT2 if i != active else (255, 214, 10)
        if r > 1:
            dr.ellipse([cx - r - 8, y - r - 8, cx + r + 8, y + r + 8], fill=(*col, 70))
            dr.ellipse([cx - r, y - r, cx + r, y + r], fill=(*col, 255))
        fd, fl = F(40), F(32, "bold")
        date, label = str(e.get("date", "")), str(e.get("label", ""))
        a = int(255 * min(1, s))
        dr.text((cx - fd.getlength(date) / 2, y - 105), date, font=fd, fill=(*WHITE, a))
        for k, ln in enumerate(_wrap(label, fl, 290)[:2]):
            dr.text((cx - fl.getlength(ln) / 2, y + 45 + k * 38), ln, font=fl, fill=(*(col if i == active else (210, 215, 230)), a))
    return im


def fmt_value(x: float, unit: str = "", signed: bool = False, ref: float | None = None, locale: str = "en") -> str:
    """$2,999 / 31 % / -5.9% / 4,3 Mio. – currency in front, sign only for changes, thousands separators."""
    ref = x if ref is None else ref
    dec = 0 if abs(ref) >= 10 or float(ref).is_integer() else 1
    num = f"{abs(x):,.{dec}f}"
    if locale == "de":
        num = num.replace(",", "X").replace(".", ",").replace("X", ".")
    sign = ("+" if x > 0 else "-" if x < 0 else "") if signed else ("-" if x < 0 else "")
    if unit in ("$", "€", "£", "¥"):
        return f"{sign}{unit}{num}" if locale != "de" or unit != "€" else f"{sign}{num} €"
    return f"{sign}{num}{unit}"


def r_bars(d: dict, t: float, dur: float, W: int) -> Image.Image:
    items = d.get("items", [])[:4]
    has_neg_pre = any(float(it.get("value", 0)) < 0 for it in items)
    w, h = int(W * 0.88), (620 if has_neg_pre else 520)
    im = _card(w, h, ACCENT)
    dr = ImageDraw.Draw(im)
    ft, fv, fl = F(44), F(52), F(32, "bold")
    if d.get("title"):
        dr.text((40, 26), d["title"], font=ft, fill=WHITE)
    if not items:
        return im
    vals = [float(it.get("value", 0)) for it in items]
    vmax = max(abs(v) for v in vals) or 1
    has_neg = any(v < 0 for v in vals)
    base = 340 if has_neg else 450
    span = 170 if has_neg else 280
    dr.line([40, base, w - 40, base], fill=(120, 130, 160, 255), width=3)
    bw = (w - 80) / len(items)
    unit = d.get("unit", "")
    for i, (it, v) in enumerate(zip(items, vals)):
        g = ease_out((t - 0.15 - i * 0.18) / 0.8)
        hh = span * abs(v) / vmax * g
        x = 40 + i * bw + bw * 0.2
        col = GREEN if v >= 0 else RED
        if v >= 0:
            dr.rounded_rectangle([x, base - hh, x + bw * 0.6, base], 12, fill=(*col, 255))
            ty = base - hh - 62
        else:
            dr.rounded_rectangle([x, base, x + bw * 0.6, base + hh], 12, fill=(*col, 255))
            ty = base + hh + 8
        shown = v * g
        txt = fmt_value(shown, unit, signed=has_neg or unit in ("%", "pp", "Pp"), ref=v, locale=d.get("locale", "en"))
        if g > 0.05:
            dr.text((x + bw * 0.3 - fv.getlength(txt) / 2, ty), txt, font=fv, fill=(*col, 255))
        lab = str(it.get("label", ""))
        ly = base + 14 if v >= 0 else base - 50
        dr.text((x + bw * 0.3 - fl.getlength(lab) / 2, ly), lab, font=fl, fill=(210, 215, 230, 255))
    return im


def r_counter(d: dict, t: float, dur: float, W: int) -> Image.Image:
    w, h = int(W * 0.8), 300
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle([0, 10, w - 1, h - 1], 40, fill=(10, 12, 22, 150))
    dr = ImageDraw.Draw(im)
    v = float(d.get("value", 0))
    dec = int(d.get("decimals", 0 if abs(v) >= 10 or v == int(v) else 1))
    shown = v * ease_out((t - 0.05) / 1.1)
    num = f"{d.get('prefix', '')}{shown:,.{dec}f}{d.get('suffix', '')}".replace(",", "X").replace(".", ",").replace("X", ".") \
        if d.get("locale", "en") == "de" else f"{d.get('prefix', '')}{shown:,.{dec}f}{d.get('suffix', '')}"
    fv = F(150)
    if fv.getlength(num) > w - 40:
        fv = F(int(150 * (w - 40) / fv.getlength(num)))
    col = GREEN if d.get("color") == "green" or (v > 0 and d.get("prefix") == "+") else RED if d.get("color") == "red" or v < 0 else ACCENT2
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(glow).text(((w - fv.getlength(num)) / 2, 40), num, font=fv, fill=(*col, 255))
    im.alpha_composite(glow.filter(ImageFilter.GaussianBlur(18)))
    dr.text(((w - fv.getlength(num)) / 2, 40), num, font=fv, fill=WHITE, stroke_width=3, stroke_fill=(*col, 255))
    lab = str(d.get("label", "")).upper()
    fl = F(38, "bold")
    dr.text(((w - fl.getlength(lab)) / 2, 225), lab, font=fl, fill=(230, 235, 245, 255))
    return im


def r_quote(d: dict, t: float, dur: float, W: int) -> Image.Image:
    w = int(W * 0.88)
    fq, fs = F(52, "xbold"), F(32, "bold")
    lines = _wrap(f"„{d.get('text', '')}“" if d.get("locale") == "de" else f"“{d.get('text', '')}”", fq, w - 100)[:4]
    h = 120 + 64 * len(lines)
    im = _card(w, h, (255, 214, 10))
    dr = ImageDraw.Draw(im)
    y = 40
    for i, ln in enumerate(lines):
        a = int(255 * ease_out((t - 0.1 - i * 0.2) / 0.4))
        dr.text((50, y), ln, font=fq, fill=(*WHITE, a))
        y += 64
    src = f"— {d.get('source', '')}"
    dr.text((50, y + 10), src, font=fs, fill=(255, 214, 10, int(255 * ease_out((t - 0.6) / 0.4))))
    return im


def _silhouette(size: int, col) -> Image.Image:
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    dr.ellipse([0, 0, size - 1, size - 1], fill=(30, 34, 52, 255), outline=(*col, 255), width=6)
    dr.ellipse([size * 0.33, size * 0.18, size * 0.67, size * 0.52], fill=(170, 178, 200, 255))
    dr.pieslice([size * 0.16, size * 0.55, size * 0.84, size * 1.2], 180, 360, fill=(170, 178, 200, 255))
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).ellipse([3, 3, size - 4, size - 4], fill=255)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(im, (0, 0), m)
    ImageDraw.Draw(out).ellipse([0, 0, size - 1, size - 1], outline=(*col, 255), width=6)
    return out


def r_person(d: dict, t: float, dur: float, W: int) -> Image.Image:
    w, h = int(W * 0.86), 250
    stc = {"green": GREEN, "red": RED, "yellow": (255, 214, 10)}.get(d.get("status_color", "green"), GREEN)
    im = _card(w, h, stc)
    im.alpha_composite(_silhouette(170, stc), (40, 40))
    dr = ImageDraw.Draw(im)
    dr.text((240, 46), str(d.get("name", "")), font=F(58), fill=WHITE)
    dr.text((240, 116), str(d.get("role", "")), font=F(34, "bold"), fill=(190, 198, 215))
    st = str(d.get("status", "")).upper()
    if st:
        s = back_out((t - 0.35) / 0.35)
        fs = F(34)
        bw = fs.getlength(st) + 40
        badge = Image.new("RGBA", (int(bw) + 4, 58), (0, 0, 0, 0))
        bd = ImageDraw.Draw(badge)
        bd.rounded_rectangle([0, 0, bw, 54], 14, fill=(*stc, 255))
        bd.text((20, 8), st, font=fs, fill=(10, 12, 20))
        if s > 0.05:
            bs = badge.resize((max(1, int(badge.width * s)), max(1, int(badge.height * s))))
            im.alpha_composite(bs, (240, 168 - bs.height // 2 + 10))
    return im


def r_checklist(d: dict, t: float, dur: float, W: int) -> Image.Image:
    items = d.get("items", [])[:4]
    w = int(W * 0.86)
    fi = F(44, "xbold")
    rows = [_wrap(str(x), fi, w - 170)[:2] for x in items]
    h = 110 + sum(58 * len(r) + 26 for r in rows)
    im = _card(w, h, GREEN)
    dr = ImageDraw.Draw(im)
    if d.get("title"):
        dr.text((40, 28), str(d["title"]).upper(), font=F(36), fill=(*GREEN, 255))
    y = 96
    step = max(0.35, min(0.8, (dur - 0.6) / max(1, len(items))))
    for i, r in enumerate(rows):
        s = back_out((t - 0.2 - i * step) / 0.3)
        if s > 0.02:
            cx, cy = 70, y + 28
            rr = 26 * s
            dr.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(*GREEN, 255))
            if s > 0.6:
                dr.line([cx - 11, cy, cx - 3, cy + 9, cx + 13, cy - 10], fill=(10, 12, 20), width=6)
            a = int(255 * min(1, s))
            for k, ln in enumerate(r):
                dr.text((120, y + k * 58), ln, font=fi, fill=(*WHITE, a))
        y += 58 * len(r) + 26
    return im


def r_compare(d: dict, t: float, dur: float, W: int) -> Image.Image:
    w, h = int(W * 0.9), 330
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    half = (w - 30) // 2
    for k, (side, col, x) in enumerate((("left", ACCENT2, 0), ("right", (255, 214, 10), half + 30))):
        s = ease_out((t - 0.1 - k * 0.3) / 0.45)
        if s <= 0.02:
            continue
        sd = d.get(side, {})
        c = _card(half, h, col)
        dr = ImageDraw.Draw(c)
        lab = _wrap(str(sd.get("label", "")).upper(), F(32, "bold"), half - 60)[:2]
        for i, ln in enumerate(lab):
            dr.text((36, 34 + i * 38), ln, font=F(32, "bold"), fill=(*col, 255))
        val = str(sd.get("value", ""))
        fv = F(76)
        if fv.getlength(val) > half - 60:
            fv = F(int(76 * (half - 60) / fv.getlength(val)))
        dr.text((36, 150), val, font=fv, fill=WHITE)
        c = _fade(c, s)
        im.alpha_composite(c, (int(x + (1 - s) * (40 if k else -40)), 0))
    vs = F(48)
    ImageDraw.Draw(im).text((w / 2 - vs.getlength("VS") / 2, h / 2 - 30), "VS", font=vs, fill=(255, 255, 255, int(255 * ease_out((t - 0.5) / 0.3))))
    return im


RENDER = {"headline": r_headline, "timeline": r_timeline, "bars": r_bars, "counter": r_counter, "quote": r_quote,
          "person": r_person, "checklist": r_checklist, "compare": r_compare}


def chrome_tag(text: str, W: int) -> Image.Image:
    f = F(30, "bold")
    tw = f.getlength(text)
    im = Image.new("RGBA", (int(tw + 70), 56), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    dr.rounded_rectangle([0, 0, tw + 66, 52], 26, fill=(12, 14, 24, 190), outline=(*ACCENT2, 170), width=2)
    dr.ellipse([18, 19, 32, 33], fill=(*RED, 255))
    dr.text((44, 10), text, font=f, fill=WHITE)
    return im


def chrome_source(text: str, W: int, lang: str = "de") -> Image.Image:
    f = F(26, "bold")
    word = "Quelle" if lang == "de" else "Source"
    s = f"{word}: {text}" if text and not text.lower().startswith(("source", "quelle")) else text
    tw = f.getlength(s)
    im = Image.new("RGBA", (int(tw + 40), 44), (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle([0, 0, tw + 36, 40], 12, fill=(0, 0, 0, 150))
    ImageDraw.Draw(im).text((18, 6), s, font=f, fill=(215, 220, 235))
    return im


class OverlayLayer:
    """Per-scene compositor used by render_scene_clip. Times are local to the scene clip."""

    def __init__(self, spec: dict, w: int, h: int):
        self.W, self.H = w, h
        self.ov = spec.get("overlay")
        self.t0 = float(spec.get("overlay_start", 0.35))
        self.t1 = float(spec.get("overlay_end", spec["duration"] - 0.2))
        self.tag = spec.get("tag")
        self.tag_from = float(spec.get("tag_from", 0.0))
        self.source = spec.get("source_label")
        self._tag_im = chrome_tag(self.tag, w) if self.tag else None
        self._src_im = chrome_source(self.source, w, spec.get("lang", "de")) if self.source else None
        self.y_center = float(spec.get("overlay_y", 0.36))
        acc = str((spec.get("params") or {}).get("accent") or "#38E1FF").lstrip("#")
        try:
            self.accent = tuple(int(acc[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            self.accent = ACCENT2

    def active(self) -> bool:
        return bool(self.ov or self._tag_im or self._src_im)

    @staticmethod
    def _comp(frame: np.ndarray, im: Image.Image, x: int, y: int) -> None:
        h, w = im.height, im.width
        H, W = frame.shape[:2]
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        a = np.asarray(im, np.float32)[y0 - y:y1 - y, x0 - x:x1 - x]
        al = a[..., 3:4] / 255
        reg = frame[y0:y1, x0:x1].astype(np.float32)
        frame[y0:y1, x0:x1] = (reg * (1 - al) + a[..., :3] * al).astype(np.uint8)

    def _glass(self, frame: np.ndarray, im: Image.Image, x: int, y: int, env: float, lt: float) -> None:
        """Frosted glass behind the card + soft accent glow around it."""
        H, W = frame.shape[:2]
        pad = 40
        x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + im.width + pad), min(H, y + im.height + pad)
        if x1 <= x0 or y1 <= y0:
            return
        region = Image.fromarray(frame[y0:y1, x0:x1])
        rw, rh = region.size
        blurred = np.asarray(region.reduce(3).filter(ImageFilter.GaussianBlur(6)).resize((rw, rh), Image.BILINEAR), np.float32)
        alpha = Image.new("L", (rw, rh), 0)
        alpha.paste(im.split()[3].point(lambda v: 255 if v > 8 else 0), (x - x0, y - y0))
        a = np.asarray(alpha, np.float32)[..., None] / 255 * env
        glow_a = np.asarray(alpha.reduce(4).filter(ImageFilter.GaussianBlur(6)).resize((rw, rh), Image.BILINEAR), np.float32)[..., None] / 255
        reg = frame[y0:y1, x0:x1].astype(np.float32)
        reg = reg * (1 - a) + (blurred * 0.85) * a
        reg = reg + np.clip(glow_a - a, 0, 1) * np.array(self.accent, np.float32) * 0.55 * env
        frame[y0:y1, x0:x1] = np.clip(reg, 0, 255).astype(np.uint8)

    @staticmethod
    def _shine(im: Image.Image, lt: float) -> Image.Image:
        """Diagonal light sweep across the card during its entrance."""
        p = (lt - 0.15) / 0.7
        if not 0 <= p <= 1:
            return im
        w, h = im.size
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        pos = (xx + yy * 0.5) / (w + h * 0.5)
        band = np.exp(-((pos - (p * 1.4 - 0.2)) / 0.05) ** 2)
        arr = np.asarray(im, np.float32).copy()
        arr[..., :3] = np.clip(arr[..., :3] + band[..., None] * 120 * (arr[..., 3:4] > 0), 0, 255)
        return Image.fromarray(arr.astype(np.uint8), "RGBA")

    def apply(self, frame: np.ndarray, t: float) -> np.ndarray:
        frame = np.ascontiguousarray(frame)
        if self._tag_im and t >= self.tag_from:
            a = min(1.0, (t - self.tag_from) / 0.3)
            self._comp(frame, _fade(self._tag_im, a), 40, int(self.H * 0.045))
        if self._src_im and self.t0 <= t:
            a = _envelope(t - self.t0, self.t1 - self.t0 + 0.2)
            self._comp(frame, _fade(self._src_im, a), 40, int(self.H * 0.915))
        if self.ov and self.t0 <= t < self.t1:
            lt = t - self.t0
            fn = RENDER.get(self.ov.get("type"))
            if fn:
                im = fn(self.ov.get("data", {}), lt, self.t1 - self.t0, self.W)
                env = _envelope(lt, self.t1 - self.t0, 0.22, 0.2)
                slide = (1 - ease_out(lt / 0.35)) * 60
                x = (self.W - im.width) // 2
                y = int(self.H * self.y_center - im.height / 2 + slide)
                self._glass(frame, im, x, y, env, lt)
                im = _fade(self._shine(im, lt), env)
                self._comp(frame, im, x, y)
        return frame
