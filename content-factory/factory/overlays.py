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
TYPES = ["headline", "timeline", "bars", "counter", "quote", "person", "checklist", "compare",
         "steps", "notify", "calc", "prompt", "stack", "meter"]


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


# ------------------------------------------------------------------ v3 creator / how-to overlays
GOLD, PINK = (255, 214, 10), (255, 92, 170)


def _monogram(label: str, size: int, col) -> Image.Image:
    """Rounded app-tile with the first letters of a tool name (no real logos)."""
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    top = tuple(min(255, int(c * 1.15) + 20) for c in col)
    for y in range(size):
        k = y / size
        dr.line([(0, y), (size, y)], fill=(*[int(top[i] * (1 - k) + col[i] * 0.55 * k) for i in range(3)], 255))
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size - 1, size - 1], size // 4, fill=255)
    im.putalpha(m)
    letters = "".join(w[0] for w in str(label).split()[:2]).upper() or "?"
    f = F(int(size * 0.42))
    ImageDraw.Draw(im).text(((size - f.getlength(letters)) / 2, size * 0.24), letters, font=f, fill=WHITE)
    return im


_PALETTE = [(56, 225, 255), (255, 92, 170), (255, 214, 10), (52, 211, 153), (124, 58, 237), (255, 140, 60)]


def r_steps(d: dict, t: float, dur: float, W: int) -> Image.Image:
    """Numbered how-to step: big index badge, title, detail line, step dots."""
    w = int(W * 0.88)
    n, total = int(d.get("n", 1)), int(d.get("total", 0) or 0)
    ft, fd = F(60), F(36, "bold")
    tl = _wrap(str(d.get("title", "")), ft, w - 250)[:2]
    dl = _wrap(str(d.get("detail", "")), fd, w - 250)[:2]
    h = max(250, 84 + 70 * len(tl) + 46 * len(dl) + (56 if total else 20))
    im = _card(w, h, GOLD)
    dr = ImageDraw.Draw(im)
    s = back_out((t - 0.05) / 0.4)
    if s > 0.02:
        r = 78 * s
        cx, cy = 120, 120
        dr.ellipse([cx - r - 10, cy - r - 10, cx + r + 10, cy + r + 10], fill=(*GOLD, 60))
        dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*GOLD, 255))
        fn = F(int(96 * s) or 1)
        dr.text((cx - fn.getlength(str(n)) / 2, cy - 58 * s), str(n), font=fn, fill=(14, 14, 22))
    y = 40
    lab = (f"STEP {n}/{total}" if total else f"STEP {n}") if d.get("locale") != "de" else (f"SCHRITT {n}/{total}" if total else f"SCHRITT {n}")
    dr.text((230, y), lab, font=F(30, "bold"), fill=(*GOLD, 255))
    y += 44
    for ln in tl:
        dr.text((230, y), ln, font=ft, fill=(*WHITE, int(255 * ease_out((t - 0.15) / 0.35))))
        y += 70
    for ln in dl:
        dr.text((230, y), ln, font=fd, fill=(200, 206, 222, int(255 * ease_out((t - 0.45) / 0.4))))
        y += 46
    if total:
        for i in range(total):
            x = 230 + i * 46
            on = i < n
            dr.rounded_rectangle([x, h - 40, x + (34 if i == n - 1 else 20), h - 26], 7, fill=(*(GOLD if on else (90, 96, 120)), 255))
    return im


def r_notify(d: dict, t: float, dur: float, W: int) -> Image.Image:
    """Stack of phone push notifications dropping in one after another."""
    items = d.get("items", [])[:3]
    w = int(W * 0.86)
    rowh, gap = 150, 22
    h = max(1, len(items)) * (rowh + gap)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    for i, it in enumerate(items):
        s = ease_out((t - 0.1 - i * 0.45) / 0.35)
        if s <= 0.02:
            continue
        col = _PALETTE[i % len(_PALETTE)]
        c = Image.new("RGBA", (w, rowh), (0, 0, 0, 0))
        cd = ImageDraw.Draw(c)
        cd.rounded_rectangle([0, 0, w - 1, rowh - 1], 38, fill=(236, 238, 246, 235))
        c.alpha_composite(_monogram(it.get("app", "App"), 92, col), (28, 29))
        cd.text((142, 24), str(it.get("app", "")).upper(), font=F(26, "bold"), fill=(110, 115, 135))
        cd.text((w - 40 - F(26, "bold").getlength(str(it.get("time", "now"))), 24), str(it.get("time", "now")), font=F(26, "bold"), fill=(130, 135, 150))
        title = str(it.get("title", ""))
        ft = F(40)
        while ft.size > 24 and ft.getlength(title) > w - 180:
            ft = F(ft.size - 2)
        cd.text((142, 58), title, font=ft, fill=(16, 18, 28))
        body = str(it.get("text", ""))
        fb = F(30, "bold")
        cd.text((142, 104), _wrap(body, fb, w - 180)[0] if body else "", font=fb, fill=(70, 75, 95))
        im.alpha_composite(_fade(c, s), (0, int(i * (rowh + gap) - (1 - s) * 80)))
    return im


def r_calc(d: dict, t: float, dur: float, W: int) -> Image.Image:
    """Worked example: a × b = result with labels (only numbers from the script/sources)."""
    terms = d.get("terms", [])[:3]
    w, h = int(W * 0.9), 420
    im = _card(w, h, GREEN)
    dr = ImageDraw.Draw(im)
    if d.get("title"):
        dr.text((40, 28), str(d["title"]).upper(), font=F(34), fill=(*GREEN, 255))
    ops = d.get("ops") or ["×"] * max(0, len(terms) - 1)
    parts = []
    for i, tm in enumerate(terms):
        parts.append(("v", str(tm.get("value", "")), str(tm.get("label", ""))))
        if i < len(terms) - 1:
            parts.append(("o", str(ops[i] if i < len(ops) else "×"), ""))
    fv, fo, fl = F(64), F(56), F(26, "bold")
    widths = [(fv if k == "v" else fo).getlength(v) + (24 if k == "v" else 40) for k, v, _ in parts]
    sc = min(1.0, (w - 80) / max(1, sum(widths)))
    if sc < 1:
        fv, fo = F(int(64 * sc)), F(int(56 * sc))
        widths = [(fv if k == "v" else fo).getlength(v) + (24 if k == "v" else 40) * sc for k, v, _ in parts]
    x = (w - sum(widths)) / 2
    for i, ((k, v, lab), wd) in enumerate(zip(parts, widths)):
        a = int(255 * ease_out((t - 0.1 - i * 0.18) / 0.3))
        if k == "v":
            dr.text((x + 12, 110), v, font=fv, fill=(*WHITE, a))
            for j, ln in enumerate(_wrap(lab.upper(), fl, wd + 20)[:2]):
                dr.text((x + wd / 2 - fl.getlength(ln) / 2, 192 + j * 30), ln, font=fl, fill=(180, 188, 210, a))
        else:
            dr.text((x + 20, 116), v, font=fo, fill=(*GOLD, a))
        x += wd
    res = str(d.get("result", ""))
    s = back_out((t - 0.25 - len(parts) * 0.18) / 0.4)
    if res and s > 0.02:
        fr = F(int(96 * max(0.2, s)))
        while fr.size > 30 and fr.getlength("= " + res) > w - 80:
            fr = F(fr.size - 4)
        txt = "= " + res
        glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(glow).text(((w - fr.getlength(txt)) / 2, 270), txt, font=fr, fill=(*GREEN, 255))
        im.alpha_composite(glow.filter(ImageFilter.GaussianBlur(14)))
        dr.text(((w - fr.getlength(txt)) / 2, 270), txt, font=fr, fill=WHITE, stroke_width=2, stroke_fill=(*GREEN, 255))
    return im


def r_prompt(d: dict, t: float, dur: float, W: int) -> Image.Image:
    """AI tool prompt box: tool chip, typed prompt with cursor, pulsing generate button."""
    w = int(W * 0.9)
    fp = F(40, "bold")
    text = str(d.get("text", ""))
    lines = _wrap(text, fp, w - 90)[:4]
    h = 210 + 52 * len(lines)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    dr.rounded_rectangle([0, 0, w - 1, h - 1], 30, fill=(16, 18, 30, 215), outline=(*PINK, 190), width=3)
    tool = str(d.get("tool", "AI"))
    im.alpha_composite(_monogram(tool, 60, PINK), (30, 24))
    dr.text((106, 36), tool, font=F(34), fill=WHITE)
    dr.text((w - 40 - F(26, "bold").getlength("PROMPT"), 42), "PROMPT", font=F(26, "bold"), fill=(*PINK, 255))
    n_chars = sum(len(x) for x in lines)
    shown = int(n_chars * min(1.0, max(0.0, (t - 0.2) / max(0.6, min(2.0, dur * 0.5)))))
    y = 108
    cur = None
    for ln in lines:
        part = ln[:max(0, shown)]
        shown -= len(ln)
        dr.text((40, y), part, font=fp, fill=(230, 234, 245))
        if part and (cur is None or len(part) < len(ln)):
            cur = (40 + fp.getlength(part) + 4, y)
            if len(part) < len(ln):
                break
        y += 52
    if cur and int(t * 2.5) % 2 == 0:
        dr.rectangle([cur[0], cur[1] + 4, cur[0] + 5, cur[1] + 46], fill=(*PINK, 255))
    btn = "Generate" if d.get("locale") != "de" else "Generieren"
    fb = F(32)
    bw = fb.getlength(btn) + 60
    pulse = 0.5 + 0.5 * math.sin(t * 5)
    bx, by = w - 40 - bw, h - 78
    dr.rounded_rectangle([bx - 6 * pulse, by - 6 * pulse, bx + bw + 6 * pulse, by + 56 + 6 * pulse], 22, fill=(*PINK, int(60 + 40 * pulse)))
    dr.rounded_rectangle([bx, by, bx + bw, by + 56], 18, fill=(*PINK, 255))
    dr.text((bx + 30, by + 9), btn, font=fb, fill=WHITE)
    return im


def r_stack(d: dict, t: float, dur: float, W: int) -> Image.Image:
    """Tool stack: app tiles (monograms) with name + role, popping in."""
    items = d.get("items", [])[:4]
    w = int(W * 0.88)
    h = 110 + 122 * max(1, len(items))
    im = _card(w, h, PINK)
    dr = ImageDraw.Draw(im)
    if d.get("title"):
        dr.text((40, 30), str(d["title"]).upper(), font=F(36), fill=(*PINK, 255))
    for i, it in enumerate(items):
        s = back_out((t - 0.15 - i * 0.3) / 0.35)
        if s <= 0.02:
            continue
        y = 96 + i * 122
        tile = _monogram(it.get("name", "?"), 96, _PALETTE[i % len(_PALETTE)])
        ts = max(1, int(96 * s))
        im.alpha_composite(tile.resize((ts, ts)), (40 + (96 - ts) // 2, y + (96 - ts) // 2))
        a = int(255 * min(1, s))
        dr.text((160, y + 6), str(it.get("name", "")), font=F(46), fill=(*WHITE, a))
        dr.text((160, y + 60), str(it.get("role", "")), font=F(30, "bold"), fill=(190, 196, 215, a))
        if it.get("price"):
            fp = F(34)
            pr = str(it["price"])
            dr.rounded_rectangle([w - 60 - fp.getlength(pr) - 24, y + 22, w - 40, y + 76], 16, fill=(*GREEN, int(a * 0.9)))
            dr.text((w - 48 - fp.getlength(pr) - 12, y + 28), pr, font=fp, fill=(10, 14, 20, a))
    return im


def r_meter(d: dict, t: float, dur: float, W: int) -> Image.Image:
    """Progress toward a threshold (e.g. 10,000 followers needed): bar + marker + labels."""
    w, h = int(W * 0.88), 300
    im = _card(w, h, ACCENT2)
    dr = ImageDraw.Draw(im)
    lab = str(d.get("label", ""))
    dr.text((40, 30), lab.upper(), font=F(36), fill=(*ACCENT2, 255))
    v, mx = float(d.get("value", 0)), float(d.get("max", 1) or 1)
    unit = str(d.get("unit", ""))
    p = ease_out((t - 0.15) / 1.0) * min(1.0, v / mx)
    x0, x1, y = 40, w - 40, 150
    dr.rounded_rectangle([x0, y, x1, y + 44], 22, fill=(40, 44, 66, 255))
    if p > 0.01:
        dr.rounded_rectangle([x0, y, x0 + (x1 - x0) * p, y + 44], 22, fill=(*GREEN, 255))
    cur = fmt_value(v * ease_out((t - 0.15) / 1.0), unit, ref=v, locale=d.get("locale", "en"))
    fm = F(56)
    dr.text((x0, 212), cur, font=fm, fill=WHITE)
    goal = fmt_value(mx, unit, locale=d.get("locale", "en"))
    gl = (d.get("goal_label") or "GOAL").upper()
    fg = F(30, "bold")
    dr.text((x1 - fg.getlength(f"{gl}: {goal}"), 228), f"{gl}: {goal}", font=fg, fill=(200, 206, 222))
    return im


def chips(items: list, t: float, W: int) -> Image.Image | None:
    """Row of small detail chips under the main card (extra facts / examples)."""
    items = [str(x)[:26] for x in (items or [])[:3] if str(x).strip()]
    if not items:
        return None
    f = F(30, "bold")
    ws = [f.getlength(x) + 44 for x in items]
    total = sum(ws) + 16 * (len(ws) - 1)
    if total > W * 0.92:
        f = F(24, "bold")
        ws = [f.getlength(x) + 36 for x in items]
        total = sum(ws) + 16 * (len(ws) - 1)
    im = Image.new("RGBA", (int(total) + 4, 60), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    x = 0
    for i, (txt, wd) in enumerate(zip(items, ws)):
        a = int(255 * ease_out((t - 0.6 - i * 0.2) / 0.3))
        if a > 4:
            col = _PALETTE[i % len(_PALETTE)]
            dr.rounded_rectangle([x, 4, x + wd, 56], 26, fill=(14, 16, 28, int(a * 0.85)), outline=(*col, a), width=3)
            dr.text((x + (wd - f.getlength(txt)) / 2, 14 if f.size > 26 else 17), txt, font=f, fill=(*WHITE, a))
        x += wd + 16
    return im


RENDER = {"headline": r_headline, "timeline": r_timeline, "bars": r_bars, "counter": r_counter, "quote": r_quote,
          "person": r_person, "checklist": r_checklist, "compare": r_compare, "steps": r_steps, "notify": r_notify,
          "calc": r_calc, "prompt": r_prompt, "stack": r_stack, "meter": r_meter}


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
                extra = 84 if self.ov.get("chips") else 0
                y = max(int(self.H * 0.085), min(y, int(self.H * 0.555) - im.height - extra))
                self._glass(frame, im, x, y, env, lt)
                im2 = _fade(self._shine(im, lt), env)
                self._comp(frame, im2, x, y)
                ch = chips(self.ov.get("chips") or [], lt, self.W)
                if ch is not None:
                    self._comp(frame, _fade(ch, env), (self.W - ch.width) // 2, y + im.height + 24)
        return frame
