"""Animated captions v2 (ASS / libass), built for retention on TikTok / Reels / Shorts.

* every word is laid out individually with real font metrics (bundled Inter Display Black)
* the spoken word gets a rounded highlight "pill" + pop/bounce, spoken/upcoming words stay white
* emphasis words are colour-highlighted and shake when spoken
* groups pop in word by word (staggered), auto-fit to the safe zone (avoids TikTok/Reels UI)
* hook title (first seconds), big kinetic stat cards with backdrop, end CTA card
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from .config import ROOT, cfg

FONT_DIR = ROOT / "assets" / "fonts"
FONT_FILE = FONT_DIR / "InterDisplay-Black.otf"
FONT_NAME = "Inter Display Black"


def _ts(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def _ass_color(hex_rgb: str, alpha: int = 0) -> str:
    h = hex_rgb.lstrip("#")
    return f"&H{alpha:02X}{h[4:6]}{h[2:4]}{h[0:2]}&".upper()


def _clean(w: str) -> str:
    return re.sub(r"[^\wäöüÄÖÜß%°'’\-.,!?$€×~²³⁰¹⁴⁵⁶⁷⁸⁹/:]", "", w)


def _norm(w: str) -> str:
    return re.sub(r"[^a-z0-9äöüß]", "", w.lower())


@lru_cache(maxsize=8)
def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.truetype(str(FONT_FILE), size)
    except OSError:
        return ImageFont.load_default()


def text_width(text: str, size: int) -> float:
    return _font(size).getlength(text)


def group_words(words: list[dict], max_words: int, max_chars: int = 16) -> list[list[dict]]:
    groups, cur = [], []
    for w in words:
        if not _clean(w["text"]):
            continue
        if cur and (sum(len(x["text"]) + 1 for x in cur) + len(w["text"]) > max_chars
                    or w["start"] - cur[-1]["end"] > 0.45 or cur[-1].get("scene") != w.get("scene")):
            groups.append(cur)
            cur = []
        cur.append(w)
        if len(cur) >= max_words or re.search(r"[.!?,;:—]$", w["text"]):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def _pill(w: float, h: float, r: float) -> str:
    """Rounded rectangle as ASS vector drawing, origin top-left."""
    k = 0.55 * r
    return (f"m {r:.0f} 0 l {w - r:.0f} 0 b {w - r + k:.0f} 0 {w:.0f} {r - k:.0f} {w:.0f} {r:.0f} l {w:.0f} {h - r:.0f} "
            f"b {w:.0f} {h - r + k:.0f} {w - r + k:.0f} {h:.0f} {w - r:.0f} {h:.0f} l {r:.0f} {h:.0f} "
            f"b {r - k:.0f} {h:.0f} 0 {h - r + k:.0f} 0 {h - r:.0f} l 0 {r:.0f} b 0 {r - k:.0f} {r - k:.0f} 0 {r:.0f} 0")


def build_ass(words: list[dict], out: Path, emphasis: list[str], stats: list[dict] | None = None,
              hook_text: str | None = None, width: int = 1080, height: int = 1920, style: dict | None = None,
              cta: dict | None = None) -> Path:
    sc = cfg().get("subtitles", {}) or {}
    st = {**sc, **(style or {})}
    size = int(st.get("size", 96))
    colors = st.get("highlight_colors", ["#FFD60A", "#38E1FF", "#FF4D6D"])
    pill_col = st.get("pill_color", "#7C3AED")
    emph_col = _ass_color(colors[0])
    y = int(height * float(st.get("position_y", 0.6)))
    maxw = int(st.get("words_per_group", 3))
    max_line = width * 0.84
    emph = {_norm(e) for phrase in emphasis for e in phrase.split()}
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Word,{FONT_NAME},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H8C000000,0,0,0,0,100,100,0,0,1,7,5,5,0,0,0,1
Style: Pill,{FONT_NAME},{size},{_ass_color(pill_col)},&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
Style: Hook,{FONT_NAME},{int(size * 1.02)},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,8,6,8,70,70,0,1
Style: Stat,{FONT_NAME},{int(size * 2.3)},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,5,40,40,0,1
Style: StatGlow,{FONT_NAME},{int(size * 2.3)},{_ass_color(colors[1])},&H00FFFFFF,{_ass_color(colors[1])},&H00000000,0,0,0,0,100,100,0,0,1,6,0,5,40,40,0,1
Style: StatSub,{FONT_NAME},{int(size * 0.6)},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,4,0,1,4,2,5,60,60,0,1
Style: Box,{FONT_NAME},10,&H00000000,&H00000000,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev: list[str] = []
    groups = group_words(words, maxw)
    space = text_width(" ", size) * 0.8
    for gi, g in enumerate(groups):
        texts = [_clean(w["text"]).upper() for w in g]
        widths = [text_width(t, size) for t in texts]
        total = sum(widths) + space * (len(g) - 1)
        fit = min(1.0, max_line / max(1.0, total * 1.06))
        x = width / 2 - total * fit / 2
        xs = []
        for wd in widths:
            xs.append(x + wd * fit / 2)
            x += (wd + space) * fit
        g_start = g[0]["start"]
        nxt = groups[gi + 1][0]["start"] if gi + 1 < len(groups) else g[-1]["end"] + 0.8
        g_end = min(g[-1]["end"] + 0.25, nxt) - 0.02
        F = int(fit * 100)
        for j, (w, t, cx, wd) in enumerate(zip(g, texts, xs, widths)):
            is_emph = _norm(w["text"]) in emph
            base_col = emph_col if is_emph else "&H00FFFFFF&"
            appear = int(j * 45)
            ev.append(f"Dialogue: 2,{_ts(g_start)},{_ts(g_end)},Word,,0,0,0,,{{\\an5\\pos({cx:.0f},{y})\\c{base_col}"
                      f"\\fscx{int(F * 0.6)}\\fscy{int(F * 0.6)}\\alpha&HFF&\\t({appear},{appear + 1},\\alpha&H00&)"
                      f"\\t({appear},{appear + 120},\\fscx{int(F * 1.08)}\\fscy{int(F * 1.08)})"
                      f"\\t({appear + 120},{appear + 200},\\fscx{F}\\fscy{F})}}{t}")
            s0 = w["start"]
            s1 = min(g[j + 1]["start"] if j + 1 < len(g) else g_end, g_end)
            if s1 - s0 < 0.04:
                continue
            pad_x, pad_y = size * 0.22, size * 0.16
            pw, ph = (wd * fit) + 2 * pad_x, size * 0.95 * fit + 2 * pad_y
            px, py = cx - pw / 2, y - ph / 2 - size * 0.02
            pill_c = _ass_color(colors[2]) if is_emph else _ass_color(pill_col)
            ev.append(f"Dialogue: 3,{_ts(s0)},{_ts(s1)},Pill,,0,0,0,,{{\\an5\\pos({cx:.0f},{y - size * 0.02:.0f})\\1c{pill_c}\\bord0\\shad0"
                      f"\\blur1\\fscx85\\fscy85\\t(0,90,\\fscx108\\fscy108)\\t(90,160,\\fscx100\\fscy100)"
                      f"\\p1}}{_pill(pw, ph, ph * 0.3)}{{\\p0}}")
            big, mid = int(F * (1.24 if is_emph else 1.14)), int(F * (1.1 if is_emph else 1.04))
            shake = "\\t(0,60,\\frz-4)\\t(60,120,\\frz3)\\t(120,170,\\frz0)" if is_emph else ""
            ev.append(f"Dialogue: 4,{_ts(s0)},{_ts(s1)},Word,,0,0,0,,{{\\an5\\pos({cx:.0f},{y})\\c&H00FFFFFF&\\bord5"
                      f"\\fscx{F}\\fscy{F}\\t(0,80,\\fscx{big}\\fscy{big})\\t(80,170,\\fscx{mid}\\fscy{mid}){shake}}}{t}")
    if hook_text:
        hw = text_width(hook_text.upper(), int(size * 1.02))
        hf = min(100, int(100 * width * 0.86 / max(1, hw)))
        ev.append(f"Dialogue: 6,{_ts(0.0)},{_ts(2.4)},Hook,,0,0,0,,{{\\an8\\pos({width // 2},{int(height * 0.14)})\\fad(60,220)"
                  f"\\fscx{int(hf * 0.6)}\\fscy{int(hf * 0.6)}\\t(0,150,\\fscx{int(hf * 1.06)}\\fscy{int(hf * 1.06)})"
                  f"\\t(150,260,\\fscx{hf}\\fscy{hf})}}{hook_text.upper()}")
    for stt in stats or []:
        s, e = stt["start"], stt["end"]
        sy = int(height * 0.34)
        pos = f"\\pos({width // 2},{sy})"
        fit = min(100, int(100 * (width * 0.86) / max(1, text_width(stt["text"], int(size * 2.3)))))
        anim = (f"\\fad(80,180)\\fscx{int(fit * 0.4)}\\fscy{int(fit * 0.4)}\\t(0,200,\\fscx{int(fit * 1.08)}\\fscy{int(fit * 1.08)})"
                f"\\t(200,340,\\fscx{fit}\\fscy{fit})")
        bw, bh = int(width * 0.88), int(size * 3.1)
        ev.append(f"Dialogue: 1,{_ts(s)},{_ts(e)},Box,,0,0,0,,{{\\an5\\pos({width // 2},{sy + int(size * 0.2)})"
                  f"\\fad(120,200)\\1c&H000000&\\1a&H68&\\bord0\\shad0\\blur36\\p1}}{_pill(bw, bh, 60)}{{\\p0}}")
        ev.append(f"Dialogue: 5,{_ts(s)},{_ts(e)},StatGlow,,0,0,0,,{{\\an5{pos}{anim}\\blur14}}{stt['text']}")
        ev.append(f"Dialogue: 6,{_ts(s)},{_ts(e)},Stat,,0,0,0,,{{\\an5{pos}{anim}}}{stt['text']}")
        if stt.get("sub"):
            ev.append(f"Dialogue: 6,{_ts(s + 0.15)},{_ts(e)},StatSub,,0,0,0,,{{\\an5\\pos({width // 2},{int(sy + size * 1.25)})"
                      f"\\fad(150,200)}}{stt['sub'].upper()}")
    if cta:
        s, e = cta["start"], cta["end"]
        cy = int(height * 0.22)
        txt = cta["text"].upper()
        cf = min(100, int(100 * width * 0.72 / max(1, text_width(txt, int(size * 1.02)))))
        bw, bh = int(width * 0.8), int(size * 1.5)
        ev.append(f"Dialogue: 6,{_ts(s)},{_ts(e)},Pill,,0,0,0,,{{\\an5\\pos({width // 2},{cy})\\1c{_ass_color(pill_col)}"
                  f"\\bord0\\shad0\\fad(150,150)\\fscx40\\fscy40\\t(0,220,\\fscx104\\fscy104)\\t(220,320,\\fscx100\\fscy100)"
                  f"\\p1}}{_pill(bw, bh, bh / 2)}{{\\p0}}")
        ev.append(f"Dialogue: 7,{_ts(s + 0.1)},{_ts(e)},Hook,,0,0,0,,{{\\an5\\pos({width // 2},{cy})\\bord0\\shad0\\fad(150,150)"
                  f"\\fscx{cf}\\fscy{cf}}}{txt}")
    out.write_text(head + "\n".join(ev) + "\n", "utf-8")
    return out


def ass_filter_arg(ass_file: Path) -> str:
    def esc(p):
        return str(p).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    return f"ass='{esc(ass_file)}':fontsdir='{esc(FONT_DIR)}'"
