"""Animated word-by-word subtitles (ASS / libass).

* 2-3 word groups, the spoken word pops (scale 118%) and turns accent colour
* emphasis words from the script are permanently coloured
* big kinetic "stat" cards and a hook title in the first seconds
"""
from __future__ import annotations

import re
from pathlib import Path

from .config import cfg


def _ts(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _ass_color(hex_rgb: str, alpha: int = 0) -> str:
    h = hex_rgb.lstrip("#")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def _clean(w: str) -> str:
    return re.sub(r"[^\wäöüÄÖÜß%°'’\-\.,!?$€]", "", w)


def _norm(w: str) -> str:
    return re.sub(r"[^a-z0-9äöüß]", "", w.lower())


def group_words(words: list[dict], max_words: int, max_chars: int = 16) -> list[list[dict]]:
    groups, cur = [], []
    for w in words:
        if cur and sum(len(x["text"]) + 1 for x in cur) + len(w["text"]) > max_chars:
            groups.append(cur)
            cur = []
        cur.append(w)
        end_punct = re.search(r"[.!?,;:—]$", w["text"])
        long_gap = False
        if len(cur) >= max_words or end_punct or long_gap:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def build_ass(words: list[dict], out: Path, emphasis: list[str], stats: list[dict] | None = None,
              hook_text: str | None = None, width: int = 1080, height: int = 1920, style: dict | None = None) -> Path:
    sc = cfg().get("subtitles", {})
    style = {**sc, **(style or {})}
    font = style.get("font", "Inter Display")
    size = int(style.get("size", 92))
    colors = style.get("highlight_colors", ["#FFD60A", "#38E1FF", "#FF4D6D"])
    active = _ass_color(colors[0])
    emph_col = _ass_color(colors[1])
    y = int(height * float(style.get("position_y", 0.66)))
    maxw = int(style.get("words_per_group", 3))
    emph = {_norm(e) for phrase in emphasis for e in phrase.split()}

    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Word,{font} Black,{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,1,0,1,7,4,5,60,60,0,1
Style: Hook,{font} Black,{int(size*1.05)},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,8,6,8,70,70,0,1
Style: Stat,{font} Black,{int(size*2.3)},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,5,40,40,0,1
Style: StatGlow,{font} Black,{int(size*2.3)},{_ass_color(colors[1])},&H00FFFFFF,{_ass_color(colors[1])},&H00000000,-1,0,0,0,100,100,0,0,1,6,0,5,40,40,0,1
Style: StatSub,{font} Bold,{int(size*0.62)},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,4,0,1,4,2,5,60,60,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev: list[str] = []
    groups = group_words(words, maxw)
    for gi, g in enumerate(groups):
        g_start, g_end = g[0]["start"], g[-1]["end"]
        nxt = groups[gi + 1][0]["start"] if gi + 1 < len(groups) else g_end + 1.0
        g_end = min(g_end + 0.06, nxt) - 0.06
        chars = sum(len(_clean(x["text"])) + 1 for x in g) - 1
        # Inter Black caps ~0.68em wide; leave room for the 118% active-word pop
        fit = min(100, int(100 * (width * 0.86) / max(1.0, chars * size * 0.68 + size * 0.5)))
        for i, w in enumerate(g):
            s = w["start"] if i else g_start
            e = g[i + 1]["start"] if i + 1 < len(g) else g_end + 0.06
            e = min(e, nxt)
            parts = []
            base = f"\\fscx{fit}\\fscy{fit}"
            for j, x in enumerate(g):
                txt = _clean(x["text"]).upper()
                if not txt:
                    continue
                if j == i:
                    big, mid = int(fit * 1.18), int(fit * 1.1)
                    parts.append("{\\c" + active + base + f"\\t(0,90,\\fscx{big}\\fscy{big})\\t(90,180,\\fscx{mid}\\fscy{mid})}}" + txt)
                elif _norm(x["text"]) in emph:
                    parts.append("{\\c" + emph_col + base + "}" + txt)
                else:
                    parts.append("{\\c&H00FFFFFF&" + base + "}" + txt)
            intro = f"\\fscx{int(fit*0.8)}\\fscy{int(fit*0.8)}" if i == 0 else ""
            ev.append(f"Dialogue: 2,{_ts(s)},{_ts(e)},Word,,0,0,0,,{{\\an5\\pos({width//2},{y})}}" + " ".join(parts))
    if hook_text:
        ev.append(f"Dialogue: 3,{_ts(0.05)},{_ts(2.6)},Hook,,0,0,0,,{{\\an8\\pos({width//2},{int(height*0.13)})\\fad(120,250)"
                  r"\fscx70\fscy70\t(0,180,\fscx104\fscy104)\t(180,300,\fscx100\fscy100)}" + hook_text.upper())
    for st in stats or []:
        s, e = st["start"], st["end"]
        pos = f"\\pos({width//2},{int(height*0.36)})"
        anim = r"\fad(80,200)\fscx40\fscy40\t(0,220,\fscx108\fscy108)\t(220,380,\fscx100\fscy100)"
        fit = min(100, int(100 * (width * 0.88) / max(1, len(st['text']) * size * 2.3 * 0.62)))
        anim = anim.replace("fscx40\\fscy40", f"fscx{int(fit*0.4)}\\fscy{int(fit*0.4)}").replace(
            "fscx108\\fscy108", f"fscx{int(fit*1.08)}\\fscy{int(fit*1.08)}").replace("fscx100\\fscy100", f"fscx{fit}\\fscy{fit}")
        bw, bh = int(width * 0.86), int(size * 2.6)
        box = f"m 0 0 l {bw} 0 l {bw} {bh} l 0 {bh}"
        ev.append(f"Dialogue: 3,{_ts(s)},{_ts(e)},Stat,,0,0,0,,{{\\an5\\pos({width//2},{int(height*0.36 + size*0.55)})"
                  f"\\fad(120,200)\\1c&H000000&\\1a&H70&\\bord0\\shad0\\blur40\\p1}}{box}{{\\p0}}")
        ev.append(f"Dialogue: 4,{_ts(s)},{_ts(e)},StatGlow,,0,0,0,,{{\\an5{pos}{anim}\\blur14}}{st['text']}")
        ev.append(f"Dialogue: 5,{_ts(s)},{_ts(e)},Stat,,0,0,0,,{{\\an5{pos}{anim}}}{st['text']}")
        if st.get("sub"):
            ev.append(f"Dialogue: 5,{_ts(s + 0.15)},{_ts(e)},StatSub,,0,0,0,,{{\\an5\\pos({width//2},{int(height*0.36 + size*1.7)})\\fad(150,200)}}{st['sub'].upper()}")
    out.write_text(head + "\n".join(ev) + "\n", "utf-8")
    return out
