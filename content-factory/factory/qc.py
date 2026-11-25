"""Automatic quality control of a finished video."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from .config import cfg
from .scriptwriter import FORMATS, fmt_bounds


def _probe(path: Path) -> dict:
    p = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                       capture_output=True, text=True)
    return json.loads(p.stdout)


def _filter_log(path: Path, af: str | None = None, vf: str | None = None) -> str:
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path)]
    if af:
        cmd += ["-af", af, "-vn"]
    if vf:
        cmd += ["-vf", vf, "-an"]
    cmd += ["-f", "null", "-"]
    return subprocess.run(cmd, capture_output=True, text=True).stderr


def contact_sheet(path: Path, out: Path, n: int = 8) -> Path:
    dur = float(_probe(path)["format"]["duration"])
    tiles = []
    for i in range(n):
        t = dur * (i + 0.5) / n
        p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(path), "-frames:v", "1",
                            "-vf", "scale=270:480", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
        if len(p.stdout) == 270 * 480 * 3:
            tiles.append(Image.frombytes("RGB", (270, 480), p.stdout))
    sheet = Image.new("RGB", (270 * 4, 480 * ((len(tiles) + 3) // 4)))
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i % 4) * 270, (i // 4) * 480))
    sheet.save(out, quality=85)
    return out


def run(video: Path, script: dict, words: list[dict], media: list[dict], voice_dur: float, extra: dict | None = None) -> dict:
    checks: list[dict] = []

    def add(name, ok, detail="", severity="error"):
        checks.append({"check": name, "ok": bool(ok), "detail": detail, "severity": severity})

    info = _probe(video)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    dur = float(info["format"]["duration"])
    W, H = cfg().get("render.width"), cfg().get("render.height")
    add("resolution", (v["width"], v["height"]) == (W, H), f"{v['width']}x{v['height']}")
    fr = v.get("avg_frame_rate", "0/1").split("/")
    fps = float(fr[0]) / max(1.0, float(fr[1]))
    add("fps", abs(fps - cfg().get("render.fps", 30)) < 0.5, f"{fps:.2f}")
    add("codec", v["codec_name"] == "h264" and v.get("pix_fmt") == "yuv420p", f"{v['codec_name']}/{v.get('pix_fmt')}")
    add("audio", a is not None and a["codec_name"] == "aac" and int(a.get("sample_rate", 0)) == 48000,
        f"{a and a['codec_name']} {a and a.get('sample_rate')}")
    lo, hi = fmt_bounds(script.get("format", "short"))
    add("duration", lo - 3 <= dur <= hi + 5, f"{dur:.1f}s (target {lo}-{hi})")
    br = int(info["format"].get("bit_rate", 0))
    add("bitrate", br > 4_000_000, f"{br/1e6:.1f} Mbit/s", "warn")

    # loudness
    log = _filter_log(video, af="ebur128=peak=true")
    m_i = re.findall(r"I:\s+(-?[\d.]+) LUFS", log)
    m_tp = re.findall(r"Peak:\s+(-?[\d.]+) dBFS", log)
    lufs = float(m_i[-1]) if m_i else -99
    tp = float(m_tp[-1]) if m_tp else 0
    target = cfg().get("render.loudness_lufs", -14)
    add("loudness", abs(lufs - target) <= 2.0, f"{lufs:.1f} LUFS (target {target})")
    add("true_peak", tp <= -0.5, f"{tp:.1f} dBTP")

    # silence / black / freeze
    sil = re.findall(r"silence_duration: ([\d.]+)", _filter_log(video, af="silencedetect=n=-45dB:d=1.0"))
    add("no_dead_air", not sil, f"{len(sil)} silent gaps >1s", "error")
    blk = re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", _filter_log(video, vf="blackdetect=d=0.6:pix_th=0.06"))
    blk = [(float(s), float(e)) for s, e in blk if float(s) > 0.5]
    add("no_black_frames", not blk, str(blk[:3]))
    frz = re.findall(r"freeze_duration: ([\d.]+)", _filter_log(video, vf="freezedetect=n=0.0008:d=4"))
    add("no_frozen_video", not frz, f"{len(frz)} frozen segments >4s", "warn")

    # subtitles vs. narration
    n_script = sum(len(sc["narration"].split()) for sc in script["scenes"])
    add("subtitle_words", len(words) == n_script, f"{len(words)} subtitle words vs {n_script} script words")
    covered = sum(w["end"] - w["start"] for w in words)
    add("subtitle_coverage", covered >= 0.6 * voice_dur, f"{covered:.1f}s of {voice_dur:.1f}s speech")
    mono = all(words[i]["start"] <= words[i + 1]["start"] for i in range(len(words) - 1))
    add("subtitle_monotonic", mono)
    ass_path = Path(video).parent / "_work" / "subs.ass"
    if not ass_path.exists():
        ass_path = Path(video).parent / "subtitles.ass"
    if ass_path.exists():
        def _t(x):
            h, m, sec = x.split(":")
            return int(h) * 3600 + int(m) * 60 + float(sec)
        evs = sorted((_t(l.split(",")[1]), _t(l.split(",")[2])) for l in ass_path.read_text("utf-8").splitlines()
                     if l.startswith("Dialogue: 4,"))
        overlaps = sum(1 for a, b in zip(evs, evs[1:]) if b[0] < a[1] - 0.005)
        add("subtitle_no_overlap", overlaps == 0, f"{overlaps} overlapping caption events")

    # script / facts / rights
    hook = script.get("hook", "")
    add("hook_length", 0 < len(hook.split()) <= 14, f"{len(hook.split())} words: {hook}")
    first_word = words[0]["start"] if words else 99
    add("hook_timing", first_word < 0.6, f"first word at {first_word:.2f}s")
    claims = script.get("claims", [])
    srcs = {c.get("source") for c in claims if c.get("source")}
    add("facts_sourced", len(claims) >= 2 and all(c.get("source") for c in claims), f"{len(claims)} claims, {len(srcs)} sources")
    fc = script.get("fact_check", {})
    add("fact_check", fc.get("verdict") in ("pass", "fix", "manual"), json.dumps(fc)[:200], "warn")
    lic_ok = all(m.get("license") for m in media)
    add("licenses", lic_ok, f"{len(media)} media items, all with licence" if lic_ok else "missing licence info")

    # visual sanity: frames not near-black, not identical
    sheet_vals = []
    for i in range(6):
        t = dur * (i + 0.5) / 6
        p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1",
                            "-vf", "scale=108:192", "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True)
        if p.stdout:
            sheet_vals.append(np.frombuffer(p.stdout, np.uint8).astype(np.float32))
    means = [x.mean() for x in sheet_vals]
    add("visual_brightness", all(m > 12 for m in means), "mean luma " + ", ".join(f"{m:.0f}" for m in means))
    diffs = [np.abs(sheet_vals[i] - sheet_vals[i + 1]).mean() for i in range(len(sheet_vals) - 1)]
    add("visual_variety", all(d > 4 for d in diffs), "frame diffs " + ", ".join(f"{d:.0f}" for d in diffs), "warn")

    title = (script.get("metadata", {}) or {}).get("title", "") or script.get("title", "")
    nums = re.findall(r"(?:Nr\.|#|No\.|Platz)\s?(\d+)", title)
    narr = " ".join(sc["narration"] for sc in script["scenes"])
    fake = [n for n in nums if not re.search(rf"\b{n}\b", narr)]
    add("title_honest", not fake, f"Titel verspricht Nr. {', '.join(fake)}, die im Video nicht vorkommt" if fake else "ok", "warn")
    extra = extra or {}
    if extra.get("shots"):
        n_cuts = extra["shots"]
        asl = dur / max(1, n_cuts)
        add("pacing", asl <= 3.2, f"Ø Einstellungslänge {asl:.1f}s ({n_cuts} Einstellungen) – Ziel ≤ 3,2 s", "warn")
    if cfg().get("mascot.enabled", True):
        add("mascot", extra.get("mascot_cues", 0) >= 2, f"{extra.get('mascot_cues', 0)} Maskottchen-Auftritte", "warn")

    errors = [c for c in checks if not c["ok"] and c["severity"] == "error"]
    warns = [c for c in checks if not c["ok"] and c["severity"] == "warn"]
    score = round(100 * sum(c["ok"] for c in checks) / len(checks))
    return {"passed": not errors, "score": score, "errors": len(errors), "warnings": len(warns),
            "duration": dur, "loudness_lufs": lufs, "checks": checks}
