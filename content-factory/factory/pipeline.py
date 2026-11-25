"""End-to-end production of one video.

topic -> research -> script + storyboard -> fact-check -> TTS (+word timings)
-> visuals (stock / AI / procedural) -> music + SFX -> scene renders -> assembly
-> QC -> metadata + thumbnail -> output/<date>/<slug>/
"""
from __future__ import annotations

import datetime as dt
import json
import random
import re
import shutil
import time
from pathlib import Path
from typing import Callable

import soundfile as sf

from . import audio, db, dedupe, metadata, qc, research, scriptwriter, subtitles, tts
from .config import cfg
from .render import CAMERA_MOVES, XFADE_SETS, assemble, render_scenes, thumbnail
from .visuals import media as media_mod

Progress = Callable[[str, float], None]


def slugify(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9äöüÄÖÜß]+", "-", s).strip("-").lower()
    return s[:60] or "video"


def _noop(stage: str, p: float) -> None:
    pass


def prepare_script(topic: str, niche: str, fmt: str, lang: str, progress: Progress = _noop,
                   research_override: dict | None = None) -> dict:
    from . import learning

    progress("research", 0.05)
    res = research.gather(topic, lang)
    if research_override and (res.get("offline") or not res.get("context")):
        res = research_override
    elif research_override:
        res["context"] = research_override["context"] + "\n\n" + res["context"]
        res["sources"] = research_override["sources"] + res["sources"]
    progress("script", 0.12)
    script, provider = scriptwriter.write(topic, niche, fmt, lang, res, learning.prompt_hints(niche))
    problems = scriptwriter.validate(script)
    if problems:
        script, provider = scriptwriter.write(topic + " (fix: " + "; ".join(problems) + ")", niche, fmt, lang, res,
                                              learning.prompt_hints(niche))
    progress("factcheck", 0.2)
    script, verdict = scriptwriter.fact_check(script, res)
    script["fact_check"] = verdict
    script["research_sources"] = res["sources"]
    script["llm_provider"] = provider
    script["topic"] = topic
    return script


def produce(script: dict, job_id: int | None = None, progress: Progress = _noop, out_root: Path | None = None) -> dict:
    t_start = time.time()
    lang = script.get("language", cfg().get("general.language", "en"))
    preset = script.get("preset") or cfg().get("render.preset", "cinematic")
    W, H, FPS = cfg().get("render.width"), cfg().get("render.height"), cfg().get("render.fps")
    date = dt.date.today().isoformat()
    slug = slugify(script.get("title") or script.get("topic", "video"))
    out_root = out_root or cfg().path("general.output_dir")
    folder = out_root / date / slug
    if folder.exists():
        folder = out_root / date / f"{slug}-{int(time.time()) % 10000}"
    work = folder / "_work"
    work.mkdir(parents=True, exist_ok=True)
    (folder / "script.json").write_text(json.dumps(script, ensure_ascii=False, indent=2), "utf-8")

    # ---- voice
    progress("voice", 0.3)
    voice_wav = work / "voice.wav"
    narrations = [sc["narration"] for sc in script["scenes"]]
    words, spans, voice_info = tts.synthesize(narrations, voice_wav, lang, script.get("voice"))
    voice_dur = sf.info(voice_wav).duration
    lead_in = 0.0
    total = voice_dur + 0.9  # outro tail

    # ---- visuals
    progress("visuals", 0.4)
    seed0 = random.randint(0, 10 ** 6)
    media_items, specs = [], []
    trans_dur = 0.4 if preset != "documentary" else 0.6
    transitions = XFADE_SETS.get(preset, XFADE_SETS["cinematic"])[:]
    random.shuffle(transitions)
    sfx_events: list[tuple[float, str, float]] = []
    stats = []
    n = len(script["scenes"])
    for i, sc in enumerate(script["scenes"]):
        s0, s1 = spans[i]
        start = 0.0 if i == 0 else s0
        end = total if i == n - 1 else spans[i + 1][0]
        dur = end - start + (trans_dur if i < n - 1 else 0)
        m = None
        if sc.get("visual_source", "auto") != "procedural":
            m = media_mod.fetch(sc, work / "media", script.get("niche", ""))
        if m:
            media_items.append({**m, "scene": i})
        else:
            media_items.append({"scene": i, "source": "procedural", "type": "generated", "kind": sc["visual"],
                                "license": "Original procedural render (own work)"})
        shakes = [0.05] if sc.get("sfx") == "impact" else []
        specs.append({"w": W, "h": H, "fps": FPS, "duration": round(dur, 3), "kind": sc["visual"],
                      "params": sc.get("params", {}), "seed": seed0 + i, "camera": sc.get("camera", CAMERA_MOVES[i % 8]),
                      "media": m, "out": str(work / f"scene_{i:02d}.mp4"), "shakes": shakes, "fade_in": i == 0,
                      "intensity": 1.0, "time_offset": random.uniform(0, 5)})
        # SFX cues
        cue = sc.get("sfx", "none")
        if i > 0:
            sfx_events.append((start - 0.25, "whoosh", 0.55))
        if cue in ("impact", "riser", "pop", "glitch", "ding"):
            at = start + (0.0 if cue != "riser" else max(0, (end - start) - 1.5))
            sfx_events.append((at, cue, 0.8 if cue == "impact" else 0.6))
        if sc.get("stat"):
            st_words = [w for w in words if w["scene"] == i]
            st0 = st_words[0]["start"] if st_words else start
            stats.append({"text": sc["stat"]["text"], "sub": sc["stat"].get("sub", ""), "start": st0 + 0.1,
                          "end": min(end - 0.1, st0 + 2.6)})
            sfx_events.append((st0 + 0.1, "pop", 0.7))

    # ---- audio beds
    progress("music", 0.48)
    lib = audio.build_sfx_library(cfg().path("general.data_dir") / "sfx")
    music_wav, sfx_wav = work / "music.wav", work / "sfx.wav"
    mood = script.get("mood") or audio.NICHE_MOOD.get(script.get("niche", ""), "cosmic")
    audio.music(total + 0.5, mood, music_wav)
    audio.sfx_track(sfx_events, total, lib, sfx_wav)

    # ---- subtitles
    emphasis = [e for sc in script["scenes"] for e in sc.get("emphasis", [])]
    ass = subtitles.build_ass(words, work / "subs.ass", emphasis, stats, script.get("hook_text"), W, H)

    # ---- render
    progress("render_scenes", 0.55)
    clips = render_scenes(specs)
    progress("assemble", 0.85)
    final = folder / f"{slug}.mp4"
    assemble(clips, [s["duration"] for s in specs], transitions, trans_dur, voice_wav, music_wav, sfx_wav, ass, final, total)
    (work / "render.json").write_text(json.dumps({"clips": clips, "durations": [s["duration"] for s in specs],
                                                   "transitions": transitions, "trans_dur": trans_dur, "total": total,
                                                   "stats": stats, "emphasis": emphasis}), "utf-8")

    # ---- QC + packaging
    progress("qc", 0.93)
    report = qc.run(final, script, words, media_items, voice_dur)
    qc.contact_sheet(final, folder / "qc_contact_sheet.jpg")
    meta = metadata.build(script)
    thumb_at = stats[0]["start"] + 0.4 if stats else 1.2
    thumbnail(final, thumb_at, script.get("hook_text") or script["title"], folder / "thumbnail.jpg")
    (folder / "words.json").write_text(json.dumps(words), "utf-8")
    shutil.copy(ass, folder / "subtitles.ass")
    _write_srt(words, folder / "subtitles.srt")
    (folder / "qc.json").write_text(json.dumps(report, indent=2), "utf-8")
    (folder / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    (folder / "licenses.json").write_text(json.dumps({"media": media_items, "music": f"Original procedural score ({mood})",
                                                       "sfx": "Original synthesised SFX", "voice": voice_info,
                                                       "font": "Inter (SIL Open Font License)"}, indent=2), "utf-8")
    (folder / "sources.md").write_text(_sources_md(script), "utf-8")
    if not cfg().get("general.keep_work_files", True):  # needed for subtitle re-render; pruned by worker
        shutil.rmtree(work, ignore_errors=True)

    video_id = db.insert("videos", job_id=job_id, slug=slug, title=meta["title"], topic=script.get("topic", script["title"]),
                         niche=script.get("niche"), format=script.get("format"), language=lang, duration=report["duration"],
                         path=str(final), folder=str(folder), hook=script.get("hook"), qc_passed=int(report["passed"]),
                         qc=report, metadata=meta, fingerprint=dedupe.fingerprint(script.get("topic", script["title"])),
                         created_at=time.time())
    progress("done", 1.0)
    return {"video_id": video_id, "path": str(final), "folder": str(folder), "qc": report, "metadata": meta,
            "render_seconds": round(time.time() - t_start, 1)}


def _write_srt(words: list[dict], out: Path) -> None:
    def ts(t):
        ms = int(round(t * 1000))
        return f"{ms//3600000:02d}:{ms//60000%60:02d}:{ms//1000%60:02d},{ms%1000:03d}"
    lines, idx = [], 1
    for g in subtitles.group_words(words, 4):
        lines += [str(idx), f"{ts(g[0]['start'])} --> {ts(g[-1]['end'])}", " ".join(w["text"] for w in g), ""]
        idx += 1
    out.write_text("\n".join(lines), "utf-8")


def _sources_md(script: dict) -> str:
    out = [f"# Quellen & Faktencheck – {script.get('title')}", ""]
    for c in script.get("claims", []):
        out.append(f"- **{c.get('claim')}** — {c.get('source')}")
    if script.get("research_sources"):
        out += ["", "## Recherche-Quellen"] + [f"- [{s['title']}]({s['url']})" for s in script["research_sources"]]
    fc = script.get("fact_check", {})
    out += ["", f"## Faktencheck: {fc.get('verdict', 'n/a')}"]
    for i in fc.get("issues", []):
        out.append(f"- {i}")
    return "\n".join(out) + "\n"


def rerender_subtitles(video_id: int) -> dict:
    """Re-burn subtitles after edits in the dashboard subtitle editor (words.json)."""
    v = db.row("SELECT * FROM videos WHERE id=?", video_id)
    if not v:
        raise ValueError("video not found")
    folder = Path(v["folder"])
    work = folder / "_work"
    rj = work / "render.json"
    if not rj.exists():
        raise FileNotFoundError("Arbeitsdateien wurden bereits aufgeräumt – Re-Render nicht mehr möglich")
    r = json.loads(rj.read_text("utf-8"))
    script = json.loads((folder / "script.json").read_text("utf-8"))
    words = json.loads((folder / "words.json").read_text("utf-8"))
    W, H = cfg().get("render.width"), cfg().get("render.height")
    ass = subtitles.build_ass(words, work / "subs.ass", r["emphasis"], r["stats"], script.get("hook_text"), W, H)
    final = Path(v["path"])
    tmp = final.with_suffix(".new.mp4")
    assemble(r["clips"], r["durations"], r["transitions"], r["trans_dur"], work / "voice.wav", work / "music.wav",
             work / "sfx.wav", ass, tmp, r["total"])
    tmp.replace(final)
    shutil.copy(ass, folder / "subtitles.ass")
    _write_srt(words, folder / "subtitles.srt")
    report = qc.run(final, script, words, json.loads((folder / "licenses.json").read_text("utf-8"))["media"],
                    sf.info(work / "voice.wav").duration)
    (folder / "qc.json").write_text(json.dumps(report, indent=2), "utf-8")
    db.update("videos", video_id, qc=report, qc_passed=int(report["passed"]))
    return {"ok": True, "qc": report}


def prune_work_dirs(days: int = 7) -> int:
    n = 0
    cutoff = time.time() - days * 86400
    for w in cfg().path("general.output_dir").glob("*/*/_work"):
        if w.stat().st_mtime < cutoff:
            shutil.rmtree(w, ignore_errors=True)
            n += 1
    return n
