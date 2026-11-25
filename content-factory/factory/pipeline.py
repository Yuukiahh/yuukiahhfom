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
from .render import CAMERA_MOVES, assemble, render_scenes, thumbnail
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
    if research_override and research_override.get("context"):
        res = research_override  # research already done (e.g. by the trend engine or a previous run)
    else:
        res = research.gather_auto(topic, lang)
    progress("script", 0.12)
    script, provider = scriptwriter.write(topic, niche, fmt, lang, res, learning.prompt_hints(niche))
    problems = scriptwriter.validate(script)
    if problems:
        script, provider = scriptwriter.write(topic + " (fix: " + "; ".join(problems) + ")", niche, fmt, lang, res,
                                              learning.prompt_hints(niche))
    progress("factcheck", 0.2)
    script, verdict = scriptwriter.fact_check(script, res)
    script["fact_check"] = verdict
    if cfg().get("overlays.enabled", True):
        progress("overlays", 0.24)
        try:
            script = scriptwriter.plan_overlays(script, res)
        except Exception as e:  # overlays are an enhancement - never block a video
            script["overlays_error"] = str(e)[:200]
    script["research_sources"] = res["sources"]
    script["llm_provider"] = provider
    script["topic"] = topic
    return script


def _shots(dur: float, first_move: str, rng: random.Random) -> list[tuple[float, str]]:
    """Split long scenes into 2-3 s shots with different camera moves (pattern interrupts)."""
    shots = [(0.0, first_move)]
    seg = float(cfg().get("render.shot_length", 2.6))
    t = seg
    while dur - t > 1.4:
        prev = shots[-1][1]
        shots.append((round(t, 3), rng.choice([m for m in CAMERA_MOVES if m != prev])))
        t += seg + rng.uniform(-0.4, 0.4)
    return shots


def produce(script: dict, job_id: int | None = None, progress: Progress = _noop, out_root: Path | None = None) -> dict:
    from . import mascot
    from .render import render_transitions
    from .transitions import SFX_FOR
    from .transitions import SETS as TSETS

    t_start = time.time()
    lang = script.get("language", cfg().get("general.language", "de"))
    preset = script.get("preset") or cfg().get("render.preset", "cinematic")
    W, H, FPS = cfg().get("render.width"), cfg().get("render.height"), cfg().get("render.fps", 60)
    date = dt.date.today().isoformat()
    slug = slugify(script.get("title") or script.get("topic", "video"))
    out_root = out_root or cfg().path("general.output_dir")
    folder = out_root / date / slug
    if folder.exists():
        folder = out_root / date / f"{slug}-{int(time.time()) % 10000}"
    work = folder / "_work"
    work.mkdir(parents=True, exist_ok=True)
    (folder / "script.json").write_text(json.dumps(script, ensure_ascii=False, indent=2), "utf-8")
    rng = random.Random(script.get("seed") or random.randint(0, 10 ** 9))
    user_images = _user_media(script.get("user_media"))

    # ---- voice
    progress("voice", 0.3)
    voice_wav = work / "voice.wav"
    narrations = [sc["narration"] for sc in script["scenes"]]
    words, spans, voice_info = tts.synthesize(narrations, voice_wav, lang, script.get("voice"))
    voice_dur = sf.info(voice_wav).duration
    # length guard: if the voice overshoots the target length, re-synthesise up to 15 % faster
    _lo, _hi = scriptwriter.fmt_bounds(script.get("format", "short"))
    if voice_dur + 1.0 > _hi + 1.5 and not script.get("no_speed_fit"):
        base_voice = script.get("voice") or ""
        parts = base_voice.split("|") if "|" in base_voice else []
        cur = float(parts[2]) if len(parts) > 2 else float(cfg().get("tts.speed", 1.0) if not isinstance(cfg().get("tts.speed"), dict) else 1.0)
        factor = min(1.15, (voice_dur + 1.0) / (_hi + 0.5))
        new_speed = round(cur * factor, 3)
        if parts:
            fitted = "|".join([parts[0], parts[1], str(new_speed)])
        else:
            fitted = None
            cfg().data.setdefault("tts", {})["speed"] = new_speed
        progress("voice_fit", 0.33)
        words, spans, voice_info = tts.synthesize(narrations, voice_wav, lang, fitted or script.get("voice"))
        voice_info["speed_fit"] = {"from": cur, "to": new_speed, "before_s": round(voice_dur, 2)}
        voice_dur = sf.info(voice_wav).duration
    total = voice_dur + 1.0  # outro tail (CTA)

    # ---- timeline: scene i spans [b_{i-1}-T/2, b_i+T/2]; transition centred on each cut b_i
    T = float(cfg().get("render.transition_seconds", 0.3))
    n = len(script["scenes"])
    cuts = [spans[i + 1][0] for i in range(n - 1)]
    progress("visuals", 0.4)
    seed0 = rng.randint(0, 10 ** 6)
    media_items, specs = [], []
    tkinds = (TSETS.get(preset) or TSETS["cinematic"])[:]
    rng.shuffle(tkinds)
    sfx_events: list[tuple[float, str, float]] = [(0.0, "impact", 0.85), (0.05, "braam", 0.45), (0.25, "sparkle", 0.35)]
    stats = []
    emph_norm = {re.sub(r"[^a-z0-9äöüß]", "", e.lower()) for sc in script["scenes"] for e in sc.get("emphasis", [])}
    for i, sc in enumerate(script["scenes"]):
        g0 = 0.0 if i == 0 else cuts[i - 1] - T / 2
        g1 = total if i == n - 1 else cuts[i] + T / 2
        dur = round(g1 - g0, 4)
        m = None
        if user_images and sc.get("visual") in ("holo_cards", "cards", "trading_cards", "tcg", "card_pack", "collectibles", "pack_opening"):
            sc = {**sc, "params": {**sc.get("params", {}), "images": user_images[i % len(user_images):] + user_images[:i % len(user_images)]}}
        if m is None and isinstance(sc.get("photo"), str) and Path(sc["photo"]).exists():  # explicit image for this scene
            m = {"path": sc["photo"], "type": "image", "source": "user",
                 "license": "User-provided image (own photo or own AI generation)"}
        elif m is None and user_images and (sc.get("photo") is True or (sc.get("photo") is None and i % 2 == 0)):
            # user photos: explicit "photo" scenes, otherwise every other non-card scene shows one full-frame
            k = sum(1 for j in range(i) if j % 2 == 0)
            m = {"path": user_images[k % len(user_images)], "type": "image", "source": "user",
                 "license": "User-provided image (own photo or own AI generation)"}
        if m is None and sc.get("visual_source", "auto") != "procedural":
            m = media_mod.fetch(sc, work / "media", script.get("niche", ""))
        media_items.append({**m, "scene": i} if m else {"scene": i, "source": "procedural-3d", "type": "generated",
                                                       "kind": sc["visual"], "license": "Original render (own work)"
                                                       + ("; card textures = user-provided photos" if sc.get("params", {}).get("images") else "")})
        is_photo = bool(m and m.get("type") == "image")
        bumps = [round(w["start"] - g0, 3) for w in words if w["scene"] == i
                 and re.sub(r"[^a-z0-9äöüß]", "", w["text"].lower()) in emph_norm]
        specs.append({"w": W, "h": H, "fps": FPS, "duration": dur, "kind": sc["visual"], "params": sc.get("params", {}),
                      "seed": seed0 + i, "camera": sc.get("camera", CAMERA_MOVES[i % len(CAMERA_MOVES)]),
                      "shots": _shots(dur, sc.get("camera", CAMERA_MOVES[i % len(CAMERA_MOVES)]), rng), "bumps": bumps,
                      "media": m, "out": str(work / f"scene_{i:02d}.mp4"), "shakes": [0.02] if i == 0 or sc.get("sfx") == "impact" else [],
                      "fade_in": i == 0, "time_offset": rng.uniform(0, 5),
                      "overlay": sc.get("overlay") if cfg().get("overlays.enabled", True) else None,
                      "overlay_start": round((0.0 if i == 0 else cuts[i - 1]) - g0 + (2.5 if i == 0 else 0.12) + (0.9 if is_photo else 0.0), 3),
                      "overlay_end": round(((total - 2.25) if i == n - 1 else cuts[i]) - g0 - 0.08, 3),  # clear of CTA card
                      "overlay_y": 0.205 if is_photo else float(cfg().get("overlays.y", 0.36)),
                      "dof": not is_photo,  # never blur the user's photos
                      "tag": script.get("tag") if cfg().get("overlays.enabled", True) else None,
                      "tag_from": max(0.0, 2.4 - g0), "lang": lang,
                      "source_label": sc.get("source_label") if cfg().get("overlays.enabled", True) else None})
        start = 0.0 if i == 0 else cuts[i - 1]
        end = total if i == n - 1 else cuts[i]
        if sc.get("overlay") and cfg().get("overlays.enabled", True):
            sfx_events.append((start + (2.5 if i == 0 else 0.12), "swoosh_reverse", 0.3))
        cue = sc.get("sfx", "none")
        if cue in ("riser",) and end - start > 2:
            sfx_events.append((end - 1.6, "riser", 0.5))
        elif cue in ("glitch", "ding", "pop", "impact", "heartbeat", "typing", "notification") and i > 0:
            sfx_events.append((start + 0.05, cue, 0.55))
        for w in words:
            if w["scene"] == i and re.sub(r"[^a-z0-9äöüß]", "", w["text"].lower()) in emph_norm:
                sfx_events.append((w["start"], "swish_text", 0.22))
        if sc.get("stat") and not (sc.get("overlay") and cfg().get("overlays.enabled", True)):
            st_words = [w for w in words if w["scene"] == i]
            st0 = st_words[0]["start"] if st_words else start
            stats.append({"text": sc["stat"]["text"], "sub": sc["stat"].get("sub", ""), "start": st0 + 0.08,
                          "end": min(end - 0.08, st0 + 2.6)})
            sfx_events += [(st0 + 0.08, "pop", 0.7), (st0 + 0.08, "sub_drop", 0.5)]
    tspecs, tnames = [], []
    for i, b in enumerate(cuts):
        k = tkinds[i % len(tkinds)]
        tnames.append(k)
        tspecs.append({"clip_a": specs[i]["out"], "clip_b": specs[i + 1]["out"], "kind": k, "dur": T, "fps": FPS,
                       "w": W, "h": H, "out": str(work / f"trans_{i:02d}.mp4"), "seed": seed0 + 100 + i})
        sfx_events.append((b - T / 2 - 0.08, SFX_FOR.get(k, "whoosh"), 0.6))
    cta_text = cfg().lang_value("brand.cta", lang) or ("Folge für mehr!" if lang == "de" else "Follow for more!")
    cta = {"text": cta_text, "start": max(0.0, total - 2.1), "end": total}
    sfx_events.append((cta["start"], "notification", 0.45))

    # ---- audio beds
    progress("music", 0.48)
    lib = audio.build_sfx_library(cfg().path("general.data_dir") / "sfx")
    music_wav, sfx_wav = work / "music.wav", work / "sfx.wav"
    mood = script.get("mood") or audio.NICHE_MOOD.get(script.get("niche", ""), "cosmic")
    music_info = f"Original procedural score ({mood})"
    track = getattr(audio, "pick_library_track", lambda *a: None)(mood, total)
    if track and hasattr(audio, "fit_track"):
        audio.fit_track(track, total + 0.5, music_wav)
        side = Path(track).with_suffix(".json")
        music_info = json.loads(side.read_text("utf-8")) if side.exists() else str(track)
    else:
        try:
            audio.music(total + 0.5, mood, music_wav, seed=rng.randint(0, 10 ** 6), markers=cuts)
        except TypeError:
            audio.music(total + 0.5, mood, music_wav)
    audio.sfx_track(sfx_events, total, lib, sfx_wav)

    # ---- captions + mascot
    emphasis = [e for sc in script["scenes"] for e in sc.get("emphasis", [])]
    theme = _theme(script.get("accent"))
    ass = subtitles.build_ass(words, work / "subs.ass", emphasis, stats, script.get("hook_text"), W, H, cta=cta,
                              style=theme.get("subs"))
    mascot_track, cues = None, []
    if cfg().get("mascot.enabled", True) and mascot.available():
        cues = mascot.plan_cues(script, words, spans, total, stats)
        mascot_track = mascot.render_track(cues, voice_wav, total, FPS, work / "mascot.mov",
                                           int(cfg().get("mascot.size", 440)))

    # ---- render
    progress("render_scenes", 0.55)
    clips = render_scenes(specs)
    progress("transitions", 0.8)
    trans_clips = render_transitions(tspecs)
    progress("assemble", 0.86)
    final = folder / f"{slug}.mp4"
    assemble(clips, [s["duration"] for s in specs], tnames, T, voice_wav, music_wav, sfx_wav, ass, final, total,
             trans_clips=trans_clips, mascot_track=mascot_track, progress_color=theme.get("progress", "#FFD60A"))
    (work / "render.json").write_text(json.dumps({"clips": clips, "durations": [s["duration"] for s in specs],
                                                   "transitions": tnames, "trans_dur": T, "total": total,
                                                   "trans_clips": trans_clips, "mascot_track": str(mascot_track or ""),
                                                   "stats": stats, "emphasis": emphasis, "cta": cta,
                                                   "shots": sum(len(s["shots"]) for s in specs)}), "utf-8")

    # ---- QC + packaging
    progress("qc", 0.93)
    report = qc.run(final, script, words, media_items, voice_dur,
                    extra={"shots": sum(len(s["shots"]) for s in specs), "mascot_cues": len(cues)})
    qc.contact_sheet(final, folder / "qc_contact_sheet.jpg")
    meta = metadata.build(script)
    thumb_at = stats[0]["start"] + 0.45 if stats else 1.3
    thumbnail(final, thumb_at, script.get("hook_text") or script["title"], folder / "thumbnail.jpg")
    (folder / "words.json").write_text(json.dumps(words), "utf-8")
    shutil.copy(ass, folder / "subtitles.ass")
    _write_srt(words, folder / "subtitles.srt")
    (folder / "qc.json").write_text(json.dumps(report, indent=2), "utf-8")
    (folder / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    (folder / "licenses.json").write_text(json.dumps({"media": media_items, "music": music_info,
                                                       "sfx": "Original synthesised SFX (assets/sfx, CC0 by owner)",
                                                       "voice": voice_info, "mascot": "Nova – original Blender render (own work)",
                                                       "font": "Inter (SIL Open Font License)"}, indent=2, ensure_ascii=False), "utf-8")
    (folder / "sources.md").write_text(_sources_md(script), "utf-8")
    if not cfg().get("general.keep_work_files", True):
        shutil.rmtree(work, ignore_errors=True)

    video_id = db.insert("videos", job_id=job_id, slug=slug, title=meta["title"], topic=script.get("topic", script["title"]),
                         niche=script.get("niche"), format=script.get("format"), language=lang, duration=report["duration"],
                         path=str(final), folder=str(folder), hook=script.get("hook"), qc_passed=int(report["passed"]),
                         qc=report, metadata=meta, fingerprint=dedupe.fingerprint(script.get("topic", script["title"])),
                         created_at=time.time())
    progress("done", 1.0)
    return {"video_id": video_id, "path": str(final), "folder": str(folder), "qc": report, "metadata": meta,
            "render_seconds": round(time.time() - t_start, 1), "voice": voice_info}


IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")


def _user_media(folder: str | None) -> list[str]:
    """Photos the user provides (e.g. their own card photos) - used as card textures / photo scenes."""
    if not folder:
        return []
    p = Path(folder)
    if not p.is_absolute():
        p = cfg().path("general.data_dir").parent / folder if not (Path.cwd() / folder).exists() else Path.cwd() / folder
    if not p.exists():
        return []
    return sorted(str(x) for x in p.iterdir() if x.suffix.lower() in IMG_EXT)


def _theme(accent: str | None) -> dict:
    """Derive caption pill / progress-bar colours from the topic accent (keeps white text readable)."""
    if not accent or not re.fullmatch(r"#[0-9a-fA-F]{6}", accent):
        return {}
    r, g, b = (int(accent[i:i + 2], 16) for i in (1, 3, 5))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    k = min(1.0, 105 / max(lum, 1))  # darken bright colours so white text stays readable on the pill
    pill = "#{:02X}{:02X}{:02X}".format(int(r * k), int(g * k), int(b * k))
    return {"subs": {"pill_color": pill}, "progress": accent}


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
    theme = _theme(script.get("accent"))
    ass = subtitles.build_ass(words, work / "subs.ass", r["emphasis"], r["stats"], script.get("hook_text"), W, H,
                              cta=r.get("cta"), style=theme.get("subs"))
    final = Path(v["path"])
    tmp = final.with_suffix(".new.mp4")
    mascot_track = Path(r["mascot_track"]) if r.get("mascot_track") else None
    from . import mascot
    if cfg().get("mascot.enabled", True) and mascot.available():
        # rebuild with the current mascot settings (size/mode); spans reconstructed from word timings
        n = len(script["scenes"])
        starts = [min([w["start"] for w in words if w["scene"] == i] or [0.0]) for i in range(n)]
        spans = [(starts[i], starts[i + 1] if i + 1 < n else r["total"]) for i in range(n)]
        cues = mascot.plan_cues(script, words, spans, r["total"], r["stats"])
        mascot_track = mascot.render_track(cues, work / "voice.wav", r["total"], cfg().get("render.fps", 60),
                                           work / "mascot.mov", int(cfg().get("mascot.size", 440)))
    elif not cfg().get("mascot.enabled", True):
        mascot_track = None
    assemble(r["clips"], r["durations"], r["transitions"], r["trans_dur"], work / "voice.wav", work / "music.wav",
             work / "sfx.wav", ass, tmp, r["total"], trans_clips=r.get("trans_clips"), mascot_track=mascot_track,
             progress_color=theme.get("progress", "#FFD60A"))
    tmp.replace(final)
    shutil.copy(ass, folder / "subtitles.ass")
    _write_srt(words, folder / "subtitles.srt")
    report = qc.run(final, script, words, json.loads((folder / "licenses.json").read_text("utf-8"))["media"],
                    sf.info(work / "voice.wav").duration,
                    extra={"shots": r.get("shots", 0), "mascot_cues": len(locals().get("cues") or [])})
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
