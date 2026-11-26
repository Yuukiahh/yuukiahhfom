"""End-to-end run: rights -> SupoClip (transcribe, select, edit) -> QA/repair -> score -> export.

``process`` does everything deterministic. ``finalize`` applies Claude's editorial review
(better titles/descriptions/hashtags, emotion/entertainment ratings, drops) and rewrites the
project. Both are idempotent and keep their state in <project>/_work/state.json.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

from . import fix, package, qa, review
from .config import Settings
from .rights import SourceInfo, inspect_source
from .supoclip_api import SupoClip, SupoClipAPIError


class RightsConfirmationNeeded(RuntimeError):
    def __init__(self, info: SourceInfo):
        self.info = info
        super().__init__(
            f'Rechte unklar: "{info.title}" von {info.channel} ({info.license or "Standardlizenz"}). '
            "Bitte bestätigen, dass du dieses Video nutzen darfst (eigener Kanal, Erlaubnis oder Lizenz)."
        )


def log(msg: str) -> None:
    print(f"[autoclipper] {msg}", file=sys.stderr, flush=True)


def build_task_body(settings: Settings, url: str, title: str, api: SupoClip | None) -> dict:
    t = settings.task
    caption = t.caption_template
    intensity = t.intensity
    edit_style = dict(t.edit_style)
    if api is not None:
        fx_on = False
        try:
            fx_on = bool(api.edit_styles().get("enabled"))
        except SupoClipAPIError:
            fx_on = False
        if not fx_on:
            log("dynamic edits are not enabled for this user: using classic captions without effects")
            intensity, edit_style, caption = "off", {}, t.fallback_caption_template
        else:
            try:
                templates = api.caption_templates()
                if templates and caption not in templates:
                    caption = t.fallback_caption_template
            except SupoClipAPIError:
                pass
        if not settings.pexels_configured:
            try:
                settings.task.include_broll = bool(api.broll_status().get("configured"))
            except SupoClipAPIError:
                pass

    body: dict[str, Any] = {
        "source": {"url": url[:2000], **({"title": title[:300]} if title else {})},
        "processing_mode": t.processing_mode,
        "clip_length": t.clip_length,
        "output_format": t.output_format,
        "add_subtitles": t.add_subtitles,
        "caption_template": caption,
        "include_broll": t.include_broll,
        "cut_long_pauses": t.cut_long_pauses,
        "remove_filler_words": t.remove_filler_words,
    }
    if t.workflow != "highlights":
        body["workflow"] = t.workflow
    if t.instructions:
        body["instructions"] = t.instructions[:500]
    if intensity != "off":
        body["edit_style"] = {"intensity": intensity, **edit_style}
    return body


def _state_path(project: Path) -> Path:
    return project / "_work" / "state.json"


def _save_state(project: Path, state: dict) -> None:
    path = _state_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def load_state(project: Path) -> dict:
    return json.loads(_state_path(project).read_text(encoding="utf-8"))


def quality_pass(file: Path, settings: Settings, review_dir: Path, *, max_rounds: int = 2) -> dict:
    """QA a clip, repair what is fixable, QA again; write contact sheet and hook strip."""
    report = qa.check(file, target_lufs=settings.target_lufs, true_peak=settings.true_peak)
    rounds = 0
    while report.fixable_codes and rounds < max_rounds:
        rounds += 1
        log(f"{file.name}: repairing {sorted(report.fixable_codes)}")
        fix.repair(file, report, target_lufs=settings.target_lufs, true_peak=settings.true_peak,
                   encoder=settings.encoder)
        report = qa.check(file, target_lufs=settings.target_lufs, true_peak=settings.true_peak)
    result = report.to_dict()
    result["repair_rounds"] = rounds
    try:
        result["contact_sheet"] = str(review.contact_sheet(file, review_dir / f"{file.stem}_sheet.png"))
        result["hook_strip"] = str(review.hook_strip(file, review_dir / f"{file.stem}_hook.png"))
    except Exception as exc:  # review images are a help, not a requirement
        result["review_error"] = str(exc)
    backup = file.with_suffix(".orig.mp4")
    if backup.exists():
        work = file.parent / "_work" / "originals"
        work.mkdir(parents=True, exist_ok=True)
        shutil.move(str(backup), work / backup.name)
    return result


def _select(items: list[dict], settings: Settings) -> tuple[list[dict], list[dict]]:
    """Keep clips that pass QA and the score bar; keep at least ``min_clips`` non-failing ones."""
    usable = [i for i in items if i["qa"]["status"] != "fail" and not i.get("drop")]
    hard_rejects = [dict(i, reason=i.get("drop_reason") or "QA fehlgeschlagen: " + ", ".join(
        x["code"] for x in i["qa"]["issues"] if x["severity"] == "fail")) for i in items if i not in usable]
    usable.sort(key=lambda i: i["score"]["final"], reverse=True)
    keep = [i for i in usable if i["score"]["final"] >= settings.min_virality]
    if len(keep) < settings.min_clips:
        keep = usable[: max(settings.min_clips, len(keep))]
    keep = keep[: settings.max_clips]
    low = [dict(i, reason=f"Virality Score {i['score']['final']} unter Schwelle {settings.min_virality}")
           for i in usable if i not in keep]
    return keep, hard_rejects + low


def _apply_meta(item: dict, meta: dict | None, settings: Settings, language: str, use_ollama: bool) -> None:
    """Merge one review entry into a clip. Every finalize is a delta: keys not given keep their
    previous value (so a later review cannot silently undo titles, ratings or drops)."""
    clip = item["clip"]
    meta = meta or {}
    if "review" in meta:
        item["review"] = {**item.get("review", {}), **(meta.get("review") or {})}
    item["score"] = package.score_clip(clip, item["qa"].get("penalty", 0), item.get("review"))
    current = item.get("meta") or {}
    if meta.get("shorts_title") or meta.get("tiktok_description") or meta.get("hashtags"):
        item["meta"] = {
            "shorts_title": meta.get("shorts_title") or current.get("shorts_title") or "Clip",
            "tiktok_description": meta.get("tiktok_description") or current.get("tiktok_description", ""),
            "hashtags": [h if h.startswith("#") else f"#{h}"
                         for h in (meta.get("hashtags") or current.get("hashtags") or [])],
            "source": "claude-review",
        }
    elif not current:
        item["meta"] = ((package.ollama_metadata(clip, settings, language) if use_ollama else None)
                        or package.heuristic_metadata(clip, language))
    if "why" in meta:
        item["why"] = meta["why"]
    if "drop" in meta:
        item["drop"] = bool(meta["drop"])
        item["drop_reason"] = meta.get("drop_reason", "") if item["drop"] else ""


def _export(project: Path, state: dict, settings: Settings) -> dict:
    items = state["items"]
    # Bring back anything an earlier pass rejected, so each export decides afresh.
    for item in items:
        f = Path(item["file"])
        if not f.exists() and (project / "_rejected" / f.name).exists():
            (project / "_rejected" / f.name).replace(f)
    keep, rejected = _select(items, settings)
    for r in rejected:
        f = Path(r["file"])
        if f.exists():
            dest = package.move_rejected(project, f)
            r["filename"] = dest.name
            for item in items:
                if item["clip"]["id"] == r["clip"]["id"]:
                    item["file"] = str(dest.parent.parent / dest.name)  # path it is restored to
    summary = package.write_project(project, keep, state["source"],
                                    rejected=[{"filename": r.get("filename") or Path(r["file"]).name,
                                               "reason": r["reason"], "clip_id": r["clip"]["id"]} for r in rejected],
                                    language=state.get("language", "de"))
    _save_state(project, state)
    return {"project": str(project), "report": str(project / "REPORT.md"),
            "clips": [{"rank": c["rank"], "file": c["file"], "score": c["score"]["final"],
                       "title": c["meta"]["shorts_title"], "qa": c["qa"]["status"],
                       "contact_sheet": c["qa"].get("contact_sheet"), "hook_strip": c["qa"].get("hook_strip"),
                       "source": f"{package.hms(c.get('source_start'))}-{package.hms(c.get('source_end'))}",
                       "text": (c["clip"].get("text") or "")[:1200]}
                      for c in summary["clips"]],
            "rejected": summary["rejected"]}


def process(source: str, settings: Settings, *, confirmed: bool = False, title: str = "",
            use_ollama: bool = True) -> dict:
    info = inspect_source(source, settings, confirmed=confirmed)
    if not info.allowed:
        raise RightsConfirmationNeeded(info)
    if info.is_live:
        raise RuntimeError("Livestreams, die noch laufen, können nicht geclippt werden.")
    log(f"source ok ({info.rights}): {info.title or info.source}")

    api = SupoClip(settings)
    api.health()
    project = package.project_dir(settings, title or info.title or "video")
    log(f"project folder: {project}")

    url = info.source
    if info.kind == "local":
        log("uploading local video to SupoClip ...")
        url = api.upload(Path(info.source))

    body = build_task_body(settings, url, title or info.title, api)
    task = api.create_task(body)
    task_id = task.get("task_id") or task.get("id")
    log(f"SupoClip task {task_id} started")
    state: dict[str, Any] = {"task_id": task_id, "request": body, "source": info.to_dict(),
                             "language": info.language or "de", "items": []}
    _save_state(project, state)

    last_msg = [""]

    def progress(t: dict) -> None:
        msg = f"{t.get('status')} {t.get('progress', 0)}% {t.get('progress_message', '')}".strip()
        if msg != last_msg[0]:
            log(msg)
            last_msg[0] = msg

    done = api.wait(task_id, on_progress=progress)
    if done.get("status") != "completed":
        raise RuntimeError(f"SupoClip task {task_id} ended with {done.get('status')}: "
                           f"{done.get('error_message') or done.get('error_code') or done.get('progress_message')}")
    return collect(project, settings, use_ollama=use_ollama)


def collect(project: Path, settings: Settings, *, use_ollama: bool = True) -> dict:
    """Download, QA, score and export the clips of the task recorded in the project state."""
    state = load_state(project)
    api = SupoClip(settings)
    clips = api.list_clips(state["task_id"])
    if not clips:
        raise RuntimeError("SupoClip hat keine Clips erzeugt (zu kurzes oder zu stilles Video?).")
    review_dir = project / "review"
    items = []
    for n, clip in enumerate(clips, start=1):
        dest = project / f"clip_{n:02d}_{clip['id'][:8]}.mp4"
        if not dest.exists():
            log(f"downloading clip {n}/{len(clips)}")
            api.download_clip(state["task_id"], clip["id"], dest)
        qa_result = quality_pass(dest, settings, review_dir)
        source_start = package.parse_time(clip.get("start_time"))
        source_end = package.parse_time(clip.get("end_time"))
        items.append({"clip": clip, "file": str(dest), "qa": qa_result,
                      "source_start": source_start, "source_end": source_end,
                      "duration": qa_result.get("duration", 0), "why": clip.get("reasoning") or ""})
    for item in items:
        _apply_meta(item, None, settings, state.get("language", "de"), use_ollama)
    state["items"] = items
    return _export(project, state, settings)


def finalize(project: Path, meta_file: Path, settings: Settings) -> dict:
    """Apply a review file: {"<clip_id>": {shorts_title, tiktok_description, hashtags, why,
    review: {hook, retention, emotion, entertainment, virality}, drop, drop_reason}}."""
    state = load_state(project)
    meta = json.loads(Path(meta_file).read_text(encoding="utf-8-sig"))
    for item in state["items"]:
        cid = item["clip"]["id"]
        entry = meta.get(cid) or meta.get(cid[:8])
        if entry is None:
            # Also accept the current file name or rank as key.
            entry = meta.get(Path(item["file"]).name) or meta.get(str(item.get("rank", "")))
        _apply_meta(item, entry, settings, state.get("language", "de"), use_ollama=False)
    return _export(project, state, settings)
