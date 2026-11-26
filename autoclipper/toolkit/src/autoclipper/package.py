"""Scoring, publishing metadata and the export folder on the Desktop.

Project layout (one folder per video):

    AI Clips/2026-11-26_<title>/
        01_<slug>.mp4 ...           finished clips, numbered in recommended posting order
        REPORT.md                   overview: scores, timestamps, titles, descriptions, hashtags, plan
        clips.json                  everything machine-readable
        posting_plan.csv            day/time/file/title/hashtags, for a scheduler or spreadsheet
        texts/01_tiktok.txt ...     ready to paste: description + hashtags
        texts/01_shorts.txt ...     ready to paste: Shorts title + description
        review/                     contact sheets and hook strips used during QA
        _rejected/                  clips that failed the quality bar (kept, not deleted)
        source.json                 source, licence, credit line
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

from .config import Settings

GENERIC_TAGS = {
    "de": ["#fyp", "#viral", "#foryou", "#shorts", "#deutsch"],
    "en": ["#fyp", "#viral", "#foryou", "#shorts"],
}
STOPWORDS = set("""
aber alle also auch auf aus bei bin bis bist das dass dem den der des die dir doch du ein eine einem einen
einer er es für hab habe hat hatte ich ihr im in ist ja jetzt kann man mal mehr mich mir mit nach nicht noch
nur oder schon sein sich sie sind so und uns von war was weil wenn wie wir wird zu zum zur über
the a an and or but if then so to of in on at for with is are was were be been it this that you i we they
he she my your our their what when where who how just like really very can will would should not no yes
""".split())


def slugify(text: str, max_len: int = 48) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return (text[:max_len].rstrip("-")) or "clip"


def hms(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    frac = int(round((seconds - int(seconds)) * 10)) % 10
    return f"{h:02d}:{m:02d}:{s:02d}.{frac}"


def parse_time(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    parts = str(value).split(":")
    try:
        total = 0.0
        for p in parts:
            total = total * 60 + float(p)
        return total
    except ValueError:
        return None


# -- scoring ------------------------------------------------------------------------------
def score_clip(clip: dict, qa_penalty: int = 0, review: dict | None = None) -> dict:
    """Combine SupoClip's AI sub-scores, Claude's review and QA into one 0-100 virality score.

    Criteria: hook, retention, emotion, entertainment, virality (shareability). SupoClip rates
    hook/engagement/value/shareability 0-25 each; the review adds emotion and entertainment 0-100
    and may correct the others.
    """
    review = review or {}
    sub = {
        "hook": clip.get("hook_score", 0) * 4,
        "retention": clip.get("engagement_score", 0) * 4,
        "value": clip.get("value_score", 0) * 4,
        "virality": clip.get("shareability_score", 0) * 4,
    }
    has_sub = any(sub.values())
    for key in ("hook", "retention", "virality"):
        if key in review:
            sub[key] = float(review[key])
    sub["emotion"] = float(review.get("emotion", sub["value"]))
    sub["entertainment"] = float(review.get("entertainment", sub["value"]))
    if has_sub or review:
        base = (0.25 * sub["hook"] + 0.20 * sub["retention"] + 0.20 * sub["emotion"]
                + 0.15 * sub["entertainment"] + 0.20 * sub["virality"])
    else:
        base = float(clip.get("virality_score") or 0)
    final = max(0, min(100, round(base - qa_penalty)))
    return {"final": final, "base": round(base, 1), "qa_penalty": qa_penalty,
            "supoclip_virality": clip.get("virality_score"), **{k: round(v) for k, v in sub.items()}}


# -- metadata -----------------------------------------------------------------------------
def heuristic_metadata(clip: dict, language: str = "de") -> dict:
    text = (clip.get("text") or "").strip()
    hook = (clip.get("hook_title") or "").strip()
    first = re.split(r"(?<=[.!?])\s+", text)[0] if text else ""
    title = hook or first[:70] or "Clip"
    words = [w.strip(".,!?\"'():;").lower() for w in text.split()]
    common = [w for w, _ in Counter(w for w in words if len(w) > 4 and w not in STOPWORDS).most_common(4)]
    tags = [f"#{slugify(w, 24).replace('-', '')}" for w in common if slugify(w, 24)]
    tags += GENERIC_TAGS.get(language, GENERIC_TAGS["en"])
    return {
        "shorts_title": title[:95],
        "tiktok_description": (first[:140] + " 👀") if first else title,
        "hashtags": list(dict.fromkeys(tags))[:8],
        "source": "heuristic",
    }


def ollama_metadata(clip: dict, settings: Settings, language: str = "de") -> dict | None:
    """Local LLM metadata (free). Returns None when Ollama is not reachable."""
    lang = "German" if language.startswith("de") else "the language of the transcript"
    prompt = (
        f"You write short-form social media copy in {lang}. Clip transcript:\n\"\"\"{(clip.get('text') or '')[:3000]}\"\"\"\n"
        "Return JSON with keys: shorts_title (max 70 chars, curiosity hook, no clickbait lies), "
        "tiktok_description (1-2 sentences, max 150 chars, ends with a question or call to action), "
        "hashtags (5-8 items, each starting with #, mix of specific and broad)."
    )
    try:
        resp = httpx.post(f"{settings.ollama_url}/api/chat", timeout=180, json={
            "model": settings.ollama_model, "stream": False, "format": "json",
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0.6},
        })
        resp.raise_for_status()
        data = json.loads(resp.json()["message"]["content"])
        tags = [t if t.startswith("#") else f"#{t}" for t in data.get("hashtags", []) if isinstance(t, str)]
        return {"shorts_title": str(data["shorts_title"])[:95],
                "tiktok_description": str(data["tiktok_description"])[:2000],
                "hashtags": tags[:10], "source": f"ollama:{settings.ollama_model}"}
    except Exception:
        return None


# -- posting plan -------------------------------------------------------------------------
def posting_order(items: list[dict]) -> list[dict]:
    """Strongest clip first; then avoid two consecutive clips from neighbouring source moments.

    Neighbouring moments usually share a topic, and back-to-back posts on one topic cannibalise
    each other. Ties keep the score order.
    """
    remaining = sorted(items, key=lambda i: i["score"]["final"], reverse=True)
    ordered: list[dict] = []
    while remaining:
        pick = remaining[0]
        if ordered:
            prev = ordered[-1]
            for cand in remaining:
                close = abs((cand.get("source_start") or 0) - (prev.get("source_start") or 0)) < 90
                if not close:
                    pick = cand if cand["score"]["final"] >= remaining[0]["score"]["final"] - 12 else remaining[0]
                    break
        ordered.append(pick)
        remaining.remove(pick)
    return ordered


def schedule(n: int, start: dt.date | None = None) -> list[str]:
    """One clip per day at 18:00 on weekdays and 12:00 on weekends (when short-form reach peaks)."""
    day = start or (dt.date.today() + dt.timedelta(days=1))
    slots = []
    for _ in range(n):
        time = "12:00" if day.weekday() >= 5 else "18:00"
        slots.append(f"{day.isoformat()} {time}")
        day += dt.timedelta(days=1)
    return slots


# -- export -------------------------------------------------------------------------------
def project_dir(settings: Settings, title: str, when: dt.date | None = None) -> Path:
    when = when or dt.date.today()
    root = Path(settings.output_root)
    base = root / f"{when.isoformat()}_{slugify(title, 40)}"
    path, n = base, 2
    while path.exists() and any(path.glob("*.mp4")):
        path = base.with_name(f"{base.name}-{n}")
        n += 1
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_project(project: Path, items: list[dict], source: dict, *, rejected: list[dict] | None = None,
                  language: str = "de") -> dict:
    """Rename clips into posting order and write REPORT.md, clips.json, posting_plan.csv and texts/."""
    ordered = posting_order(items)
    slots = schedule(len(ordered))
    texts = project / "texts"
    texts.mkdir(exist_ok=True)
    for old in texts.glob("*.txt"):
        old.unlink()

    staged = []
    for rank, (item, slot) in enumerate(zip(ordered, slots), start=1):
        src = Path(item["file"])
        name = f"{rank:02d}_{slugify(item['meta']['shorts_title'], 40)}.mp4"
        tmp = src.with_name(f".staging_{rank:02d}_{src.name}")
        if src.exists():
            src.replace(tmp)
        staged.append((item, slot, rank, tmp, project / name))
    for item, slot, rank, tmp, dest in staged:
        if tmp.exists():
            tmp.replace(dest)
        item.update(rank=rank, file=str(dest), filename=dest.name, post_at=slot)
        tags = " ".join(item["meta"]["hashtags"])
        credit = f"\n\n{source['credit_line']}" if source.get("credit_line") else ""
        (texts / f"{rank:02d}_tiktok.txt").write_text(
            f"{item['meta']['tiktok_description']}\n\n{tags}{credit}\n", encoding="utf-8")
        (texts / f"{rank:02d}_shorts.txt").write_text(
            f"TITEL:\n{item['meta']['shorts_title']} #shorts\n\nBESCHREIBUNG:\n"
            f"{item['meta']['tiktok_description']}\n\n{tags}{credit}\n", encoding="utf-8")

    with (project / "posting_plan.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["Reihenfolge", "Veröffentlichen am", "Datei", "Virality Score", "Shorts-Titel",
                    "TikTok-Beschreibung", "Hashtags", "Original Start", "Original Ende"])
        for it in ordered:
            w.writerow([it["rank"], it["post_at"], it["filename"], it["score"]["final"],
                        it["meta"]["shorts_title"], it["meta"]["tiktok_description"],
                        " ".join(it["meta"]["hashtags"]), hms(it.get("source_start")), hms(it.get("source_end"))])

    summary = {"source": source, "created": dt.datetime.now().isoformat(timespec="seconds"),
               "clips": ordered, "rejected": rejected or []}
    (project / "clips.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (project / "source.json").write_text(json.dumps(source, indent=2, ensure_ascii=False), encoding="utf-8")
    (project / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
    return summary


def render_report(summary: dict) -> str:
    src = summary["source"]
    clips = summary["clips"]
    lines = [
        f"# AI Clips – {src.get('title') or 'Projekt'}",
        "",
        f"- Quelle: {src.get('webpage_url') or src.get('source')}",
        f"- Kanal: {src.get('channel') or '–'} · Lizenz/Rechte: {src.get('license') or src.get('rights')}",
        f"- Erstellt: {summary['created']} · Clips: {len(clips)} · Aussortiert: {len(summary['rejected'])}",
    ]
    if src.get("credit_line"):
        lines.append(f"- **Pflicht-Credit (CC BY):** {src['credit_line']}")
    lines += ["", "## Veröffentlichungsreihenfolge", "",
              "| # | Wann | Score | Titel | Original | Länge | QA |",
              "|---|------|------:|-------|----------|------:|----|"]
    for c in clips:
        lines.append(f"| {c['rank']} | {c['post_at']} | **{c['score']['final']}** | {c['meta']['shorts_title']} | "
                     f"{hms(c.get('source_start'))}–{hms(c.get('source_end'))} | {c.get('duration', 0):.0f}s | "
                     f"{c.get('qa', {}).get('status', '?')} |")
    lines += ["", "Strategie: stärkster Clip zuerst (er bestimmt, wie viel Reichweite die Folgeclips bekommen), "
              "danach abwechselnd Themen statt zwei Clips aus derselben Videostelle hintereinander; "
              "ein Clip pro Tag, werktags 18:00, am Wochenende 12:00. Freigabe und Upload macht nur der Kanalinhaber.",
              ""]
    for c in clips:
        s = c["score"]
        lines += [
            f"## {c['rank']:02d} · {c['meta']['shorts_title']}",
            "",
            f"- Datei: `{c['filename']}`",
            f"- **Virality Score: {s['final']}/100** (Hook {s.get('hook')}, Retention {s.get('retention')}, "
            f"Emotion {s.get('emotion')}, Unterhaltung {s.get('entertainment')}, Viralität {s.get('virality')}, "
            f"QA-Abzug {s.get('qa_penalty')})",
            f"- Original-Zeitstempel: {hms(c.get('source_start'))} – {hms(c.get('source_end'))}",
            f"- YouTube-Shorts-Titel: {c['meta']['shorts_title']}",
            f"- TikTok-Beschreibung: {c['meta']['tiktok_description']}",
            f"- Hashtags: {' '.join(c['meta']['hashtags'])}",
        ]
        if c.get("why"):
            lines.append(f"- Warum dieser Moment: {c['why']}")
        issues = [i for i in c.get("qa", {}).get("issues", [])]
        if issues:
            lines.append("- Offene QA-Hinweise: " + "; ".join(f"{i['code']} ({i['detail']})" for i in issues))
        lines.append("")
    if summary["rejected"]:
        lines += ["## Aussortiert (in `_rejected/`)", ""]
        for r in summary["rejected"]:
            lines.append(f"- {r.get('filename')}: {r.get('reason')}")
    return "\n".join(lines) + "\n"


def move_rejected(project: Path, file: Path) -> Path:
    dest_dir = project / "_rejected"
    dest_dir.mkdir(exist_ok=True)
    dest = dest_dir / file.name
    shutil.move(str(file), dest)
    return dest
