"""Fetch REAL performance metrics from the official platform APIs into `metrics`.

Never fabricates numbers: a value the API did not return is stored as NULL, and a
platform without connection / scope is skipped with a note. After a refresh the
learning engine (`learning.analyse`) is re-run.
"""
from __future__ import annotations

import time
from collections import defaultdict

from . import db, learning
from .publish import PublishError, redact

METRIC_FIELDS = ("views", "likes", "comments", "shares", "avg_view_duration", "avg_view_pct", "subscribers_gained")


def _uploaded(platform: str) -> list[dict]:
    return db.rows("SELECT p.*, v.created_at AS video_created FROM publications p JOIN videos v ON v.id=p.video_id "
                   "WHERE p.platform=? AND p.status='uploaded' AND p.remote_id IS NOT NULL", platform)


def _store(video_id: int, platform: str, m: dict) -> bool:
    vals = {k: m.get(k) for k in METRIC_FIELDS}
    if all(v is None for v in vals.values()):
        return False
    db.insert("metrics", video_id=video_id, platform=platform, fetched_at=time.time(), **vals)
    return True


def merge_youtube(stats: dict | None, report: dict | None) -> dict:
    """Data API counts are near-realtime; Analytics adds retention/shares/subs
    (lagging 1-3 days). Prefer live counts, fill the rest from the report."""
    out = {k: None for k in METRIC_FIELDS}
    for src in (report or {}, stats or {}):  # stats last -> wins for views/likes/comments
        for k, v in src.items():
            if k in out and v is not None:
                out[k] = v
    return out


def _refresh_youtube() -> dict:
    from .publish import youtube
    pubs = _uploaded("youtube")
    if not pubs:
        return {"updated": 0, "note": "keine hochgeladenen Videos"}
    if not youtube.is_connected():
        return {"updated": 0, "skipped": len(pubs), "note": "YouTube nicht verbunden – übersprungen"}
    notes, n = [], 0
    stats = youtube.video_statistics([p["remote_id"] for p in pubs])
    for p in pubs:
        report = None
        try:
            start = time.strftime("%Y-%m-%d", time.localtime(p.get("published_at") or p["video_created"]))
            report = youtube.video_analytics(p["remote_id"], start)
        except PublishError as e:
            notes.append(f"{p['remote_id']}: Analytics nicht verfügbar ({str(e)[:120]})")
        if _store(p["video_id"], "youtube", merge_youtube(stats.get(p["remote_id"]), report)):
            n += 1
    return {"updated": n, "notes": notes}


def _refresh_tiktok() -> dict:
    from .publish import tiktok
    pubs = [p for p in _uploaded("tiktok") if str(p["remote_id"]).isdigit()]  # only real post ids
    if not pubs:
        return {"updated": 0, "note": "keine öffentlichen TikTok-Post-IDs (private Posts liefern keine ID)"}
    if not tiktok.is_connected():
        return {"updated": 0, "skipped": len(pubs), "note": "TikTok nicht verbunden – übersprungen"}
    if "video.list" not in tiktok.granted_scopes():
        return {"updated": 0, "skipped": len(pubs), "note": "Scope video.list fehlt – übersprungen"}
    data = tiktok.video_metrics([str(p["remote_id"]) for p in pubs])
    n = sum(1 for p in pubs if data.get(str(p["remote_id"])) and _store(p["video_id"], "tiktok",
                                                                         data[str(p["remote_id"])]))
    return {"updated": n}


def _refresh_instagram() -> dict:
    from .publish import instagram
    pubs = _uploaded("instagram")
    if not pubs:
        return {"updated": 0, "note": "keine hochgeladenen Reels"}
    if not instagram.is_connected():
        return {"updated": 0, "skipped": len(pubs), "note": "Instagram nicht verbunden – übersprungen"}
    n, notes = 0, []
    for p in pubs:
        try:
            if _store(p["video_id"], "instagram", instagram.media_insights(str(p["remote_id"]))):
                n += 1
        except PublishError as e:
            notes.append(f"{p['remote_id']}: {str(e)[:120]}")
    return {"updated": n, "notes": notes}


def refresh() -> dict:
    out: dict = {}
    for name, fn in (("youtube", _refresh_youtube), ("tiktok", _refresh_tiktok), ("instagram", _refresh_instagram)):
        try:
            out[name] = fn()
        except Exception as e:  # noqa: BLE001 - one platform failing must not stop the others
            msg = redact(f"{type(e).__name__}: {e}")[:300]
            out[name] = {"updated": 0, "error": msg}
            db.log_event(f"[analytics] {name}: {msg}", "error")
    out["learning"] = learning.analyse()
    return out


def latest_per_video() -> list[dict]:
    return db.rows("""
        SELECT m.*, v.title, v.niche, v.format, v.slug
        FROM metrics m JOIN videos v ON v.id = m.video_id
        WHERE m.id = (SELECT id FROM metrics m2 WHERE m2.video_id = m.video_id AND m2.platform = m.platform
                      ORDER BY fetched_at DESC, id DESC LIMIT 1)
        ORDER BY m.video_id, m.platform""")


def summary() -> dict:
    latest = latest_per_video()
    totals = {k: 0 for k in ("views", "likes", "comments", "shares", "subscribers_gained")}
    for m in latest:
        for k in totals:
            totals[k] += m.get(k) or 0
    niches: dict[str, dict] = defaultdict(lambda: {"videos": set(), "views": 0, "likes": 0, "pct": []})
    for m in latest:
        n = niches[m.get("niche") or "unknown"]
        n["videos"].add(m["video_id"])
        n["views"] += m.get("views") or 0
        n["likes"] += m.get("likes") or 0
        if m.get("avg_view_pct") is not None:
            n["pct"].append(m["avg_view_pct"])
    per_niche = {k: {"videos": len(v["videos"]), "views": v["views"], "likes": v["likes"],
                     "avg_view_pct": round(sum(v["pct"]) / len(v["pct"]), 2) if v["pct"] else None}
                 for k, v in niches.items()}
    return {"totals": totals, "videos_with_metrics": len({m["video_id"] for m in latest}),
            "per_video": latest, "per_niche": per_niche,
            "learnings": db.kv_get("learnings", {}) or {}}
