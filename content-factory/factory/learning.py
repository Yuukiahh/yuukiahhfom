"""Self-improvement engine.

Uses only REAL metrics stored in the `metrics` table (fetched from the platform
analytics APIs). With too little data it says so and changes nothing.

Learns:
  * niche weights         -> trend scorer prefers niches that retain viewers
  * format split          -> short vs long share in the daily plan
  * hook patterns         -> question / number / "you" / negation hooks
  * best publish hours    -> publishing slots
and writes a compact "learnings" text injected into the scriptwriter prompt.
"""
from __future__ import annotations

import re
import statistics
import time
from collections import defaultdict

from . import db

MIN_VIDEOS = 5


def _latest_metrics() -> list[dict]:
    return db.rows("""
        SELECT v.id, v.niche, v.format, v.hook, v.duration, v.created_at, p.published_at,
               m.views, m.likes, m.comments, m.shares, m.avg_view_pct, m.avg_view_duration, m.subscribers_gained
        FROM videos v
        JOIN metrics m ON m.id = (SELECT id FROM metrics WHERE video_id = v.id ORDER BY fetched_at DESC LIMIT 1)
        LEFT JOIN publications p ON p.video_id = v.id AND p.status = 'uploaded'
    """)


def performance_score(m: dict) -> float:
    views = m.get("views") or 0
    if views <= 0:
        return 0.0
    eng = ((m.get("likes") or 0) + 3 * (m.get("comments") or 0) + 5 * (m.get("shares") or 0)) / views
    ret = (m.get("avg_view_pct") or 0) / 100
    subs = (m.get("subscribers_gained") or 0) / views * 100
    import math
    return round(math.log10(views + 1) * 0.35 + ret * 2.0 + eng * 10 + subs * 0.5, 4)


def hook_features(hook: str) -> list[str]:
    f = []
    h = (hook or "").lower()
    if "?" in h:
        f.append("question")
    if re.search(r"\d", h):
        f.append("number")
    if re.search(r"\byou\b|\bdu\b", h):
        f.append("second_person")
    if re.search(r"n't|never|nobody|nie|kein", h):
        f.append("negation")
    if len(h.split()) <= 8:
        f.append("very_short")
    return f or ["plain"]


def analyse() -> dict:
    data = _latest_metrics()
    if len(data) < MIN_VIDEOS:
        res = {"status": "insufficient_data", "videos_with_metrics": len(data), "needed": MIN_VIDEOS,
               "updated_at": time.time()}
        db.kv_set("learnings", res)
        return res
    for d in data:
        d["score"] = performance_score(d)
    overall = statistics.mean(d["score"] for d in data) or 1e-6

    def group(key):
        g = defaultdict(list)
        for d in data:
            ks = key(d)
            for k in (ks if isinstance(ks, list) else [ks]):
                g[k].append(d["score"])
        return {k: {"n": len(v), "lift": round(statistics.mean(v) / overall, 3)} for k, v in g.items()}

    niches = group(lambda d: d["niche"] or "unknown")
    formats = group(lambda d: d["format"] or "short")
    hooks = group(lambda d: hook_features(d["hook"]))
    hours = group(lambda d: time.localtime(d["published_at"]).tm_hour if d.get("published_at") else -1)
    hours.pop(-1, None)
    # niche weights with shrinkage toward 1.0 for small samples
    weights = {k: round(1 + (v["lift"] - 1) * v["n"] / (v["n"] + 5), 3) for k, v in niches.items()}
    long_lift = formats.get("long", {}).get("lift", 1)
    short_lift = formats.get("short", {}).get("lift", 1)
    long_share = min(0.5, max(0.15, 0.3 * long_lift / max(1e-6, short_lift)))
    best_hours = sorted(((v["lift"], k) for k, v in hours.items() if v["n"] >= 2), reverse=True)[:3]
    top = sorted(data, key=lambda d: d["score"], reverse=True)[:3]
    res = {"status": "ok", "videos_with_metrics": len(data), "niche_weights": weights, "formats": formats,
           "recommended_long_share": round(long_share, 2), "hook_features": hooks,
           "best_hours": [h for _, h in best_hours], "top_hooks": [t["hook"] for t in top], "updated_at": time.time()}
    db.kv_set("learnings", res)
    return res


def niche_weight(niche: str) -> float:
    l = db.kv_get("learnings", {}) or {}
    return float((l.get("niche_weights") or {}).get(niche, 1.0))


def prompt_hints(niche: str | None = None) -> str:
    l = db.kv_get("learnings", {}) or {}
    if l.get("status") != "ok":
        return ""
    lines = []
    hf = sorted((l.get("hook_features") or {}).items(), key=lambda kv: -kv[1]["lift"])
    good = [k for k, v in hf if v["lift"] > 1.1 and v["n"] >= 3]
    bad = [k for k, v in hf if v["lift"] < 0.9 and v["n"] >= 3]
    if good:
        lines.append("Hooks that performed better: " + ", ".join(good))
    if bad:
        lines.append("Hook styles that underperformed: " + ", ".join(bad))
    if l.get("top_hooks"):
        lines.append("Best real hooks so far (do not copy, learn the pattern): " + " | ".join(l["top_hooks"]))
    return "\n".join(lines)
