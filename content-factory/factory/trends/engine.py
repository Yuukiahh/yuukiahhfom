"""Autonomous trend engine.

1. collect signals from all reachable sources (failures are recorded, never faked)
2. classify each signal into a niche (keyword lexicon)
3. normalise engagement per source (percentile rank) + recency
4. cluster similar signals across sources -> candidate topics
5. score = virality (engagement pct, velocity) x niche fit x cross-source corroboration
           x learned niche weight x low-competition bonus (YouTube, when key present)
6. optional LLM step turns raw headlines into concrete video ideas (keeps evidence)
"""
from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict

from .. import db, dedupe, learning
from ..config import cfg
from . import sources

NICHES = {
    "space": "space nasa black hole galaxy planet star universe cosmos mars moon asteroid comet telescope webb hubble orbit "
             "astronaut rocket spacex nebula supernova sun solar exoplanet gravity eclipse meteor voyager spacecraft probe "
             "interstellar europa jupiter saturn galaxies quasar quasars jwst euclid light-day lightyear cosmic dark matter",
    "ai": "ai artificial intelligence gpt chatgpt claude gemini llm openai anthropic neural model robot deepmind "
          "machine learning agent chatbot agi",
    "technology": "tech technology chip nvidia apple google microsoft samsung smartphone iphone battery quantum computer "
                  "software hardware gpu cpu semiconductor internet app gadget device 5g satellite",
    "future": "future fusion energy invention breakthrough scientists discover new material lab 2030 2050 "
              "longevity crispr gene brain implant neuralink hypersonic flying car",
    "facts": "til fact facts record largest smallest oldest world history animal ocean earth human body science study",
    "mystery": "mystery unsolved disappeared strange unexplained ancient lost secret cold case ghost cave signal "
               "anomaly cursed creepy hidden",
    "psychology": "psychology brain mind habit behavior emotion memory sleep dopamine anxiety study shows "
                  "personality cognitive bias motivation loneliness",
    "gaming": "game gaming nintendo playstation xbox steam gta minecraft fortnite esports switch zelda pokemon valve "
              "speedrun trailer",
}
_LEX = {n: set(w.split()) for n, w in NICHES.items()}


def classify(title: str, hint: str = "") -> tuple[str | None, float]:
    text = (title + " " + hint).lower()
    toks = set(re.findall(r"[a-z0-9]+", text))
    best, score = None, 0.0
    for n, lex in _LEX.items():
        hits = len(toks & lex) + sum(1 for w in lex if " " in w and w in text)
        if hits > score or (hits == score and hits > 0 and best == "facts"):
            best, score = n, float(hits)
    return best, min(1.0, score / 2)


SUB_NICHE = {"space": "space", "technology": "technology", "futurology": "future", "todayilearned": "facts",
             "artificial": "ai", "psychology": "psychology", "gaming": "gaming", "unresolvedmysteries": "mystery",
             "askscience": "facts", "interestingasfuck": "facts"}


def collect() -> dict:
    c = cfg()
    jobs = [("google_trends", lambda g=g: sources.google_trends(g)) for g in c.get("trends.geos", ["US"])]
    jobs += [("reddit", lambda s=s: sources.reddit(s)) for s in c.get("trends.subreddits", [])]
    jobs += [("youtube", lambda: sources.youtube("US")), ("wikipedia", lambda: sources.wikipedia_top("en")),
             ("hackernews", sources.hackernews), ("manual", sources.manual_imports),
             ("instagram", lambda: sources.instagram_hashtags(["space", "ai", "technology", "facts", "psychology"]))]
    from concurrent.futures import ThreadPoolExecutor
    status: dict[str, dict] = defaultdict(lambda: {"ok": 0, "failed": 0, "signals": 0, "errors": []})
    now = time.time()
    total = 0

    def run(item):
        name, fn = item
        try:
            return name, fn(), None
        except Exception as e:
            return name, [], f"{type(e).__name__}: {str(e)[:160]}"

    with ThreadPoolExecutor(8) as ex:
        for name, sigs, err in ex.map(run, jobs):
            st = status[name]
            if err:
                st["failed"] += 1
                st["errors"].append(err)
                continue
            st["ok"] += 1
            for s in sigs:
                hint = SUB_NICHE.get(s.get("extra", {}).get("subreddit", "").lower(), "")
                niche, fit = classify(s["title"] + " " + " ".join(s.get("extra", {}).get("news", [])), hint)
                if hint and not niche:
                    niche, fit = hint, 0.5
                db.insert("trend_signals", source=s["source"], title=s["title"], url=s.get("url"),
                          metric_name=s.get("metric_name"), metric_value=s.get("metric_value"), niche=niche,
                          raw={**s, "niche_fit": fit}, fetched_at=now)
                st["signals"] += 1
                total += 1
    summary = {"fetched_at": now, "signals": total, "sources": dict(status)}
    db.kv_set("trend_source_status", summary)
    return summary


def _pct_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    for r, i in enumerate(order):
        out[i] = (r + 1) / len(values)
    return out


def score_topics(lookback_h: float | None = None) -> list[dict]:
    lookback_h = lookback_h or cfg().get("trends.lookback_days", 3) * 24
    sigs = db.rows("SELECT * FROM trend_signals WHERE fetched_at > ?", time.time() - lookback_h * 3600)
    if not sigs:
        return []
    # keep latest observation per (source,title)
    latest: dict[tuple, dict] = {}
    for s in sigs:
        k = (s["source"], s["title"])
        if k not in latest or s["fetched_at"] > latest[k]["fetched_at"]:
            latest[k] = s
    sigs = list(latest.values())
    by_src = defaultdict(list)
    for s in sigs:
        by_src[s["source"]].append(s)
    for src, lst in by_src.items():
        ranks = _pct_ranks([float(x["metric_value"] or 0) for x in lst])
        for x, r in zip(lst, ranks):
            x["eng_pct"] = r if any((y["metric_value"] or 0) > 0 for y in lst) else 0.6
            raw = x["raw"] if isinstance(x["raw"], dict) else json.loads(x["raw"] or "{}")
            x["fit"] = raw.get("niche_fit", 0)
            ex = raw.get("extra", {})
            x["velocity"] = ex.get("upvotes_per_hour") or ex.get("views_per_hour") or 0
            pub = raw.get("published")
            x["age_h"] = (time.time() - pub) / 3600 if pub else 24
    # cluster by token similarity
    clusters: list[list[dict]] = []
    for s in sorted(sigs, key=lambda x: -x["eng_pct"]):
        if not s["niche"]:
            continue
        for c in clusters:
            if c[0]["niche"] == s["niche"] and dedupe.similarity(c[0]["title"], s["title"]) >= 0.34:
                c.append(s)
                break
        else:
            clusters.append([s])
    topics = []
    for c in clusters:
        lead = c[0]
        srcs = {x["source"] for x in c}
        virality = max(x["eng_pct"] for x in c)
        recency = math.exp(-min(x["age_h"] for x in c) / 48)
        fit = max(x["fit"] for x in c)
        corroboration = 1 + 0.25 * (len(srcs) - 1) + 0.05 * (len(c) - 1)
        nw = learning.niche_weight(lead["niche"])
        score = (0.5 * virality + 0.2 * recency + 0.3 * fit) * corroboration * nw
        topics.append({"title": lead["title"], "niche": lead["niche"], "score": round(min(1.5, score), 4),
                       "score_detail": {"virality_pct": round(virality, 3), "recency": round(recency, 3), "niche_fit": fit,
                                        "sources": sorted(srcs), "signals": len(c), "learned_niche_weight": nw},
                       "evidence": [{"source": x["source"], "title": x["title"], "url": x["url"],
                                     "metric": f"{x['metric_name']}={x['metric_value']:.0f}"} for x in c[:6]]})
    topics.sort(key=lambda t: -t["score"])
    return topics


def ideate(topics: list[dict], n: int = 12) -> list[dict]:
    """Optional: LLM turns headlines into evergreen-angled video ideas. Keeps evidence; never invents trend data."""
    from .. import llm
    if not topics:
        return []
    listing = "\n".join(f"{i}. [{t['niche']}] {t['title']}" for i, t in enumerate(topics[:40]))
    prompt = ("Below are REAL trending headlines/topics with their niche. Pick the {n} with the best potential for a "
              "25-90 s educational vertical video (high curiosity, explainable visually, not a tragedy/politics/celebrity gossip, "
              "safe for advertisers). For each, write a concrete video topic in English as a viewer-facing question or promise.\n"
              "Return JSON: {{\"ideas\": [{{\"index\": int, \"video_topic\": str, \"why\": str}}]}}\n\n{listing}").format(n=n, listing=listing)
    try:
        res, _ = llm.complete_json(prompt, "You are a viral content strategist. JSON only.")
    except Exception:
        return topics[:n]
    out = []
    for idea in res.get("ideas", []):
        i = idea.get("index")
        if isinstance(i, int) and 0 <= i < len(topics):
            t = dict(topics[i])
            t["source_title"] = t["title"]
            t["title"] = idea.get("video_topic") or t["title"]
            t["score_detail"] = {**t["score_detail"], "llm_reason": idea.get("why")}
            out.append(t)
    return out or topics[:n]


def refresh(use_llm: bool = True) -> dict:
    summary = collect()
    topics = score_topics()
    min_score = cfg().get("trends.min_score", 0.35)
    topics = [t for t in topics if t["score"] >= min_score]
    if use_llm:
        topics = ideate(topics)
    now = time.time()
    with db.tx() as c:
        c.execute("UPDATE topics SET status='expired' WHERE status='candidate'")
    added = 0
    for t in topics:
        dup, _ = dedupe.is_duplicate(t["title"])
        if dup:
            continue
        db.insert("topics", title=t["title"], niche=t["niche"], score=t["score"], score_detail=t["score_detail"],
                  evidence=t["evidence"], status="candidate", created_at=now)
        added += 1
    summary["topics_added"] = added
    db.kv_set("trend_last_refresh", summary)
    return summary


def top_topics(limit: int = 30) -> list[dict]:
    return db.rows("SELECT * FROM topics WHERE status='candidate' ORDER BY score DESC LIMIT ?", limit)


def niche_report() -> list[dict]:
    rows = db.rows("SELECT niche, COUNT(*) n, AVG(score) avg_score, MAX(score) max_score FROM topics "
                   "WHERE status='candidate' GROUP BY niche ORDER BY avg_score DESC")
    for r in rows:
        r["sample_titles"] = [x["title"] for x in db.rows(
            "SELECT title FROM topics WHERE status='candidate' AND niche=? ORDER BY score DESC LIMIT 3", r["niche"])]
        r["learned_weight"] = learning.niche_weight(r["niche"])
        r["score"] = round((r["avg_score"] or 0) * 0.6 + (r["max_score"] or 0) * 0.4, 3)
    return sorted(rows, key=lambda r: -r["score"])


def source_status() -> dict:
    return db.kv_get("trend_source_status", {})
