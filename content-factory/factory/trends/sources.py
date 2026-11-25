"""Trend sources. Each returns a list of signal dicts:
{source, title, url, metric_name, metric_value, published (ts|None), extra{}}

Only public / official endpoints. No scraping of TikTok/Instagram web pages
(against their terms):
  * TikTok   – no free public trends API; Research API needs approval -> manual import
  * Instagram – Graph API hashtag search when an IG business account is connected
"""
from __future__ import annotations

import datetime as dt
import html
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

from .. import secrets
from ..config import cfg

UA = {"User-Agent": "ai-content-factory/1.0 (trend research; contact: local user)"}
T = 20


def google_trends(geo: str = "US") -> list[dict]:
    r = requests.get(f"https://trends.google.com/trending/rss?geo={geo}", headers=UA, timeout=T)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    ns = {"ht": "https://trends.google.com/trending/rss"}
    out = []
    for it in root.iter("item"):
        traffic = (it.findtext("ht:approx_traffic", default="0", namespaces=ns) or "0")
        val = float(re.sub(r"[^\d]", "", traffic) or 0)
        news = [n.findtext("ht:news_item_title", namespaces=ns) for n in it.findall("ht:news_item", ns)]
        pub = it.findtext("pubDate")
        ts = None
        if pub:
            try:
                ts = dt.datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %z").timestamp()
            except ValueError:
                pass
        out.append({"source": "google_trends", "title": it.findtext("title"), "url": f"https://trends.google.com/trending?geo={geo}",
                    "metric_name": "approx_searches", "metric_value": val, "published": ts,
                    "extra": {"geo": geo, "news": [html.unescape(n) for n in news if n][:3]}})
    return out


def reddit(sub: str) -> list[dict]:
    r = requests.get(f"https://www.reddit.com/r/{sub}/top.json", params={"t": "day", "limit": 25}, headers=UA, timeout=T)
    r.raise_for_status()
    out = []
    for c in r.json()["data"]["children"]:
        d = c["data"]
        if d.get("over_18") or d.get("stickied"):
            continue
        age_h = max(1.0, (time.time() - d["created_utc"]) / 3600)
        out.append({"source": "reddit", "title": html.unescape(d["title"]), "url": "https://www.reddit.com" + d["permalink"],
                    "metric_name": "upvotes", "metric_value": float(d["score"]), "published": d["created_utc"],
                    "extra": {"subreddit": sub, "comments": d["num_comments"], "upvotes_per_hour": round(d["score"] / age_h, 1),
                              "upvote_ratio": d.get("upvote_ratio")}})
    return out


def youtube(region: str = "US") -> list[dict]:
    key = secrets.get("youtube_api_key")
    if not key:
        return []
    out = []
    for cat in ("28", "27", "20"):  # Science&Tech, Education, Gaming
        r = requests.get("https://www.googleapis.com/youtube/v3/videos", timeout=T, params={
            "part": "snippet,statistics,contentDetails", "chart": "mostPopular", "regionCode": region,
            "videoCategoryId": cat, "maxResults": 25, "key": key})
        if r.status_code != 200:
            continue
        for v in r.json().get("items", []):
            st = v.get("statistics", {})
            pub = dt.datetime.fromisoformat(v["snippet"]["publishedAt"].replace("Z", "+00:00")).timestamp()
            age_h = max(1.0, (time.time() - pub) / 3600)
            views = float(st.get("viewCount", 0))
            dur = v.get("contentDetails", {}).get("duration", "")
            out.append({"source": "youtube", "title": v["snippet"]["title"], "url": f"https://youtu.be/{v['id']}",
                        "metric_name": "views", "metric_value": views, "published": pub,
                        "extra": {"category": cat, "views_per_hour": round(views / age_h), "likes": int(st.get("likeCount", 0)),
                                  "comments": int(st.get("commentCount", 0)), "duration": dur,
                                  "is_short": bool(re.fullmatch(r"PT(\d+S|[0-2]M(\d+S)?)", dur or ""))}})
    return out


def youtube_competition(query: str) -> dict | None:
    """Supply vs demand for a topic among recent Shorts (costs 100 quota units per call)."""
    key = secrets.get("youtube_api_key")
    if not key:
        return None
    after = (dt.datetime.utcnow() - dt.timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    r = requests.get("https://www.googleapis.com/youtube/v3/search", timeout=T, params={
        "part": "id", "q": query, "type": "video", "videoDuration": "short", "publishedAfter": after,
        "order": "viewCount", "maxResults": 10, "key": key})
    if r.status_code != 200:
        return None
    ids = [i["id"]["videoId"] for i in r.json().get("items", [])]
    total = r.json().get("pageInfo", {}).get("totalResults", 0)
    views = []
    if ids:
        s = requests.get("https://www.googleapis.com/youtube/v3/videos", timeout=T,
                         params={"part": "statistics", "id": ",".join(ids), "key": key}).json()
        views = [int(x["statistics"].get("viewCount", 0)) for x in s.get("items", [])]
    return {"recent_shorts": total, "top10_median_views": sorted(views)[len(views) // 2] if views else 0}


def wikipedia_top(lang: str = "en") -> list[dict]:
    day = dt.date.today() - dt.timedelta(days=1)
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/{lang}.wikipedia/all-access/{day:%Y/%m/%d}"
    r = requests.get(url, headers=UA, timeout=T)
    r.raise_for_status()
    out = []
    skip = re.compile(r"^(Main_Page|Special:|Wikipedia:|Portal:|File:|Help:|Hauptseite|Spezial:)")
    for a in r.json()["items"][0]["articles"][:120]:
        if skip.match(a["article"]):
            continue
        out.append({"source": "wikipedia", "title": a["article"].replace("_", " "),
                    "url": f"https://{lang}.wikipedia.org/wiki/{a['article']}", "metric_name": "daily_pageviews",
                    "metric_value": float(a["views"]), "published": None, "extra": {"lang": lang, "rank": a["rank"]}})
    return out


def hackernews() -> list[dict]:
    r = requests.get("https://hn.algolia.com/api/v1/search", params={"tags": "front_page", "hitsPerPage": 40}, timeout=T, headers=UA)
    r.raise_for_status()
    return [{"source": "hackernews", "title": h["title"], "url": f"https://news.ycombinator.com/item?id={h['objectID']}",
             "metric_name": "points", "metric_value": float(h.get("points") or 0), "published": h.get("created_at_i"),
             "extra": {"comments": h.get("num_comments")}} for h in r.json()["hits"]]


def instagram_hashtags(tags: list[str]) -> list[dict]:
    token, uid = secrets.get("instagram_access_token"), secrets.get("instagram_user_id")
    if not token or not uid:
        return []
    out = []
    for tag in tags[:10]:  # API limit: 30 unique hashtags / 7 days
        s = requests.get("https://graph.facebook.com/v21.0/ig_hashtag_search", timeout=T,
                         params={"user_id": uid, "q": tag, "access_token": token}).json()
        if not s.get("data"):
            continue
        hid = s["data"][0]["id"]
        m = requests.get(f"https://graph.facebook.com/v21.0/{hid}/top_media", timeout=T, params={
            "user_id": uid, "fields": "caption,like_count,comments_count,media_type,permalink,timestamp",
            "access_token": token}).json()
        for x in m.get("data", [])[:15]:
            out.append({"source": "instagram", "title": (x.get("caption") or "")[:140], "url": x.get("permalink"),
                        "metric_name": "likes", "metric_value": float(x.get("like_count") or 0), "published": None,
                        "extra": {"hashtag": tag, "comments": x.get("comments_count"), "type": x.get("media_type")}})
    return out


def manual_imports() -> list[dict]:
    """JSON/CSV files dropped into data/trends_import/ (e.g. TikTok Creative Center exports
    or web-research results). Each JSON item: {source,title,url,metric_name,metric_value}."""
    folder = cfg().path("general.data_dir") / "trends_import"
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for f in sorted(folder.glob("*.json")):
        try:
            items = json.loads(f.read_text("utf-8"))
        except ValueError:
            continue
        for it in items:
            if it.get("title"):
                it.setdefault("source", "manual")
                it.setdefault("metric_name", "manual_rank")
                it.setdefault("metric_value", 0)
                it.setdefault("extra", {})
                it["extra"]["import_file"] = f.name
                out.append(it)
    for f in sorted(folder.glob("*.csv")):
        import csv
        with open(f, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                title = row.get("title") or row.get("hashtag") or row.get("keyword")
                if title:
                    out.append({"source": row.get("source", "tiktok_import"), "title": title, "url": row.get("url"),
                                "metric_name": row.get("metric_name", "views"), "metric_value": float(row.get("metric_value") or row.get("views") or 0),
                                "published": None, "extra": {"import_file": f.name}})
    return out


TREND_PROMPT = """Find what is CURRENTLY trending/viral (last 7 days, today is {today}) that would make great
educational short videos (TikTok/Reels/Shorts) for a {region} audience in these niches: incredible facts, AI,
technology, space & black holes, mysterious stories, psychology, gaming, future tech.
Use web search: look at news, Google Trends-style coverage, Reddit/YouTube/TikTok trend reports, science news.
Return ONLY JSON: {{"items": [{{"title": "short topic headline", "niche": "space|ai|technology|future|facts|mystery|psychology|gaming",
"url": "a real URL you found that shows this is current", "evidence": "what shows it is trending (as stated by the source)",
"published": "YYYY-MM-DD or null"}}]}} with 12-20 items, most viral first. Only items you actually found; never invent URLs."""


def claude_websearch_trends(region: str = "German-speaking") -> list[dict]:
    """Trend discovery through the local `claude` CLI web search (Claude subscription, no API cost)."""
    import datetime as dt
    import shutil
    import subprocess

    from ..llm import extract_json

    exe = shutil.which("claude") or shutil.which("claude.cmd")
    if not exe:
        return []
    p = subprocess.run([exe, "-p", "--allowedTools", "WebSearch", "WebFetch", "--output-format", "text"],
                       input=TREND_PROMPT.format(today=dt.date.today().isoformat(), region=region),
                       capture_output=True, text=True, encoding="utf-8", timeout=600)
    items = extract_json(p.stdout).get("items", [])
    out = []
    n = len(items)
    for i, it in enumerate(items):
        if not str(it.get("url", "")).startswith("http") or not it.get("title"):
            continue
        ts = None
        try:
            ts = dt.datetime.fromisoformat(str(it.get("published"))).timestamp() if it.get("published") else None
        except ValueError:
            pass
        out.append({"source": "websearch", "title": it["title"], "url": it["url"], "metric_name": "websearch_rank_score",
                    "metric_value": float(n - i), "published": ts,
                    "extra": {"evidence": it.get("evidence"), "niche_hint": it.get("niche"), "method": "claude_cli_websearch"}})
    return out
