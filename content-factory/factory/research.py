"""Fact research: Wikipedia (REST + search) as primary free source.

The scriptwriter gets the extracted text as grounding context and must cite the
returned URLs; the fact-checker re-verifies every claim against the same context.
"""
from __future__ import annotations

import requests

UA = {"User-Agent": "ai-content-factory/1.0 (educational short-video research)"}


def wiki_search(query: str, lang: str = "en", limit: int = 4) -> list[str]:
    r = requests.get(f"https://{lang}.wikipedia.org/w/api.php", headers=UA, timeout=20, params={
        "action": "query", "list": "search", "srsearch": query, "srlimit": limit, "format": "json"})
    r.raise_for_status()
    return [x["title"] for x in r.json()["query"]["search"]]


def wiki_extract(title: str, lang: str = "en", chars: int = 6000) -> dict:
    r = requests.get(f"https://{lang}.wikipedia.org/w/api.php", headers=UA, timeout=20, params={
        "action": "query", "prop": "extracts|info", "explaintext": 1, "exsectionformat": "plain",
        "titles": title, "inprop": "url", "format": "json", "redirects": 1})
    r.raise_for_status()
    page = next(iter(r.json()["query"]["pages"].values()))
    return {"title": page.get("title"), "url": page.get("fullurl"), "text": (page.get("extract") or "")[:chars]}


def gather(topic: str, lang: str = "en", max_pages: int = 3) -> dict:
    """Returns {'sources': [{title,url,text}], 'context': str, 'offline': bool}."""
    try:
        titles = wiki_search(topic, lang)[:max_pages]
        sources = [wiki_extract(t, lang) for t in titles]
        sources = [s for s in sources if s["text"]]
        if lang != "en" and len(sources) < 2:
            sources += [wiki_extract(t, "en") for t in wiki_search(topic, "en")[:2]]
        ctx = "\n\n".join(f"### SOURCE [{i+1}] {s['title']} ({s['url']})\n{s['text']}" for i, s in enumerate(sources))
        return {"sources": [{k: s[k] for k in ("title", "url")} for s in sources], "context": ctx, "offline": False}
    except Exception as e:
        return {"sources": [], "context": "", "offline": True, "error": str(e)}
