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


# ------------------------------------------------------------------ web research via Claude Code CLI
WEB_PROMPT = """Research this topic for a factual short video: "{topic}".
Use web search (and fetch pages when needed). Prefer primary/authoritative sources (NASA, ESA, universities,
peer-reviewed journals, Wikipedia, major newspapers). Today is {today}. If the topic is about a recent event,
find the newest reliable reports.
Return ONLY JSON:
{{"sources": [{{"title": str, "url": str, "date": str|null}}],
  "facts": [{{"fact": "one precise, verifiable statement with numbers/dates exactly as the source states", "source": "<url>"}}],
  "angle": "the most surprising, curiosity-driving angle for a viral explainer (1 sentence)"}}
Include 8-15 facts, each backed by one of the listed URLs. Never invent facts, numbers or URLs."""


def web_research(topic: str, lang: str = "de", timeout: int = 420) -> dict:
    """Autonomous web research through the local `claude` CLI (uses the Claude subscription, no API cost)."""
    import datetime as dt
    import shutil
    import subprocess

    from .llm import extract_json

    exe = shutil.which("claude") or shutil.which("claude.cmd")
    if not exe:
        return {"sources": [], "context": "", "offline": True, "error": "claude CLI fehlt"}
    p = subprocess.run([exe, "-p", "--allowedTools", "WebSearch", "WebFetch", "--output-format", "text"],
                       input=WEB_PROMPT.format(topic=topic, today=dt.date.today().isoformat()),
                       capture_output=True, text=True, encoding="utf-8", timeout=timeout)
    try:
        data = extract_json(p.stdout)
    except Exception as e:
        return {"sources": [], "context": "", "offline": True, "error": f"web research parse: {e}"}
    srcs = [s for s in data.get("sources", []) if str(s.get("url", "")).startswith("http")]
    urls = {s["url"] for s in srcs}
    facts = [f for f in data.get("facts", []) if f.get("source") in urls]
    ctx = "\n".join(f"- {f['fact']} (SOURCE: {f['source']})" for f in facts)
    if data.get("angle"):
        ctx = f"SUGGESTED ANGLE: {data['angle']}\n" + ctx
    return {"sources": [{"title": s.get("title"), "url": s["url"]} for s in srcs], "context": ctx,
            "offline": not facts, "method": "claude_websearch"}


def gather_auto(topic: str, lang: str = "de") -> dict:
    """Best available research: web research via Claude CLI, enriched with Wikipedia when reachable."""
    web = web_research(topic, lang)
    wiki = gather(topic, "de" if lang == "de" else "en", max_pages=2)
    if web.get("context") and wiki.get("context"):
        return {"sources": web["sources"] + wiki["sources"], "context": web["context"] + "\n\n" + wiki["context"],
                "offline": False, "method": "claude_websearch+wikipedia"}
    return web if web.get("context") else wiki
