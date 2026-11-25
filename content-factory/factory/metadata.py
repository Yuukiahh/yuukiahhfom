"""Titles, descriptions and hashtags per platform."""
from __future__ import annotations

import re

NICHE_TAGS = {
    "space": ["space", "astronomy", "universe", "science", "blackhole", "cosmos"],
    "ai": ["ai", "artificialintelligence", "tech", "future", "chatgpt", "machinelearning"],
    "technology": ["technology", "tech", "innovation", "gadgets", "science"],
    "future": ["future", "futuretech", "innovation", "technology", "science"],
    "facts": ["facts", "didyouknow", "mindblown", "interesting", "learnontiktok"],
    "mystery": ["mystery", "unsolved", "creepy", "strange", "history"],
    "psychology": ["psychology", "mindset", "brain", "psychologyfacts", "selfimprovement"],
    "gaming": ["gaming", "gamer", "videogames", "gamingfacts"],
}


def _tag(s: str) -> str:
    return re.sub(r"[^\wäöüß]", "", s.lower())


def build(script: dict) -> dict:
    m = script.get("metadata", {})
    niche = script.get("niche", "facts")
    title = (m.get("title") or script.get("title") or "").strip()
    tags = [_tag(t) for t in m.get("hashtags", [])] + NICHE_TAGS.get(niche, [])
    seen, uniq = set(), []
    for t in tags:
        if t and t not in seen:
            seen.add(t)
            uniq.append(t)
    uniq = uniq[:12]
    desc = (m.get("description") or script.get("hook", "")).strip()
    sources = [c.get("source") for c in script.get("claims", []) if c.get("source")]
    src_line = ("\n\nSources: " + " | ".join(dict.fromkeys(sources))) if sources else ""
    yt_title = title if len(title) <= 90 else title[:87] + "…"
    return {
        "title": title,
        "youtube": {"title": (yt_title + " #Shorts") if len(yt_title) < 92 else yt_title,
                    "description": desc + src_line + "\n\n" + " ".join("#" + t for t in uniq[:8]),
                    "tags": uniq, "categoryId": "28" if niche in ("ai", "technology", "future") else "27"},
        "tiktok": {"caption": (desc.split(". ")[0] + " " + " ".join("#" + t for t in uniq[:6]))[:2150]},
        "instagram": {"caption": desc + "\n\n" + " ".join("#" + t for t in uniq[:10])},
        "hashtags": uniq,
        "ai_generated_disclosure": True,
    }
