"""Script + storyboard generation and fact-checking via the LLM layer.

Output schema (also used for hand-written / Claude-Code-written scripts):
{
  "title": str, "hook": str, "hook_text": str (<=5 words on screen), "niche": str, "mood": str,
  "format": "short"|"long", "language": "en"|"de",
  "scenes": [{
      "narration": str, "visual": str (procedural kind), "params": {},
      "camera": str, "stock_query": str, "image_prompt": str,
      "emphasis": [str], "sfx": "whoosh|impact|riser|pop|glitch|ding|none",
      "stat": {"text": str, "sub": str} | null
  }],
  "claims": [{"claim": str, "source": url}],
  "metadata": {"title": str, "description": str, "hashtags": [str]}
}
"""
from __future__ import annotations

import json
import re

from . import llm
from .render import CAMERA_MOVES
from .visuals.procedural import KINDS

WPS = {"en": 2.65, "de": 2.55}  # words per second at speed ~1.08
FORMATS = {"short": (25, 45), "long": (60, 90)}

SYSTEM = """You are a top-tier short-form video scriptwriter (TikTok / Instagram Reels / YouTube Shorts) and a rigorous
science journalist. You write ORIGINAL scripts in the requested language (native, natural, spoken style - for German:
idiomatic "du"-Form, kurze Sätze, keine Anglizismen-Flut). Never copy text from sources, never invent facts, numbers or
quotes. Every factual claim must be supported by the provided SOURCES; if they do not support it, leave it out.
RETENTION RULES: (1) the first sentence is the hook and must land in < 1.5 s: shocking number, bold claim, direct
"du"-question or visual promise - no greeting, no "In diesem Video", no "Wusstest du". (2) open a loop in the first
5 s that is only closed near the end. (3) a new visual/idea every 2-3 seconds -> many short scenes. (4) one
"Aber hier kommt der Twist / Doch dann ..." pattern interrupt around the middle. (5) concrete images, comparisons
and numbers. (6) satisfying payoff, and a last line that loops back to the hook so the video rewatches seamlessly.
Write numbers as digits (they are spoken correctly by the TTS). No emojis in narration. Output JSON only."""

PROMPT = """Write a {format} vertical video script about: "{topic}"
Niche: {niche}. Language: {language_name}. Target length: {min_s}-{max_s} seconds = {min_w}-{max_w} words of narration TOTAL.
{learnings}
SOURCES (only facts from here are allowed, cite by URL):
{context}

Return ONE JSON object with exactly these keys:
- "title": catchy, <= 60 chars
- "hook": the first spoken sentence (<= 14 words, must create curiosity instantly)
- "hook_text": 2-5 word on-screen hook caption
- "niche": one of space, ai, technology, future, facts, mystery, psychology, gaming
- "mood": one of cosmic, mystery, pulse, epic, playful
- "scenes": {n_min}-{n_max} scenes (each 1-2 short sentences, about 2-4 seconds spoken). The first scene's narration
    starts with the hook. Each scene:
    "narration" (1-2 short sentences), "visual" (one of: {kinds} - prefer the dynamic 3D kinds ending in 3d where they fit),
    "params" (optional, e.g. {{"palette":"ember|violet|teal|ice"}}), "camera" (one of: {cams}),
    "stock_query" (2-4 word English stock-footage search), "image_prompt" (detailed English image prompt for an AI image),
    "emphasis" (1-3 key words from this narration to colour-highlight), "sfx" (whoosh|impact|riser|pop|glitch|ding|none),
    "stat" (optional big on-screen number card, max 10 characters: {{"text":"10¹² K","sub":"Kerntemperatur"}} or null;
            use 2-4 stat cards per video on the most impressive numbers),
    "mascot" (optional reaction of our 3D mascot Nova: wow|think|happy|point|wave or null; use 1-3 times)
- "claims": list of {{"claim": "...", "source": "<url from SOURCES>"}} for every factual statement
- "metadata": {{"title": "<=90 chars incl. 1 emoji, in the video language; must truthfully describe the video - no fake list numbers, no promises the video does not keep", "description": "2-3 sentences + sources line",
                "hashtags": [8-12 tags without #, mix of the video language and English]}}
"""

VERIFY = """You are a strict fact-checker. Check every factual statement in the SCRIPT against the SOURCES.
SOURCES:
{context}

SCRIPT (JSON):
{script}

Return JSON: {{"verdict": "pass"|"fix", "issues": [{{"text": "...", "problem": "...", "fix": "..."}}],
"script": <the corrected full script JSON with the same schema, unchanged if verdict is pass>}}"""


def fmt_bounds(fmt: str) -> tuple[int, int]:
    """'short' | 'long' | exact length like '30s' (target ±3 s)."""
    if fmt in FORMATS:
        return FORMATS[fmt]
    m = re.fullmatch(r"(\d{2,3})s", str(fmt))
    if m:
        n = int(m.group(1))
        return (max(10, n - 3), n + 2)
    return FORMATS["short"]


def target_words(fmt: str, lang: str) -> tuple[int, int]:
    lo, hi = fmt_bounds(fmt)
    wps = WPS.get(lang, 2.5)
    return int(lo * wps * 0.92), int(hi * wps * 0.9)


def write(topic: str, niche: str, fmt: str, lang: str, research: dict, learnings: str = "") -> tuple[dict, str]:
    lo, hi = fmt_bounds(fmt)
    min_w, max_w = target_words(fmt, lang)
    n_min, n_max = (9, 14) if fmt == "short" else (16, 26) if fmt == "long" else (max(5, hi // 4), max(7, hi // 3))
    prompt = PROMPT.format(format=fmt, topic=topic, niche=niche, language_name={"en": "English", "de": "German (Deutsch)"}.get(lang, lang),
                           min_s=lo, max_s=hi, min_w=min_w, max_w=max_w, n_min=n_min, n_max=n_max,
                           kinds=", ".join(sorted(set(KINDS))), cams=", ".join(CAMERA_MOVES),
                           context=research.get("context") or "(offline – use only well-established textbook facts and keep numbers conservative)",
                           learnings=("WHAT WORKED BEFORE (performance learnings):\n" + learnings) if learnings else "")
    script, provider = llm.complete_json(prompt, SYSTEM)
    return normalise(script, fmt, lang), provider


def fact_check(script: dict, research: dict) -> tuple[dict, dict]:
    if not research.get("context"):
        return script, {"verdict": "skipped", "reason": "no sources available"}
    res, _ = llm.complete_json(VERIFY.format(context=research["context"], script=json.dumps(script, ensure_ascii=False)),
                               "You are a meticulous fact-checker. Output JSON only.")
    fixed = res.get("script") if isinstance(res.get("script"), dict) and res.get("verdict") == "fix" else script
    return normalise(fixed, script.get("format", "short"), script.get("language", "en")), {
        "verdict": res.get("verdict"), "issues": res.get("issues", [])}


VISUAL_ALIASES = {
    "physics": "atom3d", "quantum": "atom3d", "atom": "atom3d", "particle": "atom3d", "particles": "atom3d",
    "flyover": "terrain3d", "landscape": "terrain3d", "mountains": "terrain3d", "terrain": "terrain3d",
    "data": "datacity3d", "server": "datacity3d", "city": "datacity3d", "computer": "datacity3d", "robot": "datacity3d",
    "genome": "dna3d", "helix": "dna3d", "gene": "dna3d", "cell": "dna3d",
    "globe": "globe3d", "world": "globe3d", "map": "globe3d",
    "hyperspace": "warp3d", "speed": "warp3d", "travel": "warp3d", "light": "warp3d",
    "milkyway": "galaxy3d", "milky_way": "galaxy3d", "universe": "galaxy3d", "quasar": "galaxy3d", "cosmos": "galaxy3d",
    "saturn": "ringplanet3d", "rings": "ringplanet3d", "jupiter": "planet", "moon": "planet",
    "satellite": "probe", "rocket": "probe", "telescope": "probe",
    "network": "neural3d", "mind": "neural3d", "thought": "neural3d",
}


def normalise(s: dict, fmt: str, lang: str) -> dict:
    s.setdefault("format", fmt)
    s.setdefault("language", lang)
    s.setdefault("mood", "cosmic")
    s.setdefault("claims", [])
    s.setdefault("metadata", {})
    for i, sc in enumerate(s.get("scenes", [])):
        v = str(sc.get("visual", "")).lower().strip()
        if v in VISUAL_ALIASES and VISUAL_ALIASES[v] in KINDS:
            sc["visual"] = VISUAL_ALIASES[v]
        if sc.get("visual") not in KINDS:
            sc["visual"] = {"space": "nebula", "ai": "neural", "technology": "circuit", "future": "grid",
                            "mystery": "fog"}.get(s.get("niche"), "abstract")
        if sc.get("camera") not in CAMERA_MOVES:
            sc["camera"] = CAMERA_MOVES[i % len(CAMERA_MOVES)]
        sc.setdefault("emphasis", [])
        sc.setdefault("sfx", "whoosh" if i else "impact")
        sc["narration"] = re.sub(r"\s+", " ", sc.get("narration", "")).strip()
    return s


def word_count(s: dict) -> int:
    return sum(len(sc["narration"].split()) for sc in s["scenes"])


def validate(s: dict) -> list[str]:
    problems = []
    if not s.get("scenes"):
        return ["no scenes"]
    lo, hi = target_words(s["format"], s.get("language", "en"))
    wc = word_count(s)
    if wc < lo * 0.85 or wc > hi * 1.15:
        problems.append(f"word count {wc} outside {lo}-{hi}")
    if len(s.get("hook", "").split()) > 16:
        problems.append("hook too long")
    if not s["scenes"][0]["narration"].lower().startswith(s.get("hook", "")[:20].lower()):
        problems.append("first scene does not start with hook")
    for sc in s["scenes"]:
        if not sc["narration"]:
            problems.append("empty narration")
    return problems


# ------------------------------------------------------------------ overlay + background planner
VISUAL_CATALOG = {
    "gpu_chip": "close-up GPU / semiconductor chip (chips, Nvidia, AMD, hardware, compute)",
    "stock_chart": "3D candlestick stock chart; params.trend up|down|volatile (shares, markets, money, valuations)",
    "ticker_wall": "stock-exchange ticker boards (Wall Street, trading, market reaction)",
    "corporate": "night skyline of corporate glass towers (companies, CEOs, boards, business decisions)",
    "newsroom": "TV news studio with world map wall (breaking news, announcements, reports)",
    "radar": "intelligence radar sweep over world map (spies, security, investigations, threats)",
    "capitol": "neoclassical government building at dusk (politics, presidents, laws, governments)",
    "datacenter": "server hall aisle with blinking racks (AI compute, cloud, data, servers)",
    "calendar": "3D flipping calendar day cards; params.highlight = day number (dates, deadlines, 'days later')",
    "courtroom": "court bench and gavel (lawsuits, trials, judges)",
    "holo_cards": "floating holographic trading cards; params.mode float|fan|burst (trading cards, TCG, collectibles, packs, pulls, rare cards)",
    "blackhole": "black hole with lensed accretion disk (black holes, gravity, extreme physics)",
    "galaxy3d": "rotating 3D spiral galaxy (galaxies, universe, cosmic scale)",
    "nebula": "colourful nebula and stars (space, stars, cosmos)",
    "planet": "shaded planet (planets, moons, solar system)",
    "earth": "Earth-like planet (Earth, climate, our planet)",
    "ringplanet3d": "ringed planet like Saturn",
    "probe": "deep-space probe / satellite (spacecraft, telescopes, missions)",
    "warp3d": "hyperspace star-streak flight (speed, rapid change, the future arriving)",
    "wormhole": "glowing tunnel (time, journeys, mysteries, transitions)",
    "globe3d": "holographic globe (global, worldwide, countries, geopolitics)",
    "neural3d": "3D neural network of glowing nodes (AI, brains, learning, networks)",
    "datacity3d": "flight over a glowing data city (digital economy, tech industry)",
    "circuit": "circuit board with pulsing traces (electronics, generic tech)",
    "atom3d": "atom with orbiting electrons (physics, quantum, particles)",
    "dna3d": "3D DNA double helix (genes, biology, medicine)",
    "fusion": "fusion plasma sphere (energy, reactors, power)",
    "terrain3d": "retro wireframe landscape flyover (journeys, retro future, gaming)",
    "fog": "foggy dark forest (mystery, horror, unknown)",
    "abstract": "soft abstract gradients (psychology, emotions, ideas)",
}


def visual_catalog() -> dict:
    return {k: v for k, v in VISUAL_CATALOG.items() if k in KINDS}


# ------------------------------------------------------------------ overlay planner (motion graphics)
OVERLAY_PROMPT = """You are a motion-graphics editor for viral explainer shorts. For the SCRIPT below, choose for EVERY
scene one on-screen infographic overlay that illustrates exactly what is being said (concrete examples, dates, numbers,
people, comparisons). Use ONLY facts, numbers and dates that appear in the script or the SOURCES - never invent data.
Vary the types; do not use the same type more than 3 times. Write overlay text in {lang_name}, short and punchy.

Overlay types and their "data":
- headline  {{"kicker": "BREAKING|NEWS|UPDATE|FACT", "text": "<= 70 chars", "source": "outlet", "date": "short date"}}
- timeline  {{"events": [{{"date": "...", "label": "<= 18 chars"}}] (2-4 events, chronological), "active": index}}
- bars      {{"title": "...", "unit": "%|...", "items": [{{"label": "...", "value": number}}] (2-4, negative allowed)}}
- counter   {{"value": number, "prefix": "+|", "suffix": "%| Mio.|...", "label": "<= 30 chars", "decimals": 0|1}}
- quote     {{"text": "<= 110 chars, verbatim or close paraphrase from a source", "source": "who said it / outlet"}}
- person    {{"name": "...", "role": "...", "status": "<= 16 chars", "status_color": "green|red|yellow"}}
- checklist {{"title": "...", "items": ["<= 28 chars", ...] (2-4)}}
- compare   {{"left": {{"label": "...", "value": "<= 12 chars"}}, "right": {{"label": "...", "value": "<= 12 chars"}}}}
Use "overlay": null for a scene only if a big visual moment needs a clean frame (max 2 scenes).

ALSO choose for every scene the BACKGROUND "visual" that literally depicts the concrete subject of that sentence (a viewer
must instantly see what it is about). Pick from this catalogue (key: what it shows):
{catalog}
Rules: stock/share-price sentences -> stock_chart (params.trend up/down) or ticker_wall; a company/CEO/board -> corporate;
chips/GPUs -> gpu_chip; government/president/politics -> capitol; spies/intelligence/investigation -> radar;
dates/"x days later" -> calendar (params.highlight = day); breaking news -> newsroom; AI/compute -> datacenter or neural3d.
Avoid repeating the same visual in consecutive scenes. Also choose one "accent" hex colour that fits the topic/brand
(e.g. Nvidia -> "#76B900", space -> "#7C3AED", finance red/green themes...).

Return JSON: {{"tag": "<= 22 chars topic tag, e.g. NVIDIA · NOV 2026", "accent": "#RRGGBB", "scenes": [{{"visual": "<catalogue key>", "params": {{...}},
"overlay": {{"type": "...", "data": {{...}}}} | null,
"source_label": "short outlet name(s) backing this scene, or null"}}, ...]}} with exactly {n} scene entries in order.

SOURCES:
{context}

SCRIPT:
{script}"""


def _valid_overlay(o) -> bool:
    if not isinstance(o, dict) or not isinstance(o.get("data"), dict):
        return False
    t, d = o.get("type"), o["data"]
    need = {"headline": ["text"], "timeline": ["events"], "bars": ["items"], "counter": ["value"], "quote": ["text"],
            "person": ["name"], "checklist": ["items"], "compare": ["left", "right"]}
    if t not in need or not all(d.get(k) not in (None, "", []) for k in need[t]):
        return False
    if t == "bars":
        return all(isinstance(i.get("value"), (int, float)) for i in d["items"])
    if t == "counter":
        return isinstance(d["value"], (int, float))
    return True


def plan_overlays(script: dict, research: dict | None = None) -> dict:
    lang = script.get("language", "de")
    n = len(script["scenes"])
    narr = "\n".join(f"{i + 1}. {sc['narration']}" for i, sc in enumerate(script["scenes"]))
    cat = "\n".join(f"- {k}: {v}" for k, v in visual_catalog().items())
    prompt = OVERLAY_PROMPT.format(lang_name={"de": "German", "en": "English"}.get(lang, lang), n=n, script=narr, catalog=cat,
                                   context=(research or {}).get("context") or json.dumps(script.get("claims", []), ensure_ascii=False))
    res, _ = llm.complete_json(prompt, "You design accurate, punchy infographic overlays. JSON only.")
    plans = res.get("scenes", []) if isinstance(res, dict) else []
    used = 0
    for sc, pl in zip(script["scenes"], plans):
        ov = (pl or {}).get("overlay")
        if _valid_overlay(ov):
            sc["overlay"] = {"type": ov["type"], "data": {**ov["data"], "locale": lang}}
            used += 1
        if (pl or {}).get("source_label"):
            sc["source_label"] = str(pl["source_label"])[:40]
        vis = str((pl or {}).get("visual") or "").strip()
        if vis in KINDS:
            sc["visual"] = vis
            prm = (pl or {}).get("params")
            sc["params"] = prm if isinstance(prm, dict) else {}
            hl = sc["params"].get("highlight")
            if hl is not None and not (isinstance(hl, int) and 1 <= hl <= 31):
                sc["params"].pop("highlight")
    if isinstance(res, dict) and res.get("tag"):
        script["tag"] = str(res["tag"])[:26].upper()
    if isinstance(res, dict) and re.fullmatch(r"#[0-9a-fA-F]{6}", str(res.get("accent", ""))):
        script["accent"] = res["accent"]
        for sc in script["scenes"]:
            sc.setdefault("params", {}).setdefault("accent", res["accent"])
    script["overlays_planned"] = used
    return script
