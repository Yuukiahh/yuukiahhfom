"""Obsidian documentation sync.

Writes Markdown notes into <vault>/<obsidian.project_folder>/. Generated content
sits between AUTO markers; anything you write outside the markers is preserved on
the next sync. Without a configured vault, `sync()` exports the same notes to
docs/obsidian-template/ so they can be copied manually.

SECURITY: every note passes `_guard()` – if any configured secret value (or a
token-shaped string) appears, that note is NOT written.
"""
from __future__ import annotations

import datetime as _dt
import importlib.util
import json
import re
import shutil
import time
from pathlib import Path

from . import db
from .config import ROOT, cfg

AUTO_START = "<!-- cf:auto:start – dieser Block wird automatisch erzeugt -->"
AUTO_END = "<!-- cf:auto:end -->"
TEMPLATE_DIR = ROOT / "docs" / "obsidian-template"


class SecretLeakError(RuntimeError):
    pass


# --------------------------------------------------------------------------- guard

def _guard(text: str, values: list[str]) -> None:
    from .publish import _TOKEN_PATTERNS
    for v in values:
        if v and v in text:
            raise SecretLeakError("note contains a configured secret value")
    for p in _TOKEN_PATTERNS:
        if p.search(text):
            raise SecretLeakError(f"note contains a token-like string ({p.pattern[:20]}…)")


# --------------------------------------------------------------------------- helpers

def _fmt_ts(ts) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "–"


def _safe(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|#^\[\]]+', "-", name).strip()[:120] or "video"


def _yaml(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(_yaml(x) for x in v) + "]"
    return json.dumps(str(v), ensure_ascii=False)


def _frontmatter(d: dict) -> str:
    return "---\n" + "\n".join(f"{k}: {_yaml(v)}" for k, v in d.items()) + "\n---\n"


def _q(sql: str, *a) -> list[dict]:
    try:
        return db.rows(sql, *a)
    except Exception:
        return []


def _table(headers: list[str], rows: list[list]) -> str:
    def cell(x):
        return str("–" if x is None or x == "" else x).replace("|", "\\|").replace("\n", " ")
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(cell(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


# --------------------------------------------------------------------------- notes

STATIC_DECISIONS = [
    ("ADR-001", "Prozedurale Original-Visuals als Standard",
     "Eigene, prozedural gerenderte Visuals (plus Public-Domain/NASA und lizenzfreie Stock-APIs) statt "
     "fremder Clips.", "Keine Urheberrechtsprobleme, einzigartiger Look, keine Kosten."),
    ("ADR-002", "Kokoro TTS (lokal) / Edge TTS",
     "Englisch mit Kokoro-ONNX lokal auf der GPU, Deutsch über Edge TTS.", "Kostenlos, hohe Qualität, offline (Kokoro)."),
    ("ADR-003", "Free-first LLM-Reihenfolge claude_cli → ollama",
     "Zuerst Claude Code CLI (Abo), dann lokales Ollama. Claude API nur mit ausdrücklicher Freigabe.",
     "Keine laufenden API-Kosten; Budget-Grenze in costs.monthly_budget_eur."),
    ("ADR-004", "Veröffentlichung standardmäßig AUS",
     "publish.enabled=false, manuelle Freigabe je Video, Standard-Privatsphäre privat.",
     "Kein ungewolltes öffentliches Posten; nur offizielle APIs."),
    ("ADR-005", "Eigene Render-Engine + MoneyPrinterTurbo optional",
     "Eigene FFmpeg/NumPy-Render-Pipeline ist Standard; MoneyPrinterTurbo wird optional als Backend installiert.",
     "Volle Kontrolle über Qualität/Untertitel; MPT als Alternative für Experimente."),
]

KNOWN_ISSUES = [
    ("h264_nvenc nicht verfügbar", "Alter NVIDIA-Treiber / FFmpeg ohne NVENC",
     "Treiber aktualisieren, Gyan.FFmpeg (full) installieren; Fallback libx264 greift automatisch."),
    ("onnxruntime-gpu lädt nicht (CUDA)", "CUDA/cuDNN-Version passt nicht",
     "`pip install onnxruntime` (CPU) – Kokoro läuft auch auf CPU schnell genug."),
    ("YouTube-Upload bleibt privat", "API-Projekt nicht von Google geprüft",
     "Audit im Google Cloud Projekt beantragen oder manuell in YouTube Studio umstellen."),
    ("TikTok: nur SELF_ONLY", "App nicht von TikTok geprüft (Audit)", "App-Audit beantragen; bis dahin privat posten."),
    ("Instagram: 'needs public video URL'", "Meta lädt Videos nur von öffentlicher URL",
     "publish.instagram_public_base_url setzen (öffentliches HTTPS-Hosting des output/-Ordners)."),
    ("Edge TTS Fehler 403", "Netzwerk/Service-Änderung", "edge-tts aktualisieren (`pip install -U edge-tts`)."),
    ("Ollama nicht erreichbar", "Dienst läuft nicht", "`ollama serve` starten bzw. Ollama-App öffnen."),
]


def _status() -> dict:
    from . import secrets as sec
    try:
        from .publish import publish_status
        pub = publish_status()
    except Exception:
        pub = {}
    return {
        "jobs": {r["status"]: r["n"] for r in _q("SELECT status, COUNT(*) n FROM jobs GROUP BY status")},
        "videos": (_q("SELECT COUNT(*) n FROM videos") or [{"n": 0}])[0]["n"],
        "approval": {r["approval"]: r["n"] for r in _q("SELECT approval, COUNT(*) n FROM videos GROUP BY approval")},
        "publications": {r["status"]: r["n"] for r in _q("SELECT status, COUNT(*) n FROM publications GROUP BY status")},
        "metrics": (_q("SELECT COUNT(*) n FROM metrics") or [{"n": 0}])[0]["n"],
        "publish": pub,
        "secrets": sec.status(),  # booleans only
    }


def note_overview(st: dict) -> str:
    pub_rows = [[p, "ja" if v["connected"] else "nein", "ja" if v["enabled"] else "nein", v["note"]]
                for p, v in st["publish"].items()]
    return f"""# AI Content Factory – Übersicht

Stand: {_fmt_ts(time.time())}

## Status
- Videos produziert: **{st['videos']}**
- Jobs: {', '.join(f'{k}: {v}' for k, v in st['jobs'].items()) or 'keine'}
- Freigaben: {', '.join(f'{k}: {v}' for k, v in st['approval'].items()) or 'keine'}
- Veröffentlichungen: {', '.join(f'{k}: {v}' for k, v in st['publications'].items()) or 'keine'}
- Metrik-Datensätze: {st['metrics']}
- Veröffentlichung global: **{'AN' if cfg().get('publish.enabled') else 'AUS'}** (Privatsphäre: {cfg().get('publish.default_privacy', 'private')})

## Plattformen
{_table(['Plattform', 'verbunden', 'aktiv', 'Hinweis'], pub_rows)}

## Notizen
- [[Systemarchitektur]] · [[Installierte Tools]] · [[Entscheidungen]]
- [[Content-Strategie]] · [[Erfolgreiche Formate]] · [[Fortschritt]]
- [[Fehler und Lösungen]] · [[Offene Aufgaben]]
- Videos: Ordner `Videos/`
"""


TOOLS = [
    ("python", "Python 3.12", "Laufzeit"), ("git", "Git", "Versionsverwaltung"),
    ("ffmpeg", "FFmpeg", "Rendern, Encoding (h264_nvenc), Loudness"), ("nvidia-smi", "NVIDIA-Treiber", "GPU"),
    ("claude", "Claude Code CLI", "LLM (Abo, kostenlos)"), ("ollama", "Ollama", "lokales LLM"),
    ("uv", "uv", "Python-Umgebung für MoneyPrinterTurbo"),
]
PACKAGES = ["numpy", "PIL", "soundfile", "requests", "fastapi", "uvicorn", "kokoro_onnx", "onnxruntime",
            "edge_tts", "cryptography", "keyring", "pydantic", "faster_whisper"]


def note_tools() -> str:
    rows = [[label, "✅" if shutil.which(cmd) or (cmd == "python") else "❌", use] for cmd, label, use in TOOLS]
    pk = [[p, "✅" if importlib.util.find_spec(p) else "❌"] for p in PACKAGES]
    models = [[f, "✅" if (ROOT / "models" / f).exists() else "❌"] for f in ("kokoro-v1.0.onnx", "voices-v1.0.bin")]
    mpt = (ROOT.parent / "MoneyPrinterTurbo").exists()
    return f"""# Installierte Tools

Automatisch geprüft auf diesem Rechner ({_fmt_ts(time.time())}). Installation: `install/windows/install.ps1`.

## Programme
{_table(['Tool', 'gefunden', 'Zweck'], rows)}

## Python-Pakete
{_table(['Paket', 'installiert'], pk)}

## Modelle
{_table(['Datei', 'vorhanden'], models)}

## Optional
- MoneyPrinterTurbo (`..\\MoneyPrinterTurbo`): {'✅' if mpt else '❌'}
- Ollama-Modell laut Config: `{cfg().get('llm.ollama_model')}`
"""


def note_architecture() -> str:
    return """# Systemarchitektur

```mermaid
flowchart LR
    subgraph Recherche
        T[Trend-Quellen<br/>Google Trends · Reddit · YouTube · Wikipedia · HN] --> S[Trend-Scoring<br/>+ Lern-Gewichte]
    end
    S --> Q[(Job-Queue<br/>SQLite)]
    Q --> W[Worker]
    W --> R[Recherche + Fakten]
    R --> SC[Skript<br/>claude_cli → ollama]
    SC --> TTS[TTS<br/>Kokoro / Edge]
    SC --> V[Visuals<br/>prozedural · NASA · Pexels · ComfyUI]
    TTS --> SUB[Untertitel<br/>Wort-Timing]
    V --> REN[Render-Engine<br/>FFmpeg h264_nvenc]
    SUB --> REN
    TTS --> REN
    REN --> QC[Qualitätskontrolle]
    QC --> META[Metadaten + Thumbnail]
    META --> APP{Manuelle Freigabe}
    APP -->|approved + publish.enabled + verbunden| PUB[Publisher<br/>YouTube · TikTok · Instagram]
    PUB --> AN[Analytics<br/>offizielle APIs]
    AN --> L[Lern-Engine]
    L --> S
    L --> SC
    DB[(SQLite)] -.-> OBS[Obsidian-Sync]
    D[Dashboard FastAPI] -.-> Q
    D -.-> APP
```

## Komponenten
| Modul | Aufgabe |
|---|---|
| `factory/trends` | Trend-Signale sammeln und bewerten |
| `factory/research.py`, `scriptwriter.py`, `llm.py` | Recherche, Skript, Faktencheck |
| `factory/tts.py`, `audio.py` | Stimme, Musik, SFX, Loudness |
| `factory/visuals`, `render.py`, `subtitles.py` | Bild, Animation, Untertitel, Encoding |
| `factory/qc.py`, `metadata.py` | Qualitätskontrolle, Titel/Beschreibung/Hashtags |
| `factory/publish/*` | Freigabe, Zeitplan, Uploads (offizielle APIs) |
| `factory/analytics.py`, `learning.py` | echte Metriken, Selbstoptimierung |
| `factory/secrets.py` | Schlüssel im Windows-Anmeldeinformationsspeicher |
| `factory/obsidian.py` | diese Dokumentation |
"""


def note_decisions() -> str:
    out = ["# Entscheidungen (ADR-Log)", "", "Append-only. Neue Entscheidungen werden in der DB (kv `decisions`) ergänzt.", ""]
    for num, title, decision, why in STATIC_DECISIONS:
        out += [f"## {num}: {title}", f"- **Entscheidung:** {decision}", f"- **Begründung:** {why}",
                "- **Status:** angenommen", ""]
    for i, d in enumerate(db.kv_get("decisions", []) or [], start=len(STATIC_DECISIONS) + 1):
        if isinstance(d, str):
            d = {"title": d}
        out += [f"## ADR-{i:03d}: {d.get('title', 'Entscheidung')}",
                f"- **Datum:** {d.get('date') or _fmt_ts(d.get('ts'))}",
                f"- **Entscheidung:** {d.get('decision', d.get('title', ''))}",
                f"- **Begründung:** {d.get('reason', '–')}", f"- **Status:** {d.get('status', 'angenommen')}", ""]
    return "\n".join(out)


def _video_notes() -> dict[str, str]:
    notes = {}
    pubs = _q("SELECT * FROM publications ORDER BY id")
    try:
        from .analytics import latest_per_video
        latest = latest_per_video()
    except Exception:
        latest = []
    for v in _q("SELECT * FROM videos ORDER BY id"):
        date = _dt.date.fromtimestamp(v["created_at"]).isoformat()
        vp = [p for p in pubs if p["video_id"] == v["id"]]
        urls = [p["url"] for p in vp if p.get("url")]
        vm = [m for m in latest if m["video_id"] == v["id"]]
        qc = v.get("qc") if isinstance(v.get("qc"), dict) else {}
        fm = {"video_id": v["id"], "title": v.get("title"), "niche": v.get("niche"), "format": v.get("format"),
              "language": v.get("language"), "duration": round(v["duration"], 1) if v.get("duration") else None,
              "qc_score": qc.get("score"), "qc_passed": bool(v.get("qc_passed")), "approval": v.get("approval"),
              "published_urls": urls, "created": _fmt_ts(v["created_at"])}
        for m in vm:
            for k in ("views", "likes", "comments", "shares", "avg_view_pct"):
                fm[f"{m['platform']}_{k}"] = m.get(k)
        body = [f"# {v.get('title') or v.get('slug')}", "", f"- Thema: {v.get('topic')}", f"- Hook: {v.get('hook')}",
                f"- Ordner: `{v.get('folder')}`", ""]
        if vp:
            body += ["## Veröffentlichungen", _table(["Plattform", "Status", "geplant", "veröffentlicht", "URL", "Fehler"],
                                                     [[p["platform"], p["status"], _fmt_ts(p.get("scheduled_for")),
                                                       _fmt_ts(p.get("published_at")), p.get("url"), p.get("error")]
                                                      for p in vp]), ""]
        if vm:
            body += ["## Metriken (neueste, echte API-Werte)",
                     _table(["Plattform", "Views", "Likes", "Kommentare", "Shares", "Ø Dauer s", "Ø %", "Abos+",
                             "Stand"], [[m["platform"], m.get("views"), m.get("likes"), m.get("comments"),
                                        m.get("shares"), m.get("avg_view_duration"), m.get("avg_view_pct"),
                                        m.get("subscribers_gained"), _fmt_ts(m.get("fetched_at"))] for m in vm]), ""]
        else:
            body += ["_Noch keine Metriken._", ""]
        notes[f"Videos/{date} {_safe(v.get('slug') or str(v['id']))}.md"] = _frontmatter(fm) + "\n".join(body)
    return notes


def note_formats() -> str:
    l = db.kv_get("learnings", {}) or {}
    if l.get("status") != "ok":
        return ("# Erfolgreiche Formate\n\nNoch zu wenige echte Daten "
                f"({l.get('videos_with_metrics', 0)}/{l.get('needed', 5)} Videos mit Metriken). "
                "Es werden keine Zahlen geschätzt.\n")
    fm = sorted((l.get("formats") or {}).items(), key=lambda kv: -kv[1]["lift"])
    hk = sorted((l.get("hook_features") or {}).items(), key=lambda kv: -kv[1]["lift"])
    nw = sorted((l.get("niche_weights") or {}).items(), key=lambda kv: -kv[1])
    return f"""# Erfolgreiche Formate

Stand: {_fmt_ts(l.get('updated_at'))} · Basis: {l.get('videos_with_metrics')} Videos mit echten Metriken

## Formate (Lift ggü. Durchschnitt)
{_table(['Format', 'n', 'Lift'], [[k, v['n'], v['lift']] for k, v in fm])}

## Hook-Merkmale
{_table(['Merkmal', 'n', 'Lift'], [[k, v['n'], v['lift']] for k, v in hk])}

## Nischen-Gewichte
{_table(['Nische', 'Gewicht'], [[k, v] for k, v in nw])}

- Empfohlener Long-Anteil: {l.get('recommended_long_share')}
- Beste Upload-Stunden: {', '.join(str(h) for h in l.get('best_hours', [])) or '–'}

## Beste Hooks
""" + "\n".join(f"- {h}" for h in l.get("top_hooks", [])) + "\n"


def note_errors() -> str:
    ev = _q("SELECT * FROM events WHERE level='error' ORDER BY created_at DESC LIMIT 50")
    return f"""# Fehler und Lösungen

## Bekannte Probleme
{_table(['Problem', 'Ursache', 'Lösung'], [list(x) for x in KNOWN_ISSUES])}

## Letzte Fehler aus dem Log (max. 50)
{_table(['Zeit', 'Job', 'Meldung'], [[_fmt_ts(e['created_at']), e.get('job_id'), (e.get('message') or '')[:300]] for e in ev]) if ev else '_Keine Fehler protokolliert._'}
"""


def note_strategy() -> str:
    from . import learning
    hints = learning.prompt_hints() or "_noch keine datenbasierten Hinweise_"
    return f"""# Content-Strategie

- Sprache: `{cfg().get('general.language')}` · Videos pro Tag: {cfg().get('schedule.videos_per_day')}
- Format-Mix: {cfg().get('schedule.format_mix')}
- Aktive Stunden: {cfg().get('schedule.active_hours')}
- Plattformen: {', '.join(cfg().get('publish.platforms', []))} · Upload-Slots: {', '.join(cfg().get('publish.slots', []))}
- Trend-Quellen/Regionen: {', '.join(cfg().get('trends.geos', []))}

## Grundsätze
- Originale, faktengeprüfte Inhalte mit Quellenangaben; keine fremden Clips.
- KI-Inhalte werden auf allen Plattformen als KI-generiert gekennzeichnet.
- Starker Hook in den ersten 1–2 Sekunden, große Wort-Untertitel, Loop-fähiges Ende.
- Entscheidungen basieren nur auf echten Analytics-Daten.

## Hinweise der Lern-Engine
{hints}
"""


def note_progress(st: dict) -> str:
    phases = [("1", "Grundgerüst, Config, DB, Secrets", True), ("2", "Trend-Recherche", True),
              ("3", "Skript + Faktencheck", True), ("4", "TTS, Visuals, Render, Untertitel", True),
              ("5", "QC + Metadaten", True), ("6", "Freigabe + Publishing (gesperrt bis Aktivierung)", True),
              ("7", "Analytics + Lern-Engine", True), ("8", "Obsidian-Doku + Windows-Installer", True)]
    return f"""# Fortschritt

Stand: {_fmt_ts(time.time())}

## Phasen
""" + "\n".join(f"- [{'x' if done else ' '}] Phase {n}: {t}" for n, t, done in phases) + f"""

## Zahlen
- Videos: {st['videos']} · Jobs: {st['jobs']} · Veröffentlichungen: {st['publications']} · Metriken: {st['metrics']}
"""


def note_todo(st: dict) -> str:
    items = []
    if not cfg().get("publish.enabled"):
        items.append("Veröffentlichung ist AUS – bei Bedarf `python -m factory publish enable` (bewusste Entscheidung)")
    for p, v in st["publish"].items():
        if p in (cfg().get("publish.platforms") or []) and not v["connected"]:
            items.append(f"{p} verbinden: `python -m factory publish connect {p}`")
    missing = [k for k, ok in st["secrets"].items() if not ok and not k.endswith("_token")]
    if missing:
        items.append("Optionale Secrets fehlen: " + ", ".join(missing))
    pend = _q("SELECT id, title FROM videos WHERE approval='pending' ORDER BY id")
    for v in pend[:30]:
        items.append(f"Video #{v['id']} prüfen und freigeben/ablehnen: {v.get('title')}")
    for j in _q("SELECT id, topic FROM jobs WHERE status='failed' ORDER BY id DESC LIMIT 10"):
        items.append(f"Fehlgeschlagenen Job #{j['id']} prüfen: {j.get('topic')}")
    if not cfg().get("obsidian.vault_path"):
        items.append("Obsidian-Vault in config/local.toml eintragen (obsidian.vault_path)")
    return "# Offene Aufgaben\n\n" + ("\n".join(f"- [ ] {i}" for i in items) if items else "_Nichts offen._") + "\n"


def build_notes(include_videos: bool = True) -> dict[str, str]:
    st = _status()
    notes = {
        "00 Übersicht.md": note_overview(st),
        "Installierte Tools.md": note_tools(),
        "Systemarchitektur.md": note_architecture(),
        "Entscheidungen.md": note_decisions(),
        "Erfolgreiche Formate.md": note_formats(),
        "Fehler und Lösungen.md": note_errors(),
        "Content-Strategie.md": note_strategy(),
        "Fortschritt.md": note_progress(st),
        "Offene Aufgaben.md": note_todo(st),
    }
    if include_videos:
        notes.update(_video_notes())
    return notes


# --------------------------------------------------------------------------- writing

def _wrap(generated: str) -> tuple[str, str]:
    """Split generated text into (frontmatter, body) and wrap body in markers."""
    fm = ""
    if generated.startswith("---\n"):
        end = generated.index("\n---\n", 4) + 5
        fm, generated = generated[:end], generated[end:]
    return fm, f"{AUTO_START}\n{generated.rstrip()}\n{AUTO_END}\n"


def merge_note(existing: str | None, generated: str) -> str:
    fm, block = _wrap(generated)
    if not existing or AUTO_START not in existing or AUTO_END not in existing:
        if existing:  # unknown user file: keep it below the generated block
            if existing.startswith("---\n") and "\n---\n" in existing[4:]:
                existing = existing[existing.index("\n---\n", 4) + 5:]
            return fm + block + "\n" + existing
        return fm + block
    if existing.startswith("---\n") and "\n---\n" in existing[4:]:
        existing = existing[existing.index("\n---\n", 4) + 5:]
    pre, rest = existing.split(AUTO_START, 1)
    _, post = rest.split(AUTO_END, 1)
    return fm + pre + block.rstrip("\n") + post if post.strip() else fm + pre + block


def write_notes(target: Path, notes: dict[str, str]) -> dict:
    from .publish import known_secret_values
    values = known_secret_values(min_len=6)
    written, refused = [], []
    for rel, text in notes.items():
        p = target / rel
        try:
            _guard(text, values)
            existing = p.read_text("utf-8") if p.exists() else None
            final = merge_note(existing, text)
            _guard(final, values)
        except SecretLeakError as e:
            refused.append(rel)
            db.log_event(f"[obsidian] Notiz '{rel}' NICHT geschrieben: {e}", "error")
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        if existing != final:
            p.write_text(final, "utf-8")
        written.append(rel)
    return {"path": str(target), "written": written, "refused": refused}


def export_template(path: str | Path = TEMPLATE_DIR, include_videos: bool = True) -> dict:
    return write_notes(Path(path), build_notes(include_videos))


def sync() -> dict:
    vault = (cfg().get("obsidian.vault_path") or "").strip()
    if not vault:
        res = export_template(TEMPLATE_DIR)
        res["status"] = "no_vault – Vorlage exportiert"
        return res
    vp = Path(vault).expanduser()
    if not vp.exists():
        return {"status": "error", "message": f"Vault-Pfad existiert nicht: {vp}"}
    res = write_notes(vp / cfg().get("obsidian.project_folder", "AI Content Factory"), build_notes(True))
    res["status"] = "ok"
    return res
