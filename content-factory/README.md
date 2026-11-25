# AI Content Factory

Ein lokales, weitgehend autonomes System für **TikTok, YouTube Shorts und Instagram Reels**. Es
recherchiert Trends, schreibt und prüft Skripte und produziert Premium-Hochformatvideos mit
1080×1920 px und 30 fps. Dazu gehören Wort-für-Wort-Untertitel, eigene Musik und Soundeffekte,
eine Qualitätskontrolle und ein Dashboard. Veröffentlichen kann es auch, das ist aber ausgeschaltet,
bis du es freigibst.

Das System kostet im Standard-Setup **0 €**. Es nutzt dein Claude-Code-Abo (über die `claude` CLI),
lokale Modelle auf deiner RTX 5070 Ti und kostenlose APIs.

```
Trends ─▶ Thema ─▶ Recherche ─▶ Skript+Storyboard ─▶ Faktencheck ─▶ Stimme (+Wort-Timings)
   ─▶ Visuals (eigene Renders / Stock / lokale KI) ─▶ Musik+SFX ─▶ Szenen-Render (parallel)
   ─▶ Schnitt + Übergänge + Untertitel + Mix ─▶ Qualitätskontrolle ─▶ Titel/Beschreibung/Hashtags
   ─▶ Freigabe ─▶ (optional) Veröffentlichung ─▶ Analytics ─▶ Self-Improvement
```

## Schnellstart (Windows 11)

```powershell
# Im Ordner content-factory, PowerShell als normaler Benutzer:
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\install\windows\install.ps1          # Python, FFmpeg, Git, venv, Modelle, MoneyPrinterTurbo
.\.venv\Scripts\Activate.ps1
python -m factory doctor               # Systemcheck: GPU-Encoder, LLMs, TTS, Keys
python -m factory make "What happens if you fall into a black hole?"
python -m factory dashboard            # http://127.0.0.1:8765
.\install\windows\register-autostart.ps1   # Hintergrund-Worker bei Anmeldung starten
```

## Wichtige Befehle

| Befehl | Zweck |
|---|---|
| `python -m factory doctor` | Prüft FFmpeg/NVENC, ONNX-CUDA, Kokoro, Claude CLI, Ollama und Keys |
| `python -m factory trends` | Trends abrufen, Themen bewerten, Top-Liste anzeigen |
| `python -m factory make "<Thema>" --format short\|long --lang en\|de` | Ein Video sofort produzieren |
| `python -m factory make --script scripts/test1_black_hole.json` | Ein fertiges Skript rendern |
| `python -m factory queue "<Thema>"` | Ein Thema in die Warteschlange stellen |
| `python -m factory worker` | Hintergrundprozess (plant 3–5 Videos pro Tag, mit Retries) |
| `python -m factory start\|pause\|stop\|status` | Worker steuern |
| `python -m factory secrets set pexels_api_key` | Key verschlüsselt speichern (Windows-Anmeldeinformationsverwaltung) |
| `python -m factory publish status\|connect youtube\|approve <id>\|enable` | Veröffentlichung |
| `python -m factory analytics` | Echte Kennzahlen holen und Learnings aktualisieren |
| `python -m factory obsidian` | Projektgedächtnis im Obsidian-Vault aktualisieren |

## Komponenten

| Modul | Aufgabe |
|---|---|
| `factory/trends/` | Google Trends RSS, Reddit, YouTube Data API, Wikipedia-Seitenaufrufe, Hacker News, Instagram-Hashtag-API und manueller Import für TikTok-Creative-Center-Exporte. Clustering und Scoring nach Viralität, Aktualität, Nischen-Fit, Bestätigung über mehrere Quellen und gelernten Gewichten. **Keine erfundenen Daten**: Fehlgeschlagene Quellen werden protokolliert. |
| `factory/research.py` | Wikipedia-Recherche als Grundlage für das Skript |
| `factory/scriptwriter.py` | Skript und Storyboard im JSON-Schema, danach ein zweiter Faktencheck-Durchlauf gegen die Quellen |
| `factory/llm.py` | Provider der Reihe nach: **claude CLI (Abo)** → Ollama (lokal) → Anthropic API (nur mit Budget) |
| `factory/tts.py` | Kokoro (lokal, ONNX/GPU, Englisch) und Edge TTS (Deutsch), mit Wort-Zeitstempeln; optional faster-whisper |
| `factory/visuals/procedural.py` | Eigene Original-Visuals: Schwarzes Loch mit Gravitationslinse, Nebel, Planeten, Wurmloch, Synthwave-Grid, Platine, neuronales Netz, Raumsonde, Fusionsplasma, DNA, Nebelwald, Abstrakt |
| `factory/visuals/media.py` | Pexels, Pixabay, NASA (Public Domain) und ComfyUI (lokale SDXL/Flux-Bilder auf der GPU). Lizenzen werden pro Video protokolliert. |
| `factory/audio.py` | Prozedural komponierte Musik in 5 Stimmungen und synthetische Soundeffekte (Whoosh, Impact, Riser, Pop, Glitch, Ding), dadurch keine Content-ID-Konflikte |
| `factory/subtitles.py` | Animierte Wort-für-Wort-Untertitel (ASS): Pop-Animation, farbige Schlüsselwörter, Stat-Karten, Hook-Titel |
| `factory/render.py` | Kamerafahrten (Push, Pull, Pan, Orbit, Drift, Shake), Grading, Vignette, Grain, xfade-Übergänge, Fortschrittsbalken, Ducking, −14 LUFS, **NVENC automatisch** |
| `factory/qc.py` | 22 automatische Prüfungen: Format, fps, Codec, Dauer, Lautheit, True Peak, Stille, Schwarzbilder, Standbilder, Untertitel-Sync, Hook-Timing, Quellen, Lizenzen, Bildvielfalt |
| `factory/worker.py` | Tagesplanung, Warteschlange, Retry mit Backoff, Duplikatschutz, Pause/Stopp, Single-Instance-Lock |
| `factory/publish/` | Offizielle APIs: YouTube Data API, TikTok Content Posting API, Instagram Graph API. **Standardmäßig aus.** |
| `factory/analytics.py`, `factory/learning.py` | Echte Kennzahlen abrufen; Nischen-Gewichte, Formatmix, Hook-Muster und Posting-Zeiten lernen |
| `factory/obsidian.py` | Projektgedächtnis im Obsidian-Vault, mit Schutz gegen das Schreiben von Secrets |
| `factory/mpt_adapter.py` | Optionaler zweiter Render-Stil über eine lokal laufende **MoneyPrinterTurbo**-API |
| `dashboard/` | Lokales Dark-Mode-Dashboard (FastAPI und Vanilla-JS, offline-fähig, nur 127.0.0.1) |

## Ausgabeordner
```
output/2026-11-25/what-happens-if-you-fall-into-a-black-hole/
  what-happens-if-you-fall-into-a-black-hole.mp4   fertiges Video
  thumbnail.jpg  qc_contact_sheet.jpg              Vorschaubild, QC-Übersicht
  script.json  sources.md  licenses.json           Skript, Quellen/Faktencheck, Lizenzen
  metadata.json                                    Titel/Beschreibung/Hashtags je Plattform
  subtitles.srt  subtitles.ass  words.json         Untertitel (editierbar im Dashboard)
  qc.json                                          Prüfprotokoll
```

## Kosten und Sicherheit
* Kostenpflichtige Dienste (Anthropic API, KI-Videogeneratoren) werden nur genutzt, wenn du in `config/local.toml`
  ein Budget setzt **und** sie in der Provider-Liste stehen. Jeder Aufruf wird in der Kostenübersicht erfasst.
* API-Keys und Tokens landen in der Windows-Anmeldeinformationsverwaltung bzw. in einer Fernet-verschlüsselten Datei,
  nie in der Config, nie im Obsidian-Vault, nie in Git (siehe `.gitignore`).
* Öffentliche Veröffentlichung erfordert alles drei: `python -m factory publish enable` (mit Bestätigungsphrase),
  ein verbundenes Konto und die Freigabe des einzelnen Videos. Der Standard-Datenschutzstatus ist „privat“.
  Plattformbeschränkungen werden nicht umgangen. Bei TikTok können ungeprüfte Apps nur privat posten,
  bei Instagram ist ein Business-Konto nötig.
* KI-generierte Inhalte werden gekennzeichnet (YouTube `containsSyntheticMedia`).

## Was du noch tun musst (einmalig)
1. `install.ps1` ausführen.
2. Optional, kostenlos: Keys für Pexels, Pixabay und die YouTube Data API → mehr Bildmaterial und YouTube-Trenddaten.
3. Optional: Ollama und ComfyUI für komplett lokale Skripte und KI-Bilder auf der GPU.
4. Für Uploads: ein Google-OAuth-Client (YouTube), eine TikTok-Developer-App und ein Instagram-Business-Konto.
5. Pfad zum Obsidian-Vault in `config/local.toml` eintragen (`[obsidian] vault_path = "..."`).
