# AutoClipper – vollautomatisches, lokales AI-Video-Clipping

Aus einem langen Video (YouTube-Link oder Datei) werden 5–15 fertige, vertikale Clips mit
animierten Untertiteln, Virality Score, Titeln, Beschreibungen, Hashtags und Posting-Plan –
komplett lokal auf deinem PC (RTX-GPU), ohne kostenpflichtige APIs, ohne Wasserzeichen.

Basis ist [SupoClip](https://github.com/FujiwaraChoki/supoclip) (gepinnt auf geprüften Commit
`29ca74f`), selbst gehostet mit **Whisper** (Transkription mit Wort-Zeitstempeln, CUDA) und
**Ollama** (`gpt-oss:20b`, Momentauswahl), dazu der offizielle **SupoClip-MCP-Server** für
Claude Code. AutoClipper ergänzt Rechteprüfung, technische Qualitätskontrolle mit
automatischer Reparatur, visuelle Review-Bögen, Scoring, Texte und den Export.

## Installation (einmalig, Windows 10/11)

1. Dieses Repository herunterladen (Branch `claude/supoclip-setup-windows-37hmju`), z. B.
   `git clone -b claude/supoclip-setup-windows-37hmju https://github.com/yuukiahh/yuukiahhfom.git`
2. PowerShell (normal, nicht als Admin) öffnen und ausführen:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\yuukiahhfom\autoclipper\install\Install-AutoClipper.ps1
   ```
3. Falls der Installer „WSL2 installiert – Windows neu starten" meldet: neu starten und denselben
   Befehl noch einmal ausführen. Docker Desktop fragt beim ersten Start evtl. nach Zustimmung.

Der Installer ist wiederholbar und macht:
winget-Installation von Git, FFmpeg, uv, Ollama, Docker Desktop (+WSL2) · GPU-/Treiberprüfung
(RTX 50 braucht Treiber ≥ 572) · Ollama-Modell + 32k-Kontext · SupoClip klonen (gepinnt),
`.env` mit Whisper/Ollama/GPU und zufälligen Secrets · GPU für den Worker (Compose-Override) ·
Build + Start · CUDA-Test im Container · lokales Konto + Dynamic Edits · AutoClipper-CLI und
SupoClip-MCP (via `uv tool`) · MCP-Registrierung in Claude Code · Skill nach
`~\.claude\skills\autoclipper` · Block in `~\.claude\CLAUDE.md` · Ordner `Desktop\AI Clips` ·
Selbsttest mit einem selbst erzeugten Sprachvideo.

## Benutzung

In Claude Code einfach einen YouTube-Link schicken. Fertig. Ergebnis:

```
Desktop\AI Clips\2026-11-26_<titel>\
  01_<titel>.mp4 …        fertige Clips in empfohlener Reihenfolge
  REPORT.md               Scores, Original-Zeitstempel, Titel, Beschreibungen, Hashtags, Plan
  posting_plan.csv        Veröffentlichungsplan (Excel-tauglich)
  texts\01_tiktok.txt …   Copy-&-Paste-Texte für TikTok und Shorts
  review\                 Kontaktbögen der Qualitätskontrolle
  _rejected\              aussortierte Clips mit Begründung
```

Ohne Claude: `autoclipper process "<link oder datei>"` (CLI-Hilfe: `autoclipper -h`).
Nach einem Neustart: `~\AutoClipper\bin\Start-AutoClipper.ps1`.

## Rechte & Veröffentlichung

- Creative-Commons-Videos und eigene Kanäle (`own_channels` in `~\AutoClipper\config.json`)
  laufen ohne Rückfrage; bei allem anderen fragt Claude einmal nach, ob du es nutzen darfst.
  Bei CC-BY steht der Pflicht-Credit automatisch in den Texten.
- Es wird nichts veröffentlicht. Upload und Freigabe machst du.
- SupoClip steht unter der *SupoClip Noncommercial Source License*: Selbst-Hosting für eigene
  (auch monetarisierte) Kanäle von Einzel-Creatorn ist erlaubt; Agentur-/Firmennutzung braucht
  eine Genehmigung von Shiori LLC.

## Optional

- **B-Roll:** kostenloser Pexels-API-Key (pexels.com/api) → in `~\AutoClipper\secrets.json` als
  `"pexels_api_key"` eintragen und den Installer erneut ausführen.
- **Weniger VRAM/schneller:** `-OllamaModel qwen3:8b` bzw. `-WhisperModel medium` beim Installer.

## Entwicklung

`toolkit/` ist ein Python-Paket (`autoclipper`), Tests: `cd toolkit && uv run --extra test pytest`
(erzeugt synthetische Videos mit eingebauten Fehlern und testet die Pipeline gegen einen
nachgebauten SupoClip-Server).
