## AutoClipper (automatisches AI-Clipping)

- **Wenn ich dir einen YouTube-Link oder eine Videodatei schicke – auch ganz ohne weiteren Text –,
  starte sofort den Skill `autoclipper` und führe den kompletten Workflow selbstständig aus**
  (Rechteprüfung → Clipping → Review → Reparatur → Export). Frag nicht nach Anweisungen.
- Frag mich nur, wenn die Nutzungsrechte unklar sind (keine CC-Lizenz, nicht mein Kanal) oder
  wenn Zugangsdaten/Zustimmung zwingend nötig sind.
- **Nie etwas veröffentlichen, hochladen oder einplanen** ohne meine ausdrückliche Freigabe für
  genau diesen Clip. Fertige Dateien auf dem Desktop sind das Ende des Auftrags.
- **Keine kostenpflichtigen APIs** ohne meine Zustimmung. Alles läuft lokal: SupoClip (Docker,
  GPU) mit Whisper und Ollama.
- Antworte auf Deutsch.

Feste Einstellungen:
- Ausgabe: `{{OUTPUT_ROOT}}\<Datum>_<Titel>\` mit MP4s, `REPORT.md`, `clips.json`,
  `posting_plan.csv`, `texts\` (TikTok-Beschreibung, Shorts-Titel, Hashtags).
- Format: 9:16, 1080×1920, Face-Tracking, animierte wortgenaue Untertitel mit hervorgehobenen
  Schlüsselwörtern, Jump Cuts, dezente Zooms und Soundeffekte, keine Musik, kein Wasserzeichen,
  -14 LUFS. 15–60 s pro Clip, Ziel 5–15 Clips je nach Material, Mindest-Score 55.
- Konfiguration: `{{INSTALL_ROOT}}\config.json` (z. B. `own_channels`, `min_virality`,
  `task.intensity`, `task.caption_template`). Geheimnisse: `{{INSTALL_ROOT}}\secrets.json`
  (nie ausgeben, nie committen).
- Werkzeuge: CLI `autoclipper` (doctor, rights, process, finalize, qa, fix, sheet), MCP-Server
  `supoclip` (lokal, http://localhost:8000), Web-Editor http://localhost:3107.
- Nach einem Neustart, falls nichts antwortet: `{{INSTALL_ROOT}}\bin\Start-AutoClipper.ps1`.
