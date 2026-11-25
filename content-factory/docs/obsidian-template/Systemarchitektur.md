<!-- cf:auto:start – dieser Block wird automatisch erzeugt -->
# Systemarchitektur

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
<!-- cf:auto:end -->
