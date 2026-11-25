# Musik-Bibliothek (lizenzierte Tracks)

Standardmäßig erzeugt ContentFactory für jedes Video einen **eigenen, prozedural
komponierten Soundtrack** (`factory/music_gen.py`) – ohne Urheberrechts- oder
Content-ID-Risiko. Wenn du lieber „echte“ Musik verwenden willst, lege
lizenzierte Tracks hier ab (`assets/music/`) oder in `<data_dir>/music/`
(Standard: `data/music/`, wird nicht ins Git eingecheckt).

Die Pipeline nimmt automatisch einen passenden Track, sobald einer für die
Stimmung (Mood) des Videos vorhanden ist; sonst wird weiter prozedural komponiert.
Der Track wird auf die Videolänge gekürzt bzw. mit Crossfades geloopt, ein- und
ausgeblendet und auf -14 LUFS normalisiert.

## Wo bekomme ich legal kostenlose Musik?

| Quelle | Lizenz | Hinweise |
|---|---|---|
| **YouTube Audio-Mediathek** (YouTube Studio → Audio-Mediathek) | YouTube-Lizenz, teils „Namensnennung erforderlich“ | Für YouTube gedacht; für TikTok/Instagram die Nutzungsbedingungen der Mediathek prüfen. |
| **Pixabay Music** (pixabay.com/music) | Pixabay Content License | Kostenlos, auch kommerziell, keine Namensnennung nötig. Einige Tracks sind trotzdem bei Content-ID registriert → Lizenzzertifikat herunterladen und aufbewahren. |
| **Free Music Archive** (freemusicarchive.org) | je Track verschieden | Nur **CC0** oder **CC BY** verwenden. Bei CC BY ist eine **Namensnennung** (Titel, Künstler, Lizenz, Link) in der Videobeschreibung Pflicht. |

**Nicht verwenden:** Lizenzen mit **NC** (non-commercial, z. B. CC BY-NC) – monetarisierte
Kanäle sind kommerziell. Solche Tracks werden automatisch ignoriert. Ebenso keine
Musik aus Spotify, YouTube-Videos, TikTok-Sounds o. Ä. herunterladen.

## Dateiformat

Pro Track eine Audiodatei (`.mp3`, `.wav`, `.ogg`, `.flac`, `.m4a`, `.opus`) und
daneben eine **gleichnamige JSON-Datei** („Sidecar“). Ohne Sidecar wird der Track
nicht verwendet.

```
assets/music/
  deep-space-drift.mp3
  deep-space-drift.json
```

`deep-space-drift.json`:

```json
{
  "title": "Deep Space Drift",
  "artist": "Beispiel-Künstler",
  "license": "CC BY 4.0",
  "source_url": "https://freemusicarchive.org/music/...",
  "moods": ["cosmic", "epic"]
}
```

* `title`, `artist`, `license`, `source_url` sind Pflichtfelder (für die Namensnennung
  und deinen Lizenznachweis; sie landen in `licenses.json` des Videos).
* `moods`: eine oder mehrere von `cosmic`, `mystery`, `pulse`, `epic`, `playful`,
  `lofi`, `inspire` – oder `any` für „passt überall“.
* Bewahre Lizenznachweise (Screenshot/Zertifikat der Download-Seite) auf.
