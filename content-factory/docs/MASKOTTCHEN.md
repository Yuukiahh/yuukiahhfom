# Machbarkeit: eigene animierte 3D-Maskottchen

**Ziel:** eine wiedererkennbare Figur, die im Hook und im Outro auftaucht („Nova“, ein kleiner
schwebender Sonden-Roboter mit Leuchtaugen). Sie passt zu den Nischen Weltraum, KI und Zukunft.

## Optionen

| Weg | Kosten | Qualität | Automatisierbar | Bewertung |
|---|---|---|---|---|
| **A. Blender (lokal, RTX 5070 Ti, Cycles/EEVEE)**: Figur einmal modellieren und riggen, dazu 6–10 Animations-Loops (Winken, Zeigen, Staunen, Idle), die als PNG-Sequenz mit Alpha gerendert werden | 0 € | sehr hoch, konsistent | ja: die Loops werden einmal gerendert und vom Compositor als Overlay eingesetzt | **empfohlen** |
| B. Prozedurale 2D-/2.5D-Figur direkt im Compositor (PIL/numpy) | 0 € | mittel, „Motion-Graphics-Look“ | voll | guter Zwischenschritt |
| C. KI-Videogeneratoren (z. B. Higgsfield, Kling, Veo, Seedance) mit Charakter-Referenz | kostenpflichtig (Credits) | hoch, aber Figur driftet teils zwischen Clips | teilweise | erst nach deiner Kostenfreigabe |
| D. Lokale Video-Diffusion (Wan 2.x, LTX-Video) über ComfyUI auf der 5070 Ti, 16 GB VRAM | 0 € (Strom) | mittel bis hoch, langsam (Minuten pro 3–5 s) | ja, über die ComfyUI-API | sinnvoll für Hintergrund-B-Roll, für eine konsistente Figur weniger geeignet |

## Empfehlung
1. Die Figur in Blender bauen (Low-Poly, Emission-Shader in der Markenfarbe Violett→Cyan, große Augen für Emotion).
2. Die Loops einmal in 1080×1080 mit Alpha rendern und unter `assets/mascot/<emotion>/` ablegen.
3. Der Compositor bekommt eine Overlay-Spur `mascot`: Das Skript-JSON kann pro Szene
   `"mascot": "point_left" | "wow" | "idle"` setzen. Der Renderer legt die passende PNG-Sequenz mit Schatten über die Szene.
4. Optional: eine Lippen-/Augen-Synchronisation auf Wortgrenzen (`words.json`), z. B. Augen-Blinzeln beim Akzentwort.

Für Schritt 1 brauche ich dein Okay zum Design (Name, Farben, Charakter). Die Integration in den
Renderer (Schritt 3) ist vorbereitet: Szenen sind Layer, ein zusätzlicher Overlay-Layer ist eine kleine Erweiterung.
Kostenpflichtige KI-Videodienste (C) nutze ich nicht ohne deine ausdrückliche Freigabe.
