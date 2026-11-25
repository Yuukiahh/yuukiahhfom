<!-- cf:auto:start – dieser Block wird automatisch erzeugt -->
# Entscheidungen (ADR-Log)

Append-only. Neue Entscheidungen werden in der DB (kv `decisions`) ergänzt.

## ADR-001: Prozedurale Original-Visuals als Standard
- **Entscheidung:** Eigene, prozedural gerenderte Visuals (plus Public-Domain/NASA und lizenzfreie Stock-APIs) statt fremder Clips.
- **Begründung:** Keine Urheberrechtsprobleme, einzigartiger Look, keine Kosten.
- **Status:** angenommen

## ADR-002: Kokoro TTS (lokal) / Edge TTS
- **Entscheidung:** Englisch mit Kokoro-ONNX lokal auf der GPU, Deutsch über Edge TTS.
- **Begründung:** Kostenlos, hohe Qualität, offline (Kokoro).
- **Status:** angenommen

## ADR-003: Free-first LLM-Reihenfolge claude_cli → ollama
- **Entscheidung:** Zuerst Claude Code CLI (Abo), dann lokales Ollama. Claude API nur mit ausdrücklicher Freigabe.
- **Begründung:** Keine laufenden API-Kosten; Budget-Grenze in costs.monthly_budget_eur.
- **Status:** angenommen

## ADR-004: Veröffentlichung standardmäßig AUS
- **Entscheidung:** publish.enabled=false, manuelle Freigabe je Video, Standard-Privatsphäre privat.
- **Begründung:** Kein ungewolltes öffentliches Posten; nur offizielle APIs.
- **Status:** angenommen

## ADR-005: Eigene Render-Engine + MoneyPrinterTurbo optional
- **Entscheidung:** Eigene FFmpeg/NumPy-Render-Pipeline ist Standard; MoneyPrinterTurbo wird optional als Backend installiert.
- **Begründung:** Volle Kontrolle über Qualität/Untertitel; MPT als Alternative für Experimente.
- **Status:** angenommen
<!-- cf:auto:end -->
