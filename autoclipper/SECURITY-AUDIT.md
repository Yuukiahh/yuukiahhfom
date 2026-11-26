# Sicherheitsprüfung (2026-11-26)

Geprüft wurde SupoClip auf dem gepinnten Commit `29ca74f9f38f7107bfca1939fd017ae7a0f147a8`
sowie das AutoClipper-Toolkit.

## Abhängigkeiten (bekannte Schwachstellen)

| Komponente | Quelle | Pakete | Befund |
|---|---|---:|---|
| SupoClip Backend (inkl. CUDA-Torch) | `backend/uv.lock`, pip-audit (PyPI-Advisory-DB) | 213 | keine bekannten Schwachstellen |
| SupoClip MCP-Server | `mcp/uv.lock`, pip-audit | 36 | keine bekannten Schwachstellen |
| AutoClipper-Toolkit | installierte Umgebung, pip-audit | alle | keine bekannten Schwachstellen |
| SupoClip Frontend | `frontend/pnpm-lock.yaml`, pnpm audit | – | 1× hoch: `effect` ≤ 3.22.0, ReDoS (GHSA-7589-6fvf-x4pw), nur über `prisma > @prisma/config` |

Bewertung des Frontend-Befunds: `effect` wird nur von Prismas Konfigurations-Loader beim
Build/Start benutzt und verarbeitet keine Eingaben von außen; das Frontend lauscht nur auf
`localhost`. Für eine Ein-Personen-Installation auf dem eigenen PC ist das Risiko gering.
Behoben wird es, sobald SupoClip Prisma aktualisiert (Installer mit neuerem `-SupoClipCommit`
erneut ausführen).

## Container-Images
Offizielle bzw. gepinnte Images: `python:3.11.17-slim-trixie` (Backend/Worker),
`node:22.23.4-bookworm-slim` (Frontend), `python:3.12-slim` (MCP), `postgres:15-alpine`,
`redis:7-alpine`, `brainicism/bgutil-ytdlp-pot-provider:2.0.2` (YouTube-PO-Token-Sidecar,
von SupoClip vorgesehen, Port nicht nach außen veröffentlicht). Das optionale Log-Shipping
(`vector`, Axiom) bleibt aus.

## Konfiguration
- Alle SupoClip-Ports sind an `127.0.0.1` gebunden (Frontend 3107, Backend 8000, Redis 6379, MCP 9100).
  Nichts ist aus dem Netzwerk erreichbar.
- Zufällige 256-Bit-Secrets für Backend-Auth, Better Auth, Settings-Verschlüsselung und Redis;
  `.env` und `secrets.json` nur für deinen Windows-Benutzer lesbar (NTFS-ACL).
- `ALLOW_UNSIGNED_BACKEND_AUTH=false`: MCP und Toolkit signieren jede Anfrage (HMAC-SHA256).
- Telemetrie aus (PostHog/DataFast leer), keine Cloud-KI-Schlüssel gesetzt; alle LLM-Rollen
  (Auswahl, Rerank, Edit-Plan, Schnittkanten, AI-Edit) zeigen auf das lokale Ollama.
- Ollama lauscht nur auf localhost. Nur falls Docker es so nicht erreicht, stellt der
  Installer auf `0.0.0.0` um und weist darauf hin, in der Firewall nur private Netze zu erlauben.
- Installationen über winget (Hash-geprüfte Herstellerinstaller), SupoClip auf geprüftem Commit
  gepinnt, keine `curl | sh`-Pipes außer dem offiziellen Claude-Code-Installer, und den nur,
  falls Claude Code fehlt.

## AutoClipper-Toolkit
- Externe Programme nur als Argumentlisten (kein `shell=True`), keine Ausführung heruntergeladener Inhalte.
- Rechteprüfung vor jedem Download; Credits für CC-BY automatisch in den Texten.
- Keine Upload-/Veröffentlichungsfunktion.
