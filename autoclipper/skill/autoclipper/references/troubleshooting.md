# Troubleshooting

All commands run in PowerShell. SupoClip lives in `%USERPROFILE%\AutoClipper\supoclip`.

| Symptom | Fix |
|---|---|
| `supoclip: UNREACHABLE` | `%USERPROFILE%\AutoClipper\bin\Start-AutoClipper.ps1`; then `docker compose ps` in the supoclip folder |
| `ollama: UNREACHABLE` | start "Ollama" from the Start menu; `ollama list` |
| Task error mentioning the LLM / connection refused to host.docker.internal | `docker exec supoclip-worker curl -s http://host.docker.internal:11434/api/version`; if empty set user env `OLLAMA_HOST=0.0.0.0:11434`, restart Ollama (allow private networks only in the firewall prompt) |
| Task error `VIDEO_METADATA_TIMEOUT` / YouTube download fails ("Sign in to confirm", 403) | update yt-dlp: in the supoclip folder `docker compose build --no-cache backend worker; docker compose up -d`; the bgutil PO-token sidecar is already on (profile `youtube-proxy`) |
| Transcription very slow | `docker exec supoclip-worker .venv/bin/python -c "import torch;print(torch.cuda.is_available())"` must print True; otherwise update the NVIDIA driver (RTX 50 needs 572+), restart Docker Desktop |
| `CUDA error: no kernel image` | torch too old for the GPU: rebuild `docker compose build --no-cache worker` |
| Out of GPU memory | set `WHISPER_MODEL_SIZE=medium` in supoclip\.env or a smaller `ollama_model` (e.g. `qwen3:8b`) and `LLM=ollama:qwen3:8b`; `docker compose up -d` |
| `DYNAMIC_EDITS_NOT_AVAILABLE` | rerun the installer (it switches the flag on for the local admin user) |
| `401` from SupoClip | secrets.json and supoclip\.env disagree: rerun the installer |
| No clips produced | video too short/silent or language not detected: set `TRANSCRIPTION_LANGUAGE=de` in supoclip\.env, `docker compose up -d` |
| Clip QA `FACE_CUT` on most clips | rerun with `--output-format vertical_fit` |
| Logs | `docker compose logs --tail 200 worker` (or backend) in the supoclip folder; `%USERPROFILE%\AutoClipper\install.log` |
| Update SupoClip | rerun `Install-AutoClipper.ps1 -SupoClipCommit <newer sha>` after reviewing the changes |
