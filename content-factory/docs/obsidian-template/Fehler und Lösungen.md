<!-- cf:auto:start – dieser Block wird automatisch erzeugt -->
# Fehler und Lösungen

## Bekannte Probleme
| Problem | Ursache | Lösung |
|---|---|---|
| h264_nvenc nicht verfügbar | Alter NVIDIA-Treiber / FFmpeg ohne NVENC | Treiber aktualisieren, Gyan.FFmpeg (full) installieren; Fallback libx264 greift automatisch. |
| onnxruntime-gpu lädt nicht (CUDA) | CUDA/cuDNN-Version passt nicht | `pip install onnxruntime` (CPU) – Kokoro läuft auch auf CPU schnell genug. |
| YouTube-Upload bleibt privat | API-Projekt nicht von Google geprüft | Audit im Google Cloud Projekt beantragen oder manuell in YouTube Studio umstellen. |
| TikTok: nur SELF_ONLY | App nicht von TikTok geprüft (Audit) | App-Audit beantragen; bis dahin privat posten. |
| Instagram: 'needs public video URL' | Meta lädt Videos nur von öffentlicher URL | publish.instagram_public_base_url setzen (öffentliches HTTPS-Hosting des output/-Ordners). |
| Edge TTS Fehler 403 | Netzwerk/Service-Änderung | edge-tts aktualisieren (`pip install -U edge-tts`). |
| Ollama nicht erreichbar | Dienst läuft nicht | `ollama serve` starten bzw. Ollama-App öffnen. |

## Letzte Fehler aus dem Log (max. 50)
| Zeit | Job | Meldung |
|---|---|---|
| 2026-11-25 14:57 | 1 | Job #1 Fehler (Versuch 1/3): RuntimeError: QC fehlgeschlagen: visual_brightness Traceback (most recent call last):   File "/home/user/yuukiahhfom/content-factory/factory/worker.py", line 146, in process     raise RuntimeError("QC fehlgeschlagen: " + ", ".join(c["check"] for c in res["qc"]["checks"]  |
<!-- cf:auto:end -->
