"""Optional backend: hand a finished script to a locally running MoneyPrinterTurbo.

MoneyPrinterTurbo (https://github.com/harry0703/MoneyPrinterTurbo) is installed next to
this project by install/windows/install.ps1. Start its API with
    cd ..\\MoneyPrinterTurbo ; uv run python main.py      (default http://127.0.0.1:8080)
Its LLM can be set to the `claude_code` provider (uses your Claude subscription via the
claude CLI) in its config.toml. Our own engine remains the default because it produces
original visuals, word-level animated subtitles, SFX and QC; MPT is useful as a second
style (stock-footage montage via Pexels/Pixabay).
"""
from __future__ import annotations

import time

import requests

DEFAULT_URL = "http://127.0.0.1:8080"


def available(url: str = DEFAULT_URL) -> bool:
    try:
        return requests.get(url + "/docs", timeout=2).status_code == 200
    except Exception:
        return False


def generate(script: dict, url: str = DEFAULT_URL, voice: str = "en-US-AndrewMultilingualNeural-Male",
             source: str = "pexels", timeout_s: int = 1800, token: str | None = None) -> dict:
    narration = " ".join(sc["narration"] for sc in script["scenes"])
    terms = list(dict.fromkeys(sc.get("stock_query", "") for sc in script["scenes"] if sc.get("stock_query")))
    body = {
        "video_subject": script.get("topic") or script["title"],
        "video_script": narration,
        "video_terms": terms[:8],
        "video_aspect": "9:16",
        "video_source": source,
        "voice_name": voice,
        "voice_rate": 1.05,
        "bgm_type": "random",
        "bgm_volume": 0.2,
        "subtitle_enabled": True,
        "subtitle_position": "bottom",
        "video_count": 1,
        "video_clip_duration": 4,
        "video_language": script.get("language", "en"),
    }
    headers = {"x-api-key": token} if token else {}
    r = requests.post(url + "/api/v1/videos", json=body, headers=headers, timeout=30)
    r.raise_for_status()
    task_id = r.json()["data"]["task_id"]
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        time.sleep(5)
        s = requests.get(f"{url}/api/v1/tasks/{task_id}", headers=headers, timeout=15).json().get("data", {})
        if s.get("state") == 1 and s.get("videos"):
            return {"task_id": task_id, "videos": s["videos"], "backend": "moneyprinterturbo"}
        if s.get("state") == -1:
            raise RuntimeError(f"MoneyPrinterTurbo task failed: {s}")
    raise TimeoutError("MoneyPrinterTurbo task timed out")
