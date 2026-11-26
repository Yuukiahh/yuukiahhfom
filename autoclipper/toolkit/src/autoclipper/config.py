"""Settings, read from ~/AutoClipper/config.json and AUTOCLIPPER_* / SUPOCLIP_* environment variables.

The Windows installer writes config.json (including the resolved Desktop path, which may be
redirected to OneDrive) and secrets.json (backend auth secret and local user id). Nothing here
calls a paid service: SupoClip runs locally with Whisper and Ollama.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

HOME_DIR = Path(os.environ.get("AUTOCLIPPER_HOME") or (Path.home() / "AutoClipper"))


def _default_output_root() -> str:
    return str(Path.home() / "Desktop" / "AI Clips")


@dataclass
class TaskDefaults:
    """What every SupoClip task is created with. Chosen for professional, uncluttered shorts."""

    processing_mode: str = "quality"
    output_format: str = "vertical"          # face-tracked dynamic 9:16 reframing
    add_subtitles: bool = True
    caption_template: str = "highlighter_swipe"  # animated, key word highlighted; falls back to hormozi
    fallback_caption_template: str = "hormozi"
    include_broll: bool = False              # turned on automatically when PEXELS_API_KEY is configured
    cut_long_pauses: bool = True
    remove_filler_words: bool = True
    clip_length: str = "auto"                # 15-60 s
    workflow: str = "highlights"
    intensity: str = "bold"                  # zooms, hits on cuts; edit_style below removes the noisy parts
    edit_style: dict = field(default_factory=lambda: {
        "sfx": True,
        "music": False,
        "zooms": True,
        "shake": False,
        "transitions": True,
        "speed": False,
        "grain": False,
        "progress_bar": False,
        "loop_ending": False,
        "reactions": False,
        "grade": "auto",
    })
    instructions: str = (
        "Only pick moments that are understandable without the rest of the video, "
        "open with a strong hook in the first 1-3 seconds, and end on a complete thought. "
        "Prefer emotional, funny, surprising or controversial moments. Skip intros, outros and sponsor reads."
    )


@dataclass
class Settings:
    api_url: str = "http://localhost:8000"
    user_id: str = ""
    auth_secret: str = ""
    api_key: str = ""
    output_root: str = field(default_factory=_default_output_root)
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "gpt-oss:20b"
    own_channels: list = field(default_factory=list)  # channels you own: no rights question
    min_virality: int = 55                   # clips below this final score are dropped (kept in _rejected)
    min_clips: int = 5
    max_clips: int = 15
    target_lufs: float = -14.0
    true_peak: float = -1.0
    encoder: str = "auto"                    # auto | h264_nvenc | libx264
    pexels_configured: bool = False
    task: TaskDefaults = field(default_factory=TaskDefaults)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["auth_secret"] = "***" if self.auth_secret else ""
        data["api_key"] = "***" if self.api_key else ""
        return data


def _merge(target: Any, values: dict) -> None:
    for key, value in values.items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        if isinstance(current, TaskDefaults) and isinstance(value, dict):
            _merge(current, value)
        elif isinstance(current, dict) and isinstance(value, dict):
            current.update(value)
        else:
            setattr(target, key, value)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {}


def load_settings(home: Path | None = None) -> Settings:
    home = home or HOME_DIR
    settings = Settings()
    _merge(settings, _read_json(home / "config.json"))
    _merge(settings, _read_json(home / "secrets.json"))
    env = {
        "api_url": os.environ.get("SUPOCLIP_API_URL"),
        "user_id": os.environ.get("SUPOCLIP_USER_ID"),
        "auth_secret": os.environ.get("SUPOCLIP_AUTH_SECRET"),
        "api_key": os.environ.get("SUPOCLIP_API_KEY"),
        "output_root": os.environ.get("AUTOCLIPPER_OUTPUT_ROOT"),
        "ollama_url": os.environ.get("AUTOCLIPPER_OLLAMA_URL"),
        "ollama_model": os.environ.get("AUTOCLIPPER_OLLAMA_MODEL"),
        "encoder": os.environ.get("AUTOCLIPPER_ENCODER"),
    }
    _merge(settings, {k: v for k, v in env.items() if v})
    settings.api_url = settings.api_url.rstrip("/")
    if settings.pexels_configured:
        settings.task.include_broll = True
    return settings
