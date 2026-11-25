"""Configuration loading: config/default.toml overlaid by config/local.toml."""
from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    def __init__(self, data: dict[str, Any]):
        self.data = data

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def path(self, dotted: str) -> Path:
        p = Path(self.get(dotted))
        return p if p.is_absolute() else ROOT / p

    def lang_value(self, dotted: str, lang: str | None = None):
        """Values like tts.voice may be a per-language table."""
        v = self.get(dotted)
        if isinstance(v, dict):
            lang = lang or self.get("general.language", "en")
            return v.get(lang) or next(iter(v.values()))
        return v

    def set_local(self, dotted: str, value: Any) -> None:
        """Persist a change into config/local.toml (used by the dashboard)."""
        local_file = CONFIG_DIR / "local.toml"
        local = tomllib.loads(local_file.read_text("utf-8")) if local_file.exists() else {}
        cur = local
        parts = dotted.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value
        local_file.write_text(_dump_toml(local), "utf-8")
        reload()


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k} = {_toml_value(x)}" for k, x in v.items()) + " }"
    raise TypeError(type(v))


def _dump_toml(d: dict) -> str:
    lines: list[str] = []
    for section, body in d.items():
        if isinstance(body, dict):
            lines.append(f"[{section}]")
            for k, v in body.items():
                lines.append(f"{k} = {_toml_value(v)}")
            lines.append("")
        else:
            lines.insert(0, f"{section} = {_toml_value(body)}")
    return "\n".join(lines)


_cfg: Config | None = None


def load() -> Config:
    data = tomllib.loads((CONFIG_DIR / "default.toml").read_text("utf-8"))
    local = CONFIG_DIR / "local.toml"
    if local.exists():
        data = _merge(data, tomllib.loads(local.read_text("utf-8")))
    extra = os.environ.get("CF_CONFIG")
    if extra and Path(extra).exists():
        data = _merge(data, tomllib.loads(Path(extra).read_text("utf-8")))
    return Config(data)


def reload() -> Config:
    global _cfg
    _cfg = load()
    return _cfg


def cfg() -> Config:
    global _cfg
    if _cfg is None:
        _cfg = load()
    return _cfg
