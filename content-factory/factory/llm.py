"""LLM abstraction with free-first provider order.

claude_cli : `claude -p` (Claude Code headless) – uses the user's existing Claude
             subscription, no per-token API cost. Same approach MoneyPrinterTurbo's
             `claude_code` provider uses.
ollama     : local model on the GPU (free).
anthropic  : paid Claude API – only used when explicitly listed AND a budget > 0.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess

import requests

from . import secrets
from .config import cfg


class LLMError(RuntimeError):
    pass


def _claude_cli(prompt: str, system: str, timeout: float) -> str:
    exe = shutil.which("claude") or shutil.which("claude.cmd")
    if not exe:
        raise LLMError("claude CLI nicht gefunden")
    env = {k: v for k, v in os.environ.items() if not k.startswith("ANTHROPIC_")}  # force subscription auth
    cmd = [exe, "-p", "--output-format", "text", "--append-system-prompt", system, "--tools", ""]
    p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                       timeout=timeout, env=env)
    if p.returncode != 0 or not p.stdout.strip():
        raise LLMError(f"claude CLI Fehler: {p.stderr[-500:]}")
    return p.stdout


def _ollama(prompt: str, system: str, timeout: float) -> str:
    url = cfg().get("llm.ollama_url").rstrip("/") + "/api/chat"
    r = requests.post(url, timeout=timeout, json={
        "model": cfg().get("llm.ollama_model"),
        "stream": False,
        "options": {"temperature": 0.8, "num_ctx": 16384},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
    })
    r.raise_for_status()
    return r.json()["message"]["content"]


def _anthropic(prompt: str, system: str, timeout: float) -> str:
    if float(cfg().get("costs.monthly_budget_eur", 0)) <= 0:
        raise LLMError("Anthropic API ist kostenpflichtig und Budget = 0 -> übersprungen")
    key = secrets.get("anthropic_api_key")
    if not key:
        raise LLMError("anthropic_api_key fehlt")
    r = requests.post("https://api.anthropic.com/v1/messages", timeout=timeout, headers={
        "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json={"model": cfg().get("llm.anthropic_model"), "max_tokens": 4096, "system": system,
              "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    data = r.json()
    usage = data.get("usage", {})
    from . import costs
    costs.record_tokens("anthropic", usage.get("input_tokens", 0), usage.get("output_tokens", 0))
    return "".join(b.get("text", "") for b in data["content"])


PROVIDERS = {"claude_cli": _claude_cli, "ollama": _ollama, "anthropic": _anthropic}


def available() -> dict[str, bool]:
    out = {"claude_cli": bool(shutil.which("claude") or shutil.which("claude.cmd"))}
    try:
        requests.get(cfg().get("llm.ollama_url") + "/api/tags", timeout=2).raise_for_status()
        out["ollama"] = True
    except Exception:
        out["ollama"] = False
    out["anthropic"] = bool(secrets.get("anthropic_api_key")) and float(cfg().get("costs.monthly_budget_eur", 0)) > 0
    return out


def complete(prompt: str, system: str = "You are a helpful assistant.") -> tuple[str, str]:
    errors = []
    timeout = float(cfg().get("llm.timeout_seconds", 300))
    for name in cfg().get("llm.providers", []):
        fn = PROVIDERS.get(name)
        if not fn:
            continue
        try:
            return fn(prompt, system, timeout), name
        except Exception as e:  # try next provider
            errors.append(f"{name}: {e}")
    raise LLMError("Kein LLM-Provider verfügbar: " + " | ".join(errors))


def extract_json(text: str):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise LLMError("Keine JSON-Antwort")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise LLMError("Unvollständiges JSON")


def complete_json(prompt: str, system: str, retries: int = 2):
    last = None
    for _ in range(retries + 1):
        text, provider = complete(prompt, system)
        try:
            return extract_json(text), provider
        except Exception as e:
            last = e
            prompt += "\n\nIMPORTANT: answer with ONE valid JSON object only."
    raise LLMError(f"JSON parse failed: {last}")
