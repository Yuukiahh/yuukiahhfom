"""Cost tracking. Default setup is 100% free; every paid call is recorded."""
from __future__ import annotations

import time

from . import db

# EUR per 1M tokens (approx., update when pricing changes)
TOKEN_PRICES = {"anthropic": (2.8, 14.0)}


def record(service: str, amount_eur: float, units: float = 0, unit_name: str = "", job_id: int | None = None):
    db.insert("costs", service=service, amount_eur=amount_eur, units=units, unit_name=unit_name,
              job_id=job_id, created_at=time.time())


def record_tokens(service: str, inp: int, out: int, job_id: int | None = None):
    pi, po = TOKEN_PRICES.get(service, (0, 0))
    record(service, inp / 1e6 * pi + out / 1e6 * po, inp + out, "tokens", job_id)


def summary() -> dict:
    month_start = time.time() - 30 * 86400
    rows = db.rows("SELECT service, SUM(amount_eur) eur, SUM(units) units FROM costs "
                   "WHERE created_at > ? GROUP BY service", month_start)
    total = sum(r["eur"] or 0 for r in rows)
    from .config import cfg
    return {"last_30_days_eur": round(total, 4), "by_service": rows,
            "budget_eur": cfg().get("costs.monthly_budget_eur", 0),
            "free_services": ["Claude Code CLI (Abo)", "Ollama lokal", "Kokoro TTS lokal", "Edge TTS",
                              "Prozedurale Visuals", "NASA Public Domain", "Pexels/Pixabay (kostenlos)",
                              "YouTube Data API (Quota)", "Google Trends RSS", "Reddit JSON", "Wikipedia"]}
