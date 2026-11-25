"""Background worker: plans the day, works the queue, retries failures, publishes (if allowed).

States: running | paused (finishes nothing new, keeps state) | stopped (loop exits).
Single instance enforced with a lock file. Safe to run via Windows Task Scheduler at logon.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import random
import signal
import time
import traceback
from pathlib import Path

from . import db, dedupe
from .config import cfg, reload

POLL = 15


# ------------------------------------------------------------------ state
def get_state() -> dict:
    st = db.kv_get("worker_state", {"state": "stopped"})
    hb = db.kv_get("worker_heartbeat", 0) or 0
    st["alive"] = time.time() - hb < POLL * 4
    st["current_job"] = db.kv_get("worker_current_job")
    return st


def set_state(state: str) -> dict:
    if state not in ("running", "paused", "stopped"):
        raise ValueError(state)
    db.kv_set("worker_state", {"state": state, "changed_at": time.time()})
    db.log_event(f"Worker-Status -> {state}")
    return get_state()


# ------------------------------------------------------------------ queue
def enqueue(topic: str, niche: str | None = None, fmt: str = "short", lang: str | None = None, preset: str | None = None,
            voice: str | None = None, options: dict | None = None, topic_id: int | None = None, force: bool = False) -> int:
    if not force:
        dup, other = dedupe.is_duplicate(topic)
        if dup:
            raise ValueError(f"Duplikat: ähnliches Thema bereits produziert/geplant: {other}")
    if not niche:
        from .trends.engine import classify
        niche = classify(topic)[0] or "facts"
    now = time.time()
    jid = db.insert("jobs", topic=topic, topic_id=topic_id, niche=niche, format=fmt,
                    language=lang or cfg().get("general.language", "en"),
                    preset=preset or cfg().get("render.preset", "cinematic"), voice=voice, status="queued",
                    options=options or {}, created_at=now, updated_at=now, not_before=0)
    if topic_id:
        db.update("topics", topic_id, status="queued")
    db.log_event(f"Job #{jid} eingeplant: {topic} ({fmt})", job_id=jid)
    return jid


def cancel(job_id: int) -> None:
    db.update("jobs", job_id, status="cancelled")


def retry(job_id: int) -> None:
    db.update("jobs", job_id, status="queued", attempts=0, error=None, not_before=0)


def _today_start() -> float:
    d = dt.datetime.combine(dt.date.today(), dt.time())
    return d.timestamp()


def status() -> dict:
    counts = {r["status"]: r["n"] for r in db.rows("SELECT status, COUNT(*) n FROM jobs GROUP BY status")}
    produced = db.row("SELECT COUNT(*) n FROM videos WHERE created_at > ?", _today_start())["n"]
    qc_rate = db.row("SELECT AVG(qc_passed) r, COUNT(*) n FROM videos")
    return {"state": get_state(), "queue": counts, "produced_today": produced,
            "videos_per_day": cfg().get("schedule.videos_per_day"), "qc_pass_rate": qc_rate["r"], "videos_total": qc_rate["n"],
            "active_hours": cfg().get("schedule.active_hours")}


# ------------------------------------------------------------------ planning
def plan_today() -> list[int]:
    """Top up today's jobs to videos_per_day using the best fresh trend topics, diversified by niche."""
    target = int(cfg().get("schedule.videos_per_day", 4))
    planned = db.row("SELECT COUNT(*) n FROM jobs WHERE created_at > ? AND status != 'cancelled'", _today_start())["n"]
    need = target - planned
    if need <= 0:
        return []
    from .trends import engine
    last = (db.kv_get("trend_last_refresh") or {}).get("fetched_at", 0)
    if time.time() - last > 6 * 3600 or not engine.top_topics(1):
        try:
            engine.refresh()
        except Exception as e:
            db.log_event(f"Trend-Refresh fehlgeschlagen: {e}", "error")
    cands = engine.top_topics(40)
    mix = cfg().get("schedule.format_mix", {"short": 0.7, "long": 0.3})
    from . import learning
    lr = db.kv_get("learnings", {}) or {}
    long_share = lr.get("recommended_long_share", mix.get("long", 0.3)) if lr.get("status") == "ok" else mix.get("long", 0.3)
    used_niches: dict[str, int] = {}
    ids = []
    for t in cands:
        if len(ids) >= need:
            break
        if used_niches.get(t["niche"], 0) >= 2:  # diversity
            continue
        fmt = "long" if random.random() < long_share else "short"
        try:
            ids.append(enqueue(t["title"], t["niche"], fmt, topic_id=t["id"], options={"evidence": t.get("evidence")}))
            used_niches[t["niche"]] = used_niches.get(t["niche"], 0) + 1
        except ValueError:
            db.update("topics", t["id"], status="rejected")
    if len(ids) < need:
        db.log_event(f"Nur {len(ids)}/{need} Themen aus Trends verfügbar", "warn")
    return ids


# ------------------------------------------------------------------ execution
def process(job: dict) -> None:
    from . import pipeline

    jid = job["id"]
    db.update("jobs", jid, status="running", attempts=(job["attempts"] or 0) + 1, error=None)
    db.kv_set("worker_current_job", jid)

    def prog(stage, p):
        db.update("jobs", jid, stage=stage, progress=p)
        if get_state()["state"] == "stopped":
            raise KeyboardInterrupt("stopped by user")

    opts = job.get("options") or {}
    try:
        if opts.get("script"):
            script = opts["script"]
        else:
            script = pipeline.prepare_script(job["topic"], job["niche"], job["format"], job["language"], prog)
        script.setdefault("topic", job["topic"])
        script["preset"] = job.get("preset") or script.get("preset")
        if job.get("voice"):
            script["voice"] = job["voice"]
        res = pipeline.produce(script, jid, prog)
        if not res["qc"]["passed"]:
            raise RuntimeError("QC fehlgeschlagen: " + ", ".join(c["check"] for c in res["qc"]["checks"]
                                                                 if not c["ok"] and c["severity"] == "error"))
        db.update("jobs", jid, status="done", progress=1.0, stage="done", video_id=res["video_id"])
        if job.get("topic_id"):
            db.update("topics", job["topic_id"], status="used")
        db.log_event(f"Video fertig: {res['path']} (QC {res['qc']['score']})", job_id=jid)
        _after_video(res["video_id"])
    except KeyboardInterrupt:
        db.update("jobs", jid, status="queued", stage="interrupted")
        raise
    except Exception as e:
        attempts = (job["attempts"] or 0) + 1
        maxr = int(cfg().get("schedule.max_retries", 3))
        err = f"{type(e).__name__}: {e}"
        db.log_event(f"Job #{jid} Fehler (Versuch {attempts}/{maxr}): {err}\n{traceback.format_exc()[-1500:]}", "error", jid)
        if attempts < maxr:
            back = cfg().get("schedule.retry_backoff_seconds", 120) * 2 ** (attempts - 1)
            db.update("jobs", jid, status="queued", error=err, not_before=time.time() + back, stage="retry-wait")
        else:
            db.update("jobs", jid, status="failed", error=err)
    finally:
        db.kv_set("worker_current_job", None)


def _after_video(video_id: int) -> None:
    try:
        from . import publish
        if cfg().get("publish.enabled"):
            publish.auto_schedule(video_id)
    except Exception as e:
        db.log_event(f"Auto-Scheduling übersprungen: {e}", "warn")


def _periodic() -> None:
    now = time.time()
    tasks = [("analytics", 6 * 3600, "factory.analytics", "refresh"), ("obsidian", 12 * 3600, "factory.obsidian", "sync"),
             ("publish", 300, "factory.publish", "run_due")]
    for name, every, mod, fn in tasks:
        k = f"periodic_{name}"
        if now - (db.kv_get(k, 0) or 0) < every:
            continue
        db.kv_set(k, now)
        try:
            import importlib
            getattr(importlib.import_module(mod), fn)()
        except Exception as e:
            db.log_event(f"{name}: {e}", "warn")
    if now - (db.kv_get("periodic_prune", 0) or 0) > 86400:
        db.kv_set("periodic_prune", now)
        from .pipeline import prune_work_dirs
        prune_work_dirs()


def _in_active_hours() -> bool:
    a, b = cfg().get("schedule.active_hours", [0, 24])
    h = dt.datetime.now().hour
    return a <= h < b


def _lock() -> Path:
    lock = cfg().path("general.data_dir") / "worker.lock"
    if lock.exists():
        try:
            pid = int(lock.read_text())
            if pid != os.getpid() and _pid_alive(pid):
                raise SystemExit(f"Worker läuft bereits (PID {pid})")
        except ValueError:
            pass
    lock.write_text(str(os.getpid()))
    return lock


def _pid_alive(pid: int) -> bool:
    try:
        if os.name == "nt":
            import ctypes
            h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if h:
                ctypes.windll.kernel32.CloseHandle(h)
                return True
            return False
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def run_forever(start: bool = True) -> None:
    lock = _lock()
    if start:
        set_state("running")
    # recover jobs interrupted by a crash / shutdown
    with db.tx() as c:
        c.execute("UPDATE jobs SET status='queued', stage='recovered' WHERE status='running'")
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *a: stop.update(flag=True))
    db.log_event("Worker gestartet")
    try:
        while not stop["flag"]:
            db.kv_set("worker_heartbeat", time.time())
            reload()
            st = get_state()["state"]
            if st == "stopped":
                break
            if st == "running" and _in_active_hours():
                try:
                    plan_today()
                except Exception as e:
                    db.log_event(f"Planung fehlgeschlagen: {e}", "error")
                job = db.row("SELECT * FROM jobs WHERE status='queued' AND not_before <= ? ORDER BY id LIMIT 1", time.time())
                if job:
                    try:
                        process(job)
                    except KeyboardInterrupt:
                        break
                    continue
            _periodic()
            time.sleep(POLL)
    finally:
        db.log_event("Worker beendet")
        lock.unlink(missing_ok=True)
