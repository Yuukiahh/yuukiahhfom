"""ContentFactory dashboard – FastAPI backend.

Runs locally only (127.0.0.1:8765). Serves the single-page app from
dashboard/static/ and a JSON API under /api.

Modules that are developed in parallel (worker, trends engine, publish,
analytics, pipeline.rerender_subtitles) are imported lazily inside the
endpoints, so the dashboard starts even when they are missing; the affected
endpoints answer with 503 (module missing) or 501 (function missing).

Security:
  * bound to 127.0.0.1 only, Host header must be localhost/127.0.0.1
    (protects against DNS rebinding)
  * every mutating request must carry the header ``X-CF-Dashboard: 1``
    (forces a CORS preflight for cross-site requests, which is never granted)
  * secrets can be set but are never returned
  * media files are only served from inside the configured output dir
"""
from __future__ import annotations

import datetime as dt
import importlib
import json
import re
import shutil
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from factory import db
from factory.config import CONFIG_DIR, cfg

HOST = "127.0.0.1"
PORT = 8765
STATIC_DIR = Path(__file__).resolve().parent / "static"
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
CSRF_HEADER = "x-cf-dashboard"

app = FastAPI(title="ContentFactory Dashboard", version="1.0", docs_url="/api/docs", redoc_url=None,
              openapi_url="/api/openapi.json")


# ---------------------------------------------------------------- middleware
@app.middleware("http")
async def _guard(request: Request, call_next):
    host = (request.headers.get("host") or "").rsplit(":", 1)[0].lower()
    if host and host not in ALLOWED_HOSTS:
        return JSONResponse({"error": "Host nicht erlaubt"}, status_code=403)
    if request.method not in ("GET", "HEAD", "OPTIONS") and request.headers.get(CSRF_HEADER) != "1":
        return JSONResponse({"error": f"Header {CSRF_HEADER}: 1 fehlt"}, status_code=403)
    resp = await call_next(request)
    if request.url.path.startswith("/api"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


# ---------------------------------------------------------------- helpers
def _err(status: int, msg: str, **extra: Any) -> HTTPException:
    return HTTPException(status_code=status, detail={"error": msg, **extra})


@app.exception_handler(HTTPException)
async def _http_exc(_request: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {"error": str(exc.detail)}
    return JSONResponse(detail, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def _validation_exc(_request: Request, exc: RequestValidationError):
    problems = [f"{'.'.join(str(x) for x in e.get('loc', []) if x != 'body')}: {e.get('msg')}" for e in exc.errors()]
    return JSONResponse({"error": "Ungültige Eingabe", "problems": problems[:20]}, status_code=422)


def _module(name: str):
    """Import a factory module lazily; 503 if it does not exist (yet) or fails to import."""
    try:
        return importlib.import_module(name)
    except Exception as e:  # ImportError, but also syntax errors in half-written modules
        raise _err(503, f"Modul {name} nicht verfügbar", detail=f"{type(e).__name__}: {e}", module=name)


def _func(module: str, fn: str) -> Callable:
    mod = _module(module)
    f = getattr(mod, fn, None)
    if not callable(f):
        raise _err(501, f"{module}.{fn}() ist noch nicht implementiert", module=module, function=fn)
    return f


def _call(module: str, fn: str, *args: Any, _log: bool = True, **kwargs: Any) -> Any:
    f = _func(module, fn)
    try:
        return f(*args, **kwargs)
    except HTTPException:
        raise
    except ValueError as e:  # domain validation from factory modules, e.g. duplicate topic
        raise _err(409 if "uplikat" in str(e) else 400, str(e), kind="value_error")
    except Exception as e:
        if _log:
            db_log(f"Dashboard: {module}.{fn} fehlgeschlagen: {e}", "error")
        raise _err(500, f"{module}.{fn}() fehlgeschlagen: {e}", trace=traceback.format_exc()[-1500:])


def _try(module: str, fn: str, *args: Any, **kwargs: Any) -> tuple[Any, str | None]:
    """Like _call but never raises: returns (result, error_message)."""
    try:
        return _call(module, fn, *args, _log=False, **kwargs), None
    except HTTPException as e:
        d = e.detail if isinstance(e.detail, dict) else {"error": str(e.detail)}
        return None, d.get("error")


def db_log(message: str, level: str = "info", job_id: int | None = None) -> None:
    try:
        db.log_event(message, level, job_id)
    except Exception:
        pass


def _jsonable(v: Any) -> Any:
    return json.loads(json.dumps(v, default=str))


def _output_root() -> Path:
    return cfg().path("general.output_dir").resolve()


def _safe_path(p: str | Path | None) -> Path:
    """Resolve a path from the DB and make sure it lies inside the output dir."""
    if not p:
        raise _err(404, "Kein Pfad gespeichert")
    path = Path(p).resolve()
    root = _output_root()
    if path != root and root not in path.parents:
        raise _err(403, "Pfad liegt außerhalb des Ausgabeordners")
    return path


def _video(video_id: int) -> dict:
    v = db.row("SELECT * FROM videos WHERE id=?", video_id)
    if not v:
        raise _err(404, f"Video {video_id} nicht gefunden")
    return v


def _folder(v: dict) -> Path:
    if v.get("folder"):
        return _safe_path(v["folder"])
    return _safe_path(Path(v["path"]).parent)


def _today_start() -> float:
    return dt.datetime.combine(dt.date.today(), dt.time()).timestamp()


# background tasks (trend refresh, analytics refresh, subtitle re-render)
_bg_lock = threading.Lock()
_bg: dict[str, dict] = {}


def _run_bg(key: str, module: str, fn: str, *args: Any) -> dict:
    f = _func(module, fn)  # raise 503/501 synchronously
    with _bg_lock:
        cur = _bg.get(key)
        if cur and cur.get("running"):
            return cur

        state = {"running": True, "started_at": time.time(), "finished_at": None, "result": None, "error": None}
        _bg[key] = state

    def work():
        try:
            state["result"] = _jsonable(f(*args))
            db_log(f"Dashboard: {module}.{fn} abgeschlossen")
        except Exception as e:
            state["error"] = f"{type(e).__name__}: {e}"
            db_log(f"Dashboard: {module}.{fn} fehlgeschlagen: {e}", "error")
        finally:
            state["running"] = False
            state["finished_at"] = time.time()

    threading.Thread(target=work, name=f"cf-{key}", daemon=True).start()
    return state


def _bg_state(key: str) -> dict:
    return _bg.get(key) or {"running": False, "started_at": None, "finished_at": None, "result": None, "error": None}


# ---------------------------------------------------------------- worker / state
class StateIn(BaseModel):
    state: str


@app.get("/api/state")
def get_state():
    return _call("factory.worker", "get_state")


@app.post("/api/state")
def set_state(body: StateIn):
    aliases = {"start": "running", "run": "running", "pause": "paused", "stop": "stopped"}
    st = aliases.get(body.state, body.state)
    if st not in ("running", "paused", "stopped"):
        raise _err(400, "state muss running | paused | stopped sein")
    _call("factory.worker", "set_state", st)
    db_log(f"Dashboard: Worker-Status → {st}")
    return _call("factory.worker", "get_state")


@app.get("/api/status")
def status():
    """Overview KPIs. DB-based numbers always work; worker info is added if available."""
    today = _today_start()
    counts = {r["status"]: r["n"] for r in db.rows("SELECT status, COUNT(*) n FROM jobs GROUP BY status")}
    produced_today = (db.row("SELECT COUNT(*) n FROM videos WHERE created_at>=?", today) or {}).get("n", 0)
    qc = db.row("SELECT COUNT(*) n, SUM(CASE WHEN qc_passed=1 THEN 1 ELSE 0 END) ok FROM videos") or {}
    approvals = {r["approval"] or "pending": r["n"] for r in
                 db.rows("SELECT approval, COUNT(*) n FROM videos GROUP BY approval")}
    out: dict[str, Any] = {
        "videos_per_day": cfg().get("schedule.videos_per_day", 4),
        "produced_today": produced_today,
        "queue": counts,
        "videos_total": qc.get("n") or 0,
        "qc_passed": qc.get("ok") or 0,
        "qc_pass_rate": round((qc.get("ok") or 0) / qc["n"], 3) if qc.get("n") else None,
        "approvals": approvals,
        "publish_enabled": bool(cfg().get("publish.enabled", False)),
        "active_hours": cfg().get("schedule.active_hours"),
    }
    costs, cerr = _try("factory.costs", "summary")
    out["costs"] = costs
    w, werr = _try("factory.worker", "status")
    out["worker"] = _jsonable(w) if w is not None else None
    ws, wserr = _try("factory.worker", "get_state")
    out["worker_state"] = _jsonable(ws) if ws is not None else None
    out["errors"] = {k: v for k, v in {"costs": cerr, "worker": werr or wserr}.items() if v}
    return out


# ---------------------------------------------------------------- queue
class EnqueueIn(BaseModel):
    topic: str = Field(min_length=2, max_length=300)
    niche: str | None = None
    format: str = "short"
    language: str | None = None
    preset: str | None = None
    voice: str | None = None
    auto_script: bool = True
    script: dict | None = None      # optional hand-written script (used when auto_script is False)
    force: bool = False             # enqueue even if the topic looks like a duplicate
    options: dict | None = None


@app.get("/api/queue")
def queue(status: str | None = None, limit: int = Query(200, ge=1, le=1000)):
    order = ("CASE status WHEN 'running' THEN 0 WHEN 'queued' THEN 1 WHEN 'review' THEN 2 ELSE 3 END, "
             "CASE WHEN status IN ('running','queued') THEN created_at END ASC, updated_at DESC")
    if status:
        jobs = db.rows(f"SELECT * FROM jobs WHERE status=? ORDER BY {order} LIMIT ?", status, limit)
    else:
        jobs = db.rows(f"SELECT * FROM jobs ORDER BY {order} LIMIT ?", limit)
    return {"jobs": jobs, "now": time.time()}


@app.post("/api/queue")
def enqueue(body: EnqueueIn):
    fmt = body.format if body.format in ("short", "long") else None
    if not fmt:
        raise _err(400, "format muss short oder long sein")
    if body.language and body.language not in ("en", "de"):
        raise _err(400, "language muss en oder de sein")
    opts = dict(body.options or {})
    opts["auto_script"] = body.auto_script
    if not body.auto_script:
        if not body.script or not isinstance(body.script.get("scenes"), list) or not body.script["scenes"]:
            raise _err(400, "Ohne automatisches Skript wird ein Skript-JSON mit 'scenes' benötigt")
        script = dict(body.script)
        script.setdefault("language", body.language or cfg().get("general.language", "en"))
        script.setdefault("format", fmt)
        if body.niche:
            script.setdefault("niche", body.niche)
        opts["script"] = script
    kwargs: dict[str, Any] = {"niche": body.niche or None, "fmt": fmt, "lang": body.language or None,
                              "preset": body.preset or None, "voice": body.voice or None, "options": opts}
    if body.force:
        kwargs["force"] = True
    job_id = _call("factory.worker", "enqueue", body.topic.strip(), **kwargs)
    db_log(f"Dashboard: Job {job_id} eingereiht – {body.topic.strip()}", job_id=job_id)
    return {"job_id": job_id}


@app.post("/api/queue/plan")
def plan_today():
    ids = _call("factory.worker", "plan_today")
    return {"job_ids": ids}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: int):
    _call("factory.worker", "cancel", job_id)
    return {"ok": True, "job": db.row("SELECT * FROM jobs WHERE id=?", job_id)}


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: int):
    _call("factory.worker", "retry", job_id)
    return {"ok": True, "job": db.row("SELECT * FROM jobs WHERE id=?", job_id)}


# ---------------------------------------------------------------- trends
class TopicEnqueueIn(BaseModel):
    format: str = "short"
    language: str | None = None
    force: bool = False


@app.get("/api/trends")
def trends(limit: int = Query(30, ge=1, le=200)):
    out: dict[str, Any] = {"refresh": _bg_state("trends")}
    topics, terr = _try("factory.trends.engine", "top_topics", limit=limit)
    if topics is None:  # fallback: read the topics table directly
        topics = db.rows("SELECT * FROM topics WHERE status='candidate' ORDER BY score DESC LIMIT ?", limit)
    niches, nerr = _try("factory.trends.engine", "niche_report")
    sources, serr = _try("factory.trends.engine", "source_status")
    out.update(topics=_jsonable(topics), niches=_jsonable(niches or []), sources=_jsonable(sources),
               errors={k: v for k, v in {"top_topics": terr, "niche_report": nerr, "source_status": serr}.items() if v})
    last = db.row("SELECT MAX(fetched_at) t, COUNT(*) n FROM trend_signals") or {}
    out["signals"] = {"count": last.get("n") or 0, "last_fetch": last.get("t")}
    return out


@app.post("/api/trends/refresh")
def trends_refresh():
    return _run_bg("trends", "factory.trends.engine", "refresh")


@app.post("/api/topics/{topic_id}/enqueue")
def topic_enqueue(topic_id: int, body: TopicEnqueueIn | None = None):
    t = db.row("SELECT * FROM topics WHERE id=?", topic_id)
    if not t:
        raise _err(404, f"Thema {topic_id} nicht gefunden")
    body = body or TopicEnqueueIn()
    if body.format not in ("short", "long"):
        raise _err(400, "format muss short oder long sein")
    kwargs: dict[str, Any] = {"niche": t.get("niche"), "fmt": body.format, "lang": body.language or None,
                              "options": {"evidence": t.get("evidence"), "auto_script": True}, "topic_id": topic_id}
    if body.force:
        kwargs["force"] = True
    job_id = _call("factory.worker", "enqueue", t["title"], **kwargs)
    db_log(f"Dashboard: Trend-Thema {topic_id} eingereiht als Job {job_id}", job_id=job_id)
    return {"job_id": job_id}


# ---------------------------------------------------------------- catalog
@app.get("/api/catalog")
def catalog():
    """Voices, design presets and defaults for the generator form."""
    out: dict[str, Any] = {"errors": {}}
    try:
        tts = importlib.import_module("factory.tts")
        out["voices"] = getattr(tts, "VOICES", {})
    except Exception as e:
        out["voices"] = {}
        out["errors"]["voices"] = f"factory.tts nicht verfügbar: {e}"
    try:
        render = importlib.import_module("factory.render")
        out["presets"] = list(getattr(render, "XFADE_SETS", {}).keys())
    except Exception as e:
        out["presets"] = ["cinematic", "energetic", "documentary"]
        out["errors"]["presets"] = f"factory.render nicht verfügbar: {e}"
    c = cfg()
    out["engines"] = c.get("tts.engine")
    out["defaults"] = {"language": c.get("general.language", "en"), "preset": c.get("render.preset", "cinematic"),
                       "voice": c.get("tts.voice"), "format_mix": c.get("schedule.format_mix")}
    niches = {r["niche"] for r in db.rows("SELECT DISTINCT niche FROM topics WHERE niche IS NOT NULL")}
    niches |= {r["niche"] for r in db.rows("SELECT DISTINCT niche FROM videos WHERE niche IS NOT NULL")}
    try:
        audio = importlib.import_module("factory.audio")
        niches |= set(getattr(audio, "NICHE_MOOD", {}).keys())
    except Exception:
        pass
    out["niches"] = sorted(n for n in niches if n)
    return out


# ---------------------------------------------------------------- videos
def _video_summary(v: dict) -> dict:
    meta = v.get("metadata") if isinstance(v.get("metadata"), dict) else {}
    qc = v.get("qc") if isinstance(v.get("qc"), dict) else {}
    return {
        "id": v["id"], "job_id": v.get("job_id"), "slug": v.get("slug"), "title": v.get("title"),
        "topic": v.get("topic"), "niche": v.get("niche"), "format": v.get("format"), "language": v.get("language"),
        "duration": v.get("duration"), "hook": v.get("hook"), "qc_passed": bool(v.get("qc_passed")),
        "qc_score": qc.get("score"), "approval": v.get("approval") or "pending", "created_at": v.get("created_at"),
        "hashtags": meta.get("hashtags", []),
        "urls": {"video": f"/media/{v['id']}/video", "thumb": f"/media/{v['id']}/thumb",
                 "contact": f"/media/{v['id']}/contact"},
    }


@app.get("/api/videos")
def videos(limit: int = Query(100, ge=1, le=1000), approval: str | None = None):
    if approval:
        vs = db.rows("SELECT * FROM videos WHERE approval=? ORDER BY created_at DESC LIMIT ?", approval, limit)
    else:
        vs = db.rows("SELECT * FROM videos ORDER BY created_at DESC LIMIT ?", limit)
    pubs: dict[int, list] = {}
    for p in db.rows("SELECT * FROM publications ORDER BY COALESCE(scheduled_for, published_at) DESC"):
        pubs.setdefault(p["video_id"], []).append(p)
    return {"videos": [{**_video_summary(v), "publications": pubs.get(v["id"], [])} for v in vs]}


def _read_text(p: Path, limit: int = 200_000) -> str | None:
    try:
        return p.read_text("utf-8")[:limit] if p.is_file() else None
    except OSError:
        return None


def _read_json(p: Path) -> Any:
    t = _read_text(p, 5_000_000)
    if t is None:
        return None
    try:
        return json.loads(t)
    except ValueError:
        return None


@app.get("/api/videos/{video_id}")
def video_detail(video_id: int):
    v = _video(video_id)
    out = {**_video_summary(v), "metadata": v.get("metadata"), "qc": v.get("qc"), "path": v.get("path"),
           "folder": v.get("folder")}
    try:
        folder = _folder(v)
        script = _read_json(folder / "script.json") or {}
        out["sources_md"] = _read_text(folder / "sources.md")
        out["licenses"] = _read_json(folder / "licenses.json")
        out["claims"] = script.get("claims", [])
        out["research_sources"] = script.get("research_sources", [])
        out["fact_check"] = script.get("fact_check")
        out["files"] = sorted(p.name for p in folder.iterdir() if p.is_file()) if folder.is_dir() else []
    except HTTPException as e:
        out["folder_error"] = e.detail
    out["publications"] = db.rows("SELECT * FROM publications WHERE video_id=? ORDER BY id DESC", video_id)
    out["metrics"] = db.rows("SELECT * FROM metrics WHERE video_id=? ORDER BY fetched_at DESC LIMIT 50", video_id)
    job = db.row("SELECT * FROM jobs WHERE id=?", v["job_id"]) if v.get("job_id") else None
    out["job"] = job
    return out


def _file(path: Path, media_type: str | None = None) -> FileResponse:
    if not path.is_file():
        raise _err(404, f"Datei fehlt: {path.name}")
    return FileResponse(path, media_type=media_type, headers={"Cache-Control": "no-cache"})


@app.get("/media/{video_id}/video")
def media_video(video_id: int):
    v = _video(video_id)
    return _file(_safe_path(v.get("path")), "video/mp4")


@app.get("/media/{video_id}/thumb")
def media_thumb(video_id: int):
    return _file(_folder(_video(video_id)) / "thumbnail.jpg", "image/jpeg")


@app.get("/media/{video_id}/contact")
def media_contact(video_id: int):
    return _file(_folder(_video(video_id)) / "qc_contact_sheet.jpg", "image/jpeg")


# ---------------------------------------------------------------- subtitle editor
class Word(BaseModel):
    text: str = Field(min_length=1, max_length=80)
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    scene: int = Field(0, ge=0)


class WordsIn(BaseModel):
    words: list[Word]


@app.get("/api/videos/{video_id}/words")
def get_words(video_id: int):
    v = _video(video_id)
    folder = _folder(v)
    f = folder / "words.json"
    words = _read_json(f)
    if words is None:
        raise _err(404, "words.json nicht gefunden")
    return {"video_id": video_id, "words": words, "duration": v.get("duration"),
            "has_backup": (folder / "words.orig.json").exists(),
            "rerender": _bg_state(f"rerender:{video_id}")}


@app.put("/api/videos/{video_id}/words")
def put_words(video_id: int, body: WordsIn):
    v = _video(video_id)
    folder = _folder(v)
    f = folder / "words.json"
    if not body.words:
        raise _err(400, "Wortliste ist leer")
    problems = []
    prev_start = -1.0
    dur = float(v.get("duration") or 1e9)
    for i, w in enumerate(body.words):
        if w.end < w.start:
            problems.append(f"Wort {i + 1} ('{w.text}'): Ende vor Start")
        if w.start < prev_start:
            problems.append(f"Wort {i + 1} ('{w.text}'): Start vor dem vorherigen Wort")
        if w.end > dur + 1:
            problems.append(f"Wort {i + 1} ('{w.text}'): Ende nach Videoende")
        prev_start = w.start
    if problems:
        raise _err(422, "Ungültige Zeitangaben", problems=problems[:20])
    backup = folder / "words.orig.json"
    if f.exists() and not backup.exists():
        shutil.copy(f, backup)
    data = [{"text": w.text.strip(), "start": round(w.start, 3), "end": round(w.end, 3), "scene": w.scene}
            for w in body.words]
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
    tmp.replace(f)
    db_log(f"Dashboard: Untertitel von Video {video_id} bearbeitet ({len(data)} Wörter)")
    return {"ok": True, "count": len(data)}


@app.post("/api/videos/{video_id}/words/restore")
def restore_words(video_id: int):
    folder = _folder(_video(video_id))
    backup = folder / "words.orig.json"
    if not backup.exists():
        raise _err(404, "Kein Original-Backup vorhanden")
    shutil.copy(backup, folder / "words.json")
    return {"ok": True}


@app.post("/api/videos/{video_id}/rerender-subtitles")
def rerender_subtitles(video_id: int):
    _video(video_id)
    return _run_bg(f"rerender:{video_id}", "factory.pipeline", "rerender_subtitles", video_id)


@app.get("/api/videos/{video_id}/rerender-subtitles")
def rerender_status(video_id: int):
    return _bg_state(f"rerender:{video_id}")


# ---------------------------------------------------------------- approval / publishing
class ScheduleIn(BaseModel):
    platform: str
    when_ts: float


def _set_approval(video_id: int, fn: str, value: str) -> dict:
    _video(video_id)
    try:
        mod = importlib.import_module("factory.publish")
        f = getattr(mod, fn, None)
    except Exception:
        f = None
    if callable(f):
        try:
            f(video_id)
        except Exception as e:
            raise _err(500, f"factory.publish.{fn}() fehlgeschlagen: {e}")
        via = "publish"
    else:  # approval itself is purely local state – safe fallback
        db.update("videos", video_id, approval=value)
        via = "db"
    db_log(f"Dashboard: Video {video_id} → {value}")
    return {"ok": True, "approval": (db.row("SELECT approval FROM videos WHERE id=?", video_id) or {}).get("approval"),
            "via": via}


@app.post("/api/videos/{video_id}/approve")
def approve(video_id: int):
    return _set_approval(video_id, "approve", "approved")


@app.post("/api/videos/{video_id}/reject")
def reject(video_id: int):
    return _set_approval(video_id, "reject", "rejected")


@app.post("/api/videos/{video_id}/schedule")
def schedule(video_id: int, body: ScheduleIn):
    v = _video(video_id)
    if (v.get("approval") or "pending") != "approved":
        raise _err(409, "Video muss zuerst freigegeben werden")
    if body.when_ts < time.time() - 60:
        raise _err(400, "Zeitpunkt liegt in der Vergangenheit")
    res = _call("factory.publish", "schedule", video_id, body.platform, body.when_ts)
    db_log(f"Dashboard: Video {video_id} für {body.platform} geplant "
           f"({dt.datetime.fromtimestamp(body.when_ts):%d.%m.%Y %H:%M})")
    return {"ok": True, "result": _jsonable(res)}


@app.get("/api/publish")
def publish_overview():
    st, err = _try("factory.publish", "publish_status")
    c = cfg()
    pubs = db.rows("""SELECT p.*, v.title, v.slug FROM publications p LEFT JOIN videos v ON v.id = p.video_id
                      ORDER BY COALESCE(p.scheduled_for, p.published_at) DESC LIMIT 200""")
    return {"platforms": _jsonable(st) if st is not None else None, "error": err,
            "enabled": bool(c.get("publish.enabled", False)),
            "require_manual_approval": bool(c.get("publish.require_manual_approval", True)),
            "default_privacy": c.get("publish.default_privacy", "private"),
            "configured_platforms": c.get("publish.platforms", []), "slots": c.get("publish.slots", []),
            "publications": pubs,
            "approved_unscheduled": db.rows(
                """SELECT id, title, created_at FROM videos WHERE approval='approved' AND id NOT IN
                   (SELECT video_id FROM publications) ORDER BY created_at DESC LIMIT 50""")}


# ---------------------------------------------------------------- analytics / learning / costs
@app.get("/api/analytics")
def analytics():
    summary, err = _try("factory.analytics", "summary")
    metrics = db.rows("""
        SELECT v.id video_id, v.title, v.niche, v.format, m.platform, m.views, m.likes, m.comments, m.shares,
               m.avg_view_pct, m.avg_view_duration, m.subscribers_gained, m.fetched_at
        FROM metrics m JOIN videos v ON v.id = m.video_id
        WHERE m.id IN (SELECT MAX(id) FROM metrics GROUP BY video_id, platform)
        ORDER BY m.views DESC LIMIT 200""")
    return {"summary": _jsonable(summary), "error": err, "metrics": metrics, "refresh": _bg_state("analytics")}


@app.post("/api/analytics/refresh")
def analytics_refresh():
    return _run_bg("analytics", "factory.analytics", "refresh")


@app.get("/api/learnings")
def learnings():
    return {"learnings": db.kv_get("learnings")}


@app.post("/api/learnings/analyse")
def learnings_analyse():
    return {"learnings": _jsonable(_call("factory.learning", "analyse"))}


@app.get("/api/costs")
def costs():
    s = _call("factory.costs", "summary")
    recent = db.rows("SELECT * FROM costs ORDER BY created_at DESC LIMIT 100")
    days = db.rows("""SELECT date(created_at, 'unixepoch', 'localtime') day, SUM(amount_eur) eur
                      FROM costs WHERE created_at > ? GROUP BY day ORDER BY day""", time.time() - 30 * 86400)
    return {**_jsonable(s), "recent": recent, "by_day": days}


# ---------------------------------------------------------------- settings
HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")
SLOT = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


def _v_int(lo: int, hi: int):
    def f(x):
        if isinstance(x, bool) or not isinstance(x, (int, float)) or int(x) != x or not lo <= x <= hi:
            raise ValueError(f"Ganzzahl {lo}–{hi} erwartet")
        return int(x)
    return f


def _v_float(lo: float, hi: float):
    def f(x):
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not lo <= x <= hi:
            raise ValueError(f"Zahl {lo}–{hi} erwartet")
        return float(x)
    return f


def _v_choice(*opts: str):
    def f(x):
        if x not in opts:
            raise ValueError("erlaubt: " + ", ".join(opts))
        return x
    return f


def _v_format_mix(x):
    if not isinstance(x, dict) or set(x) - {"short", "long"}:
        raise ValueError("{short, long} erwartet")
    s, l_ = float(x.get("short", 0)), float(x.get("long", 0))
    if s < 0 or l_ < 0 or s + l_ <= 0:
        raise ValueError("Anteile müssen ≥ 0 sein")
    tot = s + l_
    return {"short": round(s / tot, 3), "long": round(l_ / tot, 3)}


def _v_voice(x):
    if not isinstance(x, dict) or not x or set(x) - {"en", "de"} or not all(isinstance(v, str) and v for v in x.values()):
        raise ValueError("{en: Stimme, de: Stimme} erwartet")
    return {k: v.strip() for k, v in x.items()}


def _v_colors(x):
    if not isinstance(x, list) or not 1 <= len(x) <= 6 or not all(isinstance(c, str) and HEX.match(c) for c in x):
        raise ValueError("1–6 Farben im Format #RRGGBB")
    return [c.upper() for c in x]


def _v_slots(x):
    if not isinstance(x, list) or len(x) > 12 or not all(isinstance(s, str) and SLOT.match(s) for s in x):
        raise ValueError("Liste von Uhrzeiten HH:MM")
    return sorted(set(x))


def _v_hours(x):
    if (not isinstance(x, list) or len(x) != 2 or not all(isinstance(h, int) and 0 <= h <= 24 for h in x)
            or x[0] >= x[1]):
        raise ValueError("[von, bis] in Stunden 0–24")
    return x


def _v_preset(x):
    try:
        presets = list(importlib.import_module("factory.render").XFADE_SETS)
    except Exception:
        presets = ["cinematic", "energetic", "documentary"]
    return _v_choice(*presets)(x)


SETTINGS: dict[str, tuple[str, Callable]] = {
    "schedule.videos_per_day": ("Videos pro Tag", _v_int(0, 20)),
    "schedule.format_mix": ("Format-Mix (short/long)", _v_format_mix),
    "schedule.active_hours": ("Aktive Stunden", _v_hours),
    "general.language": ("Standardsprache", _v_choice("en", "de")),
    "tts.voice": ("TTS-Stimme je Sprache", _v_voice),
    "tts.speed": ("Sprechtempo", _v_float(0.7, 1.5)),
    "render.preset": ("Design-Preset", _v_preset),
    "subtitles.highlight_colors": ("Untertitel-Highlightfarben", _v_colors),
    "subtitles.size": ("Untertitel-Größe (px)", _v_int(40, 200)),
    "subtitles.words_per_group": ("Wörter pro Einblendung", _v_int(1, 6)),
    "subtitles.position_y": ("Untertitel-Position (Anteil Höhe)", _v_float(0.3, 0.9)),
    "publish.slots": ("Veröffentlichungs-Slots", _v_slots),
    "costs.monthly_budget_eur": ("Monatsbudget (EUR)", _v_float(0, 10_000)),
}
READ_ONLY = {"publish.enabled": "Öffentliche Veröffentlichung (nur per CLI: python -m factory publish enable)",
             "publish.require_manual_approval": "Manuelle Freigabe erforderlich",
             "publish.default_privacy": "Standard-Sichtbarkeit"}


@app.get("/api/settings")
def get_settings():
    c = cfg()
    return {"values": {k: c.get(k) for k in SETTINGS}, "labels": {k: v[0] for k, v in SETTINGS.items()},
            "read_only": {k: {"label": lbl, "value": c.get(k)} for k, lbl in READ_ONLY.items()},
            "local_file": str(CONFIG_DIR / "local.toml")}


@app.put("/api/settings")
def put_settings(values: dict[str, Any]):
    unknown = [k for k in values if k not in SETTINGS]
    if unknown:
        raise _err(400, "Nicht änderbare Einstellungen: " + ", ".join(unknown))
    clean, problems = {}, {}
    for k, v in values.items():
        try:
            clean[k] = SETTINGS[k][1](v)
        except (ValueError, TypeError) as e:
            problems[k] = str(e)
    if problems:
        raise _err(422, "Ungültige Werte", problems=problems)
    c = cfg()
    changed = []
    for k, v in clean.items():
        if c.get(k) != v:
            c.set_local(k, v)
            c = cfg()
            changed.append(k)
    if changed:
        db_log("Dashboard: Einstellungen geändert: " + ", ".join(changed))
    return {"ok": True, "changed": changed, "values": {k: cfg().get(k) for k in SETTINGS}}


# ---------------------------------------------------------------- secrets / llm / events
class SecretIn(BaseModel):
    value: str = Field(min_length=1, max_length=20_000)


@app.get("/api/secrets")
def secrets_status():
    sec = _module("factory.secrets")
    try:
        st = sec.status()
    except Exception as e:
        raise _err(500, f"Secret-Speicher nicht lesbar: {e}")
    return {"secrets": [{"name": k, "description": d, "configured": bool(st.get(k)),
                         "automatic": d.startswith("(automatisch)")} for k, d in sec.KNOWN.items()]}


@app.post("/api/secrets/{name}")
def secrets_set(name: str, body: SecretIn):
    sec = _module("factory.secrets")
    if name not in sec.KNOWN:
        raise _err(404, f"Unbekannter Schlüssel {name}")
    try:
        sec.set(name, body.value.strip())
    except Exception as e:
        raise _err(500, f"Speichern fehlgeschlagen: {type(e).__name__}")
    db_log(f"Dashboard: Secret {name} gesetzt")
    return {"ok": True, "name": name, "configured": True}  # never echo the value


@app.delete("/api/secrets/{name}")
def secrets_delete(name: str):
    sec = _module("factory.secrets")
    if name not in sec.KNOWN:
        raise _err(404, f"Unbekannter Schlüssel {name}")
    sec.delete(name)
    db_log(f"Dashboard: Secret {name} gelöscht")
    return {"ok": True, "name": name, "configured": False}


@app.get("/api/llm")
def llm():
    return {"available": _call("factory.llm", "available"), "providers": cfg().get("llm.providers", [])}


@app.get("/api/events")
def events(limit: int = Query(200, ge=1, le=1000), level: str | None = None):
    if level:
        ev = db.rows("SELECT * FROM events WHERE level=? ORDER BY id DESC LIMIT ?", level, limit)
    else:
        ev = db.rows("SELECT * FROM events ORDER BY id DESC LIMIT ?", limit)
    return {"events": ev}


@app.get("/api/health")
def health():
    mods = {}
    for m in ("factory.worker", "factory.trends.engine", "factory.publish", "factory.analytics", "factory.pipeline"):
        try:
            importlib.import_module(m)
            mods[m] = True
        except Exception:
            mods[m] = False
    return {"ok": True, "time": time.time(), "modules": mods}


@app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "DELETE"])
def api_404(rest: str):
    raise _err(404, f"Unbekannter Endpunkt /api/{rest}")


# static SPA last so it never shadows /api or /media
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
