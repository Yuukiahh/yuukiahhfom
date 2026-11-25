"""Publishing: approval, scheduling and gated uploads to YouTube / TikTok / Instagram.

SAFETY GATES (all must hold before ANY upload, checked in `publish_one`):
  1. config `publish.enabled == true`   (only set via `enable(confirm=True)`)
  2. the video's `approval == 'approved'` (manual approval in dashboard/CLI)
  3. the platform account is connected  (OAuth token / access token present)
  4. the platform is listed in `publish.platforms`
The platform modules additionally refuse to upload while `publish.enabled` is false
(defense in depth). Default privacy is "private" (config `publish.default_privacy`).

Only official APIs are used. Tokens are never logged: every message that may reach
the DB / console passes through `redact()`.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
import time
from typing import Any

from .. import db
from ..config import cfg

PLATFORMS = ("youtube", "tiktok", "instagram")
PRIVACY_VALUES = ("private", "unlisted", "public")


class PublishError(RuntimeError):
    """A platform call failed. Message is always redacted."""

    def __init__(self, message: str):
        super().__init__(redact(str(message)))


class PublishRefused(PublishError):
    """A safety gate blocked the upload."""


# --------------------------------------------------------------------------- secrets / redaction

_TOKEN_PATTERNS = [
    re.compile(r"ya29\.[\w\-.]{20,}"),            # Google access token
    re.compile(r"1//[\w\-]{20,}"),                 # Google refresh token
    re.compile(r"\bEAA[A-Za-z0-9]{30,}"),          # Meta / Facebook token
    re.compile(r"\bIG[A-Za-z0-9]{40,}"),           # Instagram Login token
    re.compile(r"\b(?:act|rft)\.[\w\-.!]{20,}"),   # TikTok access / refresh token
    re.compile(r"sk-ant-[\w\-]{20,}"),             # Anthropic key
    re.compile(r"GOCSPX-[\w\-]{10,}"),             # Google OAuth client secret
    re.compile(r"(?i)(access_token|refresh_token|client_secret)=[^&\s\"']+"),
]


def _strings(v: Any) -> list[str]:
    if isinstance(v, str):
        out = [v]
        if v[:1] in "{[":
            try:
                out += _strings(json.loads(v))
            except ValueError:
                pass
        return out
    if isinstance(v, dict):
        return [s for x in v.values() for s in _strings(x)]
    if isinstance(v, list):
        return [s for x in v for s in _strings(x)]
    return []


def known_secret_values(min_len: int = 8) -> list[str]:
    """All configured secret values (and string fields inside JSON secrets such as
    OAuth tokens/client JSON). Used only for redaction / leak guards."""
    from .. import secrets

    vals: set[str] = set()
    for name in secrets.KNOWN:
        try:
            v = secrets.get(name)
        except Exception:
            v = None
        if not v:
            continue
        for s in _strings(v):
            # skip short / harmless values (e.g. "Bearer", URLs of Google endpoints)
            if len(s) >= min_len and not s.startswith("https://") and " " not in s.strip():
                vals.add(s)
    return sorted(vals, key=len, reverse=True)


def redact(text: str, values: list[str] | None = None) -> str:
    if not text:
        return text
    for v in (values if values is not None else known_secret_values()):
        if v in text:
            text = text.replace(v, "***")
    for p in _TOKEN_PATTERNS:
        text = p.sub(lambda m: (m.group(1) + "=***") if m.lastindex else "***", text)
    return text


def _log(msg: str, level: str = "info") -> None:
    db.log_event(redact("[publish] " + msg), level)


# --------------------------------------------------------------------------- platform modules

def _module(platform: str):
    if platform == "youtube":
        from . import youtube as m
    elif platform == "tiktok":
        from . import tiktok as m
    elif platform == "instagram":
        from . import instagram as m
    else:
        raise PublishError(f"unknown platform: {platform}")
    return m


def is_enabled() -> bool:
    return cfg().get("publish.enabled", False) is True


def require_enabled() -> None:
    """Called by every platform upload function (defense in depth)."""
    if not is_enabled():
        raise PublishRefused("Veröffentlichung ist deaktiviert (publish.enabled = false).")


def default_privacy() -> str:
    p = str(cfg().get("publish.default_privacy", "private")).lower()
    return p if p in PRIVACY_VALUES else "private"


def publish_status() -> dict[str, dict]:
    globally = is_enabled()
    active = cfg().get("publish.platforms", []) or []
    out: dict[str, dict] = {}
    for p in PLATFORMS:
        try:
            connected = bool(_module(p).is_connected())
        except Exception:
            connected = False
        enabled = globally and p in active
        notes = []
        if not globally:
            notes.append("Veröffentlichung global AUS (python -m factory publish enable)")
        if p not in active:
            notes.append("nicht in publish.platforms")
        if not connected:
            notes.append("Konto nicht verbunden")
        if p == "tiktok":
            notes.append("ungeprüfte TikTok-Apps dürfen nur privat (SELF_ONLY) posten")
        if p == "instagram":
            notes.append("braucht öffentliche Video-URL (publish.instagram_public_base_url); Reels sind immer öffentlich")
        out[p] = {"connected": connected, "enabled": enabled,
                  "note": "; ".join(notes) if notes else f"bereit (Privatsphäre: {default_privacy()})"}
    return out


def enable(confirm: bool = False) -> dict:
    """Turn publishing on. Requires an explicit confirm=True (CLI asks the user)."""
    if confirm is not True:
        return {"ok": False, "enabled": is_enabled(),
                "message": "Nicht aktiviert: ausdrückliche Bestätigung erforderlich (confirm=True)."}
    cfg().set_local("publish.enabled", True)
    _log("Veröffentlichung wurde vom Nutzer aktiviert", "warning")
    return {"ok": True, "enabled": is_enabled(), "privacy": default_privacy()}


def disable() -> dict:
    cfg().set_local("publish.enabled", False)
    _log("Veröffentlichung deaktiviert")
    return {"ok": True, "enabled": is_enabled()}


# --------------------------------------------------------------------------- approval

def _video(video_id: int) -> dict:
    v = db.row("SELECT * FROM videos WHERE id=?", int(video_id))
    if not v:
        raise PublishError(f"Video {video_id} nicht gefunden")
    return v


def approve(video_id: int) -> dict:
    _video(video_id)
    db.update("videos", int(video_id), approval="approved")
    _log(f"Video {video_id} freigegeben")
    return {"video_id": int(video_id), "approval": "approved"}


def reject(video_id: int) -> dict:
    _video(video_id)
    db.update("videos", int(video_id), approval="rejected")
    with db.tx() as c:  # cancel pending uploads of a rejected video
        c.execute("UPDATE publications SET status='failed', error='video rejected' "
                  "WHERE video_id=? AND status='scheduled'", (int(video_id),))
    _log(f"Video {video_id} abgelehnt")
    return {"video_id": int(video_id), "approval": "rejected"}


# --------------------------------------------------------------------------- scheduling

def schedule(video_id: int, platform: str, when_ts: float) -> int:
    if platform not in PLATFORMS:
        raise PublishError(f"unknown platform: {platform}")
    v = _video(video_id)
    if v.get("approval") == "rejected":
        raise PublishRefused(f"Video {video_id} ist abgelehnt")
    pid = db.insert("publications", video_id=int(video_id), platform=platform, status="scheduled",
                    scheduled_for=float(when_ts))
    _log(f"Video {video_id} für {platform} geplant: {time.strftime('%Y-%m-%d %H:%M', time.localtime(when_ts))}")
    return pid


def due(now: float | None = None) -> list[dict]:
    now = time.time() if now is None else now
    return db.rows("SELECT * FROM publications WHERE status='scheduled' AND scheduled_for <= ? "
                   "ORDER BY scheduled_for", now)


def _tz():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(cfg().get("general.timezone", "Europe/Berlin"))
    except Exception:  # Windows without tzdata -> local time
        return None


def _slot_times(day: _dt.date, tz) -> list[float]:
    out = []
    for s in cfg().get("publish.slots", []) or []:
        try:
            hh, mm = (int(x) for x in str(s).split(":"))
        except ValueError:
            continue
        dt = _dt.datetime.combine(day, _dt.time(hh, mm), tzinfo=tz)
        out.append(dt.timestamp())
    return sorted(out)


def next_free_slot(platform: str, now: float | None = None, min_lead_s: float = 600) -> float | None:
    now = time.time() if now is None else now
    tz = _tz()
    taken = {round(r["scheduled_for"] or 0) for r in db.rows(
        "SELECT scheduled_for FROM publications WHERE platform=? AND status IN ('scheduled','uploading','uploaded')",
        platform)}
    start = _dt.datetime.fromtimestamp(now, tz).date()
    for d in range(60):
        for ts in _slot_times(start + _dt.timedelta(days=d), tz):
            if ts >= now + min_lead_s and round(ts) not in taken:
                return ts
    return None


def auto_schedule(video_id: int, now: float | None = None) -> list[dict]:
    """Assign the next free slot on each platform in publish.platforms.
    Scheduling is harmless: run_due() still enforces every gate."""
    _video(video_id)
    out = []
    for platform in cfg().get("publish.platforms", []) or []:
        if platform not in PLATFORMS:
            continue
        exists = db.row("SELECT id, scheduled_for FROM publications WHERE video_id=? AND platform=? "
                        "AND status IN ('scheduled','uploading','uploaded')", int(video_id), platform)
        if exists:
            out.append({"platform": platform, "publication_id": exists["id"], "when": exists["scheduled_for"],
                        "note": "bereits geplant/hochgeladen"})
            continue
        ts = next_free_slot(platform, now)
        if ts is None:
            out.append({"platform": platform, "note": "keine Slots konfiguriert (publish.slots)"})
            continue
        out.append({"platform": platform, "publication_id": schedule(video_id, platform, ts), "when": ts})
    return out


# --------------------------------------------------------------------------- gates + upload

def check_gates(video: dict, platform: str) -> list[str]:
    """Return a list of reasons why this video may NOT be published (empty = OK)."""
    reasons = []
    if not is_enabled():
        reasons.append("disabled")
    if video.get("approval") != "approved":
        reasons.append("not_approved" if video.get("approval") != "rejected" else "rejected")
    if platform not in (cfg().get("publish.platforms", []) or []):
        reasons.append("platform_not_enabled")
    try:
        connected = bool(_module(platform).is_connected())
    except Exception:
        connected = False
    if not connected:
        reasons.append("not_connected")
    return reasons


def _platform_meta(video: dict, platform: str) -> dict:
    meta = video.get("metadata") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except ValueError:
            meta = {}
    m = dict(meta.get(platform) or {})
    m.setdefault("title", meta.get("title") or video.get("title") or "")
    m.setdefault("hashtags", meta.get("hashtags", []))
    m.setdefault("language", video.get("language"))
    return m


def publish_one(pub: dict) -> dict:
    """Upload a single publication row. Enforces all gates. Updates the row."""
    video = _video(pub["video_id"])
    platform = pub["platform"]
    reasons = check_gates(video, platform)
    if reasons:
        if "rejected" in reasons:
            db.update("publications", pub["id"], status="failed", error="video rejected")
        else:  # keep it scheduled; it may go out once the gate is satisfied
            db.update("publications", pub["id"], error="blockiert: " + ",".join(reasons))
        return {"publication_id": pub["id"], "status": "blocked", "reasons": reasons}
    dup = db.row("SELECT id FROM publications WHERE video_id=? AND platform=? AND status='uploaded' AND id<>?",
                 video["id"], platform, pub["id"])
    if dup:
        db.update("publications", pub["id"], status="failed", error=f"bereits hochgeladen (#{dup['id']})")
        return {"publication_id": pub["id"], "status": "duplicate"}
    db.update("publications", pub["id"], status="uploading", error=None)
    thumb = None
    if video.get("folder"):
        from pathlib import Path
        t = Path(video["folder"]) / "thumbnail.jpg"
        thumb = str(t) if t.exists() else None
    try:
        res = _module(platform).upload(video["path"], _platform_meta(video, platform), default_privacy(),
                                       thumbnail=thumb)
    except Exception as e:  # noqa: BLE001
        err = redact(f"{type(e).__name__}: {e}")[:1000]
        db.update("publications", pub["id"], status="failed", error=err)
        _log(f"Upload {platform} für Video {video['id']} fehlgeschlagen: {err}", "error")
        return {"publication_id": pub["id"], "status": "failed", "error": err}
    db.update("publications", pub["id"], status="uploaded", remote_id=res.get("id"), url=res.get("url"),
              published_at=time.time(), error=redact(res.get("note") or "") or None)
    _log(f"Video {video['id']} auf {platform} hochgeladen ({default_privacy()}): {res.get('url') or res.get('id')}")
    return {"publication_id": pub["id"], "status": "uploaded", **{k: res.get(k) for k in ("id", "url")}}


def run_due(now: float | None = None) -> list[dict]:
    """Publish everything that is due. Never raises for single items."""
    return [publish_one(p) for p in due(now)]


# --------------------------------------------------------------------------- OAuth helpers (shared)

def pkce_pair(hex_challenge: bool = False) -> tuple[str, str]:
    """(code_verifier, code_challenge). RFC 7636 S256 (base64url). TikTok's desktop
    flow documents a hex-encoded SHA-256 challenge instead -> hex_challenge=True."""
    import base64
    import hashlib
    import secrets as _pysecrets

    verifier = _pysecrets.token_urlsafe(64)[:96]
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = digest.hex() if hex_challenge else base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


def loopback_authorize(build_auth_url, port: int = 0, path: str = "/", timeout: float = 300,
                       open_browser: bool = True) -> tuple[str, str]:
    """Run a one-shot local HTTP server on 127.0.0.1, open the consent page in the
    browser and wait for the redirect. `build_auth_url(redirect_uri, state)` returns
    the URL. Returns (code, redirect_uri). Raises PublishError on error/timeout/state mismatch."""
    import http.server
    import secrets as _pysecrets
    import threading
    import urllib.parse
    import webbrowser

    state = _pysecrets.token_urlsafe(24)
    result: dict[str, str] = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            u = urllib.parse.urlparse(self.path)
            q = dict(urllib.parse.parse_qsl(u.query))
            if "code" not in q and "error" not in q:
                self.send_response(404)
                self.end_headers()
                return
            result.update(q)
            ok = "code" in q and q.get("state") == state
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            msg = ("Verbunden! Du kannst dieses Fenster schließen." if ok
                   else "Anmeldung fehlgeschlagen. Bitte erneut versuchen.")
            self.wfile.write(f"<html><body style='font-family:sans-serif'><h2>{msg}</h2></body></html>".encode())

        def log_message(self, *a):  # never print the request line (contains the code)
            pass

    srv = http.server.HTTPServer(("127.0.0.1", port), Handler)
    redirect_uri = f"http://127.0.0.1:{srv.server_address[1]}{path}"
    url = build_auth_url(redirect_uri, state)
    t = threading.Thread(target=lambda: _serve_until(srv, result, timeout), daemon=True)
    t.start()
    print("Browser wird geöffnet. Falls nicht, diese Adresse öffnen:\n" + url)
    if open_browser:
        webbrowser.open(url)
    t.join(timeout + 5)
    srv.server_close()
    if "error" in result:
        raise PublishError(f"OAuth abgelehnt: {result.get('error')} {result.get('error_description', '')}")
    if "code" not in result:
        raise PublishError("OAuth Zeitüberschreitung – kein Code empfangen")
    if result.get("state") != state:
        raise PublishError("OAuth state stimmt nicht überein (möglicher CSRF) – abgebrochen")
    return result["code"], redirect_uri


def _serve_until(srv, result: dict, timeout: float) -> None:
    srv.timeout = 1
    end = time.time() + timeout
    while time.time() < end and "code" not in result and "error" not in result:
        srv.handle_request()


def http_error(resp) -> PublishError:
    """Build a redacted error from a requests.Response (never includes headers)."""
    try:
        body = resp.text[:600]
    except Exception:
        body = ""
    return PublishError(f"HTTP {resp.status_code}: {body}")
