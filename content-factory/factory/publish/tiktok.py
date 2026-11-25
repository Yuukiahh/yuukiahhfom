"""TikTok Content Posting API v2 (official, plain `requests`).

Setup:
  1. developers.tiktok.com -> create an app, add products "Login Kit" and
     "Content Posting API" (enable "Direct Post" if you want direct posting).
  2. Register the redirect URI exactly as used here:
     http://127.0.0.1:<publish.tiktok_redirect_port, default 8766>/callback/
  3. `python -m factory secrets set tiktok_client_key` / `tiktok_client_secret`
  4. `python -m factory publish connect tiktok` (OAuth + PKCE in the browser).

IMPORTANT platform restrictions (we respect them, never circumvent them):
  * Until your app has passed TikTok's audit, ALL content posted by it is
    restricted to private viewing (SELF_ONLY), and only a few users may authorize
    it. Public direct posting requires the app audit.
  * Direct Post must only use a privacy level returned by creator_info for this
    user; we query creator_info before every post and refuse otherwise.
  * Mode "draft" (config `publish.tiktok_mode = "draft"`) uses the inbox upload
    endpoint: the video lands in the creator's TikTok inbox and the creator
    finishes/publishes it inside the TikTok app (scope video.upload only).
  * Analytics (/v2/video/query/) need the `video.list` scope – add it via
    `publish.tiktok_extra_scopes = ["video.list"]` if your app has it.
  * TikTok returns a public post id only for publicly visible posts.
"""
from __future__ import annotations

import json
import math
import time
import urllib.parse
from pathlib import Path

import requests

from .. import secrets
from ..config import cfg
from . import PublishError, PublishRefused, http_error, loopback_authorize, pkce_pair, require_enabled

NAME = "tiktok"
BASE_SCOPES = ["user.info.basic", "video.upload", "video.publish"]
AUTH_URL = "https://www.tiktok.com/v2/auth/authorize/"
API = "https://open.tiktokapis.com"
TOKEN_URL = API + "/v2/oauth/token/"
MIN_CHUNK = 5 * 1024 * 1024
MAX_CHUNK = 64 * 1024 * 1024
DEFAULT_CHUNK = 10 * 1024 * 1024
TIMEOUT = 60
PRIVACY_MAP = {"private": "SELF_ONLY", "unlisted": "SELF_ONLY", "public": "PUBLIC_TO_EVERYONE"}


def scopes() -> list[str]:
    extra = cfg().get("publish.tiktok_extra_scopes", []) or []
    return BASE_SCOPES + [s for s in extra if s not in BASE_SCOPES]


# --------------------------------------------------------------------------- credentials

def _client() -> tuple[str, str]:
    key, sec = secrets.get("tiktok_client_key"), secrets.get("tiktok_client_secret")
    if not key or not sec:
        raise PublishError("Secrets 'tiktok_client_key' / 'tiktok_client_secret' fehlen.")
    return key, sec


def _token() -> dict | None:
    raw = secrets.get("tiktok_oauth_token")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return None


def _save_token(tok: dict) -> None:
    secrets.set("tiktok_oauth_token", json.dumps(tok))


def is_connected() -> bool:
    t = _token()
    return bool(t and t.get("access_token") and secrets.get("tiktok_client_key"))


def granted_scopes() -> set[str]:
    t = _token() or {}
    return {s.strip() for s in str(t.get("scope", "")).split(",") if s.strip()}


def _redirect_port() -> int:
    return int(cfg().get("publish.tiktok_redirect_port", 8766))


def build_auth_url(client_key: str, redirect_uri: str, state: str, challenge: str) -> str:
    return AUTH_URL + "?" + urllib.parse.urlencode({
        "client_key": client_key, "response_type": "code", "scope": ",".join(scopes()),
        "redirect_uri": redirect_uri, "state": state,
        "code_challenge": challenge, "code_challenge_method": "S256"})


def connect(open_browser: bool = True, timeout: float = 300) -> dict:
    key, sec = _client()
    # TikTok's desktop PKCE documents SHA256(verifier) as HEX string.
    verifier, challenge = pkce_pair(hex_challenge=True)
    code, redirect_uri = loopback_authorize(lambda ru, st: build_auth_url(key, ru, st, challenge),
                                            port=_redirect_port(), path="/callback/", timeout=timeout,
                                            open_browser=open_browser)
    r = requests.post(TOKEN_URL, data={"client_key": key, "client_secret": sec, "code": code,
                                       "grant_type": "authorization_code", "redirect_uri": redirect_uri,
                                       "code_verifier": verifier},
                      headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=TIMEOUT)
    tok = _check_token_response(r)
    _save_token(tok)
    return {"connected": True, "scopes": tok.get("scope", "")}


def _check_token_response(r) -> dict:
    if r.status_code != 200:
        raise http_error(r)
    tok = r.json()
    if "access_token" not in tok:
        raise PublishError(f"TikTok Token-Fehler: {tok.get('error')} {tok.get('error_description', '')}")
    now = time.time()
    tok["expires_at"] = now + int(tok.get("expires_in", 86400)) - 60
    tok["refresh_expires_at"] = now + int(tok.get("refresh_expires_in", 0))
    return tok


def disconnect() -> None:
    tok = _token()
    if tok:
        try:
            key, sec = _client()
            requests.post(API + "/v2/oauth/revoke/", data={"client_key": key, "client_secret": sec,
                                                            "token": tok.get("access_token")}, timeout=15)
        except Exception:
            pass
    secrets.delete("tiktok_oauth_token")


def access_token() -> str:
    tok = _token()
    if not tok:
        raise PublishError("TikTok nicht verbunden (python -m factory publish connect tiktok).")
    if tok.get("expires_at", 0) > time.time():
        return tok["access_token"]
    if not tok.get("refresh_token"):
        raise PublishError("TikTok Token abgelaufen – bitte neu verbinden.")
    key, sec = _client()
    r = requests.post(TOKEN_URL, data={"client_key": key, "client_secret": sec, "grant_type": "refresh_token",
                                       "refresh_token": tok["refresh_token"]},
                      headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=TIMEOUT)
    new = _check_token_response(r)
    _save_token(new)
    return new["access_token"]


def _api(path: str, body: dict | None = None, params: dict | None = None) -> dict:
    r = requests.post(API + path, params=params, json=body or {},
                      headers={"Authorization": "Bearer " + access_token(),
                               "Content-Type": "application/json; charset=UTF-8"}, timeout=TIMEOUT)
    try:
        payload = r.json()
    except ValueError:
        raise http_error(r) from None
    err = payload.get("error") or {}
    if r.status_code != 200 or (err.get("code") and err.get("code") != "ok"):
        raise PublishError(f"TikTok {path}: HTTP {r.status_code} {err.get('code')} {err.get('message', '')}")
    return payload.get("data") or {}


# --------------------------------------------------------------------------- posting

def creator_info() -> dict:
    return _api("/v2/post/publish/creator_info/query/")


def chunk_plan(size: int, chunk_size: int = DEFAULT_CHUNK) -> tuple[int, int]:
    """(chunk_size, total_chunk_count) following TikTok's media transfer rules:
    files < 5 MB go in one chunk; chunks 5–64 MB; the last chunk takes the
    remainder (may be up to 128 MB); count = floor(size / chunk_size)."""
    if size <= 0:
        raise PublishError("leere Videodatei")
    if size < MIN_CHUNK:
        return size, 1
    chunk_size = max(MIN_CHUNK, min(MAX_CHUNK, chunk_size))
    if size <= chunk_size:
        return size, 1
    return chunk_size, max(1, math.floor(size / chunk_size))


def resolve_privacy(privacy: str, options: list[str]) -> str:
    want = PRIVACY_MAP.get(privacy, "SELF_ONLY")
    if want not in options:
        if "SELF_ONLY" in options and want != "PUBLIC_TO_EVERYONE":
            return "SELF_ONLY"
        raise PublishRefused(f"Privatsphäre {want} für dieses Konto nicht erlaubt (erlaubt: {options}). "
                             "Ungeprüfte Apps dürfen nur SELF_ONLY posten.")
    return want


def upload(video_path: str, meta: dict, privacy: str = "private", thumbnail: str | None = None,
           mode: str | None = None, chunk_size: int = DEFAULT_CHUNK, poll: bool = True) -> dict:
    """Upload via FILE_UPLOAD. mode 'direct' (Direct Post) or 'draft' (inbox).
    Returns {"id", "url", "publish_id", "status", "note"}. `thumbnail` unused
    (TikTok picks the cover via video_cover_timestamp_ms)."""
    require_enabled()
    mode = mode or cfg().get("publish.tiktok_mode", "direct")
    path = Path(video_path)
    if not path.exists():
        raise PublishError(f"Videodatei fehlt: {path}")
    size = path.stat().st_size
    csize, count = chunk_plan(size, chunk_size)
    source = {"source": "FILE_UPLOAD", "video_size": size, "chunk_size": csize, "total_chunk_count": count}
    note = ""
    if mode == "draft":
        data = _api("/v2/post/publish/inbox/video/init/", {"source_info": source})
        note = "Als Entwurf in die TikTok-Inbox gesendet – in der App fertigstellen."
    else:
        info = creator_info()
        max_dur = info.get("max_video_post_duration_sec")
        if max_dur and meta.get("duration") and float(meta["duration"]) > float(max_dur):
            raise PublishRefused(f"Video länger als erlaubt ({max_dur}s)")
        level = resolve_privacy(privacy, info.get("privacy_level_options") or ["SELF_ONLY"])
        post_info = {
            "title": (meta.get("caption") or meta.get("title") or "")[:2200],
            "privacy_level": level,
            "disable_comment": bool(info.get("comment_disabled", False)),
            "disable_duet": bool(info.get("duet_disabled", False)),
            "disable_stitch": bool(info.get("stitch_disabled", False)),
            "video_cover_timestamp_ms": int(meta.get("cover_ms", 1000)),
            "is_aigc": True,  # official AI-generated content label
        }
        data = _api("/v2/post/publish/video/init/", {"post_info": post_info, "source_info": source})
        if level == "SELF_ONLY":
            note = "Privat (SELF_ONLY) gepostet."
    publish_id, upload_url = data.get("publish_id"), data.get("upload_url")
    if not publish_id or not upload_url:
        raise PublishError("TikTok init ohne publish_id/upload_url")
    _put_chunks(upload_url, path, size, csize, count)
    status = fetch_status(publish_id) if poll else {"status": "PROCESSING_UPLOAD"}
    if poll:
        deadline = time.time() + 600
        while status.get("status") in ("PROCESSING_UPLOAD", "PROCESSING_DOWNLOAD") and time.time() < deadline:
            time.sleep(5)
            status = fetch_status(publish_id)
    if status.get("status") == "FAILED":
        raise PublishError(f"TikTok Verarbeitung fehlgeschlagen: {status.get('fail_reason')}")
    post_ids = status.get("publicaly_available_post_id") or status.get("publicly_available_post_id") or []
    post_id = str(post_ids[0]) if post_ids else None
    url = f"https://www.tiktok.com/video/{post_id}" if post_id else None
    return {"id": post_id or publish_id, "publish_id": publish_id, "url": url,
            "status": status.get("status"), "note": note}


def _put_chunks(upload_url: str, path: Path, size: int, csize: int, count: int) -> None:
    with path.open("rb") as f:
        for i in range(count):
            start = i * csize
            end = size - 1 if i == count - 1 else start + csize - 1
            f.seek(start)
            data = f.read(end - start + 1)
            r = requests.put(upload_url, data=data, timeout=300, headers={
                "Content-Type": "video/mp4", "Content-Length": str(len(data)),
                "Content-Range": f"bytes {start}-{end}/{size}"})
            if r.status_code not in (200, 201, 206):
                raise http_error(r)


def fetch_status(publish_id: str) -> dict:
    return _api("/v2/post/publish/status/fetch/", {"publish_id": publish_id})


# --------------------------------------------------------------------------- analytics

VIDEO_FIELDS = "id,like_count,comment_count,share_count,view_count"


def video_metrics(video_ids: list[str]) -> dict[str, dict]:
    if "video.list" not in granted_scopes():
        raise PublishError("Scope video.list nicht erteilt – TikTok-Statistiken nicht verfügbar.")
    out: dict[str, dict] = {}
    for i in range(0, len(video_ids), 20):  # max 20 ids per request
        data = _api("/v2/video/query/", {"filters": {"video_ids": video_ids[i:i + 20]}},
                    params={"fields": VIDEO_FIELDS})
        out.update(parse_video_query(data))
    return out


def parse_video_query(data: dict) -> dict[str, dict]:
    out = {}
    for v in data.get("videos", []):
        out[str(v.get("id"))] = {"views": v.get("view_count"), "likes": v.get("like_count"),
                                 "comments": v.get("comment_count"), "shares": v.get("share_count")}
    return out
