"""YouTube Data API v3 upload + YouTube Analytics (official APIs, plain `requests`).

Setup (once):
  1. Google Cloud Console -> new project -> enable "YouTube Data API v3" and
     "YouTube Analytics API".
  2. OAuth consent screen (External, add yourself as test user) and an OAuth client
     of type "Desktop app". Download the client JSON.
  3. `python -m factory secrets set youtube_oauth_client` and paste the JSON.
  4. `python -m factory publish connect youtube` -> browser consent (loopback +
     PKCE). The token is stored in secret `youtube_oauth_token` (never in config).

Notes:
  * Videos uploaded by API projects that have not passed Google's API audit are
    locked to *private* by YouTube. Private is also our default.
  * `status.containsSyntheticMedia = true` is the official "altered or synthetic
    content" disclosure field; we always set it (AI voice + generated visuals).
  * Custom thumbnails require a verified channel; a failure there is only a note.
"""
from __future__ import annotations

import json
import time
import urllib.parse
from pathlib import Path

import requests

from .. import secrets
from . import PublishError, http_error, loopback_authorize, pkce_pair, require_enabled

NAME = "youtube"
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]
AUTH_URI = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"
API = "https://www.googleapis.com/youtube/v3"
UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status"
THUMB_URL = "https://www.googleapis.com/upload/youtube/v3/thumbnails/set"
ANALYTICS_URL = "https://youtubeanalytics.googleapis.com/v2/reports"
CHUNK = 8 * 1024 * 1024  # multiple of 256 KiB as required by the resumable protocol
TIMEOUT = 60


# --------------------------------------------------------------------------- credentials

def _client() -> dict:
    raw = secrets.get("youtube_oauth_client")
    if not raw:
        raise PublishError("Secret 'youtube_oauth_client' fehlt (Google OAuth Client JSON, Typ Desktop-App).")
    try:
        data = json.loads(raw)
    except ValueError:
        raise PublishError("Secret 'youtube_oauth_client' ist kein gültiges JSON.") from None
    c = data.get("installed") or data.get("web") or data
    if not c.get("client_id"):
        raise PublishError("OAuth Client JSON ohne client_id.")
    return c


def _token() -> dict | None:
    raw = secrets.get("youtube_oauth_token")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return None


def _save_token(tok: dict) -> None:
    secrets.set("youtube_oauth_token", json.dumps(tok))


def is_connected() -> bool:
    t = _token()
    return bool(t and (t.get("refresh_token") or t.get("access_token")) and secrets.get("youtube_oauth_client"))


def build_auth_url(client_id: str, redirect_uri: str, state: str, challenge: str) -> str:
    return AUTH_URI + "?" + urllib.parse.urlencode({
        "client_id": client_id, "redirect_uri": redirect_uri, "response_type": "code",
        "scope": " ".join(SCOPES), "access_type": "offline", "prompt": "consent",
        "include_granted_scopes": "true", "state": state,
        "code_challenge": challenge, "code_challenge_method": "S256",
    })


def connect(open_browser: bool = True, timeout: float = 300) -> dict:
    """Interactive OAuth 2.0 installed-app flow (loopback redirect + PKCE)."""
    c = _client()
    verifier, challenge = pkce_pair()
    code, redirect_uri = loopback_authorize(
        lambda ru, st: build_auth_url(c["client_id"], ru, st, challenge), timeout=timeout, open_browser=open_browser)
    r = requests.post(c.get("token_uri", TOKEN_URI), data={
        "code": code, "client_id": c["client_id"], "client_secret": c.get("client_secret", ""),
        "redirect_uri": redirect_uri, "grant_type": "authorization_code", "code_verifier": verifier,
    }, timeout=TIMEOUT)
    if r.status_code != 200:
        raise http_error(r)
    tok = r.json()
    tok["expires_at"] = time.time() + int(tok.get("expires_in", 3600)) - 60
    _save_token(tok)
    info = {"connected": True, "scopes": tok.get("scope", "")}
    try:
        ch = channel_stats()
        info["channel"] = ch.get("title")
    except Exception:
        pass
    return info


def disconnect() -> None:
    tok = _token()
    if tok:
        try:  # best effort revoke
            requests.post("https://oauth2.googleapis.com/revoke",
                          data={"token": tok.get("refresh_token") or tok.get("access_token")}, timeout=15)
        except requests.RequestException:
            pass
    secrets.delete("youtube_oauth_token")


def access_token() -> str:
    tok = _token()
    if not tok:
        raise PublishError("YouTube nicht verbunden (python -m factory publish connect youtube).")
    if tok.get("access_token") and tok.get("expires_at", 0) > time.time():
        return tok["access_token"]
    if not tok.get("refresh_token"):
        raise PublishError("YouTube Token abgelaufen und kein refresh_token – bitte neu verbinden.")
    c = _client()
    r = requests.post(c.get("token_uri", TOKEN_URI), data={
        "client_id": c["client_id"], "client_secret": c.get("client_secret", ""),
        "refresh_token": tok["refresh_token"], "grant_type": "refresh_token"}, timeout=TIMEOUT)
    if r.status_code != 200:
        raise http_error(r)
    new = r.json()
    tok.update({k: v for k, v in new.items() if k != "refresh_token" or v})
    tok["expires_at"] = time.time() + int(new.get("expires_in", 3600)) - 60
    _save_token(tok)
    return tok["access_token"]


def _auth() -> dict:
    return {"Authorization": "Bearer " + access_token()}


# --------------------------------------------------------------------------- upload

def video_resource(meta: dict, privacy: str) -> dict:
    if privacy not in ("private", "unlisted", "public"):
        privacy = "private"
    snippet = {
        "title": (meta.get("title") or "Untitled")[:100],
        "description": (meta.get("description") or "")[:5000],
        "tags": [t for t in meta.get("tags", meta.get("hashtags", [])) if t][:30],
        "categoryId": str(meta.get("categoryId") or "27"),
    }
    if meta.get("language"):
        snippet["defaultLanguage"] = snippet["defaultAudioLanguage"] = meta["language"]
    return {"snippet": snippet,
            "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False,
                       "containsSyntheticMedia": True, "embeddable": True}}


def upload(video_path: str, meta: dict, privacy: str = "private", thumbnail: str | None = None,
           chunk_size: int = CHUNK, max_retries: int = 5) -> dict:
    """Resumable upload. Returns {"id", "url", "privacy", "note"}."""
    require_enabled()
    path = Path(video_path)
    if not path.exists():
        raise PublishError(f"Videodatei fehlt: {path}")
    total = path.stat().st_size
    body = video_resource(meta, privacy)
    r = requests.post(UPLOAD_URL, headers={**_auth(), "Content-Type": "application/json; charset=UTF-8",
                                           "X-Upload-Content-Length": str(total),
                                           "X-Upload-Content-Type": "video/mp4"},
                      data=json.dumps(body), timeout=TIMEOUT)
    if r.status_code != 200 or not r.headers.get("Location"):
        raise http_error(r)
    session = r.headers["Location"]
    result = _send_chunks(session, path, total, chunk_size, max_retries)
    vid = result.get("id")
    if not vid:
        raise PublishError("Upload ohne Video-ID beendet")
    note = ""
    actual = (result.get("status") or {}).get("privacyStatus")
    if actual and actual != body["status"]["privacyStatus"]:
        note = f"YouTube hat Privatsphäre auf '{actual}' gesetzt (ungeprüftes API-Projekt?)"
    if thumbnail:
        try:
            set_thumbnail(vid, thumbnail)
        except PublishError as e:
            note = (note + "; " if note else "") + f"Thumbnail nicht gesetzt: {e}"
    return {"id": vid, "url": f"https://www.youtube.com/shorts/{vid}", "privacy": actual or privacy, "note": note}


def _send_chunks(session: str, path: Path, total: int, chunk_size: int, max_retries: int) -> dict:
    offset, retries = 0, 0
    with path.open("rb") as f:
        while True:
            f.seek(offset)
            data = f.read(chunk_size)
            end = offset + len(data) - 1
            headers = {**_auth(), "Content-Length": str(len(data)), "Content-Type": "video/mp4"}
            if total:
                headers["Content-Range"] = f"bytes {offset}-{end}/{total}"
            try:
                r = requests.put(session, headers=headers, data=data, timeout=300)
            except requests.RequestException as e:
                r, err = None, e
            else:
                err = None
            if r is not None and r.status_code in (200, 201):
                return r.json()
            if r is not None and r.status_code == 308:
                rng = r.headers.get("Range")  # "bytes=0-12345"
                offset = int(rng.split("-")[-1]) + 1 if rng else 0
                retries = 0
                continue
            if r is not None and r.status_code not in (500, 502, 503, 504):
                raise http_error(r)
            retries += 1
            if retries > max_retries:
                raise PublishError(f"Upload abgebrochen nach {max_retries} Versuchen: {err or r.status_code}")
            time.sleep(min(60, 2 ** retries))
            offset = _query_offset(session, total)


def _query_offset(session: str, total: int) -> int:
    try:
        r = requests.put(session, headers={**_auth(), "Content-Length": "0", "Content-Range": f"bytes */{total}"},
                         timeout=TIMEOUT)
    except requests.RequestException:
        return 0
    if r.status_code == 308 and r.headers.get("Range"):
        return int(r.headers["Range"].split("-")[-1]) + 1
    return 0


def set_thumbnail(video_id: str, image_path: str) -> None:
    p = Path(image_path)
    ctype = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    r = requests.post(THUMB_URL, params={"videoId": video_id},
                      headers={**_auth(), "Content-Type": ctype}, data=p.read_bytes(), timeout=TIMEOUT)
    if r.status_code != 200:
        raise http_error(r)


# --------------------------------------------------------------------------- stats / analytics

def channel_stats() -> dict:
    r = requests.get(f"{API}/channels", params={"part": "snippet,statistics", "mine": "true"},
                     headers=_auth(), timeout=TIMEOUT)
    if r.status_code != 200:
        raise http_error(r)
    items = r.json().get("items", [])
    if not items:
        return {}
    it = items[0]
    st = it.get("statistics", {})
    return {"id": it.get("id"), "title": it.get("snippet", {}).get("title"),
            "subscribers": _int(st.get("subscriberCount")), "views": _int(st.get("viewCount")),
            "videos": _int(st.get("videoCount")), "hidden_subscribers": st.get("hiddenSubscriberCount", False)}


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def video_statistics(video_ids: list[str]) -> dict[str, dict]:
    """videos.list part=statistics, 50 ids per call. Missing fields stay None."""
    out: dict[str, dict] = {}
    for i in range(0, len(video_ids), 50):
        r = requests.get(f"{API}/videos", params={"part": "statistics", "id": ",".join(video_ids[i:i + 50])},
                         headers=_auth(), timeout=TIMEOUT)
        if r.status_code != 200:
            raise http_error(r)
        out.update(parse_video_statistics(r.json()))
    return out


def parse_video_statistics(payload: dict) -> dict[str, dict]:
    out = {}
    for it in payload.get("items", []):
        st = it.get("statistics", {})
        out[it["id"]] = {"views": _int(st.get("viewCount")), "likes": _int(st.get("likeCount")),
                         "comments": _int(st.get("commentCount"))}
    return out


ANALYTICS_METRICS = ["views", "likes", "comments", "shares", "averageViewDuration",
                     "averageViewPercentage", "subscribersGained"]


def video_analytics(video_id: str, start_date: str, end_date: str | None = None) -> dict | None:
    end_date = end_date or time.strftime("%Y-%m-%d")
    r = requests.get(ANALYTICS_URL, params={
        "ids": "channel==MINE", "startDate": start_date, "endDate": end_date,
        "metrics": ",".join(ANALYTICS_METRICS), "filters": f"video=={video_id}"}, headers=_auth(), timeout=TIMEOUT)
    if r.status_code != 200:
        raise http_error(r)
    return parse_analytics_report(r.json())


def parse_analytics_report(payload: dict) -> dict | None:
    """Map columnHeaders -> first row. Returns None when the report has no rows
    (YouTube Analytics lags 1-3 days) – never invents zeros."""
    rows = payload.get("rows") or []
    if not rows:
        return None
    names = [h.get("name") for h in payload.get("columnHeaders", [])]
    rec = dict(zip(names, rows[0]))
    return {"views": _int(rec.get("views")), "likes": _int(rec.get("likes")), "comments": _int(rec.get("comments")),
            "shares": _int(rec.get("shares")),
            "avg_view_duration": float(rec["averageViewDuration"]) if rec.get("averageViewDuration") is not None else None,
            "avg_view_pct": float(rec["averageViewPercentage"]) if rec.get("averageViewPercentage") is not None else None,
            "subscribers_gained": _int(rec.get("subscribersGained"))}

