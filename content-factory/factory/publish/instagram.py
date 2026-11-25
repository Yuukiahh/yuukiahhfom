"""Instagram Graph API – Reels publishing (official Content Publishing API).

Requirements (Meta's rules, not ours):
  * Instagram *Business* or *Creator* account (linked to a Facebook Page when
    using Facebook Login), a Meta app with `instagram_content_publish` (or
    `instagram_business_content_publish`) permission. Using it for accounts other
    than your own app roles requires Meta App Review.
  * Secrets: `instagram_access_token` (long-lived token), `instagram_user_id`.
  * Meta downloads the video from a PUBLIC URL (`video_url`); local files cannot be
    sent. Configure `publish.instagram_public_base_url` – a public HTTPS base that
    serves the `output/` folder (e.g. your own web space / CDN). Without it the
    upload is refused with a clear error.
  * Reels published through the API are always public; there is no private mode.
    We therefore refuse Instagram uploads unless publish.default_privacy="public".
  * API limit: 50 published posts per 24 h per account.

Config (optional): publish.instagram_graph_host (default graph.facebook.com,
use graph.instagram.com for "Instagram API with Instagram Login"),
publish.instagram_api_version (default v21.0).
"""
from __future__ import annotations

import time
import urllib.parse
from pathlib import Path

import requests

from .. import secrets
from ..config import cfg
from . import PublishError, PublishRefused, http_error, require_enabled

NAME = "instagram"
TIMEOUT = 60


def _base() -> str:
    host = cfg().get("publish.instagram_graph_host", "graph.facebook.com")
    ver = cfg().get("publish.instagram_api_version", "v21.0")
    return f"https://{host}/{ver}"


def is_connected() -> bool:
    return bool(secrets.get("instagram_access_token") and secrets.get("instagram_user_id"))


def _creds() -> tuple[str, str]:
    tok, uid = secrets.get("instagram_access_token"), secrets.get("instagram_user_id")
    if not tok or not uid:
        raise PublishError("Instagram nicht verbunden (Secrets instagram_access_token / instagram_user_id).")
    return tok, uid


def _req(method: str, path: str, params: dict | None = None) -> dict:
    tok, _ = _creds()
    # token goes in the Authorization header, never in the URL (URLs end up in errors/logs)
    r = requests.request(method, _base() + path, params=params or {},
                         headers={"Authorization": "Bearer " + tok}, timeout=TIMEOUT)
    try:
        data = r.json()
    except ValueError:
        raise http_error(r) from None
    if r.status_code != 200 or "error" in data:
        e = data.get("error", {})
        raise PublishError(f"Instagram {path}: HTTP {r.status_code} {e.get('type', '')} {e.get('message', '')}")
    return data


def public_video_url(video_path: str) -> str:
    base = (cfg().get("publish.instagram_public_base_url") or "").strip()
    if not base:
        raise PublishRefused("needs public video URL: Instagram lädt Videos nur von einer öffentlichen URL. "
                             "Setze publish.instagram_public_base_url (öffentlicher HTTPS-Ordner, der output/ "
                             "ausliefert).")
    if not base.lower().startswith("https://"):
        raise PublishRefused("publish.instagram_public_base_url muss mit https:// beginnen")
    p = Path(video_path).resolve()
    out_dir = cfg().path("general.output_dir").resolve()
    try:
        rel = p.relative_to(out_dir).as_posix()
    except ValueError:
        rel = p.name
    return base.rstrip("/") + "/" + urllib.parse.quote(rel)


def upload(video_path: str, meta: dict, privacy: str = "private", thumbnail: str | None = None,
           poll_interval: float = 5, max_wait: float = 900) -> dict:
    require_enabled()
    if privacy != "public":
        raise PublishRefused("Instagram Reels sind über die API immer öffentlich – Upload nur mit "
                             "publish.default_privacy = \"public\" erlaubt.")
    _, uid = _creds()
    video_url = public_video_url(video_path)
    container = _req("POST", f"/{uid}/media", {
        "media_type": "REELS", "video_url": video_url, "caption": (meta.get("caption") or "")[:2200],
        "share_to_feed": "true"})
    cid = container.get("id")
    if not cid:
        raise PublishError("Instagram: kein Container erstellt")
    deadline = time.time() + max_wait
    while True:
        st = _req("GET", f"/{cid}", {"fields": "status_code,status"})
        code = st.get("status_code")
        if code == "FINISHED":
            break
        if code in ("ERROR", "EXPIRED"):
            raise PublishError(f"Instagram Container {code}: {st.get('status', '')}")
        if time.time() > deadline:
            raise PublishError("Instagram: Verarbeitung dauert zu lange (Timeout)")
        time.sleep(poll_interval)
    pub = _req("POST", f"/{uid}/media_publish", {"creation_id": cid})
    mid = pub.get("id")
    url = None
    try:
        url = _req("GET", f"/{mid}", {"fields": "permalink"}).get("permalink")
    except PublishError:
        pass
    return {"id": mid, "url": url, "note": "Instagram Reels sind öffentlich"}


# --------------------------------------------------------------------------- insights

INSIGHT_SETS = [
    "views,reach,likes,comments,shares,saved",   # Graph API >= v22 ("plays" -> "views")
    "plays,reach,likes,comments,shares,saved",   # older API versions
]


def media_insights(media_id: str) -> dict:
    last: Exception | None = None
    for metrics in INSIGHT_SETS:
        try:
            return parse_insights(_req("GET", f"/{media_id}/insights", {"metric": metrics}))
        except PublishError as e:
            last = e
    raise last or PublishError("Instagram insights nicht verfügbar")


def parse_insights(payload: dict) -> dict:
    vals: dict[str, int | None] = {}
    for item in payload.get("data", []):
        v = None
        if item.get("values"):
            v = item["values"][0].get("value")
        elif isinstance(item.get("total_value"), dict):
            v = item["total_value"].get("value")
        vals[item.get("name")] = v
    return {"views": vals.get("views", vals.get("plays")), "likes": vals.get("likes"),
            "comments": vals.get("comments"), "shares": vals.get("shares"),
            "reach": vals.get("reach"), "saved": vals.get("saved")}
