"""Minimal synchronous client for a self-hosted SupoClip backend.

Uses the same auth as the official SupoClip MCP server: an API key, or HMAC-signed
x-supoclip-* headers built from the local user id and BACKEND_AUTH_SECRET.
"""

from __future__ import annotations

import hashlib
import hmac
import mimetypes
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import httpx

from .config import Settings

TERMINAL = {"completed", "error", "cancelled"}


class SupoClipAPIError(RuntimeError):
    def __init__(self, status: int, detail: Any, path: str):
        self.status, self.detail, self.path = status, detail, path
        super().__init__(f"SupoClip {path} -> HTTP {status}: {detail}")

    @property
    def code(self) -> str | None:
        return self.detail.get("code") if isinstance(self.detail, dict) else None


class SupoClip:
    def __init__(self, settings: Settings, timeout: float = 60.0):
        self.s = settings
        self.timeout = timeout

    def _auth(self) -> dict[str, str]:
        if self.s.api_key:
            return {"Authorization": f"Bearer {self.s.api_key}"}
        if self.s.user_id and self.s.auth_secret:
            ts = str(int(time.time()))
            sig = hmac.new(self.s.auth_secret.encode(), f"{self.s.user_id}:{ts}".encode(), hashlib.sha256).hexdigest()
            return {"x-supoclip-user-id": self.s.user_id, "x-supoclip-ts": ts, "x-supoclip-signature": sig}
        if self.s.user_id:
            return {"x-supoclip-user-id": self.s.user_id}
        raise SupoClipAPIError(401, "no SupoClip credentials configured (run the installer)", "-")

    def request(self, method: str, path: str, *, auth: bool = True, **kw) -> Any:
        headers = {"Accept": "application/json", "User-Agent": "autoclipper/1.0"}
        if auth:
            headers.update(self._auth())
        with httpx.Client(timeout=kw.pop("timeout", self.timeout)) as client:
            resp = client.request(method, f"{self.s.api_url}{path}", headers=headers, **kw)
        if resp.status_code >= 400:
            try:
                body = resp.json()
                detail = body.get("detail", body) if isinstance(body, dict) else body
            except ValueError:
                detail = resp.text[:500]
            raise SupoClipAPIError(resp.status_code, detail, path)
        return resp.json() if resp.content else {}

    # -- endpoints ------------------------------------------------------------------------
    def health(self) -> dict:
        return self.request("GET", "/health", auth=False)

    def edit_styles(self) -> dict:
        return self.request("GET", "/tasks/edit-styles")

    def caption_templates(self) -> list[str]:
        data = self.request("GET", "/caption-templates", auth=False, params={"include": "fx"})
        items = data.get("templates", data) if isinstance(data, dict) else data
        ids = []
        for item in items or []:
            ids.append(item.get("id") if isinstance(item, dict) else item)
        return [i for i in ids if i]

    def broll_status(self) -> dict:
        return self.request("GET", "/broll/status", auth=False)

    def upload(self, file: Path) -> str:
        """Upload a local video; returns the ``upload://...`` source to create a task with."""
        file = Path(file)
        mime = mimetypes.guess_type(file.name)[0] or "video/mp4"
        headers = {"User-Agent": "autoclipper/1.0", "Idempotency-Key": f"autoclipper-{uuid.uuid4().hex}"}
        headers.update(self._auth())
        with file.open("rb") as fh, httpx.Client(timeout=None) as client:
            resp = client.post(f"{self.s.api_url}/upload", headers=headers,
                               files={"video": (file.name, fh, mime)})
        if resp.status_code >= 400:
            raise SupoClipAPIError(resp.status_code, resp.text[:500], "/upload")
        return resp.json()["video_path"]

    def create_task(self, body: dict) -> dict:
        return self.request("POST", "/tasks/", json=body)

    def get_task(self, task_id: str) -> dict:
        return self.request("GET", f"/tasks/{task_id}")

    def list_clips(self, task_id: str) -> list[dict]:
        data = self.request("GET", f"/tasks/{task_id}/clips")
        return data.get("clips", []) if isinstance(data, dict) else data

    def wait(self, task_id: str, *, poll: float = 10.0, timeout: float = 6 * 3600,
             on_progress: Callable[[dict], None] | None = None) -> dict:
        deadline = time.monotonic() + timeout
        last: dict = {}
        while time.monotonic() < deadline:
            try:
                last = self.get_task(task_id)
            except (httpx.HTTPError, SupoClipAPIError) as exc:
                # The backend restarts a worker now and then; keep waiting.
                if isinstance(exc, SupoClipAPIError) and exc.status in (401, 403, 404):
                    raise
                time.sleep(poll)
                continue
            if on_progress:
                on_progress(last)
            if str(last.get("status")) in TERMINAL:
                return last
            time.sleep(poll)
        last["timed_out"] = True
        return last

    def download_clip(self, task_id: str, clip_id: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        headers = {"User-Agent": "autoclipper/1.0", **self._auth()}
        with httpx.Client(timeout=None, follow_redirects=True) as client:
            with client.stream("GET", f"{self.s.api_url}/tasks/{task_id}/clips/{clip_id}/file",
                               headers=headers) as resp:
                if resp.status_code >= 400:
                    raise SupoClipAPIError(resp.status_code, resp.read()[:300], "clip file")
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_bytes(1 << 20):
                        fh.write(chunk)
        tmp.replace(dest)
        return dest
