"""Secret storage.

Keys/tokens are stored in the OS credential store (Windows Credential Manager via
`keyring`). If no keyring backend exists (headless Linux), they are stored in
data/secrets.enc, encrypted with Fernet; the encryption key comes from the env
var CF_MASTER_KEY or is kept in the OS keyring. Secrets never go into the config,
the Obsidian vault, or git.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

SERVICE = "ai-content-factory"

KNOWN = {
    "pexels_api_key": "Pexels API (kostenlos) – lizenzfreie Stockvideos",
    "pixabay_api_key": "Pixabay API (kostenlos) – lizenzfreie Stockvideos",
    "youtube_api_key": "YouTube Data API v3 Key (kostenlos, Quota) – Trend-Recherche",
    "youtube_oauth_client": "Google OAuth Client JSON – Upload + Analytics",
    "youtube_oauth_token": "(automatisch) YouTube OAuth Token",
    "tiktok_client_key": "TikTok Content Posting API – Client Key",
    "tiktok_client_secret": "TikTok Content Posting API – Client Secret",
    "tiktok_oauth_token": "(automatisch) TikTok OAuth Token",
    "instagram_access_token": "Instagram Graph API – Long-lived Token",
    "instagram_user_id": "Instagram Business Account ID",
    "reddit_client_id": "Reddit API (optional, höhere Limits)",
    "reddit_client_secret": "Reddit API Secret (optional)",
    "anthropic_api_key": "Claude API (KOSTENPFLICHTIG – nur mit Freigabe)",
}


def _keyring():
    try:
        import keyring
        from keyring.backends import fail

        kr = keyring.get_keyring()
        if isinstance(kr, fail.Keyring) or "fail" in type(kr).__module__ or "null" in type(kr).__module__:
            return None
        return keyring
    except Exception:
        return None


def _enc_file() -> Path:
    from .config import cfg

    d = cfg().path("general.data_dir")
    d.mkdir(parents=True, exist_ok=True)
    return d / "secrets.enc"


def _fernet():
    from cryptography.fernet import Fernet

    key = os.environ.get("CF_MASTER_KEY")
    kr = _keyring()
    if not key and kr:
        key = kr.get_password(SERVICE, "__master__")
        if not key:
            key = Fernet.generate_key().decode()
            kr.set_password(SERVICE, "__master__", key)
    if not key:
        kf = _enc_file().with_suffix(".key")
        if not kf.exists():
            kf.write_text(Fernet.generate_key().decode())
            try:
                os.chmod(kf, 0o600)
            except OSError:
                pass
        key = kf.read_text().strip()
    return Fernet(key.encode())


def _load_file() -> dict:
    f = _enc_file()
    if not f.exists():
        return {}
    return json.loads(_fernet().decrypt(f.read_bytes()))


def _save_file(d: dict) -> None:
    _enc_file().write_bytes(_fernet().encrypt(json.dumps(d).encode()))


def get(name: str) -> str | None:
    env = os.environ.get("CF_" + name.upper())
    if env:
        return env
    kr = _keyring()
    if kr:
        try:
            v = kr.get_password(SERVICE, name)
            if v:
                return v
        except Exception:
            pass
    return _load_file().get(name)


def set(name: str, value: str) -> None:  # noqa: A001
    kr = _keyring()
    if kr:
        try:
            kr.set_password(SERVICE, name, value)
            return
        except Exception:
            pass
    d = _load_file()
    d[name] = value
    _save_file(d)


def delete(name: str) -> None:
    kr = _keyring()
    if kr:
        try:
            kr.delete_password(SERVICE, name)
        except Exception:
            pass
    d = _load_file()
    if d.pop(name, None) is not None:
        _save_file(d)


def status() -> dict[str, bool]:
    """Which secrets are configured (never returns values)."""
    return {k: bool(get(k)) for k in KNOWN}
