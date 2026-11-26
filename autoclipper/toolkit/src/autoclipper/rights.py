"""Usage-rights check before anything is downloaded or processed.

Classifies a source as:
  own      - the uploader is one of your channels (config.own_channels)
  cc       - YouTube marks it Creative Commons (reuse allowed, credit required)
  local    - a file on your computer (you vouch for it)
  unknown  - standard licence or unknown: ask the owner of the PC before processing
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Settings


@dataclass
class SourceInfo:
    kind: str                 # youtube | url | local
    source: str
    rights: str               # own | cc | local | confirmed | unknown
    title: str = ""
    channel: str = ""
    channel_url: str = ""
    license: str = ""
    duration: float = 0.0
    upload_date: str = ""
    language: str = ""
    is_live: bool = False
    webpage_url: str = ""
    credit_line: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def allowed(self) -> bool:
        return self.rights in {"own", "cc", "local", "confirmed"}


def inspect_source(source: str, settings: Settings, *, confirmed: bool = False) -> SourceInfo:
    path = Path(source.strip('"'))
    if path.exists() and path.is_file():
        return SourceInfo(kind="local", source=str(path.resolve()), rights="local", title=path.stem)

    import yt_dlp  # imported lazily: heavy

    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        meta = ydl.extract_info(source, download=False)

    extractor = (meta.get("extractor_key") or "").lower()
    lic = meta.get("license") or ""
    channel = meta.get("channel") or meta.get("uploader") or ""
    own = {c.strip().lower() for c in settings.own_channels if c}
    ids = {channel.lower(), (meta.get("channel_id") or "").lower(), (meta.get("uploader_id") or "").lower()}
    if own & ids:
        rights = "own"
    elif "creative commons" in lic.lower():
        rights = "cc"
    elif confirmed:
        rights = "confirmed"
    else:
        rights = "unknown"

    info = SourceInfo(
        kind="youtube" if "youtube" in extractor else "url",
        source=source,
        rights=rights,
        title=meta.get("title") or "",
        channel=channel,
        channel_url=meta.get("channel_url") or meta.get("uploader_url") or "",
        license=lic,
        duration=float(meta.get("duration") or 0),
        upload_date=meta.get("upload_date") or "",
        language=meta.get("language") or "",
        is_live=bool(meta.get("is_live")),
        webpage_url=meta.get("webpage_url") or source,
    )
    if rights == "cc":
        info.credit_line = f'Quelle: "{info.title}" von {channel} ({info.webpage_url}), CC BY'
    return info
