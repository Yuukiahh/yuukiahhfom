"""Duplicate-topic protection."""
from __future__ import annotations

import re

from . import db

STOP = set("the a an of in on to and is are what why how if you your into for with from at by it this that der die das ein eine und ist was wie wenn du in im zu von mit".split())


def tokens(text: str) -> set[str]:
    words = re.findall(r"[a-zäöüß0-9]+", text.lower())
    out = set()
    for w in words:
        if w in STOP or len(w) < 3:
            continue
        out.add(w[:-1] if w.endswith("s") and len(w) > 4 else w)
    return out


def fingerprint(text: str) -> str:
    return " ".join(sorted(tokens(text)))


def similarity(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def is_duplicate(topic: str, threshold: float = 0.55, days: int = 120) -> tuple[bool, str | None]:
    import time

    since = time.time() - days * 86400
    existing = db.rows("SELECT topic FROM videos WHERE created_at > ? UNION SELECT topic FROM jobs "
                       "WHERE status IN ('queued','running','review') AND created_at > ?", since, since)
    for r in existing:
        if similarity(topic, r["topic"]) >= threshold:
            return True, r["topic"]
    return False, None
