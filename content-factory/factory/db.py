"""SQLite persistence: trends, topics, jobs (queue), videos, metrics, costs, events."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .config import cfg

SCHEMA = """
CREATE TABLE IF NOT EXISTS trend_signals (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,          -- google_trends | reddit | youtube | wikipedia | hackernews | manual
    title TEXT NOT NULL,
    url TEXT,
    metric_name TEXT,              -- e.g. upvotes, approx_traffic, views
    metric_value REAL,
    niche TEXT,
    raw JSON,
    fetched_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_signals_time ON trend_signals(fetched_at);

CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    niche TEXT,
    score REAL,
    score_detail JSON,
    evidence JSON,                 -- list of signal ids / urls backing this topic
    status TEXT DEFAULT 'candidate', -- candidate | queued | used | rejected
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY,
    topic TEXT NOT NULL,
    topic_id INTEGER,
    niche TEXT,
    format TEXT DEFAULT 'short',   -- short | long
    language TEXT DEFAULT 'en',
    preset TEXT DEFAULT 'cinematic',
    voice TEXT,
    status TEXT DEFAULT 'queued',  -- queued | running | done | failed | cancelled | review
    attempts INTEGER DEFAULT 0,
    stage TEXT,
    progress REAL DEFAULT 0,
    error TEXT,
    video_id INTEGER,
    options JSON,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    not_before REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY,
    job_id INTEGER,
    slug TEXT,
    title TEXT,
    topic TEXT,
    niche TEXT,
    format TEXT,
    language TEXT,
    duration REAL,
    path TEXT,
    folder TEXT,
    hook TEXT,
    qc_passed INTEGER,
    qc JSON,
    metadata JSON,
    fingerprint TEXT,
    approval TEXT DEFAULT 'pending', -- pending | approved | rejected
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS publications (
    id INTEGER PRIMARY KEY,
    video_id INTEGER NOT NULL,
    platform TEXT NOT NULL,
    remote_id TEXT,
    url TEXT,
    status TEXT,                   -- scheduled | uploaded | failed
    scheduled_for REAL,
    published_at REAL,
    error TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY,
    video_id INTEGER NOT NULL,
    platform TEXT NOT NULL,
    views INTEGER, likes INTEGER, comments INTEGER, shares INTEGER,
    avg_view_duration REAL, avg_view_pct REAL, subscribers_gained INTEGER,
    fetched_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS costs (
    id INTEGER PRIMARY KEY,
    service TEXT NOT NULL,
    amount_eur REAL NOT NULL,
    units REAL, unit_name TEXT,
    job_id INTEGER,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    level TEXT, message TEXT, job_id INTEGER, created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value JSON);
"""

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None


def db_path() -> Path:
    d = cfg().path("general.data_dir")
    d.mkdir(parents=True, exist_ok=True)
    return d / "factory.sqlite3"


def conn() -> sqlite3.Connection:
    global _conn
    with _lock:
        if _conn is None:
            _conn = sqlite3.connect(db_path(), check_same_thread=False, timeout=30)
            _conn.row_factory = sqlite3.Row
            _conn.execute("PRAGMA journal_mode=WAL")
            _conn.executescript(SCHEMA)
        return _conn


@contextmanager
def tx() -> Iterator[sqlite3.Connection]:
    with _lock:
        c = conn()
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise


def _enc(v: Any) -> Any:
    return json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v


def insert(table: str, **fields: Any) -> int:
    keys = list(fields)
    with tx() as c:
        cur = c.execute(
            f"INSERT INTO {table} ({','.join(keys)}) VALUES ({','.join('?' * len(keys))})",
            [_enc(fields[k]) for k in keys],
        )
        return int(cur.lastrowid)


def update(table: str, row_id: int, **fields: Any) -> None:
    if table == "jobs":
        fields.setdefault("updated_at", time.time())
    sets = ",".join(f"{k}=?" for k in fields)
    with tx() as c:
        c.execute(f"UPDATE {table} SET {sets} WHERE id=?", [_enc(v) for v in fields.values()] + [row_id])


def rows(sql: str, *args: Any) -> list[dict]:
    with _lock:
        out = []
        for r in conn().execute(sql, args).fetchall():
            d = dict(r)
            for k, v in d.items():
                if isinstance(v, str) and v[:1] in "[{":
                    try:
                        d[k] = json.loads(v)
                    except ValueError:
                        pass
            out.append(d)
        return out


def row(sql: str, *args: Any) -> dict | None:
    r = rows(sql, *args)
    return r[0] if r else None


def kv_get(key: str, default: Any = None) -> Any:
    r = row("SELECT value FROM kv WHERE key=?", key)
    if not r:
        return default
    v = r["value"]
    return json.loads(v) if isinstance(v, str) else v


def kv_set(key: str, value: Any) -> None:
    with tx() as c:
        c.execute("INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                  (key, json.dumps(value)))


def log_event(message: str, level: str = "info", job_id: int | None = None) -> None:
    insert("events", level=level, message=message, job_id=job_id, created_at=time.time())
