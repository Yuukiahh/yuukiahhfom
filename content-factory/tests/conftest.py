"""Test fixtures: isolated config dir, data dir and in-memory secrets."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Returns a helper with .config(**sections) to set CF_CONFIG overrides,
    .secrets dict (in-memory secret store) and .video(...) to create a video row."""
    from factory import config, db, secrets

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    shutil.copy(ROOT / "config" / "default.toml", cfg_dir / "default.toml")
    monkeypatch.setattr(config, "CONFIG_DIR", cfg_dir)  # set_local() writes here, not into the repo
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "output"
    extra = tmp_path / "test.toml"
    monkeypatch.setenv("CF_CONFIG", str(extra))

    store: dict[str, str] = {}
    monkeypatch.setattr(secrets, "get", lambda n: store.get(n))
    monkeypatch.setattr(secrets, "set", lambda n, v: store.__setitem__(n, v))
    monkeypatch.setattr(secrets, "delete", lambda n: store.pop(n, None))
    monkeypatch.setattr(secrets, "status", lambda: {k: bool(store.get(k)) for k in secrets.KNOWN})

    class Env:
        secrets = store
        tmp = tmp_path

        def config(self, publish=None, obsidian=None):
            lines = ["[general]", f'data_dir = "{data_dir.as_posix()}"', f'output_dir = "{out_dir.as_posix()}"', ""]
            for name, sec in (("publish", publish), ("obsidian", obsidian)):
                if sec:
                    lines.append(f"[{name}]")
                    lines += [f"{k} = {config._toml_value(v)}" for k, v in sec.items()]
                    lines.append("")
            extra.write_text("\n".join(lines), "utf-8")
            config.reload()

        def video(self, approval="pending", **kw):
            import time
            out_dir.mkdir(parents=True, exist_ok=True)
            vf = out_dir / "2026-11-25" / "test" / "test.mp4"
            vf.parent.mkdir(parents=True, exist_ok=True)
            vf.write_bytes(b"\x00" * 1024)
            fields = dict(slug="test-video", title="Test Video", topic="Black holes", niche="space", format="short",
                          language="en", duration=31.5, path=str(vf), folder=str(vf.parent), hook="What if?",
                          qc_passed=1, qc={"score": 96, "passed": True}, approval=approval,
                          metadata={"title": "Test Video", "youtube": {"title": "Test #Shorts", "description": "d",
                                                                         "tags": ["space"], "categoryId": "27"},
                                    "tiktok": {"caption": "cap"}, "instagram": {"caption": "cap"},
                                    "hashtags": ["space"]},
                          created_at=time.time())
            fields.update(kw)
            return db.insert("videos", **fields)

    e = Env()
    e.config()
    if db._conn is not None:
        db._conn.close()
    monkeypatch.setattr(db, "_conn", None)
    yield e
    if db._conn is not None:
        db._conn.close()
    db._conn = None
    config._cfg = None  # next cfg() call reloads with the real config
