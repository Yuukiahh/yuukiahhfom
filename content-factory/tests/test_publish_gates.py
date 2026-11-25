"""Publishing must be impossible unless enabled AND approved AND connected."""
from __future__ import annotations

import json
import time
from unittest import mock

import pytest

YT_TOKEN = {"access_token": "ya29.SUPERSECRETACCESSTOKEN0123456789abcdef", "refresh_token": "1//REFRESHSECRET0123456789abcdef",
            "expires_at": time.time() + 3600, "scope": "x"}
YT_CLIENT = {"installed": {"client_id": "cid.apps.googleusercontent.com", "client_secret": "GOCSPX-clientsecret123456"}}


def connect_youtube(env):
    env.secrets["youtube_oauth_client"] = json.dumps(YT_CLIENT)
    env.secrets["youtube_oauth_token"] = json.dumps(YT_TOKEN)


def _due(env, vid, platform="youtube"):
    from factory import publish
    return publish.schedule(vid, platform, time.time() - 5)


def test_default_config_is_disabled_and_private(env):
    from factory import publish
    assert publish.is_enabled() is False
    assert publish.default_privacy() == "private"
    st = publish.publish_status()
    assert set(st) == {"youtube", "tiktok", "instagram"}
    assert all(not v["enabled"] and not v["connected"] for v in st.values())


def test_refused_when_disabled(env):
    from factory import db, publish
    connect_youtube(env)
    vid = env.video(approval="approved")
    pid = _due(env, vid)
    with mock.patch("factory.publish.youtube.upload") as up, mock.patch("requests.post") as rp, \
            mock.patch("requests.put") as rput:
        res = publish.run_due()
    up.assert_not_called(); rp.assert_not_called(); rput.assert_not_called()
    assert res[0]["status"] == "blocked" and "disabled" in res[0]["reasons"]
    assert db.row("SELECT status FROM publications WHERE id=?", pid)["status"] == "scheduled"


def test_refused_when_not_approved(env):
    from factory import publish
    env.config(publish={"enabled": True, "platforms": ["youtube"]})
    connect_youtube(env)
    vid = env.video(approval="pending")
    _due(env, vid)
    with mock.patch("factory.publish.youtube.upload") as up:
        res = publish.run_due()
    up.assert_not_called()
    assert "not_approved" in res[0]["reasons"]


def test_rejected_video_cancels_schedule(env):
    from factory import db, publish
    env.config(publish={"enabled": True, "platforms": ["youtube"]})
    connect_youtube(env)
    vid = env.video(approval="approved")
    pid = _due(env, vid)
    publish.reject(vid)
    assert db.row("SELECT status FROM publications WHERE id=?", pid)["status"] == "failed"
    with pytest.raises(publish.PublishRefused):
        publish.schedule(vid, "youtube", time.time())


def test_refused_when_not_connected(env):
    from factory import publish
    env.config(publish={"enabled": True, "platforms": ["youtube"]})
    vid = env.video(approval="approved")
    _due(env, vid)
    with mock.patch("factory.publish.youtube.upload") as up:
        res = publish.run_due()
    up.assert_not_called()
    assert res[0]["reasons"] == ["not_connected"]


def test_refused_when_platform_not_listed(env):
    from factory import publish
    env.config(publish={"enabled": True, "platforms": ["youtube"]})
    env.secrets.update({"instagram_access_token": "EAAinstatoken", "instagram_user_id": "1784"})
    vid = env.video(approval="approved")
    _due(env, vid, "instagram")
    with mock.patch("factory.publish.instagram.upload") as up:
        res = publish.run_due()
    up.assert_not_called()
    assert "platform_not_enabled" in res[0]["reasons"]


def test_platform_modules_refuse_directly_when_disabled(env, tmp_path):
    from factory.publish import PublishRefused, instagram, tiktok, youtube
    f = tmp_path / "v.mp4"; f.write_bytes(b"x")
    for mod in (youtube, tiktok, instagram):
        with mock.patch("requests.post") as rp, pytest.raises(PublishRefused):
            mod.upload(str(f), {"title": "t"}, "public")
        rp.assert_not_called()


def test_enable_requires_confirm(env):
    from factory import publish
    assert publish.enable(False)["ok"] is False
    assert publish.is_enabled() is False
    assert (env.tmp / "config" / "local.toml").exists() is False
    assert publish.enable(confirm=True)["enabled"] is True
    assert "enabled = true" in (env.tmp / "config" / "local.toml").read_text()


def _resp(status=200, json_body=None, headers=None, text=""):
    r = mock.Mock()
    r.status_code = status
    r.headers = headers or {}
    r.json.return_value = json_body or {}
    r.text = text or json.dumps(json_body or {})
    return r


def test_full_youtube_upload_when_all_gates_pass(env):
    from factory import db, publish
    env.config(publish={"enabled": True, "platforms": ["youtube"]})
    connect_youtube(env)
    vid = env.video(approval="approved")
    pid = _due(env, vid)
    init = _resp(200, headers={"Location": "https://upload.example/session"})
    done = _resp(200, {"id": "abc123", "status": {"privacyStatus": "private"}})
    with mock.patch("requests.post", return_value=init) as rp, mock.patch("requests.put", return_value=done) as rput:
        res = publish.run_due()
    assert res[0]["status"] == "uploaded", res
    body = json.loads(rp.call_args_list[0].kwargs["data"])
    assert body["status"]["privacyStatus"] == "private"
    assert body["status"]["containsSyntheticMedia"] is True
    assert body["status"]["selfDeclaredMadeForKids"] is False
    assert rput.call_args.kwargs["headers"]["Content-Range"] == "bytes 0-1023/1024"
    row = db.row("SELECT * FROM publications WHERE id=?", pid)
    assert row["status"] == "uploaded" and row["remote_id"] == "abc123"
    assert row["url"].endswith("abc123")


def test_upload_failure_is_redacted(env):
    from factory import db, publish
    env.config(publish={"enabled": True, "platforms": ["youtube"]})
    connect_youtube(env)
    vid = env.video(approval="approved")
    pid = _due(env, vid)
    leak = _resp(403, text="forbidden for token " + YT_TOKEN["access_token"] + " and " + YT_TOKEN["refresh_token"])
    with mock.patch("requests.post", return_value=leak):
        res = publish.run_due()
    assert res[0]["status"] == "failed"
    row = db.row("SELECT * FROM publications WHERE id=?", pid)
    dump = json.dumps(db.rows("SELECT * FROM events")) + row["error"]
    assert YT_TOKEN["access_token"] not in dump and YT_TOKEN["refresh_token"] not in dump
    assert "***" in row["error"]


def test_auto_schedule_uses_slots(env):
    from factory import publish
    env.config(publish={"platforms": ["youtube", "tiktok"], "slots": ["12:00", "18:00"]})
    vid = env.video()
    vid2 = env.video()
    a = publish.auto_schedule(vid)
    b = publish.auto_schedule(vid2)
    assert [x["platform"] for x in a] == ["youtube", "tiktok"]
    assert all(x["when"] > time.time() for x in a)
    assert a[0]["when"] != b[0]["when"]  # next free slot
    again = publish.auto_schedule(vid)
    assert all("bereits" in x["note"] for x in again)
