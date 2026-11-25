"""TikTok / Instagram / YouTube request logic with mocked HTTP."""
from __future__ import annotations

import json
import time
from unittest import mock

import pytest

from test_publish_gates import _resp


def _tiktok(env):
    env.config(publish={"enabled": True, "platforms": ["tiktok"]})
    env.secrets.update({"tiktok_client_key": "awkey123", "tiktok_client_secret": "tsecret123456",
                        "tiktok_oauth_token": json.dumps({"access_token": "act.SECRETTIKTOKTOKEN1234567890",
                                                          "expires_at": time.time() + 999, "scope": "video.publish"})})


def test_tiktok_chunk_plan():
    from factory.publish.tiktok import chunk_plan
    assert chunk_plan(1000) == (1000, 1)                      # < 5 MB -> one chunk
    assert chunk_plan(25 * 2**20, 10 * 2**20) == (10 * 2**20, 2)  # remainder goes into last chunk
    assert chunk_plan(8 * 2**20, 10 * 2**20) == (8 * 2**20, 1)


def test_tiktok_direct_post_private(env, tmp_path):
    from factory.publish import tiktok
    _tiktok(env)
    f = tmp_path / "v.mp4"; f.write_bytes(b"x" * 2048)
    calls = []

    def post(url, **kw):
        calls.append((url, kw.get("json")))
        if url.endswith("creator_info/query/"):
            return _resp(200, {"data": {"privacy_level_options": ["SELF_ONLY"], "max_video_post_duration_sec": 600},
                               "error": {"code": "ok"}})
        if url.endswith("video/init/"):
            return _resp(200, {"data": {"publish_id": "v_pub_1", "upload_url": "https://up.example/x"},
                               "error": {"code": "ok"}})
        if url.endswith("status/fetch/"):
            return _resp(200, {"data": {"status": "PUBLISH_COMPLETE"}, "error": {"code": "ok"}})
        raise AssertionError(url)

    with mock.patch("requests.post", side_effect=post), mock.patch("requests.put", return_value=_resp(201)) as put:
        res = tiktok.upload(str(f), {"caption": "hi #space"}, "private")
    init = [b for u, b in calls if u.endswith("/v2/post/publish/video/init/")][0]
    assert init["post_info"]["privacy_level"] == "SELF_ONLY"
    assert init["post_info"]["is_aigc"] is True
    assert init["source_info"] == {"source": "FILE_UPLOAD", "video_size": 2048, "chunk_size": 2048,
                                   "total_chunk_count": 1}
    assert put.call_args.kwargs["headers"]["Content-Range"] == "bytes 0-2047/2048"
    assert calls[0][0].endswith("creator_info/query/")  # creator info always first
    assert res["publish_id"] == "v_pub_1" and res["url"] is None


def test_tiktok_public_refused_if_not_offered(env):
    from factory.publish import PublishRefused, tiktok
    with pytest.raises(PublishRefused):
        tiktok.resolve_privacy("public", ["SELF_ONLY"])
    assert tiktok.resolve_privacy("public", ["SELF_ONLY", "PUBLIC_TO_EVERYONE"]) == "PUBLIC_TO_EVERYONE"


def test_tiktok_draft_mode_uses_inbox(env, tmp_path):
    from factory.publish import tiktok
    _tiktok(env)
    f = tmp_path / "v.mp4"; f.write_bytes(b"x" * 10)
    urls = []

    def post(url, **kw):
        urls.append(url)
        if "inbox/video/init" in url:
            return _resp(200, {"data": {"publish_id": "p2", "upload_url": "https://up"}, "error": {"code": "ok"}})
        return _resp(200, {"data": {"status": "SEND_TO_USER_INBOX"}, "error": {"code": "ok"}})

    with mock.patch("requests.post", side_effect=post), mock.patch("requests.put", return_value=_resp(201)):
        res = tiktok.upload(str(f), {}, "private", mode="draft")
    assert any(u.endswith("/v2/post/publish/inbox/video/init/") for u in urls)
    assert not any("creator_info" in u for u in urls)
    assert res["status"] == "SEND_TO_USER_INBOX"


def test_instagram_needs_public_url(env, tmp_path):
    from factory.publish import PublishRefused, instagram
    env.config(publish={"enabled": True, "default_privacy": "public", "platforms": ["instagram"]})
    env.secrets.update({"instagram_access_token": "EAAsecret", "instagram_user_id": "17841"})
    f = tmp_path / "v.mp4"; f.write_bytes(b"x")
    with mock.patch("requests.request") as rq, pytest.raises(PublishRefused, match="needs public video URL"):
        instagram.upload(str(f), {"caption": "c"}, "public")
    rq.assert_not_called()


def test_instagram_refuses_private(env, tmp_path):
    from factory.publish import PublishRefused, instagram
    env.config(publish={"enabled": True, "instagram_public_base_url": "https://cdn.example/out"})
    env.secrets.update({"instagram_access_token": "EAAsecret", "instagram_user_id": "17841"})
    with pytest.raises(PublishRefused):
        instagram.upload(str(tmp_path / "v.mp4"), {}, "private")


def test_instagram_reels_flow(env):
    from factory.publish import instagram
    env.config(publish={"enabled": True, "default_privacy": "public",
                        "instagram_public_base_url": "https://cdn.example/out"})
    env.secrets.update({"instagram_access_token": "EAAsecrettoken", "instagram_user_id": "17841"})
    vid_path = env.tmp / "output" / "2026-11-25" / "slug" / "slug.mp4"
    seq = []

    def req(method, url, params=None, headers=None, timeout=None):
        seq.append((method, url.split("/v21.0")[1], params))
        assert "EAAsecrettoken" not in url and "access_token" not in (params or {})
        assert headers["Authorization"] == "Bearer EAAsecrettoken"
        if url.endswith("/17841/media"):
            return _resp(200, {"id": "C1"})
        if url.endswith("/C1"):
            return _resp(200, {"status_code": "FINISHED"})
        if url.endswith("/media_publish"):
            return _resp(200, {"id": "M1"})
        return _resp(200, {"permalink": "https://www.instagram.com/reel/xyz/"})

    with mock.patch("requests.request", side_effect=req):
        res = instagram.upload(str(vid_path), {"caption": "c"}, "public", poll_interval=0)
    assert seq[0][2]["media_type"] == "REELS"
    assert seq[0][2]["video_url"] == "https://cdn.example/out/2026-11-25/slug/slug.mp4"
    assert seq[2][2] == {"creation_id": "C1"}
    assert res == {"id": "M1", "url": "https://www.instagram.com/reel/xyz/", "note": mock.ANY}


def test_youtube_auth_url_has_pkce_and_scopes():
    from urllib.parse import parse_qs, urlparse
    from factory.publish import pkce_pair, youtube
    v, c = pkce_pair()
    q = parse_qs(urlparse(youtube.build_auth_url("cid", "http://127.0.0.1:5555/", "st", c)).query)
    assert q["code_challenge_method"] == ["S256"] and q["code_challenge"] == [c]
    assert "youtube.upload" in q["scope"][0] and "yt-analytics.readonly" in q["scope"][0]
    assert q["access_type"] == ["offline"] and q["redirect_uri"] == ["http://127.0.0.1:5555/"]
    assert 43 <= len(v) <= 128


def test_youtube_token_refresh(env):
    from factory.publish import youtube
    env.secrets["youtube_oauth_client"] = json.dumps({"installed": {"client_id": "c", "client_secret": "s"}})
    env.secrets["youtube_oauth_token"] = json.dumps({"access_token": "old", "refresh_token": "r", "expires_at": 0})
    with mock.patch("requests.post", return_value=_resp(200, {"access_token": "new", "expires_in": 3600})) as rp:
        assert youtube.access_token() == "new"
    assert rp.call_args.kwargs["data"]["grant_type"] == "refresh_token"
    tok = json.loads(env.secrets["youtube_oauth_token"])
    assert tok["refresh_token"] == "r" and tok["access_token"] == "new"


def test_youtube_resumable_chunks(env, tmp_path):
    from factory.publish import youtube
    env.config(publish={"enabled": True})
    env.secrets["youtube_oauth_client"] = json.dumps({"installed": {"client_id": "c"}})
    env.secrets["youtube_oauth_token"] = json.dumps({"access_token": "a", "expires_at": time.time() + 999})
    f = tmp_path / "v.mp4"; f.write_bytes(b"x" * 600)
    puts = [_resp(308, headers={"Range": "bytes=0-255"}), _resp(503), _resp(308, headers={"Range": "bytes=0-255"}),
            _resp(308, headers={"Range": "bytes=0-511"}), _resp(201, {"id": "vid9", "status": {"privacyStatus": "private"}})]
    with mock.patch("requests.post", return_value=_resp(200, headers={"Location": "https://s"})), \
            mock.patch("requests.put", side_effect=puts) as put, mock.patch("time.sleep"):
        res = youtube.upload(str(f), {"title": "t"}, "private", chunk_size=256)
    ranges = [c.kwargs["headers"].get("Content-Range") for c in put.call_args_list]
    assert ranges == ["bytes 0-255/600", "bytes 256-511/600", "bytes */600", "bytes 256-511/600", "bytes 512-599/600"]
    assert res["id"] == "vid9"


def test_youtube_connect_loopback_flow(env, capsys):
    """Simulates the browser redirect against the real local http.server."""
    import threading
    import urllib.parse
    import urllib.request
    from factory.publish import youtube
    env.secrets["youtube_oauth_client"] = json.dumps({"installed": {"client_id": "cid", "client_secret": "GOCSPX-abcdefghijklmnop"}})

    def fake_browser(url):
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
        cb = q["redirect_uri"] + "?" + urllib.parse.urlencode({"code": "4/CODE", "state": q["state"]})
        threading.Thread(target=lambda: urllib.request.urlopen(cb, timeout=5).read(), daemon=True).start()
        return True

    tok = {"access_token": "ya29.NEWACCESSTOKEN_abcdefghijklmnop", "refresh_token": "1//NEWREFRESH_abcdefghijklmnop",
           "expires_in": 3599, "scope": " ".join(youtube.SCOPES)}
    with mock.patch("webbrowser.open", side_effect=fake_browser), \
            mock.patch("requests.post", return_value=_resp(200, tok)) as rp, \
            mock.patch("requests.get", return_value=_resp(200, {"items": []})), \
            mock.patch.dict("os.environ", {"NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1"}):
        info = youtube.connect(timeout=10)
    assert info["connected"] is True
    data = rp.call_args.kwargs["data"]
    assert data["code"] == "4/CODE" and data["code_verifier"] and data["grant_type"] == "authorization_code"
    assert json.loads(env.secrets["youtube_oauth_token"])["refresh_token"] == tok["refresh_token"]
    printed = capsys.readouterr().out
    assert tok["access_token"] not in printed and "GOCSPX" not in printed
    assert youtube.is_connected()
