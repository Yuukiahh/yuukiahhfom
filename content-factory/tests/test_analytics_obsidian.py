"""Analytics parsing (fixtures), refresh without connection, Obsidian export + secret guard."""
from __future__ import annotations

import json
import time
from unittest import mock

import pytest

from test_publish_gates import YT_TOKEN, _resp, connect_youtube

YT_STATS = {"items": [{"id": "abc123", "statistics": {"viewCount": "1520", "likeCount": "88", "commentCount": "7",
                                                       "favoriteCount": "0"}}]}
YT_REPORT = {"kind": "youtubeAnalytics#resultTable",
             "columnHeaders": [{"name": n, "columnType": "METRIC"} for n in
                               ["views", "likes", "comments", "shares", "averageViewDuration",
                                "averageViewPercentage", "subscribersGained"]],
             "rows": [[1400, 80, 6, 12, 24, 77.5, 3]]}
TT_QUERY = {"videos": [{"id": "7300000000000000001", "view_count": 999, "like_count": 50, "comment_count": 4,
                        "share_count": 2}], "cursor": 0, "has_more": False}
IG_INSIGHTS = {"data": [{"name": "views", "period": "lifetime", "values": [{"value": 321}]},
                        {"name": "reach", "values": [{"value": 300}]},
                        {"name": "likes", "values": [{"value": 21}]},
                        {"name": "comments", "values": [{"value": 2}]},
                        {"name": "shares", "values": [{"value": 1}]},
                        {"name": "saved", "total_value": {"value": 5}}]}


def test_parsers():
    from factory.publish import instagram, tiktok, youtube
    assert youtube.parse_video_statistics(YT_STATS) == {"abc123": {"views": 1520, "likes": 88, "comments": 7}}
    r = youtube.parse_analytics_report(YT_REPORT)
    assert r == {"views": 1400, "likes": 80, "comments": 6, "shares": 12, "avg_view_duration": 24.0,
                 "avg_view_pct": 77.5, "subscribers_gained": 3}
    assert youtube.parse_analytics_report({"columnHeaders": YT_REPORT["columnHeaders"], "rows": []}) is None
    assert tiktok.parse_video_query(TT_QUERY)["7300000000000000001"]["views"] == 999
    ig = instagram.parse_insights(IG_INSIGHTS)
    assert ig["views"] == 321 and ig["saved"] == 5 and ig["shares"] == 1


def test_merge_never_fabricates():
    from factory.analytics import merge_youtube
    m = merge_youtube({"views": 1520, "likes": 88, "comments": None}, None)
    assert m["views"] == 1520 and m["comments"] is None and m["shares"] is None and m["avg_view_pct"] is None
    m2 = merge_youtube({"views": 1520, "likes": 88, "comments": 7}, youtube_report := {"views": 1400, "shares": 12})
    assert m2["views"] == 1520 and m2["shares"] == 12 and youtube_report


def _uploaded(env, platform="youtube", remote="abc123"):
    from factory import db
    vid = env.video(approval="approved")
    db.insert("publications", video_id=vid, platform=platform, remote_id=remote, status="uploaded",
              published_at=time.time() - 86400, url="https://youtube.com/shorts/" + remote)
    return vid


def test_refresh_youtube_inserts_metrics(env):
    from factory import analytics, db
    connect_youtube(env)
    vid = _uploaded(env)

    def get(url, params=None, headers=None, timeout=None):
        return _resp(200, YT_STATS if url.endswith("/videos") else YT_REPORT)

    with mock.patch("requests.get", side_effect=get):
        out = analytics.refresh()
    assert out["youtube"]["updated"] == 1
    m = db.row("SELECT * FROM metrics WHERE video_id=?", vid)
    assert (m["views"], m["likes"], m["comments"], m["shares"], m["avg_view_pct"], m["subscribers_gained"]) == \
        (1520, 88, 7, 12, 77.5, 3)
    assert out["learning"]["status"] == "insufficient_data"
    s = analytics.summary()
    assert s["totals"]["views"] == 1520 and s["per_niche"]["space"]["videos"] == 1


def test_refresh_skips_unconnected(env):
    from factory import analytics, db
    _uploaded(env)
    _uploaded(env, "instagram", "M1")
    with mock.patch("requests.get") as g, mock.patch("requests.request") as rq:
        out = analytics.refresh()
    g.assert_not_called(); rq.assert_not_called()
    assert "nicht verbunden" in out["youtube"]["note"] and "nicht verbunden" in out["instagram"]["note"]
    assert db.rows("SELECT * FROM metrics") == []


def test_obsidian_export_writes_expected_files(env, tmp_path):
    from factory import db, obsidian
    vid = _uploaded(env)
    db.insert("metrics", video_id=vid, platform="youtube", views=10, likes=1, comments=0, shares=0,
              avg_view_duration=5.0, avg_view_pct=50.0, subscribers_gained=0, fetched_at=time.time())
    db.kv_set("decisions", [{"title": "Testentscheidung", "reason": "weil"}])
    db.log_event("boom happened", "error")
    out = tmp_path / "tpl"
    res = obsidian.export_template(out)
    expected = {"00 Übersicht.md", "Installierte Tools.md", "Systemarchitektur.md", "Entscheidungen.md",
                "Erfolgreiche Formate.md", "Fehler und Lösungen.md", "Content-Strategie.md", "Fortschritt.md",
                "Offene Aufgaben.md"}
    assert expected <= {p.name for p in out.iterdir()}
    assert res["refused"] == []
    vids = list((out / "Videos").glob("*.md"))
    assert len(vids) == 1
    note = vids[0].read_text("utf-8")
    assert note.startswith("---\n") and 'niche: "space"' in note and "qc_score: 96" in note
    assert "https://youtube.com/shorts/abc123" in note and "youtube_views: 10" in note
    assert "```mermaid" in (out / "Systemarchitektur.md").read_text("utf-8")
    dec = (out / "Entscheidungen.md").read_text("utf-8")
    assert "Kokoro" in dec and "MoneyPrinterTurbo" in dec and "Testentscheidung" in dec
    assert "boom happened" in (out / "Fehler und Lösungen.md").read_text("utf-8")


def test_obsidian_preserves_user_text(env, tmp_path):
    from factory import obsidian
    out = tmp_path / "v"
    obsidian.export_template(out, include_videos=False)
    p = out / "Fortschritt.md"
    p.write_text(p.read_text("utf-8") + "\n## Meine Notizen\nbitte behalten\n", "utf-8")
    obsidian.export_template(out, include_videos=False)
    txt = p.read_text("utf-8")
    assert "bitte behalten" in txt and txt.count(obsidian.AUTO_START) == 1


def test_obsidian_sync_to_vault(env, tmp_path):
    from factory import obsidian
    vault = tmp_path / "vault"; vault.mkdir()
    env.config(obsidian={"vault_path": vault.as_posix(), "project_folder": "AI Content Factory"})
    res = obsidian.sync()
    assert res["status"] == "ok"
    assert (vault / "AI Content Factory" / "00 Übersicht.md").exists()


def test_obsidian_refuses_secret_and_tokens_never_written(env, tmp_path):
    from factory import db, obsidian
    connect_youtube(env)
    env.secrets["anthropic_api_key"] = "sk-ant-api03-SECRETSECRETSECRET123"
    # an error event that (wrongly) contains a token must not end up in the vault
    db.log_event("upload failed with " + YT_TOKEN["access_token"], "error")
    out = tmp_path / "o"
    res = obsidian.export_template(out)
    assert "Fehler und Lösungen.md" in res["refused"]
    blob = "".join(p.read_text("utf-8") for p in out.rglob("*.md"))
    for v in (YT_TOKEN["access_token"], YT_TOKEN["refresh_token"], "GOCSPX-clientsecret123456",
              "sk-ant-api03-SECRETSECRETSECRET123"):
        assert v not in blob
    with pytest.raises(obsidian.SecretLeakError):
        obsidian._guard("x " + YT_TOKEN["refresh_token"], [])


def test_redact():
    from factory.publish import redact
    s = redact("tok ya29.abcdefghijklmnopqrstuvwxyz0123 and refresh_token=foo&x=1 EAAB" + "c" * 40, values=[])
    assert "ya29.abc" not in s and "foo" not in s and "EAAB" not in s
