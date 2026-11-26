"""Full pipeline against a fake SupoClip backend that speaks the real API shapes."""

import hashlib
import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from autoclipper import package, pipeline

SECRET = "test-secret"
USER = "user-123"


class FakeSupoClip:
    def __init__(self, clip_files: dict[str, Path]):
        self.clip_files = clip_files
        self.created: list[dict] = []
        self.uploads = 0
        self.polls = 0
        self.bad_auth = 0


def handler_for(state: FakeSupoClip):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authed(self):
            ts = self.headers.get("x-supoclip-ts", "")
            want = hmac.new(SECRET.encode(), f"{USER}:{ts}".encode(), hashlib.sha256).hexdigest()
            ok = self.headers.get("x-supoclip-user-id") == USER and self.headers.get("x-supoclip-signature") == want
            if not ok:
                state.bad_auth += 1
                self._json(401, {"detail": "bad auth"})
            return ok

        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/health":
                return self._json(200, {"status": "healthy"})
            if p == "/caption-templates":
                return self._json(200, {"templates": [{"id": "hormozi"}, {"id": "highlighter_swipe"}]})
            if p == "/broll/status":
                return self._json(200, {"configured": False, "provider": None})
            if not self._authed():
                return
            if p == "/tasks/edit-styles":
                return self._json(200, {"enabled": True})
            if p == "/tasks/t1":
                state.polls += 1
                status = "completed" if state.polls >= 2 else "processing"
                return self._json(200, {"id": "t1", "status": status, "progress": 50 * state.polls})
            if p == "/tasks/t1/clips":
                clips = []
                for n, (cid, _) in enumerate(state.clip_files.items()):
                    clips.append({
                        "id": cid, "filename": f"{cid}.mp4", "start_time": f"00:0{n}:10",
                        "end_time": f"00:0{n}:40", "text": f"Das ist der spannende Moment Nummer {n}. Wahnsinn!",
                        "virality_score": 80 - n * 15, "hook_score": 22 - n * 4, "engagement_score": 20 - n * 4,
                        "value_score": 18 - n * 3, "shareability_score": 20 - n * 4,
                        "hook_title": f"Moment {n} schockiert alle", "reasoning": "strong hook",
                    })
                return self._json(200, {"task_id": "t1", "clips": clips, "total_clips": len(clips)})
            if p.startswith("/tasks/t1/clips/") and p.endswith("/file"):
                cid = p.split("/")[4]
                data = state.clip_files[cid].read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self._json(404, {"detail": "not found"})

        def do_POST(self):
            if not self._authed():
                return
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length)
            if self.path == "/upload":
                state.uploads += 1
                return self._json(200, {"video_path": "upload://abc.mp4"})
            if self.path == "/tasks/":
                state.created.append(json.loads(body))
                return self._json(200, {"task_id": "t1", "job_id": "j1"})
            self._json(404, {"detail": "not found"})

    return H


@pytest.fixture
def fake_backend(clip_factory):
    clips = {
        "c0aaaaaaaaaa": clip_factory("c0.mp4", seconds=18),                                          # defects, fixable
        "c1bbbbbbbbbb": clip_factory("c1.mp4", size="1080x1920", black_start=0, gap=None, volume=0.2),
        "c2cccccccccc": clip_factory("c2.mp4", size="1080x1920", black_start=0, gap=None, volume=0.2),
    }
    state = FakeSupoClip(clips)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield state, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_local_file_end_to_end(fake_backend, settings, clip_factory, monkeypatch):
    state, url = fake_backend
    settings.api_url, settings.user_id, settings.auth_secret = url, USER, SECRET
    monkeypatch.setattr("autoclipper.supoclip_api.time.sleep", lambda s: None)
    source = clip_factory("my video.mp4", size="1920x1080", black_start=0, gap=None)

    result = pipeline.process(str(source), settings, title="Mein Testvideo", use_ollama=False)

    assert state.uploads == 1 and state.bad_auth == 0
    body = state.created[0]
    assert body["source"]["url"] == "upload://abc.mp4"
    assert body["output_format"] == "vertical" and body["caption_template"] == "highlighter_swipe"
    assert body["edit_style"]["intensity"] == "bold" and body["edit_style"]["music"] is False
    assert body["cut_long_pauses"] and body["remove_filler_words"]

    project = Path(result["project"])
    assert project.parent == Path(settings.output_root)
    names = sorted(p.name for p in project.glob("*.mp4"))
    assert names and all(n[:2].isdigit() for n in names), names
    for f in ("REPORT.md", "clips.json", "posting_plan.csv", "source.json", "texts/01_tiktok.txt"):
        assert (project / f).exists(), f
    report = (project / "REPORT.md").read_text(encoding="utf-8")
    assert "Virality Score" in report and "Original-Zeitstempel" in report
    # Clip 0 had fixable defects: repaired and kept at 1080x1920.
    from autoclipper import qa
    for p in project.glob("*.mp4"):
        r = qa.check(p)
        assert (r.width, r.height) == (1080, 1920) and r.status != "fail", (p, r.issues)
    for c in result["clips"]:
        assert Path(c["contact_sheet"]).exists() and Path(c["hook_strip"]).exists()


def test_finalize_applies_review(fake_backend, settings, clip_factory, monkeypatch):
    state, url = fake_backend
    settings.api_url, settings.user_id, settings.auth_secret = url, USER, SECRET
    monkeypatch.setattr("autoclipper.supoclip_api.time.sleep", lambda s: None)
    source = clip_factory("v.mp4", size="1920x1080", black_start=0, gap=None)
    result = pipeline.process(str(source), settings, title="Finalize", use_ollama=False)
    project = Path(result["project"])

    meta = {
        "c2cccccccccc": {"shorts_title": "Niemand hat DAS erwartet", "tiktok_description": "Warte bis zum Ende 😳",
                         "hashtags": ["#podcast", "talk"], "why": "Überraschung",
                         "review": {"hook": 95, "retention": 90, "emotion": 90, "entertainment": 90, "virality": 95}},
        "c0aaaaaaaaaa": {"drop": True, "drop_reason": "braucht Kontext"},
    }
    meta_file = project / "_work" / "review.json"
    meta_file.write_text(json.dumps(meta), encoding="utf-8")
    out = pipeline.finalize(project, meta_file, settings)

    assert out["clips"][0]["title"] == "Niemand hat DAS erwartet"
    assert out["clips"][0]["score"] >= 90
    assert any(r["reason"] == "braucht Kontext" for r in out["rejected"])
    assert (project / "_rejected").exists()
    txt = (project / "texts" / "01_tiktok.txt").read_text(encoding="utf-8")
    assert "#talk" in txt and "Warte bis zum Ende" in txt
    # A later review is a delta: undoing the drop restores the clip, earlier ratings stay.
    meta_file.write_text(json.dumps({"c0aaaaaaaaaa": {"drop": False}}), encoding="utf-8")
    again = pipeline.finalize(project, meta_file, settings)
    assert len(again["clips"]) == 3, again["rejected"]
    assert again["clips"][0]["title"] == "Niemand hat DAS erwartet" and again["clips"][0]["score"] >= 90
    assert not any((project / "_rejected").glob("*.mp4"))


def test_scoring_and_order():
    items = [
        {"score": {"final": 70}, "source_start": 10},
        {"score": {"final": 68}, "source_start": 20},
        {"score": {"final": 66}, "source_start": 500},
        {"score": {"final": 40}, "source_start": 900},
    ]
    order = package.posting_order(items)
    assert order[0]["score"]["final"] == 70
    assert order[1]["source_start"] == 500  # not the neighbouring moment
    s = package.score_clip({"hook_score": 25, "engagement_score": 25, "value_score": 25, "shareability_score": 25})
    assert s["final"] == 100
    assert package.score_clip({"virality_score": 64}, qa_penalty=4)["final"] == 60
    assert package.hms(3725.4) == "01:02:05.4"
    assert package.parse_time("01:02:05.5") == 3725.5
