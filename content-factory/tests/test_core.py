"""Core unit tests (no network, fast)."""
import json
import os
import tempfile
from pathlib import Path

import numpy as np

TMP = tempfile.mkdtemp()
cfg_file = Path(TMP) / "test.toml"
cfg_file.write_text(f'[general]\ndata_dir = "{TMP}/data"\noutput_dir = "{TMP}/out"\n'.replace("\\", "/"))
os.environ["CF_CONFIG"] = str(cfg_file)

from factory import audio, dedupe, learning, llm, metadata, scriptwriter, subtitles  # noqa: E402
from factory.trends import engine  # noqa: E402
from factory.visuals.procedural import KINDS, make_scene  # noqa: E402


def test_extract_json_variants():
    assert llm.extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm.extract_json('<think>x</think> here: {"a": {"b": "}"}} trailing') == {"a": {"b": "}"}}


def test_classify_niches():
    assert engine.classify("Voyager 1 reaches one light-day from Earth")[0] == "space"
    assert engine.classify("New GPT model beats benchmark")[0] == "ai"
    assert engine.classify("Nintendo announces new Zelda")[0] == "gaming"


def test_dedupe_similarity():
    assert dedupe.similarity("What happens if you fall into a black hole", "Falling into a black hole: what happens?") > 0.5
    assert dedupe.similarity("black hole facts", "quantum computer chip") == 0


def test_all_scenes_render():
    for kind in set(KINDS.values()):
        name = next(k for k, v in KINDS.items() if v is kind)
        s = make_scene(name, 108, 192, 1)
        f = s.render(0.5)
        assert f.shape[:2] == (s.H, s.W) and np.isfinite(f).all()


def test_subtitles_fit_and_groups(tmp_path):
    words = [{"text": w, "start": i * 0.3, "end": i * 0.3 + 0.28, "scene": 0}
             for i, w in enumerate("That difference stretches you thinner spaghettification".split())]
    for g in subtitles.group_words(words, 3):
        assert sum(len(w["text"]) + 1 for w in g) - 1 <= 16 or len(g) == 1
    out = subtitles.build_ass(words, tmp_path / "s.ass", ["spaghettification"], [{"text": "4,000,000×", "sub": "x", "start": 0, "end": 1}], "HOOK")
    txt = out.read_text()
    assert "Dialogue" in txt and "\\fscx" in txt


def test_audio_lengths(tmp_path):
    for d, mood in ((0.3, "cosmic"), (2.77, "pulse"), (6.1, "mystery")):  # remainders shorter than the pad attack
        assert len(audio.music(d, mood)) == int(d * audio.SR)
    m = audio.music(3.0, "pulse")
    assert abs(len(m) / audio.SR - 3.0) < 0.01 and np.abs(m).max() <= 0.81
    lib = audio.build_sfx_library(tmp_path / "sfx")
    assert set(lib) == set(audio.SFX)


def test_script_validation_and_metadata():
    s = {"title": "T", "hook": "Hook here", "niche": "space", "format": "short", "language": "en",
         "scenes": [{"narration": "Hook here and " + "word " * 80, "visual": "nonexistent"}],
         "claims": [{"claim": "c", "source": "https://x"}], "metadata": {"title": "Title", "hashtags": ["#Space", "space"]}}
    s = scriptwriter.normalise(s, "short", "en")
    assert s["scenes"][0]["visual"] in KINDS
    assert scriptwriter.validate(s) == []
    m = metadata.build(s)
    assert m["hashtags"].count("space") == 1 and "#Shorts" in m["youtube"]["title"]


def test_learning_needs_real_data():
    res = learning.analyse()
    assert res["status"] == "insufficient_data"
    assert learning.prompt_hints() == ""
