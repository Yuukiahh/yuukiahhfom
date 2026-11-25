"""Text-to-speech with word-level timestamps and an engine fallback chain.

Engines (config tts.engine.<lang> is a list, first working engine wins):
  elevenlabs : most natural (premium). REST API /with-timestamps -> exact per-character
               alignment. Needs secret `elevenlabs_api_key`; usage is capped by
               tts.elevenlabs_monthly_chars and logged in the cost overview.
  edge       : Microsoft Edge neural voices (free, online, very natural German e.g.
               de-DE-SeraphinaMultilingualNeural / FlorianMultilingualNeural). Native word boundaries.
  coqui      : Coqui VITS "Thorsten" (free, offline, German). Runs in a worker process
               (tts.coqui_python) so torch stays out of the main process.
  kokoro     : Kokoro ONNX (free, offline, English).
Display words (subtitles) and spoken words (TTS) are mapped 1:1 via factory.textnorm, so
numbers/units are spoken correctly while subtitles show digits.
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

from . import secrets, textnorm
from .config import ROOT, cfg

SR = 24000


@dataclass
class Word:
    text: str
    start: float
    end: float
    scene: int = 0


class TTSError(RuntimeError):
    pass


# ------------------------------------------------------------------ helpers
def _syllables(w: str) -> float:
    w = re.sub(r"[^a-zäöüß0-9 ]", "", w.lower())
    if not w.strip():
        return 0.3
    groups = re.findall(r"[aeiouyäöü]+", w)
    return max(1, len(groups)) + 0.06 * len(w)


def _phrases(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?…;:,—])\s+", text.strip())
    out: list[str] = []
    for p in parts:
        if out and len(p.split()) <= 2 and not re.search(r"[.!?]$", out[-1]):
            out[-1] += " " + p
        else:
            out.append(p)
    return [p for p in out if p.strip()]


def _trim(audio: np.ndarray, sr: int, thresh: float = 0.012) -> np.ndarray:
    idx = np.where(np.abs(audio) > thresh)[0]
    if len(idx) == 0:
        return audio
    pad = int(0.02 * sr)
    return audio[max(0, idx[0] - pad): min(len(audio), idx[-1] + pad)]


def _resample(a: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return a
    x = np.linspace(0, len(a) - 1, int(len(a) * SR / sr))
    return np.interp(x, np.arange(len(a)), a).astype(np.float32)


def _gap_after(text: str, last_in_scene: bool) -> float:
    if last_in_scene:
        return 0.32
    c = text.rstrip()[-1:]
    return 0.26 if c in ".!?…" else 0.12 if c in ",;:—" else 0.05


def _assemble_phrase_audio(scenes: list[str], lang: str, synth_phrases, out_wav: Path):
    """Generic phrase-level assembly for engines without word timestamps."""
    words, spans, chunks, t = [], [], [], 0.0
    plan = [(si, ph) for si, sc in enumerate(scenes) for ph in _phrases(sc)]
    audios = synth_phrases([textnorm.spoken_text(ph, lang) for _, ph in plan])
    scene_start = {}
    for k, ((si, ph), audio) in enumerate(zip(plan, audios)):
        scene_start.setdefault(si, t)
        audio = _trim(audio, SR)
        dur = len(audio) / SR
        disp = ph.split()
        spoken = textnorm.spoken_words(disp, lang)
        weights = np.array([_syllables(s) for s in spoken])
        usable = max(0.05, dur - 0.04)
        cum = np.concatenate([[0], np.cumsum(weights)]) / weights.sum() * usable + 0.02
        for wi, w in enumerate(disp):
            words.append(Word(w, round(t + cum[wi], 3), round(t + cum[wi + 1], 3), si))
        chunks.append(audio)
        t += dur
        last = k + 1 == len(plan) or plan[k + 1][0] != si
        gap = _gap_after(ph, last)
        chunks.append(np.zeros(int(gap * SR), np.float32))
        t += gap
        if last:
            spans.append((scene_start[si], t))
    sf.write(out_wav, np.concatenate(chunks), SR, subtype="PCM_16")
    return words, spans


# ------------------------------------------------------------------ kokoro (en)
@lru_cache(maxsize=1)
def _kokoro():
    import kokoro_onnx

    model, voices = cfg().path("tts.kokoro_model"), cfg().path("tts.kokoro_voices")
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers():
            sess = ort.InferenceSession(str(model), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            return kokoro_onnx.Kokoro.from_session(sess, str(voices))
    except Exception:
        pass
    return kokoro_onnx.Kokoro(str(model), str(voices))


def synth_kokoro(scenes, voice, speed, lang, out_wav):
    k = _kokoro()
    kl = "en-gb" if voice[:1] == "b" else "en-us"

    def run(phr):
        return [k.create(p, voice=voice, speed=speed, lang=kl)[0].astype(np.float32) for p in phr]
    return _assemble_phrase_audio(scenes, lang, run, out_wav)


# ------------------------------------------------------------------ coqui (de, offline)
COQUI_WORKER = Path(__file__).with_name("tts_coqui_worker.py")


def synth_coqui(scenes, voice, speed, lang, out_wav):
    py = cfg().get("tts.coqui_python") or sys.executable
    py = str(ROOT / py) if py and not Path(py).is_absolute() and "/" in py else py
    model_dir = cfg().path("tts.coqui_model_dir")
    if not (model_dir / "model_file.pth").exists():
        raise TTSError(f"Coqui-Modell fehlt: {model_dir}")

    def run(phr):
        with tempfile.TemporaryDirectory() as td:
            req = {"model_dir": str(model_dir), "phrases": phr, "out_dir": td, "speed": speed}
            p = subprocess.run([py, "-I", str(COQUI_WORKER)], input=json.dumps(req), capture_output=True, text=True,
                               timeout=1800)
            if p.returncode != 0:
                raise TTSError("Coqui-Worker: " + p.stderr[-800:])
            res = json.loads(p.stdout.strip().splitlines()[-1])
            out = []
            for f in res["files"]:
                a, sr = sf.read(f, dtype="float32")
                out.append(_resample(a, sr))
            return out
    return _assemble_phrase_audio(scenes, lang, run, out_wav)


# ------------------------------------------------------------------ edge (free, online)
async def _edge_async(text: str, voice: str, rate: str, out_mp3: Path):
    import edge_tts

    comm = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
    marks = []
    with open(out_mp3, "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                marks.append((chunk["offset"] / 1e7, (chunk["offset"] + chunk["duration"]) / 1e7))
    return marks


def _decode(path: Path) -> np.ndarray:
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.float32).copy()


def _map_marks(disp: list[str], spoken: list[str], marks: list[tuple[float, float]], t0: float, dur: float):
    """Spoken tokens may be more than display words (16,1 -> sechzehn Komma eins)."""
    counts = [max(1, len(s.split())) if s else 0 for s in spoken]
    out, k = [], 0
    for w, c in zip(disp, counts):
        if c == 0 or k >= len(marks):
            prev_end = out[-1][2] if out else t0
            out.append((w, prev_end, prev_end + 0.05))
            continue
        seg = marks[k:k + c]
        out.append((w, t0 + seg[0][0], t0 + seg[-1][1]))
        k += c
    return out


def synth_edge(scenes, voice, speed, lang, out_wav):
    rate = f"{int(round((speed - 1) * 100)):+d}%"
    words, spans, pieces, t = [], [], [], 0.0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "part.mp3"
        for si, scene_text in enumerate(scenes):
            disp = scene_text.split()
            spoken = textnorm.spoken_words(disp, lang)
            marks = asyncio.run(_edge_async(" ".join(s for s in spoken if s), voice, rate, tmp))
            a = _decode(tmp)
            for w, s, e in _map_marks(disp, spoken, marks, t, len(a) / SR):
                words.append(Word(w, round(s, 3), round(e, 3), si))
            gap = np.zeros(int(0.3 * SR), np.float32)
            pieces += [a, gap]
            spans.append((t, t + len(a) / SR + 0.3))
            t += len(a) / SR + 0.3
    sf.write(out_wav, np.concatenate(pieces), SR, subtype="PCM_16")
    return words, spans


# ------------------------------------------------------------------ elevenlabs (premium)
def _el_month_usage() -> int:
    from . import db
    r = db.row("SELECT SUM(units) n FROM costs WHERE service='elevenlabs' AND created_at > ?", time.time() - 30 * 86400)
    return int((r or {}).get("n") or 0)


def synth_elevenlabs(scenes, voice, speed, lang, out_wav):
    import requests

    key = secrets.get("elevenlabs_api_key")
    if not key:
        raise TTSError("elevenlabs_api_key fehlt")
    cap = int(cfg().get("tts.elevenlabs_monthly_chars", 0) or 0)
    chars = sum(len(textnorm.spoken_text(s, lang)) for s in scenes)
    if cap <= 0 or _el_month_usage() + chars > cap:
        raise TTSError(f"ElevenLabs-Zeichenbudget erreicht/deaktiviert (cap={cap})")
    model = cfg().get("tts.elevenlabs_model", "eleven_multilingual_v2")
    words, spans, pieces, t = [], [], [], 0.0
    spoken_scenes = [textnorm.spoken_text(s, lang) for s in scenes]
    for si, scene_text in enumerate(scenes):
        disp = scene_text.split()
        spoken = textnorm.spoken_words(disp, lang)
        text = " ".join(s for s in spoken if s)
        r = requests.post(f"https://api.elevenlabs.io/v1/text-to-speech/{voice}/with-timestamps",
                          params={"output_format": "mp3_44100_128"}, timeout=120,
                          headers={"xi-api-key": key, "Content-Type": "application/json"},
                          json={"text": text, "model_id": model, "language_code": lang,
                                "previous_text": " ".join(spoken_scenes[max(0, si - 2):si])[-600:] or None,
                                "next_text": (spoken_scenes[si + 1] if si + 1 < len(scenes) else None),
                                "voice_settings": {"stability": 0.42, "similarity_boost": 0.8, "style": 0.35,
                                                   "use_speaker_boost": True, "speed": max(0.7, min(1.2, speed))}})
        if r.status_code != 200:
            raise TTSError(f"ElevenLabs HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(base64.b64decode(data["audio_base64"]))
            mp3 = Path(f.name)
        a = _decode(mp3)
        mp3.unlink(missing_ok=True)
        al = data.get("alignment") or data.get("normalized_alignment") or {}
        chars_, st, en = al.get("characters", []), al.get("character_start_times_seconds", []), al.get("character_end_times_seconds", [])
        marks, cur = [], None
        for ch, s0, e0 in zip(chars_, st, en):
            if ch.isspace():
                if cur:
                    marks.append(cur)
                    cur = None
            else:
                cur = (cur[0], e0) if cur else (s0, e0)
        if cur:
            marks.append(cur)
        for w, s, e in _map_marks(disp, spoken, marks, t, len(a) / SR):
            words.append(Word(w, round(s, 3), round(e, 3), si))
        gap = np.zeros(int(0.25 * SR), np.float32)
        pieces += [a, gap]
        spans.append((t, t + len(a) / SR + 0.25))
        t += len(a) / SR + 0.25
    from . import costs
    costs.record("elevenlabs", 0.0, chars, "characters (im Abo enthalten)")
    sf.write(out_wav, np.concatenate(pieces), SR, subtype="PCM_16")
    return words, spans


ENGINES = {"elevenlabs": synth_elevenlabs, "edge": synth_edge, "coqui": synth_coqui, "kokoro": synth_kokoro}


def synthesize(scenes: list[str], out_wav: Path, lang: str | None = None, voice: str | None = None):
    lang = lang or cfg().get("general.language", "de")
    chain = cfg().lang_value("tts.engine", lang)
    chain = [chain] if isinstance(chain, str) else list(chain)
    voices = cfg().get("tts.voices", {}) or {}
    speed = float(cfg().get("tts.speed", 1.0))
    errors = []
    for engine in chain:
        fn = ENGINES.get(engine)
        if not fn:
            continue
        v = voice if (voice and engine == (cfg().get("tts.voice_engine_hint") or engine)) else None
        v = v or (voices.get(engine, {}) or {}).get(lang) or voice or ""
        try:
            words, spans = fn(scenes, v, speed, lang, out_wav)
            return [asdict(w) for w in words], spans, {"engine": engine, "voice": v, "fallback_errors": errors}
        except Exception as e:
            errors.append(f"{engine}: {type(e).__name__}: {str(e)[:200]}")
    raise TTSError("Keine TTS-Engine verfügbar: " + " | ".join(errors))


VOICES = {
    "elevenlabs": ["IWm8DnJ4NGjFI7QAM5lM (Stephan – Infotainment)", "FTNCalFNG5bRnkkaP5Ug (Otto – ruhig)",
                   "kkJxCnlRCckmfFvzDW5Q (Alexander – tief)", "AnvlJBAqSLDzEevYr9Ap (Ava – jung)",
                   "uvysWDLbKpA4XvpD3GI6 (Leonie)"],
    "edge": ["de-DE-FlorianMultilingualNeural", "de-DE-SeraphinaMultilingualNeural", "de-DE-ConradNeural",
             "de-DE-KillianNeural", "de-DE-KatjaNeural", "de-DE-AmalaNeural", "en-US-AndrewMultilingualNeural"],
    "coqui": ["thorsten (de)"],
    "kokoro": ["am_michael", "am_onyx", "am_adam", "af_heart", "af_bella", "bm_george"],
}
