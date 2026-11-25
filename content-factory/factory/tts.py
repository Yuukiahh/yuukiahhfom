"""Text-to-speech with word-level timestamps.

kokoro : local ONNX model (free, offline, GPU via onnxruntime-gpu). Synthesised phrase
         by phrase; phrase boundaries are exact, word timings inside a phrase are
         distributed by phoneme weight (optional faster-whisper re-alignment).
edge   : Microsoft Edge read-aloud voices (free, online) – native WordBoundary events.
"""
from __future__ import annotations

import asyncio
import re
import subprocess
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import cfg

SR = 24000


@dataclass
class Word:
    text: str
    start: float
    end: float
    scene: int = 0


def _syllables(w: str) -> float:
    w = re.sub(r"[^a-zäöüß0-9]", "", w.lower())
    if not w:
        return 0.3
    if w.isdigit():
        return 1.2 * len(w)
    groups = re.findall(r"[aeiouyäöü]+", w)
    return max(1, len(groups)) + 0.08 * len(w)


def _phrases(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?…;:,—])\s+", text.strip())
    out: list[str] = []
    for p in parts:
        if out and len(p.split()) <= 2 and not re.search(r"[.!?]$", out[-1]):
            out[-1] += " " + p
        else:
            out.append(p)
    return [p for p in out if p.strip()]


def _trim(audio: np.ndarray, thresh: float = 0.012) -> np.ndarray:
    idx = np.where(np.abs(audio) > thresh)[0]
    if len(idx) == 0:
        return audio
    pad = int(0.025 * SR)
    return audio[max(0, idx[0] - pad): min(len(audio), idx[-1] + pad)]


@lru_cache(maxsize=1)
def _kokoro():
    import kokoro_onnx

    providers = None
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass
    model, voices = cfg().path("tts.kokoro_model"), cfg().path("tts.kokoro_voices")
    if providers:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(model), providers=providers)
        return kokoro_onnx.Kokoro.from_session(sess, str(voices))
    return kokoro_onnx.Kokoro(str(model), str(voices))


def synth_kokoro(scenes: list[str], voice: str, speed: float, out_wav: Path) -> tuple[list[Word], list[tuple[float, float]]]:
    k = _kokoro()
    lang = "en-gb" if voice[:1] == "b" else "en-us"
    chunks, words, scene_spans, t = [], [], [], 0.0
    for si, scene_text in enumerate(scenes):
        s_start = t
        phrases = _phrases(scene_text)
        for pi, ph in enumerate(phrases):
            audio, _ = k.create(ph, voice=voice, speed=speed, lang=lang)
            audio = _trim(audio.astype(np.float32))
            dur = len(audio) / SR
            toks = ph.split()
            weights = np.array([_syllables(w) for w in toks])
            # phrase-internal onset/offset trim compensation
            usable = max(0.05, dur - 0.05)
            cum = np.concatenate([[0], np.cumsum(weights)]) / weights.sum() * usable + 0.025
            for wi, w in enumerate(toks):
                words.append(Word(w, round(t + cum[wi], 3), round(t + cum[wi + 1], 3), si))
            chunks.append(audio)
            t += dur
            end_char = ph.rstrip()[-1:]
            gap = 0.30 if end_char in ".!?…" else 0.14 if end_char in ",;:—" else 0.06
            if pi == len(phrases) - 1:
                gap = 0.38
            chunks.append(np.zeros(int(gap * SR), np.float32))
            t += gap
        scene_spans.append((s_start, t))
    audio = np.concatenate(chunks)
    sf.write(out_wav, audio, SR, subtype="PCM_16")
    return words, scene_spans


async def _edge_async(text: str, voice: str, rate: str, out_mp3: Path):
    import edge_tts

    comm = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
    marks = []
    with open(out_mp3, "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                marks.append((chunk["text"], chunk["offset"] / 1e7, (chunk["offset"] + chunk["duration"]) / 1e7))
    return marks


def synth_edge(scenes: list[str], voice: str, speed: float, out_wav: Path):
    rate = f"{int(round((speed - 1) * 100)):+d}%"
    words, spans, pieces, t = [], [], [], 0.0
    tmp = out_wav.with_suffix(".part.mp3")
    for si, scene_text in enumerate(scenes):
        marks = asyncio.run(_edge_async(scene_text, voice, rate, tmp))
        wav_piece = out_wav.with_suffix(f".s{si}.wav")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-ar", str(SR), "-ac", "1", str(wav_piece)], check=True)
        a, _ = sf.read(wav_piece, dtype="float32")
        for w, s, e in marks:
            words.append(Word(w, round(t + s, 3), round(t + e, 3), si))
        gap = np.zeros(int(0.3 * SR), np.float32)
        pieces += [a, gap]
        spans.append((t, t + len(a) / SR + 0.3))
        t += len(a) / SR + 0.3
        wav_piece.unlink(missing_ok=True)
    tmp.unlink(missing_ok=True)
    sf.write(out_wav, np.concatenate(pieces), SR, subtype="PCM_16")
    return words, spans


def realign_whisper(wav: Path, words: list[Word]) -> list[Word]:
    """Optional: refine timings with faster-whisper (GPU) if installed."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return words
    try:
        model = WhisperModel("small", device="auto", compute_type="auto")
        segs, _ = model.transcribe(str(wav), word_timestamps=True)
        recog = [w for s in segs for w in (s.words or [])]
    except Exception:
        return words
    norm = lambda s: re.sub(r"[^a-z0-9äöüß]", "", s.lower())  # noqa: E731
    j = 0
    for w in words:
        for k in range(j, min(j + 4, len(recog))):
            if norm(recog[k].word) == norm(w.text):
                w.start, w.end, j = round(recog[k].start, 3), round(recog[k].end, 3), k + 1
                break
    return words


def synthesize(scenes: list[str], out_wav: Path, lang: str | None = None, voice: str | None = None):
    lang = lang or cfg().get("general.language", "en")
    engine = cfg().lang_value("tts.engine", lang)
    voice = voice or cfg().lang_value("tts.voice", lang)
    speed = float(cfg().get("tts.speed", 1.0))
    if engine == "edge":
        words, spans = synth_edge(scenes, voice, speed, out_wav)
    else:
        words, spans = synth_kokoro(scenes, voice, speed, out_wav)
        words = realign_whisper(out_wav, words)
    return [asdict(w) for w in words], spans, {"engine": engine, "voice": voice}


VOICES = {
    "kokoro": ["am_michael", "am_onyx", "am_adam", "am_fenrir", "am_puck", "af_heart", "af_bella",
               "af_nicole", "af_sarah", "bm_george", "bm_fable", "bm_daniel", "bf_emma"],
    "edge": ["de-DE-ConradNeural", "de-DE-KillianNeural", "de-DE-FlorianMultilingualNeural",
             "de-DE-KatjaNeural", "de-DE-SeraphinaMultilingualNeural", "en-US-AndrewMultilingualNeural",
             "en-US-GuyNeural", "en-US-AriaNeural"],
}
