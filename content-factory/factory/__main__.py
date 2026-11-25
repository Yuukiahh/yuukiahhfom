"""CLI:  python -m factory <command>

  doctor                      check installation (ffmpeg, GPU encoder, LLMs, TTS, secrets)
  trends [--no-llm]           refresh trend data and list top topics
  video "<Thema>"             EIN Befehl: recherchieren, Skript, Stimme, 3D-Animation, Schnitt -> fertige MP4
  make "<topic>" [--format short|long] [--lang de|en] [--script file.json]
  mascot build|pack|status    3D-Maskottchen in Blender rendern / packen
  worker                      run the background worker (start state 'running')
  start | pause | stop        control the worker
  queue "<topic>" [...]       add a job
  dashboard                   start the dashboard on http://127.0.0.1:8765
  secrets list|set <name>|delete <name>
  publish enable|disable|status|approve <video_id>
  analytics                   fetch real metrics + update learnings
  obsidian                    sync project notes into the Obsidian vault
"""
from __future__ import annotations

import argparse
import getpass
from pathlib import Path
import json
import shutil
import subprocess
import sys


def doctor() -> int:
    from . import llm, secrets
    from .config import cfg
    from .render import video_encoder

    ok = True

    def line(name, good, detail=""):
        nonlocal ok
        ok &= bool(good) or name.startswith("(optional)")
        print(f"  [{'OK' if good else '--'}] {name:34s} {detail}")

    print("ContentFactory – Systemcheck")
    line("ffmpeg", shutil.which("ffmpeg"), shutil.which("ffmpeg") or "fehlt")
    enc = video_encoder()
    line("(optional) NVENC GPU-Encoder", "h264_nvenc" in enc, enc[1])
    try:
        import onnxruntime as ort
        prov = ort.get_available_providers()
        line("(optional) ONNX CUDA (Kokoro auf GPU)", "CUDAExecutionProvider" in prov, ",".join(prov))
    except ImportError:
        line("onnxruntime", False, "fehlt")
    line("Kokoro Modell", cfg().path("tts.kokoro_model").exists(), str(cfg().path("tts.kokoro_model")))
    for k, v in llm.available().items():
        line(f"(optional) LLM {k}", v)
    if not any(llm.available().values()):
        line("mindestens ein LLM", False, "claude CLI oder Ollama installieren")
    if shutil.which("nvidia-smi"):
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        line("(optional) GPU", True, out)
    for k, v in secrets.status().items():
        if not k.endswith("_token"):
            line(f"(optional) Key {k}", v)
    print("Ergebnis:", "bereit" if ok else "es fehlt noch etwas (siehe oben)")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="factory")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    t = sub.add_parser("trends")
    t.add_argument("--no-llm", action="store_true")
    for name in ("make", "queue", "video"):
        m = sub.add_parser(name)
        m.add_argument("topic", nargs="?")
        m.add_argument("--format", default="short", help="short (25-45 s) | long (60-90 s) | exakte Länge z.B. 30s")
        m.add_argument("--lang", default=None)
        m.add_argument("--niche", default=None)
        m.add_argument("--preset", default=None)
        m.add_argument("--voice", default=None)
        m.add_argument("--script", default=None, help="fertiges Skript-JSON verwenden")
        m.add_argument("--media", default=None, help="Ordner mit eigenen Fotos (z.B. Fotos deiner Karten) – werden im Video verwendet")
    sub.add_parser("worker")
    for s in ("start", "pause", "stop", "status"):
        sub.add_parser(s)
    sub.add_parser("dashboard")
    sp = sub.add_parser("secrets")
    sp.add_argument("action", choices=["list", "set", "delete"])
    sp.add_argument("name", nargs="?")
    pp = sub.add_parser("publish")
    pp.add_argument("action", choices=["enable", "disable", "status", "approve", "reject", "connect", "run"])
    pp.add_argument("arg", nargs="?")
    sub.add_parser("analytics")
    sub.add_parser("obsidian")
    mp = sub.add_parser("mascot")
    mp.add_argument("action", choices=["build", "pack", "status"])
    mp.add_argument("--size", type=int, default=512)
    a = ap.parse_args(argv)

    if a.cmd == "doctor":
        return doctor()
    if a.cmd == "trends":
        from .trends import engine
        print(json.dumps(engine.refresh(use_llm=not a.no_llm), indent=2, default=str)[:3000])
        for t in engine.top_topics(20):
            print(f"{t['score']:.3f}  [{t['niche']}]  {t['title']}")
        return 0
    if a.cmd == "mascot":
        from . import mascot
        from .config import cfg
        frames = cfg().path("general.data_dir") / "mascot_frames"
        if a.action == "build":
            import shutil as _sh
            exe = _sh.which("blender") or r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"
            subprocess.run([exe, "-b", "--factory-startup", "--python", str(Path(mascot.__file__).with_name("build_mascot.py")),
                            "--", str(frames), "--size", str(a.size)], check=True)
        if a.action in ("build", "pack"):
            print([str(x) for x in mascot.pack(frames)])
        print("Maskottchen-Assets vorhanden:", mascot.available())
        return 0
    if a.cmd == "video":
        a.cmd = "make"
    if a.cmd in ("make", "queue"):
        from . import pipeline, worker
        script = json.load(open(a.script, encoding="utf-8")) if a.script else None
        topic = a.topic or (script or {}).get("topic")
        if not topic:
            ap.error("Thema oder --script angeben")
        if a.cmd == "queue":
            print("Job", worker.enqueue(topic, a.niche, a.format, a.lang, a.preset, a.voice, {"script": script} if script else None))
            return 0
        from .config import cfg
        lang = a.lang or (script or {}).get("language") or cfg().get("general.language")
        if not script:
            from .trends.engine import classify
            script = pipeline.prepare_script(topic, a.niche or classify(topic)[0] or "facts", a.format, lang,
                                             lambda s, p: print(f"[{p:4.0%}] {s}", flush=True))
        if a.voice:
            script["voice"] = a.voice
        if a.media:
            script["user_media"] = str(Path(a.media).resolve())
        if a.preset:
            script["preset"] = a.preset
        res = pipeline.produce(script, progress=lambda s, p: print(f"[{p:4.0%}] {s}", flush=True))
        print(json.dumps({k: res[k] for k in ("path", "render_seconds")}, indent=2))
        print("QC:", "BESTANDEN" if res["qc"]["passed"] else "FEHLER", res["qc"]["score"])
        return 0 if res["qc"]["passed"] else 2
    if a.cmd == "worker":
        from . import worker
        worker.run_forever()
        return 0
    if a.cmd in ("start", "pause", "stop"):
        from . import worker
        print(worker.set_state({"start": "running", "pause": "paused", "stop": "stopped"}[a.cmd]))
        return 0
    if a.cmd == "status":
        from . import worker
        print(json.dumps(worker.status(), indent=2, default=str))
        return 0
    if a.cmd == "dashboard":
        from dashboard.app import main as dash
        dash()
        return 0
    if a.cmd == "secrets":
        from . import secrets
        if a.action == "list":
            for k, v in secrets.status().items():
                print(f"  {'✓' if v else '·'} {k:24s} {secrets.KNOWN[k]}")
        elif a.action == "set":
            if a.name == "youtube_oauth_client":
                path = input("Pfad zur client_secret_*.json: ").strip().strip('"')
                secrets.set(a.name, open(path, encoding="utf-8").read())
            else:
                secrets.set(a.name, getpass.getpass(f"{a.name}: "))
            print("gespeichert (verschlüsselt / Windows-Anmeldeinformationsverwaltung)")
        else:
            secrets.delete(a.name)
        return 0
    if a.cmd == "publish":
        from . import publish
        if a.action == "enable":
            print("ACHTUNG: Damit werden freigegebene Videos automatisch auf verbundenen Konten veröffentlicht.")
            ok = input("Zum Bestätigen 'JA VERÖFFENTLICHEN' eingeben: ").strip() == "JA VERÖFFENTLICHEN"
            publish.enable(ok)
            print("aktiviert" if ok else "abgebrochen")
        elif a.action == "disable":
            from .config import cfg
            cfg().set_local("publish.enabled", False)
            print("deaktiviert")
        elif a.action == "status":
            print(json.dumps(publish.publish_status(), indent=2))
        elif a.action == "approve":
            publish.approve(int(a.arg))
        elif a.action == "reject":
            publish.reject(int(a.arg))
        elif a.action == "connect":
            import importlib
            importlib.import_module(f"factory.publish.{a.arg}").connect()
        elif a.action == "run":
            print(publish.run_due())
        return 0
    if a.cmd == "analytics":
        from . import analytics
        print(json.dumps(analytics.refresh(), indent=2, default=str))
        return 0
    if a.cmd == "obsidian":
        from . import obsidian
        print(obsidian.sync())
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
