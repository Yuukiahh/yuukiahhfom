"""Command line: autoclipper <command> ...

  doctor                       check GPU, ffmpeg, SupoClip, Ollama, output folder
  rights <url|file>            licence/rights check (JSON)
  process <url|file>           full run; exit code 3 = rights confirmation needed
  collect <project>            re-download / re-QA clips of an already finished task
  finalize <project> <meta>    apply Claude's review (titles, scores, drops) and rewrite the export
  qa <clip.mp4>...             quality report (JSON)
  fix <clip.mp4>...            QA + automatic repair in place
  sheet <clip.mp4>             contact sheet + hook strip PNGs
  config                       print effective settings (secrets masked)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__, media, qa
from .config import HOME_DIR, load_settings


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def cmd_doctor(args, settings) -> int:
    import httpx

    checks = {}
    checks["ffmpeg"] = shutil.which("ffmpeg") or "MISSING"
    checks["nvenc"] = media.nvenc_available() if shutil.which("ffmpeg") else False
    smi = shutil.which("nvidia-smi")
    if smi:
        out = subprocess.run([smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True)
        checks["gpu"] = out.stdout.strip() or out.stderr.strip()
    else:
        checks["gpu"] = "nvidia-smi not found"
    try:
        checks["supoclip"] = httpx.get(f"{settings.api_url}/health", timeout=10).json()
    except Exception as exc:
        checks["supoclip"] = f"UNREACHABLE: {exc}"
    try:
        tags = httpx.get(f"{settings.ollama_url}/api/tags", timeout=10).json()
        names = [m["name"] for m in tags.get("models", [])]
        checks["ollama"] = {"models": names, "configured_model_present": settings.ollama_model in names}
    except Exception as exc:
        checks["ollama"] = f"UNREACHABLE: {exc}"
    checks["credentials"] = "ok" if (settings.api_key or settings.user_id) else "MISSING (run the installer)"
    out = Path(settings.output_root)
    checks["output_root"] = f"{out} ({'exists' if out.exists() else 'will be created'})"
    if settings.user_id or settings.api_key:
        try:
            from .supoclip_api import SupoClip
            checks["dynamic_edits"] = SupoClip(settings).edit_styles().get("enabled")
        except Exception as exc:
            checks["dynamic_edits"] = f"unknown: {exc}"
    _print(checks)
    bad = [k for k, v in checks.items() if isinstance(v, str) and ("MISSING" in v or "UNREACHABLE" in v)]
    return 1 if bad else 0


def cmd_rights(args, settings) -> int:
    from .rights import inspect_source

    info = inspect_source(args.source, settings, confirmed=args.confirmed)
    _print({**info.to_dict(), "allowed": info.allowed})
    return 0 if info.allowed else 3


def cmd_process(args, settings) -> int:
    from .pipeline import RightsConfirmationNeeded, process

    if args.min_score is not None:
        settings.min_virality = args.min_score
    for key in ("intensity", "caption_template", "clip_length", "output_format"):
        value = getattr(args, key)
        if value:
            setattr(settings.task, key, value)
    if args.instructions:
        settings.task.instructions = args.instructions
    try:
        result = process(args.source, settings, confirmed=args.confirmed, title=args.title or "",
                         use_ollama=not args.no_ollama)
    except RightsConfirmationNeeded as exc:
        _print({"rights_confirmation_needed": True, "message": str(exc), "source": exc.info.to_dict()})
        return 3
    _print(result)
    return 0


def cmd_collect(args, settings) -> int:
    from .pipeline import collect

    _print(collect(Path(args.project), settings, use_ollama=not args.no_ollama))
    return 0


def cmd_finalize(args, settings) -> int:
    from .pipeline import finalize

    if args.min_score is not None:
        settings.min_virality = args.min_score
    _print(finalize(Path(args.project), Path(args.meta), settings))
    return 0


def cmd_qa(args, settings) -> int:
    reports = qa.check_many([Path(p) for p in args.clips], target_lufs=settings.target_lufs,
                            true_peak=settings.true_peak)
    print(qa.dump(reports))
    return 0 if all(r.status != "fail" for r in reports) else 2


def cmd_fix(args, settings) -> int:
    from .pipeline import quality_pass

    results = []
    for p in args.clips:
        path = Path(p)
        results.append(quality_pass(path, settings, path.parent / "review"))
    _print(results)
    return 0 if all(r["status"] != "fail" for r in results) else 2


def cmd_sheet(args, settings) -> int:
    from . import review

    clip = Path(args.clip)
    out = Path(args.out or clip.parent / "review")
    _print({"contact_sheet": str(review.contact_sheet(clip, out / f"{clip.stem}_sheet.png")),
            "hook_strip": str(review.hook_strip(clip, out / f"{clip.stem}_hook.png"))})
    return 0


def cmd_config(args, settings) -> int:
    _print({"home": str(HOME_DIR), **settings.to_dict()})
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
        except (AttributeError, ValueError):
            pass
    p = argparse.ArgumentParser(prog="autoclipper", description="AI clipping on a local SupoClip")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)
    sub.add_parser("config").set_defaults(fn=cmd_config)

    r = sub.add_parser("rights")
    r.add_argument("source")
    r.add_argument("--confirmed", action="store_true")
    r.set_defaults(fn=cmd_rights)

    pr = sub.add_parser("process")
    pr.add_argument("source", help="YouTube/video URL or local file")
    pr.add_argument("--confirmed", action="store_true", help="owner confirmed the rights to use this video")
    pr.add_argument("--title")
    pr.add_argument("--min-score", type=int)
    pr.add_argument("--intensity", choices=["off", "subtle", "bold", "max"])
    pr.add_argument("--caption-template")
    pr.add_argument("--clip-length", choices=["auto", "short", "medium", "long"])
    pr.add_argument("--output-format", choices=["vertical", "vertical_fit", "vertical_pan", "vertical_split"])
    pr.add_argument("--instructions")
    pr.add_argument("--no-ollama", action="store_true", help="skip local LLM metadata (heuristic only)")
    pr.set_defaults(fn=cmd_process)

    c = sub.add_parser("collect")
    c.add_argument("project")
    c.add_argument("--no-ollama", action="store_true")
    c.set_defaults(fn=cmd_collect)

    f = sub.add_parser("finalize")
    f.add_argument("project")
    f.add_argument("meta")
    f.add_argument("--min-score", type=int)
    f.set_defaults(fn=cmd_finalize)

    q = sub.add_parser("qa")
    q.add_argument("clips", nargs="+")
    q.set_defaults(fn=cmd_qa)

    fx = sub.add_parser("fix")
    fx.add_argument("clips", nargs="+")
    fx.set_defaults(fn=cmd_fix)

    s = sub.add_parser("sheet")
    s.add_argument("clip")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_sheet)

    args = p.parse_args(argv)
    settings = load_settings()
    try:
        return args.fn(args, settings)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        _print({"error": type(exc).__name__, "message": str(exc)})
        return 1


if __name__ == "__main__":
    sys.exit(main())
