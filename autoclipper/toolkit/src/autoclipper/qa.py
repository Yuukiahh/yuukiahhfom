"""Technical quality control for finished clips.

Every check returns issues with a code, a severity (warn / fail) and whether ``fix.repair`` can
repair it automatically. Things a machine cannot judge reliably (caption spelling, whether a
moment needs context, caption placement) are left to the visual review of the contact sheet.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import media

TARGET_W, TARGET_H = 1080, 1920
EDGE_EPS = 0.25          # seconds of black at start/end that counts as a bad cut
DEAD_AIR_SECONDS = 1.5   # internal silence longer than this is "boring"


@dataclass
class Issue:
    code: str
    severity: str          # "warn" | "fail"
    fixable: bool
    detail: str
    data: dict = field(default_factory=dict)


@dataclass
class QAReport:
    path: str
    status: str = "ok"     # ok | warn | fail
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    loudness_lufs: float | None = None
    true_peak: float | None = None
    faces_checked: int = 0
    issues: list[Issue] = field(default_factory=list)

    def add(self, issue: Issue) -> None:
        self.issues.append(issue)
        rank = {"ok": 0, "warn": 1, "fail": 2}
        if rank[issue.severity] > rank[self.status]:
            self.status = issue.severity

    @property
    def fixable_codes(self) -> set[str]:
        return {i.code for i in self.issues if i.fixable}

    @property
    def penalty(self) -> int:
        """Points removed from the virality score for problems that remain after repair."""
        return sum(15 if i.severity == "fail" else 4 for i in self.issues)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["penalty"] = self.penalty
        return data


def _face_check(path: Path, duration: float, samples: int = 16) -> tuple[int, list[Issue]]:
    try:
        import cv2  # type: ignore
    except ImportError:
        return 0, [Issue("FACE_CHECK_SKIPPED", "warn", False, "opencv not installed")]

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0, [Issue("FACE_CHECK_SKIPPED", "warn", False, "could not open video for face check")]

    frames_with_face = cut = off_center = 0
    cut_times: list[float] = []
    try:
        for k in range(samples):
            t = duration * (k + 0.5) / samples
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            min_side = max(40, int(w * 0.08))
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(min_side, min_side))
            if len(faces) == 0:
                continue
            frames_with_face += 1
            # Largest face = the speaker the reframe should follow.
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            margin = max(2, int(w * 0.01))
            if x <= margin or x + fw >= w - margin or y <= margin:
                cut += 1
                cut_times.append(round(t, 2))
            cx = (x + fw / 2) / w
            if len(faces) == 1 and not (0.18 <= cx <= 0.82):
                off_center += 1
    finally:
        cap.release()

    issues: list[Issue] = []
    if frames_with_face:
        if cut / frames_with_face >= 0.25:
            issues.append(Issue("FACE_CUT", "fail", False,
                                f"face touches the frame edge in {cut}/{frames_with_face} sampled frames",
                                {"times": cut_times}))
        elif cut:
            issues.append(Issue("FACE_CUT", "warn", False,
                                f"face touches the frame edge in {cut}/{frames_with_face} sampled frames",
                                {"times": cut_times}))
        if off_center / frames_with_face >= 0.4:
            issues.append(Issue("FACE_OFF_CENTER", "warn", False,
                                f"single face far from center in {off_center}/{frames_with_face} frames"))
    return frames_with_face, issues


def check(path: Path, *, target_lufs: float = -14.0, true_peak: float = -1.0,
          min_len: float = 12.0, max_len: float = 65.0, faces: bool = True) -> QAReport:
    path = Path(path)
    report = QAReport(path=str(path))
    try:
        info = media.probe(path)
    except media.MediaError as exc:
        report.add(Issue("UNREADABLE", "fail", False, str(exc)))
        return report

    v, a = media.stream(info, "video"), media.stream(info, "audio")
    container = float(info.get("format", {}).get("duration") or 0)
    report.duration = round(container, 3)

    if not v:
        report.add(Issue("NO_VIDEO", "fail", False, "no video stream"))
        return report
    report.width, report.height = int(v.get("width", 0)), int(v.get("height", 0))
    report.fps = round(media.fps_of(v), 3)

    if (report.width, report.height) != (TARGET_W, TARGET_H):
        report.add(Issue("RESOLUTION", "fail", True, f"{report.width}x{report.height}, expected 1080x1920"))
    if v.get("codec_name") != "h264" or v.get("pix_fmt") not in ("yuv420p", "yuvj420p"):
        report.add(Issue("CODEC", "warn", True, f"{v.get('codec_name')}/{v.get('pix_fmt')}, expected h264/yuv420p"))
    if report.fps and report.fps < 23:
        report.add(Issue("LOW_FPS", "warn", False, f"{report.fps} fps"))

    if container < min_len:
        report.add(Issue("TOO_SHORT", "fail", False, f"{container:.1f}s < {min_len}s"))
    elif container > max_len:
        report.add(Issue("TOO_LONG", "warn", False, f"{container:.1f}s > {max_len}s"))

    if not a:
        report.add(Issue("NO_AUDIO", "fail", False, "no audio stream"))
    else:
        vd, ad = media.stream_duration(v, container), media.stream_duration(a, container)
        v0, a0 = float(v.get("start_time") or 0), float(a.get("start_time") or 0)
        drift = max(abs(vd - ad), abs(v0 - a0))
        if drift > 0.12:
            report.add(Issue("AV_DESYNC", "fail" if drift > 0.3 else "warn", True,
                             f"audio/video differ by {drift:.2f}s", {"video": vd, "audio": ad}))

    # Black frames: at the edges they are bad cuts (trimmable), in the middle a render error.
    for start, end in media.black_segments(path):
        if start <= EDGE_EPS:
            report.add(Issue("BLACK_START", "fail", True, f"black {start:.2f}-{end:.2f}s", {"end": end}))
        elif end >= container - EDGE_EPS:
            report.add(Issue("BLACK_END", "fail", True, f"black {start:.2f}-{end:.2f}s", {"start": start}))
        else:
            report.add(Issue("BLACK_MIDDLE", "fail", False, f"black {start:.2f}-{end:.2f}s",
                             {"start": start, "end": end}))

    for start, end in media.freeze_segments(path):
        report.add(Issue("FROZEN", "warn", False, f"frozen picture {start:.2f}-{min(end, container):.2f}s"))

    if a:
        dead = [(s, min(e, container)) for s, e in media.silence_segments(path, min_dur=DEAD_AIR_SECONDS)
                if s > 0.3 and min(e, container) < container - 0.3]
        if dead:
            report.add(Issue("DEAD_AIR", "warn", True,
                             f"{len(dead)} silent gap(s) over {DEAD_AIR_SECONDS}s",
                             {"segments": [[round(s, 3), round(e, 3)] for s, e in dead]}))
        try:
            loud = media.loudness(path, target_lufs, true_peak)
            report.loudness_lufs = round(loud["input_i"], 2)
            report.true_peak = round(loud["input_tp"], 2)
            if abs(loud["input_i"] - target_lufs) > 1.5 or loud["input_tp"] > true_peak + 0.3:
                report.add(Issue("LOUDNESS", "warn", True,
                                 f"{loud['input_i']:.1f} LUFS / TP {loud['input_tp']:.1f} dB, "
                                 f"target {target_lufs} LUFS / {true_peak} dBTP", loud))
        except media.MediaError as exc:
            report.add(Issue("LOUDNESS_UNKNOWN", "warn", False, str(exc)))

    if faces:
        report.faces_checked, face_issues = _face_check(path, container)
        for issue in face_issues:
            report.add(issue)
    return report


def check_many(paths: list[Path], **kw) -> list[QAReport]:
    return [check(p, **kw) for p in paths]


def dump(reports: list[QAReport]) -> str:
    return json.dumps([r.to_dict() for r in reports], indent=2, ensure_ascii=False)
