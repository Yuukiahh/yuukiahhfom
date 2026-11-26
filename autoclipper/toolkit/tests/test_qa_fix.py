from autoclipper import fix, media, qa


def codes(report):
    return {i.code for i in report.issues}


def test_detects_defects(clip_factory):
    report = qa.check(clip_factory())
    found = codes(report)
    assert {"RESOLUTION", "BLACK_START", "DEAD_AIR", "LOUDNESS"} <= found
    assert report.status == "fail"
    assert {"RESOLUTION", "BLACK_START", "DEAD_AIR", "LOUDNESS"} <= report.fixable_codes


def test_repair_fixes_everything_fixable(clip_factory):
    path = clip_factory(seconds=18)
    before = qa.check(path)
    fix.repair(path, before, encoder="libx264")
    after = qa.check(path)
    assert not after.fixable_codes, after.issues
    assert (after.width, after.height) == (1080, 1920)
    assert abs(after.loudness_lufs - (-14.0)) <= 1.5
    # 0.6 s black removed, the 3 s gap shortened to 0.5 s.
    assert 14.0 < after.duration < 15.2, after.duration
    assert after.status in ("ok", "warn")
    assert path.with_suffix(".orig.mp4").exists()


def test_clean_clip_passes(clip_factory):
    path = clip_factory(size="1080x1920", black_start=0, gap=None, volume=0.3)
    fix.repair(path, qa.check(path), encoder="libx264")  # normalise loudness once
    report = qa.check(path)
    assert report.status == "ok", report.issues


def test_missing_audio_is_unfixable_fail(clip_factory):
    report = qa.check(clip_factory(audio=False, black_start=0))
    assert "NO_AUDIO" in codes(report)
    assert report.status == "fail"


def test_middle_black_is_flagged_not_trimmed(tmp_path):
    import subprocess
    p = tmp_path / "mid.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=s=1080x1920:r=30:d=14", "-f", "lavfi", "-i", "sine=f=300:d=14",
                    "-vf", "drawbox=enable='between(t,6,7)':color=black:t=fill",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                    str(p)], check=True)
    report = qa.check(p)
    assert "BLACK_MIDDLE" in codes(report)
    assert "BLACK_MIDDLE" not in report.fixable_codes


def test_encoder_args_fallback():
    args = media.video_encoder_args("libx264")
    assert "libx264" in args
