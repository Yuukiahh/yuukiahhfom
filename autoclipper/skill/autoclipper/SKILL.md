---
name: autoclipper
description: >-
  AutoClipper - turns a long video into 5-15 finished, vertical, captioned short-form clips
  (TikTok, YouTube Shorts, Reels) with a local, GPU-accelerated SupoClip (Whisper + Ollama),
  quality control, virality scores, titles, descriptions, hashtags and a posting plan, exported
  to Desktop\AI Clips. Use it immediately, without asking for instructions, whenever the user
  sends a YouTube link (youtube.com/watch, youtu.be, /shorts/, /live/) or a path to a video file
  (.mp4 .mov .mkv .webm .avi), alone or with words like "clip", "clips", "shorts", "tiktoks",
  "schneiden", "AutoClipper". Also use it to redo, improve or re-export an existing AI Clips project.
---

# AutoClipper

The user set this up so that **sending a link is the whole request**. Run the full workflow
below end to end. Ask only when a rule below says so (rights unclear, publishing). Reply in
German unless the user writes in another language.

Tools: the `autoclipper` CLI (Bash/PowerShell), the `supoclip` MCP server (`mcp__supoclip__*`),
your image reader for review PNGs. Settings: `%USERPROFILE%\AutoClipper\config.json`.
Clips: `Desktop\AI Clips\<date>_<title>\`. Everything runs locally; never add a paid API key
or paid service without the user's explicit consent.

**Never publish, upload, schedule or post anything** (SupoClip social publishing, Higgsfield
`tiktok_*`, any upload tool) unless the user explicitly asks for that specific clip in this
conversation. Producing files on the Desktop is the end of the job.

## 1. Preflight (seconds)

```
autoclipper doctor
```
- `supoclip` UNREACHABLE (Docker or Ollama not started after a reboot): run
  `powershell -ExecutionPolicy Bypass -File "%USERPROFILE%\AutoClipper\bin\Start-AutoClipper.ps1"`,
  then rerun doctor. See `references/troubleshooting.md` for every other failure.
- `ollama` without the configured model: `ollama pull <model>`.
- `dynamic_edits: false`: clips still work (classic captions); mention it in the summary.

## 2. Rights (before anything is downloaded)

```
autoclipper rights "<url-or-path>"
```
- exit 0 → `rights` is `own`, `cc` or `local`: continue. For `cc`, the export carries the
  required credit line automatically; mention it in the summary.
- exit 3 → standard licence / unknown owner. Ask once, short (AskUserQuestion when available):
  "Darfst du „<title>" von <channel> nutzen (eigener Kanal, Erlaubnis, Lizenz)?" Continue with
  `--confirmed` only after a yes. On no: stop and say why.
- Live streams still running: say so and stop.

## 3. Clip (minutes up to an hour - run in the background)

```
autoclipper process "<url-or-path>" [--confirmed] > "%TEMP%\autoclipper_run.json"
```
Run it as a background command (it can take long; progress lines go to stderr) and tell the user
it is running. When it ends, read the JSON: `project`, `clips[]` (rank, file, score, title, qa,
contact_sheet, hook_strip, source, text) and `rejected[]`.

What `process` already did - do not redo it: transcription (Whisper large-v3-turbo, word
timestamps), moment selection and scoring by the local LLM (hook / engagement / value /
shareability), 9:16 1080x1920 face-tracked reframing, animated word-synced captions with the
key word highlighted, jump cuts (pauses and filler words removed), sensible zooms and sound
effects (no music bed, no grain, no shake, no progress bar), loudness normalisation to
-14 LUFS / -1 dBTP, technical QA with automatic repair (black frames, A/V sync, resolution,
codec, dead air, loudness), contact sheets, export with numbered files, REPORT.md,
clips.json, posting_plan.csv and paste-ready texts.

Options when the default is not right for the material (see `references/editing.md`):
`--intensity subtle|bold|max`, `--caption-template <id>`, `--clip-length short|medium|long`,
`--output-format vertical_fit|vertical_split` (screen recordings, two speakers, wide shots),
`--instructions "<max 500 chars editorial guidance>"`.

If it failed:
- exit 3: rights (step 2). Exit 1 with `SupoClip task ... ended with error`: read the message,
  fix (troubleshooting), rerun. Interrupted after the task finished: `autoclipper collect "<project>"`.
- A failed SupoClip task can also be inspected/resumed with `mcp__supoclip__supoclip_get_task`
  / `supoclip_resume_task` (task id in `<project>\_work\state.json`).

## 4. Editorial review (your job - this is where quality comes from)

For **every** clip (kept and rejected), using `text` and the two PNGs:

1. **Look** at `contact_sheet` and `hook_strip` with your image reader. Check: face fully in
   frame and roughly centred, nobody cut in half; captions readable, not covering the face's
   mouth/eyes, inside the safe zone (not in the bottom 20 % or right 15 % where TikTok/Shorts UI
   sits, not in the top 10 %); no black or frozen frames; zooms not jarring; something
   happens visually in the first second.
2. **Read** the transcript. Check: understandable without the rest of the video (no unresolved
   "he/that/as I said", no missing setup); strong hook in the first 1-3 s (question, bold claim,
   number, conflict, emotion); ends on a complete thought or payoff; no boring stretch; caption
   spelling, especially names, brands, numbers (the captions are this text).
3. **Rate** 0-100 with `references/scoring.md`: `hook`, `retention`, `emotion`,
   `entertainment`, `virality`. Be strict; a 90 must be genuinely exceptional.
4. **Write** in the video's language (German if unclear), following `references/metadata.md`:
   `shorts_title`, `tiktok_description`, `hashtags`, plus `why` (one sentence: why this works).
5. **Decide** `drop: true` + `drop_reason` for clips that need context, are boring, duplicate a
   better clip, have a visible face cut, wrong captions you cannot fix, or misleading content.

Write all entries as one JSON object keyed by clip id (from `clips.json`) to
`<project>\_work\review.json` (template: `references/review-template.json`) and apply it:

```
autoclipper finalize "<project>" "<project>\_work\review.json"
```
`finalize` merges (later reviews only change the keys they contain), re-scores (your ratings
minus QA penalties), drops what you dropped or what falls below `min_virality` (default 55;
keeps at least 5 non-failing clips), renumbers files in posting order and rewrites
REPORT.md / clips.json / posting_plan.csv / texts/. Rejected files go to `_rejected/`.

## 5. Fix what review found (automatic, then re-check)

- Technical problem left in a clip: `autoclipper fix "<clip.mp4>"` (QA → repair → QA, max 2 rounds).
- Face cut / wrong framing on most clips, or a wide/two-person/screen video:
  rerun step 3 with `--output-format vertical_fit` (or `vertical_split` for two speakers) and
  review again; keep the better project, delete nothing without asking.
- Fewer than 5 usable clips from a video longer than 10 minutes: rerun step 3 once with
  `--clip-length short` and `--instructions` naming the strongest topics you saw; finalize the
  run that has more good clips. Short or thin material may honestly produce fewer clips -
  say so instead of padding with weak ones.
- Caption typo SupoClip cannot fix from here: drop the clip if enough remain; otherwise keep it,
  name the exact word in the summary and point to the editor at http://localhost:3107.
- After any re-render: review the new contact sheets again before finalizing.

## 6. Report back (short)

- Project folder path and `REPORT.md`.
- A table: rank, score, Shorts title, original timestamps, duration.
- What was dropped and why (one line each), anything the user must know (CC credit line,
  captions to check, dynamic edits off).
- One line: nothing was published; the clips are ready for their own approval.

## Settings the user may change

`config.json` (`%USERPROFILE%\AutoClipper`): `own_channels` (no rights question for these),
`min_virality`, `min_clips`, `max_clips`, `task.caption_template`, `task.intensity`,
`task.edit_style`, `task.instructions`, `ollama_model`, `encoder`. Rerunning the installer keeps
these. B-roll needs a free Pexels API key: only ask about it if the user wants B-roll.
