# Editing standards and SupoClip options

Target: professional, calm short-form editing. Effects serve the content; nothing flashes,
shakes or plays music without a reason.

## Defaults (config.json → task)
| Setting | Default | Why |
|---|---|---|
| output_format | `vertical` | 1080x1920, face tracking with dynamic reframing |
| caption_template | `highlighter_swipe` (fallback `hormozi`) | word-synced animated captions, key word highlighted |
| intensity | `bold` | sensible zooms/pushes on key moments, hits on cuts |
| edit_style | sfx on, transitions on, zooms on; music, shake, speed, grain, progress bar, loop ending, reactions off | no distracting effects, no music bed fighting the voice |
| cut_long_pauses / remove_filler_words | on | jump cuts, no dead air |
| clip_length | `auto` (15-60 s) | |
| processing_mode | `quality` | |
| include_broll | on only with a Pexels key (free, licensed stock) | |

## Choose per video
- Podcast / interview, calm tone: `--intensity subtle --caption-template podcast` or keep default.
- High-energy creator, comedy, reactions: `--intensity bold` (default) or `max` for very loud content.
- Two people side by side: `--output-format vertical_split`.
- Screen recording, slides, gameplay with facecam, wide stage shots: `--output-format vertical_fit`
  (full frame inside 9:16, nothing cut off).
- Tutorials where details matter: `--clip-length medium`.
- Long streams: pass `--instructions` with the topics or segments to prefer.

Caption templates: `default hormozi mrbeast minimal tiktok neon podcast` (classic) and with
dynamic edits `highlighter_swipe emoji_drop behind_head marker_circle kinetic_stack ...`.
`mcp__supoclip__supoclip_list_caption_templates` lists what this install renders;
`mcp__supoclip__supoclip_list_edit_styles` shows the effect switches.

## What QA repairs automatically
Black frames at start/end (trim), dead air > 1.5 s inside a clip (jump cut, 0.25 s kept each
side), wrong resolution (pad to 1080x1920, never crop), codec/pixel format, A/V drift,
loudness to -14 LUFS integrated / -1 dBTP. Encoder: NVENC on the RTX GPU, libx264 fallback.

## What QA only reports (needs a re-render or a decision)
Black frames in the middle, frozen picture, face touching the frame edge (`FACE_CUT`),
face far off centre, too short/long, no audio. See SKILL.md step 5.
