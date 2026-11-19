# Splitting a Video into Actions

Students: **Osama Najjar, Mostafa Taha** · Mentor: **Saeed Naamneh** · Moderator: **Jehad Alsanaa**

Given a first-person video of someone assembling or disassembling a PC, the system finds when each
of 13 activities happens and cuts the video at the moments the activity changes. Every clip is
named after its activity, and a player shows the video with the timeline of activities and cuts
underneath it.

## Quick commands

Run these in the VS Code terminal (PowerShell), in the project folder.

```powershell
# 1. Once per terminal: turn on the project's Python environment
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1

# 2. Once (and after requirements.txt changes): install what the project needs
pip install -r requirements.txt

# 3. Build every video's player (video + timeline + clips) and open the home page.
#    First run: a few minutes per video. After editing an annotation file: seconds.
python show_all.py

# Only one video (faster):
python show_all.py pc_disassembly_02

# 4. Just open the results again, without rebuilding anything
start outputs\index.html

# 5. Upload your changes to GitHub
git add -A
git commit -m "Describe what you changed"
git push
```

Where things are:

| What | Where |
|---|---|
| The videos (not in git) | `data\videos\` |
| Annotation files (start, end, label per activity) | `data\annotations\` |
| Home page listing all the videos | `outputs\index.html` |
| One video's player (video, timeline under it, clips) | `outputs\<video name>\viewer.html` |
| One video's clips, one per activity | `outputs\<video name>\clips\` |

Open the `.html` pages in a browser (double-click them in File Explorer, or use `start`), not in
the VS Code editor, which only shows their code.

## The activities

`CLOSE`, `COVER`, `EXTRACT`, `FIXING_FAN`, `INSERT`, `OPEN`, `PLUG`, `PLUG_FAN`, `SCREW`,
`UNCOVER`, `UNPLUG`, `UNPLUG_FAN`, `UNSCREW`. Time between activities is `BACKGROUND`.
Six of them come in opposite pairs (OPEN/CLOSE, COVER/UNCOVER, INSERT/EXTRACT, PLUG/UNPLUG,
PLUG_FAN/UNPLUG_FAN, SCREW/UNSCREW); `labels.py` defines them and their colours.

## Data

PC assembly and PC disassembly videos from the EgoProceL dataset (head-mounted camera, 4K),
with an annotation file per video giving each activity's start and end:

```
data/videos/        pc_assembly_01.mp4, pc_disassembly_01.mp4, ...   (not in git: too large)
data/annotations/   one annotation file per video
```

The videos are 3-4 GB each, so they are not in git. Download them from the
[project's Google Drive folder](https://drive.google.com/drive/folders/19QofEX1tcfmU3fO2DAwGV8iiBZPAiRuI)
into `data/videos/`, keeping their names (`pc_assembly_01.mp4`, ...).

## The player

See every video with its timeline under it and its clips:

```bat
pip install -r requirements.txt
python show_all.py
```

This builds a player for every video in `data\videos\` that has an annotation file in
`data\annotations\`, then opens `outputs\index.html` in the browser: a home page listing all the
videos, each with its timeline. Click a video to open its player and clips. Each video's player is
in `outputs\<video name>\viewer.html`. The first run takes a few minutes per 4K video; after you
edit an annotation file, run it again and it takes seconds. `python show_all.py pc_assembly_02`
builds only that video; `--no-clips` skips the clips.

To see the pages inside VS Code instead of a separate browser:

```bat
python show_all.py --serve
```

It prints an address (`http://localhost:8000/`). In VS Code press `Ctrl+Shift+P`, run
**Simple Browser: Show** and enter that address. Stop it with `Ctrl+C`. (`python serve.py` serves
the existing outputs without rebuilding anything.)

One video with any segments file:

```bat
python viewer.py --video data\videos\pc_assembly_01.mp4 --segments steps.json
```

`steps.json` is a list of `{"start": seconds, "end": seconds, "label": "SCREW"}`; an annotation
file from `data/annotations/` works too. The command writes into `outputs/<video name>/`:
`viewer.html` (open it in a browser), a browser-playable `preview.mp4`, and one clip per step in
`clips/` named after its activity, e.g. `step_07_SCREW.mp4`. Making the preview of a 4K video
takes a few minutes the first time; later runs reuse it. Add `--no-clips` to skip the clips, and
`--ground-truth annotation.json` to show the annotated activities as a second row.

## Status

- Annotation files for 5 videos (4 assembly, 1 disassembly) are drafted and being checked;
  see `data/annotations/README.md`.
- The player shows each video with its timeline and clips.
- Next: the model that labels each second of video with one of the 13 activities and cuts the
  video where the activity changes, evaluated on videos it was not trained on.
