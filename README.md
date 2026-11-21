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

# Only one video, or a few (faster); the home page still lists all the videos:
python show_all.py pc_disassembly_02
python show_all.py pc_disassembly_02 --fresh

# Redo everything from the original videos (e.g. to show someone): a few minutes per video
python show_all.py --fresh

# 4. The model: test it on every video (each one cut by a model trained on the
#    other videos), then train the final model. Opens the results page.
#    First run: 1-2 minutes per video to describe its frames, then a few minutes for the clips.
python run_model.py
python run_model.py --no-clips      # faster: scores and timelines only

# Cut a NEW video with the trained model (any .mp4, annotated or not)
python run_model.py cut C:\path\to\new_video.mp4

# 5. Just open the results again, without rebuilding anything
start outputs\index.html                        # home page: all the videos (annotations)
start outputs\results.html                      # the model's scores, all the videos
start outputs\pc_disassembly_02\model.html      # one video: the model's cuts and clips
start outputs\pc_disassembly_02\viewer.html     # one video: the annotation's cuts and clips

# 6. Upload your changes to GitHub
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
| The model's scores, all videos | `outputs\results.html` (also `results.md`, `results.json`) |
| The model's player for one video (its cuts, with the annotation underneath) | `outputs\<video name>\model.html` |
| The model's clips for one video | `outputs\<video name>\model_clips\` |
| The model's activities as a file | `outputs\<video name>\model_steps.json` |

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
in `outputs\<video name>\viewer.html`. `python show_all.py pc_assembly_02` builds only that video;
`--no-clips` skips the clips.

What is redone on each run: the timeline, the page and every clip are rebuilt from the annotation
file every time, so a change to that file always shows up. Only `preview.mp4`, the smaller copy of
the 4K video that the browser plays, is reused, because it depends only on the video; making it
takes a few minutes per video. `--fresh` deletes all earlier outputs and makes everything again,
the copies included.

One video with any segments file:

```bat
python viewer.py --video data\videos\pc_assembly_01.mp4 --segments steps.json
```

`steps.json` is a list of `{"start": seconds, "end": seconds, "label": "SCREW"}`; an annotation
file from `data/annotations/` works too. The command writes into `outputs/<video name>/`:
`viewer.html` (open it in a browser), a browser-playable `preview.mp4`, and one clip per step in
`clips/` named after its activity, e.g. `step_07_SCREW.mp4`. Making the preview of a 4K video
takes a few minutes the first time; later runs reuse it (`--fresh` makes it again). Add
`--no-clips` to skip the clips, and `--ground-truth annotation.json` to show the annotated
activities as a second row.

## The activity model

`run_model.py` finds the activities in a video by itself and cuts it where the activity changes.

1. **Describing the frames** (`features.py`). Two frames per second are taken from the video. A
   pretrained image model (ResNet-50, trained on ImageNet's 1.2 million photos; downloaded once,
   about 100 MB) turns each frame into 2048 numbers describing what is in it: hands, tools,
   cables, boards. The amount of movement in each part of the picture is measured too. This is
   saved in `features\`, so it is done once per video.
2. **Context** (`model.py`). Each moment is described by the frames around it as well (the 4 s
   before, the 4 s after, the 30 s around it), because one frame rarely shows what a hand is doing.
3. **Assembly or disassembly.** A classifier decides from the start of the video whether the PC
   is being put together or taken apart, and only that task's activities are allowed afterwards.
4. **Activity.** A second classifier (logistic regression) gives, for every half second, how
   likely each activity is. It learns this from the annotation files.
5. **Smoothing and cutting.** The labels are chosen for the whole video at once (Viterbi
   decoding): changing activity costs a penalty, and the usual order of activities learned from
   the training videos (e.g. UNSCREW is followed by EXTRACT) is preferred. The video is cut
   wherever the label changes; activities shorter than 2 s are merged into a neighbour.

**Testing** (`evaluate.py`). Each video is cut by a model trained on the *other* videos only, so it
is always new to the model, and the result is compared with its annotation file. The scores are the
standard ones for action segmentation: share of seconds labelled right, F1 per activity, segment
F1@10/25/50 (an activity found counts if it overlaps a real one enough), edit score (are the
activities in the right order) and whether the cuts are within 2 or 5 s of the real ones.

### Results (8 videos, checked annotations)

| | Seconds right | Segment F1@25 | Segment F1@50 | Order (edit) | Cuts ±5 s F1 | Assembly/disassembly |
|---|---|---|---|---|---|---|
| Mean of 8 videos | 80 % | 78 | 63 | 69 | 63 | 8 of 8 right |

Assembly videos score higher (about 86 % of seconds right) than disassembly videos (about 74 %).
Long activities are found well (F1: PLUG 97, SCREW 94, UNSCREW 87, FIXING_FAN 86); short ones less
well (UNPLUG_FAN 28, COVER 37, EXTRACT 49, UNCOVER 52): the model often merges a short step into
the long one next to it. OPEN, CLOSE and PLUG_FAN are not marked in the annotations, so the model
cannot learn them. The full tables are in `outputs\results.md`.

Keep in mind: the model's settings were chosen by comparing these same 8 test runs (on the first,
draft annotations), so the numbers are a little optimistic. A final test on 1-2 new annotated
videos, kept aside until then, is the fair measure.

## Status

- Annotation files for 8 videos (4 assembly, 4 disassembly), checked against the videos by
  Mostafa Taha and Osama Najjar; see `data/annotations/README.md`.
- The player shows each video with its timeline and clips.
- The activity model is built and tested (see above).
- Next: help the model with short activities, and a final test on new videos kept aside.
