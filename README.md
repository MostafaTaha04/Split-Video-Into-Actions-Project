# Splitting a Video into Actions

Students: **Osama Najjar, Mostafa Taha** · Mentor: **Saeed Naamneh** · Moderator: **Jehad Alsanaa**

Given a first-person video of someone assembling or disassembling a PC, the system finds when each
of 13 activities happens and cuts the video at the moments the activity changes. Every clip is
named after its activity, and a player shows the video with the timeline of activities and cuts
underneath it.

## How to run

All commands are for PowerShell (for example the VS Code terminal), in the project folder.

### 1. Install (once)

You need Python 3.10 or newer.

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Each time you open a new terminal, run only the 2nd and 3rd lines.

### 2. Run it on one video

```powershell
python run_model.py VIDEO_NAME
```

Example:

```powershell
python run_model.py pc_assembly_01
```

See the result (it also opens by itself):

```powershell
start outputs\pc_assembly_01\model.html
```

The page shows the video with the timeline of activities under it and one clip per activity.

- `VIDEO_NAME` is the name of a video in `data\videos\` or `data\new_videos\`, without `.mp4`.
- A training video (like `pc_assembly_01`) is cut by a model trained on the *other* videos only,
  so the result is a fair test; its scores are printed and its annotation is shown under the
  timeline.
- A video somewhere else: write its full path instead of the name. Example:

  ```powershell
  python run_model.py C:\Users\me\Downloads\my_video.mp4
  start outputs\my_video\model.html
  ```

  The trained model is included in the project, so this works right away on any first-person
  video of a PC being assembled or disassembled. The first run downloads the image model (about
  100 MB); after that, a video takes a few minutes.

### 3. Run the whole project

This needs the 10 training videos in `data\videos\` (see [Data](#data)); the annotation files are
already in the project.

```powershell
python run_model.py
```

See the result (it also opens by itself):

```powershell
start outputs\results.html
```

It tests the model on each of the 10 videos (each one cut by a model trained on the other 9),
builds every video's player and clips, and trains the final model. Click a video on the results
page to see its player.

### Other commands

**The real test on new videos** (every video in `data\new_videos\`, cut by a model trained on
`data\videos\`):

```powershell
python run_model.py final
start outputs\final_results.html
```

**Check the annotation files** (every video in `data\videos\` shown with its annotation):

```powershell
python show_all.py
start outputs\index.html
```

**Options** - add them after any command above:

| Option | What it does | Example |
|---|---|---|
| `--fresh` | Redoes everything from the original video instead of reusing earlier work | `python run_model.py pc_assembly_01 --fresh` |
| `--no-clips` | Skips the clips (faster) | `python run_model.py pc_assembly_01 --no-clips` |
| `--task` | Tells the model the task instead of letting it decide (one video only) | `python run_model.py pc_disassembly_05 --task disassembly` |

**To add a video:** name it like the others (`pc_assembly_07.mp4`) and put its annotation file
`pc_assembly_07.json` in `data\annotations\`. Put the video in `data\videos\` to train on it, or in
`data\new_videos\` to test on it. Open the `.html` pages in a browser, not in the VS Code editor.

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

## The activity model

`run_model.py` finds the activities in a video by itself and cuts it where the activity changes.

1. **Describing the frames** (`features.py`). Two frames per second are taken from the video. A
   pretrained image model (ResNet-50, trained on ImageNet's 1.2 million photos; downloaded once,
   about 100 MB) turns each frame into 2048 numbers describing what is in it: hands, tools,
   cables, boards. It does this twice: for the whole frame, and for a zoomed-in view of the work
   area (the centre and bottom of the frame, where the hands and the case are), where small parts
   are about twice as large. The amount of movement in each part of the picture is measured too.
   This is saved in `features\`, so it is done once per video.
2. **Context** (`model.py`). Each moment is described by the frames around it as well: windows of
   2, 4, 8, 16 and 32 s before and after it, and the 30 s around it. One frame rarely shows what a
   hand is doing; the seconds around it do.
3. **Assembly or disassembly.** A classifier decides from the start of the video whether the PC
   is being put together or taken apart, and only that task's activities are allowed afterwards.
4. **Activity.** A second classifier (logistic regression) gives, for every half second, how
   likely each activity is. It learns this from the annotation files.
5. **Smoothing and cutting** (a hidden semi-Markov model). The video is split into activities for
   the whole video at once. Each possible activity segment is scored by how well its frames fit,
   how usual its length is for that activity (learned from the training videos: EXTRACT takes a
   few seconds, SCREW minutes), and how usual it is after the previous activity (UNSCREW is
   usually followed by EXTRACT). This keeps short activities that a plain smoothing would swallow,
   and stops long ones from breaking up. The video is cut wherever the activity changes.

**Testing** (`evaluate.py`). Each video is cut by a model trained on the *other* videos only, so it
is always new to the model, and the result is compared with its annotation file. The scores are the
standard ones for action segmentation: share of seconds labelled right, F1 per activity, segment
F1@10/25/50 (an activity found counts if it overlaps a real one enough), edit score (are the
activities in the right order) and whether the cuts are within 2 or 5 s of the real ones.

### Results

**1. Leave one video out on the first 8 videos** (4 assembly, 4 disassembly). Each video
was cut by a model trained on the other 7:

| | Seconds right | Segment F1@25 | Segment F1@50 | Order (edit) | Cuts ±2 s F1 | Cuts ±5 s F1 | Assembly/disassembly |
|---|---|---|---|---|---|---|---|
| First model | 80 % | 78 | 63 | 69 | 52 | 63 | 8 of 8 right |
| **Current model** | **85 %** | **83** | **77** | **81** | **70** | **76** | **8 of 8 right** |

What changed from the first model, each kept only because it raised the scores: the
duration-aware smoothing (step 5), the longer and more varied time windows (step 2), and the
zoomed-in view of the work area (step 1). F1 per activity: PLUG 99, SCREW 97, FIXING_FAN 92,
INSERT 91, UNSCREW 87, UNCOVER 85, UNPLUG 67, EXTRACT 49, COVER 37, UNPLUG_FAN 37. The model's
design was chosen on these 8 runs, so these numbers are somewhat optimistic.

**2. Final test on 2 new videos** (the fair measure). The model above, trained on the 8 videos and
not changed afterwards, cut two videos it had never seen. They were annotated by watching the videos, without
using the model's cuts:

| Video (original name) | Seconds right | Segment F1@50 | Order (edit) | Cuts ±5 s F1 | Assembly/disassembly |
|---|---|---|---|---|---|
| `pc_disassembly_05` (`Head_17`) | 78 % | 60 | 58 | 76 | right (98 %) |
| `pc_assembly_05` (`Head_9`) | 64 % | 53 | 47 | 67 | right (99 %) |
| **Mean** | **71 %** | **57** | **53** | **71** | **2 of 2 right** |

On new videos the model loses about 14 points of "seconds right". The main mistakes: in `pc_assembly_05` the
4-minute motherboard screwing was mostly given other labels, and in `pc_disassembly_05` the cover
coming off and half of the parts being lifted out were missed.

**3. Leave one video out on all 10 videos** (after the final test, the two new videos were added to
the training videos):

| | Seconds right | Segment F1@25 | Segment F1@50 | Order (edit) | Cuts ±5 s F1 | Assembly/disassembly |
|---|---|---|---|---|---|---|
| The model decides assembly/disassembly | 77 % | 75 | 68 | 71 | 74 | 9 of 10 right |
| Told assembly or disassembly (`--task`) | 81 % | 78 | 71 | 76 | 76 | given |

The first 8 videos score 76-97 %; the two final-test videos score 60 % and 21 %. `pc_disassembly_05`
was called an assembly, so only assembly activities were allowed (21 % right; 60 % when told it is a
disassembly). The best way to make the model better is more annotated videos.

**4. Two more new videos, checked by eye.** The model trained on all 10 videos cut
`pc_assembly_06` and `pc_disassembly_06` (no annotation files, so no scores). It recognised assembly and disassembly correctly (99.7 % and
98 %). In `pc_assembly_06` every activity matched the video, in the right order. In
`pc_disassembly_06` everything matched except 3:00-4:05, where the power supply is unscrewed at the
back and the model gave no or wrong labels.

## Status

- Annotation files for 10 videos (5 assembly, 5 disassembly), checked against the
  videos by Mostafa Taha and Osama Najjar; see `data/annotations/README.md`.
- The trained model (`models/activity_model.pkl`, trained on all 10) is in the project.
- Tested on the training videos and on 4 new videos (above).
- Next: more annotated videos, starting with `pc_assembly_06` and `pc_disassembly_06`.
