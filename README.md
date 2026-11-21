# Splitting a Video into Actions

Students: **Osama Najjar, Mostafa Taha** · Mentor: **Saeed Naamneh** · Moderator: **Jehad Alsanaa**

Given a first-person video of someone assembling or disassembling a PC, the system finds when each
of 13 activities happens and cuts the video at the moments the activity changes. Every clip is
named after its activity, and a player shows the video with the timeline of activities and cuts
underneath it.

## Quick commands

Run these in the VS Code terminal (PowerShell), in the project folder.

### Every time you open a terminal

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1
pip install -r requirements.txt      # only the first time, and again when requirements.txt changes
```

### Commands for one video

Write the video's name (without `.mp4`) once, in the first line. All the other lines use it, so for
another video you only change the first line. (`$v` lasts until you close the terminal; in a new
terminal, set it again.)

```powershell
$v = "pc_disassembly_05"                         # <- the video's name, without .mp4

# Open its pages (nothing is rebuilt)
start "outputs\$v\model.html"                     # the model's cuts and clips (annotation underneath, if it has one)
start "outputs\$v\viewer.html"                    # its annotation's cuts and clips (videos in data\videos only)

# Rebuild its annotation player, e.g. after editing data\annotations\<name>.json
python show_all.py $v                            # seconds
python show_all.py $v --fresh                    # from the original video again: a few minutes

# Cut it with the trained model (run "python run_model.py" once first, to train it)
python run_model.py cut "data\new_videos\$v.mp4"   # a new test video
python run_model.py cut "data\videos\$v.mp4"       # a training video
python run_model.py cut "data\new_videos\$v.mp4" --fresh   # describe its frames again too
python run_model.py cut "data\new_videos\$v.mp4" --task disassembly   # if you know it (or --task assembly)
```

### Commands for all the videos

```powershell
# The annotation players: every video in data\videos, plus the home page
python show_all.py                               # first run: a few minutes per video; then seconds
python show_all.py --fresh                       # redo everything from the original videos

# The model: test it on every training video (each one cut by a model trained on the
# others), then train the final model. Opens outputs\results.html.
python run_model.py
python run_model.py --no-clips                   # faster: scores and timelines only

# The real test: train on data\videos, then cut every video in data\new_videos.
# Opens outputs\final_results.html. New videos with an annotation file also get a score.
python run_model.py final

# Open the summary pages again (nothing is rebuilt)
start outputs\index.html                         # all the training videos, with their annotations
start outputs\results.html                       # the model's scores on the training videos
start outputs\final_results.html                 # the real test on the new videos
```

### Adding a video

Name it like the others, with the next number (`pc_assembly_06`, `pc_disassembly_06`, ...), and use
the same name for the video and its annotation file.

**To test the model on it** (the model never learns from it): put `<name>.mp4` in
`data\new_videos\`. For a score, also put its annotation file `<name>.json` in `data\annotations\`,
made from the video and not copied from the model's cuts. Then:

```powershell
python run_model.py final
$v = "<name>"
start "outputs\$v\model.html"
```

**To train the model on it**: put `<name>.mp4` in `data\videos\` and `<name>.json` in
`data\annotations\`. Then:

```powershell
$v = "<name>"
python show_all.py $v                            # check its annotation in the player
python run_model.py                              # test and train again, now with it
```

A video is either a training video or a test video, never both. Move a test video into
`data\videos\` only after its final-test result has been written down; from then on it is no
longer a fair test.

### Upload your changes to GitHub

```powershell
git add -A
git commit -m "Describe what you changed"
git push
```

Where things are:

| What | Where |
|---|---|
| The videos the model learns from (not in git) | `data\videos\` |
| New videos kept aside for the real test (not in git) | `data\new_videos\` |
| Annotation files (start, end, label per activity) | `data\annotations\` |
| Home page listing all the videos | `outputs\index.html` |
| One video's player (video, timeline under it, clips) | `outputs\<video name>\viewer.html` |
| One video's clips, one per activity | `outputs\<video name>\clips\` |
| The model's scores, all videos | `outputs\results.html` (also `results.md`, `results.json`) |
| The real test on the new videos | `outputs\final_results.html` (also `.md`, `.json`) |
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

**1. Leave one video out on the first 8 videos** (4 assembly, 4 disassembly, one person). Each video
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
not changed afterwards, cut two videos it had never seen, of a different person (white gloves
instead of tan ones, and working more slowly). They were annotated by watching the videos, without
using the model's cuts:

| Video (original name) | Seconds right | Segment F1@50 | Order (edit) | Cuts ±5 s F1 | Assembly/disassembly |
|---|---|---|---|---|---|
| `pc_disassembly_05` (`Head_17`) | 78 % | 60 | 58 | 76 | right (98 %) |
| `pc_assembly_05` (`Head_9`) | 64 % | 53 | 47 | 67 | right (99 %) |
| **Mean** | **71 %** | **57** | **53** | **71** | **2 of 2 right** |

A new person costs about 14 points of "seconds right". The main mistakes: in `pc_assembly_05` the
4-minute motherboard screwing was mostly given other labels, and in `pc_disassembly_05` the cover
coming off and half of the parts being lifted out were missed.

**3. Leave one video out on all 10 videos** (after the final test, the two new videos were added to
the training videos):

| | Seconds right | Segment F1@25 | Segment F1@50 | Order (edit) | Cuts ±5 s F1 | Assembly/disassembly |
|---|---|---|---|---|---|---|
| The model decides assembly/disassembly | 77 % | 73 | 68 | 71 | 74 | 9 of 10 right |
| Told assembly or disassembly (`--task`) | 81 % | 78 | 71 | 76 | 76 | given |

The 8 videos of the first person score 76-97 %; the two of the second person score about 60 %.
When `pc_disassembly_05` is the test video, the only other video of that person in training is an
assembly, so the model links "white gloves" with assembly and calls it an assembly (21 % right; 60 %
when told it is a disassembly). With one person per task in the training data, the model cannot
tell the person apart from the task. The final model, trained on all 10, has seen both people do
both tasks. The best way to make it better is more videos of more people.

## Status

- Annotation files for 10 videos (5 assembly, 5 disassembly, two people), checked against the
  videos by Mostafa Taha and Osama Najjar; see `data/annotations/README.md`.
- The player shows each video with its timeline and clips.
- The activity model is built, tested on the training videos, and tested on 2 new videos (above).
- Next: more videos of more people, and a new final test on videos kept aside.
