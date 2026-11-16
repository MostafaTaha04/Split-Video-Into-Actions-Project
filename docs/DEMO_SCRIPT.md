# Demo Video Script — "Splitting a Video into Actions"

**Target length:** ~5½ minutes
**Students:** Osama Najjar, Mustafa Taha · **Mentor:** Saeed Naamneh · **Moderator:** Jehad Alsanaa

Narration script + shot list for the Project 6 demo video. Each section lists
**[SHOW]** (what's on screen) and **[SAY]** (what you narrate). Record your screen
(OBS, Xbox Game Bar `Win+G`, or a Zoom screen-share recording). The lines are a
guide, not a teleprompter.

Every number in this script is the current one (README / `extended_results.json`).
If you re-run `evaluate_extended.py` and anything changes, update the script first.

## Before recording

Produce the three runs the demo shows (a few minutes each on CPU):

```bat
python main.py --video Coolingfaninstallation.mp4 --output results\demo_coolingfan --ground-truth ground_truth\ground_truth_coolingfan_v2.json
python main.py --video CPUplacement.mp4 --output results\demo_cpu_appearance --scorer appearance --detector none --ground-truth ground_truth\ground_truth_cpuplacement.json
python main.py --video split-video-data\pc_disassembly_01.mp4 --output results\pc_disassembly_01 --scorer appearance --detector none --max-frames 1800
```

The first uses the defaults, which *are* the evaluated configuration — so the command
shown on screen is short and honest. Expected: cooling fan **F1 0.923** at 1.0 s
(6 predicted vs 7 annotated boundaries); CPU with the appearance scorer **F1 0.800**
(4 predicted vs 6 annotated, precision 1.000). The EgoProceL disassembly run (first three
minutes) ends with `Direction: 3 assembly, 7 disassembly, 7 unclear -> mostly disassembly`;
its saved results are already in `results\pc_disassembly_01`.

Then open, ready to switch to:
- a terminal in the project folder
- `results\demo_coolingfan\step_report.html` (in a browser)
- `results\demo_coolingfan\annotated_output.mp4`
- `results\pc_disassembly_01\step_report.html`
- `figures\boundary_alignment.png` and `figures\scorer_comparison.png`

---

## 0:00–0:30 — Intro & problem

**[SHOW]** Title slide or your face.

**[SAY]**
"Hi, we're Osama and Mustafa. Our project, *Splitting a Video into Actions*, takes a
first-person video of someone working on hardware — here, assembling a PC — and
automatically divides it into its individual steps. The goal is **temporal
segmentation**: finding *where* each step begins and ends. Recognising or naming the
action is explicitly not required."

---

## 0:30–1:30 — How it works

**[SHOW]** `figures\pipeline_architecture.png`.

**[SAY]**
"For every frame the pipeline extracts several signals: hand position and grip from
MediaPipe, components from a YOLO-World open-vocabulary detector, dense optical flow,
a scene-change score, and hand–object interaction. These cues are fused into a
per-frame boundary score, and a step boundary is placed at each strong peak. A single
global configuration is used for every video — nothing is tuned per clip.

We also built a second, training-free scorer. Instead of asking *is something
happening?*, it asks *does the scene look different now than it did two seconds ago?*
— comparing compact appearance descriptors before and after each moment."

---

## 1:30–2:45 — Live run

**[SHOW]** Terminal. Show the command, then cut to the finished output (don't show the
processing wait):

```bat
python main.py --video Coolingfaninstallation.mp4 --output results\demo_coolingfan --ground-truth ground_truth\ground_truth_coolingfan_v2.json
```

**[SAY]**
"No tuning flags: the defaults are exactly the configuration we evaluated. It runs hand
tracking, detection and optical flow frame by frame, then prints the detected steps and
the evaluation against our manual annotation — here an F1 of 0.92 at a one-second
tolerance."

**[SHOW]** Open `step_report.html`. Scroll through two or three steps.

**[SAY]**
"Every run produces this step-by-step breakdown: each step's time span, keyframes from
its start, middle and end, the dots showing where the hands were during the step, and —
when ground truth is available — the annotated step it corresponds to. The continuous
video becomes a structured procedure."

**[SHOW]** ~5 seconds of `annotated_output.mp4`.

---

## 2:45–4:15 — Results

**[SHOW]** `figures\boundary_alignment.png`.

**[SAY]**
"We evaluate on 30 annotated clips. Five were annotated by hand, frame by frame; the other
25 were drafted with an AI tool and then checked against the video by the two of us, and we always
report the two groups separately. The main metric is boundary F1 at one second. On
single-scene footage the rule-based method localises boundaries to within a few tenths of
a second — the cooling fan scores 0.92. Its weakness is over-segmentation: on most clips
its boundary score is almost always 'on', so it proposes a boundary every couple of seconds."

**[SHOW]** `figures\heldout_set.png`.

**[SAY]**
"The appearance scorer fixes most of that. Under leave-one-clip-out cross-validation it
improves mean F1 from 0.42 to 0.62, and on the 26 held-out clips from 0.47 to 0.58 — better
on 22 of them. It is not uniformly better: it fails on the cooling fan, whose small
mechanical sub-steps barely change the scene's appearance."

**[SHOW]** `figures\scorer_comparison.png`.

**[SAY]**
"We also checked something less flattering. Most of these videos are edited, and an
editor's cut often lands exactly on a step change. A detector that only looks for cuts
scores as well as the appearance scorer. Away from cuts, the rule scorer's motion cues
find more boundaries. The fairest test is continuous footage with no cuts, and on our
three continuous recordings both scorers beat every baseline."

**[SAY]** (limitations, briefly)
"Three honest limitations: hand tracking rarely succeeds on our overhead footage, so the
interaction cues have little to work with; only five clips were annotated fully by hand;
and most of the footage is edited, so part of every score comes from the editing. That's
why we built a tool to import public egocentric datasets such as EgoProceL's PC-assembly
videos, and we have already run two of them."

---

## 4:15–4:45 — Assembly or disassembly

**[SHOW]** The terminal output of the disassembly run, ending with
`Direction: 3 assembly, 7 disassembly, 7 unclear -> mostly disassembly`; then
`results\pc_disassembly_01\step_report.html`, scrolled to step 3 (29.5–32.1 s, the case
cover coming off), which is marked *disassembly*.

**[SAY]**
"We also asked whether a step puts a part in or takes it out. The segmentation itself
doesn't care — pulling a part out changes the scene as much as pushing one in — but our
step names did, because the rules were written for assembly. Now each step gets a
direction from motion: the hands travel in and out, but the part travels one way only.
On a disassembly video from the public EgoProceL dataset it says *mostly disassembly*,
and on an assembly video *mostly assembly*. It is a weak cue — right on about two steps
in three, and it leans on how long a step lasts — so when you know the video type, you
say so with `--task`."

---

## 4:45–5:30 — Conclusion

**[SHOW]** Title slide or your face.

**[SAY]**
"In summary: a working, interpretable system that turns a continuous hardware procedure
into trackable steps, evaluated with a single configuration, cross-validation, 26 held-out
clips and standard metrics, with its failures measured rather than hidden. Next steps are
more continuous first-person recordings, a second annotator, combining cut detection with
the motion cues, a hand–object contact detector so the interaction cues work on every
camera angle, and a video model trained on putting-in and taking-out actions. Thanks for
watching."

---

## Recording checklist

- [ ] All three demo runs produced, the F1 values match the numbers above, and the
      disassembly run ends with "mostly disassembly"
- [ ] `step_report.html` opens in the browser with keyframes visible
- [ ] Screen recorder tested (audio + screen)
- [ ] Terminal font large enough to read on video
- [ ] Processing wait edited out; keep it under ~5 minutes
