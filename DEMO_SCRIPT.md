# Demo Video Script — "Splitting a Video into Actions"

**Target length:** ~4–5 minutes
**Students:** Osama Najjar, Mustafa Taha · **Mentor:** Saeed Namnah

This is a narration script + shot list for the Task 6 demo video. Each section lists
**[SHOW]** (what's on screen) and **[SAY]** (what you narrate). Record your screen
(e.g., OBS, Xbox Game Bar `Win+G`, or Zoom screen-share recording). Speak naturally —
the lines below are a guide, not a word-for-word teleprompter.

Before recording, open these so you can switch quickly:
- A terminal in the project folder
- `results_coolingfan_v2run/timeline.png`
- `results_coolingfan_v2run/annotated_output.mp4`
- `Final_Report.docx` (or the results table)
- One short clip from `results_coolingfan_v2run/clips/` (e.g. `step_5.mp4`)

---

## 0:00–0:30 — Intro & problem (talking head or title slide)

**[SHOW]** Title slide or your face. Optionally the project title.

**[SAY]**
"Hi, we're Osama and Mustafa. Our project, *Splitting a Video into Actions*, takes a
first-person video of someone assembling PC hardware and automatically divides it into
its individual steps. Importantly, the goal is **temporal segmentation** — finding *where*
each step begins and ends — not recognising or naming the action. I'll show how it works
and how well it performs."

---

## 0:30–1:30 — How it works (pipeline overview)

**[SHOW]** The module table / a simple diagram from the report, or just talk over the code folder.

**[SAY]**
"The system is a modular pipeline. For every frame we extract several signals: hand
movement and grip from MediaPipe, component and workspace regions from a YOLO-World
open-vocabulary detector, dense optical flow for motion, a scene-change score, and
hand-object interaction. These are fused into a single per-frame *transition score*.
Wherever that score peaks — meaning several cues agree something changed — we place a
step boundary. So the algorithm is interpretable: a boundary is just a moment where the
hands and the motion strongly indicate the operator moved on to the next step."

---

## 1:30–2:45 — Live run (the core of the demo)

**[SHOW]** Terminal. Run (or play a pre-recorded run of) the cooling-fan command:

```
python main.py --video Coolingfaninstallation.mp4 --output results_demo --fps 10 --detector open_vocab --model yolov8l-worldv2.pt --resize 960x540 --open-vocab-imgsz 960 --threshold 0.55 --min-duration 2.5 --grip-window 7 --object-confidence 0.10 --ground-truth ground_truth_coolingfan_v2.json
```

> Tip: the full run takes a few minutes on CPU. **Pre-record it and fast-forward**, or
> start it, then cut to the already-finished output so you don't show dead time.

**[SAY]** (while it runs / as you cut to the finished output)
"Here we run it on the cooling-fan installation clip. It loads, enables MediaPipe hand
tracking, runs the detector, and processes the video frame by frame. When it finishes it
prints a **segmentation summary** — each detected step with its start and end time and a
short activity label — and writes out a timeline image, an annotated video, and clips for
each step."

**[SHOW]** Scroll to the printed SEGMENTATION SUMMARY in the terminal; point at a couple of rows.

**[SAY]**
"You can see it split this 48-second clip into the procedure's steps, each with a
timestamp and a label derived from the hand motion."

---

## 2:45–3:30 — Visual results

**[SHOW]** Open `timeline.png`. Then play ~10 seconds of `annotated_output.mp4`. Then open one `clips/step_*.mp4`.

**[SAY]**
"This timeline shows the detected boundaries laid over the activity signal — each band is
one step. In the annotated video you can see the hand tracking and the regions the system
is using. And because it exports each step as its own clip, the continuous video becomes a
set of labelled, trackable steps — which was the goal."

---

## 3:30–4:30 — Evaluation & results

**[SHOW]** The `--- EVALUATION ---` block in the terminal, then the results table in `Final_Report.docx`.

**[SAY]**
"To measure accuracy we compare the detected boundaries against a manual annotation, using
boundary precision, recall, and F1 at several time tolerances. On the clean clips the
system recovers **every** true boundary within three seconds, and the boundaries it matches
are accurate to within about a tenth to four-tenths of a second. On the cooling fan it
reaches an F1 around 0.63–0.71 at a one-second tolerance.

We also compared against naive baselines — splitting the video into equal parts or random
boundaries — and our method clearly wins at the tight tolerances, which shows it's
genuinely localising the steps, not just guessing. An ablation study showed optical flow
is the most important signal: remove it and the system can't find any boundaries at all.

We were also honest about the limits: on edited tutorial videos with presenter cut-aways,
the scene cuts mislead the detector, and zero-shot detection couldn't reliably identify
small parts — though that doesn't affect the segmentation goal."

---

## 4:30–5:00 — Conclusion

**[SHOW]** Title slide or your face again.

**[SAY]**
"In summary, we built a working, interpretable system that segments egocentric assembly
video into steps, evaluated it rigorously against ground truth and baselines, and
documented where it works and where it doesn't. Future work would be training a custom
detector for the parts and testing on more continuous footage. Thanks for watching."

---

## Recording checklist

- [ ] Screen recorder tested (audio + screen captured)
- [ ] Run pre-executed so you can fast-forward the slow part
- [ ] `timeline.png`, `annotated_output.mp4`, a step clip, and the report all open
- [ ] Terminal font large enough to read on video
- [ ] One practice run-through before the real take
- [ ] Keep it under ~5 minutes; it's fine to edit out the processing wait
