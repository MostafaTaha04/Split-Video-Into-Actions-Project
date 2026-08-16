# Data Collection Guide

How to grow the evaluation set from 5 clips to 30, and why the details matter.

Every limitation in the project traces back to sample size: the headline results
rest on **three clean clips**, the learned scorers are data-starved (the simplest
model wins, which is the signature of too little data), and no comparison in
§6.7 reaches statistical significance. More clips is the single highest-value
thing that can be done to the project.

---

## 1. The bottleneck is annotation, not video

This is the thing to internalise before starting.

| | Cost | Scales? |
|---|---|---|
| Getting video | minutes per clip | yes — trivially |
| **Annotating boundaries** | **15–25 min per clip** | **no — linear human time** |

Models train on `(window, is_this_a_boundary)` pairs, and every one of those
comes from a human marking a transition time. Downloading a thousand videos
does not help; a thousand *annotated* videos would.

Realistic budget with the existing tooling:

| Clips | Annotation time | What it unlocks |
|---|--:|---|
| 5 (today) | — | 29 boundaries; models overfit, nothing is significant |
| **20** | ~7 h | ~120 boundaries; learned scorers become viable |
| **30** | ~11 h | ~180 boundaries; comparisons can reach significance |
| 100 | ~35 h | end-to-end video models start to make sense |

**Target 25–30.** That fixes the small-sample criticism without becoming a
different project.

---

## 2. What footage to collect

### The rule

**Continuous, single-shot, first-person.** No cuts, no zooms, no edits.

The project's own numbers justify this: clean continuous clips score
**F1 0.727 / 0.714**; edited tutorials score **0.316 / 0.167**. Collecting more
edited footage would scale up the case the method handles worst.

### Film it yourself — don't scrape YouTube

Tempting, but wrong for this task:

- Almost all PC-build videos on YouTube are **edited tutorials** — presenter
  cut-aways, diagram overlays, jump cuts. Exactly the failure case.
- Most are **third-person tripod shots**. The brief specifies *egocentric*.
- You cannot control clip length, framing or procedure boundaries.

Filming 30 short clips takes an afternoon and gives footage that matches the
task definition. That is faster than filtering hours of unsuitable video.

### Camera setup

- **Phone on a chest mount or clamp**, angled down at the work surface — or an
  overhead arm. Anything that keeps hands and components in frame throughout.
- **1080p, 30 fps** is plenty. The pipeline downsamples to 960×540 at 10 fps.
- **Fixed camera.** Do not pan, zoom, or reframe mid-clip — camera motion feeds
  the optical-flow cue and creates false boundaries.
- **Even, consistent lighting.** Shadows across the workspace hurt both hand
  tracking and flow.
- **Hands visible.** MediaPipe hand tracking is the primary cue; if the hands
  leave frame the signal disappears.

### Clip length and content

- **45–90 seconds**, containing **5–9 distinct steps**. Long enough to be a real
  procedure, short enough to annotate in one sitting.
- **Start recording before the first action, stop after the last.** Avoid dead
  time at either end — the segmenter discards boundaries near the edges anyway.
- **One procedure per clip.** Don't chain three jobs into one recording.

### Procedures worth filming

Aim for variety in motion type, not just component type — the method keys on
*how the hands move*, so repeating similar motions adds less than it looks.

| Procedure | Motion character |
|---|---|
| RAM install / removal | press, click, lever |
| CPU seat + lock | precise placement, lever |
| Cooler mount | align, screw, clip |
| SSD / M.2 install | small screw, slot |
| GPU install | two-handed insert, screw |
| Case fan mount | repeated screwing |
| Cable / connector routing | fine finger work |
| Panel removal | unscrew, slide, lift |
| Thermal paste + spread | tool use, sustained |
| Dust cleaning | sustained, low structure |

Repeat each 2–3 times with variation — different order, different hand, slower
or faster. Variation matters more than count.

---

## 3. Onboarding a clip

```bash
# 1. run the pipeline and register the clip
python add_clip.py --video ssd_install_01.mp4 --name "SSD install 1" --footage clean

# 2. annotate
python annotate.py contact-sheet --video ssd_install_01.mp4 --step 1.0
python annotate.py review --video ssd_install_01.mp4 \
    --output ground_truth/ground_truth_ssdinstall1.json --annotator "Your Name"

# 3. validate, then re-score everything
python annotate.py validate --ground-truth ground_truth/ground_truth_ssdinstall1.json \
    --video ssd_install_01.mp4
python evaluate_extended.py --src .
```

`add_clip.py` runs the vision pipeline, measures the effective fps from the run,
writes an empty annotation stub, and registers the clip in `clips.json`. No code
changes are needed — every evaluation script reads that registry.

New clips register as **`heldout`** by default: a clip should not influence
tuning until you deliberately decide it should. Move it with `--split dev`.

**Annotate to `docs/ANNOTATION_PROTOCOL.md`.** Consistency across clips matters more
than any individual judgement call — an inconsistent 30-clip set is worse than a
consistent 10-clip one.

---

## 4. Target dataset composition

| Category | Target | Why |
|---|--:|---|
| Clean egocentric | **24** | the footage the method is designed for |
| Edited tutorial | 4 | keeps the §7.2 contrast measurable |
| Held out, never tuned on | **6** | a real test set, not one clip |
| Second-annotator overlap | 3 | inter-annotator agreement |

Two things worth doing that cost little:

- **Keep a proper held-out split.** With 30 clips you can hold out 6 and still
  have 24 for tuning — that turns your single held-out clip into a genuine test
  set with an error bar.
- **Have your partner independently annotate 3 clips.** That gives the
  inter-annotator agreement number (`annotate.py agreement`), which is the
  ceiling on what any method can score. Currently unmeasured, and it is the
  first thing a sharp examiner asks about.

---

## 5. Re-running after the dataset grows

Everything is driven by `clips.json`, so:

```bash
python evaluate_extended.py --src .     # rule-based, baselines, ablations, figures
python learned_baseline.py --src .      # learned vs hand-designed comparison
python train_boundary_model.py --src .  # retrain the shipped hybrid scorer
```

Expect the story to change, and be ready for it: with 30 clips the learned and
hybrid scorers should improve substantially, while the rule-based fusion stays
roughly flat — its thresholds are hand-set and do not benefit from more data.
**If that happens, report it.** "The learned model overtakes the hand-designed
one once the dataset is large enough" is a stronger and more interesting finding
than either method winning outright, and it would be measured rather than
asserted.
