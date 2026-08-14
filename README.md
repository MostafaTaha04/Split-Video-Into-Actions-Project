# Splitting a Video into Actions

**Temporal segmentation of egocentric hardware-assembly video.**

This project automatically divides a first-person (egocentric) video of a hardware-assembly
procedure into its individual steps. The goal is **temporal segmentation** — detecting *where*
each step begins and ends — and not action recognition. A modular, interpretable pipeline fuses
hand tracking, open-vocabulary object detection, optical flow, scene-change detection, and
hand–object interaction into a single per-frame transition signal, and places step boundaries at
its prominent peaks.

> Project 6 · Students: **Osama Najjar, Mostafa Taha** · Mentor: **Saeed Namnah**

---

## Highlights

- **Modular feature pipeline** — hands, objects/regions, optical flow, scene change, and interaction.
- **Interpretable boundaries** — a step boundary is simply a moment where multiple cues agree the action changed; no per-frame labels or training required.
- **Rigorous evaluation** — boundary precision/recall/F1 at multiple tolerances, plus segment IoU and coverage, against manually annotated ground truth.
- **No test-set tuning** — one global configuration for all clips, validated by leave-one-clip-out cross-validation, plus a clip never used for tuning at any stage.
- **Rule-based, learned, or hybrid scoring** — `--scorer rule|learned|hybrid`. A classifier trained on the pipeline's own per-frame features can replace or blend with the hand-designed fusion; the hybrid roughly doubles worst-case per-clip F1.
- **Baselines & ablations** — beats uniform, random, and a `ruptures` change-point baseline at tight tolerances; cue ablation identifies what actually carries the signal.
- **Annotation tooling** — a documented protocol, a frame-accurate annotation tool, a ground-truth validator, and an inter-annotator agreement metric (`annotate.py`).
- **Readable outputs** — annotated video, timeline plot, per-step clips, and a feature CSV per run.

---

## Results

All numbers below use a **single global configuration** (boundary threshold 0.70, minimum segment
duration 2.0 s) applied unchanged to every clip. No per-clip tuning.

| Video | Type | F1 @1.0s | F1 @3.0s | In tuning set? |
|---|---|:--:|:--:|:--:|
| Cooling fan | Clean procedure | **0.727** | 0.727 | yes |
| CPU placement | Clean procedure | **0.714** | 0.857 | yes |
| RAM install | Edited tutorial | 0.316 | 0.421 | yes |
| Cable connect | Edited tutorial | 0.167 | 0.500 | yes |
| Intel CPU install | Clean procedure | 0.286 | 0.571 | **no — held out** |

**Leave-one-clip-out cross-validation** (tune on three clips, test on the fourth) gives a held-out
mean F1 of **0.658** on the clean clips, confirming the configuration is not overfitted to them.

On clean, continuously recorded procedures the system localises matched boundaries to within
~0.1–0.4 s. Two known poor cases are analysed rather than hidden:

- **Edited tutorial footage** (RAM, Cable) — presenter cut-aways and diagram slides fire the
  scene-change cue at moments that are not assembly steps. See `Final_Report.docx` §7.2.
- **The held-out clip** (Intel CPU install, F1 0.286) — the system proposes 9 boundaries against 5
  annotated ones. Note that this clip's ground truth is one of the files that **fails**
  `annotate.py validate` (every boundary on a whole or half second), and 2 of its 3 unmatched
  references sit 1.8 s from a prediction. The number is reported as-is and should be re-measured
  after re-annotation — see `ANNOTATION_PROTOCOL.md` §4.

`figures/boundary_alignment.png` shows predicted versus annotated boundaries for all five clips on
a shared time axis. Full analysis: `Final_Report.docx`, `Extended_Evaluation.docx`.

### Learned and hybrid scorers

Supervised alternatives were trained on the pipeline's own per-frame features (3,396 labelled
frames, no new annotation) and compared under leave-one-clip-out with **identical** post-processing:

| Scorer | Mean F1 @1.0s | Worst clip | Std across clips |
|---|:--:|:--:|:--:|
| **Hybrid** (rule + logistic regression) | **0.476** | **0.316** | **0.098** |
| Logistic regression (raw features) | 0.435 | 0.211 | 0.142 |
| Rule-based fusion (hand-designed) | 0.405 | 0.133 | 0.262 |
| Logistic regression (all features) | 0.386 | — | — |
| Gradient boosting (raw features) | 0.360 | — | — |
| Neural network (all features) | 0.315 | — | — |
| Neural network (raw features) | 0.294 | — | — |
| Gradient boosting (all features) | 0.268 | — | — |

No learned model *significantly* beats the hand-designed fusion (best margin +0.030, paired
t-test p = 0.76 — five clips cannot resolve it). But they fail in opposite directions: rule-based
scores 0.720 on the clean clips vs the learned 0.544, and 0.195 on the harder clips vs 0.363.
Blending them gives the best mean, **roughly doubles the worst-case clip score**, and cuts spread
across clips by 2.7×. Note also that the *simplest* model is the best of the three and both
higher-capacity models rank below it — the expected signature of limited data.

> **Version note.** Figures produced with **scikit-learn 1.9.0**. Logistic regression, the neural
> network, the hybrid and the rule-based scorer are bit-identical under 1.7.2, but gradient boosting
> moves (0.360→0.356 raw, 0.268→0.306 all) because its implementation changed between releases. No
> conclusion depends on it, but `scikit-learn` is pinned to `>=1.9,<1.10` so the table reproduces.

```bash
pip install "scikit-learn>=1.9,<1.10"
python learned_baseline.py --src .        # held-out comparison above
python train_boundary_model.py --src .    # writes boundary_model.joblib
python main.py --video clip.mp4 --scorer hybrid ...
```

The default remains `--scorer rule`, which is still the best choice on clean footage.

`boundary_model.joblib` is **not** committed — a pickled scikit-learn estimator is tied to the
version that created it, and loading it under a different version warns and may change results.
Run `train_boundary_model.py` once after cloning; it takes seconds. The JSON sidecar recording the
blend weight and provenance *is* committed.

---

## Installation

Requires **Python 3.10+**.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt          # or: pip install -e .
```

On first run, the YOLO-World weights (`yolov8*-worldv2.pt`) and the MediaPipe
`hand_landmarker.task` model download automatically. Neither is stored in the repository.

---

## Quick start

Segment a clip and evaluate it against ground truth:

```bash
python main.py --video Coolingfaninstallation.mp4 --output results_coolingfan \
  --fps 10 --detector open_vocab --model yolov8l-worldv2.pt \
  --resize 960x540 --open-vocab-imgsz 960 --threshold 0.70 --min-duration 2.0 \
  --grip-window 7 --object-confidence 0.10 \
  --ground-truth ground_truth_coolingfan_v2.json
```

Each run writes to the `--output` folder:

- `segmentation_results.json` — segments and boundaries with timestamps
- `timeline.png` — boundaries over the activity signal
- `annotated_output.mp4` — video with hand/region overlays
- `clips/step_*.mp4` — one clip per detected step
- `features.csv` — per-frame feature values
- `evaluation_report.txt` — metrics (when `--ground-truth` is given)

---

## Key command-line options

| Option | Description |
|---|---|
| `--video` | Input video path (required). |
| `--output` | Output directory. |
| `--detector` | `workspace`, `open_vocab`, `yolo`, `hybrid`, or `none`. |
| `--model` | Detector weights (e.g. `yolov8l-worldv2.pt`, or a trained `hardware_model.pt`). |
| `--fps` | Processing rate (10 recommended). |
| `--resize` | Processing resolution, e.g. `960x540`. |
| `--threshold` | Boundary-detection sensitivity (higher = fewer boundaries). |
| `--min-duration` | Minimum segment length in seconds. |
| `--ground-truth` | Ground-truth JSON for evaluation. |
| `--scorer` | `rule` (default), `learned`, or `hybrid` boundary scoring. |
| `--boundary-model` | Trained model for `learned`/`hybrid` (default `boundary_model.joblib`). |
| `--blend-weight` | Rule-side weight for `hybrid`; defaults to the cross-validated value. |
| `--no-flow` / `--no-scene` | Disable a feature (used for the ablation study). |
| `--debug-detections` | Print raw detector candidates per frame. |
| `--max-frames` | Process only the first N frames (fast debugging). |

---

## Ground-truth annotation

Evaluation is only as trustworthy as the reference it is scored against, so annotation is treated
as a first-class part of the method. **`ANNOTATION_PROTOCOL.md` defines what counts as a step
boundary** and must be followed for every annotation.

```bash
# 1. coarse pass — timestamped frame grids to locate transitions
python annotate.py contact-sheet --video Coolingfaninstallation.mp4 --step 1.0

# 2. fine pass — refine one boundary to 0.1 s
python annotate.py contact-sheet --video Coolingfaninstallation.mp4 \
    --around 12.5 --window 1.0 --step 0.1

# 3. record — interactive frame-stepping player, writes the JSON directly
python annotate.py review --video Coolingfaninstallation.mp4 \
    --output ground_truth_coolingfan.json --annotator "Your Name"

# 4. validate — rejects placeholder annotators, template round numbers, gaps/overlaps
python annotate.py validate --ground-truth ground_truth_coolingfan.json \
    --video Coolingfaninstallation.mp4

# 5. agreement — inter-annotator ceiling from two independent annotations
python annotate.py agreement --a gt_fan_annotatorA.json --b gt_fan_annotatorB.json
```

> **Status:** four of the five annotation files currently fail `validate` and are flagged for
> re-annotation in `ANNOTATION_PROTOCOL.md` §4. This is the highest-value open task on the project.

---

## Evaluation tools

**Extended evaluation (no GPU / no re-run needed)** — every run saves a per-frame `features.csv`,
so the fast boundary-detection stage can be replayed offline to produce held-out cross-validation,
sensitivity, annotation robustness, a change-point baseline, and the cue ablation in seconds:

```bash
python evaluate_extended.py --src .      # writes extended_results.json and figures/
```

This reproduces every number and figure in the report's evaluation sections, and asserts that
re-segmenting the saved features reproduces the saved boundaries (guarding against drift).

**Baseline comparison:**

```bash
python evaluate_baselines.py --ground-truth ground_truth_cpuplacement.json \
  --results results_cpu_final/segmentation_results.json
```

**Pipeline-level ablation** — re-run with a feature disabled:

```bash
python main.py ... --no-flow      # contribution of optical flow
python main.py ... --no-scene     # contribution of scene-change detection
```

**Refresh saved run reports** — rebuild `evaluation_report.txt` for every run from its saved
features and boundaries, without re-running the vision pipeline:

```bash
python refresh_run_reports.py --dry-run   # show deltas
python refresh_run_reports.py             # apply
```

---

## Testing

46 unit and integration tests cover the evaluation metrics, the temporal segmenter (fusion, peak
detection, segment construction, edge cases), and the offline re-segmentation. They need only
`numpy`, `scipy`, and `opencv` — the heavy vision deps are imported lazily — so they run in
seconds:

```bash
pip install -r requirements-dev.txt
python -m unittest discover -s tests -v        # or: python -m pytest tests/
```

Continuous integration (`.github/workflows/ci.yml`) runs the suite, error-level lint
(`ruff check .`), the full evaluation replay, and the ground-truth validator on Python 3.10 and 3.11.

---

## Project structure

```
main.py                  Pipeline orchestration & CLI
config.py                Central configuration (FeatureParams, SegmenterParams)
video_loader.py          Frame loading / resizing / clip export
hand_tracker.py          MediaPipe hand tracking (Tasks API)
object_detector.py       YOLO-World open-vocabulary detection + workspace ROIs
optical_flow.py          Dense optical-flow features
scene_detector.py        Scene-change detection
interaction_tracker.py   Hand-object/region interaction
feature_extractor.py     Per-frame feature fusion -> transition score
temporal_segmenter.py    Boundary detection & segment construction
activity_recognizer.py   Human-readable activity labels (known-weak; see Limitations)
boundary_model.py        Learned boundary scorer: features, model loading, blending
train_boundary_model.py  Trains the learned scorer from saved features + annotations
learned_baseline.py      Held-out comparison of learned vs hand-designed scorers
evaluator.py             Boundary / segment metrics
visualizer.py            Timeline, annotated video, CSV export
utils.py                 Metrics & helpers
annotate.py              Annotation kit: contact sheets, review, validate, agreement
evaluate_baselines.py    Baseline comparison tool
evaluate_extended.py     Offline held-out CV, sensitivity, baselines, ablation (+ figures)
refresh_run_reports.py   Rebuild saved run reports from saved features
extract_training_frames.py / train_hardware_model.py   Optional custom-detector kit
tests/                   Unit + integration tests (metrics, segmenter, re-segmentation)
figures/                 Generated evaluation figures
ground_truth_*.json      Manual annotations
```

---

## Ground truth format

```json
{
  "video": "Coolingfaninstallation.mp4",
  "annotator": "Osama Najjar",
  "annotation_method": "interactive frame-stepping review (annotate.py review)",
  "steps": [
    { "id": 0, "start": 0.0, "end": 4.63, "label": "Prepare cooler area" }
  ]
}
```

Boundaries are the interior step end-times (a file with *n* steps yields *n − 1* boundaries).
Annotate from the actual video — evenly-spaced placeholder times produce misleading scores, which
is exactly what `annotate.py validate` exists to catch.

---

## Limitations

- The evaluation set is small: five clips, of which three are clean continuous procedures. The
  learned-vs-rule-based comparison cannot reach statistical significance at this size.
- **The activity labeller does not work.** Measured against annotated step labels it scores
  0.000–0.039 across every clip, emitting generic phrases like "Unspecified hand activity", because
  the component-specific path depends on the zero-shot detector that cannot identify small parts.
  Labelling is out of scope for the brief and segmentation metrics are unaffected, but the
  component is retained and its failure is reported rather than hidden (`Final_Report.docx` §7.4).
- **The rule-based scorer does not improve with more data** — its ~40 thresholds are hand-set, so a
  larger dataset needs re-tuning rather than retraining. This is part of why the learned scorer exists.
- Subtle, low-motion transitions can be missed.
- Edited, multi-shot tutorial footage is not well suited to feature-based segmentation.
- **The method is not causal.** Score normalisation uses the 5th/95th percentiles of the whole
  video, so it cannot run as a live stream without a windowed reformulation.
- Zero-shot detection cannot reliably identify small hardware parts; component labels would need a
  custom-trained detector (see `TRAINING_GUIDE.md`).
- Inference is CPU-bound (~1.5 fps); a GPU greatly speeds it up.
- Four of five ground-truth files still fail the project's own validator (see above).

---

## Documentation

- `Final_Report.docx` — full project report (design, evaluation, discussion, references).
- `Evaluation_Report.docx` — focused evaluation report.
- `Extended_Evaluation.docx` — supplement: held-out CV, sensitivity, robustness, change-point baseline, ablation.
- `ANNOTATION_PROTOCOL.md` — what counts as a boundary, and how annotations are made and checked.
- `DEMO_SCRIPT.md` — script for the demo video.
- `TRAINING_GUIDE.md` — how to train a custom hardware detector.
- `evaluate_extended.py` + `extended_results.json` + `figures/` — extended evaluation and plots.
