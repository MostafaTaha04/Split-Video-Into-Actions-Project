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
- **Baseline comparison & ablation** — the method beats uniform/random splitting at tight tolerances, and optical flow is shown to be the dominant cue.
- **Readable outputs** — annotated video, timeline plot, per-step clips, and a feature CSV per run.

---

## Results (summary)

Boundary-detection F1 against manual ground truth, shared configuration:

| Video           | Type             | F1 @1.0s | F1 @3.0s | Matched MAE | Recall @3s |
|-----------------|------------------|:--------:|:--------:|:-----------:|:----------:|
| Cooling fan     | Clean procedure  | 0.625    | 0.875    | 0.10 s      | 1.00       |
| CPU placement   | Clean procedure  | 0.615    | 0.923    | 0.43 s      | 1.00       |
| RAM install     | Edited tutorial  | 0.154    | 0.615    | 0.37 s      | 0.80       |
| Cable connect   | Edited tutorial  | 0.000    | 0.364    | —           | 0.33       |

On clean, continuously recorded procedures the system recovers every boundary within 3 s and
localises matched boundaries to within ~0.1–0.4 s. Edited tutorial footage (presenter cut-aways,
diagram slides) is a known poor case — see the report's discussion. Full analysis is in
`Final_Report.docx` / `Evaluation_Report.docx`.

---

## Installation

Requires **Python 3.10+**.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
pip install --upgrade mediapipe ultralytics
```

On first run, the YOLO-World weights (`yolov8*-worldv2.pt`) and the MediaPipe
`hand_landmarker.task` model download automatically.

---

## Quick start

Segment a clip and evaluate it against ground truth:

```bash
python main.py --video Coolingfaninstallation.mp4 --output results_coolingfan \
  --fps 10 --detector open_vocab --model yolov8l-worldv2.pt \
  --resize 960x540 --open-vocab-imgsz 960 --threshold 0.55 --min-duration 2.5 \
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
| `--no-flow` / `--no-scene` | Disable a feature (used for the ablation study). |
| `--debug-detections` | Print raw detector candidates per frame. |
| `--max-frames` | Process only the first N frames (fast debugging). |

---

## Evaluation tools

**Baseline comparison** — show the method beats naive segmentation:

```bash
python evaluate_baselines.py --ground-truth ground_truth_cpuplacement.json \
  --results results_cpu/segmentation_results.json
```

**Ablation** — re-run with a feature disabled and compare F1:

```bash
python main.py ... --no-flow      # contribution of optical flow
python main.py ... --no-scene     # contribution of scene-change detection
```

**Extended evaluation (no GPU / no re-run needed)** — every run saves a per-frame
`features.csv`, so the fast boundary-detection stage can be replayed offline to
produce held-out cross-validation, sensitivity, annotation-robustness, a
change-point baseline, and a fusion ablation in seconds:

```bash
python evaluate_extended.py --src .      # writes extended_results.json and figures/
```

This reproduces the numbers and figures used in the report's extended-evaluation
section, and asserts that re-segmenting the saved features reproduces the saved
boundaries (guarding against drift).

---

## Testing

Unit and integration tests cover the evaluation metrics and the offline
re-segmentation. They need only `numpy`, `scipy`, and `opencv` (the heavy vision
deps are imported lazily), so they run quickly:

```bash
pip install -r requirements-dev.txt
python -m unittest discover -s tests -v        # or: python -m pytest tests/
```

Continuous integration (`.github/workflows/ci.yml`) runs the suite and
error-level lint (`ruff check .`) on Python 3.10 and 3.11.

---

## Project structure

```
main.py                  Pipeline orchestration & CLI
config.py                Central configuration
video_loader.py          Frame loading / resizing / clip export
hand_tracker.py          MediaPipe hand tracking (Tasks API)
object_detector.py       YOLO-World open-vocabulary detection + workspace ROIs
optical_flow.py          Dense optical-flow features
scene_detector.py        Scene-change detection
interaction_tracker.py   Hand-object/region interaction
feature_extractor.py     Per-frame feature fusion -> transition score
temporal_segmenter.py    Boundary detection & segment construction
activity_recognizer.py   Human-readable activity labels
evaluator.py             Boundary / segment metrics
visualizer.py            Timeline, annotated video, CSV export
utils.py                 Metrics & helpers
evaluate_baselines.py    Baseline comparison tool
evaluate_extended.py     Offline held-out CV, sensitivity, baselines, ablation (+ figures)
extract_training_frames.py / train_hardware_model.py   Optional custom-detector kit
tests/                   Unit + integration tests (metrics, re-segmentation)
figures/                 Generated evaluation figures
ground_truth_*.json      Manual annotations
```

---

## Ground truth format

```json
{
  "video": "Coolingfaninstallation.mp4",
  "annotator": "manual",
  "steps": [
    { "id": 0, "start": 0.0, "end": 4.6, "label": "Prepare cooler area" }
  ]
}
```

Boundaries are derived from the step end-times. Annotate from the actual video — evenly-spaced
placeholder times produce misleading scores.

---

## Limitations

- Subtle, low-motion transitions can be missed.
- Edited, multi-shot tutorial footage is not well suited to feature-based segmentation.
- Zero-shot detection cannot reliably identify small hardware parts; component labels would need a
  custom-trained detector (see `TRAINING_GUIDE.md`).
- Inference is CPU-bound (~1.5 fps); a GPU greatly speeds it up.

---

## Documentation

- `Final_Report.docx` — full project report (design, evaluation, discussion, references).
- `Evaluation_Report.docx` — focused evaluation report.
- `Extended_Evaluation.docx` — supplement: held-out CV, sensitivity, robustness, change-point baseline, ablation.
- `DEMO_SCRIPT.md` — script for the demo video.
- `TRAINING_GUIDE.md` — how to train a custom hardware detector.
- `evaluate_extended.py` + `extended_results.json` + `figures/` — extended evaluation and plots.
