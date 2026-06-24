# Training a Custom Hardware Detector

The default pipeline uses **YOLO-World** open-vocabulary detection, which (as
documented in the report, §7.3) cannot reliably recognise small assembly parts —
clips, screws, connectors — in close-up, hand-occluded footage. Component
*labels* are therefore optional: segmentation itself is driven by motion, hands,
optical flow, and scene cues. If you do want reliable per-part labels, this guide
shows how to train a small custom detector on a few hundred labelled frames from
your own videos and plug it into the pipeline with `--detector yolo`.

> This is the future-work path referenced in `README.md` and the report. It is
> entirely optional — the system runs and is evaluated without it.

---

## Overview

```
videos ──► extract_training_frames.py ──► label (YOLO format) ──► data.yaml
       └─────────────────────────────────────────────────────────────┘
                                  │
                       train_hardware_model.py
                                  │
                          hardware_model.pt
                                  │
        main.py --detector yolo --model hardware_model.pt
```

You will need ~150–400 labelled frames for a usable detector of a handful of
classes. More frames and more visual variety (lighting, angles, hands in frame)
give a more robust model.

---

## 1. Extract frames

Sample frames from your clips so they can be labelled:

```bash
python extract_training_frames.py --videos *.mp4 --output dataset/images \
    --every 0.5 --resize 960x540
```

Options:

- `--every N` — sample one frame every N seconds (0.5 is a good start).
- `--max-per-video N` — cap frames per clip so one long video doesn't dominate.
- `--resize WxH` — optional downscale; match the resolution you run inference at.

Frames are written as `dataset/images/<clip>_f000123.jpg`.

---

## 2. Label the frames (YOLO format)

Use any YOLO-format annotation tool — labelImg, Label Studio, CVAT, or Roboflow.
Draw a bounding box around each part you care about and assign it a class.

Suggested classes (keep the set small and consistent):

```
cpu, cpu_socket, retention_lever, cooling_fan, heatsink,
mounting_clip, ram_stick, cable, connector, screw, screwdriver
```

YOLO format writes one `.txt` per image (same stem) into a `labels/` folder, one
line per box: `class_id cx cy w h` with all values normalised to `[0, 1]`.
Arrange the dataset as:

```
dataset/
  images/train/   images/val/
  labels/train/   labels/val/
```

A rough 80/20 train/val split is fine. Keep frames from the *same* video mostly
on the same side of the split so the validation set measures generalisation.

---

## 3. Create `data.yaml`

Create `dataset/data.yaml` describing the dataset and the class names (order
defines each class id, starting at 0):

```yaml
path: dataset
train: images/train
val: images/val

names:
  0: cpu
  1: cpu_socket
  2: retention_lever
  3: cooling_fan
  4: heatsink
  5: mounting_clip
  6: ram_stick
  7: cable
  8: connector
  9: screw
  10: screwdriver
```

The class names here must match the ids you used while labelling.

---

## 4. Train

```bash
python train_hardware_model.py --data dataset/data.yaml --epochs 100 \
    --base yolov8n.pt --imgsz 960 --name hardware
```

- `--base` — `yolov8n.pt` is fast and fine for a small dataset; `yolov8s.pt` is
  stronger if you have more frames and a GPU.
- `--device 0` — use the first GPU (training on CPU works but is slow).
- On success the best checkpoint is copied to `./hardware_model.pt`.

---

## 5. Run the pipeline with your detector

```bash
python main.py --video Coolingfaninstallation.mp4 --output results_custom \
    --detector yolo --model hardware_model.pt \
    --fps 10 --resize 960x540 --object-confidence 0.25 \
    --ground-truth ground_truth_coolingfan_v2.json
```

In `yolo` mode the trained model runs alongside the workspace ROIs, so the ROI
fallback is preserved on frames the detector misses. Raise `--object-confidence`
once your model is reliable to suppress false positives.

---

## Notes and expectations

- A small custom detector mainly improves *component labels and context*; it is
  not required for boundary detection and will not, on its own, change the
  segmentation metrics dramatically on the clean clips.
- Label quality matters more than quantity. Tight, consistent boxes on
  partially-occluded parts are worth more than many sloppy ones.
- If you only have the four project clips, expect overfitting; treat the result
  as a proof of concept and gather more footage for a production model.
