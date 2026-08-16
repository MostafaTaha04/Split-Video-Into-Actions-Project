"""
detector_kit.py
===============
Everything between "frames extracted" and "a custom detector that beats
zero-shot", for the perception half of the project.

Section 7.3 documents a negative result: YOLO-World, prompted zero-shot with
text like "cpu socket" and "screwdriver", scores at most **0.024** confidence on
small hardware parts in this close-up, hand-occluded footage. That failure also
breaks the activity labeller (Section 7.4), which needs component identity and
therefore emits "Unspecified hand activity" and scores 0.000-0.039.

Fine-tuning a small detector on labelled frames from the project's own videos is
the documented fix for both. This module supplies the parts that were missing:
a fixed class list, a reproducible train/val/test split, dataset validation, and
- most importantly - a like-for-like comparison against the zero-shot baseline
so the improvement is *measured* rather than assumed.

Sub-commands
------------
    classes    print the class list (and write classes.txt for labelling tools)
    prepare    split labelled frames into train/val/test and write data.yaml
    validate   check the dataset before wasting GPU time on a broken one
    compare    custom detector vs YOLO-World zero-shot, on the same frames

Workflow
--------
    # 1. frames to label
    python extract_training_frames.py --videos *.mp4 --output dataset/images --every 0.5

    # 2. class list for your labelling tool
    python detector_kit.py classes --output dataset/classes.txt

    # 3. ... label the frames in YOLO format (see TRAINING_GUIDE.md) ...

    # 4. split + data.yaml, then check it
    python detector_kit.py prepare --root dataset
    python detector_kit.py validate --root dataset

    # 5. train (GPU strongly recommended)
    python train_hardware_model.py --data dataset/data.yaml --epochs 100

    # 6. did it actually beat zero-shot?
    python detector_kit.py compare --root dataset --model hardware_model.pt

    # 7. use it
    python main.py --video clip.mp4 --detector yolo --model hardware_model.pt
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

# Deliberately SHORT — four classes, not eight.
#
# Every extra class takes examples away from the others AND adds a decision per
# frame during labelling, so a longer list costs annotation time twice over. The
# four dropped from the original list were each dropped for a reason:
#
#   motherboard  appears in nearly every frame and fills most of it, so it
#                carries almost no information about which step is happening
#   cpu          rarely visible — usually under a socket cover or inside a hand
#   ram_slot     small, dark, hard to distinguish from the surrounding board
#   screwdriver  appears in only a handful of frames across the corpus
#
# The four kept are the objects whose presence actually changes between
# procedural steps, which is what the segmentation cares about. Add classes back
# only when there is enough data to support them.
CLASSES = [
    "cpu_socket",       # socket area, lever, retention bracket
    "cooler",           # cooler/heatsink/fan assembly
    "ram_stick",        # memory module
    "connector",        # cable ends, headers, plugs
]

IMG_EXT = {".jpg", ".jpeg", ".png"}


# ------------------------------------------------------------------ classes
def cmd_classes(args):
    print("Class list (index: name) — the order defines the label IDs:\n")
    for i, c in enumerate(CLASSES):
        print(f"  {i}: {c}")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output} — load this into labelImg / CVAT / Label Studio")
    print("\nKeep the order fixed. Re-ordering silently invalidates every label "
          "file you have already drawn.")


# ------------------------------------------------------------------ prepare
def _pairs(root: Path):
    """(image, label) pairs. A frame with no .txt is treated as background."""
    img_dir, lbl_dir = root / "images", root / "labels"
    if not img_dir.is_dir():
        raise SystemExit(f"no images at {img_dir} — run extract_training_frames.py first")
    out = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() in IMG_EXT:
            out.append((p, lbl_dir / f"{p.stem}.txt"))
    if not out:
        raise SystemExit(f"no images found in {img_dir}")
    return out


def cmd_prepare(args):
    root = Path(args.root)
    pairs = _pairs(root)

    labelled = [(i, l) for i, l in pairs if l.exists()]
    unlabelled = len(pairs) - len(labelled)
    if not labelled:
        raise SystemExit(
            f"none of the {len(pairs)} frames have labels in {root/'labels'}.\n"
            "Label them first — see TRAINING_GUIDE.md.")

    # Split by SOURCE VIDEO, not at random. Frames from one clip are highly
    # correlated (0.5 s apart), so a random split leaks near-duplicates into
    # validation and reports an accuracy the detector will not reproduce.
    by_video = {}
    for img, lbl in labelled:
        stem = img.stem.rsplit("_f", 1)[0]
        by_video.setdefault(stem, []).append((img, lbl))

    videos = sorted(by_video)
    random.Random(args.seed).shuffle(videos)
    print(f"{len(labelled)} labelled frames from {len(videos)} source videos "
          f"({unlabelled} unlabelled frames ignored)\n")

    if len(videos) < 3:
        print("WARNING: fewer than 3 source videos — splitting by frame instead.\n"
              "         Validation scores will be optimistic (near-duplicate frames\n"
              "         appear in both splits). Collect more clips.")
        random.Random(args.seed).shuffle(labelled)
        n = len(labelled)
        splits = {"train": labelled[:int(.7 * n)],
                  "val": labelled[int(.7 * n):int(.85 * n)],
                  "test": labelled[int(.85 * n):]}
    else:
        n_val = max(1, round(len(videos) * 0.15))
        n_test = max(1, round(len(videos) * 0.15))
        v_test, v_val, v_train = videos[:n_test], videos[n_test:n_test + n_val], videos[n_test + n_val:]
        splits = {s: [p for v in vs for p in by_video[v]]
                  for s, vs in (("train", v_train), ("val", v_val), ("test", v_test))}
        for s, vs in (("train", v_train), ("val", v_val), ("test", v_test)):
            print(f"  {s:5s}: {len(splits[s]):4d} frames from {vs}")

    for split, items in splits.items():
        for sub in ("images", "labels"):
            (root / split / sub).mkdir(parents=True, exist_ok=True)
        for img, lbl in items:
            shutil.copy(img, root / split / "images" / img.name)
            shutil.copy(lbl, root / split / "labels" / lbl.name)

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        f"# Generated by detector_kit.py prepare — do not edit by hand.\n"
        f"path: {root.resolve().as_posix()}\n"
        f"train: train/images\nval: val/images\ntest: test/images\n\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {json.dumps(CLASSES)}\n",
        encoding="utf-8")

    print(f"\nwrote {data_yaml}")
    print(f"next: python detector_kit.py validate --root {root}")


# ------------------------------------------------------------------ validate
def cmd_validate(args):
    root = Path(args.root)
    errors, warnings_, counts = [], [], {c: 0 for c in CLASSES}
    total_boxes = n_frames = 0

    for split in ("train", "val", "test"):
        img_dir = root / split / "images"
        if not img_dir.is_dir():
            errors.append(f"missing split folder: {img_dir}")
            continue
        imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT]
        if not imgs:
            errors.append(f"{split}: no images")
        for img in imgs:
            n_frames += 1
            lbl = root / split / "labels" / f"{img.stem}.txt"
            if not lbl.exists():
                warnings_.append(f"{split}/{img.name}: no label file (treated as background)")
                continue
            for ln, line in enumerate(lbl.read_text(encoding="utf-8").splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{lbl.name}:{ln}: expected 5 values, got {len(parts)}")
                    continue
                try:
                    cid = int(parts[0])
                    vals = [float(v) for v in parts[1:]]
                except ValueError:
                    errors.append(f"{lbl.name}:{ln}: non-numeric values")
                    continue
                if not 0 <= cid < len(CLASSES):
                    errors.append(f"{lbl.name}:{ln}: class id {cid} outside 0..{len(CLASSES)-1}")
                    continue
                if any(not 0.0 <= v <= 1.0 for v in vals):
                    errors.append(f"{lbl.name}:{ln}: coords must be normalised 0..1, got {vals}")
                    continue
                if vals[2] <= 0 or vals[3] <= 0:
                    errors.append(f"{lbl.name}:{ln}: zero-area box")
                    continue
                counts[CLASSES[cid]] += 1
                total_boxes += 1

    print(f"frames : {n_frames}")
    print(f"boxes  : {total_boxes}\n")
    print("class balance:")
    for c in CLASSES:
        n = counts[c]
        flag = "  <- too few, expect poor accuracy" if 0 < n < 30 else ("  <- NONE" if n == 0 else "")
        print(f"  {c:14s} {n:5d}  {'#' * min(40, n // 5)}{flag}")

    empty = [c for c, n in counts.items() if n == 0]
    if empty:
        warnings_.append(f"classes with no examples at all: {empty}. Either label some "
                         f"or remove them from CLASSES before training.")

    print()
    for w in warnings_[:12]:
        print(f"  WARN  {w}")
    if len(warnings_) > 12:
        print(f"  WARN  ... and {len(warnings_) - 12} more")
    for e in errors[:20]:
        print(f"  ERROR {e}")
    if len(errors) > 20:
        print(f"  ERROR ... and {len(errors) - 20} more")

    print()
    if errors:
        print(f"{len(errors)} error(s) — fix these before training.")
        raise SystemExit(1)
    print(f"OK — 0 errors, {len(warnings_)} warning(s).")
    if total_boxes < 200:
        print(f"\nNote: {total_boxes} boxes is a small dataset. Expect a usable but "
              "imperfect detector; 500+ is a more comfortable target.")


# ------------------------------------------------------------------ compare
def cmd_compare(args):
    """Custom detector vs YOLO-World zero-shot, on the same held-out frames.

    This is the experiment that makes the work reportable. Section 7.3 records
    zero-shot peaking at 0.024 confidence; without a matched comparison,
    "we trained a detector" is an assertion rather than a result.
    """
    try:
        from ultralytics import YOLO, YOLOWorld
    except ImportError:
        raise SystemExit("pip install ultralytics")

    root = Path(args.root)
    img_dir = root / "test" / "images"
    if not img_dir.is_dir():
        raise SystemExit(f"no test split at {img_dir} — run prepare first")
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    if not imgs:
        raise SystemExit("test split is empty")
    imgs = imgs[:args.limit] if args.limit else imgs
    print(f"comparing on {len(imgs)} held-out frames\n")

    results = {}

    # --- custom detector -------------------------------------------------
    if os.path.exists(args.model):
        custom = YOLO(args.model)
        confs, n_det = [], 0
        for p in imgs:
            r = custom.predict(str(p), conf=args.conf, verbose=False)[0]
            c = r.boxes.conf.cpu().numpy() if r.boxes is not None else []
            n_det += len(c)
            confs.extend(float(v) for v in c)
        results["custom"] = {
            "model": args.model,
            "detections": n_det,
            "per_frame": round(n_det / len(imgs), 2),
            "mean_conf": round(float(sum(confs) / len(confs)), 4) if confs else 0.0,
            "max_conf": round(max(confs), 4) if confs else 0.0,
        }
    else:
        print(f"(no custom model at {args.model} — train it first)\n")

    # --- zero-shot baseline ----------------------------------------------
    zs = YOLOWorld(args.zeroshot_model)
    zs.set_classes(CLASSES)
    confs, n_det = [], 0
    for p in imgs:
        r = zs.predict(str(p), conf=args.conf, verbose=False)[0]
        c = r.boxes.conf.cpu().numpy() if r.boxes is not None else []
        n_det += len(c)
        confs.extend(float(v) for v in c)
    results["zero_shot"] = {
        "model": args.zeroshot_model,
        "detections": n_det,
        "per_frame": round(n_det / len(imgs), 2),
        "mean_conf": round(float(sum(confs) / len(confs)), 4) if confs else 0.0,
        "max_conf": round(max(confs), 4) if confs else 0.0,
    }

    print(f"{'':12s} {'detections':>11s} {'per frame':>10s} {'mean conf':>10s} {'max conf':>9s}")
    for k, v in results.items():
        print(f"{k:12s} {v['detections']:11d} {v['per_frame']:10.2f} "
              f"{v['mean_conf']:10.4f} {v['max_conf']:9.4f}")

    if "custom" in results:
        a, b = results["custom"]["mean_conf"], results["zero_shot"]["mean_conf"]
        print(f"\nmean-confidence ratio custom/zero-shot: "
              f"{(a / b if b else float('inf')):.1f}x")
        print("\nNote: confidence is not accuracy. For a proper mAP against the test "
              "labels use:\n    yolo val model=%s data=%s split=test" % (args.model, root / "data.yaml"))

    out = root / "detector_comparison.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("classes", help="print/write the class list")
    c.add_argument("--output", default=None)
    c.set_defaults(func=cmd_classes)

    p = sub.add_parser("prepare", help="split frames and write data.yaml")
    p.add_argument("--root", default="dataset")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_prepare)

    v = sub.add_parser("validate", help="check the dataset before training")
    v.add_argument("--root", default="dataset")
    v.set_defaults(func=cmd_validate)

    m = sub.add_parser("compare", help="custom detector vs zero-shot")
    m.add_argument("--root", default="dataset")
    m.add_argument("--model", default="hardware_model.pt")
    m.add_argument("--zeroshot-model", default="yolov8l-worldv2.pt")
    m.add_argument("--conf", type=float, default=0.01)
    m.add_argument("--limit", type=int, default=0, help="0 = all test frames")
    m.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
