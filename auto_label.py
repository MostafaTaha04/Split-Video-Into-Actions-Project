"""
auto_label.py
=============
Bootstrap bounding-box annotation with zero-shot predictions, so the human work
becomes correction rather than creation.

Drawing a box from scratch takes 2-4 minutes per frame across eight classes.
Correcting boxes that are already roughly right takes well under a minute. This
script runs YOLO-World over the extracted frames with prompts mapped onto the
project's class list and writes YOLO-format labels that ``label_boxes.py`` opens
directly.

What to expect
--------------
Section 7.3 of the report measures zero-shot detection at 0.024 confidence on
small hardware parts, so these predictions are NOT a substitute for labelling.
Large, distinctive objects (motherboard, cooler, case) come out roughly right;
small ones (connectors, sockets, RAM) are unreliable and many frames will need
boxes added by hand. The value is in removing the tedious half of the work, not
in avoiding the work.

Every prediction is a draft. Nothing here should reach training without a human
having looked at it.

Usage
-----
    python auto_label.py --root dataset                 # all frames
    python auto_label.py --root dataset --conf 0.02     # more, worse, drafts
    python auto_label.py --root dataset --overwrite     # redo existing labels

Then correct them:
    python label_boxes.py --root dataset
"""
from __future__ import annotations

import argparse
import os

# Text prompts fed to the open-vocabulary detector, mapped onto our class ids.
# Several phrasings per class because zero-shot detection is sensitive to
# wording; whichever fires, the box lands on the right class.
PROMPT_TO_CLASS = {
    "computer processor": "cpu",
    "cpu chip": "cpu",
    "cpu socket": "cpu_socket",
    "processor socket": "cpu_socket",
    "socket retention bracket": "cpu_socket",
    "cpu cooler": "cooler",
    "heatsink": "cooler",
    "computer fan": "cooler",
    "ram stick": "ram_stick",
    "memory module": "ram_stick",
    "ram slot": "ram_slot",
    "memory slot": "ram_slot",
    "cable connector": "connector",
    "wire plug": "connector",
    "motherboard header": "connector",
    "computer motherboard": "motherboard",
    "circuit board": "motherboard",
    "screwdriver": "screwdriver",
}

IMG_EXT = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--model", default="yolov8l-worldv2.pt")
    ap.add_argument("--conf", type=float, default=0.03,
                    help="low on purpose: a wrong box is quick to delete, a "
                         "missing one has to be drawn")
    ap.add_argument("--max-per-frame", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true",
                    help="replace existing label files (default: skip them)")
    args = ap.parse_args()

    try:
        from ultralytics import YOLOWorld
    except ImportError:
        raise SystemExit("pip install ultralytics")

    from detector_kit import CLASSES

    img_dir = os.path.join(args.root, "images")
    lbl_dir = os.path.join(args.root, "labels")
    if not os.path.isdir(img_dir):
        raise SystemExit(f"no images at {img_dir}")
    os.makedirs(lbl_dir, exist_ok=True)

    images = sorted(os.path.join(img_dir, f) for f in os.listdir(img_dir)
                    if os.path.splitext(f)[1].lower() in IMG_EXT)
    if not images:
        raise SystemExit("no images found")

    prompts = [p for p, c in PROMPT_TO_CLASS.items() if c in CLASSES]
    idx_of = {p: CLASSES.index(PROMPT_TO_CLASS[p]) for p in prompts}

    model = YOLOWorld(args.model)
    model.set_classes(prompts)
    print(f"{len(images)} frames  |  {len(prompts)} prompts -> {len(CLASSES)} classes")
    print(f"confidence floor {args.conf} (deliberately low)\n")

    written = skipped = total_boxes = 0
    per_class = {c: 0 for c in CLASSES}

    for i, path in enumerate(images, 1):
        out = os.path.join(lbl_dir, os.path.splitext(os.path.basename(path))[0] + ".txt")
        if os.path.exists(out) and not args.overwrite:
            skipped += 1
            continue

        r = model.predict(path, conf=args.conf, verbose=False)[0]
        lines = []
        if r.boxes is not None and len(r.boxes):
            # highest confidence first, then cap — better a few good drafts
            # than a frame buried under overlapping junk to delete.
            order = r.boxes.conf.argsort(descending=True)[:args.max_per_frame]
            for b in order:
                cls_i = int(r.boxes.cls[b])
                name = prompts[cls_i] if cls_i < len(prompts) else None
                if name is None:
                    continue
                cid = idx_of[name]
                cx, cy, w, h = (float(v) for v in r.boxes.xywhn[b])
                if w <= 0 or h <= 0:
                    continue
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                per_class[CLASSES[cid]] += 1

        with open(out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))
        written += 1
        total_boxes += len(lines)
        if i % 20 == 0 or i == len(images):
            print(f"  {i}/{len(images)}  ({total_boxes} draft boxes so far)", flush=True)

    print(f"\nwrote {written} label files ({skipped} skipped as already labelled)")
    print(f"{total_boxes} draft boxes, {total_boxes/max(written,1):.1f} per frame\n")
    print("draft boxes per class:")
    for c in CLASSES:
        n = per_class[c]
        flag = "   <- few or none; expect to draw these by hand" if n < 10 else ""
        print(f"  {c:14s} {n:5d}{flag}")

    print("\nThese are DRAFTS. Every frame still needs a human pass:")
    print(f"  python label_boxes.py --root {args.root}")
    print("Delete what is wrong, fix what is close, add what is missing.")


if __name__ == "__main__":
    main()
