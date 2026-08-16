"""
label_boxes.py
==============
Draw bounding boxes for the custom hardware detector, with no account, no
upload and no new dependencies.

The usual options all have a cost: Roboflow and CVAT require signing up and
uploading the footage, and labelImg pulls PyQt5 into the environment. This tool
uses the OpenCV GUI that the project already depends on, reads the class list
from ``detector_kit.CLASSES`` so the IDs cannot drift, and writes YOLO-format
labels straight into ``dataset/labels/``.

Controls
--------
    left-drag        draw a box for the current class
    0 - 7            choose the class
    n / SPACE        next image      (saves automatically)
    p                previous image  (saves automatically)
    u                undo the last box on this image
    r                clear every box on this image
    h                hide/show the on-screen help
    q / ESC          save and quit

Progress is written as you go, so closing the window never loses work; re-run
and it resumes at the first unlabelled image.

Usage
-----
    python label_boxes.py                      # dataset/images -> dataset/labels
    python label_boxes.py --root dataset --start 40
    python label_boxes.py --only ytbuildB      # just one source clip

Output is YOLO format, one .txt per image: ``class cx cy w h`` with all
coordinates normalised to 0..1.
"""
from __future__ import annotations

import argparse
import os

import cv2

from detector_kit import CLASSES

IMG_EXT = {".jpg", ".jpeg", ".png"}
# Distinct, high-contrast BGR colours — one per class.
COLOURS = [(60, 180, 75), (245, 130, 48), (0, 130, 200), (240, 50, 230),
           (255, 225, 25), (70, 240, 240), (250, 190, 190), (170, 110, 40)]
MAX_DISPLAY = 900          # longest on-screen edge


def label_path(img_path, root):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(root, "labels", stem + ".txt")


def load_boxes(path):
    """YOLO file -> [(cls, cx, cy, w, h)] in normalised coords."""
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        parts = line.split()
        if len(parts) == 5:
            try:
                out.append((int(parts[0]), *(float(v) for v in parts[1:])))
            except ValueError:
                pass
    return out


def save_boxes(path, boxes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for c, cx, cy, w, h in boxes:
            fh.write(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def draw(img, boxes, cls, idx, total, name, show_help, drag=None):
    vis = img.copy()
    H, W = vis.shape[:2]

    for c, cx, cy, w, h in boxes:
        x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
        x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
        col = COLOURS[c % len(COLOURS)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        label = CLASSES[c] if c < len(CLASSES) else str(c)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 6, y1), col, -1)
        cv2.putText(vis, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    if drag:
        cv2.rectangle(vis, drag[0], drag[1], COLOURS[cls % len(COLOURS)], 2)

    bar = 30 if not show_help else 30 + 22 * ((len(CLASSES) + 1) // 2) + 26
    vis = cv2.copyMakeBorder(vis, bar, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
    cv2.putText(vis, f"[{idx+1}/{total}] {name}   boxes: {len(boxes)}   "
                     f"CLASS {cls}: {CLASSES[cls]}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                COLOURS[cls % len(COLOURS)], 1, cv2.LINE_AA)

    if show_help:
        y = 48
        for i, name_ in enumerate(CLASSES):
            x = 10 if i % 2 == 0 else 300
            cv2.putText(vis, f"{i}: {name_}", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        COLOURS[i % len(COLOURS)], 1, cv2.LINE_AA)
            if i % 2 == 1:
                y += 22
        if len(CLASSES) % 2 == 1:
            y += 22
        cv2.putText(vis, "drag=box  n/p=next/prev  u=undo  r=clear  h=help  q=quit",
                    (10, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)
    return vis, bar


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--start", type=int, default=0, help="start at this index")
    ap.add_argument("--only", default=None, help="only images whose name contains this")
    ap.add_argument("--stride", type=int, default=1,
                    help="label every Nth frame (2 halves the work). Skipped frames "
                         "stay on disk but are ignored by detector_kit prepare, which "
                         "only uses frames that actually have a label file.")
    args = ap.parse_args()

    img_dir = os.path.join(args.root, "images")
    if not os.path.isdir(img_dir):
        raise SystemExit(f"no images at {img_dir} — run extract_training_frames.py first")
    images = sorted(os.path.join(img_dir, f) for f in os.listdir(img_dir)
                    if os.path.splitext(f)[1].lower() in IMG_EXT)
    if args.only:
        images = [p for p in images if args.only in os.path.basename(p)]
    if args.stride > 1:
        images = images[::args.stride]
    if not images:
        raise SystemExit("no images matched")

    done = sum(1 for p in images if load_boxes(label_path(p, args.root)))
    print(f"{len(images)} images, {done} already have labels")
    print(__doc__.split("Controls")[1].split("Usage")[0])

    # Resume at the first unlabelled image unless told otherwise.
    idx = args.start
    if not args.start:
        for i, p in enumerate(images):
            if not os.path.exists(label_path(p, args.root)):
                idx = i
                break

    win = "label_boxes  —  drag to draw, q to quit"
    try:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    except cv2.error:
        raise SystemExit(
            "OpenCV has no GUI support in this environment.\n"
            "    pip uninstall -y opencv-python-headless opencv-python\n"
            "    pip install opencv-python")

    state = {"cls": 0, "drag": None, "down": None, "bar": 30, "help": True}

    def on_mouse(event, x, y, flags, _):
        y -= state["bar"]                      # the top bar is not part of the image
        if event == cv2.EVENT_LBUTTONDOWN:
            state["down"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["down"]:
            state["drag"] = (state["down"], (x, y))
        elif event == cv2.EVENT_LBUTTONUP and state["down"]:
            state["drag"] = None
            (x0, y0), (x1, y1) = state["down"], (x, y)
            state["down"] = None
            state["new"] = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    cv2.setMouseCallback(win, on_mouse)

    boxes = load_boxes(label_path(images[idx], args.root))
    img = None
    scale = 1.0

    while True:
        if img is None:
            raw = cv2.imread(images[idx])
            if raw is None:
                idx = (idx + 1) % len(images)
                continue
            h, w = raw.shape[:2]
            scale = min(1.0, MAX_DISPLAY / max(h, w))
            img = cv2.resize(raw, (int(w * scale), int(h * scale))) if scale < 1 else raw
            boxes = load_boxes(label_path(images[idx], args.root))

        # a completed drag becomes a box
        if state.get("new"):
            x1, y1, x2, y2 = state.pop("new")
            H, W = img.shape[:2]
            x1, x2 = max(0, min(x1, W)), max(0, min(x2, W))
            y1, y2 = max(0, min(y1, H)), max(0, min(y2, H))
            if x2 - x1 > 4 and y2 - y1 > 4:      # ignore stray clicks
                boxes.append((state["cls"],
                              (x1 + x2) / 2 / W, (y1 + y2) / 2 / H,
                              (x2 - x1) / W, (y2 - y1) / H))

        vis, bar = draw(img, boxes, state["cls"], idx, len(images),
                        os.path.basename(images[idx]), state["help"], state["drag"])
        state["bar"] = bar
        cv2.imshow(win, vis)
        k = cv2.waitKey(20) & 0xFF

        if k == 255:
            continue
        if ord("0") <= k <= ord("7"):
            state["cls"] = k - ord("0")
        elif k in (ord("n"), 32, 83):
            save_boxes(label_path(images[idx], args.root), boxes)
            idx = (idx + 1) % len(images); img = None
        elif k in (ord("p"), 81):
            save_boxes(label_path(images[idx], args.root), boxes)
            idx = (idx - 1) % len(images); img = None
        elif k == ord("u") and boxes:
            boxes.pop()
        elif k == ord("r"):
            boxes = []
        elif k == ord("h"):
            state["help"] = not state["help"]
        elif k in (ord("q"), 27):
            save_boxes(label_path(images[idx], args.root), boxes)
            break

    cv2.destroyAllWindows()
    labelled = sum(1 for p in images if load_boxes(label_path(p, args.root)))
    total = sum(len(load_boxes(label_path(p, args.root))) for p in images)
    print(f"\n{labelled}/{len(images)} images labelled, {total} boxes total")
    print(f"next: python detector_kit.py validate --root {args.root}")


if __name__ == "__main__":
    main()
