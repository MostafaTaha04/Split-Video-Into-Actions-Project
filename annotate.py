"""
annotate.py — ground-truth annotation kit
=========================================

Tooling to produce *frame-accurate, human-verified* step annotations, and to
prove they are human-verified. Evaluation numbers are only as trustworthy as
the reference they are scored against (see Final_Report.docx §7.1), so this
module exists to remove any doubt about how the ground truth was made.

Four sub-commands:

  contact-sheet   Render timestamped frame grids (PNG) for a video, so an
                  annotator can read boundary times off a printed/zoomed sheet
                  instead of scrubbing a player. Two-pass workflow: a coarse
                  sheet to find each transition, then a fine sheet around it.

  review          Interactive player (OpenCV window). Step frame-by-frame,
                  press SPACE to mark a boundary at the current timestamp,
                  and write a ready-to-fill ground-truth JSON on quit.
                  Requires a desktop session — run this on your own machine.

  validate        Check a ground-truth file for the defects that silently
                  corrupt evaluation: non-monotonic or overlapping steps,
                  gaps, coverage that does not span the video, placeholder
                  annotator fields, and suspiciously round timestamps
                  (the signature of an unverified template).

  agreement       Compare two independent annotations of the same video and
                  report inter-annotator agreement (mean boundary offset and
                  F1 of one annotator against the other at several
                  tolerances). This is the number that answers "how do we
                  know your ground truth is any good?".

Typical workflow
----------------
    python annotate.py contact-sheet --video Coolingfaninstallation.mp4 --step 1.0
    # ... read approximate boundary times off the sheet ...
    python annotate.py contact-sheet --video Coolingfaninstallation.mp4 \
        --around 12.5 --window 2.0 --step 0.1        # refine one boundary
    python annotate.py validate --ground-truth ground_truth_coolingfan_v2.json \
        --video Coolingfaninstallation.mp4
    python annotate.py agreement --a gt_fan_annotatorA.json --b gt_fan_annotatorB.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Timestamps that are exact multiples of this are treated as "suspiciously
# round" — a template artefact rather than a read-off-the-video annotation.
ROUND_GRID = 1.0
ROUND_FRACTION_WARN = 0.6


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def open_video(path: str) -> Tuple[cv2.VideoCapture, float, int, float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, n, (n / fps if fps else 0.0)


def read_at(cap: cv2.VideoCapture, t: float) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, frame = cap.read()
    return frame if ok else None


def load_steps(path: str) -> Tuple[List[dict], dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    steps = data.get("steps")
    if steps is None:
        steps = [
            {"start": s["start_time"], "end": s["end_time"],
             "label": s.get("activity", s.get("label", ""))}
            for s in data.get("segments", [])
        ]
    return steps, data


def interior_boundaries(steps: List[dict]) -> List[float]:
    return [float(s["end"]) for s in steps[:-1]]


# --------------------------------------------------------------------------
# contact-sheet
# --------------------------------------------------------------------------
def cmd_contact_sheet(args):
    cap, fps, nframes, duration = open_video(args.video)

    if args.around is not None:
        t0 = max(0.0, args.around - args.window)
        t1 = min(duration, args.around + args.window)
        tag = f"around{args.around:g}"
    else:
        t0, t1 = 0.0, duration
        tag = "full"

    times = list(np.arange(t0, t1 + 1e-9, args.step))
    if not times:
        raise SystemExit("no frames selected — check --step/--window")

    outdir = args.output
    os.makedirs(outdir, exist_ok=True)

    cols = args.cols
    cell_w = args.cell_width
    per_sheet = cols * args.rows
    stem = os.path.splitext(os.path.basename(args.video))[0]

    sheets = 0
    for chunk_start in range(0, len(times), per_sheet):
        chunk = times[chunk_start:chunk_start + per_sheet]
        tiles = []
        for t in chunk:
            frame = read_at(cap, t)
            if frame is None:
                continue
            h, w = frame.shape[:2]
            cell_h = int(cell_w * h / w)
            tile = cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_AREA)

            # Burn in the timestamp and frame index. A dark strip keeps the
            # text readable regardless of the underlying image.
            strip = 26
            tile = cv2.copyMakeBorder(tile, strip, 2, 2, 2,
                                      cv2.BORDER_CONSTANT, value=(20, 20, 20))
            cv2.putText(tile, f"{t:7.2f}s   f{int(round(t * fps)):5d}",
                        (6, strip - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)

        if not tiles:
            continue

        th, tw = tiles[0].shape[:2]
        rows_needed = math.ceil(len(tiles) / cols)
        sheet = np.full((rows_needed * th, cols * tw, 3), 30, np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = tile

        name = f"{stem}_{tag}_{chunk[0]:07.2f}-{chunk[-1]:07.2f}s.png".replace(" ", "0")
        path = os.path.join(outdir, name)
        cv2.imwrite(path, sheet)
        sheets += 1
        print(f"  wrote {path}  ({len(tiles)} frames, {chunk[0]:.2f}s–{chunk[-1]:.2f}s)")

    cap.release()
    print(f"\n{sheets} contact sheet(s) in {outdir}/  "
          f"[video {duration:.2f}s @ {fps:.3f} fps, step {args.step}s]")
    print("Read the transition times off the sheets, then refine each one with:")
    print(f"  python annotate.py contact-sheet --video {args.video} "
          f"--around <t> --window 1.0 --step 0.1")


# --------------------------------------------------------------------------
# review (interactive)
# --------------------------------------------------------------------------
def _require_gui():
    """Fail early and usefully if OpenCV was built without GUI support.

    `requirements-dev.txt` installs ``opencv-python-headless`` (correct for CI,
    which has no display). If it is installed into the same environment as the
    regular ``opencv-python``, it wins, and every cv2 window function raises a
    long build-configuration error that does not say what to do about it.
    """
    try:
        cv2.namedWindow("__probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__probe__")
    except cv2.error:
        raise SystemExit(
            "\nOpenCV has no GUI support, so the interactive reviewer cannot open a window.\n"
            "This happens when 'opencv-python-headless' (installed by requirements-dev.txt\n"
            "for CI) shadows the normal 'opencv-python' in the same environment.\n"
            "\n"
            "Fix, in this environment:\n"
            "    pip uninstall -y opencv-python-headless opencv-python\n"
            "    pip install opencv-python\n"
            "\n"
            "Then re-run this command. Note that reinstalling requirements-dev.txt will\n"
            "re-introduce the headless build; keep the dev/test extras in a separate venv,\n"
            "or re-run the two commands above afterwards.\n"
            "\n"
            "Alternative that needs no GUI at all: use contact sheets and write the JSON\n"
            "by hand --\n"
            f"    python annotate.py contact-sheet --video {getattr(_require_gui, '_video', '<video>')} --step 1.0\n"
        )


def cmd_review(args):
    _require_gui._video = args.video
    _require_gui()
    cap, fps, nframes, duration = open_video(args.video)
    print(__doc__.split("Typical workflow")[0].strip()[:0] or "", end="")
    print(f"""
Interactive review — {os.path.basename(args.video)}
  {duration:.2f}s, {nframes} frames @ {fps:.3f} fps

  RIGHT / LEFT   step 1 frame        d / a   jump 1 second
  w / s          jump 5 seconds      SPACE   mark boundary here
  u              undo last mark      r       reset all marks
  q / ESC        quit and save
""")
    marks: List[float] = []
    idx = 0
    win = "annotate — SPACE marks a boundary"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            idx = max(0, idx - 1)
            continue
        t = idx / fps
        disp = frame.copy()
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 62), (20, 20, 20), -1)
        cv2.putText(disp, f"{t:7.2f}s / {duration:.2f}s   frame {idx}/{nframes}",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp, f"marks: {['%.2f' % m for m in marks[-6:]]}",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        # progress bar with marks
        bw = disp.shape[1]
        cv2.line(disp, (0, 66), (int(bw * t / duration), 66), (0, 200, 255), 3)
        for m in marks:
            x = int(bw * m / duration)
            cv2.line(disp, (x, 60), (x, 74), (0, 0, 255), 2)
        cv2.imshow(win, disp)

        k = cv2.waitKey(0) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == 32:
            if not any(abs(m - t) < 1e-6 for m in marks):
                marks.append(round(t, 2))
                marks.sort()
                print(f"  marked {t:.2f}s   ({len(marks)} total)")
        elif k == ord("u") and marks:
            print(f"  undid {marks.pop():.2f}s")
        elif k == ord("r"):
            marks.clear()
            print("  cleared all marks")
        elif k in (83, ord("l")):
            idx = min(nframes - 1, idx + 1)
        elif k in (81, ord("h")):
            idx = max(0, idx - 1)
        elif k == ord("d"):
            idx = min(nframes - 1, idx + int(fps))
        elif k == ord("a"):
            idx = max(0, idx - int(fps))
        elif k == ord("w"):
            idx = min(nframes - 1, idx + int(5 * fps))
        elif k == ord("s"):
            idx = max(0, idx - int(5 * fps))

    cap.release()
    cv2.destroyAllWindows()

    cuts = [0.0] + marks + [round(duration, 2)]
    steps = [
        {"id": i, "start": cuts[i], "end": cuts[i + 1], "label": f"step_{i + 1}", "notes": ""}
        for i in range(len(cuts) - 1)
    ]
    out = {
        "video": os.path.basename(args.video),
        "annotator": args.annotator,
        "annotation_method": "interactive frame-stepping review (annotate.py review)",
        "video_duration": round(duration, 2),
        "fps": round(fps, 3),
        "steps": steps,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.output} with {len(steps)} steps. "
          f"Fill in the 'label' fields, then run:  python annotate.py validate "
          f"--ground-truth {args.output} --video {args.video}")


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------
def cmd_validate(args):
    steps, data = load_steps(args.ground_truth)
    errors: List[str] = []
    warnings: List[str] = []

    if not steps:
        raise SystemExit("no steps/segments found in file")

    annotator = str(data.get("annotator", "")).strip()
    if not annotator or "VERIFY" in annotator.upper() or annotator.lower() in {
        "manual", "draft", "", "todo", "unknown"
    }:
        errors.append(
            f"annotator field is {annotator!r} — set it to a real person's name. "
            "An unattributed or draft-marked annotation cannot be defended in review."
        )
    if "annotation_method" not in data:
        warnings.append(
            "no 'annotation_method' field — record how the times were obtained "
            "(e.g. 'interactive frame-stepping review')."
        )

    for i, s in enumerate(steps):
        st, en = float(s["start"]), float(s["end"])
        if en <= st:
            errors.append(f"step {i}: end ({en}) <= start ({st})")
        if i > 0:
            prev_end = float(steps[i - 1]["end"])
            if abs(st - prev_end) > 1e-6:
                errors.append(
                    f"step {i}: starts at {st} but previous step ends at {prev_end} "
                    f"({'gap' if st > prev_end else 'overlap'} of {abs(st - prev_end):.2f}s)"
                )
        if not str(s.get("label", "")).strip() or str(s.get("label")).startswith("step_"):
            warnings.append(f"step {i}: placeholder or empty label {s.get('label')!r}")

    ends = [float(s["end"]) for s in steps]
    starts = [float(s["start"]) for s in steps]
    if starts and abs(starts[0]) > 1e-6:
        warnings.append(f"first step starts at {starts[0]}, not 0.0")

    # Round-number detection: the fingerprint of an unverified template.
    all_times = starts[1:] + ends[:-1]
    if all_times:
        n_round = sum(1 for t in all_times if abs(t / ROUND_GRID - round(t / ROUND_GRID)) < 1e-6)
        frac = n_round / len(all_times)
        if frac >= ROUND_FRACTION_WARN:
            errors.append(
                f"{n_round}/{len(all_times)} interior boundaries ({frac:.0%}) are exact "
                f"multiples of {ROUND_GRID}s. Real transitions almost never land on whole "
                "seconds — this looks like an unverified template. Re-annotate from the "
                "video with annotate.py review or contact-sheet."
            )

    if args.video:
        cap, fps, nframes, duration = open_video(args.video)
        cap.release()
        if abs(ends[-1] - duration) > args.duration_tolerance:
            warnings.append(
                f"last step ends at {ends[-1]:.2f}s but video is {duration:.2f}s "
                f"(difference {abs(ends[-1] - duration):.2f}s)"
            )
        quantum = 1.0 / fps
        off_grid = [t for t in all_times if abs(t / quantum - round(t / quantum)) > 0.25]
        if off_grid:
            warnings.append(
                f"{len(off_grid)} boundary time(s) do not lie on a frame boundary "
                f"(1 frame = {quantum:.4f}s); they will be rounded during evaluation."
            )

    print(f"ground truth : {args.ground_truth}")
    print(f"steps        : {len(steps)}   interior boundaries: {len(steps) - 1}")
    print(f"annotator    : {annotator or '(empty)'}")
    print(f"covers       : {starts[0]:.2f}s – {ends[-1]:.2f}s")
    print()
    for w in warnings:
        print(f"  WARN  {w}")
    for e in errors:
        print(f"  ERROR {e}")
    if not errors and not warnings:
        print("  OK — no problems found.")
    print()
    if errors:
        print(f"{len(errors)} error(s), {len(warnings)} warning(s).")
        sys.exit(1)
    print(f"0 errors, {len(warnings)} warning(s).")


# --------------------------------------------------------------------------
# agreement
# --------------------------------------------------------------------------
def cmd_agreement(args):
    from utils import MetricsCalculator

    a_steps, a_data = load_steps(args.a)
    b_steps, b_data = load_steps(args.b)
    A, B = interior_boundaries(a_steps), interior_boundaries(b_steps)

    print(f"A: {args.a}  ({a_data.get('annotator', '?')}) — {len(A)} interior boundaries")
    print(f"B: {args.b}  ({b_data.get('annotator', '?')}) — {len(B)} interior boundaries")
    print()
    print("  tol      P      R     F1    mean|Δt|   matched")
    rows = []
    for tol in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
        m = MetricsCalculator.boundary_accuracy(A, B, tolerance=tol)
        matches = MetricsCalculator.match_boundaries(A, B, tol)
        mae = float(np.mean([o for _, _, o in matches])) if matches else float("nan")
        rows.append((tol, m["precision"], m["recall"], m["f1_score"], mae, len(matches)))
        mae_txt = "      —" if math.isnan(mae) else f"{mae:7.3f}"
        print(f"  {tol:4.2f}  {m['precision']:5.3f}  {m['recall']:5.3f}  "
              f"{m['f1_score']:5.3f}   {mae_txt}s   {len(matches)}/{max(len(A), len(B))}")

    print()
    print("Report the F1 at 1.0s as inter-annotator agreement. It is the ceiling on")
    print("what any automatic method can score against this ground truth: a system")
    print("cannot meaningfully beat the agreement between two humans.")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({
                "annotator_a": a_data.get("annotator"),
                "annotator_b": b_data.get("annotator"),
                "n_boundaries_a": len(A),
                "n_boundaries_b": len(B),
                "agreement": [
                    {"tolerance": t, "precision": p, "recall": r, "f1": f1v,
                     "mean_abs_offset": None if math.isnan(mae) else round(mae, 3),
                     "matched": k}
                    for t, p, r, f1v, mae, k in rows
                ],
            }, f, indent=2)
        print(f"\nwrote {args.output}")


# --------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(
        description="Ground-truth annotation kit for workflow-video segmentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    cs = sub.add_parser("contact-sheet", help="render timestamped frame grids")
    cs.add_argument("--video", required=True)
    cs.add_argument("--output", default="annotation_kit")
    cs.add_argument("--step", type=float, default=1.0, help="seconds between frames")
    cs.add_argument("--around", type=float, default=None, help="centre on this timestamp")
    cs.add_argument("--window", type=float, default=1.5, help="+/- seconds around --around")
    cs.add_argument("--cols", type=int, default=6)
    cs.add_argument("--rows", type=int, default=5)
    cs.add_argument("--cell-width", type=int, default=320)
    cs.set_defaults(func=cmd_contact_sheet)

    rv = sub.add_parser("review", help="interactive frame-stepping annotation")
    rv.add_argument("--video", required=True)
    rv.add_argument("--output", required=True, help="ground-truth JSON to write")
    rv.add_argument("--annotator", required=True, help="your name (goes in the file)")
    rv.set_defaults(func=cmd_review)

    va = sub.add_parser("validate", help="check a ground-truth file")
    va.add_argument("--ground-truth", required=True)
    va.add_argument("--video", default=None, help="cross-check duration and frame grid")
    va.add_argument("--duration-tolerance", type=float, default=0.5)
    va.set_defaults(func=cmd_validate)

    ag = sub.add_parser("agreement", help="inter-annotator agreement between two files")
    ag.add_argument("--a", required=True)
    ag.add_argument("--b", required=True)
    ag.add_argument("--output", default=None, help="write agreement JSON here")
    ag.set_defaults(func=cmd_agreement)

    return ap


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
