"""
split_recording.py
==================
Cut one long recording into per-procedure clips, without creating the kind of
fake dataset that inflates results.

The distinction this tool enforces
----------------------------------
Cutting a recording into **disjoint segments, each covering a different
procedure**, produces real data: no frame appears twice, and each clip is a
distinct task.

Cutting it into **overlapping windows** does not. Earlier in this project a
folder of 30 "new" clips turned out to be six overlapping windows over each of
five existing videos — clips 3 and 4 shared 17 seconds of identical frames.
Registered as independent clips they would have leaked near-duplicate frames
between training and test and inflated every score. This tool refuses to
produce overlapping segments for that reason.

Clips cut from one recording are still **correlated** — same lighting, same
hardware, same operator — so they are all tagged with the same ``source``.
Evaluation groups by source, so a clip is never tested against a model trained
on its own sibling.

Usage
-----
    # 1. see what is in the recording (contact sheet every 10 s)
    python split_recording.py overview --video session1.mp4 --step 10

    # 2. write a cut list (plain text, one line per clip):
    #      start  end  name
    #      0:00   1:30  open case
    #      1:30   3:10  install RAM
    #      3:10   4:50  mount cooler

    # 3. cut them
    python split_recording.py cut --video session1.mp4 --cuts cuts.txt \\
        --source session1 --output split-video-data

    # 4. register each one (runs the pipeline, makes an annotation stub)
    python add_clip.py --video split-video-data/session1_01_open_case.mp4 \\
        --name "Open case" --footage clean --source session1
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys


def parse_time(s: str) -> float:
    """Accept 90, 1:30 or 0:01:30."""
    s = s.strip()
    if not s:
        raise ValueError("empty time")
    parts = s.split(":")
    if len(parts) > 3:
        raise ValueError(f"bad time: {s}")
    total = 0.0
    for p in parts:
        total = total * 60 + float(p)
    return total


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "clip"


def read_cuts(path: str):
    """Parse the cut list. Lines: START END NAME. '#' starts a comment."""
    cuts = []
    for ln, raw in enumerate(open(path, encoding="utf-8"), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) < 2:
            raise SystemExit(f"{path}:{ln}: expected 'START END [NAME]', got {raw!r}")
        try:
            start, end = parse_time(parts[0]), parse_time(parts[1])
        except ValueError as exc:
            raise SystemExit(f"{path}:{ln}: {exc}")
        if end <= start:
            raise SystemExit(f"{path}:{ln}: end ({parts[1]}) is not after start ({parts[0]})")
        name = parts[2].strip() if len(parts) > 2 else f"clip{len(cuts)+1}"
        cuts.append((start, end, name))
    if not cuts:
        raise SystemExit(f"{path}: no cuts found")
    return cuts


def check_disjoint(cuts):
    """Reject overlaps — the failure mode this tool exists to prevent."""
    ordered = sorted(cuts, key=lambda c: c[0])
    problems = []
    for (s1, e1, n1), (s2, e2, n2) in zip(ordered, ordered[1:]):
        if s2 < e1:
            problems.append(
                f"'{n1}' ({s1:.1f}-{e1:.1f}s) overlaps '{n2}' ({s2:.1f}-{e2:.1f}s) "
                f"by {e1 - s2:.1f}s")
    if problems:
        print("OVERLAPPING SEGMENTS — refusing to cut:\n")
        for p in problems:
            print(f"  {p}")
        print("\nOverlapping clips share frames. Registered as separate clips they leak\n"
              "near-duplicate footage between training and test, which inflates every\n"
              "score. Make the segments disjoint, or cut one long clip instead of two.")
        raise SystemExit(1)


def video_duration(path: str) -> float:
    import cv2
    c = cv2.VideoCapture(path)
    if not c.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = c.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    c.release()
    return n / fps


# ------------------------------------------------------------------ overview
def cmd_overview(args):
    """Contact sheet of the whole recording, to find where procedures start."""
    from annotate import build_parser  # reuse the existing implementation
    argv = ["contact-sheet", "--video", args.video, "--step", str(args.step),
            "--output", args.output, "--cols", "6", "--rows", "6",
            "--cell-width", "220"]
    ns = build_parser().parse_args(argv)
    ns.func(ns)
    dur = video_duration(args.video)
    print(f"\nrecording is {dur/60:.1f} minutes ({dur:.0f}s)")
    print("\nWrite a cut list, one line per procedure, e.g.:\n")
    print("  # start   end     name")
    print("  0:00     1:30    open case")
    print("  1:30     3:10    install RAM")
    print("  3:10     4:50    mount cooler")
    print("\nSegments must not overlap. Aim for 80-100s each, 5-9 steps per clip.")


# ---------------------------------------------------------------------- cut
def cmd_cut(args):
    cuts = read_cuts(args.cuts)
    check_disjoint(cuts)

    dur = video_duration(args.video)
    over = [(s, e, n) for s, e, n in cuts if e > dur + 0.5]
    if over:
        raise SystemExit(
            f"these segments run past the end of the recording ({dur:.1f}s): "
            + ", ".join(f"'{n}' ends {e:.1f}s" for _, e, n in over))

    os.makedirs(args.output, exist_ok=True)
    src_tag = args.source or os.path.splitext(os.path.basename(args.video))[0]

    print(f"recording : {args.video}  ({dur/60:.1f} min)")
    print(f"source tag: {src_tag}   (all clips share this — they are one session)\n")

    written = []
    for i, (start, end, name) in enumerate(sorted(cuts, key=lambda c: c[0]), 1):
        out = os.path.join(args.output, f"{src_tag}_{i:02d}_{slug(name)}.mp4")
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-ss", f"{start:.3f}", "-i", args.video,
               "-t", f"{end - start:.3f}"]
        if args.copy:
            # Stream copy: instant, but cuts land on keyframes so the real start
            # can drift by a second or so. Fine for training data, not for
            # anything where the exact start time matters.
            cmd += ["-c", "copy"]
        else:
            cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-an"]
        cmd.append(out)

        if args.dry_run:
            print(f"  [dry-run] {start:7.1f}-{end:7.1f}s ({end-start:5.1f}s) -> {out}")
            continue

        r = subprocess.run(cmd)
        if r.returncode != 0:
            raise SystemExit(f"ffmpeg failed on '{name}'")
        got = video_duration(out)
        flag = "" if abs(got - (end - start)) < 1.0 else f"  <- got {got:.1f}s, expected {end-start:.1f}s"
        print(f"  {start:7.1f}-{end:7.1f}s ({end-start:5.1f}s) -> {os.path.basename(out)}{flag}")
        written.append((out, name))

    if args.dry_run:
        print("\n[dry-run] nothing written")
        return

    total = sum(e - s for s, e, _ in cuts)
    print(f"\n{len(written)} clips, {total:.0f}s total ({total/dur*100:.0f}% of the recording)")
    print("\nRegister them (each needs a pipeline run and annotation):\n")
    for out, name in written:
        print(f"  python add_clip.py --video {out} \\\n"
              f"      --name \"{name}\" --footage clean --source {src_tag}")
    print(f"\nAll tagged source={src_tag}, so evaluation groups them together "
          "rather than\ntreating them as independent samples.")


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    o = sub.add_parser("overview", help="contact sheet of the whole recording")
    o.add_argument("--video", required=True)
    o.add_argument("--step", type=float, default=10.0, help="seconds between frames")
    o.add_argument("--output", default="annotation_kit")
    o.set_defaults(func=cmd_overview)

    c = sub.add_parser("cut", help="cut disjoint clips from a cut list")
    c.add_argument("--video", required=True)
    c.add_argument("--cuts", required=True, help="text file: START END NAME per line")
    c.add_argument("--output", default="split-video-data")
    c.add_argument("--source", default=None,
                   help="session tag shared by every clip (default: video filename)")
    c.add_argument("--copy", action="store_true",
                   help="stream-copy instead of re-encoding: instant, but cuts snap "
                        "to keyframes and can drift ~1s")
    c.add_argument("--dry-run", action="store_true")
    c.set_defaults(func=cmd_cut)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
