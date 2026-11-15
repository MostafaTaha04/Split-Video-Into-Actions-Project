"""
import_segments.py
==================
Convert an external step-annotation table (start, end, label per row) into
this project's ground-truth JSON, so public egocentric datasets can join the
evaluation without re-annotating them by hand.

Written for EgoProceL's PC Assembly / PC Disassembly subsets (Bansal et al.,
ECCV 2022: 14 + 15 egocentric videos, 9 key-steps each, annotated in ELAN by
two annotators and cross-verified), but it accepts any table that has a start
and an end column:

* delimiter auto-detected (comma, tab, semicolon); header row optional;
* times as seconds (``12.48``), ``MM:SS(.mmm)``, ``HH:MM:SS(.mmm)``, or frame
  numbers (``--units frames``, converted with the video's fps);
* the first two time-like columns are taken as start and end; the first
  non-numeric column, if any, as the label.

Datasets like EgoProceL annotate only the *key-steps*, leaving background
(unrelated activity) between them. This project's format tiles the whole
video, so gaps longer than ``--min-gap`` become explicit ``background`` steps;
both the start and the end of every key-step then count as a boundary, which
is how the original dataset defines a step's extent. Shorter gaps are closed
by meeting in the middle.

    python import_segments.py --video split-video-data/pc_assembly_01.mp4 \\
        --table annotations/pc_assembly/pc_assembly_01.csv \\
        --annotator "EgoProceL (Bansal et al., 2022)" \\
        --out ground_truth/ground_truth_pcassembly01.json

    python annotate.py validate --ground-truth ground_truth/ground_truth_pcassembly01.json
    python add_clip.py ...   # register it (split: dev or heldout)

Check the first converted file by eye (annotate.py contact-sheet) before
importing a whole dataset: if the table's columns are not what this script
assumes, the printed step list will look wrong immediately.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re

import cv2

_TIME = re.compile(r"^\s*(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)\s*$")


def parse_time(v: str):
    v = v.strip()
    try:
        return float(v)
    except ValueError:
        m = _TIME.match(v)
        if not m:
            return None
        h, mnt, sec = m.groups()
        return int(h or 0) * 3600 + int(mnt) * 60 + float(sec)


def read_table(path: str):
    with open(path, encoding="utf-8-sig") as fh:
        text = fh.read()
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",\t;")
    except csv.Error:
        dialect = csv.excel_tab if "\t" in text else csv.excel
    rows = [r for r in csv.reader(io.StringIO(text), dialect) if any(c.strip() for c in r)]
    out = []
    for r in rows:
        times = [(i, parse_time(c)) for i, c in enumerate(r)]
        tcols = [(i, t) for i, t in times if t is not None]
        if len(tcols) < 2:
            continue                      # header or malformed row
        (i0, t0), (i1, t1) = tcols[0], tcols[1]
        label = next((c.strip() for i, c in enumerate(r)
                      if i not in (i0, i1) and parse_time(c) is None and c.strip()), "")
        out.append((t0, t1, label))
    return out


def to_steps(segs, duration, min_gap=0.5, background="background"):
    segs = sorted((max(0.0, a), min(duration, b), lab) for a, b, lab in segs if b > a)
    steps, t = [], 0.0
    for a, b, lab in segs:
        if a - t >= min_gap:
            steps.append([t, a, background])
        elif steps and a > t:             # tiny gap: meet in the middle
            mid = (t + a) / 2
            steps[-1][1] = mid
            a = mid
        elif a < t:                        # overlap with previous: clip
            a = t
        if b > a:
            steps.append([a, b, lab or f"Step {len(steps) + 1}"])
            t = b
    if duration - t >= min_gap:
        steps.append([t, duration, background])
    elif steps:
        steps[-1][1] = duration
    return [{"id": i, "start": round(s, 3), "end": round(e, 3), "label": lab}
            for i, (s, e, lab) in enumerate(steps)]


def main():
    ap = argparse.ArgumentParser(description="Import an external step table as ground truth.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--units", choices=["seconds", "frames"], default="seconds")
    ap.add_argument("--min-gap", type=float, default=0.5,
                    help="gaps at least this long become 'background' steps (s)")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    cap.release()

    segs = read_table(a.table)
    if not segs:
        raise SystemExit(f"no (start, end) rows found in {a.table}")
    if a.units == "frames":
        segs = [(s / fps, e / fps, lab) for s, e, lab in segs]
    steps = to_steps(segs, duration, a.min_gap)

    data = {"video": os.path.basename(a.video), "annotator": a.annotator,
            "annotation_method": f"imported from {os.path.basename(a.table)} (import_segments.py)",
            "video_duration": round(duration, 2), "fps": round(fps, 3), "steps": steps}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    print(f"wrote {a.out}: {len(steps)} steps from {len(segs)} annotated segments, {duration:.1f}s")
    for s in steps:
        print(f"  {s['id']:2d}  {s['start']:8.2f} - {s['end']:8.2f}  {s['label']}")


if __name__ == "__main__":
    main()
