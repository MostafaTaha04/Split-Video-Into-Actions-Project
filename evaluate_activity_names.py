"""
evaluate_activity_names.py
==========================
Does a step's name say the right kind of motion: turning (screw / unscrew /
tighten / loosen / drive) or moving a part (take / place / lift / insert /
remove / pull ...)? The hand-motion branch of activity_recognizer.py names
these "Rotating / ... motion" and "Grip and move a component" (worded by
direction elsewhere).

Truth comes from the annotated step names, by the leading verb; steps whose
verb is neither kind (open, inspect, connect, ...) or mixes both ("Fetch and
drive the next screw") are left out. Each step is rebuilt from the saved
per-frame features, so no video is needed. Two sets of windows: the annotated
steps themselves, and the segments the appearance scorer finds at its frozen
configuration (each matched to the annotated step covering at least half of
it).

Compared: the labeller without the quick-move rule (QUICK_MOVE_SECONDS=None)
and with it at 3, 4 and 5 seconds.

    python evaluate_activity_names.py      # writes activity_names_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

TURN = {"screw", "unscrew", "tighten", "loosen", "drive", "fasten", "turn", "twist", "rotate", "thread"}
MOVE = {"take", "place", "lift", "put", "set", "remove", "insert", "slide", "pick", "fetch", "lay",
        "lower", "carry", "move", "pull", "push", "reposition", "slip", "seat", "drop", "grab",
        "collect", "gather", "unplug", "plug", "detach"}
NAMED_TURN = {"Rotating / tightening motion", "Screwdriver/screw operation",
              "CPU cooler screw operation", "Secure M.2 SSD"}
NAMED_MOVE = {"Grip and move a component"}
VARIANTS = (None, 3.0, 4.0, 5.0)


def truth(label: str):
    kinds = set()
    for clause in re.split(r"\band\b|,", label.lower()):
        words = re.findall(r"[a-z']+", clause)
        if words and words[0] in TURN:
            kinds.add("turn")
        elif words and words[0] in MOVE:
            kinds.add("move")
    return kinds.pop() if len(kinds) == 1 else None


def named(key: str):
    return "turn" if key in NAMED_TURN else "move" if key in NAMED_MOVE else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", default=".")
    ap.add_argument("--out", default="activity_names_results.json")
    args = ap.parse_args()

    import evaluate_extended as ee
    from activity_recognizer import ActivityRecognizer
    from clip_registry import load_registry, resolve_gt, resolve_results
    from refresh_run_reports import load_features
    from temporal_segmenter import TemporalSegmenter

    windows = {"annotated_steps": [], "predicted_segments": []}
    for c in load_registry(args.src):
        run = resolve_results(c["results_dir"], args.src)
        fcsv = os.path.join(run, "features.csv")
        frames = load_features(fcsv)
        with open(resolve_gt(c["ground_truth"], args.src), encoding="utf-8") as fh:
            steps = json.load(fh)["steps"]
        for s in steps:
            windows["annotated_steps"].append((frames, s["start"], s["end"], s["label"]))
        rows = ee.load_features(fcsv)
        desc = ee.load_appearance(args.src, c["results_dir"], rows)
        if desc is None:
            continue
        cuts = sorted(ee.segment_appearance(rows, desc, float(c.get("fps", 10.0)), 0.75, 1.5))
        edges = [frames[0].timestamp] + cuts + [frames[-1].timestamp + 1e-6]
        for a, b in zip(edges[:-1], edges[1:]):
            best, ov = None, 0.0
            for s in steps:
                o = min(b, s["end"]) - max(a, s["start"])
                if o > ov:
                    best, ov = s, o
            lab = best["label"] if best is not None and ov >= 0.5 * (b - a) else ""
            windows["predicted_segments"].append((frames, a, b, lab))

    seg = TemporalSegmenter(min_segment_duration=1.5, fps=10.0)
    res = {"protocol": __doc__.split("\n\n")[1].replace("\n", " ")}
    for name, wins in windows.items():
        built = []
        for frames, a, b, lab in wins:
            fs = [f for f in frames if a <= f.timestamp < b]
            if len(fs) >= 2 and truth(lab):
                built.append((seg._build_segment(0, fs), truth(lab)))
        res[name] = {"n": len(built), "turn_share": round(sum(t == "turn" for _, t in built) / len(built), 3)}
        for qm in VARIANTS:
            rec = ActivityRecognizer()
            rec.quick_move_seconds = qm
            got = [(named(rec.describe_segment(s, 1.0, 0, 3).key), t) for s, t in built]
            n_turn = sum(g == "turn" for g, _ in got)
            n_move = sum(g == "move" for g, _ in got)
            res[name][f"quick_move_{qm}"] = {
                "right_kind": round(float(np.mean([g == t for g, t in got])), 3),
                "named_turning": n_turn,
                "named_turning_right": round(sum(g == t == "turn" for g, t in got) / max(n_turn, 1), 3),
                "named_moving": n_move,
                "named_moving_right": round(sum(g == t == "move" for g, t in got) / max(n_move, 1), 3),
            }
    with open(os.path.join(args.src, args.out), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

    print("TURNING vs MOVING names (truth: annotated step verbs)")
    for name in windows:
        r = res[name]
        print(f"  {name}: n={r['n']}, turning share {r['turn_share']:.2f}")
        for qm in VARIANTS:
            v = r[f"quick_move_{qm}"]
            print(f"    quick-move {str(qm):5s}: right kind {v['right_kind']:.3f} | 'Rotating' "
                  f"{v['named_turning']:3d} named, {v['named_turning_right']:.2f} right | "
                  f"'Grip and move' {v['named_moving']:3d} named, {v['named_moving_right']:.2f} right")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
