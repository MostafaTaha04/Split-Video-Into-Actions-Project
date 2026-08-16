"""
refresh_run_reports.py
----------------------
Rebuild each saved run's segments and ``evaluation_report.txt`` from the run's
own ``features.csv`` and its already-saved boundaries, without re-executing the
heavy vision pipeline.

Why this exists
---------------
Segment construction was corrected so that consecutive segments share their
boundary frame and therefore tile the video exactly. Previously each segment
ended one frame *before* the boundary while the next began *at* it, leaving a
one-frame hole per boundary. That did not move any boundary — and so did not
change boundary precision/recall/F1 — but it understated ``coverage_ratio`` and
``segment_iou`` by about one frame per boundary.

The saved reports were generated before the fix. This script regenerates them
from the *saved boundaries*, so the refreshed numbers correspond exactly to the
runs already in the repository; no re-run and no re-tuning is involved.

Usage:
    python refresh_run_reports.py            # refresh every results_* dir
    python refresh_run_reports.py --dry-run  # show the deltas, write nothing
"""
from __future__ import annotations

import argparse
import csv
import json
import os

from evaluator import Evaluator
from feature_extractor import FrameFeatures
from temporal_segmenter import Boundary, TemporalSegmenter

def runs_from_registry(src="."):
    """results_dir -> (ground truth json, effective fps), from clips.json.

    Derived from the registry rather than hardcoded, so a clip added with
    add_clip.py is refreshed automatically.
    """
    from clip_registry import load_registry

    from clip_registry import resolve_gt, resolve_results
    return {os.path.relpath(resolve_results(c["results_dir"], src), src):
            (os.path.relpath(resolve_gt(c["ground_truth"], src), src), float(c["fps"]))
            for c in load_registry(src)}


def load_features(path: str):
    out = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ff = FrameFeatures(frame_idx=int(float(row["frame_idx"])),
                               timestamp=float(row["timestamp"]))
            for key, val in row.items():
                if key in ("frame_idx", "timestamp") or not hasattr(ff, key):
                    continue
                current = getattr(ff, key)
                try:
                    if isinstance(current, bool):
                        setattr(ff, key, float(val) > 0.5)
                    elif isinstance(current, (int, float)):
                        setattr(ff, key, type(current)(float(val)))
                except (TypeError, ValueError):
                    pass
            out.append(ff)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=".")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"{'run':28s} {'coverage':>18s} {'segment IoU':>18s} {'boundary F1':>14s}")
    print("-" * 82)

    for run, (gtf, fps) in runs_from_registry(args.src).items():
        rdir = os.path.join(args.src, run)
        fcsv = os.path.join(rdir, "features.csv")
        rjson = os.path.join(rdir, "segmentation_results.json")
        gtp = os.path.join(args.src, gtf)
        if not (os.path.exists(fcsv) and os.path.exists(rjson) and os.path.exists(gtp)):
            print(f"{run:28s}  (skipped — missing files)")
            continue

        features = load_features(fcsv)
        saved = json.load(open(rjson, encoding="utf-8"))

        # Saved boundaries record only the timestamp, so recover each one's
        # frame index from the feature stream (first frame at or after it).
        def frame_for(ts: float) -> int:
            for f in features:
                if f.timestamp >= ts - 1e-9:
                    return f.frame_idx
            return features[-1].frame_idx

        boundaries = [
            Boundary(frame_idx=int(b.get("frame_idx", frame_for(float(b["timestamp"])))),
                     timestamp=float(b["timestamp"]),
                     confidence=float(b.get("confidence", 0.0)),
                     reason=b.get("reason", ""),
                     signal_strengths=b.get("signal_strengths"))
            for b in saved["boundaries"]
        ]

        seg = TemporalSegmenter(min_segment_duration=1.5, fps=fps)
        segments = seg._create_segments(boundaries, features)
        seg._label_segments(segments, features)

        gaps = {round(b.start_time - a.end_time, 6) for a, b in zip(segments, segments[1:])}
        assert gaps in ({0.0}, set()), f"{run}: segments still not contiguous: {gaps}"
        assert len(segments) == len(boundaries) + 1, f"{run}: segment/boundary count mismatch"

        ev = Evaluator(gtp)
        old_txt = ""
        rpt_path = os.path.join(rdir, "evaluation_report.txt")
        if os.path.exists(rpt_path):
            old_txt = open(rpt_path, encoding="utf-8").read()

        def grab(text, label):
            for line in text.splitlines():
                if line.startswith(label):
                    return line.split(":")[1].strip()
            return "?"

        metrics = ev.evaluate(segments, boundaries)
        cov_old = grab(old_txt, "Coverage")
        iou_old = grab(old_txt, "Average segment IoU")
        f1_old = grab(old_txt, "F1 Score")
        print(f"{run:28s} {cov_old:>8s} -> {metrics['coverage_ratio']:<7.3f}"
              f" {iou_old:>8s} -> {metrics['segment_iou']:<7.3f}"
              f" {f1_old:>6s} -> {metrics['boundary_metrics']['f1_score']:<6.3f}")

        if args.dry_run:
            continue

        ev.generate_report(segments, boundaries, output_path=rpt_path)

        saved["segments"] = [
            {
                "id": s.segment_id,
                "start_time": round(s.start_time, 3),
                "end_time": round(s.end_time, 3),
                "duration": round(s.duration, 3),
                "activity": s.activity_description,
                "activity_raw": s.dominant_activity,
                "activity_reason": s.activity_reason,
                "activity_confidence": round(s.activity_confidence, 3),
                "real_objects": s.real_objects_used,
                "heuristic_rois": s.heuristic_regions,
                "all_objects_and_rois": s.tools_used,
                "interaction_types": s.interaction_types,
                "confidence": round(s.confidence, 3),
                "avg_activity_level": round(s.avg_activity_level, 3),
                "avg_motion_energy": round(s.avg_motion_energy, 3),
                "visual_stability": round(s.visual_stability, 3),
            }
            for s in segments
        ]
        saved["num_segments"] = len(segments)
        with open(rjson, "w", encoding="utf-8") as fh:
            json.dump(saved, fh, indent=2)

    print("\nBoundary metrics are unchanged by design — only segment extents moved.")


if __name__ == "__main__":
    main()
