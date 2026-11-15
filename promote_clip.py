"""
promote_clip.py
===============
Move a clip from the AI-draft set into the main, human-annotated evaluation
once a person has annotated it.

Recommended workflow for one clip (YT build C10 as the example):

    # 1. annotate it yourself, into a NEW file (do not overwrite the AI draft)
    python annotate.py review --video split-video-data/ytbuildC_10_seg10.mp4 \
        --output ground_truth/ground_truth_ytbuildc10_human.json --annotator "Your Name"

    # 2. promote it: validates your file, reports how far the AI draft was from
    #    your annotation, and moves the clip into clips.json (held out)
    python promote_clip.py --name "YT build C10" \
        --ground-truth ground_truth_ytbuildc10_human.json

    # 3. re-score everything
    python evaluate_extended.py --src .

Annotating into a new file keeps the AI draft for comparison: the agreement
printed in step 2 measures how good the AI drafts were, which is worth a line
in the report. Labels in your file may be left as the reviewer wrote them.

A clip whose AI draft was reviewed (not re-annotated) by people can also be
promoted, provided the file says so: ``"annotation": "ai_draft"``,
``"review_status": "human_reviewed"``, the people who checked it in ``annotator``
and the check in ``annotation_method``. It is registered with
``"annotation_source": "ai_draft_reviewed"`` so every result can still be split
by provenance.

The clip joins as ``heldout`` by default. Choose ``--split dev`` only if you
decided that before looking at any score on the clip; a clip whose scores you
have seen and then tune on is no longer a fair test.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from clip_registry import (AI_REGISTRY_FILE, add_clip, load_ai_registry, load_registry,
                           resolve_gt)

AI_MARKERS = ("ai draft", "ai-draft")   # annotator text that means "not checked by a person yet"


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def interior_boundaries(gt):
    steps = gt.get("steps") or gt.get("segments") or []
    return [float(s["end"]) for s in steps][:-1]


def agreement(a, b, tol=1.0):
    """Boundary F1 between two annotations, Hungarian one-to-one matching."""
    from utils import MetricsCalculator as M
    if not a or not b:
        return 0.0, 0
    tp = len(M.match_boundaries(a, b, tol))
    p, r = tp / len(a), tp / len(b)
    return (2 * p * r / (p + r) if p + r else 0.0), tp


def is_reviewed_draft(gt):
    return gt.get("annotation") == "ai_draft" and gt.get("review_status") == "human_reviewed"


def check_human_file(path):
    """Problems that stop a file from counting as human-checked ground truth."""
    gt = _load(path)
    problems = []
    annotator = str(gt.get("annotator", "")).strip()
    if not annotator:
        problems.append("annotator is empty")
    elif is_reviewed_draft(gt):
        if any(m in annotator.lower() for m in AI_MARKERS):
            problems.append("a reviewed AI draft must name the people who checked it in 'annotator'")
        method = str(gt.get("annotation_method", "")).lower()
        if "review" not in method and "checked" not in method:
            problems.append("a reviewed AI draft must describe the check in 'annotation_method'")
    elif any(m in annotator.lower() for m in AI_MARKERS):
        problems.append(f"annotator is {annotator!r}: this is still the AI draft, "
                        "annotate the clip yourself into a new file")
    if gt.get("annotation") == "ai_draft" and gt.get("review_status") != "human_reviewed":
        problems.append("file is marked as an unreviewed AI draft")
    if not gt.get("steps") and not gt.get("segments"):
        problems.append("file has no steps")
    return gt, problems


def promote(src, name, gt_file, split="heldout", dry_run=False):
    ai = load_ai_registry(src)
    entry = next((c for c in ai if c["name"] == name), None)
    if entry is None:
        raise SystemExit(f"{name!r} is not in {AI_REGISTRY_FILE}. "
                         f"AI-draft clips: {', '.join(c['name'] for c in ai)}")
    human_path = resolve_gt(gt_file, src)
    if not os.path.exists(human_path):
        raise SystemExit(f"annotation file not found: {gt_file}")
    gt, problems = check_human_file(human_path)
    if problems:
        raise SystemExit("cannot promote:\n  - " + "\n  - ".join(problems))

    here = os.path.dirname(os.path.abspath(__file__))
    val = subprocess.run([sys.executable, os.path.join(here, "annotate.py"), "validate",
                          "--ground-truth", human_path],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(val.stdout.strip())
    if val.returncode != 0:
        raise SystemExit("cannot promote: the validator found errors (above)")

    draft_path = resolve_gt(entry["ground_truth"], src)
    if os.path.exists(draft_path) and os.path.abspath(draft_path) != os.path.abspath(human_path):
        a, h = interior_boundaries(_load(draft_path)), interior_boundaries(gt)
        f1, tp = agreement(a, h)
        print(f"\nAI draft vs your annotation: {len(a)} vs {len(h)} boundaries, "
              f"{tp} within 1 s, agreement F1 {f1:.3f}")

    new = {k: v for k, v in entry.items() if k != "annotation"}
    new["ground_truth"] = os.path.basename(gt_file)
    new["split"] = split
    if is_reviewed_draft(gt):
        new["annotation_source"] = "ai_draft_reviewed"
        what = f"Drafted with an AI tool, checked and approved by {gt['annotator']}"
    else:
        what = f"Annotated by {gt['annotator']}"
    new["note"] = (new.get("note", "") + " " if new.get("note") else "") + \
        f"{what}; promoted from the AI-draft set."
    if dry_run:
        print(f"\n[dry-run] would move {name!r} into clips.json as {split}:")
        print(json.dumps(new, indent=2))
        return new

    path = os.path.join(src, AI_REGISTRY_FILE)
    data = _load(path)
    data["clips"] = [c for c in data["clips"] if c["name"] != name]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    add_clip(new, src)
    reg = load_registry(src)
    hand = sum("annotation_source" not in c for c in reg)
    print(f"\n{name!r} moved into clips.json as {split}. "
          f"clips.json: {len(reg)} clips ({hand} hand-annotated, {len(reg) - hand} AI-drafted "
          f"+ reviewed); AI-draft clips left: {len(load_ai_registry(src))}.")
    print("Next: python evaluate_extended.py --src .")
    return new


def main():
    ap = argparse.ArgumentParser(description="Promote an AI-draft clip after a person annotated it.")
    ap.add_argument("--name", required=True, help='registered clip name, e.g. "YT build C10"')
    ap.add_argument("--ground-truth", required=True, help="your annotation file")
    ap.add_argument("--split", choices=["heldout", "dev"], default="heldout")
    ap.add_argument("--src", default=".")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    promote(a.src, a.name, a.ground_truth, a.split, a.dry_run)


if __name__ == "__main__":
    main()
