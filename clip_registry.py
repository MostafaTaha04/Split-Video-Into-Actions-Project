"""
clip_registry.py
================
Single source of truth for which clips are in the evaluation set.

Every evaluation script reads the registry through this module, so adding a clip
is a data change (``clips.json``) rather than a code change in five places. That
matters once the dataset grows beyond a handful of clips — the previous design
hardcoded the clip list in ``evaluate_extended.py`` and ``refresh_run_reports.py``
independently, which would silently drift.

Registry fields
---------------
``name``          display name used in tables and figures
``video``         source .mp4 (gitignored; needed only to re-run the pipeline)
``results_dir``   folder holding features.csv / segmentation_results.json
``ground_truth``  annotation JSON (see ANNOTATION_PROTOCOL.md)
``fps``           effective processing rate of the saved run, not the source fps
``footage``       ``clean`` (continuous single shot) or ``edited`` (cuts/overlays)
``split``         ``dev`` (may inform tuning) or ``heldout`` (never used for tuning)
``annotation_source`` (optional) ``ai_draft_reviewed`` for a clip whose ground truth
                  was drafted by an AI and then reviewed by people before it was
                  promoted (promote_clip.py). Absent = annotated by hand.

AI-drafted annotations live in a second registry
-----------------------------------------------
``clips_ai_draft.json`` lists clips whose ground truth was drafted by an AI
from the video frames and has not yet been checked by a person. They are kept
out of ``clips.json`` on purpose: ``dev_clips``, ``heldout_clips`` and
``clean_names`` read only ``clips.json``, so an unreviewed annotation can never
enter tuning, the leave-one-clip-out study or the headline held-out number.
``ai_draft_clips`` returns them for the separately reported evaluation. Once a
person has reviewed a clip's annotation (``review_status: human_reviewed``),
move its entry from ``clips_ai_draft.json`` to ``clips.json``.

Usage
-----
    from clip_registry import load_registry, dev_clips, clean_names
    reg = load_registry(".")
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

REGISTRY_FILE = "clips.json"
AI_REGISTRY_FILE = "clips_ai_draft.json"

REQUIRED = ("name", "video", "results_dir", "ground_truth", "fps", "footage", "split")
VALID_FOOTAGE = {"clean", "edited"}
VALID_SPLIT = {"dev", "heldout"}


def load_registry(src: str = ".") -> List[dict]:
    """Read and validate clips.json. Raises SystemExit with a precise message."""
    path = os.path.join(src, REGISTRY_FILE)
    if not os.path.exists(path):
        raise SystemExit(
            f"No clip registry at {path}.\n"
            "It lists the evaluation clips. Create one with add_clip.py, or restore "
            "it from version control."
        )
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    clips = data.get("clips", [])
    if not clips:
        raise SystemExit(f"{path} contains no clips.")

    seen = set()
    for i, c in enumerate(clips):
        missing = [k for k in REQUIRED if k not in c]
        if missing:
            raise SystemExit(f"{path}: clip {i} is missing {missing}")
        if c["name"] in seen:
            raise SystemExit(f"{path}: duplicate clip name {c['name']!r}")
        seen.add(c["name"])
        if c["footage"] not in VALID_FOOTAGE:
            raise SystemExit(
                f"{path}: clip {c['name']!r} has footage={c['footage']!r}; "
                f"expected one of {sorted(VALID_FOOTAGE)}")
        if c["split"] not in VALID_SPLIT:
            raise SystemExit(
                f"{path}: clip {c['name']!r} has split={c['split']!r}; "
                f"expected one of {sorted(VALID_SPLIT)}")
    return clips


# Folders searched for run outputs and annotations, in order. The registry
# stores bare names, so the repository can be tidied — results into results/,
# annotations into ground_truth/ — without editing clips.json or any script.
RESULT_DIRS = (".", "results")
GT_DIRS = (".", "ground_truth", "annotations")


def resolve_results(name: str, src: str = ".") -> str:
    for d in RESULT_DIRS:
        p = os.path.join(src, name) if d == "." else os.path.join(src, d, name)
        if os.path.isdir(p):
            return p
    return os.path.join(src, name)      # report the conventional path in errors


def resolve_gt(name: str, src: str = ".") -> str:
    for d in GT_DIRS:
        p = os.path.join(src, name) if d == "." else os.path.join(src, d, name)
        if os.path.exists(p):
            return p
    return os.path.join(src, name)


def has_annotations(clip: dict, src: str = ".") -> bool:
    """True if the clip's ground truth actually contains steps.

    add_clip.py registers a clip with an EMPTY annotation stub so the pipeline
    run and the registration can happen before the manual annotation work. Such
    a clip must be excluded from evaluation until it is annotated: counted as
    having zero boundaries it contributes an infinite over-segmentation ratio,
    an F1 of 0, and (worse) several hundred unlabelled frames that dilute the
    positive rate for every learned scorer.
    """
    path = resolve_gt(clip["ground_truth"], src)
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(data.get("steps") or data.get("segments"))


def _as_tuples(clips, src=".") -> Dict[str, Tuple[str, str, float]]:
    """Mapping name -> (results_dir, ground_truth, fps), paths resolved.

    Paths are resolved here rather than at each call site so the repository
    layout can change without touching any evaluation script.
    """
    return {c["name"]: (os.path.relpath(resolve_results(c["results_dir"], src), src),
                        os.path.relpath(resolve_gt(c["ground_truth"], src), src),
                        float(c["fps"])) for c in clips}


def _annotated(clips, src):
    """Drop unannotated clips, saying so once so the omission is visible."""
    keep, skip = [], []
    for c in clips:
        (keep if has_annotations(c, src) else skip).append(c)
    if skip:
        names = ", ".join(f"{c['name']!r}" for c in skip)
        print(f"[registry] skipping {len(skip)} unannotated clip(s): {names}"
              f"  (annotate them, then they join the evaluation automatically)")
    return keep


def dev_clips(src: str = ".") -> Dict[str, Tuple[str, str, float]]:
    """Annotated clips that may inform tuning (the development set)."""
    return _as_tuples(_annotated([c for c in load_registry(src) if c["split"] == "dev"], src), src)


def heldout_clips(src: str = ".") -> Dict[str, Tuple[str, str, float]]:
    """Annotated clips that take no part in tuning at any stage."""
    return _as_tuples(_annotated([c for c in load_registry(src) if c["split"] == "heldout"], src), src)


def load_ai_registry(src: str = ".") -> List[dict]:
    """Read ``clips_ai_draft.json``; an absent file means no AI-drafted clips.

    Every entry must say ``"annotation": "ai_draft"`` and must not share a name
    with a clip in ``clips.json``, so a clip is never in both sets at once.
    """
    path = os.path.join(src, AI_REGISTRY_FILE)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        clips = json.load(fh).get("clips", [])
    human = {c["name"] for c in load_registry(src)}
    seen = set()
    for i, c in enumerate(clips):
        missing = [k for k in REQUIRED if k not in c]
        if missing:
            raise SystemExit(f"{path}: clip {i} is missing {missing}")
        if c.get("annotation") != "ai_draft":
            raise SystemExit(f"{path}: clip {c['name']!r} must have "
                             "\"annotation\": \"ai_draft\"")
        if c["name"] in human:
            raise SystemExit(f"{path}: clip {c['name']!r} is also in {REGISTRY_FILE}; "
                             "a reviewed clip must be removed from the AI-draft registry")
        if c["name"] in seen:
            raise SystemExit(f"{path}: duplicate clip name {c['name']!r}")
        seen.add(c["name"])
        if c["footage"] not in VALID_FOOTAGE:
            raise SystemExit(f"{path}: clip {c['name']!r} has footage={c['footage']!r}")
    return clips


def ai_draft_clips(src: str = ".") -> Dict[str, Tuple[str, str, float]]:
    """Clips annotated by an AI draft, not yet human-reviewed.

    Never mixed into dev_clips / heldout_clips: they are scored only at the
    frozen global configuration and reported in their own section.
    """
    return _as_tuples(_annotated(load_ai_registry(src), src), src)


def is_hand_annotated(clip: dict) -> bool:
    """True unless the clip's ground truth started as an AI draft."""
    return "annotation_source" not in clip


def hand_heldout_clips(src: str = ".") -> Dict[str, Tuple[str, str, float]]:
    """Held-out clips annotated by hand (no AI-drafted provenance).

    Experiments whose results are already reported on the hand-annotated set
    (learned scorers, module ablations, scorer variants) keep using this set,
    so adding reviewed AI drafts to the held-out set does not silently change
    them or put those clips into a training set.
    """
    return _as_tuples(_annotated([c for c in load_registry(src)
                                  if c["split"] == "heldout" and is_hand_annotated(c)], src), src)


def clean_names(src: str = ".", split: str = "dev") -> List[str]:
    """Names of continuously-recorded clips — the footage the method targets."""
    return [c["name"] for c in _annotated(load_registry(src), src)
            if c["footage"] == "clean" and (split is None or c["split"] == split)]


def video_for(name: str, src: str = ".") -> str:
    for c in load_registry(src):
        if c["name"] == name:
            return c["video"]
    raise SystemExit(f"clip {name!r} not in the registry")


# Where raw .mp4 files may live, in search order. The registry stores only the
# filename, so the videos can be kept at the repository root or gathered into a
# folder without editing clips.json.
VIDEO_DIRS = (".", "split-video-data", "videos", "data/videos")


def resolve_video(filename: str, src: str = ".") -> str:
    """Locate a video by filename, searching the conventional folders.

    Videos are gitignored and often reorganised (moved into a folder for a
    Drive upload, for example). Searching a few known locations means that does
    not silently break every script that reads raw frames.
    """
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
    for d in VIDEO_DIRS:
        p = os.path.join(src, d, filename) if d != "." else os.path.join(src, filename)
        if os.path.exists(p):
            return p
    searched = ", ".join(os.path.join(src, d) for d in VIDEO_DIRS)
    raise SystemExit(
        f"video not found: {filename}\n"
        f"searched: {searched}\n"
        "Videos are gitignored — keep them at the repository root or in "
        "split-video-data/."
    )


AI_REGISTRY_COMMENT = [
    "Clips whose ground truth was DRAFTED WITH AN AI TOOL from the video",
    "frames and has NOT been reviewed by a person. Same fields as clips.json,",
    "plus \"annotation\": \"ai_draft\". These clips never take part in tuning,",
    "leave-one-clip-out or the headline held-out number: evaluate_extended.py",
    "scores them at the frozen global configuration and reports them in a",
    "separate 'ai_draft_heldout' section. After a person reviews a clip's",
    "annotation, move its entry to clips.json.",
]


def add_clip(entry: dict, src: str = ".") -> None:
    """Append a validated entry to clips.json (or, for an AI-drafted
    annotation, to clips_ai_draft.json), preserving the file's comments."""
    ai = entry.get("annotation") == "ai_draft"
    path = os.path.join(src, AI_REGISTRY_FILE if ai else REGISTRY_FILE)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        data = {"_comment": AI_REGISTRY_COMMENT, "clips": []}
    taken = {c["name"] for c in load_registry(src)} | {c["name"] for c in load_ai_registry(src)}
    if entry["name"] in taken:
        raise SystemExit(f"clip {entry['name']!r} is already registered")
    missing = [k for k in REQUIRED if k not in entry]
    if missing:
        raise SystemExit(f"new clip is missing {missing}")
    data.setdefault("clips", []).append(entry)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    load_registry(src)      # re-validate the written files
    load_ai_registry(src)


def summary(src: str = ".") -> str:
    clips = load_registry(src)
    dev = [c for c in clips if c["split"] == "dev"]
    held = [c for c in clips if c["split"] == "heldout"]
    clean = [c for c in clips if c["footage"] == "clean"]
    ai = load_ai_registry(src)
    hand = sum(is_hand_annotated(c) for c in clips)
    return (f"{len(clips)} clips: {len(dev)} dev, {len(held)} held out; "
            f"{len(clean)} clean, {len(clips) - len(clean)} edited; "
            f"{hand} hand-annotated, {len(clips) - hand} AI-drafted + human-reviewed"
            + (f" | + {len(ai)} AI-draft clips (reported separately)" if ai else ""))


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "."
    print(summary(src))
    for c in load_registry(src):
        print(f"  {c['name']:22s} {c['footage']:7s} {c['split']:8s} "
              f"{c['fps']:7.3f} fps  {c['results_dir']}"
              + ("" if is_hand_annotated(c) else "  [AI draft, reviewed]"))
    for c in load_ai_registry(src):
        print(f"  {c['name']:22s} {c['footage']:7s} ai_draft "
              f"{c['fps']:7.3f} fps  {c['results_dir']}")
