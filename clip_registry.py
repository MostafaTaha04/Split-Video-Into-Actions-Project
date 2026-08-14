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


def _as_tuples(clips) -> Dict[str, Tuple[str, str, float]]:
    """Legacy mapping name -> (results_dir, ground_truth, fps)."""
    return {c["name"]: (c["results_dir"], c["ground_truth"], float(c["fps"])) for c in clips}


def dev_clips(src: str = ".") -> Dict[str, Tuple[str, str, float]]:
    """Clips that may inform tuning (the development set)."""
    return _as_tuples([c for c in load_registry(src) if c["split"] == "dev"])


def heldout_clips(src: str = ".") -> Dict[str, Tuple[str, str, float]]:
    """Clips that take no part in tuning at any stage."""
    return _as_tuples([c for c in load_registry(src) if c["split"] == "heldout"])


def clean_names(src: str = ".", split: str = "dev") -> List[str]:
    """Names of continuously-recorded clips — the footage the method targets."""
    return [c["name"] for c in load_registry(src)
            if c["footage"] == "clean" and (split is None or c["split"] == split)]


def video_for(name: str, src: str = ".") -> str:
    for c in load_registry(src):
        if c["name"] == name:
            return c["video"]
    raise SystemExit(f"clip {name!r} not in the registry")


def add_clip(entry: dict, src: str = ".") -> None:
    """Append a validated entry to clips.json, preserving the file's comments."""
    path = os.path.join(src, REGISTRY_FILE)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    names = {c["name"] for c in data.get("clips", [])}
    if entry["name"] in names:
        raise SystemExit(f"clip {entry['name']!r} is already registered")
    missing = [k for k in REQUIRED if k not in entry]
    if missing:
        raise SystemExit(f"new clip is missing {missing}")
    data.setdefault("clips", []).append(entry)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    load_registry(src)   # re-validate the written file


def summary(src: str = ".") -> str:
    clips = load_registry(src)
    dev = [c for c in clips if c["split"] == "dev"]
    held = [c for c in clips if c["split"] == "heldout"]
    clean = [c for c in clips if c["footage"] == "clean"]
    return (f"{len(clips)} clips: {len(dev)} dev, {len(held)} held out; "
            f"{len(clean)} clean, {len(clips) - len(clean)} edited")


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "."
    print(summary(src))
    for c in load_registry(src):
        print(f"  {c['name']:22s} {c['footage']:7s} {c['split']:8s} "
              f"{c['fps']:7.3f} fps  {c['results_dir']}")
