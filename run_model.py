"""
run_model.py
============
The activity model, end to end: it finds the 13 activities in a video by itself
and cuts the video where the activity changes.

    python run_model.py                   # the whole project: test + train (below)
    python run_model.py pc_assembly_01    # one video, by its name or its path
    python run_model.py test              # test on every annotated video
    python run_model.py train             # train the final model on all annotated videos
    python run_model.py final             # the real test: new videos the model never saw

  one video  Give its name (a video in data/videos or data/new_videos, with or
         without .mp4) or the path of any .mp4 file. The result is
         outputs/<name>/model.html: the video, the timeline under it and the clips.
         A training video (in data/videos, with an annotation file) is cut by a
         model trained on the OTHER videos only, as in the test, so the result is
         fair; its scores are printed and its annotation is shown underneath.
         Any other video is cut by the saved model (models/activity_model.pkl).
         Add --task assembly or --task disassembly if you know which it is.
         ("python run_model.py cut VIDEO" does the same.)
  test   Each annotated video is cut by a model trained on the OTHER videos only
         ("leave one video out"), so every video is new to the model that cuts it.
         The model's cuts are compared with the annotation file. Writes, per video,
         outputs/<name>/model.html (the player: the model's activities, with the
         annotation underneath) and its clips in outputs/<name>/model_clips/, and
         outputs/results.html with all the scores (also results.json, results.md).
  train  Trains on all annotated videos and saves models/activity_model.pkl.
  final  The real test. Trains on all annotated videos in data/videos, then cuts
         every video in data/new_videos - videos the model has never seen and
         that never take part in training or in choosing the model's settings.
         If such a video has an annotation file (data/annotations/<name>.json),
         its cuts are scored against it. Writes outputs/final_results.html and,
         per video, outputs/<name>/model.html with its clips.

Options:
  --fresh     describe the frames again instead of using features/ (a few minutes per video)
  --no-clips  do not write the per-activity clips (faster)
  --no-open   do not open the page in the browser

The first run describes the frames of every video (features.py), about 1-2
minutes per video, and downloads the image model once (about 100 MB). After
that, testing and training take well under a minute; building the players and
clips takes a little longer.
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import os
import sys
import time
import traceback
from typing import List

import numpy as np

import features
from evaluate import confusion, score
from labels import BACKGROUND, CLASSES, LABEL_COLORS
from model import ActivityModel, Settings, labels_at, task_of
from show_all import review_status
from viewer import build_viewer, open_in_browser, video_info

ROOT = os.path.dirname(os.path.abspath(__file__))
VIDEOS = os.path.join(ROOT, "data", "videos")
ANNOTATIONS = os.path.join(ROOT, "data", "annotations")
NEW_VIDEOS = os.path.join(ROOT, "data", "new_videos")   # kept aside for the real test
OUTPUTS = os.path.join(ROOT, "outputs")
MODEL_PATH = os.path.join(ROOT, "models", "activity_model.pkl")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def annotated_videos() -> List[str]:
    names = []
    for v in sorted(glob.glob(os.path.join(VIDEOS, "*.mp4"))):
        n = os.path.splitext(os.path.basename(v))[0]
        if os.path.exists(os.path.join(ANNOTATIONS, n + ".json")):
            names.append(n)
    return names


def load_video(name: str, fresh: bool = False) -> dict:
    path = os.path.join(ANNOTATIONS, name + ".json")
    with open(path, encoding="utf-8") as fh:
        ann = json.load(fh)
    feats = features.load(name, fresh=fresh)
    return {"name": name, "features": feats, "segments": ann.get("segments", []),
            "task": task_of(ann, name), "duration": float(ann.get("duration_s") or 0) or None,
            "status": review_status(path)}


def write_steps(path: str, name: str, result: dict, trained_on: List[str]) -> None:
    doc = {"video": name + ".mp4", "made_by": "run_model.py (activity model)",
           "trained_on": trained_on, "task": result["task"],
           "task_probability": {k: round(v, 3) for k, v in result["task_probs"].items()},
           "segments": result["segments"]}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)


def build_player(name: str, video: str, steps: str, ann: str = None, clips: bool = True,
                 back: str = "../results.html", back_label: str = "Results") -> str:
    return build_viewer(video, steps, out_dir=os.path.join(OUTPUTS, name), ground_truth=ann,
                        make_clips=clips, lane_title="Model's activities", back_href=back,
                        back_label=back_label, page_name="model.html", clips_dirname="model_clips")


# --------------------------------------------------------------------------
# test: leave one video out
# --------------------------------------------------------------------------
def test(names: List[str], fresh: bool, clips: bool) -> str:
    print(f"Loading {len(names)} annotated videos...")
    vids = [load_video(n, fresh) for n in names]
    tasks = {v["task"] for v in vids}
    if len(vids) < 2:
        raise SystemExit("The test needs at least 2 annotated videos.")
    rows, pairs = [], []
    t0 = time.time()
    for i, v in enumerate(vids):
        train = [w for j, w in enumerate(vids) if j != i]
        print(f"[{i + 1}/{len(vids)}] {v['name']}: training on the other {len(train)} videos, then cutting it")
        m = ActivityModel(Settings()).fit(train)
        r = m.predict(v["features"], v["duration"])
        gt = list(labels_at(v["features"]["times"], v["segments"]))
        sc = score(r["labels"], gt, float(v["features"]["fps"]))
        sc.update(name=v["name"], task=v["task"], predicted_task=r["task"], status=v["status"],
                  segments=r["segments"], annotation=v["segments"],
                  duration=v["duration"] or float(v["features"]["times"][-1]))
        rows.append(sc)
        pairs.append((r["labels"], gt))
        out_dir = os.path.join(OUTPUTS, v["name"])
        os.makedirs(out_dir, exist_ok=True)
        steps = os.path.join(out_dir, "model_steps.json")
        write_steps(steps, v["name"], r, m.trained_on)
        print(f"    {sc['frame_accuracy']:.0f}% of seconds right, {len(r['segments'])} activities found "
              f"({len(v['segments'])} annotated), task: {r['task']}"
              + ("" if r["task"] == v["task"] else f"  (WRONG, it is {v['task']})"))
        video = os.path.join(VIDEOS, v["name"] + ".mp4")
        if r["segments"]:
            try:
                build_player(v["name"], video, steps, os.path.join(ANNOTATIONS, v["name"] + ".json"), clips)
            except Exception:
                traceback.print_exc()
                print(f"    {v['name']}: the player could not be built; the scores are still saved")
    print(f"Done in {time.time() - t0:.0f} s.")
    if len(tasks) < 2:
        print("Note: all videos are of one task, so assembly vs disassembly was not really tested.")
    return write_results(rows, confusion(pairs))


# --------------------------------------------------------------------------
# results page
# --------------------------------------------------------------------------
KEYS = [("frame_accuracy", "Seconds right %"), ("macro_f1", "Activity F1"),
        ("f1@10", "Segment F1@10"), ("f1@25", "Segment F1@25"), ("f1@50", "Segment F1@50"),
        ("edit", "Order (edit)"), ("cut2", "Cuts ±2 s F1"), ("cut5", "Cuts ±5 s F1")]


def _flat(r: dict) -> dict:
    d = {k: r[k] for k, _ in KEYS if k in r}
    d["cut2"] = 100 * r["cuts"]["f1"]
    d["cut5"] = 100 * r["cuts_5s"]["f1"]
    return d


def _lane(segs: List[dict], T: float, h: int = 18) -> str:
    out = []
    for s in segs:
        c = LABEL_COLORS.get(s["label"], "#8a939c")
        tip = html.escape(f'{s["label"]}  {s["start"]:.0f}-{s["end"]:.0f} s')
        out.append(f'<div class="seg" title="{tip}" style="left:{100 * s["start"] / T:.3f}%;'
                   f'width:{100 * (s["end"] - s["start"]) / T:.3f}%;background:{c}"></div>')
    return f'<div class="lane" style="height:{h}px">{"".join(out)}</div>'


def write_results(rows: List[dict], conf: dict) -> str:
    flat = [_flat(r) for r in rows]
    mean = {k: float(np.mean([f[k] for f in flat])) for k, _ in KEYS}
    # per activity over all videos (seconds)
    act = {}
    for r in rows:
        for c, v in r["per_class"].items():
            a = act.setdefault(c, {"tp": 0.0, "real": 0.0, "pred": 0.0})
            a["real"] += v["real_s"]
            a["pred"] += v["predicted_s"]
            a["tp"] += v["recall"] * v["real_s"]
    for c, a in act.items():
        a["precision"] = a["tp"] / a["pred"] if a["pred"] else 0.0
        a["recall"] = a["tp"] / a["real"] if a["real"] else 0.0
        a["f1"] = (2 * a["precision"] * a["recall"] / (a["precision"] + a["recall"])
                   if a["precision"] + a["recall"] else 0.0)
    order = [c for c in CLASSES if c in act]
    tasks_ok = sum(r["task"] == r["predicted_task"] for r in rows)

    os.makedirs(OUTPUTS, exist_ok=True)
    data = {"videos": [{**{k: r[k] for k in ("name", "task", "predicted_task", "status", "segments")},
                        **{k: round(v, 2) for k, v in _flat(r).items()}} for r in rows],
            "mean": {k: round(v, 2) for k, v in mean.items()},
            "task_correct": f"{tasks_ok}/{len(rows)}",
            "per_activity": {c: {k: round(v, 3) if isinstance(v, float) else v for k, v in act[c].items()}
                             for c in order},
            "confusion_seconds": {g: {p: n / 2 for p, n in d.items()} for g, d in conf.items()},
            "settings": Settings().__dict__}
    with open(os.path.join(OUTPUTS, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

    # markdown, for the report
    md = ["# Results: leave one video out", "",
          f"Each video was cut by a model trained on the other {len(rows) - 1} videos. "
          f"Assembly vs disassembly was recognised correctly in {tasks_ok} of {len(rows)} videos.", "",
          "| Video | " + " | ".join(h for _, h in KEYS) + " |", "|---" * (len(KEYS) + 1) + "|"]
    for r, f in zip(rows, flat):
        md.append(f"| {r['name']} | " + " | ".join(f"{f[k]:.1f}" for k, _ in KEYS) + " |")
    md.append("| **Mean** | " + " | ".join(f"**{mean[k]:.1f}**" for k, _ in KEYS) + " |")
    md += ["", "| Activity | Annotated (s) | Precision | Recall | F1 |", "|---|---|---|---|---|"]
    for c in order:
        a = act[c]
        md.append(f"| {c} | {a['real']:.0f} | {100 * a['precision']:.0f} | {100 * a['recall']:.0f} | {100 * a['f1']:.0f} |")
    with open(os.path.join(OUTPUTS, "results.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")

    # html
    cards = "".join(
        f'<div class="kpi"><div class="v">{mean[k]:.0f}</div><div class="l">{html.escape(h)}</div></div>'
        for k, h in KEYS if k in ("frame_accuracy", "f1@25", "edit", "cut5"))
    head = "".join(f"<th>{html.escape(h)}</th>" for _, h in KEYS)
    vid_rows = []
    for r, f in zip(rows, flat):
        T = max(r["duration"], 1e-6)
        task = html.escape(r["predicted_task"]) + ("" if r["predicted_task"] == r["task"] else " ✗")
        page = f'{r["name"]}/model.html' if r["segments"] else ""
        name = f'<a href="{html.escape(page)}">{html.escape(r["name"])}</a>' if page else html.escape(r["name"])
        vid_rows.append(
            f'<tr><td class="nm">{name}<div class="small">{task}</div></td>'
            + "".join(f"<td>{f[k]:.0f}</td>" for k, _ in KEYS)
            + f'<td class="lanes"><div class="ln">model</div>{_lane(r["segments"], T)}'
              f'<div class="ln">annotation</div>{_lane(r["annotation"], T)}</td></tr>')
    vid_rows.append('<tr class="mean"><td>Mean</td>' + "".join(f"<td>{mean[k]:.1f}</td>" for k, _ in KEYS) + "<td></td></tr>")
    act_rows = "".join(
        f'<tr><td><i style="background:{LABEL_COLORS.get(c, "#8a939c")}"></i>{html.escape(c)}</td>'
        f'<td>{act[c]["real"]:.0f}</td><td>{100 * act[c]["precision"]:.0f}</td>'
        f'<td>{100 * act[c]["recall"]:.0f}</td><td><b>{100 * act[c]["f1"]:.0f}</b></td></tr>' for c in order)
    page = RESULTS.replace("__N__", str(len(rows))).replace("__N1__", str(len(rows) - 1)) \
        .replace("__TASKS__", f"{tasks_ok} of {len(rows)}").replace("__KPIS__", cards) \
        .replace("__HEAD__", head).replace("__ROWS__", "\n".join(vid_rows)).replace("__ACTS__", act_rows)
    out = os.path.join(OUTPUTS, "results.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"\nMean over {len(rows)} videos: " + ", ".join(f"{h} {mean[k]:.1f}" for k, h in KEYS))
    print(f"Assembly vs disassembly right in {tasks_ok} of {len(rows)} videos.")
    print(f"Results page: {out}")
    return out


RESULTS = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model results</title>
<style>
 :root { --ink:#1f2328; --muted:#646c76; --line:#d8dde3; --bg:#f6f7f9; --card:#fff; --accent:#0b62d6; }
 body { margin:0; font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; color:var(--ink); background:var(--bg); }
 main { max-width:1180px; margin:0 auto; padding:20px 20px 48px; }
 h1 { font-size:20px; margin:0 0 2px; } h2 { font-size:16px; margin:26px 0 8px; } .sub { color:var(--muted); margin-bottom:14px; }
 a { color:var(--accent); text-decoration:none; } a:hover { text-decoration:underline; }
 .kpis { display:flex; gap:10px; flex-wrap:wrap; margin:6px 0 4px; }
 .kpi { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 16px; min-width:150px; }
 .kpi .v { font-size:26px; font-weight:650; font-variant-numeric:tabular-nums; } .kpi .l { color:var(--muted); font-size:12px; }
 table { border-collapse:collapse; background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:hidden; width:100%; }
 th, td { padding:6px 8px; border-bottom:1px solid #eef0f3; text-align:right; font-variant-numeric:tabular-nums; vertical-align:middle; }
 th { font-size:12px; color:var(--muted); font-weight:600; background:#fafbfc; }
 td:first-child, th:first-child { text-align:left; } td.nm { white-space:nowrap; } .small { color:var(--muted); font-size:11px; }
 tr.mean td { font-weight:650; background:#fafbfc; }
 td.lanes { width:34%; min-width:260px; text-align:left; } .ln { font-size:10px; color:var(--muted); margin-top:2px; }
 .lane { position:relative; border-radius:4px; background:#eef1f4; overflow:hidden; }
 .seg { position:absolute; top:0; bottom:0; border-right:1px solid #fff; }
 td i { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:-1px; }
 .acts { max-width:620px; }
 dl { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 16px; margin:0; }
 dt { font-weight:600; margin-top:6px; } dd { margin:0 0 4px 0; color:#3b424a; }
</style></head><body><main>
<h1>Model results</h1>
<div class="sub">Each of the __N__ videos was cut by a model trained on the other __N1__ only, then compared with its
annotation file. Assembly vs disassembly was recognised correctly in __TASKS__ videos.
Click a video to open its player: the model's activities, with the annotation underneath, and the model's clips.
· <a href="index.html">Annotation players</a></div>
<div class="kpis">__KPIS__</div>
<h2>Per video</h2>
<table><tr><th>Video</th>__HEAD__<th>Timeline</th></tr>
__ROWS__
</table>
<h2>Per activity (all videos together)</h2>
<table class="acts"><tr><th>Activity</th><th>Annotated (s)</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
__ACTS__
</table>
<h2>What the numbers mean</h2>
<dl>
<dt>Seconds right %</dt><dd>Share of the video (every half second) where the model's label equals the annotation's, BACKGROUND included.</dd>
<dt>Activity F1</dt><dd>For each activity: precision = of the time the model said X, how much really was X; recall = of the real X, how much it found; F1 combines both. Averaged over the activities in the video.</dd>
<dt>Segment F1@10 / @25 / @50</dt><dd>The standard action-segmentation score: a found activity counts if it has the right label and overlaps a real one by at least 10 / 25 / 50 % (intersection over union).</dd>
<dt>Order (edit)</dt><dd>How similar the list of activities is, in order, ignoring exact times (100 = the same list).</dd>
<dt>Cuts ±2 s / ±5 s</dt><dd>Whether the model cuts where the annotation changes activity, within 2 or 5 seconds (the annotations themselves are accurate to about 2 s).</dd>
</dl>
</main></body></html>
"""


# --------------------------------------------------------------------------
# train and cut
# --------------------------------------------------------------------------
def train(names: List[str], fresh: bool) -> ActivityModel:
    vids = [load_video(n, fresh) for n in names]
    print(f"Training the final model on {len(vids)} videos...")
    m = ActivityModel(Settings()).fit(vids)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    m.save(MODEL_PATH)
    print(f"Saved: {MODEL_PATH}")
    return m


COMMANDS = ["all", "test", "train", "final", "cut"]


def find_video(arg: str) -> str:
    """The video given on the command line: a path, or just a name (pc_assembly_01) of a video
    in data/videos or data/new_videos."""
    if os.path.isfile(arg):
        return os.path.abspath(arg)
    name = os.path.basename(arg.rstrip("/\\"))
    if not name.lower().endswith(".mp4"):
        name += ".mp4"
    for folder in (VIDEOS, NEW_VIDEOS):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    raise SystemExit(f"No video called {arg!r}. Write the name of a video in data\\videos or "
                     f"data\\new_videos (for example  python run_model.py pc_assembly_01), "
                     f"or the full path of an .mp4 file.")


def _print_result(name: str, r: dict, how: str) -> None:
    print(f"{name}: {r['task']} ({how}), {len(r['segments'])} activities:")
    for i, s in enumerate(r["segments"], 1):
        print(f"  {i:2d}. {s['label']:11s} {s['start']:7.1f} - {s['end']:7.1f} s")


def _print_scores(sc: dict, task: str, true_task: str) -> None:
    print(f"Compared with the annotation file: {sc['frame_accuracy']:.0f}% of seconds right, "
          f"segment F1@50 {sc['f1@50']:.0f}, order {sc['edit']:.0f}, "
          f"cuts within 5 s F1 {100 * sc['cuts_5s']['f1']:.0f}; task "
          + ("right" if task == true_task else f"WRONG (it is {true_task})"))


def _back() -> tuple:
    for page, label in (("results.html", "Results"), ("index.html", "All videos")):
        if os.path.exists(os.path.join(OUTPUTS, page)):
            return "../" + page, label
    return None, "All videos"


def one(video: str, fresh: bool, clips: bool, task: str = None) -> str:
    """Cut one video and build its page (see 'one video' at the top)."""
    name = os.path.splitext(os.path.basename(video))[0]
    training = annotated_videos()
    if name in training and os.path.dirname(os.path.abspath(video)) == os.path.abspath(VIDEOS):
        others = [n for n in training if n != name]
        if len(others) >= 2:
            return _cut_left_out(name, others, fresh, clips, task)
    return cut(video, fresh, clips, task)


def _cut_left_out(name: str, others: List[str], fresh: bool, clips: bool, task: str = None) -> str:
    print(f"{name} is one of the training videos, so it is cut by a model trained on the other "
          f"{len(others)} videos only (as in the test).")
    v = load_video(name, fresh)
    m = ActivityModel(Settings()).fit([load_video(n) for n in others])
    r = m.predict(v["features"], v["duration"], task=task)
    out_dir = os.path.join(OUTPUTS, name)
    os.makedirs(out_dir, exist_ok=True)
    steps = os.path.join(out_dir, "model_steps.json")
    write_steps(steps, name, r, m.trained_on)
    _print_result(name, r, "given with --task" if task else f"{100 * r['task_probs'][r['task']]:.0f}% sure")
    gt = list(labels_at(v["features"]["times"], v["segments"]))
    _print_scores(score(r["labels"], gt, float(v["features"]["fps"])), r["task"], v["task"])
    if not r["segments"]:
        print("No activities found.")
        return steps
    back, label = _back()
    page = build_player(name, os.path.join(VIDEOS, name + ".mp4"), steps,
                        os.path.join(ANNOTATIONS, name + ".json"), clips, back=back, back_label=label)
    print(f"Player page: {page}")
    return page


def cut(video: str, fresh: bool, clips: bool, task: str = None) -> str:
    """Cut a video with the saved model."""
    video = find_video(video)
    if not os.path.exists(MODEL_PATH):
        raise SystemExit("No trained model yet. Run  python run_model.py train  first.")
    m = ActivityModel.load(MODEL_PATH)
    name = os.path.splitext(os.path.basename(video))[0]
    try:
        feats = features.load(name, fresh=fresh, video=video)
    except RuntimeError as e:
        raise SystemExit(str(e))
    r = m.predict(feats, video_info(video)["duration"], task=task)
    out_dir = os.path.join(OUTPUTS, name)
    os.makedirs(out_dir, exist_ok=True)
    steps = os.path.join(out_dir, "model_steps.json")
    write_steps(steps, name, r, m.trained_on)
    _print_result(name, r, "given with --task" if task else f"{100 * r['task_probs'][r['task']]:.0f}% sure")
    if name in m.trained_on:
        print("Note: the model was trained on this video, so this is not a test on a new video.")
    ann = os.path.join(ANNOTATIONS, name + ".json")
    if os.path.exists(ann):
        with open(ann, encoding="utf-8") as fh:
            a = json.load(fh)
        gt = list(labels_at(feats["times"], a.get("segments", [])))
        _print_scores(score(r["labels"], gt, float(feats["fps"])), r["task"], task_of(a, name))
    if not r["segments"]:
        print("No activities found.")
        return steps
    back, label = _back()
    page = build_player(name, video, steps, ann if os.path.exists(ann) else None, clips,
                        back=back, back_label=label)
    print(f"Player page: {page}")
    return page


# --------------------------------------------------------------------------
# final: the real test on videos kept aside
# --------------------------------------------------------------------------
def _fingerprint(path: str) -> tuple:
    import hashlib
    with open(path, "rb") as fh:
        head = hashlib.md5(fh.read(1 << 20)).hexdigest()
    return os.path.getsize(path), head


def final(names: List[str], fresh: bool, clips: bool) -> str:
    new = sorted(glob.glob(os.path.join(NEW_VIDEOS, "*.mp4")))
    if not new:
        raise SystemExit(f"No videos in {NEW_VIDEOS}. Put the new videos there (not in data/videos) "
                         "and run again.")
    need = [v for v in new if fresh or not os.path.exists(features.cache_path(os.path.splitext(os.path.basename(v))[0]))]
    if need:
        try:
            features.check_packages()
        except RuntimeError as e:
            raise SystemExit(str(e))
    train_prints = {_fingerprint(os.path.join(VIDEOS, n + ".mp4")): n for n in names}
    m = train(names, fresh)
    items = []
    for i, video in enumerate(new, 1):
        name = os.path.splitext(os.path.basename(video))[0]
        print(f"[{i}/{len(new)}] {name}")
        if name in names:
            print(f"    skipped: a training video is also called {name}.mp4 - rename one of them")
            continue
        same = train_prints.get(_fingerprint(video))
        if same:
            print(f"    skipped: this is the same file as the training video {same}.mp4, "
                  "so it would not be a fair test")
            continue
        try:
            feats = features.load(name, fresh=fresh, video=video)
            duration = video_info(video)["duration"]
            r = m.predict(feats, duration)
        except Exception:
            traceback.print_exc()
            print(f"    {name}: FAILED - see the error above; continuing with the next video")
            continue
        out_dir = os.path.join(OUTPUTS, name)
        os.makedirs(out_dir, exist_ok=True)
        steps = os.path.join(out_dir, "model_steps.json")
        write_steps(steps, name, r, m.trained_on)
        ann_path = os.path.join(ANNOTATIONS, name + ".json")
        item = {"name": name, "task": r["task"], "task_prob": r["task_probs"][r["task"]],
                "segments": r["segments"], "duration": duration, "annotation": None, "score": None}
        msg = f"    {r['task']} ({100 * item['task_prob']:.0f}% sure), {len(r['segments'])} activities found"
        if os.path.exists(ann_path):
            with open(ann_path, encoding="utf-8") as fh:
                ann = json.load(fh)
            gt = list(labels_at(feats["times"], ann.get("segments", [])))
            sc = score(r["labels"], gt, float(feats["fps"]))
            item.update(annotation=ann.get("segments", []), score=sc, true_task=task_of(ann, name))
            msg += (f" ({len(ann.get('segments', []))} annotated), {sc['frame_accuracy']:.0f}% of seconds right"
                    + ("" if item["true_task"] == r["task"] else f", task WRONG (it is {item['true_task']})"))
        else:
            msg += " (no annotation file, so no score)"
        print(msg)
        if r["segments"]:
            try:
                build_player(name, video, steps, ann_path if item["annotation"] is not None else None, clips,
                             back="../final_results.html", back_label="Final test")
            except Exception:
                traceback.print_exc()
                print(f"    {name}: the player could not be built")
        items.append(item)
    if not items:
        raise SystemExit("No new video could be cut.")
    return write_final(items, m.trained_on)


def write_final(items: List[dict], trained_on: List[str]) -> str:
    scored = [it for it in items if it["score"]]
    keys = [k for k in KEYS]
    rows = []
    for it in items:
        T = max(it["duration"], 1e-6)
        f = _flat(it["score"]) if it["score"] else None
        task = f'{html.escape(it["task"])} ({100 * it["task_prob"]:.0f}%)'
        if it["score"] and it.get("true_task") != it["task"]:
            task += " ✗"
        lanes = f'<div class="ln">model</div>{_lane(it["segments"], T)}'
        if it["annotation"] is not None:
            lanes += f'<div class="ln">annotation</div>{_lane(it["annotation"], T)}'
        cells = "".join(f"<td>{f[k]:.0f}</td>" for k, _ in keys) if f else \
            f'<td colspan="{len(keys)}" class="na">no annotation file: cuts shown, no score</td>'
        rows.append(f'<tr><td class="nm"><a href="{html.escape(it["name"])}/model.html">{html.escape(it["name"])}</a>'
                    f'<div class="small">{task} · {len(it["segments"])} activities</div></td>{cells}'
                    f'<td class="lanes">{lanes}</td></tr>')
    if scored:
        mean = {k: float(np.mean([_flat(it["score"])[k] for it in scored])) for k, _ in keys}
        rows.append('<tr class="mean"><td>Mean</td>' + "".join(f"<td>{mean[k]:.1f}</td>" for k, _ in keys) + "<td></td></tr>")
    head = "".join(f"<th>{html.escape(h)}</th>" for _, h in keys)
    page = (RESULTS
            .replace("<title>Model results</title>", "<title>Final test</title>")
            .replace("<h1>Model results</h1>", "<h1>Final test: new videos</h1>"))
    a, b = page.index('<div class="sub">'), page.index("<h2>What the numbers mean</h2>")
    intro = (f'<div class="sub">The model was trained on {len(trained_on)} videos ({html.escape(", ".join(trained_on))}) '
             f'and then cut {len(items)} new video(s) it had never seen. Click a video to open its player and clips. '
             f'· <a href="results.html">Test on the training videos</a> · <a href="index.html">Annotation players</a></div>'
             f'<table><tr><th>Video</th>{head}<th>Timeline</th></tr>{"".join(rows)}</table>')
    page = page[:a] + intro + page[b:]
    page = page.replace(" td.lanes {", " td.na { text-align:left; color:var(--muted); }\n td.lanes {")
    out = os.path.join(OUTPUTS, "final_results.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(page)
    data = [{"name": it["name"], "task": it["task"], "task_probability": round(it["task_prob"], 3),
             "segments": it["segments"],
             "scores": {k: round(v, 2) for k, v in _flat(it["score"]).items()} if it["score"] else None,
             "per_activity": ({c: {k: round(v, 3) for k, v in d.items()} for c, d in it["score"]["per_class"].items()}
                              if it["score"] else None)} for it in items]
    with open(os.path.join(OUTPUTS, "final_results.json"), "w", encoding="utf-8") as fh:
        json.dump({"trained_on": trained_on, "videos": data}, fh, indent=2)
    md = ["# Final test: new videos", "",
          f"Model trained on {len(trained_on)} videos: {', '.join(trained_on)}.", "",
          "| Video | Task | Activities | " + " | ".join(h for _, h in keys) + " |",
          "|---|---|---|" + "---|" * len(keys)]
    for it in items:
        f = _flat(it["score"]) if it["score"] else None
        md.append(f"| {it['name']} | {it['task']} | {len(it['segments'])} | "
                  + (" | ".join(f"{f[k]:.1f}" for k, _ in keys) if f else " | ".join("-" for _ in keys)) + " |")
    with open(os.path.join(OUTPUTS, "final_results.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    if scored:
        print(f"\nMean over the {len(scored)} annotated new video(s): "
              + ", ".join(f"{h} {np.mean([_flat(it['score'])[k] for it in scored]):.1f}" for k, h in keys))
    print(f"Final test page: {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Find the activities in the videos and cut them.",
        epilog="examples:  python run_model.py   |   python run_model.py pc_assembly_01   |   "
               "python run_model.py C:\\videos\\my_video.mp4   |   python run_model.py final")
    ap.add_argument("what", nargs="?", default="all", metavar="VIDEO or COMMAND",
                    help="a video (its name, e.g. pc_assembly_01, or its path) or one of: " + ", ".join(COMMANDS)
                         + " (default: all)")
    ap.add_argument("video", nargs="?", help=argparse.SUPPRESS)   # for the older form: cut VIDEO
    ap.add_argument("--fresh", action="store_true", help="describe the frames again (features/)")
    ap.add_argument("--no-clips", action="store_true", help="do not write the per-activity clips")
    ap.add_argument("--no-open", action="store_true", help="do not open the page in the browser")
    ap.add_argument("--task", choices=["assembly", "disassembly"],
                    help="for one video: say whether it is an assembly or a disassembly, instead of letting the "
                         "model decide (it then only has to find the activities)")
    a = ap.parse_args()

    if a.what == "cut" or a.what not in COMMANDS:
        target = a.video if a.what == "cut" else a.what
        if not target:
            raise SystemExit("Give the video, e.g.  python run_model.py pc_assembly_01")
        if a.what != "cut" and a.video:
            raise SystemExit(f"Give one video at a time (got {a.what!r} and {a.video!r}).")
        page = one(find_video(target), a.fresh, not a.no_clips, a.task)
    else:
        if a.video:
            raise SystemExit(f"'{a.what}' takes no video. For one video:  python run_model.py {a.video}")
        names = annotated_videos()
        if not names:
            raise SystemExit(f"No videos with an annotation file in {VIDEOS}")
        page = None
        if a.what == "final":
            page = final(names, a.fresh, not a.no_clips)
        if a.what in ("all", "test"):
            page = test(names, a.fresh, not a.no_clips)
        if a.what in ("all", "train"):
            train(names, fresh=False)
    if page and not a.no_open and page.endswith(".html"):
        open_in_browser(page)
    return 0


if __name__ == "__main__":
    sys.exit(main())
