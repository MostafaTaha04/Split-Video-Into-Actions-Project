# Test commands — post-reorganisation verification

Run from the repository root with the venv active:

```bat
.venv\Scripts\activate
```

Paste the output of each block back and I will check it against the expected
values below. **Blocks 4 and 5 are the important ones** — they exercise
mediapipe / ultralytics, which are not installed in my sandbox, so they are the
only parts of the system I have never been able to run myself. Blocks 1–3 I have
already verified; run them anyway to confirm your environment agrees with mine.

---

## Block 0 — environment

The aspect-ratio bug and the opencv/numpy breakage both showed up here first.

```bat
python -c "import cv2, numpy, sklearn, mediapipe, scipy; print('cv2', cv2.__version__); print('numpy', numpy.__version__); print('sklearn', sklearn.__version__); print('mediapipe', mediapipe.__version__); print('scipy', scipy.__version__)"
python -c "import cv2; cv2.namedWindow('t'); cv2.destroyAllWindows(); print('OpenCV GUI OK')"
```

Expected: `cv2 4.11.0`, `numpy 1.26.4`, `sklearn 1.9.x`, `mediapipe 0.10.14`, `OpenCV GUI OK`.

> `numpy` **must** stay on 1.x and `cv2` on 4.x — mediapipe breaks on numpy 2.x.
> If either has moved, fix with `pip install -r requirements.txt`, never a bare
> `pip install opencv-python`.

---

## Block 1 — static checks and the clip registry

```bat
python -m ruff check .
python -m unittest discover -s tests
python clip_registry.py
```

Expected: `All checks passed!` · `Ran 58 tests ... OK (skipped=2)` · 6 clips listed,
`YT build B1` present.

This also proves the folder reorganisation worked: the registry now resolves
`results/` and `ground_truth/` even though `clips.json` still stores bare names.

---

## Block 2 — saved runs did not drift

```bat
python refresh_run_reports.py --dry-run
```

Expected: every row shows the **same number on both sides of the arrow**, e.g.
`results/results_coolingfan_v2run  0.981 -> 0.981   0.665 -> 0.665   0.600 -> 0.600`.

Any row where the two sides differ means a saved run no longer reproduces — tell me
immediately, that would be a real regression.

---

## Block 3 — the two headline evaluations

```bat
python evaluate_extended.py --src .
python learned_baseline.py --src .
```

Expected from `evaluate_extended.py`:

| Quantity | Value |
|---|---|
| sanity check | `re-segmentation reproduces saved boundaries: OK` |
| Cooling fan / CPU / RAM / Cable F1@1s | 1.000 / 0.667 / 0.353 / 0.133 |
| mean F1@1s | **0.538** |
| LOO clean mean | **0.500** |
| held-out Intel CPU install | **0.286** |

Expected from `learned_baseline.py`:

| Scorer | mean LOO F1@1.0s |
|---|:--:|
| hybrid (rule + logistic regression) | **0.516** |
| gradient boosting (raw) | 0.508 |
| rule-based fusion | 0.354 |

with `per_clip_wins_learned: 5`, `wilcoxon p=0.0625`.

> Gradient boosting is the one row that can legitimately move between scikit-learn
> releases. If **any other** row differs from the table, that is a real change and
> I need to see it.

---

## Block 4 — full vision pipeline *(verified 2026-08-16)*

This is the actual system running on video: mediapipe hand tracking, YOLO-World
detection, optical flow, scene detection. About 8 minutes.

Use **RAM**, not the cooling fan. Hand-detection rates differ enormously by
camera angle, and picking the wrong clip makes a healthy run look broken:

| Clip | hands detected | why |
|---|:--:|---|
| Cooling fan | **0.0%** | overhead framing — MediaPipe sees no hands at any resolution or threshold (`diagnose_hands.py`) |
| CPU | 1.5% | same overhead framing |
| Cable | 17.1% | |
| RAM | **31.8%** | good angle — use this one |
| Intel CPU | 42.2% | |

```bat
python main.py --video RAMinstallation.mp4 --output results\results_ram_TEST ^
  --fps 15 --detector open_vocab --model yolov8l-worldv2.pt ^
  --resize 960x540 --open-vocab-imgsz 960 --threshold 0.50 --min-duration 3.0 ^
  --grip-window 7 --object-confidence 0.10 ^
  --ground-truth ground_truth\ground_truth_raminstallation.json
```

Expected, all confirmed on 2026-08-16:

- `source is 1080x1920 (portrait); resize 960x540 re-oriented to 540x960`
- `Effective FPS: 15.0` and **896** frames (not 1792 — see the fps note below)
- **F1 Score: 0.353** at tolerance 1.0s, **0.588** at 3.0s

Then the hand check — this is what catches the aspect-ratio bug:

```bat
python -c "import csv;r=list(csv.DictReader(open('results/results_ram_TEST/features.csv')));n=sum(1 for x in r if float(x['hands_present'])>0);print(f'{n}/{len(r)} frames with hands = {100*n/len(r):.1f}%')"
```

Expected **~32%**. A 0.0% *on this clip* means the frame loader is squashing the
aspect ratio again and every hand feature is dead. (0.0% on the cooling fan is
normal and expected — see the table above.)

> **Two defects this block found on 2026-08-16, both now fixed.**
>
> 1. `main.py` passed `--video` straight to `VideoLoader` instead of going
>    through `clip_registry.resolve_video`, so `--video RAMinstallation.mp4`
>    failed from the repository root even though the file was in
>    `split-video-data/`. The README quick-start was broken for a fresh clone.
> 2. The frame stride used `int(original_fps / target_fps)`. Real footage is
>    **29.97** fps, so `--fps 15` computed `int(1.998) = 1` — no skipping at
>    all. The run silently processed at 29.97 fps: 1792 frames instead of 896,
>    18 minutes instead of 8, and F1 0.125 instead of 0.353. Now `round()`,
>    which reproduces every rate in `clips.json` including the 14.985 the two
>    29.97 fps clips record.
>
> Both are the same failure mode as the aspect-ratio bug: a plausible-looking
> wrong answer rather than a crash.

---

## Block 5 — hybrid scorer end-to-end *(I cannot test this)*

The hybrid is the scorer that beats the hand-designed fusion on all 5 clips, so
it needs to work from a clean start.

```bat
python train_boundary_model.py --src .
python -c "from boundary_model import BoundaryScorer, load_default_blend_weight as w; s=BoundaryScorer('boundary_model.joblib'); print(s.describe()); print('blend weight w_rule =', w('boundary_model.joblib'))"
```

Expected: `logistic_regression trained on 5 clips, 3396 frames (373 positive @ tol 0.5s)`
and `blend weight w_rule = 0.8`, with **no `InconsistentVersionWarning`**.

Then run the real pipeline with it:

```bat
python main.py --video Coolingfaninstallation.mp4 --output results\results_coolingfan_HYBRID ^
  --fps 10 --detector open_vocab --model yolov8l-worldv2.pt ^
  --resize 960x540 --open-vocab-imgsz 960 --threshold 0.50 --min-duration 3.0 ^
  --grip-window 7 --object-confidence 0.10 --scorer hybrid ^
  --ground-truth ground_truth\ground_truth_coolingfan_v2.json
```

Expected: runs clean and prints which scorer it used. Paste the summary.

---

## Block 6 — annotation prep for `YT build B1`

This is the highest-value remaining task. With 5 clips the Wilcoxon test's
*floor* is p = 0.0625, so the hybrid result **cannot** reach significance no
matter how large the effect. A 6th clip drops the floor to p = 0.031.

`YT build B1` is already registered and already has `features.csv` extracted —
it only needs its boundaries marked.

```bat
python annotate.py contact-sheet --video ytbuildB_01_seg01.mp4 --step 1.0
```

This writes a contact sheet of one frame per second. Read the approximate step
boundaries off it, refine any that are unclear:

```bat
python annotate.py contact-sheet --video ytbuildB_01_seg01.mp4 --around 12.5 --window 2.0 --step 0.1
```

Then write the times into `ground_truth\ground_truth_ytbuildb1.json` (copy the
shape from `ground_truth\ground_truth_coolingfan_v2.json`) and validate:

```bat
python annotate.py validate --ground-truth ground_truth\ground_truth_ytbuildb1.json --video ytbuildB_01_seg01.mp4
```

It must pass. Two things it deliberately rejects:

- an `annotator` field still containing `VERIFY`
- ≥60% of boundaries landing on a whole second — the signature of times typed
  from memory rather than read off frames. This is exactly what four of your
  existing ground-truth files fail on.

Once it passes, re-run Block 3 — the clip joins the evaluation automatically and
the Wilcoxon p should drop to 0.031.

---

## Cleanup

I can create and move files but not delete them, so the rejected material was
quarantined instead. Delete it yourself before submitting:

```bat
rmdir /s /q _scratch_delete_me
```

It holds the 194 rejected auto-labels, `_preview`, `_results_backup`,
`_hybrid_probe.py`, `_sweep.py`, and `hand_diagnostic.png`. It is gitignored, so
nothing there is committed. Also remove the two throwaway run folders from
Blocks 4–5 once checked:

```bat
rmdir /s /q results\results_coolingfan_TEST results\results_coolingfan_HYBRID
```
