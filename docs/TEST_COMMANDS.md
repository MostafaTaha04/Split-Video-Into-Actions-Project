# Test commands — full verification

Run everything from the repository root with the venv active:

```bat
.venv\Scripts\activate
```

(PowerShell: `.venv\Scripts\Activate.ps1`.) Every command below is a single line, so it
works the same in cmd and PowerShell.

Paste the output of each block back and it will be checked against the expected values.
Blocks 1–4 were verified on Linux before handing over. **Blocks 5–7 need the YOLO-World
detector, so they are the parts only your machine can run.**

---

## Everything at once

```bat
python check_everything.py
```

Runs Blocks 0-3, the ablation table and the 30-video smoke test (8a) in order, about
25 minutes, and saves the complete output to `check_everything_log.txt`; it ends with
a PASS/FAIL line per step. `--no-smoke` skips the smoke test (~3 minutes). The long
end-to-end check (8b, `python verify_all.py`) is separate. The blocks below explain
each step and its expected output.

---

## Block 0 — environment

```bat
python -c "import sys, cv2, numpy, sklearn, mediapipe, scipy, ultralytics; print(sys.version.split()[0], 'cv2', cv2.__version__, 'numpy', numpy.__version__, 'sklearn', sklearn.__version__, 'mediapipe', mediapipe.__version__, 'ultralytics', ultralytics.__version__)"
```

Expected: `3.11.9 cv2 4.11.0 numpy 1.26.4 sklearn 1.9.0 mediapipe 0.10.14 ultralytics 8.4.75`.
No new dependencies were added — the appearance scorer and the step report use only
OpenCV and NumPy.

---

## Block 1 — static checks, tests, registry

```bat
python -m ruff check .
python -m unittest discover -s tests
python clip_registry.py
```

Expected:

```
All checks passed!
...
Ran 111 tests in 1.5s
OK
30 clips: 4 dev, 26 held out; 3 clean, 27 edited; 5 hand-annotated, 25 AI-drafted + human-reviewed
  Cooling fan            edited  dev       10.000 fps  results_coolingfan_v2run
  CPU                    edited  dev       12.500 fps  results_cpu_final
  RAM                    edited  dev       14.985 fps  results_ram_final
  Cable                  edited  dev       14.985 fps  results_cable_final
  Intel CPU install      edited  heldout   10.000 fps  results_installintelcpu
  YT build A1            edited  heldout   10.000 fps  batch/ytbuildA_01_seg01  [AI draft, reviewed]
  ...                    (25 rows marked [AI draft, reviewed]: YT build A1-A3, B1-B8,
                          C1-C11, then input22/23/24 aug, the only "clean" clips)
```

The 25 AI-drafted, human-reviewed clips are held-out clips in `clips.json`, marked
`"annotation_source": "ai_draft_reviewed"`. `clips_ai_draft.json` is empty, so no
"+ N AI-draft clips" appears.

`OK` with **no `skipped=`** and **no warnings**. Before this upgrade the suite said
`OK (skipped=2)`: the two re-segmentation drift tests were silently skipping because they
looked for the results in the repository root after `results/` was reorganised.

If you see `InconsistentVersionWarning: Trying to unpickle estimator LogisticRegression
from version 1.7.2 when using version 1.9.0`, your `boundary_model.joblib` was trained
under an older scikit-learn (the file is not committed, so each machine trains its own).
Retrain it under the installed version and repeat:

```bat
python train_boundary_model.py --src .
python -m unittest discover -s tests
```

If you see `skipped=5`, the file is missing or cannot run at all — same fix.

---

## Block 2 — saved runs did not drift

```bat
python refresh_run_reports.py --dry-run
```

Expected — the same number on both sides of every arrow:

```
results/results_coolingfan_v2run    0.981 -> 0.981      0.665 -> 0.665    0.600 -> 0.600
results/results_cpu_final       0.694 -> 0.694      0.544 -> 0.544    0.400 -> 0.400
results/results_ram_final       0.528 -> 0.528      0.379 -> 0.379    0.273 -> 0.273
results/results_cable_final     0.568 -> 0.568      0.362 -> 0.362    0.154 -> 0.154
results/results_installintelcpu    0.811 -> 0.811      0.600 -> 0.600    0.526 -> 0.526
```

(These F1 values are each saved run at its *own, older* settings — that is why they
differ from the README, which replays every clip at the global configuration. Block 7
regenerates the runs at the current defaults.)

---

## Block 3 — offline evaluation (seconds, no video needed)

### 3a — headline numbers, both scorers

```bat
python evaluate_extended.py --src .
```

Expected (ends with this summary; the full JSON is in `extended_results.json`):

```
[sanity] re-segmentation reproduces saved boundaries: OK

================ SUMMARY (F1 @1.0 s) ================
rule scorer        global thr=0.65 min_dur=1.5
  per clip         Cooling fan 0.923  CPU 0.571  RAM 0.296  Cable 0.118
  dev mean 0.477 | LOO all 0.421 | held-out Intel CPU install 0.583 (+25 reviewed-draft clips: HELD-OUT SET below)
appearance scorer  global thr=0.75 min_dur=1.5
  per clip         Cooling fan 0.364  CPU 0.800  RAM 0.609  Cable 1.000
  dev mean 0.693 | LOO all 0.622 | held-out Intel CPU install 0.706 (+25 reviewed-draft clips: HELD-OUT SET below)
standard metrics, mean over all 30 clips (rule -> appearance)
  segF1@10        0.694 -> 0.715
  segF1@25        0.685 -> 0.707
  segF1@50        0.419 -> 0.578
  MoF_hungarian   0.572 -> 0.645
  gebd_f1@0.05    0.707 -> 0.706
  pooled F1 [95% CI] rule 0.466 [0.428, 0.504] -> appearance 0.589 [0.545, 0.633]
  hand-annotated clips only (5): rule 0.471 [0.275, 0.674] -> appearance 0.667 [0.528, 0.83]
edit-cut-only baseline, hand-annotated clips (5): 83% of boundaries on a cut | mean F1@1s 0.723 | pooled 0.706 [0.565, 0.872]
  recall@1s at cut / not at cut: rule 0.66/0.83  appearance 0.72/0.33
edit-cut-only baseline, all clips (30): 45% of boundaries on a cut | mean F1@1s 0.551 | pooled 0.585 [0.518, 0.648]
  recall@1s at cut / not at cut: rule 0.60/0.65  appearance 0.71/0.33
--- HELD-OUT SET: 26 clips (25 AI-drafted + human-reviewed, 1 hand-annotated), 405 boundaries, frozen config ---
  rule       mean F1@1s 0.467 @3s 0.676 | pooled 0.468 [0.433, 0.502] | segF1@50 0.424 | pred/gt 1.68
             by source video: Intel CPU install 0.583  YT build A 0.497  YT build B 0.497  YT build C 0.448  continuous recordings 0.387
             by footage: clean (3 clips) mean 0.387 pooled 0.386 [0.256, 0.462] pred/gt 2.26  |  edited (23 clips) mean 0.478 pooled 0.477 [0.443, 0.513] pred/gt 1.63
             by annotation: AI-drafted + human-reviewed (25) mean 0.463 pooled 0.465  |  hand-annotated (1) mean 0.583 pooled 0.583
  appearance mean F1@1s 0.580 @3s 0.656 | pooled 0.584 [0.539, 0.631] | segF1@50 0.548 | pred/gt 0.67
             by source video: Intel CPU install 0.706  YT build A 0.658  YT build B 0.610  YT build C 0.552  continuous recordings 0.485
             by footage: clean (3 clips) mean 0.485 pooled 0.480 [0.235, 0.75] pred/gt 0.43  |  edited (23 clips) mean 0.592 pooled 0.592 [0.548, 0.636] pred/gt 0.70
             by annotation: AI-drafted + human-reviewed (25) mean 0.575 pooled 0.581  |  hand-annotated (1) mean 0.706 pooled 0.706
  baselines  uniform (oracle K) 0.300 | random (rule's count) 0.382
  edit cuts  42% of boundaries sit on an edit cut (172/405); cut-only baseline mean F1@1s 0.530
             recall@1s at cut / not at cut: rule 0.60/0.64  appearance 0.70/0.33
=====================================================
[figures] written to ./figures/
```

The per-clip lines, LOO, the Intel held-out score and the "hand-annotated clips only" line
must be exactly as before the 25 clips were promoted: the development clips and the frozen
configuration did not change, and the learned scorers still use only the five hand-annotated
clips. The edit-cut and held-out-set blocks need `appearance.npz` and `edit_cuts.json` in each
clip's results folder; if a line says "missing", run `python appearance.py --src .` (~3 min)
and `python edit_cuts.py --src .` (~2 min) first.

Open `figures\scorer_comparison.png` (both scorers and the edit-cut-only baseline on the
hand-annotated clips), `figures\boundary_alignment_appearance.png` and
`figures\heldout_set.png`.

### 3a+ — all 30 annotations are valid

Command Prompt:

```bat
for %f in (ground_truth\ground_truth_*.json) do @echo %f & python annotate.py validate --ground-truth %f | findstr /C:"warning(s)"
```

PowerShell:

```powershell
Get-ChildItem ground_truth\ground_truth_*.json | ForEach-Object { "$($_.Name)  " + (python annotate.py validate --ground-truth $_.FullName | Select-String "warning\(s\)") }
```

Expected: `0 errors, 0 warning(s).` for all 30 files that `clips.json` uses (the five made
by hand and the 25 AI-drafted, human-reviewed ones). The one exception is
`ground_truth_template.json`, the blank template, which reports `1 error(s), 1 warning(s).`
on purpose (its annotator is the placeholder `manual`). Files in `ground_truth\_superseded\`
are not matched by this pattern.

### 3b — the appearance descriptors reproduce on Windows

The shipped `results\*\appearance.npz` files were computed on Linux. Recompute them on
your machine (needs the videos in `split-video-data\`, ~1 minute) and re-run 3a:

```bat
python appearance.py --src .
python evaluate_extended.py --src .
```

Expected: six `[ok]` lines, then **the identical summary**. If any appearance number moves
by more than ±0.01, paste both summaries — that would mean video decoding differs
between platforms enough to matter.

### 3c — every over-segmentation fix that was tried

```bat
python compare_scorer_variants.py --src .
```

Expected:

```
rule (current default)                        LOO(all) 0.421  held-out [0.583]  grid 48
rule + stricter prominence                    LOO(all) 0.550  held-out [0.0]  grid 192
rule, top-K step-length prior                 LOO(all) 0.522  held-out [0.286]  grid 18
rule + appearance merging                     LOO(all) 0.376  held-out [0.444]  grid 288
blend 0.5*rule + 0.5*appearance               LOO(all) 0.544  held-out [0.706]  grid 48
max(rule, appearance)                         LOO(all) 0.564  held-out [0.706]  grid 48
appearance, top-K step-length prior           LOO(all) 0.528  held-out [0.632]  grid 18
appearance (adopted as --scorer appearance)   LOO(all) 0.622  held-out [0.706]  grid 48
```

### 3d — learned scorers, now with the appearance row

```bat
python learned_baseline.py --src .
```

Expected table:

```
rule-based fusion (hand-designed)                   0.445
appearance change (training-free)                   0.639
hybrid_logistic_regression__raw                     0.444
mlp_neural_net__all                                 0.413
gradient_boosting__raw                              0.412
mlp_neural_net__raw                                 0.410
logistic_regression__all                            0.410
gradient_boosting__all                              0.410
logistic_regression__raw                            0.379
```

The two `gradient_boosting` rows are version-sensitive (README, "Version note"); every
other row must match exactly.

### 3e — assembly vs disassembly per step

```bat
python evaluate_direction.py --src .
```

The first run makes one 5 fps optical-flow pass per video (a few minutes in all) and
caches it as `direction_flow.npz` in each run folder; later runs take seconds. Expected:

```
  labelled steps: {'assembly': 129, 'disassembly': 120}
  single-cue AUC: {'flux': 0.647, 'centroid': 0.608, 'log_duration': 0.304}
  annotated steps, all             n=249  AUC 0.727  acc 0.675  confident: 48% of steps, acc 0.775
  annotated steps, hardware_only   n=187  AUC 0.673  acc 0.642  confident: 49% of steps, acc 0.728
  annotated steps, packaging_only  n= 62  AUC 0.848  acc 0.774  confident: 45% of steps, acc 0.929
  predicted segments          n=168  AUC 0.670  acc 0.655  confident: 54%, acc 0.667
  with --task = video type: acc 0.643 (with strong-evidence override: 0.631)
  video verdicts (auto): {'right': 7, 'wrong': 1, 'mixed_or_unclear': 22}
```

These numbers are weak, and reported as they are: the cue is right on about two steps in
three, and the two disassembly recordings (`input22 aug`, `input23 aug`) are *not*
recognised as disassembly. Optical flow can differ slightly between OpenCV builds, so the
last digit may move; the conclusion should not.

### 3f — turning vs moving step names

```bat
python evaluate_activity_names.py --src .
```

Expected (seconds, no video needed):

```
  annotated_steps: n=260, turning share 0.27
    quick-move None : right kind 0.231 | 'Rotating' 205 named, 0.28 right | 'Grip and move'   4 named, 0.75 right
    quick-move 5.0  : right kind 0.542 | 'Rotating' 101 named, 0.45 right | 'Grip and move' 109 named, 0.88 right
  predicted_segments: n=153, turning share 0.34
    quick-move None : right kind 0.281 | 'Rotating' 104 named, 0.38 right | 'Grip and move'   5 named, 0.80 right
    quick-move 5.0  : right kind 0.458 | 'Rotating'  61 named, 0.51 right | 'Grip and move'  48 named, 0.81 right
```

(The 3 s and 4 s rows sit in between.) 5.0 is the value in `activity_recognizer.py`.

---

## Block 4 — appearance scorer, live, with the step report (~2 minutes)

```bat
python main.py --video CPUplacement.mp4 --output results\TEST_cpu_appearance --scorer appearance --detector none --ground-truth ground_truth\ground_truth_cpuplacement.json
```

Expected in the console:

- `source is 1080x1920 (portrait); resize 960x540 re-oriented to 540x960`
- `Effective FPS: 12.5`
- `Detected 5 action segments with 4 boundaries`
- `Step report: results\TEST_cpu_appearance\step_report.html`
- in the evaluation block: `Precision: 1.000`, `Recall: 0.667`, `F1 Score: 0.800`,
  `Segmental F1@10/25/50: 0.833 / 0.833 / 0.833`, `Frame accuracy (Hungarian MoF): 0.762`

Then check the boundaries are exactly the offline replay's:

```bat
python -c "import json;print([b['timestamp'] for b in json.load(open('results/TEST_cpu_appearance/segmentation_results.json'))['boundaries']])"
```

Expected: `[5.12, 21.92, 30.4, 41.44]`.

Open `results\TEST_cpu_appearance\step_report.html` in a browser: 5 steps, three
keyframes each, and an "annotated step: …" line under every step.

---

## Block 5 — full pipeline, fresh run, one clip (~8 minutes)

This is the real system — MediaPipe, YOLO-World, optical flow, scene detection — and
checks that a fresh run reproduces the saved one. RAM is used because hands are visible
in it (~32% of frames), so it also exercises hand tracking.

```bat
python run_ablations.py --variants fresh --clips RAM --report
```

Expected at the end:

```
F1 @1.0 s (rule scorer, global configuration threshold 0.65, min-duration 1.5)
clip                          full       fresh
Cooling fan                  0.923         nan
CPU                          0.571         nan
RAM                          0.296       0.296
...
F1 @1.0 s (appearance scorer, threshold 0.75, min-duration 1.5)
clip                          full       fresh
...
RAM                          0.609       0.609
```

`fresh` must equal `full` for RAM in both tables. Then the hand check:

```bat
python -c "import csv;r=list(csv.DictReader(open('results/ablations/fresh/results_ram_final/features.csv')));n=sum(1 for x in r if float(x['hands_present'])>0);print(n,'/',len(r),'frames with hands')"
```

Expected about **32%** of 896 frames. While it runs, watch Python in Task Manager:
memory should stay well under 2 GB (it used to grow by ~4 GB per minute of video).
Open `results\ablations\fresh\results_ram_final\step_report.html` — the middle keyframes
should now show hand-position dots.

---

## Block 6 — module ablations on every clip (~1–1.5 hours, can run unattended)

```bat
python run_ablations.py
```

Runs `--no-flow` and `--no-scene` on all five human-annotated clips at the evaluated
configuration and prints a table. Expected:

```
F1 @1.0 s (rule scorer, global configuration threshold 0.65, min-duration 1.5)
clip                          full      noflow     noscene
Cooling fan                  0.923       0.000       0.923
CPU                          0.571       0.000       0.571
RAM                          0.296       0.480       0.296
Cable                        0.118       0.000       0.000
Intel CPU install            0.583       0.435       0.583
mean (all clips)             0.498       0.183       0.475
```

The appearance table below it shows `nan` in the `fresh` column for every clip except RAM
until Block 7 has been run; that is expected. Interrupted? Run it again — finished runs
are skipped. `python run_ablations.py --summarize` re-prints the tables without running
anything. The ablation uses only the five hand-annotated clips; the 25 AI-drafted, human-reviewed
held-out clips are deliberately not part of it.

---

## Block 7 — regenerate every saved run (optional, ~45 minutes)

```bat
python run_ablations.py --variants fresh --report
```

Expected: `fresh` equals `full` on every clip in both tables. If it does, the saved
reports in `results\` can be replaced by these consistent ones.

---

## Block 8 — every annotated video, end to end

`verify_all.py` runs the pipeline on all 30 clips in `clips.json` (5 hand-annotated, 25
AI-drafted and human-reviewed) with their ground truth. The `group` column says which:
`dev`, `held out` (hand-annotated) or `held/rev` (held out, AI-drafted and reviewed).
`--skip-ai` runs only the five hand-annotated clips.

**8a — smoke test (~20 minutes).** First 150 processed frames of every clip, detector off:
does every video open, every annotation load, and the pipeline finish?

```bat
python verify_all.py --quick
```

Expected: the pipeline output for each clip, then

```
QUICK SMOKE TEST (first 150 processed frames, detector off)
...                  (one row per clip, status OK)
30/30 clips OK
```

**8b — full check (~5-6 hours, can run overnight; interrupted runs resume).** Each clip
in full at the evaluated configuration, compared with its saved run:

```bat
python verify_all.py
```

Expected: `fresh` equals `saved` on every row, features `identical`, and

```
Cooling fan        dev             0.923  0.923      0.364  0.364   identical    MATCH
CPU                dev             0.571  0.571      0.800  0.800   identical    MATCH
RAM                dev             0.296  0.296      0.609  0.609   identical    MATCH
Cable              dev             0.118  0.118      1.000  1.000   identical    MATCH
Intel CPU install  held out        0.583  0.583      0.706  0.706   identical    MATCH
YT build A1        held/rev        0.489  0.489      0.800  0.800   identical    MATCH
...                (25 held/rev rows)
30/30 clips reproduce their saved run exactly
```

Run `python verify_all.py --only "Cooling fan"` first (~5 minutes) to see one finished
row before starting the long run. `python verify_all.py --summarize` prints the table at
any time. A `DIFFERENT` row is not automatically a bug — paste the table and the
pipeline log for that clip.

---

## After everything passes

The upgrade adds files that must be committed, including the appearance descriptors CI
needs for the appearance section of the evaluation:

```bat
git status
git add -A
git commit -m "Appearance scorer, standard metrics, step report, memory fix, evaluated defaults"
```

`results\ablations\` holds large `annotated_output.mp4` files, which `.gitignore` already
excludes; the CSV/JSON results there are small and worth committing with the ablation
table.
