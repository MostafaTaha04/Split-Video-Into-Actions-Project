# Splitting a Video into Actions

**Temporal segmentation of egocentric hardware-assembly video.**

This project automatically divides a first-person (egocentric) video of a hardware-assembly
procedure into its individual steps. The goal is **temporal segmentation** — detecting *where*
each step begins and ends — and not action recognition. A modular, interpretable pipeline fuses
hand tracking, open-vocabulary object detection, optical flow, scene-change detection, and
hand–object interaction into a single per-frame transition signal, and places step boundaries at
its prominent peaks.

> Project 6 · Students: **Osama Najjar, Mostafa Taha** · Mentor: **Saeed Namnah**

---

## Highlights

- **Modular feature pipeline** — hands, objects/regions, optical flow, scene change, and interaction.
- **Interpretable boundaries** — a step boundary is simply a moment where multiple cues agree the action changed; no per-frame labels or training required.
- **Rigorous evaluation** — boundary precision/recall/F1 at multiple tolerances, plus segment IoU and coverage, on 30 annotated clips: 5 annotated by hand, 25 drafted with an AI tool and checked by Mostafa Taha and Osama Najjar. Provenance is recorded per clip and every summary reports the two groups separately.
- **No test-set tuning** — one global configuration for all clips, validated by leave-one-clip-out cross-validation, plus a clip never used to tune the rule scorer (it was scored when choosing among appearance-scorer variants; see below).
- **Training-free appearance scorer** — `--scorer appearance` places boundaries where the scene *looks different before vs after* (`appearance.py`). It fixes the rule scorer's systematic over-segmentation: under leave-one-clip-out it scores 0.622 vs 0.421, and 0.706 vs 0.583 on the held-out clip. It is worse on one clip (cooling fan) and not statistically significant at five clips, so it is offered alongside the default rather than replacing it. See [Appearance scorer](#appearance-scorer-fixing-over-segmentation).
- **A 26-clip held-out set** — every remaining clip in `split-video-data/` (22 edited YouTube build clips, 3 continuous recordings) has an annotation **drafted with an AI tool from the video frames and then checked against the video by Mostafa Taha and Osama Najjar**, who changed no boundary. They joined the hand-annotated Intel clip as held-out clips: they never touch tuning and are scored at the frozen configuration. Appearance beats rule on 22 of 26 (pooled F1@1 s 0.584 vs 0.468, Wilcoxon p = 0.0004). See [The held-out set](#the-held-out-set-26-clips).
- **An edit-cut-only baseline** — most clips, including three registered as single-shot, contain editor's cuts, and a cut often falls on a step change. `edit_cuts.py`, which only compares consecutive frames, scores as well as the appearance scorer (pooled 0.585 vs 0.589 on all 30 clips; 0.706 vs 0.667 on the hand-annotated five). Away from cuts the rule scorer is the better detector. See [Edit cuts](#a-baseline-every-scorer-must-beat-edit-cuts).
- **Standard metrics** — besides boundary F1, every run and the extended evaluation report segmental F1@{10,25,50}, Hungarian frame accuracy (MoF) and GEBD Rel.Dis F1, plus pooled F1 with a bootstrap CI.
- **Step report** — every run writes `step_report.html`: per-step keyframes, where the hands were during each step, and the matching annotated step.
- **Rule-based, learned, or hybrid scoring** — `--scorer rule|learned|hybrid`. A classifier trained on the pipeline's own per-frame features can replace or blend with the hand-designed fusion. Measured under leave-one-clip-out, no learned variant beats the hand-designed rules; the apparent win reported against the earlier, unvalidated annotations did not survive their correction.
- **Baselines & ablations** — on the cooling-fan and CPU clips the rule scorer beats uniform, random, and a `ruptures` change-point baseline at tight tolerances; on the two edited clips random placement and change-point (both given the true number of steps) do better. Module and cue ablations identify what actually carries the signal, and find measured evidence against two of the five feature extractors.
- **Annotation tooling** — a documented protocol, a frame-accurate annotation tool, a ground-truth validator, and an inter-annotator agreement metric (`annotate.py`).
- **Readable outputs** — annotated video, timeline plot, per-step clips, and a feature CSV per run.

---

## Results

All numbers below use a **single global configuration** (boundary threshold 0.65, minimum segment
duration 1.5 s) applied unchanged to every clip. No per-clip tuning.

| Video | Type | F1 @1.0s | F1 @3.0s | In tuning set? |
|---|---|:--:|:--:|:--:|
| Cooling fan | Single scene, jump cuts | **0.923** | 0.923 | yes |
| CPU placement | Jump cuts, product shots | **0.571** | 0.571 | yes |
| RAM install | Edited tutorial | 0.296 | 0.519 | yes |
| Cable connect | Edited tutorial | 0.118 | 0.235 | yes |
| Intel CPU install | Body-mounted, jump cuts | **0.583** | 0.833 | **no — held out** |

The cooling fan, CPU and Intel clips were first registered as `clean` (one continuous shot).
Checked frame by frame, all three contain jump cuts, often exactly at a step change, so they
were relabelled `edited` in `clips.json`. No development clip is a continuous
recording any more; the only continuous clips are three held-out recordings. See
[Edit cuts](#a-baseline-every-scorer-must-beat-edit-cuts).

**Leave-one-clip-out** over the four dev clips gives a mean of **0.421**; on the cooling fan and
CPU clips alone it is **0.684** — note that the latter averages only two numbers. (Earlier
versions called this the "clean" LOO mean; with no continuous development clip,
`evaluate_extended.py` no longer prints one.) The held-out clip, which took no part in tuning the rule scorer, scores **0.583** at
1.0 s and 0.833 at 3.0 s, with 8 of its 10 annotated boundaries matched within 1.2 s. These
cross-validated and held-out figures are the ones to trust; the per-clip table above shares two
hyper-parameters with the clips it scores. Leave-one-out tunes only those two parameters — the
~40 feature thresholds upstream were set by hand while developing on the dev clips, so even the
leave-one-out numbers are somewhat optimistic for the rule scorer.

> **All numbers were regenerated after the ground truth was corrected.** Four of the five annotation
> files had boundaries snapped to whole seconds — the signature of times estimated rather than read
> off frames — and all four failed this project's own `annotate.py validate`. Re-annotating them
> moved every result, in both directions: the per-clip mean fell from 0.538 to 0.477, while the two
> uncontaminated estimates rose sharply (leave-one-out on the cooling fan and CPU 0.500 → 0.684,
> held-out 0.286 → 0.583). The
> earlier numbers were measured against labels that could not be defended. See
> `docs/Final_Report.docx` §7.1.
>
> An earlier frame-loader defect was also fixed: portrait clips were squashed from aspect 0.56 to
> 1.78, silently zeroing every hand-derived feature on four of five clips. See §7.6.

On the single-scene clips (cooling fan, CPU, Intel) the system localises matched boundaries to within
~0.1–0.4 s. Two known poor cases are analysed rather than hidden:

- **Edited tutorial footage** (RAM, Cable) — presenter cut-aways and diagram slides fire the
  scene-change cue at moments that are not assembly steps. See `docs/Final_Report.docx` §7.2.
- **Cable connect** remains the weakest clip (F1 0.118 at 1.0 s, 0.235 at 3.0 s). Its four annotated
  steps are long — 12 to 15 s each — while the system proposes a boundary every few seconds, so
  precision collapses. The ablation below identifies the object detector as a large part of the
  cause: disabling it raises this clip from 0.118 to **0.400**, the biggest single-clip effect
  measured anywhere in the project. Analysed in `docs/Final_Report.docx` §7.2 and §7.3.

### Appearance scorer: fixing over-segmentation

**Diagnosis.** The rule scorer's boundary score is a maximum over ~15 binary cues. On four of the
five clips at least one cue fires on **73–98% of frames**, and the smoothed score stays above the
threshold for 30–63% of the video. Peak-picking inside that plateau places a boundary roughly
every `min_duration` seconds, so the threshold barely matters (RAM yields 19 boundaries at
threshold 0.32, 0.50 and 0.65 alike) and every clip but one is over-segmented: 15 predicted vs 6
annotated on CPU, 19 vs 8 on RAM, 13 vs 4 on Cable, 14 vs 10 on the held-out clip. The one clip
with a sparse score — the cooling fan — is the one that scores 0.923. The score measures *"is
something happening?"*; a step boundary is better described as *"the scene now looks different
from how it looked a moment ago"*.

**Method** (`appearance.py`, training-free, OpenCV only). Each processed frame is reduced to a
compact descriptor — a 32-px grayscale thumbnail (layout) plus a coarse HSV histogram (what is
visible). The boundary score at time *t* is the cosine distance between the mean descriptor of the
2 s before *t* and the 2 s after it. It is high when the before- and after-states differ, and low
both during steady work and during motion that does not change the scene. Smoothing, peak
picking and every other post-processing step are shared with the rule scorer. The idea is the
same as training-free action-boundary detection on deep features (ABD, Du et al., CVPR 2022),
with hand-crafted descriptors so it needs no model download.

**Results** — same protocol as the rule scorer: one global (threshold, min-duration) chosen on the
dev clips over the *same* grid (0.75 / 1.5), leave-one-clip-out, held-out clip at the frozen
configuration.

| Clip | Rule F1@1s | Appearance F1@1s | Boundaries pred / annotated (rule → appearance) |
|---|:--:|:--:|:--:|
| Cooling fan | **0.923** | 0.364 | 6 → 4 / 7 |
| CPU | 0.571 | **0.800** | 15 → 4 / 6 |
| RAM | 0.296 | **0.609** | 19 → 15 / 8 |
| Cable | 0.118 | **1.000** | 13 → 4 / 4 |
| Intel CPU install (held out) | 0.583 | **0.706** | 14 → 7 / 10 |
| **LOO mean, 4 dev clips** | 0.421 | **0.622** | |
| **LOO mean, all 5 clips** (`learned_baseline.py` protocol) | 0.445 | **0.639** | |

| Standard metric (mean, all 5 clips) | Rule | Appearance |
|---|:--:|:--:|
| Segmental F1@10 / @25 / @50 | 0.693 / 0.679 / 0.447 | **0.796 / 0.796 / 0.743** |
| Frame accuracy (Hungarian MoF) | 0.597 | **0.738** |
| GEBD F1 @Rel.Dis 0.05 | 0.616 | **0.732** |
| Pooled boundary F1 @1s [95% bootstrap CI over clips] | 0.471 [0.28, 0.67] | **0.667 [0.53, 0.83]** |

`figures/scorer_comparison.png` and `figures/boundary_alignment_appearance.png` show the same
comparison visually.

**Caveats — read before quoting these numbers.**

- **It is not uniformly better.** It loses badly on the cooling fan (0.923 → 0.364): the clip's
  mechanical sub-steps (adjusting, attaching and locking a clip) barely change the scene's
  appearance, which is exactly what the rule scorer's motion cues *do* see.
- **Not statistically significant on the five hand-annotated clips.** Better on 4 of 5 clips, but
  because the one loss is large, a Wilcoxon test gives p = 0.44 (and five clips cannot go below
  p = 0.0625 in any case). On the 26-clip held-out set it is better on 22 of 26 (p = 0.0004), but
  25 of those annotations were AI-drafted; see [The held-out set](#the-held-out-set-26-clips).
- **A detector of editor's cuts does as well.** On the hand-annotated clips 29 of 35 boundaries
  sit on an edit cut, and the edit-cut-only baseline scores 0.706 pooled against 0.667 for this
  scorer. See [Edit cuts](#a-baseline-every-scorer-must-beat-edit-cuts).
- **It was selected among eight variants.** Blends of the two scores, top-K selection with a
  step-length prior, appearance-based merging of rule boundaries and a stricter peak prominence
  were all tried; all generalise worse under leave-one-clip-out (`compare_scorer_variants.py`,
  `scorer_variants.json`). The held-out clip was scored for every variant, so it is no longer an
  untouched test for this choice — the leave-one-clip-out columns are the fair comparison.
- **It ignores hands and tools.** The project's methodology tracks hand–tool–device interaction;
  this scorer does not. That is why it is offered as `--scorer appearance` while the rule scorer
  remains the default — the choice of headline method is the authors'. That a training-free
  appearance signal outperforms the interaction cues *on this footage* is itself a finding: on
  most clips hands are rarely detected (see Limitations), so the interaction cues have little to
  work with.

```bash
python appearance.py --src .                     # descriptors for the saved runs (~40 s)
python evaluate_extended.py --src .              # both scorers, standard metrics, figures
python compare_scorer_variants.py --src .        # every variant tried, same protocol
python main.py --video CPUplacement.mp4 --scorer appearance --detector none   # live
```

`--detector none` is fine with the appearance scorer: it does not use detections, and it is ~4×
faster.

### The held-out set (26 clips)

`split-video-data/` held 25 clips without ground truth, used only for unannotated batch
statistics:

* **22 edited clips** cut from three YouTube PC-build videos (A1–A3, B1–B8, C1–C11);
* **3 continuous recordings** (`input22/23/24_aug_01`, single shot, head-mounted, a desktop
  being taken apart and reassembled), the kind of footage the project targets.

**How they were annotated.** These annotations were not made by hand, and the files say so.
Each one was **drafted with an AI tool** from the video frames, at the granularity of
`docs/ANNOTATION_PROTOCOL.md` §1: a 1 s contact sheet over every clip, and a new step whenever
the hands release one item and reach for another, change the kind of manipulation, or pick up
or put down a tool. The system's predictions were never consulted. A first, coarse version
(147 boundaries, one step every 15.5 s, kept in `ground_truth/_superseded/`) was redone at the
protocol's granularity (395 boundaries, one step every 6.3 s; the five hand-annotated clips
average 6.6 s). **Mostafa Taha and Osama Najjar then checked version 2 against the videos
and changed no boundary.**

Every file records this: `"annotator": "Mostafa Taha, Osama Najjar"`,
`"annotation": "ai_draft"`, `"review_status": "human_reviewed"`, the check in
`annotation_method` and the drafting procedure in `draft_method`. In `clips.json` each of the 25
carries `"annotation_source": "ai_draft_reviewed"`; clips without that field were annotated by
hand. `promote_clip.py` moved them from `clips_ai_draft.json` (now empty) into `clips.json` as
**held-out** clips. The development clips and the frozen configuration did not change, so no
choice of threshold, duration or method saw these clips.

**Results** — 26 held-out clips (25 AI-drafted and reviewed, plus the hand-annotated Intel clip),
405 boundaries, each scorer at the configuration chosen on the development clips. Pooled F1 @1 s
with a 95% bootstrap CI over clips:

| | Rule-based fusion (0.65 / 1.5) | Appearance change (0.75 / 1.5) | Edit cuts only (`edit_cuts.py`) |
|---|:--:|:--:|:--:|
| **All 26 held-out clips** | 0.468 [0.433, 0.502] | **0.584 [0.539, 0.631]** | 0.578 [0.504, 0.644] |
| Edited, 23 clips (22 YouTube + Intel) | 0.477 [0.443, 0.513] | 0.592 [0.548, 0.636] | **0.610 [0.545, 0.670]** |
| Continuous recordings, 3 clips | 0.386 [0.256, 0.462] | **0.480 [0.235, 0.750]** | 0.056 [0.000, 0.125] |
| AI-drafted + reviewed, 25 clips | 0.465 | **0.581** | 0.568 |
| Hand-annotated, 1 clip (Intel) | 0.583 | 0.706 | **0.857** |
| Predicted / annotated boundaries | 1.68 (too many) | 0.67 (too few) | — |

Uniform placement with the true number of steps scores 0.300 (mean), random placement with the
rule scorer's count 0.382. Over all 26, segmental F1@50 is 0.424 (rule) and 0.548 (appearance).
`figures/heldout_set.png` plots every clip with both baselines.

What this supports, and what it does not:

* **The appearance scorer beats the rule scorer on unseen clips.** Better on 22 of 26, Wilcoxon
  p = 0.0004, at the configuration chosen on the development clips. On the five hand-annotated
  clips the same comparison was not significant (p = 0.44).
* **It does not beat a detector of editor's cuts.** Against the edit-cut-only baseline it wins
  14 clips and loses 11 (p = 0.70). See [Edit cuts](#a-baseline-every-scorer-must-beat-edit-cuts).
* **The two scorers fail in opposite directions.** The rule scorer over-segments (1.7× too many
  boundaries); the appearance scorer under-segments (0.67×): it finds the large visual changes
  and misses short steps within one view.
* **The continuous recordings are the closest to the target task.** On the three of them (35
  boundaries) both scorers beat random and uniform placement (appearance 0.480, rule 0.386,
  uniform 0.244), and there the cut baseline has almost nothing to find (0.056: one cut in three
  clips). With 35 boundaries the interval is wide.
* **The numbers depend on how finely a step is defined.** Against the coarse version-1 drafts,
  the rule scorer got 0.229 and the appearance scorer 0.513. The ranking held; the absolute
  numbers mean little without the protocol's definition of a step.
* **Quote the provenance with the numbers.** These are AI-drafted annotations reviewed by people,
  not annotations made by hand. Every summary keeps the hand-annotated clips apart ("hand-annotated
  clips only" in the summary, `by_source` in `extended_results.json`), and the learned-scorer
  experiments still train and test on the five hand-annotated clips only.

```bash
python appearance.py --src .        # descriptors for every registered run
python edit_cuts.py --src .         # hard edit cuts per clip (video frames only)
python evaluate_extended.py --src . # prints the HELD-OUT SET block at the end of the summary
```

The review and promotion procedure for further clips is in `docs/ANNOTATION_PROTOCOL.md` §5.

### A baseline every scorer must beat: edit cuts

`edit_cuts.py` marks a hard cut where the picture changes abruptly between two consecutive
frames (64×36 grayscale difference, at least 4× the local median, cuts at least 1.5 s apart).
It has no model and no parameter tuned on annotations. Used as a segmenter on its own, it is a
baseline that any scorer must beat, and on this corpus it is a strong one:

| Pooled F1 @1 s [95% CI] | Rule | Appearance | Edit cuts only | Boundaries on a cut |
|---|:--:|:--:|:--:|:--:|
| Hand-annotated, 5 clips | 0.471 [0.275, 0.674] | 0.667 [0.528, 0.830] | **0.706 [0.565, 0.872]** | 29 of 35 (83%) |
| All 30 clips | 0.466 [0.428, 0.504] | **0.589 [0.545, 0.633]** | 0.585 [0.518, 0.648] | 192 of 430 (45%) |

Per hand-annotated clip, edit cuts only: Cooling fan 0.462, CPU 0.923, RAM 0.571, Cable 0.800,
Intel 0.857 (`figures/scorer_comparison.png`).

* **Three "clean" clips were not single shots.** Checked frame by frame, the cooling fan, CPU and
  Intel clips contain jump cuts, often exactly where one step ends and the next begins, and the
  CPU clip has product shots cut in at 21.8–30.5 s. They are now labelled `edited`. Only the
  three `input2x_aug` recordings are free of cuts: the one "cut" detected among them (input22 at
  81.0 s) is a hand passing in front of the camera.
* **The appearance scorer is, in large part, a cut detector.** It finds 71% of the boundaries
  that sit on a cut and 33% of the others (all 30 clips). The rule scorer finds 60% and 65%:
  **away from cuts, the hand-designed motion cues are the better detector** (on the hand-annotated
  clips it finds 5 of the 6 boundaries not on a cut; the appearance scorer finds 2).
* **What this means for the results.** Much of every score on this corpus comes from editing,
  not from understanding the assembly. A scorer should be judged by how far it gets above the cut
  baseline, and the strongest test is footage with no cuts at all (the three continuous
  recordings). Combining a cut detector with the rule scorer's motion cues is the obvious next
  step; it was not tried, because choosing it after seeing these numbers would spend the held-out
  set.

Both analyses are in `extended_results.json` (`edit_cuts.hand_annotated`, `edit_cuts.all_clips`,
`heldout_set.edit_cuts`) and in the summary printed by `evaluate_extended.py`.

### Ablation: the object detector hurts more than it helps

Watching the annotated output showed YOLO-World labelling the work surface as `cpu socket`. Measured,
it reports at least one component in **100% of frames** on every clip, averaging 3.5–4.0 detections
per frame, with its visible-object set changing on 13–26% of frames. The boundary logic consumes
*changes* in that set, so the churn manufactures spurious transitions.

Disabling it (`--detector none`) at the same global configuration:

| Clip | detector on | detector off |
|---|:--:|:--:|
| Cooling fan | **0.923** | 0.833 |
| CPU placement | 0.571 | **0.632** |
| RAM install | 0.296 | **0.370** |
| Cable connect | 0.118 | **0.400** |
| Intel CPU install (held out) | 0.583 | **0.636** |
| Dev mean | 0.477 | **0.559** |
| LOO mean, cooling fan + CPU | 0.684 | **0.769** |
| Speed | 1.7 it/s | **7.1 it/s** |

Better on four of five clips and 4.2× faster. It is nonetheless **not** adopted as the headline
configuration: across the full config grid the held-out clip averages 0.427 with the detector and
0.392 without, so removal raises the ceiling but lowers the average, and a configuration selected
honestly on the development clips without the detector scores *worse* on the held-out clip (0.471 vs
0.583). With five clips the protocols disagree and the evidence is not conclusive. Restructuring the
system on that would repeat the error documented in §7.1. Reported as an ablation and flagged for
future work — see `docs/Final_Report.docx` §7.3.

`figures/boundary_alignment.png` shows predicted versus annotated boundaries for all five clips on
a shared time axis. Full analysis: `docs/Final_Report.docx`, `docs/Extended_Evaluation.docx`.

### Ablation: optical flow carries the signal, scene detection adds nothing

`run_ablations.py` re-runs the full pipeline on all five clips with one module switched off
(`--no-flow`, `--no-scene`), at the global configuration (rule scorer, 0.65 / 1.5):

| Clip | full | no optical flow | no scene detection |
|---|:--:|:--:|:--:|
| Cooling fan | 0.923 | 0.000 (0 boundaries) | 0.923 |
| CPU placement | 0.571 | 0.000 (2 boundaries) | 0.571 |
| RAM install | 0.296 | **0.480** | 0.296 |
| Cable connect | 0.118 | 0.000 | 0.000 |
| Intel CPU install (held out) | 0.583 | 0.435 | 0.583 |
| Mean | **0.498** | 0.183 | 0.475 |

* **Optical flow is the main signal.** Without it the mean falls from 0.498 to 0.183, and on the
  two overhead clips the score almost never crosses the threshold (0 and 2 boundaries), because
  hands are rarely detected there and the remaining cues are nearly flat. RAM, an edited clip, is
  the exception: it improves without flow.
* **Scene-change detection contributes nothing measurable.** Without it every clip has the same
  number of boundaries; on four clips at most one boundary moves by 0.1 s, and the only score that
  changes is Cable, whose single correct boundary (15.68 s against 15.7 s) moves to 14.55 s. An
  earlier single-clip run, at older settings, had suggested that removing it *improved* the
  cooling fan to 1.000; at the current configuration that does not reproduce.
* Both checks were confirmed on the saved features: `flow_magnitude` is 0 in every no-flow run and
  `scene_change_score` is 0 in every no-scene run, and replaying each run's features gives the
  same F1 as the run itself reported. Full results: `ablation_results.json`.

### Learned and hybrid scorers

Supervised alternatives were trained on the pipeline's own per-frame features (3,396 labelled
frames, no new annotation) and compared under leave-one-clip-out with **identical** post-processing:

| Scorer | Mean F1 @1.0s |
|---|:--:|
| Appearance change (training-free, `--scorer appearance`) | **0.639** |
| Rule-based fusion (hand-designed) | **0.445** |
| Hybrid (rule + logistic regression) | **0.444** |
| Neural network (all features) | 0.413 |
| Gradient boosting (raw features) | 0.412 |
| Neural network (raw features) | 0.410 |
| Logistic regression (all features) | 0.410 |
| Gradient boosting (all features) | 0.410 |
| Logistic regression (raw features) | 0.379 |

**No learned scorer beats the hand-designed fusion** (the training-free appearance scorer, added
later and discussed [above](#appearance-scorer-fixing-over-segmentation), does on average). The best of them, the hybrid, is behind by
0.001 — three clips to two, Wilcoxon p = 1.0. On this data, supervised learning over these features
matches careful hand-written rules and does not exceed them.

That conclusion is the *opposite* of what this project reported before the ground truth was
corrected, and the reversal is the most interesting result here. Against the old annotations the
hybrid appeared to win decisively: 0.516 against 0.354, five clips from five. Those annotations had
boundaries snapped to whole seconds. The learned models fitted that artificial regularity; the
hand-written rules, which encode motion physics rather than label statistics, did not. Once the
labels described what actually happens on screen, the apparent advantage vanished entirely.

The practical warning generalises beyond this project: **a learned model can win a benchmark by
exploiting structure in the labels rather than in the world**, and the only way to catch it is to
verify the labels. Here the measured effect was large — a 0.16 margin with a perfect win record,
reduced to nothing.

Two secondary observations survive the correction. Model capacity still does not pay: every learned
variant lands between 0.379 and 0.413, a 0.034 spread across logistic regression, gradient boosting
and a neural network — model choice is worth almost nothing here, the expected signature of 5 clips
and 3,396 frames of which only 441 are boundary frames. And the learned scorers remain *more consistent* across clips even while
scoring no better (std 0.156 against 0.289) — they are weaker on the cooling-fan and CPU clips (0.600 against
0.661) and slightly stronger on the edited tutorials (0.341 against 0.300), which is what a model
fitted to a mixed training set would be expected to do.

> **Version note.** `scikit-learn` is pinned to `>=1.9,<1.10` because gradient boosting is not
> reproducible across releases. Measured on identical code and data, its two rows moved by 0.009
> and 0.037 between 1.7.2 and 1.9.0 — enough to reorder the middle of the table — while logistic
> regression, the neural network, the hybrid and the rule-based scorer were bit-identical. This was
> verified by running the same script under both versions rather than assumed. No conclusion depends
> on it: gradient boosting is neither the best nor the worst method under either version, and the
> headline finding concerns the rule-based and hybrid scorers, which do not move.

```bash
pip install "scikit-learn>=1.9,<1.10"
python learned_baseline.py --src .        # held-out comparison above
python train_boundary_model.py --src .    # writes boundary_model.joblib
python main.py --video clip.mp4 --scorer hybrid ...
```

The default is `--scorer rule`, which is both the best-measured option and the one that works from
a fresh clone with no training step. `--scorer hybrid` is retained because it is the strongest
learned variant and the comparison is a reported result, not because it performs better.

`boundary_model.joblib` is **not** committed — a pickled scikit-learn estimator is tied to the
version that created it, and loading it under a different version warns and may change results.
Run `train_boundary_model.py` once after cloning; it takes seconds. The JSON sidecar recording the
blend weight and provenance *is* committed.

### Assembly vs disassembly (experimental)

The segmentation is direction-blind by design: both scorers look for moments of change, and
pulling a RAM module out changes the scene just as much as pushing one in. The step *names*
were not: every rule in `activity_recognizer.py` was worded for assembly, so a disassembly
video got names like "Press RAM module into slot". Two changes fix that:

- **Names are direction-neutral unless the direction is known** ("RAM module handling",
  "Rotating / twisting motion"). With `--task assembly` or `--task disassembly` every step is
  named for that direction ("Press RAM module into slot" / "Pull RAM module out of slot").
  This is the reliable route when you know what the video shows.
- **In `--task auto` (default) each step gets a direction from motion** (`action_direction.py`):
  a separate 5 fps optical-flow pass, camera motion removed, then three numbers per step —
  net outward motion (the hands go in and out, the part goes one way only), where in the step
  the motion is concentrated (bring-then-align vs. unlatch-then-pull), and duration (removing
  is usually quicker) — combined by a three-weight logistic regression. Steps it is unsure of
  are marked `unclear`. The pass never moves a boundary.

Measured with `evaluate_direction.py`, leave-one-source-out, labels taken from the annotated
step names (249 put-in / take-out steps; taking things out of packaging counts as removing):

| | Steps | Right | Confident share | Right when confident |
|---|:--:|:--:|:--:|:--:|
| Annotated steps (true boundaries) | 249 | 67.5% | 48% | 77.5% |
| — hardware only (no packaging) | 187 | 64.2% | 49% | 72.8% |
| Segments the system finds (appearance scorer) | 168 | 65.5% | 54% | 66.7% |

Per video it is usually "mixed or unclear" (22 of 30), and it did **not** recognise the two
disassembly recordings (`input22 aug`, `input23 aug`) as disassembly; one assembly clip (RAM)
came out "mostly disassembly". So the automatic direction is a weak hint, shown with its
evidence (`direction_p_remove` per step), not a recognised action. Recognising "putting in" vs
"taking out" reliably would need a video model trained on such actions (e.g. one trained on
Something-Something v2, which has exactly these classes), or a disassembly test set such as
EgoProceL's PC Disassembly videos (`import_segments.py` reads its annotations).

---

## Installation

Requires **Python 3.11+** (scikit-learn 1.9, pinned for reproducibility, needs 3.11).

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt          # or: pip install -e .
```

On first run, the YOLO-World weights (`yolov8*-worldv2.pt`) and the MediaPipe
`hand_landmarker.task` model download automatically. Neither is stored in the repository.

---

## Quick start

Segment a clip and evaluate it against ground truth. The command-line defaults *are* the
evaluated configuration (threshold 0.65, min-duration 1.5, 10 fps, YOLO-World `yolov8l-worldv2`
at 960 px, 960x540 frames, grip window 7, object confidence 0.10), so no tuning flags are needed:

```bash
python main.py --video Coolingfaninstallation.mp4 --output results/results_coolingfan \
  --ground-truth ground_truth/ground_truth_coolingfan_v2.json

# appearance scorer (its own evaluated defaults: threshold 0.75, min-duration 1.5)
python main.py --video Coolingfaninstallation.mp4 --output results/coolingfan_appearance \
  --scorer appearance --detector none --ground-truth ground_truth/ground_truth_coolingfan_v2.json
```

`tests/test_appearance.py` fails if these defaults ever drift from the values in
`extended_results.json`. (Before this was enforced the defaults — threshold 0.32, 15 fps,
`workspace` detector, 640x480 — described a configuration that was never evaluated.)

Each run writes to the `--output` folder:

- `segmentation_results.json` — segments and boundaries with timestamps
- `timeline.png` — boundaries over the activity signal
- `annotated_output.mp4` — video with hand/region overlays
- `clips/step_*.mp4` — one clip per detected step
- `features.csv` — per-frame feature values
- `appearance.npz` — per-frame appearance descriptors (lets any run be re-scored offline)
- `interaction_points.json` — palm positions per step (where the hands worked)
- `step_report.html` — self-contained step-by-step breakdown with keyframes (skip with `--no-report`)
- `direction_flow.npz` — coarse optical flow for the assembly/disassembly pass; each segment in
  `segmentation_results.json` carries `direction`, `direction_p_remove` and `direction_source`
- `evaluation_report.txt` — metrics incl. standard metrics (when `--ground-truth` is given)

Memory use is flat in video length (~0.4 GB for a 52 s clip): frames and optical-flow fields are
no longer held in memory. Previously every processed frame kept its image and dense flow field
(~5.7 MB, ≈4 GB per minute of video), which was OOM-killed on a 52 s clip on a 4 GB machine.

---

## Key command-line options

| Option | Description |
|---|---|
| `--video` | Input video path (required). |
| `--output` | Output directory. |
| `--detector` | `open_vocab` (default), `workspace`, `yolo`, `hybrid`, or `none`. |
| `--model` | Detector weights (default `yolov8l-worldv2.pt`, or a trained `hardware_model.pt`). |
| `--fps` | Processing rate (default 10). |
| `--resize` | Processing resolution (default `960x540`; portrait video is re-oriented). |
| `--threshold` | Boundary-detection sensitivity (higher = fewer boundaries). Default: the evaluated value for the scorer (rule 0.65, appearance 0.75). |
| `--min-duration` | Minimum segment length in seconds. Default: 1.5 for both scorers. |
| `--ground-truth` | Ground-truth JSON for evaluation. |
| `--scorer` | `rule` (default), `appearance`, `learned`, or `hybrid` boundary scoring. |
| `--no-report` | Skip `step_report.html`. |
| `--task` | `auto` (default: assembly/disassembly per step from motion, uncertain steps `unclear`), `assembly` or `disassembly` (name every step for that direction). |
| `--no-direction` | Skip the assembly/disassembly pass; names stay direction-neutral. |
| `--boundary-model` | Trained model for `learned`/`hybrid` (default `boundary_model.joblib`). |
| `--blend-weight` | Rule-side weight for `hybrid`; defaults to the cross-validated value. |
| `--no-flow` / `--no-scene` | Disable a feature (used for the ablation study). |
| `--debug-detections` | Print raw detector candidates per frame. |
| `--max-frames` | Process only the first N frames (fast debugging). |

---

## Ground-truth annotation

Evaluation is only as trustworthy as the reference it is scored against, so annotation is treated
as a first-class part of the method. **`docs/ANNOTATION_PROTOCOL.md` defines what counts as a step
boundary** and must be followed for every annotation.

```bash
# 1. coarse pass — timestamped frame grids to locate transitions
python annotate.py contact-sheet --video Coolingfaninstallation.mp4 --step 1.0

# 2. fine pass — refine one boundary to 0.1 s
python annotate.py contact-sheet --video Coolingfaninstallation.mp4 \
    --around 12.5 --window 1.0 --step 0.1

# 3. record — interactive frame-stepping player, writes the JSON directly
python annotate.py review --video Coolingfaninstallation.mp4 \
    --output ground_truth/ground_truth_coolingfan.json --annotator "Your Name"

# 4. validate — rejects placeholder annotators, template round numbers, gaps/overlaps
python annotate.py validate --ground-truth ground_truth/ground_truth_coolingfan.json \
    --video Coolingfaninstallation.mp4

# 5. agreement — inter-annotator ceiling from two independent annotations
python annotate.py agreement --a gt_fan_annotatorA.json --b gt_fan_annotatorB.json
```

If you would rather read boundary times off the contact sheets than use the interactive
player, `build_ground_truth.py` turns a list of times into a valid file — filling in duration
and fps from the video, building contiguous steps, and refusing whole-second-heavy input
*before* writing rather than after `validate` rejects it:

```bash
python build_ground_truth.py --video ytbuildB_01_seg01.mp4 \
    --boundaries 11.3 24.7 38.1 51.4 --annotator "Your Name" \
    --out ground_truth/ground_truth_ytbuildb1.json
```

It deliberately does not propose boundaries. Ground truth generated by the system it scores
would make the evaluation circular.

> **Status:** all 30 annotation files pass `validate` with no errors or warnings: the five made by
> hand, and the 25 AI-drafted and human-reviewed files (`ground_truth_ytbuild*.json`,
> `ground_truth_input2*aug.json`). An AI draft that has not been reviewed still gets a warning.
> Four of the hand-made files were re-annotated after the original versions were found to have
> boundaries snapped to whole seconds; the effect of
> that correction on every reported number is documented in `docs/Final_Report.docx` §7.1.

---

## Evaluation tools

**Extended evaluation (no GPU / no re-run needed)** — every run saves a per-frame `features.csv`,
so the fast boundary-detection stage can be replayed offline to produce held-out cross-validation,
sensitivity, annotation robustness, a change-point baseline, and the cue ablation in seconds:

```bash
python evaluate_extended.py --src .      # writes extended_results.json and figures/
```

This reproduces every number and figure in the report's evaluation sections, and asserts that
re-segmenting the saved features reproduces the saved boundaries (guarding against drift).

**Module ablations and an end-to-end reproducibility check** — re-runs the pipeline per clip at
the evaluated configuration (each clip at its recorded processing rate) and tabulates F1 for both
scorers against the saved runs:

```bash
python run_ablations.py                      # --no-flow and --no-scene on every clip (~1 h on CPU)
python run_ablations.py --variants fresh     # full pipeline again: should match the saved runs
python run_ablations.py --summarize          # table from whatever has finished
```

The earlier flow/scene ablations in `results/abl_*` covered only the cooling fan, at an older
configuration (threshold 0.50, min-duration 3.0); there, disabling scene detection scored 1.000.
`run_ablations.py` repeats them properly on all clips.

**End-to-end check on every annotated video** — `verify_all.py` runs the full pipeline on all 30
annotated clips in `clips.json` (and any unreviewed drafts in `clips_ai_draft.json`) with their ground truth, then checks that each fresh run
reproduces the saved one: identical per-frame features and the same F1 for both scorers.

```bash
python verify_all.py --quick       # smoke test, first 150 frames of every clip (~20 min)
python verify_all.py               # full check, every clip (~5-6 h on CPU, resumable)
python verify_all.py --summarize   # table from whatever has finished
```

**Step report for a saved run:**

```bash
python step_report.py --run results/results_cpu_final --video split-video-data/CPUplacement.mp4 \
    --ground-truth ground_truth/ground_truth_cpuplacement.json
```

**Importing public egocentric data** — `import_segments.py` converts an external (start, end,
label) annotation table, such as EgoProceL's PC Assembly / PC Disassembly key-steps (Bansal et
al., ECCV 2022), into this project's ground-truth format, turning un-annotated gaps into explicit
`background` steps. Check the first converted file by eye before importing a whole dataset.

**Baseline comparison:**

```bash
python evaluate_baselines.py --ground-truth ground_truth/ground_truth_cpuplacement.json \
  --results results/results_cpu_final/segmentation_results.json
```

**Pipeline-level ablation** — re-run with a feature disabled:

```bash
python main.py ... --no-flow      # contribution of optical flow
python main.py ... --no-scene     # contribution of scene-change detection
```

**Refresh saved run reports** — rebuild `evaluation_report.txt` for every run from its saved
features and boundaries, without re-running the vision pipeline:

```bash
python refresh_run_reports.py --dry-run   # show deltas
python refresh_run_reports.py             # apply
```

---

## Testing

111 unit and integration tests cover the evaluation metrics (including the standard metrics), the
temporal segmenter (fusion, peak detection, segment construction, edge cases), the appearance
scorer (live path identical to the offline replay), the CLI defaults (must equal the evaluated
configuration), the annotation importer, the offline re-segmentation, the annotation provenance
(an unreviewed AI draft cannot enter `clips.json`; a promoted draft keeps its record in both
places), the edit-cut detector, and the assembly/disassembly pass (cue signs on synthetic
flow, the decision rule, the step-name rule, direction-aware wording, the quick-move naming
rule). They need only
`numpy`, `scipy`, and `opencv` — the heavy vision deps are imported lazily — so they run in
seconds:

```bash
pip install -r requirements-dev.txt
python -m unittest discover -s tests -v        # or: python -m pytest tests/
```

Continuous integration (`.github/workflows/ci.yml`) runs the suite, error-level lint
(`ruff check .`), the full evaluation replay, and the ground-truth validator on Python 3.11 and 3.12.
CI sets `REQUIRE_SAVED_RESULTS=1`, which turns "saved results not found" into a failure: the
re-segmentation drift tests had been silently *skipping* since `results/` was reorganised,
because they still looked in the repository root.

`docs/TEST_COMMANDS.md` lists every verification command with its expected output.

---

## Project structure

```
main.py                  Pipeline orchestration & CLI
config.py                Central configuration (FeatureParams, SegmenterParams)
video_loader.py          Frame loading / resizing / clip export
hand_tracker.py          MediaPipe hand tracking (Tasks API)
object_detector.py       YOLO-World open-vocabulary detection + workspace ROIs
optical_flow.py          Dense optical-flow features
scene_detector.py        Scene-change detection
interaction_tracker.py   Hand-object/region interaction
feature_extractor.py     Per-frame feature fusion -> transition score
temporal_segmenter.py    Boundary detection & segment construction
appearance.py            Training-free appearance-change scorer (descriptors + novelty)
edit_cuts.py             Hard edit-cut detector (diagnoses edited footage; no learned model)
step_report.py           Self-contained HTML step breakdown (keyframes, hand positions)
activity_recognizer.py   Human-readable activity labels (known-weak; see Limitations),
                         worded for assembly, disassembly or neither
action_direction.py      Assembly vs disassembly per step (optical-flow cue; experimental)
evaluate_direction.py    Leave-one-source-out evaluation of action_direction.py
evaluate_activity_names.py  Are steps named as turning vs moving correctly? (annotated verbs)
boundary_model.py        Learned boundary scorer: features, model loading, blending
train_boundary_model.py  Trains the learned scorer from saved features + annotations
learned_baseline.py      Held-out comparison of learned vs hand-designed scorers
evaluator.py             Boundary / segment metrics
visualizer.py            Timeline, annotated video, CSV export
utils.py                 Metrics & helpers
annotate.py              Annotation kit: contact sheets, review, validate, agreement
evaluate_baselines.py    Baseline comparison tool
evaluate_extended.py     Offline held-out CV, sensitivity, baselines, ablation, both scorers,
                         standard metrics (+ figures)
compare_scorer_variants.py  Every over-segmentation fix tried, same protocol
run_ablations.py         Module ablations / fresh full runs on every clip
import_segments.py       External annotation tables (e.g. EgoProceL) -> ground truth
refresh_run_reports.py   Rebuild saved run reports from saved features
extract_training_frames.py / train_hardware_model.py   Optional custom-detector kit

clips.json               Clip registry — the single source of truth for evaluation
clips_ai_draft.json      AI-drafted annotations awaiting review (kept apart; empty now)
promote_clip.py          Moves a reviewed clip into clips.json, keeping its provenance
ground_truth/            Manual annotations (ground_truth_*.json)
results/                 Saved runs: features.csv, boundaries, reports, per-run figures
docs/                    Report .docx files and the protocol/guide markdown
figures/                 Generated evaluation figures
notebooks/               Colab notebook for GPU training
tests/                   Unit + integration tests (metrics, segmenter, re-segmentation)
```

Modules stay flat at the repository root: they import each other directly, and
`pyproject.toml` declares them as top-level modules. Data and documents are
foldered. Scripts never hardcode a data path — `clip_registry.resolve_results`,
`resolve_gt` and `resolve_video` look the file up, so `clips.json` stores bare
names and the layout can change without editing any script.

---

## Ground truth format

```json
{
  "video": "Coolingfaninstallation.mp4",
  "annotator": "Osama Najjar",
  "annotation_method": "interactive frame-stepping review (annotate.py review)",
  "steps": [
    { "id": 0, "start": 0.0, "end": 4.63, "label": "Prepare cooler area" }
  ]
}
```

Boundaries are the interior step end-times (a file with *n* steps yields *n − 1* boundaries).
Annotate from the actual video — evenly-spaced placeholder times produce misleading scores, which
is exactly what `annotate.py validate` exists to catch.

---

## Limitations

- Only five clips were annotated by hand. The learned-vs-rule-based comparison uses only these and
  cannot reach statistical significance at this size. The other 25 annotations were drafted by an
  AI and reviewed by people; they make the held-out set larger, but they are not independent
  hand annotations.
- **Most of the footage is edited.** 45% of all boundaries (83% on the hand-annotated clips) sit
  on an editor's cut, and a cut detector alone matches the best scorer. Three clips first
  registered as "clean" turned out to contain jump cuts and were relabelled; only the three
  `input2x_aug` recordings are single shots, and none of them is a development clip.
- **Hand tracking is unavailable on most of this corpus.** Detection rates are 0.0% and 1.5% on the two
  overhead clips, 17–32% on the others, and 42–71% only on body-mounted footage. Camera placement, not
  configuration, decides this (`docs/Final_Report.docx` §7.7). On most clips the system is effectively an
  optical-flow segmenter.
- **The activity labeller does not work.** Measured against annotated step labels it scores
  0.000–0.039 across every clip, emitting generic phrases like "Unspecified hand activity", because
  the component-specific path depends on the zero-shot detector that cannot identify small parts.
  Labelling is out of scope for the brief and segmentation metrics are unaffected, but the
  component is retained and its failure is reported rather than hidden (`docs/Final_Report.docx` §7.4).
  One fix was measurable: the curved-trajectory test named most steps "Rotating", right only
  28% of the time (no better than the 27% base rate). Short (≤5 s) gripping steps with strong
  motion are now named "Grip and move a component" instead; on the 260 annotated steps whose
  verb is a turn or a move, the right kind of name rose from 23% to 54% (153 segments the
  system finds: 28% to 46%), `evaluate_activity_names.py`. Names still say the kind of motion,
  never the part.
- **Assembly vs disassembly is only weakly detected.** Names no longer claim installing by
  default, and `--task` names every step correctly when you know the video type, but the
  automatic per-step direction is right on about two steps in three and did not recognise the
  two disassembly recordings (see "Assembly vs disassembly (experimental)").
- **The rule-based scorer does not improve with more data** — its ~40 thresholds are hand-set, so a
  larger dataset needs re-tuning rather than retraining. This is part of why the learned scorer exists.
- Subtle, low-motion transitions can be missed.
- Edited, multi-shot tutorial footage is not well suited to feature-based segmentation.
- **The method is not causal.** Score normalisation uses the 5th/95th percentiles of the whole
  video, so it cannot run as a live stream without a windowed reformulation.
- Zero-shot detection cannot reliably identify small hardware parts; component labels would need a
  custom-trained detector (see `docs/TRAINING_GUIDE.md`).
- Inference is CPU-bound (~1.5 fps with the detector, ~6 fps without); a GPU greatly speeds it up.
- **The rule scorer over-segments** on four of five clips (its score saturates; see
  [Appearance scorer](#appearance-scorer-fixing-over-segmentation)). The appearance scorer fixes
  this on average but fails on the cooling fan.
- All hand-made ground truth was annotated by one person, who also built the system, and the
  held-out Intel clip was re-annotated after its earlier results were known. The other 25 clips
  were drafted by an AI that also wrote parts of the evaluation code, and the review changed no
  boundary. A second, blind annotator (`annotate.py agreement`) would give a human ceiling and
  remove both doubts.
- The saved `results/*/evaluation_report.txt` files were produced at each run's own (older)
  settings, so their F1 differs from the tables here, which replay every clip at the single global
  configuration. `run_ablations.py --variants fresh` regenerates the runs at the current defaults.
- RAM and Cable were evaluated at 14.985 fps (their runs used `--fps 15`), the other clips at
  10–12.5 fps; `clips.json` records each rate and the scripts reproduce it.

---

## Documentation

- `docs/Final_Report.docx` — full project report (design, evaluation, discussion, references).
- `docs/Evaluation_Report.docx` — focused evaluation report.
- `docs/Extended_Evaluation.docx` — supplement: held-out CV, sensitivity, robustness, change-point baseline, ablation.
- `docs/ANNOTATION_PROTOCOL.md` — what counts as a boundary, and how annotations are made and checked.
- `docs/DEMO_SCRIPT.md` — script for the demo video.
- `docs/TRAINING_GUIDE.md` — how to train a custom hardware detector.
- `evaluate_extended.py` + `extended_results.json` + `figures/` — extended evaluation and plots.
- `docs/TEST_COMMANDS.md` — how to verify everything, with expected output.
