# Ground-Truth Annotation Protocol

This document defines what counts as a step boundary in this project, how
boundaries are recorded, and how the resulting annotations are checked. It
exists because the measured F1 of the system is bounded above by the quality of
the reference it is scored against — an unverified reference makes every number
in the evaluation meaningless (see `Final_Report.docx` §7.1).

Every ground-truth file in this repository must be produced by this procedure
and must pass `python annotate.py validate`.

---

## 1. What is a step boundary?

A **step** is one elementary operation that a technician would name as a
separate instruction in a repair manual — "open the socket lever", "seat the
cooler", "attach the mounting clip".

A **boundary** is the instant the operator stops pursuing one such goal and
begins pursuing the next. Concretely, mark the boundary at the **first frame in
which the hands are committed to the new goal**.

### Mark a boundary when

- The hands **release one component and reach for another** — mark at the frame
  the release completes, not when the next object is touched.
- The **manipulation changes kind** on the same component: aligning becomes
  pressing, positioning becomes locking.
- A **tool is picked up or put down**.
- The operator **finishes seating/locking** a part and withdraws the hands.

### Do NOT mark a boundary when

- The hands **re-grip the same component** to continue the same operation.
- The camera moves, zooms, or refocuses but the activity is unchanged.
- There is a brief pause **within** an operation (hesitation, inspection glance).
- The video **cuts to a presenter, a diagram, or an overlay**. Editing cuts are
  not procedural boundaries. (Clips containing these are flagged as *edited
  tutorials* in the evaluation and analysed separately.)

### Tie-breaking rules

- If a transition is gradual, mark the **midpoint of the ambiguous interval**
  and record the interval width in the step's `notes` field.
- If two annotators disagree by more than 1.0 s, the segment is re-reviewed
  jointly rather than averaged.
- The first step always starts at `0.0`; the last step always ends at the video
  duration. Steps must **tile the video with no gaps or overlaps**.

---

## 2. Procedure

### Step 1 — coarse pass (find approximate boundaries)

```bash
python annotate.py contact-sheet --video Coolingfaninstallation.mp4 --step 1.0
```

Writes timestamped frame grids to `annotation_kit/`. Scan them and note the
approximate second at which each transition occurs. One second of granularity
is enough for this pass.

### Step 2 — fine pass (localise each boundary)

For each approximate boundary `t` from step 1:

```bash
python annotate.py contact-sheet --video Coolingfaninstallation.mp4 \
    --around 12.5 --window 1.0 --step 0.1
```

This renders ±1 s around `t` at 0.1 s granularity. Read off the exact frame at
which the hands commit to the new goal.

### Step 3 — record the annotation

Either fill in the JSON by hand, or use the interactive player, which writes a
correctly-structured file directly:

```bash
python annotate.py review --video Coolingfaninstallation.mp4 \
    --output ground_truth_coolingfan.json --annotator "Your Name"
```

Controls: `←`/`→` step one frame, `a`/`d` jump one second, `w`/`s` jump five
seconds, `SPACE` mark a boundary, `u` undo, `q` save and quit.

**The `annotator` field must contain a real person's name.** Values such as
`manual`, `draft`, or anything containing `VERIFY` are rejected by the
validator — an unattributed annotation cannot be defended in review.

### Step 4 — validate

```bash
python annotate.py validate --ground-truth ground_truth_coolingfan.json \
    --video Coolingfaninstallation.mp4
```

The validator fails on:

| Check | Why it matters |
|---|---|
| `annotator` missing, placeholder, or marked `VERIFY` | Provenance cannot be established. |
| ≥60 % of interior boundaries on exact whole seconds | The fingerprint of an unadjusted template, not a real annotation. |
| `end <= start` on any step | Malformed interval. |
| Gap or overlap between consecutive steps | Steps must tile the video; coverage metrics assume this. |
| Last step end ≠ video duration | Ground truth does not span the footage. |

It warns on placeholder labels, missing `annotation_method`, and boundary times
that do not lie on a frame boundary.

### Step 5 — second annotator and agreement

At least one clip must be annotated **independently** by a second person, who
must not see the first annotation before finishing.

```bash
python annotate.py agreement --a gt_fan_osama.json --b gt_fan_mostafa.json \
    --output agreement_coolingfan.json
```

Report the **F1 at 1.0 s tolerance** as the inter-annotator agreement. This is
the ceiling on what any automatic method can meaningfully score: a system that
matches human annotation as well as two humans match each other has saturated
the benchmark. Quote it alongside the system's F1 in the report.

---

## 3. File format

```json
{
  "video": "Coolingfaninstallation.mp4",
  "annotator": "Osama Najjar",
  "annotation_method": "interactive frame-stepping review (annotate.py review)",
  "video_duration": 48.13,
  "fps": 30.0,
  "steps": [
    { "id": 0, "start": 0.0, "end": 4.63, "label": "Prepare cooler area", "notes": "" }
  ]
}
```

Boundaries used by the evaluator are the **interior step end-times** — that is,
`end` of every step except the last. A file with *n* steps yields *n − 1*
boundaries.

---

## 4. Current status of the annotations in this repository

| File | Annotator field | Passes validator | Action required |
|---|---|---|---|
| `ground_truth_coolingfan_v2.json` | `manual` | ✗ (name only) | Set a real name, add `annotation_method`. Times are genuine (non-round). |
| `ground_truth_cpuplacement.json` | `draft_from_frame_review_VERIFY` | ✗ | **Re-annotate** — all boundaries on whole seconds. |
| `ground_truth_raminstallation.json` | `draft_from_frame_review_VERIFY` | ✗ | **Re-annotate.** |
| `ground_truth_cableconnection.json` | `draft_from_frame_review_VERIFY` | ✗ | **Re-annotate.** |
| `ground_truth_installintelcpu.json` | `draft_from_frame_review_VERIFY` | ✗ | **Re-annotate.** |
| `ground_truth_coolingfan.json` (v1) | `manual` | ✗ | Superseded by v2; kept only for the annotation-robustness comparison. |

Re-annotating the four flagged files, and producing one second-annotator file
for the agreement number, is the single highest-value remaining task on the
project.

### Why there are two cooling-fan annotations

`ground_truth_coolingfan.json` (6 steps) was a first coarse pass.
`ground_truth_coolingfan_v2.json` (8 steps) is a finer re-annotation at a
consistent level of granularity, made **before** the reported runs and used for
every number in the report. v1 is retained deliberately: comparing results
against both quantifies how much the choice of annotation granularity moves the
score, which is reported in the annotation-robustness analysis. The v2 file was
not produced by re-annotating until the score improved — the granularity rule
in §1 was fixed first and applied uniformly.
