# Ground-Truth Annotation Protocol

This document defines what counts as a step boundary in this project, how
boundaries are recorded, and how the resulting annotations are checked. It
exists because the measured F1 of the system is bounded above by the quality of
the reference it is scored against — an unverified reference makes every number
in the evaluation meaningless (see `docs/Final_Report.docx` §7.1).

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
    --output ground_truth/ground_truth_coolingfan.json --annotator "Your Name"
```

Controls: `←`/`→` step one frame, `a`/`d` jump one second, `w`/`s` jump five
seconds, `SPACE` mark a boundary, `u` undo, `q` save and quit.

**The `annotator` field must contain a real person's name.** Values such as
`manual`, `draft`, or anything containing `VERIFY` are rejected by the
validator — an unattributed annotation cannot be defended in review.

### Step 4 — validate

```bash
python annotate.py validate --ground-truth ground_truth/ground_truth_coolingfan.json \
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

| File | Annotator | Passes validator | Notes |
|---|---|:--:|---|
| `ground_truth_coolingfan_v2.json` | Mostafa Taha | ✓ | Times read from frames; annotator field and method added. |
| `ground_truth_cpuplacement.json` | Mostafa Taha | ✓ | Re-annotated from frames; 7 steps. |
| `ground_truth_raminstallation.json` | Mostafa Taha | ✓ | Re-annotated from frames; 9 steps. |
| `ground_truth_cableconnection.json` | Mostafa Taha | ✓ | Re-annotated from frames; 5 steps. |
| `ground_truth_installintelcpu.json` | Mostafa Taha | ✓ | Re-annotated from frames; 11 steps (held-out clip). |
| `ground_truth_ytbuild*.json` (22), `ground_truth_input2*aug.json` (3) | Mostafa Taha, Osama Najjar | ✓ | Drafted with an AI tool, checked against the video by Mostafa Taha and Osama Najjar; see §5. Held-out clips. |

**All 30 evaluation files pass `annotate.py validate` with no errors or warnings**, and
CI enforces this on every push.

Four of them previously failed: their boundaries were snapped to whole seconds,
which is the signature of times estimated rather than read from frames. Correcting
them moved every reported number — the development mean fell from 0.538 to 0.477
while the two uncontaminated estimates rose sharply (leave-one-out 0.500 → 0.684,
held-out 0.286 → 0.583) and the learned-versus-rule-based comparison reversed
entirely. That episode is analysed in `Final_Report.docx` §7.1 and is the reason
this validator exists.

The remaining open item is a **second independent annotation** of at least one
clip, which would give an inter-annotator agreement ceiling and bound how much of
the residual error belongs to the system rather than to the reference.

### Why there are two cooling-fan annotations

`ground_truth_coolingfan.json` (6 steps) was a first coarse pass.
`ground_truth_coolingfan_v2.json` (8 steps) is a finer re-annotation at a
consistent level of granularity, made **before** the reported runs and used for
every number in the report. v1 is retained deliberately: comparing results
against both quantifies how much the choice of annotation granularity moves the
score, which is reported in the annotation-robustness analysis. The v2 file was
not produced by re-annotating until the score improved — the granularity rule
in §1 was fixed first and applied uniformly.

## 5. AI-drafted, human-reviewed annotations (25 clips in `split-video-data/`)

`ground_truth/ground_truth_ytbuild{a1..a3,b1..b8,c1..c11}.json` (22 edited YouTube
build clips) and `ground_truth/ground_truth_input{22,23,24}aug.json` (3 continuous
recordings) were **drafted with an AI tool**, not placed by hand, so that these clips
could be scored at all, and then **checked by Mostafa Taha and Osama Najjar**. They follow §1 as written. The
drafts were made the way a person would use §2 without the interactive player:

1. a 2 s contact sheet over the whole clip to find candidate changes of goal;
2. 0.5 s strips (±2 s) around every candidate — 0.1–0.2 s for the three continuous
   recordings — to place the boundary on the first frame where the hands commit to
   the new goal;
2b. (version 2) a 1 s contact sheet over the whole clip to add the elementary steps
   of §1 that the first pass merged: a tool picked up or put down, a release and a
   reach for another item, a change in the kind of manipulation. Version 1 had one
   step every 15.5 s against 6.6 s in the hand-made files; version 2 has one every
   6.3 s. Version 1 is kept in `ground_truth/_superseded/version1_drafts/`;
3. where the boundary falls on a hard edit cut, it was snapped to the cut frame;
   an edit cut on its own was **not** a boundary unless the operation changed;
4. the system's predictions were never opened while annotating.

**Review.** Mostafa Taha and Osama Najjar checked version 2
against the videos and changed no boundary. Each file records both stages:

```json
"annotator": "Mostafa Taha, Osama Najjar",
"annotation": "ai_draft",
"review_status": "human_reviewed",
"annotation_method": "Boundaries drafted with an AI tool from the video frames (version 2), then checked against the video and approved by Mostafa Taha and Osama Najjar; no boundaries changed.",
"draft_method": "<how the draft was made: steps 1-4 above>"
```

`promote_clip.py` then moved the 25 clips from `clips_ai_draft.json` into `clips.json`
as **held-out** clips with `"annotation_source": "ai_draft_reviewed"`, decided before any
of them was scored as a held-out clip. They are not used for tuning, and the learned
scorers still train and test on the five hand-made files only.

What still makes them weaker than the hand-made files: the boundaries were placed from
1 s sheets and 0.5 s strips, not by frame-stepping; the review confirmed them rather than
placing them independently; and there is no blind second annotation. Quote results on
these clips as "AI-drafted, human-reviewed annotations", and keep them apart from the
hand-annotated results (`by_source` in `extended_results.json` does this).

### Reviewing and promoting a further drafted clip

A new draft goes into `clips_ai_draft.json` with `"review_status": "pending_human_review"`;
the validator warns about it and it is reported separately. To review it:

```bash
python annotate.py review --video split-video-data/<clip>.mp4 \
    --output ground_truth/ground_truth_<clip>.json --annotator "Your Name"
```

Check every boundary against §1 (move, add or delete as needed). Then set
`"review_status": "human_reviewed"`, put the people who checked it in `annotator`,
describe the check in `annotation_method` (say that the boundaries were drafted with an
AI tool if they still are), and run `annotate.py validate` — the warning
disappears. Finally:

```bash
python promote_clip.py --name "<clip name>" --ground-truth ground_truth_<clip>.json   # held out by default
```

Decide whether it is `dev` or `heldout` **before** looking at its score; a reviewed
clip that you then tune on is a development clip.

Keep in mind when interpreting the YouTube clips: they are edited tutorials, and 42% of
the held-out boundaries lie on an editor's cut (`edit_cuts.py`; 83% on the five
hand-annotated clips). A step change that the editor marked with a cut is easy for any
detector of sudden picture change, so the boundaries that are *not* at a cut are the
informative ones. The three continuous recordings have no cuts and are the closest to
the target footage.
