# Re-annotation plan

> **STATUS: COMPLETED.** All four flagged files were re-annotated and all five
> evaluation files now pass `annotate.py validate` with no errors; CI enforces
> this on every push. The effect on every reported number is analysed in
> `Final_Report.docx` §7.1 and summarised in `ANNOTATION_PROTOCOL.md` §4.
> This document is retained as a record of the procedure that was followed, and
> as the template for annotating any future clip.
>
> The one item still open is the optional inter-annotator agreement study at the
> end of this document.

## Why

At the time this plan was written, all five ground-truth files failed `annotate.py validate`. Four failed because their
boundary times are exact whole seconds, which is the signature of times estimated
by scrubbing a player rather than read off frames. Real transitions do not land on
whole seconds; five doing so by chance is roughly a 1-in-100,000 event.

This costs measured F1 directly. At the 0.5 s tolerance a prediction only counts
if it lands within half a second of the annotated time. If the true transition is
at 6.4 s, the file says 6.0 s, and the system predicts 6.35 s, the system is
**correct and scored wrong**.

The evidence that this is happening:

| Clip | ground truth | F1 @1.0s |
|---|---|:--:|
| Cooling fan | real measured times | **1.000** |
| CPU placement | 83% whole seconds | 0.667 |
| RAM install | 100% whole seconds | 0.353 |
| Intel CPU install | 80% whole seconds | 0.286 |
| Cable connect | 100% whole seconds | 0.133 |

The only clip with a properly annotated answer key is the only clip that scores
perfectly. Fixing the other four may well raise the reported numbers, and will
make them defensible either way.

---

## Split

| Clip | File to rewrite | Sheets | Owner |
|---|---|---|---|
| Cable connect | `ground_truth/ground_truth_cableconnection.json` | `annotation_sheets/Cableconnection/` | |
| RAM install | `ground_truth/ground_truth_raminstallation.json` | `annotation_sheets/RAMinstallation/` | |
| CPU placement | `ground_truth/ground_truth_cpuplacement.json` | `annotation_sheets/CPUplacement/` | |
| Intel CPU install | `ground_truth/ground_truth_installintelcpu.json` | `annotation_sheets/installintelcpu/` | |

Two each is sensible. Do them independently — do not annotate side by side, and do
not look at the existing file first. Both would bias you toward the old times.

**Do not open any `annotated_output.mp4`.** Those have the system's predicted
boundaries drawn on the frames. Annotating from them makes the evaluation
circular and the results worthless.

---

## Method, per clip

**1. Read the sheets.** Each shows one frame per second with its timestamp. Note
roughly where one step ends and the next begins.

**2. Refine each boundary to 0.1 s.** For a boundary near 24 s:

```bat
python annotate.py contact-sheet --video split-video-data\RAMinstallation.mp4 --around 24.0 --window 1.0 --step 0.1
```

That writes a sheet of 21 frames covering 23.0–25.0 s into `annotation_kit\`.
Find the first frame where the new step has begun; use that timestamp.

**3. Write the file.**

```bat
python build_ground_truth.py --video RAMinstallation.mp4 ^
  --boundaries 6.3 21.8 32.4 41.2 47.6 ^
  --annotator "Your Name" ^
  --out ground_truth\ground_truth_raminstallation.json
```

It refuses input that is mostly whole seconds before writing anything.

**4. Validate.** Must pass with no errors:

```bat
python annotate.py validate --ground-truth ground_truth\ground_truth_raminstallation.json --video RAMinstallation.mp4
```

---

## What counts as a boundary

See `ANNOTATION_PROTOCOL.md` §2 for the full definition. In short: the moment the
operator stops working on one component and starts on the next. Use the frame
where the *new* action has visibly begun, not the last frame of the old one, and
be consistent about that choice across every clip — consistency matters more than
which convention you pick.

---

## After both of you finish

```bat
python evaluate_extended.py --src .
python learned_baseline.py --src .
python refresh_run_reports.py
```

Every number in the README and both reports is derived from these, so they will
need updating afterwards. The boundary *predictions* do not change — only what
they are measured against — so this is a re-measurement, not a re-run.

---

## Optional: inter-annotator agreement

Pick one clip you have both annotated independently and save the two files
separately, then:

```bat
python annotate.py agreement --a gt_ram_mostafa.json --b gt_ram_osama.json
```

This reports the mean boundary offset between two humans and their F1 against
each other. It is the ceiling on what any system could score — if two annotators
differ by 0.4 s on average, a system accurate to 0.5 s is at human level, and the
report can say so. Very few student projects measure this, and the tool is already
built.
