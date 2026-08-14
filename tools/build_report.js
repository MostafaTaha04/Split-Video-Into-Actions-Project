/**
 * build_report.js — generates Final_Report.docx.
 *
 * The report is built from source rather than hand-edited so that its numbers,
 * tables and figures stay consistent with the results files they come from. The
 * figures are read live from figures/, so regenerating after a new evaluation
 * run picks them up automatically.
 *
 * Usage (needs Node.js). Run from the repository root:
 *     npm install docx          # creates node_modules/ (gitignored)
 *     node tools/build_report.js . Final_Report.docx
 *
 * Node resolves require() from the script's directory upwards, so the `docx`
 * package must be installed in the repository root (or reachable via NODE_PATH)
 * — installing it inside tools/ will not work.
 *
 * The first argument is the repository root (used to locate figures/), the
 * second is the output path.
 *
 * NOTE: Word will not populate the table of contents automatically. Open the
 * document, click the Contents field and press F9.
 *
 * If you prefer to edit the .docx directly from here on, that is fine — just be
 * aware this script would overwrite those edits if re-run.
 */
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  TableOfContents, PageBreak, ImageRun,
} = require("docx");

const REPO = process.argv[2];
const OUT = process.argv[3];

// ---------------------------------------------------------------- helpers
const P = (text, opts = {}) => new Paragraph({
  spacing: { after: opts.after ?? 140, line: 276 },
  alignment: opts.align,
  children: [new TextRun({ text, italics: opts.italics, bold: opts.bold, size: opts.size ?? 22 })],
});

const H1 = (text) => new Paragraph({
  text, heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 160 },
});
const H2 = (text) => new Paragraph({
  text, heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 120 },
});

const bullet = (text) => new Paragraph({
  text, bullet: { level: 0 }, spacing: { after: 80, line: 276 },
});

const caption = (text) => new Paragraph({
  spacing: { before: 60, after: 220 },
  children: [new TextRun({ text, italics: true, size: 18, color: "444444" })],
});

// Table with dual widths (DXA) as required for Google Docs compatibility.
function makeTable(headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (text, opts = {}) => new TableCell({
    width: { size: opts.w, type: WidthType.DXA },
    shading: opts.head
      ? { type: ShadingType.CLEAR, fill: "E8EEF4", color: "auto" }
      : undefined,
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    children: [new Paragraph({
      alignment: opts.align,
      spacing: { after: 0, line: 240 },
      children: [new TextRun({ text: String(text), bold: opts.head, size: 19 })],
    })],
  });

  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({
        tableHeader: true,
        children: headers.map((h, i) =>
          cell(h, { head: true, w: widths[i], align: i ? AlignmentType.CENTER : AlignmentType.LEFT })),
      }),
      ...rows.map((r) => new TableRow({
        children: r.map((v, i) =>
          cell(v, { w: widths[i], align: i ? AlignmentType.CENTER : AlignmentType.LEFT })),
      })),
    ],
  });
}

function figure(file, widthPx, heightPx, cap) {
  const p = path.join(REPO, "figures", file);
  if (!fs.existsSync(p)) return [];
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 40 },
      children: [new ImageRun({
        type: "png",
        data: fs.readFileSync(p),
        transformation: { width: widthPx, height: heightPx },
      })],
    }),
    caption(cap),
  ];
}

// ---------------------------------------------------------------- content
const children = [];

// Title block
children.push(
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 60 },
    children: [new TextRun({ text: "Splitting a Video into Actions", bold: true, size: 40 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: [new TextRun({
      text: "Temporal Segmentation of Egocentric Hardware-Assembly Video", size: 26,
    })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 240 },
    children: [new TextRun({ text: "Final Project Report", italics: true, size: 24 })],
  }),
  P("Osama Najjar  ·  Mostafa Taha", { align: AlignmentType.CENTER, after: 40 }),
  P("Mentor: Saeed Namnah", { align: AlignmentType.CENTER, after: 40 }),
  P("Project 6  ·  June / July 2026", { align: AlignmentType.CENTER, after: 320 }),
);

children.push(H1("Contents"));
children.push(new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ---- Abstract
children.push(H1("Abstract"));
children.push(P(
  "This project builds a system that automatically divides an egocentric (first-person) video of a " +
  "hardware-assembly procedure into its constituent steps. The task is temporal segmentation — locating " +
  "the boundaries between procedural steps in time — and explicitly excludes action recognition. A modular, " +
  "feature-based pipeline fuses hand tracking, open-vocabulary object detection with workspace regions of " +
  "interest, dense optical flow, scene-change detection, and hand-object interaction cues into a per-frame " +
  "transition score, from which step boundaries are extracted as prominent peaks."
));
children.push(P(
  "The system is evaluated on five PC-assembly clips against manually annotated ground truth using boundary " +
  "precision, recall, and F1 at several tolerances. All headline results use a single global configuration " +
  "(boundary threshold 0.70, minimum segment duration 2.0 s) applied unchanged to every clip, with no " +
  "per-clip tuning: it reaches F1 0.727 and 0.714 at a one-second tolerance on the two clean development " +
  "clips, localising matched boundaries to within 0.1–0.4 s, and it outperforms uniform, random, and " +
  "change-point baselines at the tight tolerances that matter. Leave-one-clip-out cross-validation gives a " +
  "clean held-out mean F1 of 0.658, and a fifth clip that took no part in tuning at any stage is reported " +
  "separately at F1 0.286 (1.0 s) / 0.571 (2.0 s), with its failure mode diagnosed rather than omitted."
));
children.push(P(
  "A cue ablation showed that four of the six original fusion channels never improved the result and on 42 " +
  "of 192 configurations actively degraded it; removing them simplified the method without any loss. " +
  "Supervised alternatives were also trained and evaluated rather than merely argued against: three " +
  "classifiers learned from the project's own per-frame features do not significantly beat the " +
  "hand-designed fusion, and the simplest model outperforms the neural network — the expected signature of " +
  "limited data. The two approaches prove complementary, and a hybrid scorer that blends them roughly " +
  "doubles the worst-case per-clip F1 (0.316 against 0.133) and cuts the spread across clips by a factor " +
  "of 2.7, at no cost to the clean-clip result."
));
children.push(P(
  "The work yields four methodological findings: evaluation quality is dominated by the fidelity of the " +
  "ground-truth annotation — which motivated a formal annotation protocol, a frame-accurate annotation " +
  "tool, and an automatic validator; edited tutorial footage is poorly suited to feature-based " +
  "segmentation; zero-shot open-vocabulary detection cannot identify small hardware parts in close-up, " +
  "hand-occluded footage; and rule-based and learned scorers fail in opposite directions, which is what " +
  "makes combining them worthwhile."
));

// ---- 1 Introduction
children.push(H1("1. Introduction"));
children.push(P(
  "Manual procedures — maintenance, assembly, and repair — are increasingly captured on first-person " +
  "cameras for training and documentation. Turning such a continuous recording into a structured, " +
  "step-by-step record by hand is tedious, which motivates automatic segmentation. This project addresses " +
  "that problem for PC-hardware assembly: given an egocentric video of an operator performing a task, the " +
  "system deconstructs the continuous procedure into distinct, ordered steps."
));
children.push(P(
  "Following the project brief, the goal is strictly temporal segmentation based on procedural boundaries; " +
  "action detection or recognition is not required. The system therefore tracks the interplay among the " +
  "operator's hand movements, the tools and components in view, and the points of physical interaction on " +
  "the device, and it marks a step boundary wherever these signals indicate a transition. The primary " +
  "measure of success is the accuracy of the detected boundaries relative to a human annotation."
));
children.push(P("The contributions of this report are:"));
[
  "A complete modular segmentation pipeline requiring no per-frame labels and no training.",
  "An evaluation across five clips with multi-tolerance boundary metrics, under a single global configuration with no per-clip tuning.",
  "A baseline comparison against uniform, random, and change-point segmentation, demonstrating the method adds value at tight tolerances.",
  "A two-level ablation — pipeline features and fusion channels — that both justifies the multi-feature design and identifies redundant channels, which were then removed.",
  "A formal ground-truth annotation protocol with supporting tooling (contact sheets, frame-accurate review, an automatic validator, and inter-annotator agreement).",
  "A discussion of three practical findings: annotation fidelity dominates measured scores, source-footage type determines feasibility, and zero-shot detection fails on small hardware parts.",
].forEach((t) => children.push(bullet(t)));

// ---- 2 Background
children.push(H1("2. Background and Related Work"));
children.push(P(
  "Methods for dividing a video in time fall into two broad families: supervised temporal action " +
  "segmentation, which learns to label frames and reads boundaries from label changes, and unsupervised or " +
  "rule-based change-point methods, which detect where low-level signals shift without learning specific " +
  "actions. This project, constrained to segmentation rather than recognition and limited to a handful of " +
  "videos, follows the second family. The relevant prior work and the components reused are reviewed below."
));
children.push(H2("2.1 Supervised temporal action segmentation"));
children.push(P(
  "The dominant academic approach learns a per-frame action label and derives segment boundaries where the " +
  "predicted label changes. Temporal convolutional models such as MS-TCN [7] refine frame-wise predictions " +
  "over multiple stages and achieve strong results on densely labelled benchmarks. More recent work replaces " +
  "the frame-wise backbone with self-supervised video transformers: VideoMAE [10] pre-trains a masked " +
  "autoencoder on unlabelled video and is then fine-tuned for clip-level action classification, and applying " +
  "such a classifier over a sliding window is a common route to a segmented timeline. These methods are " +
  "accurate but depend on large, fully annotated training sets, a fixed (closed) set of action classes, and " +
  "substantial GPU training. Neither the data nor the task definition of this project supports that " +
  "approach: only a few clips are available, no per-frame labels exist, and the brief explicitly excludes " +
  "action recognition."
));
children.push(P(
  "Rather than leaving this as an assumption, the claim is tested directly in Section 6.7, where supervised " +
  "classifiers are trained on the project's own per-frame features under leave-one-clip-out cross-validation " +
  "and compared with the hand-designed fusion under identical post-processing. Supervised segmentation is " +
  "therefore both reviewed as context and evaluated as an alternative."
));
children.push(H2("2.2 Change-point and shot-boundary detection"));
children.push(P(
  "The alternative is to locate the moments at which the observed signals change, without naming the " +
  "activity. This connects to two classical areas: change-point detection in time series, which flags " +
  "abrupt shifts in a signal's statistics, and shot- or scene-boundary detection in video, which finds " +
  "editing cuts using histogram and structural differences between frames. The present system is an " +
  "instance of this family applied to multiple fused cues: it builds a transition signal from motion, hand, " +
  "and scene features and locates step boundaries as its prominent peaks. This makes the method " +
  "interpretable and label-free, at the cost of relying on the chosen low-level cues being informative for " +
  "the procedure at hand. A dynamic-programming change-point baseline (ruptures) is included in the " +
  "evaluation for direct comparison."
));
children.push(H2("2.3 Egocentric video and hand-object interaction"));
children.push(P(
  "First-person procedural video has been studied at scale through datasets such as EPIC-KITCHENS [8], " +
  "which highlight that the hands and their contact with objects are the most informative signals for what " +
  "the operator is doing. Work on hand-object contact detection at internet scale [9] similarly shows that " +
  "hand state and contact transitions align with action transitions. This motivates the present design, " +
  "which uses hand motion, grip onset and release, and hand-region contact as primary boundary cues — the " +
  "points at which the hands change what they are doing are exactly where steps tend to begin and end."
));
children.push(H2("2.4 Building blocks used in this work"));
children.push(P(
  "The pipeline reuses established components rather than inventing new ones. Hand pose comes from " +
  "MediaPipe Hand Landmarker [1]; component presence is probed with the open-vocabulary detector " +
  "YOLO-World [2] via the Ultralytics library [6]; frame-to-frame motion uses Farnebäck dense optical " +
  "flow [3] from OpenCV [4]; and boundary peaks are located with peak finding from SciPy [5]. The " +
  "contribution of this project is not any single component but their fusion into one transition signal " +
  "tailored to procedural assembly video, together with a careful, baseline- and ablation-supported " +
  "evaluation methodology."
));

// ---- 3 System design
children.push(H1("3. System Design"));
children.push(P(
  "The system is a frame-by-frame pipeline. Each sampled frame passes through independent feature " +
  "extractors; their outputs are fused into two per-frame quantities — an activity level and a transition " +
  "score — and step boundaries are the prominent, well-separated peaks of the smoothed transition score."
));
children.push(makeTable(
  ["Module (file)", "Responsibility"],
  [
    ["video_loader.py", "Frame extraction, resizing, FPS sub-sampling, clip export."],
    ["hand_tracker.py", "MediaPipe hand landmarks: position, velocity, grip, curvature."],
    ["object_detector.py", "YOLO-World open-vocabulary detection + workspace ROI fallback."],
    ["optical_flow.py", "Dense optical flow: magnitude, direction, discontinuity."],
    ["scene_detector.py", "Histogram/structural scene-change score."],
    ["interaction_tracker.py", "Hand-object/region contact and contact-point dynamics."],
    ["feature_extractor.py", "Fuses all signals into per-frame activity and transition scores."],
    ["temporal_segmenter.py", "Peak detection, filtering, segment construction."],
    ["activity_recognizer.py", "Human-readable activity label per segment."],
    ["evaluator.py", "Boundary P/R/F1, IoU, coverage against ground truth."],
    ["visualizer.py", "Timeline, annotated video, feature CSV export."],
    ["annotate.py", "Annotation kit: contact sheets, frame-accurate review, validator, agreement."],
    ["boundary_model.py", "Learned boundary scorer: feature construction, model loading, blending."],
    ["train_boundary_model.py", "Trains the learned scorer from saved features and annotations."],
  ],
  [2600, 6400],
));
children.push(caption("Table 1. Pipeline modules and their responsibilities."));

children.push(...figure("pipeline_architecture.png", 560, 300,
  "Figure 1. Pipeline architecture: independent feature extractors feed a per-frame fusion stage, " +
  "whose smoothed peaks become step boundaries."));

children.push(H2("3.1 Feature fusion and boundary detection"));
children.push(P(
  "Each extractor contributes evidence of a transition: a change in the number of visible hands or real " +
  "components, the onset or release of a grip, a change in interaction type, a spike in optical-flow " +
  "discontinuity or direction change, and a scene-change event. These are combined by taking the strongest " +
  "active cue, producing a transition score in the range zero to one. The score is smoothed with a Gaussian " +
  "filter, and peaks above a threshold and separated by a minimum duration become boundaries."
));
children.push(P(
  "The transition logic was refined for robustness so that the heuristic workspace regions — which switch " +
  "on and off frame-to-frame with motion — no longer create spurious boundaries, and additional cues fire " +
  "when motion resumes after a pause or settles after a burst, improving sensitivity to low-motion " +
  "transitions. The boundary-score fusion itself was reduced from six channels to two following the " +
  "ablation reported in §6.5; every threshold and weight in both stages is centralised in config.py " +
  "(FeatureParams and SegmenterParams) rather than embedded in the code."
));
children.push(H2("3.2 Segment construction"));
children.push(P(
  "Segments are the intervals between consecutive boundaries. Adjacent segments share their boundary frame, " +
  "so the segments tile the video exactly with no gaps. (An earlier version ended each segment one frame " +
  "before its boundary while the next began at it, leaving a one-frame hole per boundary; this did not move " +
  "any boundary, and so did not affect boundary F1, but it understated the reported coverage ratio and " +
  "segment IoU by roughly one frame per boundary. The corrected figures are used throughout this report.)"
));
children.push(H2("3.3 Activity labelling"));
children.push(P(
  "After segmentation, each segment is given a human-readable label. When a real component is recognised, " +
  "the label is component-specific (for example “CPU cooler installation”). When no component is detected, " +
  "the label is derived from the dominant hand-motion phase of the segment — reaching, gripping and moving, " +
  "fine adjustment, or pause — so that segments remain informative even without object detection."
));

children.push(H2("3.4 Selectable boundary scorers"));
children.push(P(
  "The per-frame boundary score can be produced in three ways, selected with --scorer, after which " +
  "smoothing, peak detection and segment construction are identical. This makes the alternatives " +
  "directly comparable, since any difference in the result is attributable to the score itself."
));
[
  "rule — the hand-designed fusion described above. This is the default and remains the strongest choice on clean, continuously recorded footage.",
  "learned — the probability, from a classifier trained on the project's own per-frame features, that a frame lies on a step boundary (Section 6.7).",
  "hybrid — a convex blend of the two, w·(rule) + (1−w)·(learned), with w chosen by cross-validation. This is the most robust option across footage types.",
].forEach((t) => children.push(bullet(t)));
children.push(P(
  "The learned model is trained by train_boundary_model.py from the saved per-frame features and the " +
  "existing boundary annotations, so no additional labelling is required. Feature construction is " +
  "shared between training and inference, and the saved model records the feature layout it expects; " +
  "a mismatch is rejected at load time rather than silently degrading accuracy."
));

// ---- 4 Implementation
children.push(H1("4. Implementation"));
children.push(P(
  "The system is implemented in Python using OpenCV for image processing, MediaPipe for hand tracking, the " +
  "Ultralytics library for YOLO-World, and NumPy/SciPy for signal processing. It is configured through a " +
  "single configuration object and a command-line interface exposing the detector mode, processing rate, " +
  "resolution, boundary threshold, minimum segment duration, and the evaluation ground truth. Each run " +
  "produces a results JSON, a timeline figure, an annotated video, per-segment clips, and a feature CSV."
));
children.push(P(
  "Two robustness fixes were required during development. Hand tracking initially failed because the " +
  "packaged MediaPipe graph assets could not be located; this was resolved by adopting the modern MediaPipe " +
  "Tasks API and loading the model from an in-memory buffer, which also avoids a failure mode on file paths " +
  "containing non-ASCII characters. Open-vocabulary detection was made deterministic (run on a fixed frame " +
  "schedule) and given higher inference resolution to give small parts the best possible chance of detection."
));
children.push(H2("4.1 Engineering quality and reproducibility"));
children.push(P(
  "The repository is covered by 46 unit and integration tests exercising the evaluation metrics, the " +
  "temporal segmenter (fusion, peak detection, segment construction, and edge cases such as empty input, " +
  "flat signals, and clips shorter than one minimum segment), and an end-to-end regression test asserting " +
  "that re-segmenting saved per-frame features reproduces the saved boundaries exactly. Continuous " +
  "integration runs the suite, error-level linting, the full evaluation replay, and the ground-truth " +
  "validator on Python 3.10 and 3.11. Dependencies are version-pinned with upper bounds, because both " +
  "MediaPipe and Ultralytics make breaking API changes across minor versions."
));

// ---- 5 Dataset
children.push(H1("5. Dataset and Ground Truth"));
children.push(P(
  "Five short PC-assembly clips were used. Four form the development set: cooling-fan installation and CPU " +
  "placement (single continuous recordings, referred to as clean procedures), and RAM installation and " +
  "cable connection (edited tutorials containing presenter cut-aways, diagram slides, on-screen overlays, " +
  "and multiple motherboards). A fifth clip, an Intel CPU installation, was added after the configuration " +
  "was frozen and took no part in tuning at any stage; it is reported separately as a true held-out result " +
  "in §6.4."
));
children.push(makeTable(
  ["Clip", "Type", "Duration", "Steps", "Role"],
  [
    ["Cooling fan installation", "Clean procedure", "48.1 s", "8", "Development"],
    ["CPU placement", "Clean procedure", "51.8 s", "7", "Development"],
    ["RAM installation", "Edited tutorial", "59.8 s", "6", "Development"],
    ["Cable connection", "Edited tutorial", "57.4 s", "7", "Development"],
    ["Intel CPU install", "Clean procedure", "50.8 s", "6", "Held out"],
  ],
  [2700, 1900, 1200, 900, 1600],
));
children.push(caption("Table 2. The evaluation clips. Only the four development clips were used to choose the global configuration."));

children.push(H2("5.1 Annotation protocol"));
children.push(P(
  "Ground truth for each clip is a list of step intervals with start and end times; the boundaries used by " +
  "the evaluator are the interior step end-times. Because the measured score is bounded above by the " +
  "fidelity of this reference (§7.1), annotation is treated as a first-class part of the method rather than " +
  "an incidental step. A written protocol (ANNOTATION_PROTOCOL.md) defines what does and does not " +
  "constitute a step boundary — a boundary is marked at the first frame in which the hands are committed to " +
  "a new goal; re-grips within an operation, camera movement, brief within-step pauses, and editing cuts " +
  "are explicitly not boundaries — together with tie-breaking rules for gradual transitions."
));
children.push(P("The protocol is supported by tooling in annotate.py:"));
[
  "contact-sheet — renders timestamped frame grids so transitions are read off a grid rather than by scrubbing a player; a coarse pass at 1.0 s locates each transition and a fine pass at 0.1 s localises it.",
  "review — an interactive frame-stepping player that marks boundaries and writes a correctly structured ground-truth file directly.",
  "validate — rejects the defects that silently corrupt evaluation: placeholder or draft-marked annotator fields, gaps or overlaps between steps, coverage that does not span the video, and the signature defect of an unadjusted template, namely a majority of boundaries lying on exact whole seconds.",
  "agreement — computes inter-annotator agreement between two independent annotations of the same clip, which is the ceiling on what any automatic method can meaningfully score.",
].forEach((t) => children.push(bullet(t)));
children.push(P(
  "Applying the validator to the annotations produced during development revealed that four of the five " +
  "files fail it: their annotator field is a draft marker and every interior boundary lies on a whole or " +
  "half second, which is the fingerprint of a template that was never adjusted to the footage. These files " +
  "are flagged for re-annotation in ANNOTATION_PROTOCOL.md §4. This is disclosed explicitly because it " +
  "bounds the confidence that can be placed in the per-clip numbers for those clips, and it is discussed " +
  "further in §7.1 and §7.4. The cooling-fan annotation used for the headline clean-clip result does not " +
  "exhibit this defect."
));

// ---- 6 Evaluation
children.push(H1("6. Evaluation"));
children.push(H2("6.1 Metrics"));
children.push(P(
  "Predicted boundaries are compared to ground-truth boundaries using precision, recall, and F1 at temporal " +
  "tolerances of 0.5, 1.0, 1.5, 2.0, and 3.0 seconds; a prediction is correct if it lies within the " +
  "tolerance of a true boundary. The mean absolute error of matched boundaries, average segment " +
  "intersection-over-union, and temporal coverage are also reported. Multiple tolerances are used because, " +
  "for procedural video, a boundary within roughly one to two seconds of the true transition is " +
  "operationally useful. Matching of predicted to ground-truth boundaries is computed optimally as a " +
  "one-to-one assignment (the Hungarian algorithm), so the reported F1 and the matched-boundary error are " +
  "mutually consistent and independent of the order in which boundaries are considered."
));

children.push(H2("6.2 Results under a single global configuration"));
children.push(P(
  "To remove any dependence on per-clip tuning, one configuration — boundary threshold 0.70, minimum " +
  "segment duration 2.0 s — was selected by grid search over the four development clips and applied " +
  "unchanged to every clip. These are the headline results of the project."
));
children.push(makeTable(
  ["Clip", "Type", "F1 @1.0s", "F1 @3.0s", "In tuning set"],
  [
    ["Cooling fan", "Clean procedure", "0.727", "0.727", "yes"],
    ["CPU placement", "Clean procedure", "0.714", "0.857", "yes"],
    ["RAM install", "Edited tutorial", "0.316", "0.421", "yes"],
    ["Cable connect", "Edited tutorial", "0.167", "0.500", "yes"],
    ["Intel CPU install", "Clean procedure", "0.286", "0.571", "no — held out"],
  ],
  [2100, 1900, 1300, 1300, 1700],
));
children.push(caption("Table 3. Boundary-detection F1 under the single global configuration (threshold 0.70, minimum duration 2.0 s). No per-clip tuning."));
children.push(P(
  "On clean development procedures the system localises matched boundaries to within 0.1–0.4 s on average. " +
  "Performance drops sharply on the two edited tutorials, for the reasons analysed in §7.2."
));

children.push(...figure("boundary_alignment.png", 620, 578,
  "Figure 2. Predicted (dashed red) versus annotated (solid green) boundaries for all five clips on the " +
  "smoothed boundary score, under the single global configuration; green shading marks a matched pair " +
  "within 1.0 s. On the clean clips predictions align tightly with annotations. On the edited tutorials " +
  "the score saturates just above the threshold for long stretches, producing many crossings — the direct " +
  "cause of over-segmentation on that footage."));

children.push(H2("6.3 Leave-one-clip-out cross-validation"));
children.push(P(
  "To confirm the global configuration is not merely fitted to the four development clips, the threshold " +
  "and minimum duration were re-selected on three clips and evaluated on the fourth, for each choice of " +
  "held-out clip."
));
children.push(makeTable(
  ["Held-out clip", "Selected threshold", "Selected min. duration", "F1 @1.0s", "F1 @3.0s"],
  [
    ["Cooling fan", "0.70", "2.0 s", "0.727", "0.727"],
    ["CPU placement", "0.65", "2.0 s", "0.588", "0.706"],
    ["RAM install", "0.65", "3.5 s", "0.133", "0.533"],
    ["Cable connect", "0.70", "2.0 s", "0.167", "0.500"],
  ],
  [2000, 1900, 2000, 1200, 1200],
));
children.push(caption("Table 4. Leave-one-clip-out cross-validation. Clean-clip held-out mean F1 @1.0s = 0.658."));
children.push(P(
  "The clean-clip held-out mean of 0.658 is close to the 0.721 obtained when tuning on all four clips, " +
  "indicating limited overfitting of the two hyper-parameters. The selected threshold is stable at " +
  "0.65–0.70 across folds."
));

children.push(H2("6.4 Held-out clip"));
children.push(P(
  "The Intel CPU installation clip was filmed and annotated after the configuration was frozen and took no " +
  "part in tuning at any stage, making it the only fully untainted generalisation estimate in the project. " +
  "Scored with the frozen configuration it reaches F1 0.286 at 1.0 s and 0.571 at 2.0 s — the weakest " +
  "result in the project. It is reported rather than omitted, and the cause was investigated rather than " +
  "assumed."
));
children.push(P(
  "The initial hypothesis — that the opening unboxing phase, which the annotation treats as a single step, " +
  "accounts for the error — was tested and rejected: re-scoring with the first 10 s removed lowers F1 " +
  "further, to 0.182. What the data actually shows is over-segmentation across the whole clip (9 predicted " +
  "boundaries against 5 annotated, a ratio of 1.8) combined with probable annotation error: of the three " +
  "unmatched references, two lie 1.8 s from a prediction, and this clip's ground truth is one of the files " +
  "that fails the validator, with every boundary on a whole or half second. The honest conclusion is that " +
  "this number bounds performance from below and should be re-measured once the clip is re-annotated under " +
  "the protocol in §5.1."
));

children.push(H2("6.5 Ablation studies"));
children.push(P(
  "Two ablations were run at different levels of the pipeline. The first disables an entire feature " +
  "extractor and re-runs the vision pipeline; the second removes a channel from the boundary-score fusion " +
  "and replays the saved per-frame features."
));
children.push(makeTable(
  ["Pipeline configuration", "Segments", "F1 @1.0s", "F1 @3.0s"],
  [
    ["Full (optical flow + scene change)", "10", "0.625", "0.875"],
    ["Without optical flow", "1", "0.000", "0.000"],
    ["Without scene-change detection", "9", "0.533", "0.933"],
  ],
  [3400, 1400, 1500, 1500],
));
children.push(caption("Table 5. Pipeline-level ablation on the cooling-fan clip (per-clip configuration, threshold 0.55)."));
children.push(P(
  "Optical flow is the dominant signal: without it the transition score never crosses the threshold and the " +
  "video collapses into a single segment. Scene-change detection is a useful secondary cue, its removal " +
  "lowering F1 from 0.625 to 0.533 at 1.0 s while coarse structure is preserved. This justifies the " +
  "multi-feature design."
));
children.push(makeTable(
  ["Fusion channel removed", "Mean F1 @1.0s (clean clips)"],
  [
    ["None (full fusion)", "0.721"],
    ["transition", "0.000"],
    ["activity_change", "0.607"],
  ],
  [4400, 3000],
));
children.push(caption("Table 6. Fusion-level ablation over the two retained channels."));
children.push(P(
  "The fusion originally combined six channels. Sweeping all 192 combinations of clip, threshold, and " +
  "minimum duration showed that four of them — the frame-to-frame deltas of flow magnitude, hand count, " +
  "interaction count, and tool count — never improved F1 on any configuration, produced an identical result " +
  "on 150 of 192, and actively degraded it on 11 of the 42 configurations where they made any difference " +
  "(mean F1 over the differing configurations: 0.633 with them, 0.644 without; they won in zero cases). The " +
  "explanation is structural: because fusion takes the maximum over channels, an extra channel can only " +
  "raise the score, and these four raised it mainly at frames that are not step boundaries, injecting " +
  "low-magnitude spurious peaks that survive at lower thresholds. They were therefore removed, simplifying " +
  "the method with no loss."
));
children.push(P(
  "This is not in tension with optical flow being the dominant cue. Flow enters the pipeline upstream, " +
  "through the feature extractor, which folds flow discontinuity, direction change, and uniformity into the " +
  "transition score. What the ablation removed is the redundant re-derivation of a flow delta at the fusion " +
  "stage, not optical flow itself — as the pipeline-level ablation in Table 5 confirms."
));

children.push(H2("6.6 Comparison with baselines"));
children.push(P(
  "To show the method does more than split the video arbitrarily, it is compared under the identical metric " +
  "with three baselines: a uniform split using the true number of boundaries (oracle K), random boundaries " +
  "respecting a minimum gap and averaged over 300 trials, and a dynamic-programming change-point detector " +
  "(ruptures, l2 cost) also given the true number of boundaries."
));
children.push(makeTable(
  ["Method", "Fan @1.0s", "Fan @3.0s", "CPU @1.0s", "CPU @3.0s"],
  [
    ["Full method", "0.727", "0.727", "0.714", "0.857"],
    ["Change-point (oracle K)", "0.143", "0.714", "0.500", "0.667"],
    ["Uniform (oracle K)", "0.286", "0.857", "0.333", "0.833"],
    ["Random (300-trial mean)", "0.229", "0.601", "0.290", "0.685"],
  ],
  [2600, 1500, 1500, 1500, 1500],
));
children.push(caption("Table 7. Method versus baselines on the clean clips, under the global configuration."));
children.push(P(
  "At the tight 1.0 s tolerance the method clearly outperforms every baseline on both clean clips, " +
  "including a change-point detector handed the correct number of boundaries. At the loose 3.0 s tolerance " +
  "an evenly-spaced split sometimes catches up, because that much slack lets arbitrary points fall near a " +
  "true boundary — which is precisely why tight tolerances are the meaningful comparison."
));
children.push(...figure("baseline_comparison.png", 460, 307,
  "Figure 3. Method versus baselines at the 1.0 s tolerance on the clean clips."));

children.push(H2("6.7 Learned boundary detectors and the hybrid scorer"));
children.push(P(
  "Section 2.1 argues on a priori grounds that supervised temporal action segmentation is " +
  "inapplicable to this project. That argument was tested empirically rather than left as an " +
  "assertion, which is inexpensive because the training data already exists: every pipeline run " +
  "saves a per-frame feature CSV and every clip is annotated, giving 3,396 labelled frames across " +
  "five clips with no additional annotation."
));
children.push(P(
  "A frame is labelled positive if it lies within 0.5 s of an annotated boundary (373 frames, " +
  "11.0% of the data). Each frame is represented by its 23 primitive features at offsets " +
  "{-4, -2, 0, +2, +4} plus local first differences, giving a receptive field of about one second. " +
  "Three classifiers were trained — logistic regression, gradient boosting, and a two-layer neural " +
  "network. Crucially, the predicted probability is passed through the identical smoothing, " +
  "peak-finding and filtering used by the rule-based method (peaks_from_score), so the two " +
  "approaches differ only in how the per-frame score is produced. The protocol is leave-one-clip-out " +
  "with fully nested selection: the model, the feature scaling, the peak threshold, the minimum " +
  "duration and the blend weight are all fitted on the four training clips only."
));
children.push(P(
  "Two feature sets were compared: raw excludes the hand-designed composites (transition score " +
  "and activity level) so the model cannot read the answer off the existing fusion, while all " +
  "includes them."
));
children.push(makeTable(
  ["Scorer", "Mean F1 @1.0s", "Worst clip", "Std across clips"],
  [
    ["Hybrid (rule + logistic regression)", "0.476", "0.316", "0.098"],
    ["Logistic regression (raw features)", "0.435", "0.211", "0.142"],
    ["Rule-based fusion (hand-designed)", "0.405", "0.133", "0.262"],
    ["Logistic regression (all features)", "0.386", "—", "—"],
    ["Gradient boosting (raw features)", "0.360", "—", "—"],
    ["Neural network (all features)", "0.315", "—", "—"],
    ["Neural network (raw features)", "0.294", "—", "—"],
    ["Gradient boosting (all features)", "0.268", "—", "—"],
  ],
  [3600, 1600, 1400, 1700],
));
children.push(caption("Table 8. Leave-one-clip-out comparison of learned and hand-designed boundary scorers over all five clips, under identical post-processing. Produced with scikit-learn 1.9.0; see the note on version sensitivity below."));
children.push(P(
  "Three findings, in order of importance. First, no learned model significantly outperforms the " +
  "hand-designed fusion on mean F1: the best margin is +0.030 with a paired t-test p of 0.76, which " +
  "at five clips is indistinguishable from noise. Second, the two approaches fail in opposite " +
  "directions — on the two clean clips the rule-based fusion scores 0.720 against the learned 0.544, " +
  "while on the three harder clips the ordering reverses, 0.195 against 0.363. The hand-designed " +
  "cues are better precisely on the footage they were designed for. Third, model capacity hurts: the " +
  "simplest model, logistic regression, is the best of the three, and both higher-capacity models — " +
  "gradient boosting and the neural network — rank below it. Every model also scores worse when the " +
  "hand-designed composites are supplied as extra inputs. Both observations are the expected " +
  "signature of too little data, and together they convert the argument of Section 2.1 from an " +
  "assertion into a measured result."
));
children.push(P(
  "One reproducibility caveat should be recorded. Repeating the experiment under scikit-learn 1.7.2 " +
  "rather than 1.9.0 leaves the logistic-regression, neural-network, hybrid and rule-based figures " +
  "bit-identical, but moves gradient boosting: 0.360 becomes 0.356 on raw features and 0.268 becomes " +
  "0.306 on the full feature set — enough to change which model ranks last. The gradient-boosting " +
  "implementation changed between those releases. None of the conclusions depend on it: logistic " +
  "regression is the best learned model and the hybrid the best scorer under either version. The " +
  "library is nevertheless pinned in requirements.txt and the version of record is stated in the " +
  "caption, because an unpinned dependency silently changing a reported number is precisely the " +
  "failure mode the pinning policy of Section 4.1 exists to prevent."
));
children.push(P(
  "Because the two scorers are complementary, they were combined: the hybrid score is a convex blend " +
  "w·(rule) + (1−w)·(learned), with w selected inside the same nested cross-validation. The hybrid " +
  "attains the best mean F1 (0.476), and more importantly it roughly doubles the worst-case clip " +
  "score (0.316 against 0.133) and reduces the spread across clips by a factor of 2.7 (std 0.098 " +
  "against 0.262). The mean improvement is not statistically significant at five clips (p = 0.50), " +
  "so the defensible claim is about robustness rather than peak accuracy — and robustness across " +
  "footage types is exactly the weakness identified in Section 7.2. The hybrid is available in the " +
  "pipeline as --scorer hybrid, with the trained model produced by train_boundary_model.py; the " +
  "default remains the rule-based scorer, which is still the best choice on clean footage."
));
children.push(...figure("learned_vs_rulebased.png", 600, 280,
  "Figure 4. Per-clip leave-one-clip-out F1 for the hand-designed fusion and each learned scorer. " +
  "The rule-based method peaks highest on the clean clips (cooling fan, CPU) and collapses on the " +
  "rest; the learned and hybrid scorers are flatter across footage types."));

children.push(H2("6.8 Sensitivity and annotation robustness"));
children.push(P(
  "F1 was swept across boundary thresholds from 0.45 to 0.80. Performance on the clean clips rises to a " +
  "maximum of 0.721 at the selected threshold of 0.70 and collapses to zero at 0.75 and above, where the " +
  "score no longer crosses the threshold at all. The operating point therefore sits at the top of a broad " +
  "plateau but close to a cliff; this is a genuine fragility of the method and is noted in §8."
));
children.push(...figure("sensitivity_threshold.png", 460, 307,
  "Figure 5. Sensitivity of mean F1 @1.0s to the boundary threshold (minimum duration 2.0 s)."));
children.push(P(
  "As a quantitative stand-in for inter-annotator variation, the ground-truth boundaries were perturbed " +
  "with Gaussian jitter and the frozen predictions re-scored over 300 trials. Under 0.25 s of jitter the " +
  "cooling-fan F1 is unchanged at 0.727 and the CPU F1 falls only from 0.714 to 0.688; under 0.5 s they " +
  "fall to 0.690 and 0.614 respectively. The clean-clip results are therefore stable against annotation " +
  "noise of the order of a quarter-second, but not against the multi-second uncertainty implied by the " +
  "unvalidated template files."
));

// ---- 7 Discussion
children.push(H1("7. Discussion"));
children.push(H2("7.1 Ground-truth quality dominates the measured score"));
children.push(P(
  "Scored against the unadjusted CPU template, the system reached only F1 0.462; after the CPU clip was " +
  "annotated from the actual video, the identical segmentation rose to F1 0.615 at 1.0 s and 0.923 at " +
  "3.0 s. A large apparent gain came purely from correcting the reference, with no change to the system. " +
  "Boundary metrics are only meaningful against faithfully annotated ground truth."
));
children.push(P(
  "This finding is the reason the annotation protocol, tooling, and validator of §5.1 exist, and it cuts " +
  "both ways: it also means the numbers reported here for the clips whose annotations still fail the " +
  "validator are themselves uncertain. Rather than quietly re-annotating until scores improved, the " +
  "granularity rule was fixed in writing first and the validator was then applied uniformly, including to " +
  "files whose failure is inconvenient. Re-annotating the four flagged files, and producing a second " +
  "independent annotation of at least one clip to obtain an inter-annotator agreement ceiling, is the " +
  "highest-value remaining task on the project."
));
children.push(H2("7.2 Clean procedures versus edited tutorials"));
children.push(P(
  "The method assumes continuous first-person footage. The RAM and cable clips violate this assumption: " +
  "their scene-change cues fire on video edits — cuts to a presenter or a diagram — that do not correspond " +
  "to assembly steps, producing extra and misaligned boundaries. Figure 2 shows the mechanism directly: on " +
  "these clips the boundary score sits just above the detection threshold for long stretches rather than " +
  "forming isolated peaks, so many crossings are produced. The boundaries that do match remain well " +
  "localised, confirming the issue lies in the source material rather than in the peak-detection logic."
));
children.push(H2("7.3 Object detection: a documented negative result"));
children.push(P(
  "Zero-shot open-vocabulary detection could not reliably recognise small assembly parts in this close-up, " +
  "hand-occluded footage; at a confidence floor of 0.01 the strongest score for any real part across sixty " +
  "frames was 0.024. Larger components were occasionally detected (the CPU as “processor”, and “header” and " +
  "“motherboard” on the cable clip) but not consistently. As recognition is out of scope this does not " +
  "affect the segmentation goal, but it shows that reliable component labelling would require a detector " +
  "trained on the project's own footage."
));
children.push(H2("7.4 Activity labelling: a second documented negative result"));
children.push(P(
  "The pipeline attaches a human-readable label to each detected segment " +
  "(activity_recognizer.py). Measured against the annotated step labels with a keyword-overlap " +
  "score, this component does not work: it scores 0.000 on the cooling-fan, RAM and cable clips, " +
  "0.016 on CPU placement and 0.039 on the held-out Intel clip. In practice it emits generic " +
  "phrases such as “Unspecified hand activity” and “Unspecified adjustment”, because the " +
  "component-specific branch depends on the zero-shot detector that Section 7.3 shows cannot " +
  "identify small hardware parts. With no component recognised, the labeller falls back to a " +
  "hand-motion phase description that carries almost no task-specific information."
));
children.push(P(
  "This is reported rather than quietly left in the output for two reasons. It is out of scope — " +
  "the brief states that the actions need not be detected or recognised, and segmentation metrics " +
  "are entirely unaffected by it — but it is nonetheless a shipped component that does not achieve " +
  "its stated purpose, and a reader inspecting the output video would notice. The score is therefore " +
  "surfaced automatically in extended_results.json alongside the segmentation metrics. Making " +
  "labelling work would require the same custom-trained detector identified in Section 7.3, and is " +
  "listed under future work rather than claimed."
));
children.push(H2("7.5 What the held-out clip teaches"));
children.push(P(
  "The held-out result in §6.4 is the most useful single number in the report precisely because it is the " +
  "worst. It shows that the two clean development clips, on which the configuration was chosen, are not by " +
  "themselves sufficient evidence of generalisation, and that the dominant failure mode on unseen clean " +
  "footage is over-segmentation rather than missed boundaries. Combined with the threshold-sensitivity " +
  "cliff in Section 6.8, this suggests the single global threshold is the method's main structural weakness: it is " +
  "a fixed cut-point on a signal whose scale varies between clips. An adaptive or per-clip-normalised " +
  "threshold is the most promising direction for future work."
));

// ---- 8 Limitations
children.push(H1("8. Limitations"));
[
  "The evaluation set is small: five clips, of which three are clean continuous procedures. All headline claims rest on a correspondingly small sample, and the comparisons in Section 6.7 cannot reach statistical significance at this size.",
  "Four of the five ground-truth files currently fail the project's own validator and are pending re-annotation; the numbers for those clips are correspondingly uncertain.",
  "No inter-annotator agreement has yet been measured, so the ceiling on achievable F1 against this ground truth is unknown.",
  "The activity labeller does not work, scoring 0.000–0.039 against annotated labels (Section 7.4). It is retained but out of scope.",
  "The rule-based scorer does not improve with more data. Its roughly forty thresholds and weights are hand-set, so a larger dataset would require re-tuning rather than simply retraining — one of the reasons the learned scorer of Section 6.7 was added.",
  "The method is not causal: score normalisation uses the 5th and 95th percentiles of the entire video, so it cannot run as a live stream without a windowed reformulation.",
  "Performance is sensitive to the boundary threshold, collapsing to zero above 0.75 (Section 6.8).",
  "Subtle, low-motion transitions can produce weak signals and be missed.",
  "Edited, multi-shot footage is unsuitable for feature-based procedural segmentation; the hybrid scorer reduces but does not eliminate this.",
  "Zero-shot detection cannot identify small hardware parts; component labels need a trained detector.",
  "Inference is CPU-bound at roughly 1.5 fps; a GPU greatly speeds it up.",
].forEach((t) => children.push(bullet(t)));

// ---- 9 Conclusion
children.push(H1("9. Conclusion and Future Work"));
children.push(P(
  "The system meets the project objective on clean egocentric procedures: it locates step boundaries " +
  "accurately and precisely using only motion, hand, flow, and scene features, with no dependence on action " +
  "recognition, no per-frame labels, and no training. Under a single global configuration with no per-clip " +
  "tuning it reaches F1 0.727 and 0.714 at a one-second tolerance on the two clean development clips, " +
  "outperforms uniform, random, and change-point baselines at the tolerances that matter, and generalises " +
  "with a clean held-out cross-validation mean of 0.658. A fifth, fully held-out clip scores 0.286, which " +
  "is reported openly and diagnosed."
));
children.push(P(
  "Beyond the working system, the project contributes a reusable methodology: a written annotation protocol " +
  "with tooling and an automatic validator that catches the exact defect — unadjusted template annotations " +
  "— that was silently distorting this project's own early results; a two-level ablation discipline that " +
  "identified and removed four redundant fusion channels; and an empirical rather than assumed answer to " +
  "whether a learned model should replace the hand-designed one."
));
children.push(P("Future work, in order of expected value:"));
[
  "Re-annotate the four flagged clips under the protocol and measure inter-annotator agreement, so that every reported number has a known confidence bound.",
  "Collect substantially more clips. Every limitation above traces back to a sample of five, and the learned scorer of Section 6.7 would benefit most directly — it is currently data-starved, which is why the simplest model wins.",
  "Replace the fixed global boundary threshold with an adaptive or per-clip-normalised criterion, addressing both the sensitivity cliff and the over-segmentation seen on the held-out clip.",
  "Reformulate normalisation over a sliding window to make the method causal and therefore usable on a live stream.",
  "Train a lightweight custom detector on project footage, which would fix both component labelling (Section 7.4) and the detection negative result (Section 7.3).",
  "With a larger dataset, revisit end-to-end video models such as VideoMAE [10], which were ruled out here on data grounds rather than on principle.",
  "GPU acceleration to enable larger experiments.",
].forEach((t) => children.push(bullet(t)));

// ---- References
children.push(H1("References"));
[
  "F. Zhang et al., “MediaPipe Hands: On-device Real-time Hand Tracking,” arXiv:2006.10214, 2020.",
  "T. Cheng et al., “YOLO-World: Real-Time Open-Vocabulary Object Detection,” IEEE/CVF CVPR, 2024.",
  "G. Farnebäck, “Two-Frame Motion Estimation Based on Polynomial Expansion,” Scandinavian Conf. on Image Analysis (SCIA), 2003.",
  "G. Bradski, “The OpenCV Library,” Dr. Dobb's Journal of Software Tools, 2000.",
  "P. Virtanen et al., “SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python,” Nature Methods, 2020.",
  "G. Jocher et al., “Ultralytics YOLO,” open-source software, 2023.",
  "Y. Abu Farha and J. Gall, “MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation,” IEEE/CVF CVPR, 2019.",
  "D. Damen et al., “Scaling Egocentric Vision: The EPIC-KITCHENS Dataset,” European Conf. on Computer Vision (ECCV), 2018.",
  "D. Shan, J. Geng, M. Shu, and D. F. Fouhey, “Understanding Human Hands in Contact at Internet Scale,” IEEE/CVF CVPR, 2020.",
  "C. Truong, L. Oudre, and N. Vayatis, “Selective review of offline change point detection methods,” Signal Processing, vol. 167, 2020. (ruptures)",
  "Z. Tong, Y. Song, J. Wang, and L. Wang, “VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training,” Advances in Neural Information Processing Systems (NeurIPS), 2022.",
  "F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” Journal of Machine Learning Research, vol. 12, 2011.",
].forEach((r, i) => children.push(new Paragraph({
  spacing: { after: 90, line: 264 },
  children: [new TextRun({ text: `[${i + 1}]  ${r}`, size: 21 })],
})));

// ---- Appendix
children.push(H1("Appendix A. Reproduction"));
children.push(P("A representative run on the cooling-fan clip under the global configuration:"));
children.push(new Paragraph({
  spacing: { after: 160 },
  shading: { type: ShadingType.CLEAR, fill: "F2F2F2", color: "auto" },
  children: [new TextRun({
    text: "python main.py --video Coolingfaninstallation.mp4 --output results_coolingfan " +
      "--fps 10 --detector open_vocab --model yolov8l-worldv2.pt --resize 960x540 " +
      "--open-vocab-imgsz 960 --threshold 0.70 --min-duration 2.0 --grip-window 7 " +
      "--object-confidence 0.10 --ground-truth ground_truth_coolingfan_v2.json",
    font: "Consolas", size: 18,
  })],
}));
children.push(P(
  "Every number and figure in §6 is regenerated from the saved per-frame features, with no GPU and no " +
  "re-run of the vision pipeline, by:"
));
children.push(new Paragraph({
  spacing: { after: 160 },
  shading: { type: ShadingType.CLEAR, fill: "F2F2F2", color: "auto" },
  children: [new TextRun({ text: "python evaluate_extended.py --src .", font: "Consolas", size: 18 })],
}));
children.push(P(
  "This script also asserts that re-segmenting the saved features reproduces the saved boundaries exactly, " +
  "so any drift in the boundary logic fails the run. Pipeline-level ablations use the --no-flow and " +
  "--no-scene flags (Section 6.5). The learned and hybrid scorers of Section 6.7 are reproduced with:"
));
children.push(new Paragraph({
  spacing: { after: 160 },
  shading: { type: ShadingType.CLEAR, fill: "F2F2F2", color: "auto" },
  children: [new TextRun({
    text: "pip install scikit-learn\n"
      + "python learned_baseline.py --src .        # held-out comparison (Table 8)\n"
      + "python train_boundary_model.py --src .    # writes boundary_model.joblib\n"
      + "python main.py --video clip.mp4 --scorer hybrid ...",
    font: "Consolas", size: 18,
  })],
}));
children.push(P("Ground-truth files are checked with:"));
children.push(new Paragraph({
  spacing: { after: 160 },
  shading: { type: ShadingType.CLEAR, fill: "F2F2F2", color: "auto" },
  children: [new TextRun({
    text: "python annotate.py validate --ground-truth ground_truth_coolingfan_v2.json " +
      "--video Coolingfaninstallation.mp4",
    font: "Consolas", size: 18,
  })],
}));

// ---------------------------------------------------------------- build
const doc = new Document({
  creator: "Osama Najjar, Mostafa Taha",
  title: "Splitting a Video into Actions — Final Project Report",
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, color: "1F4E79", font: "Calibri" },
        paragraph: { spacing: { before: 320, after: 160 } },
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, color: "2E74B5", font: "Calibri" },
        paragraph: { spacing: { before: 240, after: 120 } },
      },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1080, bottom: 1080, left: 1180, right: 1180 },
      },
    },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log("wrote", OUT, buf.length, "bytes");
});
