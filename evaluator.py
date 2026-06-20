import json
import numpy as np
from typing import List, Optional, Tuple

from utils import MetricsCalculator
from temporal_segmenter import ActionSegment, Boundary


class Evaluator:
    """
    Evaluate step-boundary and segment quality against manually annotated ground truth.

    Supports two JSON formats:

    1. Original format:
       {
         "segments": [
           {"start_time": 0, "end_time": 4, "activity": "..."}
         ],
         "boundaries": [
           {"timestamp": 4.0}
         ]
       }

    2. Simpler annotation format:
       {
         "steps": [
           {"start": 0, "end": 4, "label": "..."}
         ]
       }
    """

    def __init__(self, ground_truth_path: str):
        with open(ground_truth_path, encoding="utf-8") as f:
            self.gt = json.load(f)

        self.gt_segments, self.gt_labels = self._load_segments(self.gt)
        self.gt_boundaries = self._load_boundaries(self.gt, self.gt_segments)

    @staticmethod
    def _load_segments(gt: dict) -> Tuple[List[Tuple[float, float]], List[str]]:
        if "steps" in gt:
            segments = [
                (float(s["start"]), float(s["end"]))
                for s in gt.get("steps", [])
            ]

            labels = [
                s.get("label", s.get("activity", "unknown"))
                for s in gt.get("steps", [])
            ]

            return segments, labels

        segments = [
            (float(s["start_time"]), float(s["end_time"]))
            for s in gt.get("segments", [])
        ]

        labels = [
            s.get("activity", s.get("label", "unknown"))
            for s in gt.get("segments", [])
        ]

        return segments, labels

    @staticmethod
    def _load_boundaries(gt: dict, segments: List[Tuple[float, float]]) -> List[float]:
        if "boundaries" in gt and gt["boundaries"]:
            return [
                float(b["timestamp"])
                for b in gt.get("boundaries", [])
            ]

        # Derive boundaries from segment/step ends except final segment.
        return [float(end) for _, end in segments[:-1]]

    def evaluate(
        self,
        segments: List[ActionSegment],
        boundaries: List[Boundary],
        tolerance: float = 1.0,
    ) -> dict:
        pred_boundaries = [
            float(b.timestamp)
            for b in boundaries
        ]

        pred_segments = [
            (float(s.start_time), float(s.end_time))
            for s in segments
        ]

        pred_labels = [
            s.activity_description or s.dominant_activity
            for s in segments
        ]

        boundary_metrics = MetricsCalculator.boundary_accuracy(
            pred_boundaries,
            self.gt_boundaries,
            tolerance=tolerance,
        )

        segment_iou = MetricsCalculator.segment_iou(
            pred_segments,
            self.gt_segments,
        )

        over_seg = len(segments) / len(self.gt_segments) if self.gt_segments else 0
        under_seg = len(self.gt_segments) / len(segments) if segments else 0

        boundary_offset = self._compute_boundary_offset(pred_boundaries)
        coverage = self._compute_coverage(pred_segments)

        matched_offsets = self._matched_boundary_offsets(
            pred_boundaries,
            tolerance,
        )

        mean_abs_error = float(np.mean(matched_offsets)) if matched_offsets else 0.0

        label_score = self._rough_activity_label_score(pred_labels)

        return {
            "boundary_metrics": boundary_metrics,
            "segment_iou": round(float(segment_iou), 3),
            "over_segmentation_ratio": round(over_seg, 3),
            "under_segmentation_ratio": round(under_seg, 3),
            "avg_boundary_offset_seconds": round(boundary_offset, 3),
            "matched_boundary_mae_seconds": round(mean_abs_error, 3),
            "coverage_ratio": round(coverage, 3),
            "rough_activity_label_score": round(label_score, 3),
            "num_predicted": len(segments),
            "num_ground_truth": len(self.gt_segments),
            "num_pred_boundaries": len(pred_boundaries),
            "num_gt_boundaries": len(self.gt_boundaries),
        }

    def _compute_boundary_offset(self, predicted: List[float]) -> float:
        if not predicted or not self.gt_boundaries:
            return 0.0

        offsets = [
            min(abs(pred - gt) for gt in self.gt_boundaries)
            for pred in predicted
        ]

        return float(np.mean(offsets))

    def _matched_boundary_offsets(
        self,
        predicted: List[float],
        tolerance: float,
    ) -> List[float]:
        """Greedy one-to-one boundary matches within tolerance."""
        remaining_gt = list(self.gt_boundaries)
        offsets = []

        for pred in sorted(predicted):
            if not remaining_gt:
                break

            closest = min(
                remaining_gt,
                key=lambda gt: abs(pred - gt),
            )

            offset = abs(pred - closest)

            if offset <= tolerance:
                offsets.append(offset)
                remaining_gt.remove(closest)

        return offsets

    def _compute_coverage(
        self,
        predicted_segments: List[Tuple[float, float]],
    ) -> float:
        if not self.gt_segments or not predicted_segments:
            return 0.0

        total_gt_duration = sum(
            end - start
            for start, end in self.gt_segments
        )

        covered = 0.0

        for gt_start, gt_end in self.gt_segments:
            best_overlap = 0.0

            for pred_start, pred_end in predicted_segments:
                overlap_start = max(gt_start, pred_start)
                overlap_end = min(gt_end, pred_end)

                best_overlap = max(
                    best_overlap,
                    max(0, overlap_end - overlap_start),
                )

            covered += best_overlap

        return covered / total_gt_duration if total_gt_duration > 0 else 0.0

    def _rough_activity_label_score(self, predicted_labels: List[str]) -> float:
        """
        Simple keyword-overlap label score.

        This is useful for a report, but it is not a strict semantic metric.
        """
        if not predicted_labels or not self.gt_labels:
            return 0.0

        n = min(len(predicted_labels), len(self.gt_labels))

        scores = []

        stopwords = {
            "the",
            "or",
            "and",
            "to",
            "of",
            "in",
            "on",
            "area",
            "step",
        }

        for pred, gt in zip(predicted_labels[:n], self.gt_labels[:n]):
            pred_words = {
                w.lower().strip("/,_-")
                for w in pred.split()
            } - stopwords

            gt_words = {
                w.lower().strip("/,_-")
                for w in gt.split()
            } - stopwords

            if not pred_words or not gt_words:
                scores.append(0.0)
            else:
                scores.append(
                    len(pred_words & gt_words) / len(pred_words | gt_words)
                )

        return float(np.mean(scores)) if scores else 0.0

    def evaluate_at_tolerances(
        self,
        segments: List[ActionSegment],
        boundaries: List[Boundary],
        tolerances: Optional[List[float]] = None,
    ) -> dict:
        if tolerances is None:
            tolerances = [0.5, 1.0, 1.5, 2.0, 3.0]

        results = {}

        pred_boundaries = [
            b.timestamp
            for b in boundaries
        ]

        for tol in tolerances:
            metrics = MetricsCalculator.boundary_accuracy(
                pred_boundaries,
                self.gt_boundaries,
                tolerance=tol,
            )

            offsets = self._matched_boundary_offsets(
                pred_boundaries,
                tol,
            )

            metrics["matched_mae"] = (
                round(float(np.mean(offsets)), 3)
                if offsets
                else 0.0
            )

            results[f"tolerance_{tol}s"] = metrics

        return results

    def generate_report(
        self,
        segments: List[ActionSegment],
        boundaries: List[Boundary],
        output_path: Optional[str] = None,
    ) -> str:
        metrics = self.evaluate(segments, boundaries)
        multi_tol = self.evaluate_at_tolerances(segments, boundaries)

        lines = [
            "=" * 60,
            "SEGMENTATION EVALUATION REPORT",
            "=" * 60,
            "",
            f"Predicted segments: {metrics['num_predicted']}",
            f"Ground truth segments: {metrics['num_ground_truth']}",
            f"Predicted boundaries: {metrics['num_pred_boundaries']}",
            f"Ground truth boundaries: {metrics['num_gt_boundaries']}",
            f"Over-segmentation ratio: {metrics['over_segmentation_ratio']:.3f}",
            f"Under-segmentation ratio: {metrics['under_segmentation_ratio']:.3f}",
            "",
            "--- Boundary Detection ---",
            f"Precision: {metrics['boundary_metrics']['precision']:.3f}",
            f"Recall: {metrics['boundary_metrics']['recall']:.3f}",
            f"F1 Score: {metrics['boundary_metrics']['f1_score']:.3f}",
            f"Avg nearest-boundary offset: {metrics['avg_boundary_offset_seconds']:.3f}s",
            f"Matched boundary MAE: {metrics['matched_boundary_mae_seconds']:.3f}s",
            "",
            "--- Segment / Activity Quality ---",
            f"Average segment IoU: {metrics['segment_iou']:.3f}",
            f"Coverage: {metrics['coverage_ratio']:.3f}",
            f"Rough activity label score: {metrics['rough_activity_label_score']:.3f}",
            "",
            "--- Multi-Tolerance Results ---",
        ]

        for key, val in multi_tol.items():
            lines.append(
                f"  {key}: "
                f"P={val['precision']:.3f} "
                f"R={val['recall']:.3f} "
                f"F1={val['f1_score']:.3f} "
                f"MAE={val['matched_mae']:.3f}s"
            )

        lines.extend(["", "=" * 60])

        report = "\n".join(lines)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)

        return report


def create_ground_truth_template(num_segments: int, output_path: str):
    """Create a simple ground truth JSON template for manual annotation."""
    template = {
        "video": "path/to/video.mp4",
        "annotator": "",
        "steps": [
            {
                "id": i,
                "start": 0.0,
                "end": 0.0,
                "label": f"step_{i + 1}",
                "notes": "",
            }
            for i in range(num_segments)
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2)

    print(f"Ground truth template saved to: {output_path}")