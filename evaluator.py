import json
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
from utils import MetricsCalculator
from temporal_segmenter import ActionSegment, Boundary


class Evaluator:
    """Evaluate segmentation against ground truth annotations."""

    def __init__(self, ground_truth_path: str):
        with open(ground_truth_path) as f:
            self.gt = json.load(f)

        self.gt_boundaries = [b['timestamp'] for b in self.gt.get('boundaries', [])]
        self.gt_segments = [
            (s['start_time'], s['end_time']) for s in self.gt.get('segments', [])
        ]
        self.gt_labels = [
            s.get('activity', 'unknown') for s in self.gt.get('segments', [])
        ]

    def evaluate(self, segments: List[ActionSegment],
                 boundaries: List[Boundary],
                 tolerance: float = 1.0) -> dict:
        """Run full evaluation."""
        pred_boundaries = [b.timestamp for b in boundaries]
        pred_segments = [(s.start_time, s.end_time) for s in segments]

        boundary_metrics = MetricsCalculator.boundary_accuracy(
            pred_boundaries, self.gt_boundaries, tolerance=tolerance
        )
        segment_iou = MetricsCalculator.segment_iou(
            pred_segments, self.gt_segments
        )

        over_seg = len(segments) / len(self.gt_segments) if self.gt_segments else 0
        under_seg = len(self.gt_segments) / len(segments) if segments else 0

        boundary_offset = self._compute_boundary_offset(pred_boundaries)
        coverage = self._compute_coverage(pred_segments)

        return {
            "boundary_metrics": boundary_metrics,
            "segment_iou": segment_iou,
            "over_segmentation_ratio": round(over_seg, 3),
            "under_segmentation_ratio": round(under_seg, 3),
            "avg_boundary_offset_seconds": round(boundary_offset, 3),
            "coverage_ratio": round(coverage, 3),
            "num_predicted": len(segments),
            "num_ground_truth": len(self.gt_segments)
        }

    def _compute_boundary_offset(self, predicted: List[float]) -> float:
        """Average temporal offset of predicted boundaries from nearest GT."""
        if not predicted or not self.gt_boundaries:
            return 0.0

        offsets = []
        for pred in predicted:
            min_offset = min(abs(pred - gt) for gt in self.gt_boundaries)
            offsets.append(min_offset)

        return float(np.mean(offsets))

    def _compute_coverage(self, predicted_segments: List[Tuple[float, float]]) -> float:
        """How much of the GT segments are covered by predictions."""
        if not self.gt_segments or not predicted_segments:
            return 0.0

        total_gt_duration = sum(end - start for start, end in self.gt_segments)
        covered = 0.0

        for gt_start, gt_end in self.gt_segments:
            gt_duration = gt_end - gt_start
            best_overlap = 0.0

            for pred_start, pred_end in predicted_segments:
                overlap_start = max(gt_start, pred_start)
                overlap_end = min(gt_end, pred_end)
                overlap = max(0, overlap_end - overlap_start)
                best_overlap = max(best_overlap, overlap)

            covered += best_overlap

        return covered / total_gt_duration if total_gt_duration > 0 else 0.0

    def evaluate_at_tolerances(self, segments: List[ActionSegment],
                                boundaries: List[Boundary],
                                tolerances: Optional[List[float]] = None) -> dict:
        """Evaluate at multiple tolerance levels."""
        if tolerances is None:
            tolerances = [0.5, 1.0, 1.5, 2.0, 3.0]

        results = {}
        for tol in tolerances:
            pred_boundaries = [b.timestamp for b in boundaries]
            metrics = MetricsCalculator.boundary_accuracy(
                pred_boundaries, self.gt_boundaries, tolerance=tol
            )
            results[f"tolerance_{tol}s"] = metrics

        return results

    def generate_report(self, segments: List[ActionSegment],
                        boundaries: List[Boundary],
                        output_path: Optional[str] = None) -> str:
        """Generate a formatted evaluation report."""
        metrics = self.evaluate(segments, boundaries)
        multi_tol = self.evaluate_at_tolerances(segments, boundaries)

        lines = [
            "=" * 60,
            "SEGMENTATION EVALUATION REPORT",
            "=" * 60,
            "",
            f"Predicted segments: {metrics['num_predicted']}",
            f"Ground truth segments: {metrics['num_ground_truth']}",
            f"Over-segmentation ratio: {metrics['over_segmentation_ratio']:.3f}",
            f"Under-segmentation ratio: {metrics['under_segmentation_ratio']:.3f}",
            "",
            "--- Boundary Detection ---",
            f"Precision: {metrics['boundary_metrics']['precision']:.3f}",
            f"Recall: {metrics['boundary_metrics']['recall']:.3f}",
            f"F1 Score: {metrics['boundary_metrics']['f1_score']:.3f}",
            f"Avg boundary offset: {metrics['avg_boundary_offset_seconds']:.3f}s",
            "",
            "--- Segment Quality ---",
            f"Average IoU: {metrics['segment_iou']:.3f}",
            f"Coverage: {metrics['coverage_ratio']:.3f}",
            "",
            "--- Multi-Tolerance Results ---",
        ]

        for key, val in multi_tol.items():
            lines.append(f"  {key}: P={val['precision']:.3f} R={val['recall']:.3f} F1={val['f1_score']:.3f}")

        lines.extend(["", "=" * 60])
        report = "\n".join(lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)

        return report


def create_ground_truth_template(num_segments: int, output_path: str):
    """Create a ground truth JSON template for manual annotation."""
    template = {
        "video_path": "path/to/video.mp4",
        "annotator": "",
        "segments": [
            {
                "id": i,
                "start_time": 0.0,
                "end_time": 0.0,
                "activity": f"step_{i+1}",
                "description": ""
            }
            for i in range(num_segments)
        ],
        "boundaries": [
            {"timestamp": 0.0, "description": ""}
            for _ in range(num_segments - 1)
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)

    print(f"Ground truth template saved to: {output_path}")
