import json
import numpy as np
from typing import List
from utils import MetricsCalculator
from temporal_segmenter import ActionSegment, Boundary


class Evaluator:
    """Evaluate segmentation against ground truth annotations."""

    def __init__(self, ground_truth_path: str):
        with open(ground_truth_path) as f:
            self.gt = json.load(f)

        self.gt_boundaries = [b['timestamp'] for b in self.gt['boundaries']]
        self.gt_segments = [
            (s['start_time'], s['end_time']) for s in self.gt['segments']
        ]

    def evaluate(self, segments: List[ActionSegment],
                 boundaries: List[Boundary]) -> dict:
        """Run full evaluation."""
        pred_boundaries = [b.timestamp for b in boundaries]
        pred_segments = [(s.start_time, s.end_time) for s in segments]

        boundary_metrics = MetricsCalculator.boundary_accuracy(
            pred_boundaries, self.gt_boundaries, tolerance=1.0
        )
        segment_iou = MetricsCalculator.segment_iou(
            pred_segments, self.gt_segments
        )
        over_seg = len(segments) / len(self.gt_segments) if self.gt_segments else 0
        under_seg = len(self.gt_segments) / len(segments) if segments else 0

        return {
            "boundary_metrics": boundary_metrics,
            "segment_iou": segment_iou,
            "over_segmentation_ratio": over_seg,
            "under_segmentation_ratio": under_seg,
            "num_predicted": len(segments),
            "num_ground_truth": len(self.gt_segments)
        }
