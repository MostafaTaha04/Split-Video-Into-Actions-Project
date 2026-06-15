import numpy as np
from typing import List, Tuple
import time


class FPSCounter:
    """Tracks processing FPS."""

    def __init__(self, window: int = 30):
        self.window = window
        self.timestamps = []

    def tick(self):
        self.timestamps.append(time.time())
        if len(self.timestamps) > self.window:
            self.timestamps.pop(0)

    def get_fps(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0.0


class SignalSmoother:
    """Applies various smoothing techniques to time series."""

    @staticmethod
    def exponential_moving_average(signal: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        result = np.zeros_like(signal)
        result[0] = signal[0]
        for i in range(1, len(signal)):
            result[i] = alpha * signal[i] + (1 - alpha) * result[i - 1]
        return result

    @staticmethod
    def median_filter(signal: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        from scipy.ndimage import median_filter as mf
        return mf(signal, size=kernel_size)

    @staticmethod
    def adaptive_threshold(signal: np.ndarray, window: int = 30,
                           multiplier: float = 1.5) -> np.ndarray:
        """Compute adaptive threshold based on local statistics."""
        thresholds = np.zeros_like(signal)
        for i in range(len(signal)):
            start = max(0, i - window)
            end = min(len(signal), i + window)
            local = signal[start:end]
            thresholds[i] = np.mean(local) + multiplier * np.std(local)
        return thresholds


class MetricsCalculator:
    """Compute evaluation metrics for segmentation quality."""

    @staticmethod
    def boundary_accuracy(predicted_boundaries: List[float],
                          ground_truth_boundaries: List[float],
                          tolerance: float = 1.0) -> dict:
        """Compute precision, recall, F1 for boundary detection."""
        tp = 0
        matched_gt = set()

        for pred in predicted_boundaries:
            for i, gt in enumerate(ground_truth_boundaries):
                if abs(pred - gt) <= tolerance and i not in matched_gt:
                    tp += 1
                    matched_gt.add(i)
                    break

        precision = tp / len(predicted_boundaries) if predicted_boundaries else 0
        recall = tp / len(ground_truth_boundaries) if ground_truth_boundaries else 0
        f1 = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0)

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": tp,
            "false_positives": len(predicted_boundaries) - tp,
            "false_negatives": len(ground_truth_boundaries) - tp,
            "tolerance_seconds": tolerance
        }

    @staticmethod
    def segment_iou(predicted_segments: List[Tuple[float, float]],
                    ground_truth_segments: List[Tuple[float, float]]) -> float:
        """Compute average IoU between predicted and ground truth segments."""
        if not predicted_segments or not ground_truth_segments:
            return 0.0

        ious = []
        for pred_start, pred_end in predicted_segments:
            best_iou = 0.0
            for gt_start, gt_end in ground_truth_segments:
                intersection_start = max(pred_start, gt_start)
                intersection_end = min(pred_end, gt_end)
                intersection = max(0, intersection_end - intersection_start)

                union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
                iou = intersection / union if union > 0 else 0.0
                best_iou = max(best_iou, iou)
            ious.append(best_iou)

        return float(np.mean(ious))
