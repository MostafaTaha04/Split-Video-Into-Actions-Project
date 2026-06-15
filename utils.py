import numpy as np
from typing import List, Tuple
import time
import json
from pathlib import Path


class FPSCounter:
    """Tracks processing FPS."""

    def __init__(self, window: int = 30):
        self.window = window
        self.timestamps = []
        self.frame_count = 0

    def tick(self):
        self.timestamps.append(time.time())
        self.frame_count += 1
        if len(self.timestamps) > self.window:
            self.timestamps.pop(0)

    def get_fps(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0.0

    def get_elapsed(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]


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
    def gaussian_smooth(signal: np.ndarray, sigma: float = 3.0) -> np.ndarray:
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(signal, sigma)

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

    @staticmethod
    def savgol_smooth(signal: np.ndarray, window: int = 11,
                      order: int = 3) -> np.ndarray:
        """Savitzky-Golay filter for smooth derivatives."""
        from scipy.signal import savgol_filter
        if len(signal) < window:
            return signal
        return savgol_filter(signal, window, order)


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
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
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

    @staticmethod
    def edit_distance(predicted_labels: List[str],
                      ground_truth_labels: List[str]) -> int:
        """Compute edit distance between label sequences."""
        n, m = len(predicted_labels), len(ground_truth_labels)
        dp = [[0] * (m + 1) for _ in range(n + 1)]

        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if predicted_labels[i-1] == ground_truth_labels[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        return dp[n][m]

    @staticmethod
    def normalized_edit_distance(predicted_labels: List[str],
                                  ground_truth_labels: List[str]) -> float:
        """Normalized edit distance (0=identical, 1=completely different)."""
        edit_dist = MetricsCalculator.edit_distance(predicted_labels, ground_truth_labels)
        max_len = max(len(predicted_labels), len(ground_truth_labels))
        return edit_dist / max_len if max_len > 0 else 0.0


class AnnotationTool:
    """Simple ground truth annotation helper."""

    @staticmethod
    def create_template(video_duration: float, estimated_steps: int,
                        output_path: str):
        """Create annotation template with evenly spaced boundaries."""
        step_duration = video_duration / estimated_steps
        segments = []
        boundaries = []

        for i in range(estimated_steps):
            start = i * step_duration
            end = (i + 1) * step_duration
            segments.append({
                "id": i,
                "start_time": round(start, 2),
                "end_time": round(end, 2),
                "activity": "",
                "description": ""
            })
            if i > 0:
                boundaries.append({
                    "timestamp": round(start, 2),
                    "description": ""
                })

        template = {
            "video_duration": video_duration,
            "segments": segments,
            "boundaries": boundaries
        }

        with open(output_path, 'w') as f:
            json.dump(template, f, indent=2)

    @staticmethod
    def validate_annotation(annotation_path: str) -> Tuple[bool, List[str]]:
        """Validate that an annotation file is properly formatted."""
        errors = []

        try:
            with open(annotation_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            return False, [str(e)]

        if "segments" not in data:
            errors.append("Missing 'segments' field")
        if "boundaries" not in data:
            errors.append("Missing 'boundaries' field")

        if "segments" in data:
            for i, seg in enumerate(data["segments"]):
                if seg.get("start_time", 0) >= seg.get("end_time", 0):
                    errors.append(f"Segment {i}: start_time >= end_time")
                if i > 0:
                    prev_end = data["segments"][i-1].get("end_time", 0)
                    curr_start = seg.get("start_time", 0)
                    if abs(curr_start - prev_end) > 0.1:
                        errors.append(f"Segment {i}: gap/overlap with previous segment")

        return len(errors) == 0, errors
