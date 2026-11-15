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
    def match_boundaries(predicted_boundaries: List[float],
                         ground_truth_boundaries: List[float],
                         tolerance: float = 1.0) -> List[Tuple[int, int, float]]:
        """Optimal one-to-one matching of predicted to ground-truth boundaries.

        Uses the Hungarian algorithm (scipy.optimize.linear_sum_assignment) to
        find the assignment that maximises the number of matches within
        ``tolerance`` and, among those, minimises the total time offset. This is
        the standard, order-independent boundary-matching scheme and removes the
        ambiguity of greedy matching when boundaries are densely spaced.

        Returns a list of ``(pred_index, gt_index, offset_seconds)`` tuples.
        Falls back to a greedy closest-match if SciPy is unavailable.
        """
        if not predicted_boundaries or not ground_truth_boundaries:
            return []

        pred = np.asarray(predicted_boundaries, dtype=float)
        gt = np.asarray(ground_truth_boundaries, dtype=float)
        cost = np.abs(pred[:, None] - gt[None, :])

        try:
            from scipy.optimize import linear_sum_assignment

            # Heavily penalise out-of-tolerance pairs so the optimiser prefers
            # the maximum number of valid matches first, then the smallest
            # offsets; any forced out-of-tolerance pairs are discarded below.
            big = float(cost.max()) * (cost.size + 1) + 1.0
            penalised = np.where(cost <= tolerance, cost, big)
            rows, cols = linear_sum_assignment(penalised)
            matches = [
                (int(r), int(c), float(cost[r, c]))
                for r, c in zip(rows, cols)
                if cost[r, c] <= tolerance
            ]
        except Exception:
            # Greedy fallback: nearest unused GT within tolerance.
            used = set()
            matches = []
            order = sorted(range(len(pred)), key=lambda i: pred[i])
            for i in order:
                candidates = [
                    (abs(pred[i] - gt[j]), j)
                    for j in range(len(gt))
                    if j not in used and abs(pred[i] - gt[j]) <= tolerance
                ]
                if candidates:
                    off, j = min(candidates)
                    used.add(j)
                    matches.append((i, j, float(off)))

        return matches

    @staticmethod
    def boundary_accuracy(predicted_boundaries: List[float],
                          ground_truth_boundaries: List[float],
                          tolerance: float = 1.0) -> dict:
        """Compute precision, recall, F1 for boundary detection.

        Uses optimal one-to-one matching (see ``match_boundaries``).
        """
        tp = len(MetricsCalculator.match_boundaries(
            predicted_boundaries, ground_truth_boundaries, tolerance
        ))

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


    # ------------------------------------------------------------------
    # Standard temporal-segmentation metrics.
    #
    # Boundary F1 at a fixed tolerance (above) is this project's primary
    # metric. The three below are the ones the literature reports, so results
    # can be compared with published work. All are class-agnostic, because this
    # project segments steps without recognising them.
    # ------------------------------------------------------------------
    @staticmethod
    def boundaries_to_segments(boundaries: List[float], duration: float,
                               start: float = 0.0) -> List[Tuple[float, float]]:
        """Turn interior boundary times into contiguous (start, end) segments."""
        cuts = sorted(float(b) for b in boundaries if start < float(b) < duration)
        edges = [float(start)] + cuts + [float(duration)]
        return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

    @staticmethod
    def _overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))

    @staticmethod
    def segmental_f1(predicted_segments: List[Tuple[float, float]],
                     ground_truth_segments: List[Tuple[float, float]],
                     overlap: float) -> float:
        """Class-agnostic segmental F1@overlap (Lea et al., CVPR 2017).

        The standard action-segmentation metric (reported as F1@{10,25,50}).
        Each predicted segment, in temporal order, is matched to the
        ground-truth segment it overlaps most; it is a true positive if that
        IoU is at least ``overlap`` and the ground-truth segment has not
        already been claimed, otherwise a false positive. Unclaimed
        ground-truth segments are false negatives. Unlike boundary F1, this
        penalises over-segmentation through the IoU of the pieces, not only
        through the number of extra boundaries.
        """
        if not predicted_segments or not ground_truth_segments:
            return 0.0
        used = [False] * len(ground_truth_segments)
        tp = fp = 0
        for ps in predicted_segments:
            best_j, best_iou = -1, 0.0
            for j, gs in enumerate(ground_truth_segments):
                inter = MetricsCalculator._overlap(ps, gs)
                union = (ps[1] - ps[0]) + (gs[1] - gs[0]) - inter
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_j, best_iou = j, iou
            if best_j >= 0 and best_iou >= overlap and not used[best_j]:
                tp += 1
                used[best_j] = True
            else:
                fp += 1
        fn = len(ground_truth_segments) - sum(used)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return (2 * precision * recall / (precision + recall)
                if precision + recall > 0 else 0.0)

    @staticmethod
    def frame_accuracy_hungarian(predicted_segments: List[Tuple[float, float]],
                                 ground_truth_segments: List[Tuple[float, float]]) -> float:
        """Class-agnostic frame accuracy (MoF) with one-to-one segment matching.

        The usual "accuracy" for unsupervised / class-agnostic segmentation:
        predicted segments are matched one-to-one to ground-truth segments by
        the Hungarian algorithm so as to maximise total temporal overlap, and
        the score is the fraction of ground-truth time that lands in its
        matched predicted segment. Computed from exact interval overlaps, so it
        is independent of the frame rate. Over-segmentation lowers it because
        each ground-truth step can claim only one predicted piece.
        """
        if not predicted_segments or not ground_truth_segments:
            return 0.0
        total = sum(e - s for s, e in ground_truth_segments)
        if total <= 0:
            return 0.0
        overlap = np.array([[MetricsCalculator._overlap(p, g)
                             for g in ground_truth_segments]
                            for p in predicted_segments])
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(-overlap)
        return float(overlap[rows, cols].sum() / total)

    GEBD_THRESHOLDS = tuple(round(0.05 * k, 2) for k in range(1, 11))

    @staticmethod
    def gebd_f1(predicted_boundaries: List[float],
                ground_truth_boundaries: List[float],
                duration: float,
                thresholds=None) -> dict:
        """Boundary F1 under the Generic Event Boundary Detection protocol.

        Kinetics-GEBD (Shou et al., ICCV 2021) scores a detection as correct
        when its Relative Distance -- the absolute time error divided by the
        video length -- is at most a threshold, swept from 0.05 to 0.5. The
        official challenge metric is F1@0.05. Returns ``{"0.05": f1, ...,
        "avg": mean}``. Matching is one-to-one (Hungarian), as everywhere else
        in this project.

        For the ~50-60 s clips in this project, Rel.Dis 0.05 corresponds to a
        tolerance of about 2.5-3 s, i.e. it is a loose metric here; the
        fixed-tolerance F1 at 1.0 s remains the primary number.
        """
        thresholds = thresholds or MetricsCalculator.GEBD_THRESHOLDS
        out = {}
        for th in thresholds:
            m = MetricsCalculator.boundary_accuracy(
                predicted_boundaries, ground_truth_boundaries,
                tolerance=th * max(duration, 1e-9))
            out[f"{th:.2f}"] = float(m["f1_score"])
        out["avg"] = float(np.mean([out[f"{th:.2f}"] for th in thresholds]))
        return out


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
