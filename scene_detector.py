import cv2
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class SceneChangeData:
    """Stores scene change analysis for a frame."""
    histogram_change: float
    structural_change: float
    combined_score: float
    is_boundary: bool


class SceneChangeDetector:
    """Detects scene-level visual changes using multiple methods."""

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self.prev_hist = None
        self.prev_gray = None
        self.prev_edges = None
        self.change_scores: List[float] = []
        self.structural_scores: List[float] = []

    def process_frame(self, frame: np.ndarray) -> SceneChangeData:
        """Analyze frame for scene-level changes."""
        hist_change = self._compute_histogram_change(frame)
        struct_change = self._compute_structural_change(frame)

        combined = 0.6 * hist_change + 0.4 * struct_change
        is_boundary = combined > self.threshold

        self.change_scores.append(combined)
        if len(self.change_scores) > 120:
            self.change_scores.pop(0)

        return SceneChangeData(
            histogram_change=hist_change,
            structural_change=struct_change,
            combined_score=combined,
            is_boundary=is_boundary
        )

    def _compute_histogram_change(self, frame: np.ndarray) -> float:
        """Compare current frame histogram with previous."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60],
                            [0, 180, 0, 256])
        cv2.normalize(hist, hist)

        if self.prev_hist is None:
            self.prev_hist = hist
            return 0.0

        score = 1.0 - cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_CORREL)
        self.prev_hist = hist

        return max(0.0, score)

    def _compute_structural_change(self, frame: np.ndarray) -> float:
        """Detect structural changes using edge comparison."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)

        if self.prev_edges is None:
            self.prev_edges = edges
            self.prev_gray = gray
            return 0.0

        diff = cv2.absdiff(edges, self.prev_edges)
        change_ratio = float(diff.sum()) / (edges.shape[0] * edges.shape[1] * 255)

        ssim = self._compute_ssim(self.prev_gray, gray)
        structural_change = 1.0 - ssim

        self.prev_edges = edges
        self.prev_gray = gray
        self.structural_scores.append(structural_change)
        if len(self.structural_scores) > 120:
            self.structural_scores.pop(0)

        return 0.5 * change_ratio * 10 + 0.5 * structural_change

    def _compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Simplified SSIM computation."""
        c1 = 6.5025
        c2 = 58.5225

        mu1 = cv2.GaussianBlur(img1.astype(np.float64), (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2.astype(np.float64), (11, 11), 1.5)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(
            img1.astype(np.float64) ** 2, (11, 11), 1.5
        ) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(
            img2.astype(np.float64) ** 2, (11, 11), 1.5
        ) - mu2_sq
        sigma12 = cv2.GaussianBlur(
            img1.astype(np.float64) * img2.astype(np.float64), (11, 11), 1.5
        ) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

        return float(ssim_map.mean())

    def get_scene_boundaries(self) -> List[int]:
        """Return frame indices where scene changes were detected."""
        return [i for i, s in enumerate(self.change_scores) if s > self.threshold]

    def get_recent_change_level(self, window: int = 15) -> float:
        """Get average scene change level over recent frames."""
        if not self.change_scores:
            return 0.0
        recent = self.change_scores[-window:]
        return float(np.mean(recent))

    def get_visual_stability(self, window: int = 30) -> float:
        """Measure visual stability (inverse of scene change frequency)."""
        if len(self.change_scores) < window:
            return 1.0
        recent = self.change_scores[-window:]
        boundary_count = sum(1 for s in recent if s > self.threshold)
        return 1.0 - (boundary_count / len(recent))

    def reset(self):
        """Reset detector state."""
        self.prev_hist = None
        self.prev_gray = None
        self.prev_edges = None
        self.change_scores = []
        self.structural_scores = []
