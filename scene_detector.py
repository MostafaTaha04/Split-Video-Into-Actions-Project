import cv2
import numpy as np
from typing import List


class SceneChangeDetector:
    """Detects scene-level visual changes using histogram comparison."""

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self.prev_hist = None
        self.change_scores = []

    def compute_change(self, frame: np.ndarray) -> float:
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
        self.change_scores.append(score)

        return score

    def get_scene_boundaries(self) -> List[int]:
        """Return frame indices where scene changes were detected."""
        return [i for i, s in enumerate(self.change_scores) if s > self.threshold]
