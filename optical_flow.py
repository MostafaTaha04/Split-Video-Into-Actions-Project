import cv2
import numpy as np
from typing import Optional, Tuple


class OpticalFlowAnalyzer:
    """Dense optical flow for global motion analysis."""

    def __init__(self):
        self.prev_gray: Optional[np.ndarray] = None
        self.flow_magnitude_history = []

    def compute(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """Compute dense optical flow and return magnitude."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return None, 0.0

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        self.prev_gray = gray

        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        avg_magnitude = float(magnitude.mean())
        self.flow_magnitude_history.append(avg_magnitude)

        return flow, avg_magnitude

    def get_motion_discontinuity(self, window: int = 15) -> float:
        """Detect sudden changes in global motion patterns."""
        if len(self.flow_magnitude_history) < window * 2:
            return 0.0

        recent = np.array(self.flow_magnitude_history[-window:])
        previous = np.array(self.flow_magnitude_history[-window*2:-window])

        return abs(recent.mean() - previous.mean()) / (previous.std() + 1e-6)
