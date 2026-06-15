import cv2
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class FlowData:
    """Stores optical flow analysis results for a frame."""
    flow_field: Optional[np.ndarray]
    magnitude_mean: float
    magnitude_max: float
    dominant_direction: float
    motion_uniformity: float
    discontinuity_score: float


class OpticalFlowAnalyzer:
    """Dense optical flow for global and local motion analysis."""

    def __init__(self, window: int = 15):
        self.prev_gray: Optional[np.ndarray] = None
        self.flow_magnitude_history: List[float] = []
        self.flow_direction_history: List[float] = []
        self.window = window

    def compute(self, frame: np.ndarray) -> FlowData:
        """Compute dense optical flow and return structured analysis."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return FlowData(
                flow_field=None,
                magnitude_mean=0.0,
                magnitude_max=0.0,
                dominant_direction=0.0,
                motion_uniformity=1.0,
                discontinuity_score=0.0
            )

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        self.prev_gray = gray

        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        direction = np.arctan2(flow[..., 1], flow[..., 0])

        mag_mean = float(magnitude.mean())
        mag_max = float(magnitude.max())

        mask = magnitude > mag_mean
        if mask.sum() > 0:
            dominant_dir = float(np.median(direction[mask]))
        else:
            dominant_dir = 0.0

        motion_uniformity = self._compute_uniformity(direction, magnitude)
        discontinuity = self._compute_discontinuity(mag_mean)

        self.flow_magnitude_history.append(mag_mean)
        self.flow_direction_history.append(dominant_dir)
        if len(self.flow_magnitude_history) > 120:
            self.flow_magnitude_history.pop(0)
            self.flow_direction_history.pop(0)

        return FlowData(
            flow_field=flow,
            magnitude_mean=mag_mean,
            magnitude_max=mag_max,
            dominant_direction=dominant_dir,
            motion_uniformity=motion_uniformity,
            discontinuity_score=discontinuity
        )

    def _compute_uniformity(self, direction: np.ndarray,
                            magnitude: np.ndarray) -> float:
        """Measure how uniform the flow directions are (1=all same, 0=chaotic)."""
        mask = magnitude > 1.0
        if mask.sum() < 10:
            return 1.0

        dirs = direction[mask]
        mean_vec = np.array([np.cos(dirs).mean(), np.sin(dirs).mean()])
        uniformity = float(np.linalg.norm(mean_vec))

        return uniformity

    def _compute_discontinuity(self, current_magnitude: float) -> float:
        """Detect sudden changes in global motion patterns."""
        if len(self.flow_magnitude_history) < self.window * 2:
            return 0.0

        recent = np.array(self.flow_magnitude_history[-self.window:])
        previous = np.array(
            self.flow_magnitude_history[-self.window * 2:-self.window]
        )

        prev_std = previous.std()
        if prev_std < 0.01:
            prev_std = 0.01

        return abs(recent.mean() - previous.mean()) / prev_std

    def get_motion_energy(self, window: int = 15) -> float:
        """Get average motion energy over recent frames."""
        if not self.flow_magnitude_history:
            return 0.0
        recent = self.flow_magnitude_history[-window:]
        return float(np.mean(recent))

    def get_motion_variance(self, window: int = 15) -> float:
        """Get variance in motion energy (indicates transitions)."""
        if len(self.flow_magnitude_history) < 3:
            return 0.0
        recent = self.flow_magnitude_history[-window:]
        return float(np.var(recent))

    def get_direction_change(self, window: int = 10) -> float:
        """Measure how much the dominant motion direction changed recently."""
        if len(self.flow_direction_history) < 3:
            return 0.0

        recent = np.array(self.flow_direction_history[-window:])
        diffs = np.abs(np.diff(recent))
        diffs = np.minimum(diffs, 2 * np.pi - diffs)

        return float(np.mean(diffs))

    def visualize_flow(self, flow: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Create a visualization of optical flow overlaid on frame."""
        h, w = flow.shape[:2]
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 1] = 255

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        blended = cv2.addWeighted(frame, 0.7, flow_rgb, 0.3, 0)

        return blended

    def reset(self):
        """Reset flow state."""
        self.prev_gray = None
        self.flow_magnitude_history = []
        self.flow_direction_history = []
