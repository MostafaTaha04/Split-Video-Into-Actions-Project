import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class HandData:
    """Stores per-frame hand tracking data."""
    landmarks: np.ndarray
    pixel_coords: np.ndarray
    handedness: str
    confidence: float
    bounding_box: Tuple[int, int, int, int]
    fingertip_positions: np.ndarray
    palm_center: np.ndarray
    is_gripping: bool = False
    velocity: float = 0.0
    acceleration: float = 0.0
    direction: float = 0.0  # angle of motion in radians


class HandTracker:
    """Tracks hands using MediaPipe and extracts motion features."""

    FINGERTIP_IDS = [4, 8, 12, 16, 20]
    PALM_IDS = [0, 5, 9, 13, 17]
    FINGER_MCP_IDS = [5, 9, 13, 17]

    def __init__(self, detection_confidence: float = 0.7,
                 tracking_confidence: float = 0.6, max_hands: int = 2):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.previous_positions = {}
        self.previous_velocities = {}
        self.velocity_history = {"Left": [], "Right": []}
        self.position_history = {"Left": [], "Right": []}

    def process_frame(self, frame: np.ndarray) -> List[HandData]:
        """Detect and track hands in a single frame."""
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        hands_data = []
        if not results.multi_hand_landmarks:
            return hands_data

        for hand_landmarks, handedness_info in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            landmarks = np.array([
                [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
            ])

            pixel_coords = np.array([
                [int(lm.x * w), int(lm.y * h)]
                for lm in hand_landmarks.landmark
            ])

            fingertips = pixel_coords[self.FINGERTIP_IDS]
            palm_points = pixel_coords[self.PALM_IDS]
            palm_center = palm_points.mean(axis=0).astype(int)

            x_min = pixel_coords[:, 0].min()
            y_min = pixel_coords[:, 1].min()
            x_max = pixel_coords[:, 0].max()
            y_max = pixel_coords[:, 1].max()
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

            handedness = handedness_info.classification[0].label
            confidence = handedness_info.classification[0].score

            velocity = self._compute_velocity(handedness, palm_center)
            acceleration = self._compute_acceleration(handedness, velocity)
            direction = self._compute_direction(handedness, palm_center)
            is_gripping = self._detect_grip(landmarks)

            hand_data = HandData(
                landmarks=landmarks,
                pixel_coords=pixel_coords,
                handedness=handedness,
                confidence=confidence,
                bounding_box=bbox,
                fingertip_positions=fingertips,
                palm_center=palm_center,
                is_gripping=is_gripping,
                velocity=velocity,
                acceleration=acceleration,
                direction=direction
            )
            hands_data.append(hand_data)

        return hands_data

    def _compute_velocity(self, handedness: str, current_pos: np.ndarray) -> float:
        """Compute hand movement velocity between frames."""
        if handedness in self.previous_positions:
            prev = self.previous_positions[handedness]
            velocity = float(np.linalg.norm(current_pos - prev))
        else:
            velocity = 0.0

        self.previous_positions[handedness] = current_pos.copy()
        self.velocity_history[handedness].append(velocity)
        self.position_history[handedness].append(current_pos.copy())

        if len(self.velocity_history[handedness]) > 60:
            self.velocity_history[handedness].pop(0)
        if len(self.position_history[handedness]) > 60:
            self.position_history[handedness].pop(0)

        return velocity

    def _compute_acceleration(self, handedness: str, current_velocity: float) -> float:
        """Compute hand acceleration."""
        if handedness in self.previous_velocities:
            acceleration = current_velocity - self.previous_velocities[handedness]
        else:
            acceleration = 0.0

        self.previous_velocities[handedness] = current_velocity
        return acceleration

    def _compute_direction(self, handedness: str, current_pos: np.ndarray) -> float:
        """Compute direction of hand movement in radians."""
        if handedness in self.previous_positions:
            prev = self.previous_positions[handedness]
            diff = current_pos - prev
            return float(np.arctan2(diff[1], diff[0]))
        return 0.0

    def _detect_grip(self, landmarks: np.ndarray) -> bool:
        """Detect if the hand is in a gripping pose based on finger curl."""
        curl_count = 0
        for finger_idx in range(1, 5):
            tip = landmarks[finger_idx * 4 + 4]
            pip = landmarks[finger_idx * 4 + 2]
            mcp = landmarks[finger_idx * 4 + 1]

            tip_to_mcp = np.linalg.norm(tip[:2] - mcp[:2])
            pip_to_mcp = np.linalg.norm(pip[:2] - mcp[:2])

            if tip_to_mcp < pip_to_mcp * 1.1:
                curl_count += 1

        return curl_count >= 3

    def get_average_velocity(self, handedness: str, window: int = 10) -> float:
        """Get smoothed velocity over recent frames."""
        history = self.velocity_history.get(handedness, [])
        if not history:
            return 0.0
        recent = history[-window:]
        return float(np.mean(recent))

    def get_velocity_variance(self, handedness: str, window: int = 15) -> float:
        """Get velocity variance over recent frames (indicates rhythm change)."""
        history = self.velocity_history.get(handedness, [])
        if len(history) < 3:
            return 0.0
        recent = history[-window:]
        return float(np.var(recent))

    def get_trajectory_curvature(self, handedness: str, window: int = 10) -> float:
        """Compute trajectory curvature (high = direction changes)."""
        history = self.position_history.get(handedness, [])
        if len(history) < 3:
            return 0.0

        recent = np.array(history[-window:])
        if len(recent) < 3:
            return 0.0

        vectors = np.diff(recent, axis=0)
        norms = np.linalg.norm(vectors, axis=1)
        norms[norms == 0] = 1e-6

        unit_vectors = vectors / norms[:, np.newaxis]
        angle_changes = np.arccos(np.clip(
            np.sum(unit_vectors[:-1] * unit_vectors[1:], axis=1), -1, 1
        ))

        return float(np.mean(angle_changes))

    def reset(self):
        """Reset tracking state."""
        self.previous_positions = {}
        self.previous_velocities = {}
        self.velocity_history = {"Left": [], "Right": []}
        self.position_history = {"Left": [], "Right": []}
