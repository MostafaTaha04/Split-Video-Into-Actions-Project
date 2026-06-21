import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from collections import deque

try:
    from mediapipe.python.solutions import hands as mp_hands
except Exception:
    mp_hands = None


@dataclass
class HandData:
    """Stores per-frame hand tracking data."""
    landmarks: np.ndarray
    pixel_coords: np.ndarray
    handedness: str
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    fingertip_positions: np.ndarray
    palm_center: np.ndarray
    is_gripping: bool = False
    velocity: float = 0.0
    acceleration: float = 0.0
    direction: float = 0.0


class HandTracker:
    """
    Tracks hands using MediaPipe when available.

    V3 update:
    - Real MediaPipe landmarks are now used when available.
    - Grip state is smoothed using hysteresis to prevent frame-to-frame flicker.
    - Fallback motion tracking remains only as an emergency fallback.
    """

    FINGERTIP_IDS = [4, 8, 12, 16, 20]
    PALM_IDS = [0, 5, 9, 13, 17]

    def __init__(
        self,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.6,
        max_hands: int = 2,
        grip_smoothing_window: int = 5,
    ):
        self.max_hands = max_hands
        self.mp_hands = mp_hands
        self.hands = None
        self.use_mediapipe = False
        self.previous_gray: Optional[np.ndarray] = None

        self.grip_smoothing_window = max(3, int(grip_smoothing_window))
        self.grip_history: Dict[str, deque] = {
            "Left": deque(maxlen=self.grip_smoothing_window),
            "Right": deque(maxlen=self.grip_smoothing_window),
            "Motion": deque(maxlen=self.grip_smoothing_window),
        }
        self.grip_state: Dict[str, bool] = {
            "Left": False,
            "Right": False,
            "Motion": False,
        }

        if self.mp_hands is not None:
            try:
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=max_hands,
                    min_detection_confidence=detection_confidence,
                    min_tracking_confidence=tracking_confidence,
                )
                self.use_mediapipe = True
                print("HandTracker: MediaPipe Hands enabled.")
            except Exception as exc:
                print("WARNING: MediaPipe Hands could not be initialized.")
                print(f"         Reason: {exc}")
                print("         Falling back to motion-based hand tracking.")
                print("         To repair MediaPipe later, run:")
                print("         python -m pip uninstall mediapipe -y")
                print("         python -m pip install --no-cache-dir --force-reinstall mediapipe==0.10.14")
        else:
            print("WARNING: MediaPipe is not available. Using motion-based hand tracking.")

        self.previous_positions = {}
        self.previous_velocities = {}
        self.velocity_history = {"Left": [], "Right": [], "Motion": []}
        self.position_history = {"Left": [], "Right": [], "Motion": []}

    def process_frame(self, frame: np.ndarray) -> List[HandData]:
        """Detect and track hands in a single frame."""
        if self.use_mediapipe and self.hands is not None:
            try:
                return self._process_frame_mediapipe(frame)
            except Exception as exc:
                print("WARNING: MediaPipe failed during processing.")
                print(f"         Reason: {exc}")
                print("         Switching to motion-based hand tracking for the rest of this run.")
                self.use_mediapipe = False
                self.hands = None

        return self._process_frame_motion(frame)

    def _process_frame_mediapipe(self, frame: np.ndarray) -> List[HandData]:
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        hands_data: List[HandData] = []

        if not results.multi_hand_landmarks:
            return hands_data

        for hand_landmarks, handedness_info in zip(
            results.multi_hand_landmarks,
            results.multi_handedness,
        ):
            landmarks = np.array([
                [lm.x, lm.y, lm.z]
                for lm in hand_landmarks.landmark
            ], dtype=float)

            pixel_coords = np.array([
                [int(lm.x * w), int(lm.y * h)]
                for lm in hand_landmarks.landmark
            ], dtype=int)

            fingertips = pixel_coords[self.FINGERTIP_IDS]
            palm_points = pixel_coords[self.PALM_IDS]
            palm_center = palm_points.mean(axis=0).astype(int)

            x_min = int(pixel_coords[:, 0].min())
            y_min = int(pixel_coords[:, 1].min())
            x_max = int(pixel_coords[:, 0].max())
            y_max = int(pixel_coords[:, 1].max())

            pad = 8
            x_min = max(0, x_min - pad)
            y_min = max(0, y_min - pad)
            x_max = min(w - 1, x_max + pad)
            y_max = min(h - 1, y_max + pad)

            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

            handedness = handedness_info.classification[0].label
            confidence = float(handedness_info.classification[0].score)

            direction = self._compute_direction(handedness, palm_center)
            velocity = self._compute_velocity(handedness, palm_center)
            acceleration = self._compute_acceleration(handedness, velocity)

            raw_grip = self._detect_grip_raw(landmarks)
            is_gripping = self._smooth_grip(handedness, raw_grip)

            hands_data.append(HandData(
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
                direction=direction,
            ))

        return hands_data

    def _process_frame_motion(self, frame: np.ndarray) -> List[HandData]:
        """
        Emergency fallback tracker based on moving blobs.

        This should not be used for final reported results if MediaPipe works.
        """
        h, w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if self.previous_gray is None:
            self.previous_gray = gray
            return []

        diff = cv2.absdiff(gray, self.previous_gray)
        self.previous_gray = gray

        _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        hands_data: List[HandData] = []
        used_labels = set()

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < 450:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)

            if bw < 15 or bh < 15:
                continue

            if x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2:
                continue

            center = np.array([x + bw // 2, y + bh // 2], dtype=int)

            label = "Left" if center[0] < w / 2 else "Right"

            if label in used_labels:
                label = "Motion"

            used_labels.add(label)

            pixel_coords = self._make_pseudo_landmarks(x, y, bw, bh)

            landmarks = np.column_stack([
                pixel_coords[:, 0] / max(w, 1),
                pixel_coords[:, 1] / max(h, 1),
                np.zeros(21),
            ])

            fingertips = pixel_coords[self.FINGERTIP_IDS]

            direction = self._compute_direction(label, center)
            velocity = self._compute_velocity(label, center)
            acceleration = self._compute_acceleration(label, velocity)

            hands_data.append(HandData(
                landmarks=landmarks,
                pixel_coords=pixel_coords,
                handedness=label,
                confidence=float(min(0.95, area / (w * h) * 40.0)),
                bounding_box=(int(x), int(y), int(bw), int(bh)),
                fingertip_positions=fingertips,
                palm_center=center,
                is_gripping=False,
                velocity=velocity,
                acceleration=acceleration,
                direction=direction,
            ))

            if len(hands_data) >= self.max_hands:
                break

        return hands_data

    def _make_pseudo_landmarks(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Create 21 pseudo-landmarks for fallback mode only."""
        points = []

        for row in range(5):
            for col in range(4):
                px = x + int((col + 0.5) * w / 4)
                py = y + int((row + 0.5) * h / 5)
                points.append([px, py])

        points.append([x + w // 2, y + h // 2])

        return np.array(points[:21], dtype=int)

    def _detect_grip_raw(self, landmarks: np.ndarray) -> bool:
        """
        Estimate grip/pinch from MediaPipe normalized landmarks.

        This is intentionally conservative because false grip flicker creates
        false segmentation boundaries.
        """
        wrist = landmarks[0, :2]
        palm = landmarks[self.PALM_IDS, :2].mean(axis=0)

        hand_scale = np.linalg.norm(landmarks[9, :2] - wrist) + 1e-6

        thumb_tip = landmarks[4, :2]
        index_tip = landmarks[8, :2]
        middle_tip = landmarks[12, :2]
        ring_tip = landmarks[16, :2]
        pinky_tip = landmarks[20, :2]

        finger_tips = [index_tip, middle_tip, ring_tip, pinky_tip]

        tip_distances = [
            np.linalg.norm(tip - palm) / hand_scale
            for tip in finger_tips
        ]

        thumb_index_distance = np.linalg.norm(thumb_tip - index_tip) / hand_scale

        folded_count = sum(d < 1.35 for d in tip_distances)
        pinch = thumb_index_distance < 0.65

        return bool(pinch or folded_count >= 3)

    def _smooth_grip(self, label: str, raw_state: bool) -> bool:
        """
        Hysteresis smoothing:
        - To switch False -> True, most recent frames must mostly be True.
        - To switch True -> False, most recent frames must mostly be False.
        """
        if label not in self.grip_history:
            self.grip_history[label] = deque(maxlen=self.grip_smoothing_window)
            self.grip_state[label] = False

        hist = self.grip_history[label]
        hist.append(bool(raw_state))

        if len(hist) < self.grip_smoothing_window:
            return self.grip_state.get(label, False)

        true_count = sum(hist)
        current = self.grip_state.get(label, False)

        if not current:
            if true_count >= self.grip_smoothing_window - 1:
                self.grip_state[label] = True
        else:
            if true_count <= 1:
                self.grip_state[label] = False

        return self.grip_state[label]

    def _compute_velocity(self, label: str, position: np.ndarray) -> float:
        if label not in self.previous_positions:
            self.previous_positions[label] = position
            return 0.0

        prev = self.previous_positions[label]
        velocity = float(np.linalg.norm(position - prev))

        self.previous_positions[label] = position

        self.velocity_history.setdefault(label, []).append(velocity)

        if len(self.velocity_history[label]) > 120:
            self.velocity_history[label].pop(0)

        self.position_history.setdefault(label, []).append(position.copy())

        if len(self.position_history[label]) > 120:
            self.position_history[label].pop(0)

        return velocity

    def _compute_acceleration(self, label: str, velocity: float) -> float:
        prev_velocity = self.previous_velocities.get(label, velocity)
        acceleration = velocity - prev_velocity
        self.previous_velocities[label] = velocity
        return float(acceleration)

    def _compute_direction(self, label: str, position: np.ndarray) -> float:
        if label not in self.previous_positions:
            return 0.0

        prev = self.previous_positions[label]
        dx = position[0] - prev[0]
        dy = position[1] - prev[1]

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0

        return float(np.arctan2(dy, dx))

    def get_trajectory_curvature(self, label: str, window: int = 10) -> float:
        """Estimate how much the hand trajectory bends over a short window."""
        points = self.position_history.get(label, [])

        if len(points) < 3:
            return 0.0

        recent = points[-window:]

        if len(recent) < 3:
            return 0.0

        angles = []

        for i in range(1, len(recent) - 1):
            v1 = recent[i] - recent[i - 1]
            v2 = recent[i + 1] - recent[i]

            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)

            if n1 < 1e-6 or n2 < 1e-6:
                continue

            cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            angles.append(np.arccos(cosang))

        return float(np.mean(angles)) if angles else 0.0