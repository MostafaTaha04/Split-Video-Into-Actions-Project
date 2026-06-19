import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

# MediaPipe is useful when correctly installed, but the pipeline must not crash
# if the MediaPipe model files are missing from the virtual environment.
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
    direction: float = 0.0  # angle of motion in radians


class HandTracker:
    """
    Tracks hands using MediaPipe when available.

    If MediaPipe is missing or its internal .binarypb model files are missing,
    the class automatically falls back to a lightweight motion/blob tracker.
    This is important on Windows/OneDrive environments where MediaPipe can be
    partially installed and throw errors like:

        FileNotFoundError: hand_landmark_tracking_cpu.binarypb

    The fallback does not estimate real hand landmarks, but it provides hand-like
    moving regions so the rest of the action segmentation pipeline can still run.
    """

    FINGERTIP_IDS = [4, 8, 12, 16, 20]
    PALM_IDS = [0, 5, 9, 13, 17]

    def __init__(
        self,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.6,
        max_hands: int = 2,
    ):
        self.max_hands = max_hands
        self.mp_hands = mp_hands
        self.hands = None
        self.use_mediapipe = False
        self.previous_gray: Optional[np.ndarray] = None

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
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        hands_data: List[HandData] = []
        if not results.multi_hand_landmarks:
            return hands_data

        for hand_landmarks, handedness_info in zip(
            results.multi_hand_landmarks,
            results.multi_handedness,
        ):
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            pixel_coords = np.array([
                [int(lm.x * w), int(lm.y * h)] for lm in hand_landmarks.landmark
            ])

            fingertips = pixel_coords[self.FINGERTIP_IDS]
            palm_points = pixel_coords[self.PALM_IDS]
            palm_center = palm_points.mean(axis=0).astype(int)

            x_min = int(pixel_coords[:, 0].min())
            y_min = int(pixel_coords[:, 1].min())
            x_max = int(pixel_coords[:, 0].max())
            y_max = int(pixel_coords[:, 1].max())
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

            handedness = handedness_info.classification[0].label
            confidence = float(handedness_info.classification[0].score)

            direction = self._compute_direction(handedness, palm_center)
            velocity = self._compute_velocity(handedness, palm_center)
            acceleration = self._compute_acceleration(handedness, velocity)
            is_gripping = self._detect_grip(landmarks)

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
        Fallback hand tracker based on moving regions.

        It is not a true landmark detector. It finds the largest moving blobs,
        which usually correspond to hands in your close-up CPU/motherboard video.
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

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

            # Ignore tiny border flicker.
            if x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2:
                continue

            center = np.array([x + bw // 2, y + bh // 2], dtype=int)

            if center[0] < w / 2:
                label = "Left"
            else:
                label = "Right"

            if label in used_labels:
                label = "Motion"
            used_labels.add(label)

            # Create pseudo landmarks inside the blob so downstream code works.
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
        """Create 21 stable pseudo-landmarks inside a bounding box."""
        xs = [x + int(w * p) for p in [0.20, 0.35, 0.50, 0.65, 0.80]]
        ys = [y + int(h * p) for p in [0.20, 0.40, 0.60, 0.80]]
        pts = []
        pts.append([x + w // 2, y + h // 2])
        for yy in ys:
            for xx in xs:
                pts.append([xx, yy])
        pts = np.array(pts[:21], dtype=int)
        while len(pts) < 21:
            pts = np.vstack([pts, pts[-1]])
        return pts

    def _compute_velocity(self, handedness: str, current_pos: np.ndarray) -> float:
        if handedness in self.previous_positions:
            prev = self.previous_positions[handedness]
            velocity = float(np.linalg.norm(current_pos - prev))
        else:
            velocity = 0.0

        self.previous_positions[handedness] = current_pos.copy()
        self.velocity_history.setdefault(handedness, []).append(velocity)
        self.position_history.setdefault(handedness, []).append(current_pos.copy())

        if len(self.velocity_history[handedness]) > 60:
            self.velocity_history[handedness].pop(0)
        if len(self.position_history[handedness]) > 60:
            self.position_history[handedness].pop(0)

        return velocity

    def _compute_acceleration(self, handedness: str, current_velocity: float) -> float:
        if handedness in self.previous_velocities:
            acceleration = current_velocity - self.previous_velocities[handedness]
        else:
            acceleration = 0.0
        self.previous_velocities[handedness] = current_velocity
        return float(acceleration)

    def _compute_direction(self, handedness: str, current_pos: np.ndarray) -> float:
        if handedness in self.previous_positions:
            prev = self.previous_positions[handedness]
            diff = current_pos - prev
            return float(np.arctan2(diff[1], diff[0]))
        return 0.0

    def _detect_grip(self, landmarks: np.ndarray) -> bool:
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
        history = self.velocity_history.get(handedness, [])
        if not history:
            return 0.0
        return float(np.mean(history[-window:]))

    def get_velocity_variance(self, handedness: str, window: int = 15) -> float:
        history = self.velocity_history.get(handedness, [])
        if len(history) < 3:
            return 0.0
        return float(np.var(history[-window:]))

    def get_trajectory_curvature(self, handedness: str, window: int = 10) -> float:
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
        self.previous_gray = None
        self.previous_positions = {}
        self.previous_velocities = {}
        self.velocity_history = {"Left": [], "Right": [], "Motion": []}
        self.position_history = {"Left": [], "Right": [], "Motion": []}