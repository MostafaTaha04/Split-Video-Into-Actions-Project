import os
import urllib.request
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from collections import deque

import cv2
import numpy as np

# Tasks API (no binarypb needed) first, then legacy Solutions, then motion.
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _MP_TASKS_AVAILABLE = True
except Exception:
    mp = None
    mp_python = None
    mp_vision = None
    _MP_TASKS_AVAILABLE = False

try:
    from mediapipe.python.solutions import hands as mp_hands_legacy
except Exception:
    mp_hands_legacy = None

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


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
    direction: float = 0.0


class HandTracker:
    FINGERTIP_IDS = [4, 8, 12, 16, 20]
    PALM_IDS = [0, 5, 9, 13, 17]

    def __init__(self, detection_confidence=0.7, tracking_confidence=0.6,
                 max_hands=2, grip_smoothing_window=5, model_asset_path=None):
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.tasks_landmarker = None
        self.legacy_hands = None
        self.backend = "motion"
        self.use_mediapipe = False
        self._video_timestamp_ms = 0
        self.previous_gray = None
        self.grip_smoothing_window = max(3, int(grip_smoothing_window))
        self.grip_history = {k: deque(maxlen=self.grip_smoothing_window)
                             for k in ("Left", "Right", "Motion")}
        self.grip_state = {"Left": False, "Right": False, "Motion": False}
        self.previous_positions = {}
        self.previous_velocities = {}
        self.velocity_history = {"Left": [], "Right": [], "Motion": []}
        self.position_history = {"Left": [], "Right": [], "Motion": []}

        if _MP_TASKS_AVAILABLE:
            self._try_init_tasks(model_asset_path)
        if not self.use_mediapipe and mp_hands_legacy is not None:
            self._try_init_legacy()
        if not self.use_mediapipe:
            print("WARNING: MediaPipe Hands unavailable. Using motion-based hand tracking.")
            print("         To enable real landmarks: python -m pip install --upgrade mediapipe")

    def _resolve_model_path(self, model_asset_path):
        candidates = []
        if model_asset_path:
            candidates.append(model_asset_path)
        candidates.append(os.path.join(os.path.dirname(__file__), "hand_landmarker.task"))
        candidates.append("hand_landmarker.task")
        for path in candidates:
            if path and os.path.exists(path):
                return path
        target = candidates[1]
        try:
            print("HandTracker: downloading hand_landmarker.task (~7 MB, one time)...")
            urllib.request.urlretrieve(HAND_LANDMARKER_URL, target)
            print(f"HandTracker: saved model to {target}")
            return target
        except Exception as exc:
            print(f"WARNING: could not download hand_landmarker.task: {exc}")
            return None

    def _try_init_tasks(self, model_asset_path):
        model_path = self._resolve_model_path(model_asset_path)
        if not model_path:
            return
        try:
            # Read bytes in Python (handles non-ASCII Windows paths like מסמכים);
            # MediaPipe's C++ path loader cannot, so we pass a buffer instead.
            with open(model_path, "rb") as f:
                model_bytes = f.read()
            base_options = mp_python.BaseOptions(model_asset_buffer=model_bytes)
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=self.max_hands,
                min_hand_detection_confidence=self.detection_confidence,
                min_hand_presence_confidence=self.tracking_confidence,
                min_tracking_confidence=self.tracking_confidence,
            )
            self.tasks_landmarker = mp_vision.HandLandmarker.create_from_options(options)
            self.backend = "tasks"
            self.use_mediapipe = True
            print("HandTracker: MediaPipe Tasks HandLandmarker enabled.")
        except Exception as exc:
            print("WARNING: MediaPipe Tasks HandLandmarker could not be initialized.")
            print(f"         Reason: {exc}")
            self.tasks_landmarker = None

    def _try_init_legacy(self):
        try:
            self.legacy_hands = mp_hands_legacy.Hands(
                static_image_mode=False, max_num_hands=self.max_hands,
                min_detection_confidence=self.detection_confidence,
                min_tracking_confidence=self.tracking_confidence)
            self.backend = "legacy"
            self.use_mediapipe = True
            print("HandTracker: MediaPipe legacy Solutions Hands enabled.")
        except Exception as exc:
            print(f"WARNING: MediaPipe legacy Hands failed: {exc}")
            self.legacy_hands = None

    def process_frame(self, frame):
        if self.backend == "tasks" and self.tasks_landmarker is not None:
            try:
                return self._process_frame_tasks(frame)
            except Exception as exc:
                print(f"WARNING: Tasks failed: {exc}; switching to motion.")
                self.backend = "motion"; self.use_mediapipe = False
        if self.backend == "legacy" and self.legacy_hands is not None:
            try:
                return self._process_frame_legacy(frame)
            except Exception as exc:
                print(f"WARNING: legacy failed: {exc}; switching to motion.")
                self.backend = "motion"; self.use_mediapipe = False
        return self._process_frame_motion(frame)

    def _process_frame_tasks(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._video_timestamp_ms += 33
        result = self.tasks_landmarker.detect_for_video(mp_image, self._video_timestamp_ms)
        hands_data = []
        if not result.hand_landmarks:
            return hands_data
        for idx, hand_landmarks in enumerate(result.hand_landmarks):
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=float)
            pixel_coords = np.array([[int(lm.x * w), int(lm.y * h)] for lm in hand_landmarks], dtype=int)
            if result.handedness and idx < len(result.handedness):
                cat = result.handedness[idx][0]
                handedness, confidence = cat.category_name, float(cat.score)
            else:
                handedness, confidence = "Right", 1.0
            hands_data.append(self._build_hand_data(landmarks, pixel_coords, handedness, confidence, w, h))
        return hands_data

    def _process_frame_legacy(self, frame):
        h, w = frame.shape[:2]
        results = self.legacy_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hands_data = []
        if not results.multi_hand_landmarks:
            return hands_data
        for hand_landmarks, hinfo in zip(results.multi_hand_landmarks, results.multi_handedness):
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=float)
            pixel_coords = np.array([[int(lm.x * w), int(lm.y * h)] for lm in hand_landmarks.landmark], dtype=int)
            handedness = hinfo.classification[0].label
            confidence = float(hinfo.classification[0].score)
            hands_data.append(self._build_hand_data(landmarks, pixel_coords, handedness, confidence, w, h))
        return hands_data

    def _build_hand_data(self, landmarks, pixel_coords, handedness, confidence, w, h):
        fingertips = pixel_coords[self.FINGERTIP_IDS]
        palm_center = pixel_coords[self.PALM_IDS].mean(axis=0).astype(int)
        pad = 8
        x_min = max(0, int(pixel_coords[:, 0].min()) - pad)
        y_min = max(0, int(pixel_coords[:, 1].min()) - pad)
        x_max = min(w - 1, int(pixel_coords[:, 0].max()) + pad)
        y_max = min(h - 1, int(pixel_coords[:, 1].max()) + pad)
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        direction = self._compute_direction(handedness, palm_center)
        velocity = self._compute_velocity(handedness, palm_center)
        acceleration = self._compute_acceleration(handedness, velocity)
        is_gripping = self._smooth_grip(handedness, self._detect_grip_raw(landmarks))
        return HandData(landmarks, pixel_coords, handedness, confidence, bbox,
                        fingertips, palm_center, is_gripping, velocity, acceleration, direction)

    def _process_frame_motion(self, frame):
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
        hands_data = []
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
                np.zeros(21)])
            fingertips = pixel_coords[self.FINGERTIP_IDS]
            direction = self._compute_direction(label, center)
            velocity = self._compute_velocity(label, center)
            acceleration = self._compute_acceleration(label, velocity)
            hands_data.append(HandData(
                landmarks=landmarks, pixel_coords=pixel_coords, handedness=label,
                confidence=float(min(0.95, area / (w * h) * 40.0)),
                bounding_box=(int(x), int(y), int(bw), int(bh)),
                fingertip_positions=fingertips, palm_center=center,
                is_gripping=False, velocity=velocity,
                acceleration=acceleration, direction=direction))
            if len(hands_data) >= self.max_hands:
                break
        return hands_data

    def _make_pseudo_landmarks(self, x, y, w, h):
        points = []
        for row in range(5):
            for col in range(4):
                points.append([x + int((col + 0.5) * w / 4), y + int((row + 0.5) * h / 5)])
        points.append([x + w // 2, y + h // 2])
        return np.array(points[:21], dtype=int)

    def _detect_grip_raw(self, landmarks):
        wrist = landmarks[0, :2]
        palm = landmarks[self.PALM_IDS, :2].mean(axis=0)
        hand_scale = np.linalg.norm(landmarks[9, :2] - wrist) + 1e-6
        thumb_tip = landmarks[4, :2]; index_tip = landmarks[8, :2]
        middle_tip = landmarks[12, :2]; ring_tip = landmarks[16, :2]; pinky_tip = landmarks[20, :2]
        finger_tips = [index_tip, middle_tip, ring_tip, pinky_tip]
        tip_distances = [np.linalg.norm(tip - palm) / hand_scale for tip in finger_tips]
        thumb_index_distance = np.linalg.norm(thumb_tip - index_tip) / hand_scale
        folded_count = sum(d < 1.35 for d in tip_distances)
        pinch = thumb_index_distance < 0.65
        return bool(pinch or folded_count >= 3)

    def _smooth_grip(self, label, raw_state):
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

    def _compute_velocity(self, label, position):
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

    def _compute_acceleration(self, label, velocity):
        prev_velocity = self.previous_velocities.get(label, velocity)
        acceleration = velocity - prev_velocity
        self.previous_velocities[label] = velocity
        return float(acceleration)

    def _compute_direction(self, label, position):
        if label not in self.previous_positions:
            return 0.0
        prev = self.previous_positions[label]
        dx = position[0] - prev[0]; dy = position[1] - prev[1]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0
        return float(np.arctan2(dy, dx))

    def get_trajectory_curvature(self, label, window=10):
        points = self.position_history.get(label, [])
        if len(points) < 3:
            return 0.0
        recent = points[-window:]
        if len(recent) < 3:
            return 0.0
        angles = []
        for i in range(1, len(recent) - 1):
            v1 = recent[i] - recent[i - 1]; v2 = recent[i + 1] - recent[i]
            n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                continue
            cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            angles.append(np.arccos(cosang))
        return float(np.mean(angles)) if angles else 0.0