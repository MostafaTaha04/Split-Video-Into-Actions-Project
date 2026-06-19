import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO  # optional; only needed for --detector yolo/hybrid
except Exception:  # pragma: no cover - this is allowed in workspace mode
    YOLO = None


@dataclass
class DetectedObject:
    """Stores detected object or semantic work-region information."""
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)
    center: np.ndarray
    area: float
    track_id: Optional[int] = None
    mask: Optional[np.ndarray] = None


class ObjectDetector:
    """
    Detects useful regions for action segmentation.

    Why this file is different from the original:
    - The original code expected a missing custom model: hardware_model.pt.
    - yolov8n.pt is trained on COCO objects and cannot recognize CPU/motherboard parts.
    - For your test_video_01.mp4, the correct fallback is a visual workspace detector.

    Modes:
    - workspace: no YOLO. Detects motherboard/work area + active motion region.
    - yolo: uses a custom YOLO model only. Use this when you really have hardware_model.pt.
    - hybrid: YOLO if available, plus workspace regions.
    - none: returns no objects.
    """

    WORKSPACE_CLASSES = {
        "motherboard_workspace",
        "cpu_socket_region",
        "active_motion_region",
    }

    COCO_NAMES = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
        "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
        "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush",
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = 0.35,
        tool_classes: Optional[List[str]] = None,
        detector_mode: str = "workspace",
    ):
        self.detector_mode = (detector_mode or "workspace").lower()
        self.model_path = model_path
        self.confidence = confidence
        self.tool_classes = tool_classes or []

        self.model = None
        self.class_names: Dict[int, str] = {}

        self.previous_gray: Optional[np.ndarray] = None
        self.previous_detections: List[DetectedObject] = []
        self.detection_history: List[List[DetectedObject]] = []
        self.active_tools: Set[str] = set()
        self.tool_presence_history: List[Set[str]] = []

        if self.detector_mode in ("yolo", "hybrid"):
            self._try_load_yolo()
            if self.model is None and self.detector_mode == "yolo":
                print("WARNING: YOLO mode requested, but no usable YOLO model was loaded.")
                print("         Falling back to workspace detector.")
                self.detector_mode = "workspace"

    def _try_load_yolo(self):
        """Load YOLO only when a valid custom model is available."""
        if YOLO is None:
            print("WARNING: ultralytics is not installed. Workspace detector will be used.")
            return

        if not self.model_path:
            print("WARNING: no --model path was provided. Workspace detector will be used.")
            return

        if not os.path.exists(self.model_path):
            print(f"WARNING: model file not found: {self.model_path}")
            print("         Workspace detector will be used instead.")
            return

        self.model = YOLO(self.model_path)
        self.class_names = self.model.names

        # Warn if the model looks like stock COCO, because it will not detect CPU/motherboard.
        names = set(self.class_names.values()) if isinstance(self.class_names, dict) else set(self.class_names)
        if len(names.intersection(self.COCO_NAMES)) > 50:
            print("WARNING: The loaded YOLO model looks like a COCO model.")
            print("         COCO does not contain motherboard/cpu/screwdriver classes.")
            print("         Use --detector workspace or train a custom hardware_model.pt.")

    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        """Run the selected detector on one frame."""
        if self.detector_mode == "none":
            detections: List[DetectedObject] = []
        elif self.detector_mode == "workspace":
            detections = self._detect_workspace_regions(frame)
        elif self.detector_mode == "yolo":
            detections = self._detect_yolo(frame)
        elif self.detector_mode == "hybrid":
            detections = self._detect_yolo(frame) + self._detect_workspace_regions(frame)
        else:
            print(f"WARNING: unknown detector mode '{self.detector_mode}'. Using workspace mode.")
            self.detector_mode = "workspace"
            detections = self._detect_workspace_regions(frame)

        self._update_history(detections)
        return detections

    def detect_with_tracking(self, frame: np.ndarray) -> List[DetectedObject]:
        """Compatibility method. Workspace mode has no track IDs."""
        if self.detector_mode in ("workspace", "none") or self.model is None:
            return self.detect(frame)

        results = self.model.track(frame, conf=self.confidence, persist=True, verbose=False)
        detections = self._parse_yolo_results(results, with_track_id=True)
        self._update_history(detections)
        return detections

    def _detect_yolo(self, frame: np.ndarray) -> List[DetectedObject]:
        if self.model is None:
            return []
        results = self.model(frame, conf=self.confidence, verbose=False)
        return self._parse_yolo_results(results, with_track_id=False)

    def _parse_yolo_results(self, results, with_track_id: bool = False) -> List[DetectedObject]:
        detections: List[DetectedObject] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = self.class_names[cls_id]
                conf = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                area = float((x2 - x1) * (y2 - y1))
                track_id = int(box.id[0]) if with_track_id and box.id is not None else None

                detections.append(DetectedObject(
                    class_name=class_name,
                    confidence=conf,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    center=center,
                    area=area,
                    track_id=track_id,
                ))
        return detections

    def _detect_workspace_regions(self, frame: np.ndarray) -> List[DetectedObject]:
        """
        Create useful semantic regions for PC-building / motherboard videos.

        This is not object recognition. It gives the segmenter a stable region that hands can
        interact with, plus a changing motion region. This fixes the dead feature columns
        num_tools=0 and num_interactions=0 when no custom YOLO hardware model exists.
        """
        detections: List[DetectedObject] = []

        workspace_box = self._estimate_workspace_box(frame)
        detections.append(self._make_detection("motherboard_workspace", workspace_box, 0.78))

        socket_box = self._estimate_cpu_socket_box(frame, workspace_box)
        detections.append(self._make_detection("cpu_socket_region", socket_box, 0.55))

        motion_box = self._estimate_active_motion_box(frame)
        if motion_box is not None:
            detections.append(self._make_detection("active_motion_region", motion_box, 0.65))

        return detections

    def _estimate_workspace_box(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """Estimate a stable motherboard/workbench region from edges and texture."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(gray, 40, 120)
        kernel = np.ones((9, 9), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 0.12 * w * h
        best = None
        best_area = 0

        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            if area > min_area and area > best_area:
                best = (x, y, x + bw, y + bh)
                best_area = area

        if best is None:
            # Robust fallback for your top-down motherboard assembly video.
            best = (int(0.05 * w), int(0.10 * h), int(0.95 * w), int(0.96 * h))

        return self._expand_box(best, frame.shape, pad_ratio=0.04)

    def _estimate_cpu_socket_box(
        self,
        frame: np.ndarray,
        workspace_box: Tuple[int, int, int, int],
    ) -> Tuple[int, int, int, int]:
        """
        Approximate the CPU/socket working zone.

        In the absence of a trained detector, this central sub-region works well for close-up
        motherboard videos where the action is around the socket/retention latch.
        """
        x1, y1, x2, y2 = workspace_box
        ww = x2 - x1
        wh = y2 - y1
        sx1 = x1 + int(0.24 * ww)
        sy1 = y1 + int(0.18 * wh)
        sx2 = x1 + int(0.72 * ww)
        sy2 = y1 + int(0.74 * wh)
        return self._expand_box((sx1, sy1, sx2, sy2), frame.shape, pad_ratio=0.00)

    def _estimate_active_motion_box(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect the region that changed compared with the previous processed frame."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.previous_gray is None:
            self.previous_gray = gray
            return None

        diff = cv2.absdiff(gray, self.previous_gray)
        self.previous_gray = gray

        _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        boxes = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            if area > 0.003 * w * h:
                boxes.append((x, y, x + bw, y + bh))

        if not boxes:
            return None

        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)
        return self._expand_box((x1, y1, x2, y2), frame.shape, pad_ratio=0.05)

    @staticmethod
    def _expand_box(box: Tuple[int, int, int, int], shape, pad_ratio: float = 0.05) -> Tuple[int, int, int, int]:
        h, w = shape[:2]
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1
        pad_x = int(bw * pad_ratio)
        pad_y = int(bh * pad_ratio)
        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(w - 1, x2 + pad_x),
            min(h - 1, y2 + pad_y),
        )

    @staticmethod
    def _make_detection(class_name: str, box: Tuple[int, int, int, int], confidence: float) -> DetectedObject:
        x1, y1, x2, y2 = box
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=float)
        area = float(max(0, x2 - x1) * max(0, y2 - y1))
        return DetectedObject(
            class_name=class_name,
            confidence=confidence,
            bbox=(int(x1), int(y1), int(x2), int(y2)),
            center=center,
            area=area,
        )

    def _update_history(self, detections: List[DetectedObject]):
        current_tools = {d.class_name for d in detections}
        self.active_tools = current_tools
        self.tool_presence_history.append(current_tools)
        if len(self.tool_presence_history) > 120:
            self.tool_presence_history.pop(0)

        self.previous_detections = detections
        self.detection_history.append(detections)
        if len(self.detection_history) > 60:
            self.detection_history.pop(0)

    def filter_tools(self, detections: List[DetectedObject]) -> List[DetectedObject]:
        """Filter detections to classes useful for this project."""
        if not self.tool_classes:
            return detections
        allowed = set(self.tool_classes)
        return [d for d in detections if d.class_name in allowed]

    def get_new_objects(self, detections: List[DetectedObject], iou_threshold: float = 0.5) -> List[DetectedObject]:
        """Identify newly appeared objects compared to the previous frame."""
        if not self.previous_detections:
            return detections

        new_objects = []
        for det in detections:
            is_new = True
            for prev in self.previous_detections:
                if self._compute_iou(det.bbox, prev.bbox) > iou_threshold:
                    is_new = False
                    break
            if is_new:
                new_objects.append(det)
        return new_objects

    def get_disappeared_objects(self, detections: List[DetectedObject], iou_threshold: float = 0.5) -> List[DetectedObject]:
        """Identify objects that disappeared from the previous frame."""
        disappeared = []
        for prev in self.previous_detections:
            still_present = False
            for det in detections:
                if self._compute_iou(det.bbox, prev.bbox) > iou_threshold:
                    still_present = True
                    break
            if not still_present:
                disappeared.append(prev)
        return disappeared

    @staticmethod
    def _compute_iou(box1: tuple, box2: tuple) -> float:
        """Compute Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

    def get_tool_change_score(self) -> float:
        """Score how much the detected semantic region set changed recently."""
        if len(self.tool_presence_history) < 2:
            return 0.0

        recent = self.tool_presence_history[-1]
        previous = self.tool_presence_history[-2]
        if not recent and not previous:
            return 0.0

        union = recent.union(previous)
        diff = recent.symmetric_difference(previous)
        return len(diff) / len(union) if union else 0.0

    def get_tool_stability(self, window: int = 30) -> float:
        """Measure how stable the detected object/region set has been."""
        if len(self.tool_presence_history) < 2:
            return 1.0

        recent = self.tool_presence_history[-window:]
        if len(recent) < 2:
            return 1.0

        changes = 0
        for i in range(1, len(recent)):
            if recent[i] != recent[i - 1]:
                changes += 1
        return 1.0 - (changes / (len(recent) - 1))