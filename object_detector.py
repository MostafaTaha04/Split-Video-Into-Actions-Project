import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Set
from ultralytics import YOLO


@dataclass
class DetectedObject:
    """Stores detected object information."""
    class_name: str
    confidence: float
    bbox: tuple
    center: np.ndarray
    area: float
    track_id: Optional[int] = None
    mask: Optional[np.ndarray] = None


class ObjectDetector:
    """Detects tools and hardware components using YOLOv8."""

    def __init__(self, model_path: str = "yolov8n.pt",
                 confidence: float = 0.5,
                 tool_classes: Optional[List[str]] = None):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.tool_classes = tool_classes or []
        self.class_names = self.model.names
        self.previous_detections: List[DetectedObject] = []
        self.detection_history: List[List[DetectedObject]] = []
        self.active_tools: Set[str] = set()
        self.tool_presence_history: List[Set[str]] = []

    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        """Run object detection on a frame."""
        results = self.model(frame, conf=self.confidence, verbose=False)

        detections = []
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
                area = (x2 - x1) * (y2 - y1)

                detection = DetectedObject(
                    class_name=class_name,
                    confidence=conf,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    center=center,
                    area=area
                )
                detections.append(detection)

        current_tools = set(d.class_name for d in detections)
        self.active_tools = current_tools
        self.tool_presence_history.append(current_tools)
        if len(self.tool_presence_history) > 120:
            self.tool_presence_history.pop(0)

        self.previous_detections = detections
        self.detection_history.append(detections)
        if len(self.detection_history) > 60:
            self.detection_history.pop(0)

        return detections

    def detect_with_tracking(self, frame: np.ndarray) -> List[DetectedObject]:
        """Run detection with built-in YOLOv8 tracking for persistent IDs."""
        results = self.model.track(frame, conf=self.confidence,
                                    persist=True, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = self.class_names[cls_id]
                conf = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else None

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                area = (x2 - x1) * (y2 - y1)

                detection = DetectedObject(
                    class_name=class_name,
                    confidence=conf,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    center=center,
                    area=area,
                    track_id=track_id
                )
                detections.append(detection)

        self.previous_detections = detections
        self.detection_history.append(detections)
        if len(self.detection_history) > 60:
            self.detection_history.pop(0)

        return detections

    def filter_tools(self, detections: List[DetectedObject]) -> List[DetectedObject]:
        """Filter detections to only include tool-related objects."""
        if not self.tool_classes:
            return detections
        return [d for d in detections if d.class_name in self.tool_classes]

    def get_new_objects(self, detections: List[DetectedObject],
                        iou_threshold: float = 0.5) -> List[DetectedObject]:
        """Identify newly appeared objects compared to previous frame."""
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

    def get_disappeared_objects(self, detections: List[DetectedObject],
                                 iou_threshold: float = 0.5) -> List[DetectedObject]:
        """Identify objects that disappeared from previous frame."""
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

    def _compute_iou(self, box1: tuple, box2: tuple) -> float:
        """Compute Intersection over Union between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def get_tool_change_score(self) -> float:
        """Score how much the tool set changed in recent history."""
        if len(self.detection_history) < 2:
            return 0.0

        recent = self.detection_history[-1]
        previous = self.detection_history[-2]

        recent_classes = set(d.class_name for d in recent)
        previous_classes = set(d.class_name for d in previous)

        if not recent_classes and not previous_classes:
            return 0.0

        symmetric_diff = recent_classes.symmetric_difference(previous_classes)
        union = recent_classes.union(previous_classes)

        return len(symmetric_diff) / len(union) if union else 0.0

    def get_tool_stability(self, window: int = 30) -> float:
        """Measure how stable the tool set has been over recent frames."""
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
