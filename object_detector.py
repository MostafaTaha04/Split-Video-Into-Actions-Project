import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    from ultralytics import YOLOWorld
except Exception:
    YOLOWorld = None


@dataclass
class DetectedObject:
    """Stores detected object/component or semantic work-region information."""
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)
    center: np.ndarray
    area: float
    track_id: Optional[int] = None
    mask: Optional[np.ndarray] = None
    source: str = "unknown"  # workspace, yolo, open_vocab


class ObjectDetector:
    """
    Detect useful objects/regions for action segmentation.

    Modes:
    - workspace: hand-crafted semantic regions, no neural object recognition.
    - open_vocab: YOLO-World text-prompt detector for hardware classes.
    - yolo: custom trained YOLO .pt model.
    - hybrid: open_vocab/custom model + workspace regions.
    - none: no object detection.
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
        confidence: float = 0.20,
        tool_classes: Optional[List[str]] = None,
        detector_mode: str = "workspace",
        open_vocab_model_path: str = "yolov8s-worldv2.pt",
        open_vocab_interval: int = 3,
        open_vocab_imgsz: int = 1280,
        max_det: int = 50,
    ):
        self.detector_mode = (detector_mode or "workspace").lower()
        self.model_path = model_path
        self.open_vocab_model_path = open_vocab_model_path
        self.open_vocab_interval = max(1, int(open_vocab_interval))
        # Larger inference resolution dramatically improves detection of small
        # hardware parts (screws, clips, connectors) with YOLO-World.
        self.open_vocab_imgsz = max(320, int(open_vocab_imgsz))
        self.max_det = max(1, int(max_det))
        self.confidence = confidence
        self.tool_classes = tool_classes or []

        self.model = None
        self.open_vocab_model = None
        self.class_names: Dict[int, str] = {}
        self.open_vocab_names: Dict[int, str] = {
            i: name for i, name in enumerate(self.tool_classes)
        }

        self.previous_gray: Optional[np.ndarray] = None
        self.previous_detections: List[DetectedObject] = []
        self.previous_open_vocab_detections: List[DetectedObject] = []
        self._ov_initialized = False
        self.detection_history: List[List[DetectedObject]] = []
        self.active_tools: Set[str] = set()
        self.tool_presence_history: List[Set[str]] = []
        self.call_count = 0

        if self.detector_mode in ("yolo", "hybrid"):
            self._try_load_yolo()

        if self.detector_mode in ("open_vocab", "hybrid"):
            self._try_load_open_vocab()

        if self.detector_mode == "yolo" and self.model is None:
            print("WARNING: YOLO mode requested, but no usable YOLO model was loaded.")
            print("         Falling back to workspace detector.")
            self.detector_mode = "workspace"

        if self.detector_mode == "open_vocab" and self.open_vocab_model is None:
            print("WARNING: open_vocab mode requested, but YOLO-World could not be loaded.")
            print("         Falling back to workspace detector.")
            self.detector_mode = "workspace"

    def _try_load_yolo(self):
        """Load a custom YOLO detector if a valid .pt file exists."""
        if YOLO is None:
            print("WARNING: ultralytics YOLO is not installed. YOLO mode is unavailable.")
            return

        if not self.model_path:
            return

        if not os.path.exists(self.model_path):
            print(f"WARNING: model file not found: {self.model_path}")
            return

        self.model = YOLO(self.model_path)
        self.class_names = self.model.names

        names = (
            set(self.class_names.values())
            if isinstance(self.class_names, dict)
            else set(self.class_names)
        )

        if len(names.intersection(self.COCO_NAMES)) > 50:
            print("WARNING: The loaded YOLO model looks like a COCO model.")
            print("         COCO does not contain motherboard/cpu/socket hardware classes.")
            print("         Use --detector open_vocab or train a custom hardware_model.pt.")

    def _try_load_open_vocab(self):
        """Load YOLO-World and attach hardware text prompts."""
        if YOLOWorld is None:
            print("WARNING: YOLOWorld could not be imported from ultralytics.")
            print("         Run: python -m pip install --upgrade ultralytics")
            return

        try:
            model_name = self.model_path or self.open_vocab_model_path
            self.open_vocab_model = YOLOWorld(model_name)

            prompts = [
                c for c in self.tool_classes
                if c not in self.WORKSPACE_CLASSES
            ]

            if not prompts:
                prompts = [
                    "motherboard",
                    "cpu",
                    "cpu socket",
                    "screw",
                    "screwdriver",
                    "cooling fan",
                ]

            self.open_vocab_model.set_classes(prompts)
            self.open_vocab_names = {i: name for i, name in enumerate(prompts)}

            print(
                "ObjectDetector: YOLO-World open-vocabulary mode enabled "
                f"with {len(prompts)} prompts."
            )

        except Exception as exc:
            print("WARNING: YOLO-World could not be initialized.")
            print(f"         Reason: {exc}")
            self.open_vocab_model = None

    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        """Run selected detector on a frame."""
        self.call_count += 1

        if self.detector_mode == "none":
            detections: List[DetectedObject] = []

        elif self.detector_mode == "workspace":
            detections = self._detect_workspace_regions(frame)

        elif self.detector_mode == "open_vocab":
            detections = (
                self._detect_open_vocab(frame) +
                self._detect_workspace_regions(frame)
            )

        elif self.detector_mode == "yolo":
            detections = self._detect_yolo(frame)

        elif self.detector_mode == "hybrid":
            detections = []
            detections.extend(self._detect_yolo(frame))
            detections.extend(self._detect_open_vocab(frame))
            detections.extend(self._detect_workspace_regions(frame))

        else:
            print(f"WARNING: unknown detector mode '{self.detector_mode}'. Using workspace mode.")
            self.detector_mode = "workspace"
            detections = self._detect_workspace_regions(frame)

        detections = self._deduplicate_detections(detections)
        self._update_history(detections)

        return detections

    def detect_with_tracking(self, frame: np.ndarray) -> List[DetectedObject]:
        """Compatibility method. For open_vocab/workspace, normal detection is used."""
        if self.detector_mode in ("workspace", "open_vocab", "none") or self.model is None:
            return self.detect(frame)

        results = self.model.track(
            frame,
            conf=self.confidence,
            persist=True,
            verbose=False,
        )

        detections = self._parse_yolo_results(
            results,
            with_track_id=True,
            source="yolo",
        )

        self._update_history(detections)
        return detections

    def _detect_yolo(self, frame: np.ndarray) -> List[DetectedObject]:
        if self.model is None:
            return []

        results = self.model(frame, conf=self.confidence, verbose=False)

        return self._parse_yolo_results(
            results,
            with_track_id=False,
            source="yolo",
        )

    def _detect_open_vocab(self, frame: np.ndarray) -> List[DetectedObject]:
        """Run YOLO-World every N frames and reuse previous detections between calls."""
        if self.open_vocab_model is None:
            return []

        # Deterministic schedule: run on the 1st call and then every Nth call.
        # (call_count was already incremented in detect(), so it starts at 1.)
        run_now = ((self.call_count - 1) % self.open_vocab_interval == 0)

        # Reuse last result between scheduled runs (cache empty results too, so
        # the schedule stays strictly "every N frames").
        if not run_now and self._ov_initialized:
            return list(self.previous_open_vocab_detections)

        try:
            results = self.open_vocab_model.predict(
                frame,
                conf=self.confidence,
                imgsz=self.open_vocab_imgsz,
                max_det=self.max_det,
                verbose=False,
            )

            detections = self._parse_yolo_results(
                results,
                with_track_id=False,
                source="open_vocab",
            )

            self.previous_open_vocab_detections = detections
            self._ov_initialized = True

            return detections

        except Exception as exc:
            print(f"WARNING: YOLO-World inference failed: {exc}")
            return list(self.previous_open_vocab_detections)

    def _parse_yolo_results(
        self,
        results,
        with_track_id: bool = False,
        source: str = "yolo",
    ) -> List[DetectedObject]:
        detections: List[DetectedObject] = []

        for result in results:
            boxes = result.boxes

            if boxes is None:
                continue

            names = (
                getattr(result, "names", None)
                or (self.open_vocab_names if source == "open_vocab" else self.class_names)
            )

            for box in boxes:
                cls_id = int(box.cls[0])

                if isinstance(names, dict):
                    class_name = names.get(
                        cls_id,
                        self.open_vocab_names.get(cls_id, str(cls_id)),
                    )
                else:
                    class_name = str(cls_id)

                conf = float(box.conf[0])

                if conf < self.confidence:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=float)
                area = float(max(0, x2 - x1) * max(0, y2 - y1))

                track_id = (
                    int(box.id[0])
                    if with_track_id and box.id is not None
                    else None
                )

                detections.append(DetectedObject(
                    class_name=str(class_name),
                    confidence=conf,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    center=center,
                    area=area,
                    track_id=track_id,
                    source=source,
                ))

        return detections

    def _detect_workspace_regions(self, frame: np.ndarray) -> List[DetectedObject]:
        detections: List[DetectedObject] = []

        workspace_box = self._estimate_workspace_box(frame)

        detections.append(
            self._make_detection(
                "motherboard_workspace",
                workspace_box,
                0.78,
                source="workspace",
            )
        )

        socket_box = self._estimate_cpu_socket_box(frame, workspace_box)

        detections.append(
            self._make_detection(
                "cpu_socket_region",
                socket_box,
                0.55,
                source="workspace",
            )
        )

        motion_box = self._estimate_active_motion_box(frame)

        if motion_box is not None:
            detections.append(
                self._make_detection(
                    "active_motion_region",
                    motion_box,
                    0.65,
                    source="workspace",
                )
            )

        return detections

    def _estimate_workspace_box(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        h, w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(gray, 40, 120)

        kernel = np.ones((9, 9), np.uint8)

        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

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
            best = (
                int(0.05 * w),
                int(0.10 * h),
                int(0.95 * w),
                int(0.96 * h),
            )

        return self._expand_box(best, frame.shape, pad_ratio=0.04)

    def _estimate_cpu_socket_box(
        self,
        frame: np.ndarray,
        workspace_box: Tuple[int, int, int, int],
    ) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = workspace_box

        ww = x2 - x1
        wh = y2 - y1

        sx1 = x1 + int(0.24 * ww)
        sy1 = y1 + int(0.18 * wh)
        sx2 = x1 + int(0.72 * ww)
        sy2 = y1 + int(0.74 * wh)

        return self._expand_box((sx1, sy1, sx2, sy2), frame.shape, pad_ratio=0.00)

    def _estimate_active_motion_box(
        self,
        frame: np.ndarray,
    ) -> Optional[Tuple[int, int, int, int]]:
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

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        boxes = []

        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)

            if bw * bh > 0.003 * w * h:
                boxes.append((x, y, x + bw, y + bh))

        if not boxes:
            return None

        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)

        return self._expand_box((x1, y1, x2, y2), frame.shape, pad_ratio=0.05)

    def _deduplicate_detections(
        self,
        detections: List[DetectedObject],
        iou_threshold: float = 0.75,
    ) -> List[DetectedObject]:
        """Remove duplicate boxes, keeping real open-vocab/custom objects over workspace boxes."""
        if not detections:
            return []

        source_priority = {
            "open_vocab": 3,
            "yolo": 3,
            "workspace": 1,
            "unknown": 0,
        }

        detections = sorted(
            detections,
            key=lambda d: (
                source_priority.get(d.source, 0),
                d.confidence,
            ),
            reverse=True,
        )

        kept: List[DetectedObject] = []

        for det in detections:
            duplicate = False

            for prev in kept:
                same_name = det.class_name.lower() == prev.class_name.lower()

                if same_name and self._compute_iou(det.bbox, prev.bbox) > iou_threshold:
                    duplicate = True
                    break

            if not duplicate:
                kept.append(det)

        return kept

    @staticmethod
    def _expand_box(
        box: Tuple[int, int, int, int],
        shape,
        pad_ratio: float = 0.05,
    ) -> Tuple[int, int, int, int]:
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
    def _make_detection(
        class_name: str,
        box: Tuple[int, int, int, int],
        confidence: float,
        source: str = "workspace",
    ) -> DetectedObject:
        x1, y1, x2, y2 = box

        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=float)
        area = float(max(0, x2 - x1) * max(0, y2 - y1))

        return DetectedObject(
            class_name=class_name,
            confidence=confidence,
            bbox=(int(x1), int(y1), int(x2), int(y2)),
            center=center,
            area=area,
            source=source,
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
        if not self.tool_classes:
            return detections

        allowed = {c.lower() for c in self.tool_classes}

        return [
            d for d in detections
            if d.class_name.lower() in allowed
        ]

    def get_new_objects(
        self,
        detections: List[DetectedObject],
        iou_threshold: float = 0.5,
    ) -> List[DetectedObject]:
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

    def get_disappeared_objects(
        self,
        detections: List[DetectedObject],
        iou_threshold: float = 0.5,
    ) -> List[DetectedObject]:
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