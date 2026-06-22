from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Config:
    """Global configuration for the action-splitting pipeline."""

    # Video settings
    input_video_path: str = "input_video.mp4"
    output_dir: str = "output/"
    frame_resize: Tuple[int, int] = (640, 480)
    fps_target: int = 15

    # Hand tracking
    hand_detection_confidence: float = 0.7
    hand_tracking_confidence: float = 0.6
    max_hands: int = 2

    # Smooth MediaPipe grip state to prevent false boundaries
    grip_smoothing_window: int = 5

    # Optional explicit path to a hand_landmarker.task bundle.
    # If None, HandTracker auto-downloads it on first run.
    hand_model_path: Optional[str] = None

    # Object / component detection
    object_detector_mode: str = "workspace"
    object_model_path: Optional[str] = None

    # Lowered from 0.20 to 0.08 because YOLO-World may miss small hardware parts.
    object_confidence: float = 0.08

    open_vocab_model_path: str = "yolov8s-worldv2.pt"
    open_vocab_interval: int = 3

    # Inference resolution for YOLO-World. 1280 detects small hardware parts
    # far better than the default 640. Raise frame_resize too (see below) to
    # actually feed those pixels in.
    open_vocab_imgsz: int = 1280
    max_det: int = 50

    # Text prompts/classes for hardware assembly.
    # These are used by YOLO-World open-vocabulary detection.
    tool_classes: List[str] = field(default_factory=lambda: [
        # Heuristic ROIs
        "motherboard_workspace",
        "cpu_socket_region",
        "active_motion_region",

        # General motherboard / workspace
        "motherboard",
        "computer motherboard",
        "pc case",
        "computer case",

        # CPU / socket
        "cpu",
        "computer processor",
        "processor",
        "cpu socket",
        "socket retention lever",
        "socket retention bracket",

        # Cooling fan / cooler / heatsink
        "cpu cooler",
        "computer cpu cooler",
        "air cooler",
        "cooling fan",
        "cooler fan",
        "computer fan",
        "fan blades",
        "heatsink",
        "mounting clip",
        "retention clip",
        "mounting bracket",
        "cooler bracket",

        # Screws / screwdriver
        "screw",
        "screwdriver",

        # RAM
        "ram stick",
        "ram",
        "memory module",
        "ram slot",

        # SSD
        "ssd",
        "m.2 ssd",
        "nvme ssd",

        # Cable / connector
        "cable",
        "fan cable",
        "power cable",
        "connector",
        "plug",
        "header",

        # Thermal
        "thermal paste",
    ])

    # These are heuristic regions, not true object detections.
    roi_classes: List[str] = field(default_factory=lambda: [
        "motherboard_workspace",
        "cpu_socket_region",
        "active_motion_region",
    ])

    # Interaction tracking
    interaction_distance_threshold: int = 90
    interaction_iou_threshold: float = 0.05

    # Optical flow
    optical_flow_enabled: bool = True
    flow_discontinuity_weight: float = 0.35
    flow_window: int = 15

    # Scene change detection
    scene_detection_enabled: bool = True
    scene_change_threshold: float = 0.45
    scene_change_weight: float = 0.25

    # Segmentation
    window_size: int = 30
    boundary_threshold: float = 0.32
    min_segment_duration: float = 1.5
    smoothing_sigma: float = 2.0

    # Evaluation
    ground_truth_path: Optional[str] = None
    boundary_tolerance: float = 1.0

    # Visualization
    draw_optical_flow: bool = False
    export_clips: bool = True