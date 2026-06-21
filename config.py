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

    # New: smooth MediaPipe grip state to prevent false boundaries
    grip_smoothing_window: int = 5

    # Object / component detection
    object_detector_mode: str = "workspace"
    object_model_path: Optional[str] = None
    object_confidence: float = 0.20
    open_vocab_model_path: str = "yolov8s-worldv2.pt"
    open_vocab_interval: int = 3

    # Text prompts/classes for hardware assembly
    tool_classes: List[str] = field(default_factory=lambda: [
        # Heuristic ROIs
        "motherboard_workspace",
        "cpu_socket_region",
        "active_motion_region",

        # Real hardware/tool prompts
        "motherboard",
        "cpu",
        "computer processor",
        "processor",
        "cpu socket",
        "socket retention lever",
        "socket retention bracket",
        "screw",
        "screwdriver",
        "cooling fan",
        "fan",
        "ram stick",
        "ram",
        "ssd",
        "cable",
        "connector",
        "heatsink",
        "thermal paste",
    ])

    # These are heuristic regions, not true object detections.
    roi_classes: List[str] = field(default_factory=lambda: [
        "motherboard_workspace",
        "cpu_socket_region",
        "active_motion_region",
    ])

    # Interaction tracking
    # Updated for real MediaPipe boxes, which are tighter than fallback motion blobs.
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