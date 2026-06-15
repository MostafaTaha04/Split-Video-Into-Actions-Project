from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # Video settings
    input_video_path: str = "input_video.mp4"
    output_dir: str = "output/"
    frame_resize: tuple = (640, 480)
    fps_target: int = 15

    # Hand tracking
    hand_detection_confidence: float = 0.7
    hand_tracking_confidence: float = 0.6
    max_hands: int = 2

    # Object detection
    object_model_path: str = "yolov8n.pt"
    object_confidence: float = 0.5
    tool_classes: List[str] = field(default_factory=lambda: [
        "screwdriver", "wrench", "pliers", "hammer",
        "drill", "knife", "scissors", "bottle"
    ])

    # Interaction tracking
    interaction_distance_threshold: int = 50
    interaction_iou_threshold: float = 0.3

    # Optical flow
    optical_flow_enabled: bool = True
    flow_discontinuity_weight: float = 0.3
    flow_window: int = 15

    # Scene change detection
    scene_detection_enabled: bool = True
    scene_change_threshold: float = 0.6
    scene_change_weight: float = 0.2

    # Segmentation
    window_size: int = 30
    boundary_threshold: float = 0.4
    min_segment_duration: float = 2.0
    smoothing_sigma: float = 3.0

    # Evaluation
    ground_truth_path: Optional[str] = None
    boundary_tolerance: float = 1.0

    # Visualization
    draw_hands: bool = True
    draw_objects: bool = True
    draw_interactions: bool = True
    draw_optical_flow: bool = True
    export_clips: bool = True
