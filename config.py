from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class FeatureParams:
    """Tunable thresholds and weights for per-frame feature fusion.

    These were previously hard-coded inside ``feature_extractor.py``. They are
    centralised here, named and commented, so the segmentation sensitivity can
    be inspected and tuned in one place. The defaults reproduce the original
    behaviour exactly. Two groups:

    * ``activity_*`` weights/normalisers build the 0..1 *activity level*.
    * ``cue_*`` strengths and ``*_thresh`` triggers build the *transition score*
      (the per-frame evidence that a step boundary occurred); the score is the
      strongest active cue.
    """

    # --- Activity-level composition (weights sum-capped at 1.0) ---
    activity_real_tool_bonus: float = 0.10      # a real (non-ROI) component is visible
    activity_velocity_weight: float = 0.25
    activity_velocity_norm: float = 45.0        # hand velocity that saturates the term
    activity_flow_weight: float = 0.20
    activity_flow_norm: float = 8.0             # flow magnitude that saturates the term
    activity_interaction_weight: float = 0.18
    activity_density_weight: float = 0.12
    activity_hand_presence_weight: float = 0.10
    activity_grip_weight: float = 0.08
    activity_flow_nonuniform_weight: float = 0.07

    # --- Transition cue strengths (each is the score contributed when it fires) ---
    cue_hands_change: float = 0.65
    cue_real_component_change: float = 0.68
    cue_grip_change: float = 0.50
    cue_interaction_change: float = 0.55
    cue_activity_active_change: float = 0.55
    cue_motion_onset: float = 0.55              # motion resumes after a calm spell
    cue_motion_settle: float = 0.50             # motion settles after a burst
    cue_velocity_drop: float = 0.58
    cue_acceleration: float = 0.45
    cue_tool_changed: float = 0.75
    cue_flow_discontinuity_extra: float = 0.35  # added to flow_weight, capped at 0.8
    cue_direction_change_extra: float = 0.20    # added to flow_weight, capped at 0.65
    cue_flow_uniformity_drop: float = 0.48
    cue_curvature: float = 0.35

    # --- Transition cue trigger thresholds ---
    activity_active_thresh: float = 0.28        # "active" vs "idle" activity cutoff
    onset_recent_window: int = 8                 # frames of recent history examined
    onset_recent_min_frames: int = 5
    onset_calm_mean: float = 0.20               # recent activity below this == calm
    onset_resume_activity: float = 0.33         # current activity above this == resumed
    settle_busy_mean: float = 0.33              # recent activity above this == busy
    settle_low_activity: float = 0.18           # current activity below this == settled
    velocity_drop_min_avg: float = 5.0
    velocity_drop_ratio: float = 0.35
    acceleration_thresh: float = 25.0
    contact_shift_thresh: float = 90.0
    contact_shift_norm: float = 180.0
    flow_discontinuity_thresh: float = 1.4
    direction_change_thresh: float = 1.3
    scene_change_thresh: float = 0.35
    flow_uniformity_low: float = 0.35
    flow_uniformity_prev_high: float = 0.65
    curvature_thresh: float = 0.9

    # Caps applied to flow-derived cues.
    cue_flow_discontinuity_cap: float = 0.8
    cue_direction_change_cap: float = 0.65
    cue_scene_change_cap: float = 0.8
    cue_contact_shift_cap: float = 0.75


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

    detector_debug: bool = False
    max_frames: Optional[int] = None

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

    # Per-frame feature fusion thresholds/weights (centralised; see FeatureParams).
    feature_params: FeatureParams = field(default_factory=FeatureParams)

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
