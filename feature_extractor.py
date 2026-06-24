import numpy as np
from dataclasses import dataclass, field
from typing import List
from collections import deque

from hand_tracker import HandData
from object_detector import DetectedObject
from interaction_tracker import Interaction
from optical_flow import FlowData
from scene_detector import SceneChangeData


@dataclass
class FrameFeatures:
    """Aggregated features for a single frame."""
    frame_idx: int
    timestamp: float

    # Hand features
    hand_velocity_left: float = 0.0
    hand_velocity_right: float = 0.0
    hand_acceleration_left: float = 0.0
    hand_acceleration_right: float = 0.0
    hands_present: int = 0
    grip_state_left: bool = False
    grip_state_right: bool = False
    hand_distance: float = 0.0
    hand_direction_left: float = 0.0
    hand_direction_right: float = 0.0
    trajectory_curvature: float = 0.0

    # Object/component features
    num_objects: int = 0
    num_tools: int = 0
    tool_changed: bool = False
    dominant_tool: str = ""
    visible_tools: List[str] = field(default_factory=list)
    tool_stability: float = 1.0

    # Interaction features
    num_interactions: int = 0
    interaction_type: str = "none"
    contact_point_shift: float = 0.0
    contact_point_variance: float = 0.0
    interaction_density: float = 0.0
    interaction_rhythm: float = 0.0

    # Optical flow features
    flow_magnitude: float = 0.0
    flow_direction: float = 0.0
    flow_uniformity: float = 1.0
    flow_discontinuity: float = 0.0
    motion_energy: float = 0.0
    direction_change: float = 0.0

    # Scene features
    scene_change_score: float = 0.0
    visual_stability: float = 1.0

    # Composite signals
    activity_level: float = 0.0
    transition_score: float = 0.0


class FeatureExtractor:
    """Extracts and combines frame-level features for temporal segmentation."""

    def __init__(
        self,
        window_size: int = 30,
        flow_weight: float = 0.35,
        scene_weight: float = 0.25,
        params=None,
    ):
        from config import FeatureParams

        self.window_size = window_size
        self.flow_weight = flow_weight
        self.scene_weight = scene_weight
        # Centralised, documented thresholds/weights (defaults reproduce the
        # original hard-coded behaviour exactly).
        self.p = params or FeatureParams()
        self.feature_history: List[FrameFeatures] = []

        self.velocity_buffer_left = deque(maxlen=window_size)
        self.velocity_buffer_right = deque(maxlen=window_size)
        self.activity_buffer = deque(maxlen=window_size)
        self.flow_buffer = deque(maxlen=window_size)

        self.prev_tool = ""
        self.prev_tool_set = set()

        self.transition_score_history = deque(maxlen=window_size * 3)

    def extract(
        self,
        frame_idx: int,
        timestamp: float,
        hands: List[HandData],
        objects: List[DetectedObject],
        interactions: List[Interaction],
        contact_shift: float,
        contact_variance: float,
        interaction_density: float,
        interaction_rhythm: float,
        flow_data: FlowData,
        scene_data: SceneChangeData,
        tool_stability: float,
        trajectory_curvature: float,
    ) -> FrameFeatures:
        features = FrameFeatures(frame_idx=frame_idx, timestamp=timestamp)

        self._extract_hand_features(features, hands, trajectory_curvature)
        self._extract_object_features(features, objects, tool_stability)
        self._extract_interaction_features(
            features,
            interactions,
            contact_shift,
            contact_variance,
            interaction_density,
            interaction_rhythm,
        )
        self._extract_flow_features(features, flow_data)
        self._extract_scene_features(features, scene_data)
        self._compute_composite_signals(features)

        self.feature_history.append(features)
        self.transition_score_history.append(features.transition_score)
        self.activity_buffer.append(features.activity_level)
        self.flow_buffer.append(features.flow_magnitude)

        return features

    def _extract_hand_features(
        self,
        features: FrameFeatures,
        hands: List[HandData],
        trajectory_curvature: float,
    ):
        features.hands_present = len(hands)
        features.trajectory_curvature = trajectory_curvature

        for hand in hands:
            if hand.handedness == "Left":
                features.hand_velocity_left = hand.velocity
                features.grip_state_left = hand.is_gripping
                features.hand_acceleration_left = hand.acceleration
                features.hand_direction_left = hand.direction
                self.velocity_buffer_left.append(hand.velocity)
            else:
                features.hand_velocity_right = hand.velocity
                features.grip_state_right = hand.is_gripping
                features.hand_acceleration_right = hand.acceleration
                features.hand_direction_right = hand.direction
                self.velocity_buffer_right.append(hand.velocity)

        if len(hands) == 2:
            features.hand_distance = float(
                np.linalg.norm(hands[0].palm_center - hands[1].palm_center)
            )

    def _extract_object_features(
        self,
        features: FrameFeatures,
        objects: List[DetectedObject],
        tool_stability: float,
    ):
        features.num_objects = len(objects)
        features.tool_stability = tool_stability

        tools = [o for o in objects if o.class_name.lower() != "person"]

        features.num_tools = len(tools)
        features.visible_tools = sorted({o.class_name for o in tools})

        if tools:
            # Real detected hardware should dominate over large fallback workspace regions.
            priority = {
                "screwdriver": 100,
                "screw": 95,
                "cooling fan": 90,
                "fan": 88,
                "cpu": 85,
                "processor": 84,
                "computer processor": 84,
                "cpu socket": 82,
                "socket retention lever": 80,
                "socket retention bracket": 78,
                "ram stick": 76,
                "ram": 75,
                "cable": 74,
                "connector": 73,
                "thermal paste": 72,
                "heatsink": 71,
                "motherboard": 60,
                "cpu_socket_region": 45,
                "active_motion_region": 30,
                "motherboard_workspace": 20,
            }

            dominant = max(
                tools,
                key=lambda t: (
                    priority.get(t.class_name.lower(), 0),
                    t.confidence,
                    t.area,
                ),
            )

            features.dominant_tool = dominant.class_name

            current_set = {t.class_name for t in tools}

            features.tool_changed = bool(self.prev_tool_set) and current_set != self.prev_tool_set

            self.prev_tool_set = current_set
            self.prev_tool = dominant.class_name

    def _extract_interaction_features(
        self,
        features: FrameFeatures,
        interactions: List[Interaction],
        contact_shift: float,
        contact_variance: float,
        interaction_density: float,
        interaction_rhythm: float,
    ):
        features.num_interactions = len(interactions)
        features.contact_point_shift = contact_shift
        features.contact_point_variance = contact_variance
        features.interaction_density = interaction_density
        features.interaction_rhythm = interaction_rhythm

        if interactions:
            type_priority = {
                "use": 4,
                "grasp": 3,
                "touch": 2,
                "approach": 1,
            }

            best = max(
                interactions,
                key=lambda i: type_priority.get(i.interaction_type, 0),
            )

            features.interaction_type = best.interaction_type

    def _extract_flow_features(self, features: FrameFeatures, flow_data: FlowData):
        features.flow_magnitude = flow_data.magnitude_mean
        features.flow_direction = flow_data.dominant_direction
        features.flow_uniformity = flow_data.motion_uniformity
        features.flow_discontinuity = flow_data.discontinuity_score
        features.motion_energy = flow_data.magnitude_mean
        features.direction_change = 0.0

        if self.feature_history:
            prev_dir = self.feature_history[-1].flow_direction
            dir_diff = abs(flow_data.dominant_direction - prev_dir)
            features.direction_change = float(min(dir_diff, 2 * np.pi - dir_diff))

    def _extract_scene_features(
        self,
        features: FrameFeatures,
        scene_data: SceneChangeData,
    ):
        features.scene_change_score = scene_data.combined_score
        features.visual_stability = 1.0 - scene_data.combined_score

    def _compute_composite_signals(self, features: FrameFeatures):
        max_velocity = max(features.hand_velocity_left, features.hand_velocity_right)

        max_accel = max(
            abs(features.hand_acceleration_left),
            abs(features.hand_acceleration_right),
        )

        interaction_ratio = features.num_interactions / max(features.num_tools, 1)
        hand_presence_score = min(features.hands_present / 2.0, 1.0)

        p = self.p

        real_tool_bonus = (
            p.activity_real_tool_bonus
            if any(
                t not in {
                    "motherboard_workspace",
                    "cpu_socket_region",
                    "active_motion_region",
                }
                for t in features.visible_tools
            )
            else 0.0
        )

        features.activity_level = min(
            1.0,
            real_tool_bonus +
            p.activity_velocity_weight * min(max_velocity / p.activity_velocity_norm, 1.0) +
            p.activity_flow_weight * min(features.flow_magnitude / p.activity_flow_norm, 1.0) +
            p.activity_interaction_weight * min(interaction_ratio, 1.0) +
            p.activity_density_weight * features.interaction_density +
            p.activity_hand_presence_weight * hand_presence_score +
            p.activity_grip_weight * float(features.grip_state_left or features.grip_state_right) +
            p.activity_flow_nonuniform_weight * (1.0 - features.flow_uniformity),
        )

        transition_signals = []

        roi_set = {
            "motherboard_workspace",
            "cpu_socket_region",
            "active_motion_region",
        }

        if self.feature_history:
            prev = self.feature_history[-1]

            if features.hands_present != prev.hands_present:
                transition_signals.append(p.cue_hands_change)

            # Only treat a change in REAL components as a boundary cue. The
            # workspace ROIs (esp. active_motion_region) flicker frame-to-frame
            # with motion, which previously created spurious boundaries.
            curr_real = {t for t in features.visible_tools if t not in roi_set}
            prev_real = {t for t in prev.visible_tools if t not in roi_set}
            if curr_real != prev_real and (curr_real or prev_real):
                transition_signals.append(p.cue_real_component_change)

            # Grip onset/release reliably marks pick-up / put-down moments.
            # Strengthened now that MediaPipe grip tracking is working.
            if (
                features.grip_state_left != prev.grip_state_left
                or features.grip_state_right != prev.grip_state_right
            ):
                transition_signals.append(p.cue_grip_change)

            if features.interaction_type != prev.interaction_type:
                transition_signals.append(p.cue_interaction_change)

            prev_active = prev.activity_level > p.activity_active_thresh
            curr_active = features.activity_level > p.activity_active_thresh

            if prev_active != curr_active:
                transition_signals.append(p.cue_activity_active_change)

        # Multi-frame activity-phase change: a step often begins when motion
        # resumes after a calm spell, or ends when the hands settle after a
        # burst. This recovers low-motion transitions (e.g. seating a part or
        # closing a lever) that single-frame cues miss.
        recent = self.feature_history[-p.onset_recent_window:]
        if len(recent) >= p.onset_recent_min_frames:
            recent_mean = float(np.mean([f.activity_level for f in recent]))
            if recent_mean < p.onset_calm_mean and features.activity_level > p.onset_resume_activity:
                transition_signals.append(p.cue_motion_onset)   # onset after a pause
            elif recent_mean > p.settle_busy_mean and features.activity_level < p.settle_low_activity:
                transition_signals.append(p.cue_motion_settle)   # settling after a burst

        velocity_values = list(self.velocity_buffer_left) + list(self.velocity_buffer_right)

        if velocity_values:
            avg_vel = float(np.mean(velocity_values))

            if avg_vel > p.velocity_drop_min_avg and max_velocity < avg_vel * p.velocity_drop_ratio:
                transition_signals.append(p.cue_velocity_drop)

        if max_accel > p.acceleration_thresh:
            transition_signals.append(p.cue_acceleration)

        if features.tool_changed:
            transition_signals.append(p.cue_tool_changed)

        if features.contact_point_shift > p.contact_shift_thresh:
            transition_signals.append(
                min(features.contact_point_shift / p.contact_shift_norm, p.cue_contact_shift_cap)
            )

        if features.flow_discontinuity > p.flow_discontinuity_thresh:
            transition_signals.append(
                min(self.flow_weight + p.cue_flow_discontinuity_extra, p.cue_flow_discontinuity_cap)
            )

        if features.direction_change > p.direction_change_thresh:
            transition_signals.append(
                min(self.flow_weight + p.cue_direction_change_extra, p.cue_direction_change_cap)
            )

        if features.scene_change_score > p.scene_change_thresh:
            transition_signals.append(
                min(self.scene_weight + features.scene_change_score, p.cue_scene_change_cap)
            )

        if (
            features.flow_uniformity < p.flow_uniformity_low
            and self.feature_history
            and self.feature_history[-1].flow_uniformity > p.flow_uniformity_prev_high
        ):
            transition_signals.append(p.cue_flow_uniformity_drop)

        if features.trajectory_curvature > p.curvature_thresh:
            transition_signals.append(p.cue_curvature)

        features.transition_score = max(transition_signals) if transition_signals else 0.0

    def get_feature_matrix(self) -> np.ndarray:
        if not self.feature_history:
            return np.array([])

        return np.array([
            [
                f.hand_velocity_left,
                f.hand_velocity_right,
                f.hand_acceleration_left,
                f.hand_acceleration_right,
                float(f.hands_present),
                float(f.grip_state_left),
                float(f.grip_state_right),
                f.hand_distance,
                f.trajectory_curvature,
                float(f.num_tools),
                float(f.tool_changed),
                f.tool_stability,
                float(f.num_interactions),
                f.contact_point_shift,
                f.contact_point_variance,
                f.interaction_density,
                f.interaction_rhythm,
                f.flow_magnitude,
                f.flow_uniformity,
                f.flow_discontinuity,
                f.direction_change,
                f.scene_change_score,
                f.visual_stability,
                f.activity_level,
                f.transition_score,
            ]
            for f in self.feature_history
        ])

    def get_transition_scores(self) -> np.ndarray:
        return np.array([f.transition_score for f in self.feature_history])

    def get_feature_names(self) -> List[str]:
        return [
            "hand_velocity_left",
            "hand_velocity_right",
            "hand_acceleration_left",
            "hand_acceleration_right",
            "hands_present",
            "grip_state_left",
            "grip_state_right",
            "hand_distance",
            "trajectory_curvature",
            "num_tools",
            "tool_changed",
            "tool_stability",
            "num_interactions",
            "contact_point_shift",
            "contact_point_variance",
            "interaction_density",
            "interaction_rhythm",
            "flow_magnitude",
            "flow_uniformity",
            "flow_discontinuity",
            "direction_change",
            "scene_change_score",
            "visual_stability",
            "activity_level",
            "transition_score",
        ]