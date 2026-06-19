import numpy as np
from dataclasses import dataclass
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

    # Object / semantic workspace features
    num_objects: int = 0
    num_tools: int = 0
    tool_changed: bool = False
    dominant_tool: str = ""
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

    def __init__(self, window_size: int = 30,
                 flow_weight: float = 0.35,
                 scene_weight: float = 0.25):
        self.window_size = window_size
        self.flow_weight = flow_weight
        self.scene_weight = scene_weight
        self.feature_history: List[FrameFeatures] = []
        self.velocity_buffer_left = deque(maxlen=window_size)
        self.velocity_buffer_right = deque(maxlen=window_size)
        self.activity_buffer = deque(maxlen=window_size)
        self.flow_buffer = deque(maxlen=window_size)
        self.prev_tool = ""
        self.transition_score_history = deque(maxlen=window_size * 3)

    def extract(self, frame_idx: int, timestamp: float,
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
                trajectory_curvature: float) -> FrameFeatures:
        """Extract all features for the current frame."""
        features = FrameFeatures(frame_idx=frame_idx, timestamp=timestamp)

        self._extract_hand_features(features, hands, trajectory_curvature)
        self._extract_object_features(features, objects, tool_stability)
        self._extract_interaction_features(
            features, interactions, contact_shift,
            contact_variance, interaction_density, interaction_rhythm
        )
        self._extract_flow_features(features, flow_data)
        self._extract_scene_features(features, scene_data)
        self._compute_composite_signals(features)

        self.feature_history.append(features)
        self.transition_score_history.append(features.transition_score)
        self.activity_buffer.append(features.activity_level)
        self.flow_buffer.append(features.flow_magnitude)
        return features

    def _extract_hand_features(self, features: FrameFeatures,
                               hands: List[HandData],
                               trajectory_curvature: float):
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
            features.hand_distance = float(np.linalg.norm(
                hands[0].palm_center - hands[1].palm_center
            ))

    def _extract_object_features(self, features: FrameFeatures,
                                 objects: List[DetectedObject],
                                 tool_stability: float):
        features.num_objects = len(objects)
        features.tool_stability = tool_stability

        tools = [o for o in objects if o.class_name != "person"]
        features.num_tools = len(tools)

        if tools:
            # Prefer smaller active/CPU regions over the large workspace for semantic labels.
            priority = {
                "active_motion_region": 3,
                "cpu_socket_region": 2,
                "motherboard_workspace": 1,
            }
            dominant = max(tools, key=lambda t: (priority.get(t.class_name, 0), t.area))
            features.dominant_tool = dominant.class_name
            features.tool_changed = (
                dominant.class_name != self.prev_tool and self.prev_tool != ""
            )
            self.prev_tool = dominant.class_name

    def _extract_interaction_features(self, features: FrameFeatures,
                                      interactions: List[Interaction],
                                      contact_shift: float,
                                      contact_variance: float,
                                      interaction_density: float,
                                      interaction_rhythm: float):
        features.num_interactions = len(interactions)
        features.contact_point_shift = contact_shift
        features.contact_point_variance = contact_variance
        features.interaction_density = interaction_density
        features.interaction_rhythm = interaction_rhythm

        if interactions:
            type_priority = {"use": 4, "grasp": 3, "touch": 2, "approach": 1}
            best = max(interactions, key=lambda i: type_priority.get(i.interaction_type, 0))
            features.interaction_type = best.interaction_type

    def _extract_flow_features(self, features: FrameFeatures,
                               flow_data: FlowData):
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

    def _extract_scene_features(self, features: FrameFeatures,
                                scene_data: SceneChangeData):
        features.scene_change_score = scene_data.combined_score
        features.visual_stability = 1.0 - scene_data.combined_score

    def _compute_composite_signals(self, features: FrameFeatures):
        """
        Compute high-level action and boundary signals.

        The original version depended heavily on real tool detections. For your CPU/motherboard
        video, this version also uses hand presence, hand velocity, interaction with the semantic
        workspace, optical flow changes, and scene changes.
        """
        max_velocity = max(features.hand_velocity_left, features.hand_velocity_right)
        max_accel = max(abs(features.hand_acceleration_left), abs(features.hand_acceleration_right))

        interaction_ratio = features.num_interactions / max(features.num_tools, 1)
        hand_presence_score = min(features.hands_present / 2.0, 1.0)

        features.activity_level = (
            0.25 * min(max_velocity / 45.0, 1.0) +
            0.20 * min(features.flow_magnitude / 8.0, 1.0) +
            0.18 * min(interaction_ratio, 1.0) +
            0.12 * features.interaction_density +
            0.10 * hand_presence_score +
            0.08 * float(features.grip_state_left or features.grip_state_right) +
            0.07 * (1.0 - features.flow_uniformity)
        )

        transition_signals = []

        # 1) Hands appear/disappear: usually a real phase transition in assembly videos.
        if self.feature_history:
            prev = self.feature_history[-1]
            if features.hands_present != prev.hands_present:
                transition_signals.append(0.65)

            if (features.grip_state_left != prev.grip_state_left or
                    features.grip_state_right != prev.grip_state_right):
                transition_signals.append(0.42)

            if features.interaction_type != prev.interaction_type:
                transition_signals.append(0.55)

            # Activity regime change: moving/manipulating <-> pause/inspection.
            prev_active = prev.activity_level > 0.28
            curr_active = features.activity_level > 0.28
            if prev_active != curr_active:
                transition_signals.append(0.55)

        # 2) Velocity dip after a moving window: often the hand finishes one sub-action.
        velocity_values = list(self.velocity_buffer_left) + list(self.velocity_buffer_right)
        if velocity_values:
            avg_vel = float(np.mean(velocity_values))
            if avg_vel > 5.0 and max_velocity < avg_vel * 0.35:
                transition_signals.append(0.58)

        # 3) Large acceleration spike: start/end of manipulation.
        if max_accel > 25:
            transition_signals.append(0.45)

        # 4) Semantic region/tool change.
        if features.tool_changed:
            transition_signals.append(0.75)

        # 5) Contact point shift, when MediaPipe hand pose is confident enough.
        if features.contact_point_shift > 90:
            transition_signals.append(min(features.contact_point_shift / 180, 0.75))

        # 6) Optical-flow and scene-change signals.
        if features.flow_discontinuity > 1.4:
            transition_signals.append(min(self.flow_weight + 0.35, 0.8))

        if features.direction_change > 1.3:
            transition_signals.append(min(self.flow_weight + 0.20, 0.65))

        if features.scene_change_score > 0.35:
            transition_signals.append(min(self.scene_weight + features.scene_change_score, 0.8))

        if features.flow_uniformity < 0.35 and self.feature_history:
            if self.feature_history[-1].flow_uniformity > 0.65:
                transition_signals.append(0.48)

        if features.trajectory_curvature > 0.9:
            transition_signals.append(0.35)

        features.transition_score = max(transition_signals) if transition_signals else 0.0

    def get_feature_matrix(self) -> np.ndarray:
        """Convert feature history to a numpy matrix."""
        if not self.feature_history:
            return np.array([])

        return np.array([
            [
                f.hand_velocity_left, f.hand_velocity_right,
                f.hand_acceleration_left, f.hand_acceleration_right,
                float(f.hands_present), float(f.grip_state_left),
                float(f.grip_state_right), f.hand_distance,
                f.trajectory_curvature,
                float(f.num_tools), float(f.tool_changed),
                f.tool_stability,
                float(f.num_interactions), f.contact_point_shift,
                f.contact_point_variance,
                f.interaction_density, f.interaction_rhythm,
                f.flow_magnitude, f.flow_uniformity,
                f.flow_discontinuity, f.direction_change,
                f.scene_change_score, f.visual_stability,
                f.activity_level, f.transition_score,
            ]
            for f in self.feature_history
        ])

    def get_transition_scores(self) -> np.ndarray:
        return np.array([f.transition_score for f in self.feature_history])

    def get_feature_names(self) -> List[str]:
        return [
            "hand_velocity_left", "hand_velocity_right",
            "hand_acceleration_left", "hand_acceleration_right",
            "hands_present", "grip_state_left",
            "grip_state_right", "hand_distance",
            "trajectory_curvature",
            "num_tools", "tool_changed", "tool_stability",
            "num_interactions", "contact_point_shift",
            "contact_point_variance",
            "interaction_density", "interaction_rhythm",
            "flow_magnitude", "flow_uniformity",
            "flow_discontinuity", "direction_change",
            "scene_change_score", "visual_stability",
            "activity_level", "transition_score",
        ]