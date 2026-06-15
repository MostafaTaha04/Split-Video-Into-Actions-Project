import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from collections import deque
from hand_tracker import HandData
from object_detector import DetectedObject
from interaction_tracker import Interaction


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

    # Object features
    num_objects: int = 0
    num_tools: int = 0
    tool_changed: bool = False
    dominant_tool: str = ""

    # Interaction features
    num_interactions: int = 0
    interaction_type: str = "none"
    contact_point_shift: float = 0.0
    interaction_density: float = 0.0

    # Composite signals
    activity_level: float = 0.0
    transition_score: float = 0.0


class FeatureExtractor:
    """Extracts and combines features from all tracking modules."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.feature_history: List[FrameFeatures] = []
        self.velocity_buffer_left = deque(maxlen=window_size)
        self.velocity_buffer_right = deque(maxlen=window_size)
        self.prev_velocity_left = 0.0
        self.prev_velocity_right = 0.0
        self.prev_tool = ""

    def extract(self, frame_idx: int, timestamp: float,
                hands: List[HandData],
                objects: List[DetectedObject],
                interactions: List[Interaction],
                contact_shift: float,
                interaction_density: float) -> FrameFeatures:
        """Extract all features for the current frame."""
        features = FrameFeatures(frame_idx=frame_idx, timestamp=timestamp)

        self._extract_hand_features(features, hands)
        self._extract_object_features(features, objects)
        self._extract_interaction_features(
            features, interactions, contact_shift, interaction_density
        )
        self._compute_composite_signals(features)

        self.feature_history.append(features)
        return features

    def _extract_hand_features(self, features: FrameFeatures,
                                hands: List[HandData]):
        """Extract hand-related features."""
        features.hands_present = len(hands)

        for hand in hands:
            if hand.handedness == "Left":
                features.hand_velocity_left = hand.velocity
                features.grip_state_left = hand.is_gripping
                features.hand_acceleration_left = (
                    hand.velocity - self.prev_velocity_left
                )
                self.prev_velocity_left = hand.velocity
                self.velocity_buffer_left.append(hand.velocity)
            else:
                features.hand_velocity_right = hand.velocity
                features.grip_state_right = hand.is_gripping
                features.hand_acceleration_right = (
                    hand.velocity - self.prev_velocity_right
                )
                self.prev_velocity_right = hand.velocity
                self.velocity_buffer_right.append(hand.velocity)

        if len(hands) == 2:
            features.hand_distance = float(np.linalg.norm(
                hands[0].palm_center - hands[1].palm_center
            ))

    def _extract_object_features(self, features: FrameFeatures,
                                  objects: List[DetectedObject]):
        """Extract object-related features."""
        features.num_objects = len(objects)
        tools = [o for o in objects if o.class_name != "person"]
        features.num_tools = len(tools)

        if tools:
            dominant = max(tools, key=lambda t: t.area)
            features.dominant_tool = dominant.class_name
            features.tool_changed = (
                dominant.class_name != self.prev_tool and self.prev_tool != ""
            )
            self.prev_tool = dominant.class_name

    def _extract_interaction_features(self, features: FrameFeatures,
                                       interactions: List[Interaction],
                                       contact_shift: float,
                                       interaction_density: float):
        """Extract interaction-related features."""
        features.num_interactions = len(interactions)
        features.contact_point_shift = contact_shift
        features.interaction_density = interaction_density

        if interactions:
            type_priority = {"use": 3, "grasp": 2, "touch": 1, "approach": 0}
            best = max(interactions, key=lambda i: type_priority.get(i.interaction_type, 0))
            features.interaction_type = best.interaction_type

    def _compute_composite_signals(self, features: FrameFeatures):
        """Compute high-level composite signals for segmentation."""
        max_velocity = max(features.hand_velocity_left,
                          features.hand_velocity_right)
        features.activity_level = (
            0.4 * min(max_velocity / 50.0, 1.0) +
            0.3 * features.interaction_density +
            0.2 * (features.num_interactions / max(features.num_objects, 1)) +
            0.1 * float(features.grip_state_left or features.grip_state_right)
        )

        transition_signals = []

        if self.velocity_buffer_left:
            avg_vel = np.mean(list(self.velocity_buffer_left))
            if avg_vel > 5 and features.hand_velocity_left < avg_vel * 0.3:
                transition_signals.append(0.6)

        if features.tool_changed:
            transition_signals.append(0.8)

        if len(self.feature_history) > 0:
            prev = self.feature_history[-1]
            if (features.grip_state_left != prev.grip_state_left or
                    features.grip_state_right != prev.grip_state_right):
                transition_signals.append(0.4)

            if features.interaction_type != prev.interaction_type:
                transition_signals.append(0.5)

        if features.contact_point_shift > 100:
            transition_signals.append(0.7)

        features.transition_score = max(transition_signals) if transition_signals else 0.0

    def get_feature_matrix(self) -> np.ndarray:
        """Convert feature history to numpy matrix for analysis."""
        if not self.feature_history:
            return np.array([])

        matrix = np.array([
            [
                f.hand_velocity_left, f.hand_velocity_right,
                f.hand_acceleration_left, f.hand_acceleration_right,
                float(f.hands_present), float(f.grip_state_left),
                float(f.grip_state_right), f.hand_distance,
                float(f.num_tools), float(f.tool_changed),
                float(f.num_interactions), f.contact_point_shift,
                f.interaction_density, f.activity_level,
                f.transition_score
            ]
            for f in self.feature_history
        ])

        return matrix

    def get_transition_scores(self) -> np.ndarray:
        """Get the transition score time series."""
        return np.array([f.transition_score for f in self.feature_history])
