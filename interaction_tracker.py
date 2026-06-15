import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from hand_tracker import HandData
from object_detector import DetectedObject


@dataclass
class Interaction:
    """Represents a hand-object interaction event."""
    hand: HandData
    obj: DetectedObject
    interaction_type: str
    distance: float
    overlap_ratio: float
    frame_idx: int
    timestamp: float
    contact_point: Optional[np.ndarray] = None


@dataclass
class InteractionState:
    """Tracks ongoing interaction state."""
    active_interactions: List[Interaction] = field(default_factory=list)
    interaction_start_frame: Optional[int] = None
    current_tool: Optional[str] = None
    previous_tool: Optional[str] = None
    contact_point: Optional[np.ndarray] = None
    duration_frames: int = 0
    tool_switch_count: int = 0


class InteractionTracker:
    """Tracks physical interactions between hands and objects/hardware."""

    def __init__(self, distance_threshold: int = 50,
                 iou_threshold: float = 0.3):
        self.distance_threshold = distance_threshold
        self.iou_threshold = iou_threshold
        self.state = InteractionState()
        self.interaction_history: List[Interaction] = []
        self.contact_points_history: List[np.ndarray] = []
        self.interaction_change_scores: List[float] = []
        self.interaction_type_history: List[str] = []
        self.tool_sequence: List[str] = []

    def process_frame(self, hands: List[HandData],
                      objects: List[DetectedObject],
                      frame_idx: int, timestamp: float) -> List[Interaction]:
        """Detect interactions between hands and objects in current frame."""
        current_interactions = []

        for hand in hands:
            for obj in objects:
                interaction = self._check_interaction(
                    hand, obj, frame_idx, timestamp
                )
                if interaction:
                    current_interactions.append(interaction)

        change_score = self._compute_interaction_change(current_interactions)
        self.interaction_change_scores.append(change_score)

        dominant_type = self._get_dominant_type(current_interactions)
        self.interaction_type_history.append(dominant_type)
        if len(self.interaction_type_history) > 120:
            self.interaction_type_history.pop(0)

        self._update_state(current_interactions, frame_idx)
        self.interaction_history.extend(current_interactions)

        return current_interactions

    def _check_interaction(self, hand: HandData, obj: DetectedObject,
                           frame_idx: int, timestamp: float) -> Optional[Interaction]:
        """Check if a hand is interacting with an object."""
        distance = self._hand_object_distance(hand, obj)
        overlap = self._hand_object_overlap(hand, obj)

        if distance > self.distance_threshold and overlap < self.iou_threshold:
            return None

        interaction_type = self._classify_interaction(hand, obj, distance, overlap)

        contact_point = self._estimate_contact_point(hand, obj)
        if contact_point is not None:
            self.contact_points_history.append(contact_point)
            if len(self.contact_points_history) > 120:
                self.contact_points_history.pop(0)

        return Interaction(
            hand=hand,
            obj=obj,
            interaction_type=interaction_type,
            distance=distance,
            overlap_ratio=overlap,
            frame_idx=frame_idx,
            timestamp=timestamp,
            contact_point=contact_point
        )

    def _hand_object_distance(self, hand: HandData, obj: DetectedObject) -> float:
        """Compute minimum distance between fingertips and object center."""
        distances = np.linalg.norm(
            hand.fingertip_positions - obj.center, axis=1
        )
        return float(distances.min())

    def _hand_object_overlap(self, hand: HandData, obj: DetectedObject) -> float:
        """Compute overlap ratio between hand bbox and object bbox."""
        hx, hy, hw, hh = hand.bounding_box
        hand_box = (hx, hy, hx + hw, hy + hh)
        obj_box = obj.bbox

        x1 = max(hand_box[0], obj_box[0])
        y1 = max(hand_box[1], obj_box[1])
        x2 = min(hand_box[2], obj_box[2])
        y2 = min(hand_box[3], obj_box[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        hand_area = hw * hh

        return intersection / hand_area if hand_area > 0 else 0.0

    def _classify_interaction(self, hand: HandData, obj: DetectedObject,
                               distance: float, overlap: float) -> str:
        """Classify the type of interaction."""
        if hand.is_gripping and overlap > 0.4:
            return "grasp"
        elif hand.is_gripping and distance < self.distance_threshold * 0.5:
            return "use"
        elif overlap > 0.2:
            return "touch"
        else:
            return "approach"

    def _estimate_contact_point(self, hand: HandData,
                                 obj: DetectedObject) -> Optional[np.ndarray]:
        """Estimate the point of physical contact on the hardware."""
        if not hand.is_gripping:
            return None

        distances = np.linalg.norm(
            hand.fingertip_positions - obj.center, axis=1
        )
        closest_finger_idx = np.argmin(distances)
        contact = hand.fingertip_positions[closest_finger_idx].copy()

        return contact

    def _get_dominant_type(self, interactions: List[Interaction]) -> str:
        """Get the most significant interaction type from a list."""
        if not interactions:
            return "none"
        type_priority = {"use": 3, "grasp": 2, "touch": 1, "approach": 0}
        best = max(interactions, key=lambda i: type_priority.get(i.interaction_type, 0))
        return best.interaction_type

    def _compute_interaction_change(self,
                                     current: List[Interaction]) -> float:
        """Compute how much interactions changed from previous state."""
        prev_types = set(
            (i.hand.handedness, i.obj.class_name, i.interaction_type)
            for i in self.state.active_interactions
        )
        curr_types = set(
            (i.hand.handedness, i.obj.class_name, i.interaction_type)
            for i in current
        )

        if not prev_types and not curr_types:
            return 0.0

        union = prev_types.union(curr_types)
        diff = prev_types.symmetric_difference(curr_types)

        return len(diff) / len(union) if union else 0.0

    def _update_state(self, interactions: List[Interaction], frame_idx: int):
        """Update the interaction tracking state."""
        self.state.active_interactions = interactions

        if interactions:
            self.state.duration_frames += 1
            if self.state.interaction_start_frame is None:
                self.state.interaction_start_frame = frame_idx

            tools = [i.obj.class_name for i in interactions
                     if i.interaction_type in ("grasp", "use")]
            if tools:
                new_tool = tools[0]
                if (self.state.current_tool is not None and
                        new_tool != self.state.current_tool):
                    self.state.tool_switch_count += 1
                    self.state.previous_tool = self.state.current_tool

                self.state.current_tool = new_tool
                if not self.tool_sequence or self.tool_sequence[-1] != new_tool:
                    self.tool_sequence.append(new_tool)
        else:
            self.state.interaction_start_frame = None
            self.state.current_tool = None
            self.state.duration_frames = 0

    def get_contact_point_shift(self, window: int = 30) -> float:
        """Measure how much the contact point has shifted recently."""
        if len(self.contact_points_history) < 2:
            return 0.0

        recent = self.contact_points_history[-window:]
        if len(recent) < 2:
            return 0.0

        shifts = np.diff(recent, axis=0)
        total_shift = float(np.linalg.norm(shifts, axis=1).sum())

        return total_shift

    def get_contact_point_variance(self, window: int = 30) -> float:
        """Variance of contact point positions (high = scattered work)."""
        if len(self.contact_points_history) < 3:
            return 0.0

        recent = np.array(self.contact_points_history[-window:])
        return float(np.var(recent, axis=0).sum())

    def get_interaction_density(self, window: int = 30) -> float:
        """Fraction of recent frames that had active interactions."""
        recent_scores = self.interaction_change_scores[-window:]
        if not recent_scores:
            return 0.0
        return sum(1 for s in recent_scores if s > 0) / len(recent_scores)

    def get_interaction_rhythm(self, window: int = 60) -> float:
        """Detect rhythmic patterns in interactions (repetitive actions)."""
        if len(self.interaction_change_scores) < window:
            return 0.0

        recent = np.array(self.interaction_change_scores[-window:])
        if recent.std() < 0.01:
            return 0.0

        autocorr = np.correlate(recent - recent.mean(), recent - recent.mean(), mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        autocorr /= autocorr[0] if autocorr[0] != 0 else 1

        peaks = []
        for i in range(1, len(autocorr) - 1):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                if autocorr[i] > 0.3:
                    peaks.append(autocorr[i])

        return float(np.mean(peaks)) if peaks else 0.0
