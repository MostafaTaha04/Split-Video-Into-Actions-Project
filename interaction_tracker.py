import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
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
    """
    Tracks interactions between hands and objects/hardware.

    V3 update:
    - Thresholds are relaxed for real MediaPipe hand boxes.
    - Interaction no longer depends too strongly on is_gripping.
    - Touch/contact can be detected from fingertip distance even without a grip.
    """

    ROI_CLASSES = {
        "motherboard_workspace",
        "cpu_socket_region",
        "active_motion_region",
    }

    def __init__(
        self,
        distance_threshold: int = 90,
        iou_threshold: float = 0.05,
    ):
        self.distance_threshold = distance_threshold
        self.iou_threshold = iou_threshold
        self.state = InteractionState()

        self.interaction_history: List[Interaction] = []
        self.contact_points_history: List[np.ndarray] = []
        self.interaction_change_scores: List[float] = []
        self.interaction_type_history: List[str] = []
        self.tool_sequence: List[str] = []

    def process_frame(
        self,
        hands: List[HandData],
        objects: List[DetectedObject],
        frame_idx: int,
        timestamp: float,
    ) -> List[Interaction]:
        """Detect interactions between hands and objects in current frame."""
        current_interactions = []

        for hand in hands:
            for obj in objects:
                interaction = self._check_interaction(
                    hand,
                    obj,
                    frame_idx,
                    timestamp,
                )

                if interaction:
                    current_interactions.append(interaction)

        # Keep only the most useful interactions.
        current_interactions = self._prioritize_interactions(current_interactions)

        change_score = self._compute_interaction_change(current_interactions)
        self.interaction_change_scores.append(change_score)

        if len(self.interaction_change_scores) > 120:
            self.interaction_change_scores.pop(0)

        dominant_type = self._get_dominant_type(current_interactions)
        self.interaction_type_history.append(dominant_type)

        if len(self.interaction_type_history) > 120:
            self.interaction_type_history.pop(0)

        self._update_state(current_interactions, frame_idx)
        self.interaction_history.extend(current_interactions)

        if len(self.interaction_history) > 1000:
            self.interaction_history = self.interaction_history[-1000:]

        return current_interactions

    def _check_interaction(
        self,
        hand: HandData,
        obj: DetectedObject,
        frame_idx: int,
        timestamp: float,
    ) -> Optional[Interaction]:
        """Check if a hand is interacting with an object."""
        distance = self._hand_object_distance(hand, obj)
        overlap = self._hand_object_overlap(hand, obj)

        class_name = obj.class_name.lower()

        # Large workspace ROI should not dominate interaction decisions.
        if class_name == "motherboard_workspace":
            return None

        distance_limit = self.distance_threshold

        # For cpu_socket_region, allow slightly larger distance because it is a functional ROI.
        if class_name == "cpu_socket_region":
            distance_limit = int(self.distance_threshold * 1.25)

        if distance > distance_limit and overlap < self.iou_threshold:
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
            contact_point=contact_point,
        )

    def _prioritize_interactions(
        self,
        interactions: List[Interaction],
        max_per_frame: int = 6,
    ) -> List[Interaction]:
        """Prefer real objects and socket ROI over generic motion ROI."""
        if not interactions:
            return []

        def score(interaction: Interaction) -> float:
            cls = interaction.obj.class_name.lower()

            if cls not in self.ROI_CLASSES:
                base = 3.0
            elif cls == "cpu_socket_region":
                base = 2.0
            elif cls == "active_motion_region":
                base = 1.0
            else:
                base = 0.0

            type_bonus = {
                "use": 0.9,
                "grasp": 0.7,
                "touch": 0.5,
                "approach": 0.2,
            }.get(interaction.interaction_type, 0.0)

            distance_bonus = 1.0 / (1.0 + interaction.distance / max(self.distance_threshold, 1))

            return base + type_bonus + distance_bonus + interaction.overlap_ratio

        return sorted(interactions, key=score, reverse=True)[:max_per_frame]

    def _hand_object_distance(self, hand: HandData, obj: DetectedObject) -> float:
        """Compute minimum distance between fingertips/palm and object center."""
        points = np.vstack([
            hand.fingertip_positions,
            hand.palm_center.reshape(1, 2),
        ])

        distances = np.linalg.norm(points - obj.center, axis=1)

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
        hand_area = max(0, hw * hh)

        return intersection / hand_area if hand_area > 0 else 0.0

    def _classify_interaction(
        self,
        hand: HandData,
        obj: DetectedObject,
        distance: float,
        overlap: float,
    ) -> str:
        """Classify the type of interaction."""
        cls = obj.class_name.lower()
        is_real_object = cls not in self.ROI_CLASSES

        if hand.is_gripping and is_real_object and distance < self.distance_threshold:
            return "use"

        if hand.is_gripping and overlap > self.iou_threshold:
            return "grasp"

        if distance < self.distance_threshold * 0.45:
            return "touch"

        if overlap > self.iou_threshold:
            return "touch"

        return "approach"

    def _estimate_contact_point(
        self,
        hand: HandData,
        obj: DetectedObject,
    ) -> Optional[np.ndarray]:
        """
        Estimate contact point.

        V3: contact point does not require is_gripping anymore,
        because touching/pressing a CPU socket may not look like a grip.
        """
        points = np.vstack([
            hand.fingertip_positions,
            hand.palm_center.reshape(1, 2),
        ])

        distances = np.linalg.norm(points - obj.center, axis=1)
        closest_idx = int(np.argmin(distances))

        if distances[closest_idx] > self.distance_threshold * 1.3:
            return None

        return points[closest_idx].copy()

    def _get_dominant_type(self, interactions: List[Interaction]) -> str:
        if not interactions:
            return "none"

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

        return best.interaction_type

    def _compute_interaction_change(self, current: List[Interaction]) -> float:
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
        self.state.active_interactions = interactions

        if interactions:
            self.state.duration_frames += 1

            if self.state.interaction_start_frame is None:
                self.state.interaction_start_frame = frame_idx

            tools = [
                i.obj.class_name
                for i in interactions
                if i.interaction_type in {"use", "grasp", "touch"}
            ]

            if tools:
                current_tool = tools[0]

                if (
                    self.state.current_tool is not None
                    and current_tool != self.state.current_tool
                ):
                    self.state.tool_switch_count += 1

                self.state.previous_tool = self.state.current_tool
                self.state.current_tool = current_tool
                self.tool_sequence.append(current_tool)

                if len(self.tool_sequence) > 120:
                    self.tool_sequence.pop(0)

            contacts = [
                i.contact_point
                for i in interactions
                if i.contact_point is not None
            ]

            if contacts:
                self.state.contact_point = np.mean(contacts, axis=0)
        else:
            self.state.interaction_start_frame = None
            self.state.duration_frames = 0
            self.state.current_tool = None
            self.state.contact_point = None

    def get_contact_point_shift(self) -> float:
        """Return shift between last two contact points."""
        if len(self.contact_points_history) < 2:
            return 0.0

        return float(
            np.linalg.norm(
                self.contact_points_history[-1] - self.contact_points_history[-2]
            )
        )

    def get_contact_point_variance(self, window: int = 20) -> float:
        """Return variance of recent contact points."""
        if len(self.contact_points_history) < 3:
            return 0.0

        points = np.array(self.contact_points_history[-window:])

        if len(points) < 3:
            return 0.0

        return float(np.mean(np.var(points, axis=0)))

    def get_interaction_density(self, window: int = 30) -> float:
        """Fraction of recent frames with interaction."""
        if not self.interaction_type_history:
            return 0.0

        recent = self.interaction_type_history[-window:]

        return sum(t != "none" for t in recent) / len(recent)

    def get_interaction_rhythm(self, window: int = 60) -> float:
        """
        Estimate rhythm/regularity of interaction events.
        Higher value means more regular repeated interaction.
        """
        recent = self.interaction_history[-window:]

        if len(recent) < 3:
            return 0.0

        frames = [i.frame_idx for i in recent]
        diffs = np.diff(frames)

        if len(diffs) == 0 or np.mean(diffs) < 1e-6:
            return 0.0

        return float(1.0 / (1.0 + np.std(diffs) / (np.mean(diffs) + 1e-6)))

    def get_tool_switch_count(self) -> int:
        return self.state.tool_switch_count

    def get_current_tool(self) -> Optional[str]:
        return self.state.current_tool

    def get_interaction_change_score(self) -> float:
        if not self.interaction_change_scores:
            return 0.0

        return float(self.interaction_change_scores[-1])