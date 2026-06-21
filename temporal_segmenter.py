import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from feature_extractor import FrameFeatures
from activity_recognizer import ActivityRecognizer


@dataclass
class ActionSegment:
    """Represents a detected action segment."""
    segment_id: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float
    dominant_activity: str
    avg_activity_level: float
    tools_used: List[str]
    interaction_types: List[str]
    confidence: float
    avg_motion_energy: float = 0.0
    visual_stability: float = 1.0
    activity_description: str = ""
    activity_reason: str = ""
    activity_confidence: float = 0.0

    # V3: separate real detections from heuristic ROIs.
    real_objects_used: List[str] = field(default_factory=list)
    heuristic_regions: List[str] = field(default_factory=list)


@dataclass
class Boundary:
    """Represents a detected boundary between segments."""
    frame_idx: int
    timestamp: float
    confidence: float
    reason: str
    signal_strengths: Optional[dict] = None


class TemporalSegmenter:
    """
    Performs temporal segmentation and produces human-readable activity labels.

    V3 update:
    - Keeps ROI/object separation.
    - Activity labels are generated after segment creation.
    - Boundary refinement is not applied after segment creation in main.py anymore.
    """

    ROI_CLASSES = {
        "motherboard_workspace",
        "cpu_socket_region",
        "active_motion_region",
    }

    def __init__(
        self,
        boundary_threshold: float = 0.32,
        min_segment_duration: float = 1.5,
        smoothing_sigma: float = 2.0,
        fps: float = 15.0,
    ):
        self.boundary_threshold = boundary_threshold
        self.min_segment_duration = min_segment_duration
        self.smoothing_sigma = smoothing_sigma
        self.fps = fps
        self.min_segment_frames = max(1, int(min_segment_duration * fps))
        self.activity_recognizer = ActivityRecognizer()

    def segment(
        self,
        features: List[FrameFeatures],
    ) -> Tuple[List[ActionSegment], List[Boundary]]:
        if not features:
            return [], []

        if len(features) < self.min_segment_frames:
            segment = self._create_single_segment(features)
            self._label_segments([segment], features)
            return [segment], []

        boundary_scores = self._build_boundary_score(features)
        smoothed_scores = gaussian_filter1d(boundary_scores, self.smoothing_sigma)

        boundaries = self._detect_boundaries(smoothed_scores, features)
        boundaries = self._filter_boundaries(boundaries)
        boundaries = self._merge_close_boundaries(boundaries)
        boundaries = self._remove_edge_boundaries(boundaries, features)

        segments = self._create_segments(boundaries, features)
        self._label_segments(segments, features)

        return segments, boundaries

    def _label_segments(
        self,
        segments: List[ActionSegment],
        features: List[FrameFeatures],
    ):
        total_duration = (
            max(features[-1].timestamp - features[0].timestamp, 1e-6)
            if features
            else 1.0
        )

        for idx, segment in enumerate(segments):
            decision = self.activity_recognizer.describe_segment(
                segment=segment,
                total_duration=total_duration,
                segment_index=idx,
                num_segments=len(segments),
            )

            segment.activity_description = decision.label
            segment.activity_reason = decision.reason
            segment.activity_confidence = decision.confidence

    def _build_boundary_score(self, features: List[FrameFeatures]) -> np.ndarray:
        transition = np.array([f.transition_score for f in features], dtype=float)
        activity = np.array([f.activity_level for f in features], dtype=float)
        flow = np.array([f.flow_magnitude for f in features], dtype=float)
        hands = np.array([f.hands_present for f in features], dtype=float)
        interactions = np.array([f.num_interactions for f in features], dtype=float)
        tool_counts = np.array([f.num_tools for f in features], dtype=float)

        activity_change = np.abs(np.diff(activity, prepend=activity[0]))

        flow_norm = self._normalize(flow)
        flow_change = np.abs(np.diff(flow_norm, prepend=flow_norm[0]))

        hand_change = np.minimum(np.abs(np.diff(hands, prepend=hands[0])), 1.0)

        interaction_change = np.minimum(
            np.abs(np.diff(interactions, prepend=interactions[0])),
            1.0,
        )

        tool_count_change = np.minimum(
            np.abs(np.diff(tool_counts, prepend=tool_counts[0])) / 3.0,
            1.0,
        )

        score = np.maximum.reduce([
            transition,
            0.80 * self._normalize(activity_change),
            0.55 * self._normalize(flow_change),
            0.55 * hand_change,
            0.55 * interaction_change,
            0.45 * tool_count_change,
        ])

        warmup = min(len(score), max(3, int(0.5 * self.fps)))
        score[:warmup] = 0.0

        return np.clip(score, 0.0, 1.0)

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)

        if values.size == 0:
            return values

        lo = np.percentile(values, 5)
        hi = np.percentile(values, 95)

        if hi - lo < 1e-6:
            return np.zeros_like(values)

        return np.clip((values - lo) / (hi - lo), 0.0, 1.0)

    def _detect_boundaries(
        self,
        scores: np.ndarray,
        features: List[FrameFeatures],
    ) -> List[Boundary]:
        peaks, properties = find_peaks(
            scores,
            height=self.boundary_threshold,
            distance=self.min_segment_frames,
            prominence=0.08,
        )

        boundaries = []

        for peak_idx, height in zip(peaks, properties.get("peak_heights", [])):
            reason, strengths = self._determine_boundary_reason(
                features,
                peak_idx,
                scores,
            )

            boundaries.append(Boundary(
                frame_idx=features[peak_idx].frame_idx,
                timestamp=features[peak_idx].timestamp,
                confidence=float(min(height, 1.0)),
                reason=reason,
                signal_strengths=strengths,
            ))

        return boundaries

    def _determine_boundary_reason(
        self,
        features: List[FrameFeatures],
        idx: int,
        scores: np.ndarray,
    ) -> Tuple[str, dict]:
        feature = features[idx]
        prev = features[idx - 1] if idx > 0 else feature

        signals = {}

        if feature.tool_changed:
            signals["object_or_region_change"] = 0.75

        if feature.visible_tools != prev.visible_tools:
            signals["visible_tool_set_change"] = 0.62

        if feature.contact_point_shift > 90:
            signals["contact_shift"] = min(feature.contact_point_shift / 180, 1.0)

        if feature.hands_present != prev.hands_present:
            signals["hand_presence_change"] = 0.55

        if abs(feature.activity_level - prev.activity_level) > 0.12:
            signals["activity_change"] = min(
                abs(feature.activity_level - prev.activity_level) * 3,
                1.0,
            )

        if feature.interaction_type != prev.interaction_type:
            signals["interaction_change"] = 0.55

        if feature.hand_velocity_left < 5 and feature.hand_velocity_right < 5:
            signals["motion_pause"] = 0.45

        if feature.flow_discontinuity > 1.4:
            signals["flow_discontinuity"] = min(feature.flow_discontinuity / 3, 1.0)

        if feature.scene_change_score > 0.35:
            signals["scene_change"] = feature.scene_change_score

        if feature.direction_change > 1.3:
            signals["direction_change"] = min(feature.direction_change / np.pi, 1.0)

        if not signals:
            signals["composite_signal"] = float(scores[idx])

        reason = "|".join(
            sorted(signals.keys(), key=lambda k: signals[k], reverse=True)
        )

        return reason, signals

    def _filter_boundaries(self, boundaries: List[Boundary]) -> List[Boundary]:
        if not boundaries:
            return []

        filtered = [boundaries[0]]

        for boundary in boundaries[1:]:
            prev = filtered[-1]
            frame_gap = boundary.frame_idx - prev.frame_idx

            if frame_gap >= self.min_segment_frames:
                filtered.append(boundary)
            elif boundary.confidence > prev.confidence:
                filtered[-1] = boundary

        return filtered

    def _merge_close_boundaries(self, boundaries: List[Boundary]) -> List[Boundary]:
        if len(boundaries) <= 1:
            return boundaries

        merged = [boundaries[0]]
        merge_window = max(1, self.min_segment_frames // 2)

        for boundary in boundaries[1:]:
            if boundary.frame_idx - merged[-1].frame_idx < merge_window:
                if boundary.confidence > merged[-1].confidence:
                    merged[-1] = boundary
            else:
                merged.append(boundary)

        return merged

    def _remove_edge_boundaries(
        self,
        boundaries: List[Boundary],
        features: List[FrameFeatures],
    ) -> List[Boundary]:
        if not boundaries or not features:
            return boundaries

        cleaned = []
        video_start = features[0].timestamp
        video_end = features[-1].timestamp

        for boundary in boundaries:
            if boundary.timestamp - video_start < self.min_segment_duration:
                continue

            if video_end - boundary.timestamp < self.min_segment_duration:
                continue

            cleaned.append(boundary)

        return cleaned

    def _create_segments(
        self,
        boundaries: List[Boundary],
        features: List[FrameFeatures],
    ) -> List[ActionSegment]:
        segments = []

        start_indices = [0] + [
            self._frame_to_feature_idx(b.frame_idx, features)
            for b in boundaries
        ]

        end_indices = [
            self._frame_to_feature_idx(b.frame_idx, features)
            for b in boundaries
        ] + [len(features) - 1]

        for seg_id, (start_idx, end_idx) in enumerate(zip(start_indices, end_indices)):
            if start_idx >= end_idx:
                continue

            segment_features = features[start_idx:end_idx]

            segments.append(
                self._build_segment(seg_id, segment_features)
            )

        return segments

    def _build_segment(
        self,
        seg_id: int,
        segment_features: List[FrameFeatures],
    ) -> ActionSegment:
        start_f = segment_features[0]
        end_f = segment_features[-1]

        tool_set = set()

        for f in segment_features:
            tool_set.update(f.visible_tools)

            if f.dominant_tool:
                tool_set.add(f.dominant_tool)

        tools = sorted(tool_set)

        real_objects = sorted([
            t for t in tools
            if t not in self.ROI_CLASSES
        ])

        rois = sorted([
            t for t in tools
            if t in self.ROI_CLASSES
        ])

        interaction_types = sorted(set(
            f.interaction_type
            for f in segment_features
            if f.interaction_type != "none"
        ))

        avg_activity = float(np.mean([
            f.activity_level for f in segment_features
        ]))

        avg_motion = float(np.mean([
            f.flow_magnitude for f in segment_features
        ]))

        avg_stability = float(np.mean([
            f.visual_stability for f in segment_features
        ]))

        dominant = self._determine_dominant_activity(segment_features)
        confidence = self._compute_segment_confidence(segment_features)

        return ActionSegment(
            segment_id=seg_id,
            start_frame=start_f.frame_idx,
            end_frame=end_f.frame_idx,
            start_time=start_f.timestamp,
            end_time=end_f.timestamp,
            duration=end_f.timestamp - start_f.timestamp,
            dominant_activity=dominant,
            avg_activity_level=avg_activity,
            tools_used=tools,
            real_objects_used=real_objects,
            heuristic_regions=rois,
            interaction_types=interaction_types,
            confidence=confidence,
            avg_motion_energy=avg_motion,
            visual_stability=avg_stability,
        )

    def _determine_dominant_activity(self, features: List[FrameFeatures]) -> str:
        avg_velocity = float(np.mean([
            max(f.hand_velocity_left, f.hand_velocity_right)
            for f in features
        ]))

        avg_flow = float(np.mean([
            f.flow_magnitude for f in features
        ]))

        avg_activity = float(np.mean([
            f.activity_level for f in features
        ]))

        avg_hands = float(np.mean([
            f.hands_present for f in features
        ]))

        avg_interactions = float(np.mean([
            f.num_interactions for f in features
        ]))

        if avg_hands < 0.4 and avg_flow < 1.2:
            return "idle_no_hands"

        if avg_interactions > 0.3:
            if avg_velocity > 12 or avg_flow > 3.0 or avg_activity > 0.38:
                return "active_assembly"

            return "positioning_inspection"

        if avg_velocity > 10 or avg_flow > 3.0:
            return "transition"

        return "inspection_or_pause"

    def _compute_segment_confidence(
        self,
        features: List[FrameFeatures],
    ) -> float:
        if len(features) < 3:
            return 0.5

        activities = np.array([f.activity_level for f in features], dtype=float)

        variance = float(np.var(activities))
        consistency = 1.0 / (1.0 + variance * 10)
        duration_score = min(len(features) / max(self.min_segment_frames, 1), 1.0)

        internal_transitions = [
            f.transition_score
            for f in features[1:-1]
        ]

        low_internal = (
            1.0 - float(np.mean(internal_transitions))
            if internal_transitions
            else 1.0
        )

        low_internal = max(0.0, min(low_internal, 1.0))

        tool_specific = any(
            any(
                t not in self.ROI_CLASSES
                for t in f.visible_tools
            )
            for f in features
        )

        tool_score = 0.10 if tool_specific else 0.0

        return min(
            1.0,
            0.35 * consistency +
            0.25 * duration_score +
            0.30 * low_internal +
            tool_score,
        )

    def _frame_to_feature_idx(
        self,
        frame_idx: int,
        features: List[FrameFeatures],
    ) -> int:
        for i, f in enumerate(features):
            if f.frame_idx >= frame_idx:
                return i

        return len(features) - 1

    def _create_single_segment(self, features: List[FrameFeatures]) -> ActionSegment:
        return self._build_segment(0, features)

    def adaptive_segment(
        self,
        features: List[FrameFeatures],
        target_segments: Optional[int] = None,
    ) -> Tuple[List[ActionSegment], List[Boundary]]:
        if target_segments is None:
            return self.segment(features)

        low, high = 0.1, 0.9
        best_segments, best_boundaries = None, None

        for _ in range(10):
            mid = (low + high) / 2
            self.boundary_threshold = mid

            segments, boundaries = self.segment(features)

            if best_segments is None:
                best_segments, best_boundaries = segments, boundaries

            if len(segments) == target_segments:
                return segments, boundaries

            if len(segments) > target_segments:
                low = mid
            else:
                high = mid

            if abs(len(segments) - target_segments) < abs(len(best_segments) - target_segments):
                best_segments, best_boundaries = segments, boundaries

        return best_segments, best_boundaries