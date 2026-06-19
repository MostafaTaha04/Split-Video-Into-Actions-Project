import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass
from typing import List, Tuple, Optional

from feature_extractor import FrameFeatures


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
    Performs temporal segmentation to find action boundaries.

    Updated for CPU/motherboard assembly videos:
    - Does not rely only on tool changes.
    - Combines transition_score, activity-level changes, flow changes, and hand presence changes.
    """

    def __init__(self, boundary_threshold: float = 0.32,
                 min_segment_duration: float = 1.5,
                 smoothing_sigma: float = 2.0,
                 fps: float = 15.0):
        self.boundary_threshold = boundary_threshold
        self.min_segment_duration = min_segment_duration
        self.smoothing_sigma = smoothing_sigma
        self.fps = fps
        self.min_segment_frames = max(1, int(min_segment_duration * fps))

    def segment(self, features: List[FrameFeatures]) -> Tuple[List[ActionSegment], List[Boundary]]:
        """Perform full temporal segmentation on extracted features."""
        if not features:
            return [], []

        if len(features) < self.min_segment_frames:
            segment = self._create_single_segment(features)
            return [segment], []

        boundary_scores = self._build_boundary_score(features)
        smoothed_scores = gaussian_filter1d(boundary_scores, self.smoothing_sigma)

        boundaries = self._detect_boundaries(smoothed_scores, features)
        boundaries = self._filter_boundaries(boundaries)
        boundaries = self._merge_close_boundaries(boundaries)
        segments = self._create_segments(boundaries, features)

        return segments, boundaries

    def _build_boundary_score(self, features: List[FrameFeatures]) -> np.ndarray:
        """Build a robust boundary score from several signals."""
        transition = np.array([f.transition_score for f in features], dtype=float)
        activity = np.array([f.activity_level for f in features], dtype=float)
        flow = np.array([f.flow_magnitude for f in features], dtype=float)
        hands = np.array([f.hands_present for f in features], dtype=float)
        interactions = np.array([f.num_interactions for f in features], dtype=float)

        activity_change = np.abs(np.diff(activity, prepend=activity[0]))
        flow_norm = self._normalize(flow)
        flow_change = np.abs(np.diff(flow_norm, prepend=flow_norm[0]))
        hand_change = np.minimum(np.abs(np.diff(hands, prepend=hands[0])), 1.0)
        interaction_change = np.minimum(np.abs(np.diff(interactions, prepend=interactions[0])), 1.0)

        score = np.maximum.reduce([
            transition,
            0.85 * self._normalize(activity_change),
            0.60 * self._normalize(flow_change),
            0.65 * hand_change,
            0.50 * interaction_change,
        ])

        # Ignore the first few frames: initialization often produces false boundaries.
        warmup = min(len(score), max(3, int(0.5 * self.fps)))
        score[:warmup] = 0.0
        return np.clip(score, 0.0, 1.0)

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        """Robust 0-1 normalization."""
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return values
        lo = np.percentile(values, 5)
        hi = np.percentile(values, 95)
        if hi - lo < 1e-6:
            return np.zeros_like(values)
        return np.clip((values - lo) / (hi - lo), 0.0, 1.0)

    def _detect_boundaries(self, scores: np.ndarray,
                           features: List[FrameFeatures]) -> List[Boundary]:
        """Detect boundary candidates using peak detection."""
        peaks, properties = find_peaks(
            scores,
            height=self.boundary_threshold,
            distance=self.min_segment_frames,
            prominence=0.08,
        )

        boundaries = []
        for peak_idx, height in zip(peaks, properties.get("peak_heights", [])):
            reason, strengths = self._determine_boundary_reason(features, peak_idx, scores)
            boundaries.append(Boundary(
                frame_idx=features[peak_idx].frame_idx,
                timestamp=features[peak_idx].timestamp,
                confidence=float(min(height, 1.0)),
                reason=reason,
                signal_strengths=strengths,
            ))
        return boundaries

    def _determine_boundary_reason(self, features: List[FrameFeatures],
                                   idx: int,
                                   scores: np.ndarray) -> Tuple[str, dict]:
        """Determine the primary reason for a boundary."""
        feature = features[idx]
        prev = features[idx - 1] if idx > 0 else feature
        signals = {}

        if feature.tool_changed:
            signals["region_change"] = 0.75
        if feature.contact_point_shift > 90:
            signals["contact_shift"] = min(feature.contact_point_shift / 180, 1.0)
        if feature.hands_present != prev.hands_present:
            signals["hand_presence_change"] = 0.65
        if abs(feature.activity_level - prev.activity_level) > 0.12:
            signals["activity_change"] = min(abs(feature.activity_level - prev.activity_level) * 3, 1.0)
        if feature.interaction_type != prev.interaction_type:
            signals["interaction_change"] = 0.55
        if feature.hand_velocity_left < 5 and feature.hand_velocity_right < 5:
            signals["motion_pause"] = 0.50
        if feature.flow_discontinuity > 1.4:
            signals["flow_discontinuity"] = min(feature.flow_discontinuity / 3, 1.0)
        if feature.scene_change_score > 0.35:
            signals["scene_change"] = feature.scene_change_score
        if feature.direction_change > 1.3:
            signals["direction_change"] = min(feature.direction_change / np.pi, 1.0)

        if not signals:
            signals["composite_signal"] = float(scores[idx])

        reason = "|".join(sorted(signals.keys(), key=lambda k: signals[k], reverse=True))
        return reason, signals

    def _filter_boundaries(self, boundaries: List[Boundary]) -> List[Boundary]:
        """Filter out boundaries that are too close together."""
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
        """Merge boundaries that are very close and keep the stronger one."""
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

    def _create_segments(self, boundaries: List[Boundary],
                         features: List[FrameFeatures]) -> List[ActionSegment]:
        """Create action segments from detected boundaries."""
        segments = []
        start_indices = [0] + [self._frame_to_feature_idx(b.frame_idx, features) for b in boundaries]
        end_indices = [self._frame_to_feature_idx(b.frame_idx, features) for b in boundaries] + [len(features) - 1]

        for seg_id, (start_idx, end_idx) in enumerate(zip(start_indices, end_indices)):
            if start_idx >= end_idx:
                continue
            segment_features = features[start_idx:end_idx]
            segments.append(self._build_segment(seg_id, segment_features))
        return segments

    def _build_segment(self, seg_id: int,
                       segment_features: List[FrameFeatures]) -> ActionSegment:
        """Build a segment from its constituent features."""
        start_f = segment_features[0]
        end_f = segment_features[-1]

        tools = sorted(set(f.dominant_tool for f in segment_features if f.dominant_tool))
        interaction_types = sorted(set(
            f.interaction_type for f in segment_features if f.interaction_type != "none"
        ))

        avg_activity = float(np.mean([f.activity_level for f in segment_features]))
        avg_motion = float(np.mean([f.flow_magnitude for f in segment_features]))
        avg_stability = float(np.mean([f.visual_stability for f in segment_features]))
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
            interaction_types=interaction_types,
            confidence=confidence,
            avg_motion_energy=avg_motion,
            visual_stability=avg_stability,
        )

    def _determine_dominant_activity(self, features: List[FrameFeatures]) -> str:
        """Determine a practical activity label for a segment."""
        avg_velocity = float(np.mean([
            max(f.hand_velocity_left, f.hand_velocity_right) for f in features
        ]))
        avg_flow = float(np.mean([f.flow_magnitude for f in features]))
        avg_activity = float(np.mean([f.activity_level for f in features]))
        avg_hands = float(np.mean([f.hands_present for f in features]))
        avg_interactions = float(np.mean([f.num_interactions for f in features]))

        interaction_counts = {}
        for f in features:
            itype = f.interaction_type
            interaction_counts[itype] = interaction_counts.get(itype, 0) + 1
        dominant_interaction = max(interaction_counts, key=interaction_counts.get) if interaction_counts else "none"

        if avg_hands < 0.4 and avg_flow < 1.2:
            return "idle/no_hands"
        if avg_interactions > 0.3:
            if avg_velocity > 12 or avg_flow > 3.0 or avg_activity > 0.38:
                return "active_assembly"
            return "positioning_inspection"
        if dominant_interaction != "none":
            return dominant_interaction
        if avg_velocity > 10 or avg_flow > 3.0:
            return "transition"
        return "idle"

    def _compute_segment_confidence(self, features: List[FrameFeatures]) -> float:
        """Compute confidence that this is a coherent segment."""
        if len(features) < 3:
            return 0.5

        activities = np.array([f.activity_level for f in features], dtype=float)
        variance = float(np.var(activities))
        consistency = 1.0 / (1.0 + variance * 10)
        duration_score = min(len(features) / max(self.min_segment_frames, 1), 1.0)

        internal_transitions = [f.transition_score for f in features[1:-1]]
        low_internal = 1.0 - float(np.mean(internal_transitions)) if internal_transitions else 1.0
        low_internal = max(0.0, min(low_internal, 1.0))

        return 0.4 * consistency + 0.3 * duration_score + 0.3 * low_internal

    def _frame_to_feature_idx(self, frame_idx: int,
                              features: List[FrameFeatures]) -> int:
        """Find the feature index closest to a frame index."""
        for i, f in enumerate(features):
            if f.frame_idx >= frame_idx:
                return i
        return len(features) - 1

    def _create_single_segment(self, features: List[FrameFeatures]) -> ActionSegment:
        return self._build_segment(0, features)

    def refine_boundaries_with_energy(self, boundaries: List[Boundary],
                                      features: List[FrameFeatures],
                                      search_window: int = 10) -> List[Boundary]:
        """Refine boundary positions to align with local minimum energy points."""
        refined = []
        for boundary in boundaries:
            idx = self._frame_to_feature_idx(boundary.frame_idx, features)
            start = max(0, idx - search_window)
            end = min(len(features), idx + search_window)

            energies = [
                max(features[i].hand_velocity_left, features[i].hand_velocity_right) +
                features[i].flow_magnitude * 2
                for i in range(start, end)
            ]

            if energies:
                refined_idx = start + int(np.argmin(energies))
                refined.append(Boundary(
                    frame_idx=features[refined_idx].frame_idx,
                    timestamp=features[refined_idx].timestamp,
                    confidence=boundary.confidence,
                    reason=boundary.reason,
                    signal_strengths=boundary.signal_strengths,
                ))
            else:
                refined.append(boundary)
        return refined

    def adaptive_segment(self, features: List[FrameFeatures],
                         target_segments: Optional[int] = None) -> Tuple[List[ActionSegment], List[Boundary]]:
        """Segment with adaptive threshold to hit a target segment count."""
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
            elif len(segments) > target_segments:
                low = mid
            else:
                high = mid

            if abs(len(segments) - target_segments) < abs(len(best_segments) - target_segments):
                best_segments, best_boundaries = segments, boundaries

        return best_segments, best_boundaries