import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Optional
from hand_tracker import HandData
from object_detector import DetectedObject
from interaction_tracker import Interaction
from temporal_segmenter import ActionSegment, Boundary
from feature_extractor import FrameFeatures


class Visualizer:
    """Handles all visualization and output generation."""

    COLORS = [
        (255, 100, 100), (100, 255, 100), (100, 100, 255),
        (255, 255, 100), (255, 100, 255), (100, 255, 255),
        (200, 150, 50), (50, 200, 150), (150, 50, 200),
    ]

    def __init__(self, output_dir: str = "output/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def draw_frame(self, frame: np.ndarray,
                   hands: List[HandData],
                   objects: List[DetectedObject],
                   interactions: List[Interaction],
                   current_segment: Optional[ActionSegment] = None,
                   features: Optional[FrameFeatures] = None) -> np.ndarray:
        """Draw all annotations on a frame."""
        annotated = frame.copy()

        self._draw_objects(annotated, objects)
        self._draw_hands(annotated, hands)
        self._draw_interactions(annotated, interactions)

        if current_segment:
            self._draw_segment_info(annotated, current_segment)

        if features:
            self._draw_feature_overlay(annotated, features)

        return annotated

    def _draw_hands(self, frame: np.ndarray, hands: List[HandData]):
        """Draw hand landmarks and bounding boxes."""
        for hand in hands:
            color = (0, 255, 0) if hand.handedness == "Right" else (0, 200, 255)

            x, y, w, h = hand.bounding_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            for point in hand.fingertip_positions:
                cv2.circle(frame, tuple(point), 5, color, -1)

            cv2.circle(frame, tuple(hand.palm_center), 8, color, 2)

            if hand.is_gripping:
                cv2.putText(frame, "GRIP", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            vel_text = f"v={hand.velocity:.1f}"
            cv2.putText(frame, vel_text, (x, y + h + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def _draw_objects(self, frame: np.ndarray, objects: List[DetectedObject]):
        """Draw detected objects."""
        for obj in objects:
            x1, y1, x2, y2 = obj.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
            label = f"{obj.class_name} ({obj.confidence:.2f})"
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)

    def _draw_interactions(self, frame: np.ndarray,
                           interactions: List[Interaction]):
        """Draw interaction indicators."""
        for interaction in interactions:
            hand_center = tuple(interaction.hand.palm_center)
            obj_center = tuple(interaction.obj.center.astype(int))

            color_map = {
                "grasp": (0, 0, 255),
                "use": (255, 0, 255),
                "touch": (0, 255, 255),
                "approach": (128, 128, 128)
            }
            color = color_map.get(interaction.interaction_type, (255, 255, 255))

            cv2.line(frame, hand_center, obj_center, color, 2)
            mid = ((hand_center[0] + obj_center[0]) // 2,
                   (hand_center[1] + obj_center[1]) // 2)
            cv2.putText(frame, interaction.interaction_type, mid,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def _draw_segment_info(self, frame: np.ndarray, segment: ActionSegment):
        """Draw current segment information overlay."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h - 80), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        text = f"Step {segment.segment_id + 1}: {segment.dominant_activity}"
        cv2.putText(frame, text, (20, h - 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if segment.tools_used:
            tools_text = f"Tools: {', '.join(segment.tools_used)}"
            cv2.putText(frame, tools_text, (20, h - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def _draw_feature_overlay(self, frame: np.ndarray, features: FrameFeatures):
        """Draw feature values as a small overlay."""
        texts = [
            f"Activity: {features.activity_level:.2f}",
            f"Transition: {features.transition_score:.2f}",
        ]
        for i, text in enumerate(texts):
            cv2.putText(frame, text, (10, 20 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def generate_timeline(self, segments: List[ActionSegment],
                          boundaries: List[Boundary],
                          total_duration: float,
                          output_path: Optional[str] = None) -> np.ndarray:
        """Generate a visual timeline of all segments."""
        width = 1200
        height = 200
        timeline = np.ones((height, width, 3), dtype=np.uint8) * 30

        for i, segment in enumerate(segments):
            color = self.COLORS[i % len(self.COLORS)]
            x_start = int((segment.start_time / total_duration) * width)
            x_end = int((segment.end_time / total_duration) * width)

            cv2.rectangle(timeline, (x_start, 40), (x_end, 120), color, -1)
            cv2.rectangle(timeline, (x_start, 40), (x_end, 120), (255, 255, 255), 1)

            mid_x = (x_start + x_end) // 2
            label = f"Step {segment.segment_id + 1}"
            cv2.putText(timeline, label, (mid_x - 20, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        for boundary in boundaries:
            x = int((boundary.timestamp / total_duration) * width)
            cv2.line(timeline, (x, 30), (x, 130), (0, 0, 255), 2)

        for i in range(0, int(total_duration) + 1, max(1, int(total_duration / 10))):
            x = int((i / total_duration) * width)
            cv2.putText(timeline, f"{i}s", (x, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        path = output_path or str(self.output_dir / "timeline.png")
        cv2.imwrite(path, timeline)
        return timeline

    def export_results(self, segments: List[ActionSegment],
                       boundaries: List[Boundary],
                       output_path: Optional[str] = None):
        """Export segmentation results as JSON."""
        results = {
            "num_segments": len(segments),
            "num_boundaries": len(boundaries),
            "segments": [
                {
                    "id": s.segment_id,
                    "start_time": round(s.start_time, 2),
                    "end_time": round(s.end_time, 2),
                    "duration": round(s.duration, 2),
                    "activity": s.dominant_activity,
                    "tools": s.tools_used,
                    "interactions": s.interaction_types,
                    "confidence": round(s.confidence, 3)
                }
                for s in segments
            ],
            "boundaries": [
                {
                    "timestamp": round(b.timestamp, 2),
                    "confidence": round(b.confidence, 3),
                    "reason": b.reason
                }
                for b in boundaries
            ]
        }

        path = output_path or str(self.output_dir / "segmentation_results.json")
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)

    def create_annotated_video(self, video_loader, frames_data: list,
                                segments: List[ActionSegment],
                                output_path: Optional[str] = None):
        """Create full annotated output video."""
        path = output_path or str(self.output_dir / "annotated_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_size = video_loader.resize or (video_loader.width, video_loader.height)
        writer = cv2.VideoWriter(path, fourcc, video_loader.effective_fps, out_size)

        for frame_data in frames_data:
            annotated = self.draw_frame(
                frame_data['frame'],
                frame_data['hands'],
                frame_data['objects'],
                frame_data['interactions'],
                frame_data.get('segment'),
                frame_data.get('features')
            )
            writer.write(annotated)

        writer.release()
