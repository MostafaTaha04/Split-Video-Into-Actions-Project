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
from optical_flow import FlowData


class Visualizer:
    """Handles all visualization and output generation."""

    COLORS = [
        (255, 100, 100),
        (100, 255, 100),
        (100, 100, 255),
        (255, 255, 100),
        (255, 100, 255),
        (100, 255, 255),
        (200, 150, 50),
        (50, 200, 150),
        (150, 50, 200),
    ]

    ROI_CLASSES = {
        "motherboard_workspace",
        "cpu_socket_region",
        "active_motion_region",
    }

    def __init__(self, output_dir: str = "output/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def draw_frame(
        self,
        frame: np.ndarray,
        hands: List[HandData],
        objects: List[DetectedObject],
        interactions: List[Interaction],
        current_segment: Optional[ActionSegment] = None,
        features: Optional[FrameFeatures] = None,
        flow_data: Optional[FlowData] = None,
        draw_flow: bool = False,
    ) -> np.ndarray:
        annotated = frame.copy()

        if draw_flow and flow_data and flow_data.flow_field is not None:
            annotated = self._draw_flow_overlay(annotated, flow_data)

        self._draw_objects(annotated, objects)
        self._draw_hands(annotated, hands)
        self._draw_interactions(annotated, interactions)

        if current_segment:
            self._draw_segment_info(annotated, current_segment)

        if features:
            self._draw_feature_overlay(annotated, features)

        return annotated

    def _draw_hands(self, frame: np.ndarray, hands: List[HandData]):
        for hand in hands:
            color = (0, 255, 0) if hand.handedness == "Right" else (0, 200, 255)

            x, y, w, h = hand.bounding_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            for point in hand.fingertip_positions:
                cv2.circle(frame, tuple(point), 5, color, -1)

            cv2.circle(frame, tuple(hand.palm_center), 8, color, 2)

            if hand.is_gripping:
                cv2.putText(
                    frame,
                    "GRIP",
                    (x, max(15, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

            vel_text = f"v={hand.velocity:.1f} a={hand.acceleration:.1f}"

            cv2.putText(
                frame,
                vel_text,
                (x, y + h + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
            )

    def _draw_objects(self, frame: np.ndarray, objects: List[DetectedObject]):
        for obj in objects:
            x1, y1, x2, y2 = obj.bbox
            is_roi = obj.class_name in self.ROI_CLASSES

            color = (80, 160, 255) if is_roi else (0, 255, 255)
            thickness = 1 if is_roi else 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            prefix = "ROI" if is_roi else "OBJ"
            label = f"{prefix}: {obj.class_name} ({obj.confidence:.2f})"

            if obj.track_id is not None:
                label += f" #{obj.track_id}"

            cv2.putText(
                frame,
                label,
                (x1, max(12, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                color,
                1,
            )

    def _draw_interactions(
        self,
        frame: np.ndarray,
        interactions: List[Interaction],
    ):
        for interaction in interactions:
            hand_center = tuple(interaction.hand.palm_center)
            obj_center = tuple(interaction.obj.center.astype(int))

            color_map = {
                "grasp": (0, 0, 255),
                "use": (255, 0, 255),
                "touch": (0, 255, 255),
                "approach": (128, 128, 128),
            }

            color = color_map.get(interaction.interaction_type, (255, 255, 255))

            cv2.line(frame, hand_center, obj_center, color, 2)

            if interaction.contact_point is not None:
                cp = tuple(interaction.contact_point.astype(int))
                cv2.circle(frame, cp, 6, (0, 0, 255), -1)
                cv2.circle(frame, cp, 8, (255, 255, 255), 1)

            mid = (
                (hand_center[0] + obj_center[0]) // 2,
                (hand_center[1] + obj_center[1]) // 2,
            )

            cv2.putText(
                frame,
                interaction.interaction_type,
                mid,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
            )

    def _draw_flow_overlay(
        self,
        frame: np.ndarray,
        flow_data: FlowData,
    ) -> np.ndarray:
        flow = flow_data.flow_field
        h, w = flow.shape[:2]

        step = 16
        y_coords, x_coords = np.mgrid[step // 2:h:step, step // 2:w:step]

        fx = flow[y_coords, x_coords, 0]
        fy = flow[y_coords, x_coords, 1]

        for i in range(y_coords.shape[0]):
            for j in range(y_coords.shape[1]):
                pt1 = (int(x_coords[i, j]), int(y_coords[i, j]))
                pt2 = (
                    int(x_coords[i, j] + fx[i, j] * 3),
                    int(y_coords[i, j] + fy[i, j] * 3),
                )

                mag = np.sqrt(fx[i, j] ** 2 + fy[i, j] ** 2)

                if mag > 1.0:
                    cv2.arrowedLine(
                        frame,
                        pt1,
                        pt2,
                        (0, 255, 0),
                        1,
                        tipLength=0.3,
                    )

        return frame

    def _draw_segment_info(self, frame: np.ndarray, segment: ActionSegment):
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h - 110), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

        color = self.COLORS[segment.segment_id % len(self.COLORS)]
        activity = segment.activity_description or segment.dominant_activity

        text = f"Step {segment.segment_id + 1}: {activity}"

        cv2.putText(
            frame,
            text[:80],
            (20, h - 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

        real_objects = getattr(segment, "real_objects_used", [])
        rois = getattr(segment, "heuristic_regions", [])

        obj_text = f"Objects: {', '.join(real_objects[:4]) if real_objects else 'none'}"
        roi_text = f"ROIs: {', '.join(rois[:3]) if rois else 'none'}"

        cv2.putText(
            frame,
            obj_text[:95],
            (20, h - 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (230, 230, 230),
            1,
        )

        cv2.putText(
            frame,
            roi_text[:95],
            (20, h - 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (170, 170, 170),
            1,
        )

        conf_text = (
            f"SegConf: {segment.confidence:.2f} | "
            f"ActConf: {segment.activity_confidence:.2f} | "
            f"Motion: {segment.avg_motion_energy:.1f}"
        )

        cv2.putText(
            frame,
            conf_text,
            (20, h - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.33,
            (180, 180, 180),
            1,
        )

    def _draw_feature_overlay(self, frame: np.ndarray, features: FrameFeatures):
        texts = [
            f"Activity: {features.activity_level:.2f}",
            f"Transition: {features.transition_score:.2f}",
            f"Flow: {features.flow_magnitude:.1f}",
            f"Scene: {features.scene_change_score:.2f}",
        ]

        # Moved slightly down to reduce clutter with object labels.
        for i, text in enumerate(texts):
            cv2.putText(
                frame,
                text,
                (10, 30 + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
            )

    def generate_timeline(
        self,
        segments: List[ActionSegment],
        boundaries: List[Boundary],
        total_duration: float,
        features: Optional[List[FrameFeatures]] = None,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        width = 1200
        height = 300 if features else 200

        timeline = np.ones((height, width, 3), dtype=np.uint8) * 30

        for i, segment in enumerate(segments):
            color = self.COLORS[i % len(self.COLORS)]

            x_start = int((segment.start_time / total_duration) * width)
            x_end = int((segment.end_time / total_duration) * width)

            cv2.rectangle(timeline, (x_start, 40), (x_end, 100), color, -1)
            cv2.rectangle(timeline, (x_start, 40), (x_end, 100), (255, 255, 255), 1)

            mid_x = (x_start + x_end) // 2
            label = f"Step {segment.segment_id + 1}"

            cv2.putText(
                timeline,
                label,
                (mid_x - 20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )

        for boundary in boundaries:
            x = int((boundary.timestamp / total_duration) * width)

            cv2.line(timeline, (x, 30), (x, 110), (0, 0, 255), 2)

            cv2.putText(
                timeline,
                f"{boundary.confidence:.2f}",
                (x - 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 0, 255),
                1,
            )

        if features:
            graph_top = 130
            graph_bottom = 270
            graph_height = graph_bottom - graph_top

            cv2.rectangle(
                timeline,
                (0, graph_top - 10),
                (width, graph_bottom + 10),
                (20, 20, 20),
                -1,
            )

            cv2.putText(
                timeline,
                "Transition Score",
                (10, graph_top + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (150, 150, 150),
                1,
            )

            scores = [f.transition_score for f in features]

            if scores:
                max_score = max(max(scores), 1.0)
                points = []

                for i, score in enumerate(scores):
                    x = int((i / len(scores)) * width)
                    y = graph_bottom - int((score / max_score) * graph_height)
                    points.append((x, y))

                for i in range(1, len(points)):
                    cv2.line(timeline, points[i - 1], points[i], (0, 255, 100), 1)

                thresh_y = graph_bottom - int(
                    (self.threshold_val(boundaries) / max_score) * graph_height
                )

                cv2.line(timeline, (0, thresh_y), (width, thresh_y), (100, 100, 255), 1)

        tick = max(1, int(total_duration / 10))

        for i in range(0, int(total_duration) + 1, tick):
            x = int((i / total_duration) * width)

            cv2.putText(
                timeline,
                f"{i}s",
                (x, 118),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (200, 200, 200),
                1,
            )

        path = output_path or str(self.output_dir / "timeline.png")
        cv2.imwrite(path, timeline)

        return timeline

    def threshold_val(self, boundaries: List[Boundary]) -> float:
        if boundaries:
            return min(b.confidence for b in boundaries) * 0.9

        return 0.4

    def export_results(
        self,
        segments: List[ActionSegment],
        boundaries: List[Boundary],
        video_metadata: Optional[dict] = None,
        output_path: Optional[str] = None,
    ):
        results = {
            "video_info": video_metadata or {},
            "num_segments": len(segments),
            "num_boundaries": len(boundaries),
            "segments": [
                {
                    "id": s.segment_id,
                    "start_time": round(s.start_time, 2),
                    "end_time": round(s.end_time, 2),
                    "duration": round(s.duration, 2),
                    "activity": s.activity_description or s.dominant_activity,
                    "activity_raw": s.dominant_activity,
                    "activity_reason": getattr(s, "activity_reason", ""),
                    "activity_confidence": round(getattr(s, "activity_confidence", 0.0), 3),
                    "real_objects": getattr(s, "real_objects_used", []),
                    "heuristic_rois": getattr(s, "heuristic_regions", []),
                    "all_objects_and_rois": s.tools_used,
                    "interactions": s.interaction_types,
                    "confidence": round(s.confidence, 3),
                    "motion_energy": round(s.avg_motion_energy, 3),
                    "visual_stability": round(s.visual_stability, 3),
                }
                for s in segments
            ],
            "boundaries": [
                {
                    "timestamp": round(b.timestamp, 2),
                    "confidence": round(b.confidence, 3),
                    "reason": b.reason,
                    "signal_strengths": b.signal_strengths,
                }
                for b in boundaries
            ],
        }

        path = output_path or str(self.output_dir / "segmentation_results.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    def create_annotated_video(
        self,
        video_loader,
        frames_data: list,
        segments: List[ActionSegment],
        draw_flow: bool = False,
        output_path: Optional[str] = None,
    ):
        path = output_path or str(self.output_dir / "annotated_output.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_size = video_loader.resize or (video_loader.width, video_loader.height)

        writer = cv2.VideoWriter(
            path,
            fourcc,
            video_loader.effective_fps,
            out_size,
        )

        for frame_data in frames_data:
            annotated = self.draw_frame(
                frame_data["frame"],
                frame_data["hands"],
                frame_data["objects"],
                frame_data["interactions"],
                frame_data.get("segment"),
                frame_data.get("features"),
                frame_data.get("flow_data"),
                draw_flow=draw_flow,
            )

            writer.write(annotated)

        writer.release()

    def export_feature_csv(
        self,
        features: List[FrameFeatures],
        feature_names: List[str],
        output_path: Optional[str] = None,
    ):
        path = output_path or str(self.output_dir / "features.csv")

        with open(path, "w", encoding="utf-8") as f:
            header = "frame_idx,timestamp," + ",".join(feature_names)
            f.write(header + "\n")

            for feat in features:
                row = [
                    str(feat.frame_idx),
                    f"{feat.timestamp:.3f}",
                ]

                row.extend([
                    f"{feat.hand_velocity_left:.3f}",
                    f"{feat.hand_velocity_right:.3f}",
                    f"{feat.hand_acceleration_left:.3f}",
                    f"{feat.hand_acceleration_right:.3f}",
                    str(feat.hands_present),
                    str(int(feat.grip_state_left)),
                    str(int(feat.grip_state_right)),
                    f"{feat.hand_distance:.3f}",
                    f"{feat.trajectory_curvature:.3f}",
                    str(feat.num_tools),
                    str(int(feat.tool_changed)),
                    f"{feat.tool_stability:.3f}",
                    str(feat.num_interactions),
                    f"{feat.contact_point_shift:.3f}",
                    f"{feat.contact_point_variance:.3f}",
                    f"{feat.interaction_density:.3f}",
                    f"{feat.interaction_rhythm:.3f}",
                    f"{feat.flow_magnitude:.3f}",
                    f"{feat.flow_uniformity:.3f}",
                    f"{feat.flow_discontinuity:.3f}",
                    f"{feat.direction_change:.3f}",
                    f"{feat.scene_change_score:.3f}",
                    f"{feat.visual_stability:.3f}",
                    f"{feat.activity_level:.3f}",
                    f"{feat.transition_score:.3f}",
                ])

                f.write(",".join(row) + "\n")