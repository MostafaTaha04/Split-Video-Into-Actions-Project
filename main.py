import argparse
from pathlib import Path
from tqdm import tqdm

from config import Config
from video_loader import VideoLoader
from hand_tracker import HandTracker
from object_detector import ObjectDetector
from interaction_tracker import InteractionTracker
from feature_extractor import FeatureExtractor
from temporal_segmenter import TemporalSegmenter
from optical_flow import OpticalFlowAnalyzer, FlowData
from scene_detector import SceneChangeDetector, SceneChangeData
from visualizer import Visualizer
from evaluator import Evaluator
from utils import FPSCounter


class ActionSplitterPipeline:
    """Main pipeline that orchestrates all modules."""

    def __init__(self, config: Config):
        self.config = config

        self.video_loader = VideoLoader(
            config.input_video_path,
            resize=config.frame_resize,
            target_fps=config.fps_target,
        )

        self.hand_tracker = HandTracker(
            detection_confidence=config.hand_detection_confidence,
            tracking_confidence=config.hand_tracking_confidence,
            max_hands=config.max_hands,
            grip_smoothing_window=config.grip_smoothing_window,
            model_asset_path=config.hand_model_path,
        )

        self.object_detector = ObjectDetector(
            model_path=config.object_model_path,
            confidence=config.object_confidence,
            tool_classes=config.tool_classes,
            detector_mode=config.object_detector_mode,
            open_vocab_model_path=config.open_vocab_model_path,
            open_vocab_interval=config.open_vocab_interval,
            open_vocab_imgsz=config.open_vocab_imgsz,
            max_det=config.max_det,
            debug=config.detector_debug,
        )

        self.interaction_tracker = InteractionTracker(
            distance_threshold=config.interaction_distance_threshold,
            iou_threshold=config.interaction_iou_threshold,
        )

        self.optical_flow = (
            OpticalFlowAnalyzer(window=config.flow_window)
            if config.optical_flow_enabled
            else None
        )

        self.scene_detector = (
            SceneChangeDetector(threshold=config.scene_change_threshold)
            if config.scene_detection_enabled
            else None
        )

        self.feature_extractor = FeatureExtractor(
            window_size=config.window_size,
            flow_weight=config.flow_discontinuity_weight,
            scene_weight=config.scene_change_weight,
        )

        self.segmenter = TemporalSegmenter(
            boundary_threshold=config.boundary_threshold,
            min_segment_duration=config.min_segment_duration,
            smoothing_sigma=config.smoothing_sigma,
            fps=self.video_loader.effective_fps,
        )

        self.visualizer = Visualizer(output_dir=config.output_dir)

        self.evaluator = (
            Evaluator(config.ground_truth_path)
            if config.ground_truth_path
            else None
        )

        self.fps_counter = FPSCounter()

    def run(self) -> dict:
        print(f"Processing video: {self.config.input_video_path}")
        print(
            f"Duration: {self.video_loader.duration:.1f}s | "
            f"FPS: {self.video_loader.original_fps:.1f} | "
            f"Effective FPS: {self.video_loader.effective_fps:.1f}"
        )

        print("Modules: Hand Tracking + Tool/Component Detection + Interaction Tracking")
        print(f"         + Detector mode: {self.config.object_detector_mode}")
        print(f"         + Object confidence: {self.config.object_confidence:.2f}")

        if self.config.object_detector_mode in {"open_vocab", "hybrid"}:
            print(
                f"         + Open vocabulary model: "
                f"{self.config.object_model_path or self.config.open_vocab_model_path}"
            )
            print(
                f"         + Open vocabulary interval: "
                f"every {self.config.open_vocab_interval} processed frames"
            )

        if self.optical_flow:
            print(f"         + Optical Flow (weight={self.config.flow_discontinuity_weight})")

        if self.scene_detector:
            print(f"         + Scene Change Detection (threshold={self.config.scene_change_threshold})")

        frames_data = self._process_frames()

        features = [fd["features"] for fd in frames_data]

        segments, boundaries = self.segmenter.segment(features)

        self._assign_segments_to_frames(frames_data, segments)

        print(f"\nDetected {len(segments)} action segments with {len(boundaries)} boundaries")

        self._generate_outputs(frames_data, segments, boundaries, features)

        eval_results = None

        if self.evaluator:
            eval_results = self._run_evaluation(segments, boundaries)

        return {
            "segments": segments,
            "boundaries": boundaries,
            "total_duration": self.video_loader.duration,
            "evaluation": eval_results,
        }

    def _process_frames(self) -> list:
        frames_data = []

        total = max(
            1,
            self.video_loader.total_frames // self.video_loader.frame_skip,
        )

        max_frames = getattr(self.config, "max_frames", None)
        processed = 0
        for frame_idx, frame in tqdm(
            self.video_loader.frames(),
            total=(min(total, max_frames) if max_frames else total),
            desc="Processing",
        ):
            if max_frames and processed >= max_frames:
                break
            processed += 1
            self.fps_counter.tick()

            timestamp = self.video_loader.frame_to_time(frame_idx)

            hands = self.hand_tracker.process_frame(frame)

            objects = self.object_detector.detect(frame)
            tools = self.object_detector.filter_tools(objects)

            interactions = self.interaction_tracker.process_frame(
                hands,
                tools,
                frame_idx,
                timestamp,
            )

            if self.optical_flow:
                flow_data = self.optical_flow.compute(frame)
            else:
                flow_data = FlowData(
                    flow_field=None,
                    magnitude_mean=0.0,
                    magnitude_max=0.0,
                    dominant_direction=0.0,
                    motion_uniformity=1.0,
                    discontinuity_score=0.0,
                )

            if self.scene_detector:
                scene_data = self.scene_detector.process_frame(frame)
            else:
                scene_data = SceneChangeData(
                    histogram_change=0.0,
                    structural_change=0.0,
                    combined_score=0.0,
                    is_boundary=False,
                )

            contact_shift = self.interaction_tracker.get_contact_point_shift()
            contact_variance = self.interaction_tracker.get_contact_point_variance()
            interaction_density = self.interaction_tracker.get_interaction_density()
            interaction_rhythm = self.interaction_tracker.get_interaction_rhythm()
            tool_stability = self.object_detector.get_tool_stability()

            curvature = 0.0

            for hand in hands:
                curvature = max(
                    curvature,
                    self.hand_tracker.get_trajectory_curvature(hand.handedness),
                )

            features = self.feature_extractor.extract(
                frame_idx=frame_idx,
                timestamp=timestamp,
                hands=hands,
                objects=tools,
                interactions=interactions,
                contact_shift=contact_shift,
                contact_variance=contact_variance,
                interaction_density=interaction_density,
                interaction_rhythm=interaction_rhythm,
                flow_data=flow_data,
                scene_data=scene_data,
                tool_stability=tool_stability,
                trajectory_curvature=curvature,
            )

            frames_data.append({
                "frame_idx": frame_idx,
                "frame": frame,
                "hands": hands,
                "objects": tools,
                "interactions": interactions,
                "features": features,
                "flow_data": flow_data,
                "scene_data": scene_data,
                "segment": None,
            })

        print(f"\nProcessing complete. Avg FPS: {self.fps_counter.get_fps():.1f}")

        return frames_data

    def _assign_segments_to_frames(self, frames_data: list, segments: list):
        for frame_data in frames_data:
            fidx = frame_data["frame_idx"]

            for segment in segments:
                if segment.start_frame <= fidx <= segment.end_frame:
                    frame_data["segment"] = segment
                    break

    def _generate_outputs(
        self,
        frames_data: list,
        segments: list,
        boundaries: list,
        features: list,
    ):
        video_meta = self.video_loader.get_metadata()

        self.visualizer.export_results(segments, boundaries, video_meta)
        print(f"Results: {self.config.output_dir}/segmentation_results.json")

        self.visualizer.generate_timeline(
            segments,
            boundaries,
            self.video_loader.duration,
            features,
        )
        print(f"Timeline: {self.config.output_dir}/timeline.png")

        feature_names = self.feature_extractor.get_feature_names()

        self.visualizer.export_feature_csv(features, feature_names)
        print(f"Features: {self.config.output_dir}/features.csv")

        self.visualizer.create_annotated_video(
            self.video_loader,
            frames_data,
            segments,
            draw_flow=self.config.draw_optical_flow,
        )
        print(f"Video: {self.config.output_dir}/annotated_output.mp4")

        if self.config.export_clips:
            clips_dir = Path(self.config.output_dir) / "clips"
            clips_dir.mkdir(exist_ok=True)

            for segment in segments:
                clip_path = str(clips_dir / f"step_{segment.segment_id + 1}.mp4")
                self.video_loader.export_clip(
                    segment.start_frame,
                    segment.end_frame,
                    clip_path,
                )

            print(f"Clips: {self.config.output_dir}/clips/")

        print("\n" + "=" * 70)
        print("SEGMENTATION SUMMARY")
        print("=" * 70)

        for s in segments:
            objects = ", ".join(getattr(s, "real_objects_used", [])) or "none"
            rois = ", ".join(getattr(s, "heuristic_regions", [])) or "none"

            print(
                f"  Step {s.segment_id + 1}: "
                f"{s.start_time:.1f}s - {s.end_time:.1f}s "
                f"({s.duration:.1f}s) | "
                f"Activity: {s.activity_description or s.dominant_activity} | "
                f"Objects: {objects} | "
                f"ROIs: {rois} | "
                f"Motion: {s.avg_motion_energy:.1f} | "
                f"Conf: {s.confidence:.2f} | "
                f"ActivityConf: {s.activity_confidence:.2f}"
            )

        print("=" * 70)

    def _run_evaluation(self, segments, boundaries) -> dict:
        print("\n--- EVALUATION ---")

        results = self.evaluator.evaluate(
            segments,
            boundaries,
            tolerance=self.config.boundary_tolerance,
        )

        report = self.evaluator.generate_report(
            segments,
            boundaries,
            output_path=str(Path(self.config.output_dir) / "evaluation_report.txt"),
        )

        print(report)

        return results


def _parse_classes_arg(value: str):
    if not value:
        return None

    return [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Split egocentric video into action steps"
    )

    parser.add_argument("--video", "-v", required=True, help="Input video path")
    parser.add_argument("--output", "-o", default="output/", help="Output directory")
    parser.add_argument("--threshold", "-t", type=float, default=0.32, help="Boundary detection threshold")
    parser.add_argument("--min-duration", type=float, default=1.5, help="Minimum segment duration in seconds")
    parser.add_argument("--fps", type=int, default=15, help="Target processing FPS")
    parser.add_argument("--no-clips", action="store_true", help="Skip exporting individual clips")
    parser.add_argument("--no-flow", action="store_true", help="Disable optical flow analysis")
    parser.add_argument("--no-scene", action="store_true", help="Disable scene change detection")
    parser.add_argument("--ground-truth", "-g", type=str, default=None, help="Path to ground truth JSON for evaluation")
    parser.add_argument("--tolerance", type=float, default=1.0, help="Boundary tolerance in seconds for evaluation")

    parser.add_argument(
        "--detector",
        choices=["workspace", "open_vocab", "yolo", "hybrid", "none"],
        default="workspace",
        help="Detector mode. Use open_vocab for YOLO-World hardware prompts.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model path. For open_vocab, default is yolov8s-worldv2.pt; for yolo, use hardware_model.pt",
    )

    parser.add_argument(
        "--open-vocab-interval",
        type=int,
        default=3,
        help="Run open-vocabulary detection every N processed frames",
    )

    parser.add_argument("--debug-detections", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--open-vocab-imgsz", type=int, default=1280)
    parser.add_argument("--max-det", type=int, default=50)
    parser.add_argument("--resize", type=str, default=None)
    parser.add_argument("--hand-model", type=str, default=None)

    parser.add_argument(
        "--object-confidence",
        type=float,
        default=None,
        help="Override object detector confidence threshold, e.g. 0.08 for YOLO-World hardware.",
    )

    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Comma-separated custom hardware prompts/classes",
    )

    parser.add_argument(
        "--grip-window",
        type=int,
        default=5,
        help="Grip smoothing window in frames",
    )

    parser.add_argument("--draw-flow", action="store_true", help="Draw optical flow arrows on output video")

    args = parser.parse_args()

    config = Config(
        input_video_path=args.video,
        output_dir=args.output,
        boundary_threshold=args.threshold,
        min_segment_duration=args.min_duration,
        fps_target=args.fps,
        export_clips=not args.no_clips,
        optical_flow_enabled=not args.no_flow,
        scene_detection_enabled=not args.no_scene,
        ground_truth_path=args.ground_truth,
        boundary_tolerance=args.tolerance,
        object_detector_mode=args.detector,
        object_model_path=args.model,
        open_vocab_interval=args.open_vocab_interval,
        open_vocab_imgsz=args.open_vocab_imgsz,
        max_det=args.max_det,
        grip_smoothing_window=args.grip_window,
        hand_model_path=args.hand_model,
        draw_optical_flow=args.draw_flow,
        detector_debug=args.debug_detections,
        max_frames=args.max_frames,
    )

    if args.object_confidence is not None:
        config.object_confidence = args.object_confidence

    if args.resize:
        try:
            w_str, h_str = args.resize.lower().split("x")
            config.frame_resize = (int(w_str), int(h_str))
        except Exception:
            print("WARNING: bad --resize")

    custom_classes = _parse_classes_arg(args.classes)

    if custom_classes:
        config.tool_classes = [
            "motherboard_workspace",
            "cpu_socket_region",
            "active_motion_region",
        ] + custom_classes

    pipeline = ActionSplitterPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()