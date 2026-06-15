import cv2
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
from visualizer import Visualizer
from utils import FPSCounter


class ActionSplitterPipeline:
    """Main pipeline that orchestrates all modules."""

    def __init__(self, config: Config):
        self.config = config
        self.video_loader = VideoLoader(
            config.input_video_path,
            resize=config.frame_resize,
            fps_target=config.fps_target
        )
        self.hand_tracker = HandTracker(
            detection_confidence=config.hand_detection_confidence,
            tracking_confidence=config.hand_tracking_confidence,
            max_hands=config.max_hands
        )
        self.object_detector = ObjectDetector(
            model_path=config.object_model_path,
            confidence=config.object_confidence,
            tool_classes=config.tool_classes
        )
        self.interaction_tracker = InteractionTracker(
            distance_threshold=config.interaction_distance_threshold,
            iou_threshold=config.interaction_iou_threshold
        )
        self.feature_extractor = FeatureExtractor(
            window_size=config.window_size
        )
        self.segmenter = TemporalSegmenter(
            boundary_threshold=config.boundary_threshold,
            min_segment_duration=config.min_segment_duration,
            smoothing_sigma=config.smoothing_sigma,
            fps=self.video_loader.effective_fps
        )
        self.visualizer = Visualizer(output_dir=config.output_dir)
        self.fps_counter = FPSCounter()

    def run(self) -> dict:
        """Execute the full pipeline."""
        print(f"Processing video: {self.config.input_video_path}")
        print(f"Duration: {self.video_loader.duration:.1f}s | "
              f"FPS: {self.video_loader.original_fps:.1f} | "
              f"Effective FPS: {self.video_loader.effective_fps:.1f}")

        # Phase 1: Process all frames
        frames_data = self._process_frames()

        # Phase 2: Perform temporal segmentation
        features = [fd['features'] for fd in frames_data]
        segments, boundaries = self.segmenter.segment(features)

        # Refine boundaries to energy minima
        boundaries = self.segmenter.refine_boundaries_with_energy(
            boundaries, features
        )

        # Phase 3: Assign segments to frames for visualization
        self._assign_segments_to_frames(frames_data, segments)

        # Phase 4: Generate outputs
        print(f"\nDetected {len(segments)} action segments with "
              f"{len(boundaries)} boundaries")
        self._generate_outputs(frames_data, segments, boundaries)

        return {
            "segments": segments,
            "boundaries": boundaries,
            "total_duration": self.video_loader.duration
        }

    def _process_frames(self) -> list:
        """Process each frame through all tracking modules."""
        frames_data = []
        total = self.video_loader.total_frames // self.video_loader.frame_skip

        for frame_idx, frame in tqdm(self.video_loader.frames(),
                                      total=total, desc="Processing"):
            self.fps_counter.tick()
            timestamp = self.video_loader.frame_to_time(frame_idx)

            # Track hands
            hands = self.hand_tracker.process_frame(frame)

            # Detect objects
            objects = self.object_detector.detect(frame)
            tools = self.object_detector.filter_tools(objects)

            # Track interactions
            interactions = self.interaction_tracker.process_frame(
                hands, tools, frame_idx, timestamp
            )

            # Extract features
            contact_shift = self.interaction_tracker.get_contact_point_shift()
            interaction_density = self.interaction_tracker.get_interaction_density()

            features = self.feature_extractor.extract(
                frame_idx=frame_idx,
                timestamp=timestamp,
                hands=hands,
                objects=tools,
                interactions=interactions,
                contact_shift=contact_shift,
                interaction_density=interaction_density
            )

            frames_data.append({
                'frame_idx': frame_idx,
                'frame': frame,
                'hands': hands,
                'objects': tools,
                'interactions': interactions,
                'features': features,
                'segment': None
            })

        return frames_data

    def _assign_segments_to_frames(self, frames_data: list,
                                    segments: list):
        """Assign each frame to its corresponding segment."""
        for frame_data in frames_data:
            fidx = frame_data['frame_idx']
            for segment in segments:
                if segment.start_frame <= fidx <= segment.end_frame:
                    frame_data['segment'] = segment
                    break

    def _generate_outputs(self, frames_data: list,
                          segments: list, boundaries: list):
        """Generate all output artifacts."""
        # Export JSON results
        self.visualizer.export_results(segments, boundaries)
        print(f"Results saved to {self.config.output_dir}/segmentation_results.json")

        # Generate timeline image
        self.visualizer.generate_timeline(
            segments, boundaries, self.video_loader.duration
        )
        print(f"Timeline saved to {self.config.output_dir}/timeline.png")

        # Create annotated video
        self.visualizer.create_annotated_video(
            self.video_loader, frames_data, segments
        )
        print(f"Annotated video saved to {self.config.output_dir}/annotated_output.mp4")

        # Export individual clips
        if self.config.export_clips:
            clips_dir = Path(self.config.output_dir) / "clips"
            clips_dir.mkdir(exist_ok=True)
            for segment in segments:
                clip_path = str(clips_dir / f"step_{segment.segment_id + 1}.mp4")
                self.video_loader.export_clip(
                    segment.start_frame, segment.end_frame, clip_path
                )
            print(f"Individual clips saved to {self.config.output_dir}/clips/")

        # Print summary
        print("\n" + "=" * 60)
        print("SEGMENTATION SUMMARY")
        print("=" * 60)
        for s in segments:
            print(f"  Step {s.segment_id + 1}: "
                  f"{s.start_time:.1f}s - {s.end_time:.1f}s "
                  f"({s.duration:.1f}s) | "
                  f"Activity: {s.dominant_activity} | "
                  f"Tools: {', '.join(s.tools_used) or 'none'} | "
                  f"Confidence: {s.confidence:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Split egocentric video into action steps"
    )
    parser.add_argument("--video", "-v", required=True, help="Input video path")
    parser.add_argument("--output", "-o", default="output/", help="Output directory")
    parser.add_argument("--threshold", "-t", type=float, default=0.4,
                       help="Boundary detection threshold")
    parser.add_argument("--min-duration", type=float, default=2.0,
                       help="Minimum segment duration in seconds")
    parser.add_argument("--fps", type=int, default=15,
                       help="Target processing FPS")
    parser.add_argument("--no-clips", action="store_true",
                       help="Skip exporting individual clips")

    args = parser.parse_args()

    config = Config(
        input_video_path=args.video,
        output_dir=args.output,
        boundary_threshold=args.threshold,
        min_segment_duration=args.min_duration,
        fps_target=args.fps,
        export_clips=not args.no_clips
    )

    pipeline = ActionSplitterPipeline(config)
    results = pipeline.run()


if __name__ == "__main__":
    main()
