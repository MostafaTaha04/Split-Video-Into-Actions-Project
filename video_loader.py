import cv2
import numpy as np
from typing import Generator, Tuple, Optional


class VideoLoader:
    """Handles video loading, frame extraction, and clip export."""

    def __init__(self, video_path: str, resize: Optional[Tuple[int, int]] = None,
                 target_fps: Optional[int] = None):
        self.video_path = video_path
        self.resize = resize
        self.target_fps = target_fps

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.original_fps

        # Frame stride. This MUST round rather than truncate. Real footage is
        # 29.97 fps, not 30, so a request for 15 fps gives 29.97/15 = 1.998,
        # and int() floors that to 1 — no skipping at all. The result is a
        # silent 2x overshoot: twice the frames, twice the runtime, and a
        # different segmentation from the saved run at the same --fps value.
        # round() gives 2 and the effective rate lands on 14.985, which is what
        # clips.json records for the 29.97 fps clips.
        self.frame_skip = 1
        if target_fps and target_fps < self.original_fps:
            self.frame_skip = max(1, round(self.original_fps / target_fps))

        self.effective_fps = self.original_fps / self.frame_skip
        if target_fps and abs(self.effective_fps - target_fps) > 0.05 * target_fps:
            print(f"VideoLoader: requested {target_fps} fps; source is "
                  f"{self.original_fps:.3f} fps, so the achievable rate is "
                  f"{self.effective_fps:.3f} fps (every {self.frame_skip} frame"
                  f"{'s' if self.frame_skip > 1 else ''}).")

        # --- aspect-ratio-safe resize -------------------------------------
        # `--resize 960x540` used to be applied as a plain cv2.resize, which
        # forces landscape dimensions onto portrait footage. A 1080x1920 phone
        # clip became 960x540: aspect 0.56 squashed to 1.78, so every hand was
        # stretched ~3x horizontally. MediaPipe is trained on correctly
        # proportioned hands and detected NOTHING — hand tracking silently
        # produced all-zero features on all four portrait clips in this
        # project, including the two used for the headline results.
        #
        # The requested size is now treated as a size *budget* whose
        # orientation follows the source: a portrait video with a 960x540
        # budget is resized to 540x960. Same pixel count, correct proportions,
        # no letterboxing waste.
        self.resize = self._orient_resize(resize)
        if resize and self.resize != tuple(resize):
            print(f"VideoLoader: source is {self.width}x{self.height} "
                  f"({'portrait' if self.height > self.width else 'landscape'}); "
                  f"resize {resize[0]}x{resize[1]} re-oriented to "
                  f"{self.resize[0]}x{self.resize[1]} to preserve aspect ratio.")

    def _orient_resize(self, resize):
        """Match the requested size's orientation to the source's."""
        if not resize:
            return None
        rw, rh = int(resize[0]), int(resize[1])
        src_portrait = self.height > self.width
        req_portrait = rh > rw
        if src_portrait != req_portrait:
            rw, rh = rh, rw
        return (rw, rh)

    def frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Yield (frame_index, frame) tuples."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if frame_idx % self.frame_skip == 0:
                if self.resize:
                    frame = cv2.resize(frame, self.resize)
                yield frame_idx, frame

            frame_idx += 1

    def get_frame_at(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get a specific frame by index."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret and self.resize:
            frame = cv2.resize(frame, self.resize)
        return frame if ret else None

    def frame_to_time(self, frame_idx: int) -> float:
        """Convert frame index to timestamp in seconds."""
        return frame_idx / self.original_fps

    def time_to_frame(self, timestamp: float) -> int:
        """Convert timestamp to frame index."""
        return int(timestamp * self.original_fps)

    def export_clip(self, start_frame: int, end_frame: int, output_path: str):
        """Export a video clip between two frame indices."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_size = self.resize if self.resize else (self.width, self.height)
        writer = cv2.VideoWriter(output_path, fourcc, self.original_fps, out_size)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for idx in range(start_frame, end_frame):
            ret, frame = self.cap.read()
            if not ret:
                break
            if self.resize:
                frame = cv2.resize(frame, self.resize)
            writer.write(frame)

        writer.release()

    def get_metadata(self) -> dict:
        """Return video metadata as dictionary."""
        return {
            "path": self.video_path,
            "original_fps": self.original_fps,
            "effective_fps": self.effective_fps,
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
            "duration_seconds": self.duration,
            "frame_skip": self.frame_skip
        }

    def release(self):
        self.cap.release()

    def __del__(self):
        self.release()
