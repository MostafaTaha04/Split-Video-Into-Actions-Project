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

        self.frame_skip = 1
        if target_fps and target_fps < self.original_fps:
            self.frame_skip = max(1, int(self.original_fps / target_fps))

        self.effective_fps = self.original_fps / self.frame_skip

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
