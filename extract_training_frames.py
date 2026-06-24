"""
extract_training_frames.py
--------------------------
Sample frames from one or more assembly videos so they can be labelled for a
custom hardware detector (see TRAINING_GUIDE.md). The script writes evenly
sampled JPEG frames into an output folder ready for annotation in a YOLO-format
labelling tool (labelImg, Label Studio, Roboflow, CVAT, ...).

Examples
--------
# Sample one frame every 0.5 s from every clip into dataset/images/
python extract_training_frames.py --videos *.mp4 --output dataset/images --every 0.5

# Sample a fixed number of frames per video
python extract_training_frames.py --videos CPUplacement.mp4 --output dataset/images --max-per-video 60
"""
import argparse
import glob
import os
from pathlib import Path

import cv2


def sample_video(
    video_path: str,
    out_dir: Path,
    every_seconds: float,
    max_per_video: int | None,
    resize: tuple[int, int] | None,
) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"WARNING: could not open {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, int(round(fps * every_seconds)))

    stem = Path(video_path).stem
    saved = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % step == 0:
            if resize is not None:
                frame = cv2.resize(frame, resize)

            out_path = out_dir / f"{stem}_f{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1

            if max_per_video is not None and saved >= max_per_video:
                break

        frame_idx += 1

    cap.release()
    print(f"  {video_path}: saved {saved} frames "
          f"(fps={fps:.1f}, total_frames={total}, step={step})")
    return saved


def main():
    ap = argparse.ArgumentParser(
        description="Extract frames from videos for custom-detector labelling."
    )
    ap.add_argument(
        "--videos",
        nargs="+",
        required=True,
        help="Video paths or globs, e.g. --videos *.mp4 CPUplacement.mp4",
    )
    ap.add_argument(
        "--output",
        "-o",
        default="dataset/images",
        help="Output folder for extracted frames.",
    )
    ap.add_argument(
        "--every",
        type=float,
        default=0.5,
        help="Sample one frame every N seconds (default 0.5).",
    )
    ap.add_argument(
        "--max-per-video",
        type=int,
        default=None,
        help="Cap the number of frames saved per video.",
    )
    ap.add_argument(
        "--resize",
        type=str,
        default=None,
        help="Resize frames before saving, e.g. 960x540.",
    )
    args = ap.parse_args()

    resize = None
    if args.resize:
        try:
            w_str, h_str = args.resize.lower().split("x")
            resize = (int(w_str), int(h_str))
        except Exception:
            print("WARNING: bad --resize, ignoring.")

    # Expand globs (and de-duplicate while preserving order).
    paths: list[str] = []
    for pattern in args.videos:
        matches = glob.glob(pattern)
        paths.extend(matches if matches else [pattern])
    paths = list(dict.fromkeys(paths))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting frames to: {out_dir}")
    total_saved = 0
    for video_path in paths:
        if not os.path.exists(video_path):
            print(f"WARNING: file not found: {video_path}")
            continue
        total_saved += sample_video(
            video_path, out_dir, args.every, args.max_per_video, resize
        )

    print(f"\nDone. {total_saved} frames written to {out_dir}.")
    print("Next: label these frames (YOLO format), then run train_hardware_model.py.")


if __name__ == "__main__":
    main()
