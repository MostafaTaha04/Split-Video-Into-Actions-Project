"""
features.py
===========
Turns a video into numbers the activity model can learn from: two "feature"
vectors per second of video (one every 0.5 s).

For each sampled frame:
  * appearance - a pretrained image model (ResNet-50, trained on ImageNet's
    1.2 million photos) describes the whole frame as 2048 numbers. It already
    knows what hands, tools, cables, fans and boards look like; our own model
    only has to learn which of those descriptions mean which activity.
  * crop       - the same, for a zoomed-in view of the work area: the centre
    and bottom of the frame, where the hands and the case are (CROP below).
    Small parts - a cable, a RAM stick, the cover's edge - are about twice as
    large there as in the whole frame.
  * motion     - how much the picture changed since the previous sample, in a
    4 x 4 grid over the frame, plus overall (17 numbers). Screwing, plugging and
    lifting move in different places and amounts.

The frames come from the browser copy of the video (outputs/<name>/preview.mp4,
960 px) rather than the 4K original, which is far slower to decode; it is made
with viewer.py if it does not exist yet. The image model is downloaded once from
GitHub (about 100 MB) into models/ and checked against a known checksum.

Results are cached in features/<name>.npz, because they depend only on the
video: later runs reuse them (fresh=True makes them again).

    python features.py                      # every video in data/videos
    python features.py pc_assembly_01       # only this one
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
import urllib.request
from typing import Optional

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
VIDEOS = os.path.join(ROOT, "data", "videos")
OUTPUTS = os.path.join(ROOT, "outputs")
FEATURES = os.path.join(ROOT, "features")
MODELS = os.path.join(ROOT, "models")

FPS = 2.0                 # samples per second of video
SIZE = 224                # the image model's input size
FEATURE_VERSION = 2       # bump when the way features are made changes
# The work area: the middle half of the width, from 35 % of the height down to the bottom.
CROP = "crop=iw*0.5:ih*0.65:iw*0.25:ih*0.35"

# ResNet-50 v1 (ONNX model zoo, opset 12), from GitHub. sha256 checked after download.
BACKBONE_FILE = "resnet50-v1-12.onnx"
BACKBONE_SHA256 = "3f03fdef724b22947eed826f1eef1dc5c34151bb4c37d634f1db89dfa2dd1526"
BACKBONE_URLS = [
    "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/resnet/model/resnet50-v1-12.onnx",
    "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-12.onnx",
]
BACKBONE_FEATURE_NODE = "flatten_473"   # the 2048 numbers before the final ImageNet layer
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


# --------------------------------------------------------------------------
# the image model
# --------------------------------------------------------------------------
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def backbone_path() -> str:
    """Path of the ResNet-50 file, downloading it the first time."""
    os.makedirs(MODELS, exist_ok=True)
    path = os.path.join(MODELS, BACKBONE_FILE)
    if os.path.exists(path) and _sha256(path) == BACKBONE_SHA256:
        return path
    last = None
    for url in BACKBONE_URLS:
        try:
            print(f"Downloading the image model (about 100 MB, once) from\n  {url}")
            tmp = path + ".part"
            urllib.request.urlretrieve(url, tmp)
            if _sha256(tmp) != BACKBONE_SHA256:
                raise ValueError("downloaded file does not match the expected checksum")
            os.replace(tmp, path)
            return path
        except Exception as e:  # try the next address
            last = e
            print(f"  failed: {e}")
    raise RuntimeError(f"Could not download {BACKBONE_FILE}: {last}. Download it from "
                       f"{BACKBONE_URLS[-1]} into the models folder and run again.")


_SESSION = None


def check_packages() -> None:
    """Stop with a clear message if the packages that run the image model are missing."""
    missing = []
    for module in ("onnx", "onnxruntime"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise RuntimeError(f"Missing Python package(s) {', '.join(missing)}, needed to describe the frames of "
                           "a new video. Install them with:  pip install -r requirements.txt")


def backbone():
    """An onnxruntime session that returns the 2048 feature numbers per image."""
    global _SESSION
    if _SESSION is None:
        import onnx
        import onnxruntime as ort
        m = onnx.load(backbone_path())
        if BACKBONE_FEATURE_NODE not in [o.name for o in m.graph.output]:
            m.graph.output.append(onnx.helper.make_tensor_value_info(
                BACKBONE_FEATURE_NODE, onnx.TensorProto.FLOAT, None))
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        _SESSION = ort.InferenceSession(m.SerializeToString(), opts, providers=["CPUExecutionProvider"])
    return _SESSION


def embed(frames: np.ndarray) -> np.ndarray:
    """frames: (n, 224, 224, 3) uint8 RGB -> (n, 2048) float32."""
    x = frames.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    x = (x - MEAN) / STD
    sess = backbone()
    return sess.run([BACKBONE_FEATURE_NODE], {sess.get_inputs()[0].name: x})[0].reshape(len(frames), -1)


# --------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------
def motion_features(prev: Optional[np.ndarray], cur: np.ndarray) -> np.ndarray:
    """Mean absolute change between two samples, in a 4 x 4 grid and overall (17 numbers, 0-1)."""
    g = cur.astype(np.float32).mean(axis=2)[::4, ::4] / 255.0          # 56 x 56 grey
    if prev is None:
        return np.zeros(17, dtype=np.float32), g
    d = np.abs(g - prev)
    cells = d.reshape(4, 14, 4, 14).mean(axis=(1, 3)).ravel()
    return np.concatenate([cells, [d.mean()]]).astype(np.float32), g


def ensure_preview(name: str, video: Optional[str] = None) -> str:
    """outputs/<name>/preview.mp4, made from the original video if it does not exist yet."""
    preview = os.path.join(OUTPUTS, name, "preview.mp4")
    video = video or os.path.join(VIDEOS, name + ".mp4")
    if os.path.exists(preview) and (not os.path.exists(video) or os.path.getmtime(preview) > os.path.getmtime(video)):
        return preview
    if not os.path.exists(video):
        raise FileNotFoundError(video)
    from viewer import find_ffmpeg, make_preview, video_info
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found: pip install -r requirements.txt")
    os.makedirs(os.path.dirname(preview), exist_ok=True)
    print(f"  making a smaller copy of {name}.mp4 first (a few minutes for a 4K video)...")
    make_preview(ffmpeg, video, preview + ".part.mp4", video_info(video))
    os.replace(preview + ".part.mp4", preview)
    return preview


def read_frames(video: str, fps: float = FPS, size: int = SIZE, crop: str = ""):
    """Yield (time_s, 224x224x3 RGB uint8) samples, fps per second, the whole frame (or the crop)
    squeezed to a square."""
    from viewer import find_ffmpeg
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found: pip install -r requirements.txt")
    vf = f"fps={fps}," + (crop + "," if crop else "") + f"scale={size}:{size}:flags=area"
    cmd = [ffmpeg, "-v", "error", "-i", video, "-an", "-vf", vf,
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = size * size * 3
    i = 0
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield i / fps, np.frombuffer(buf, dtype=np.uint8).reshape(size, size, 3)
            i += 1
    finally:
        if p.poll() is None:   # stopped early: end ffmpeg quietly
            p.kill()
        p.stdout.close()
        p.wait()


# --------------------------------------------------------------------------
# per video, cached
# --------------------------------------------------------------------------
def cache_path(name: str) -> str:
    return os.path.join(FEATURES, name + ".npz")


def compute(name: str, video: Optional[str] = None, batch: int = 16) -> dict:
    """video: the original video (default data/videos/<name>.mp4); frames are read from its preview."""
    check_packages()   # before the slow part, so a missing package is reported at once
    src = ensure_preview(name, video)
    t0 = time.time()
    times, motion, emb, cemb, buf, cbuf = [], [], [], [], [], []
    prev = None
    for (t, fr), (_, cr) in zip(read_frames(src), read_frames(src, crop=CROP)):
        m, prev = motion_features(prev, fr)
        times.append(t)
        motion.append(m)
        buf.append(fr)
        cbuf.append(cr)
        if len(buf) == batch:
            emb.append(embed(np.stack(buf)))
            cemb.append(embed(np.stack(cbuf)))
            buf, cbuf = [], []
            if len(times) % (batch * 20) == 0:
                print(f"  {name}: {times[-1] / 60:.1f} min of video done ({time.time() - t0:.0f} s)")
    if buf:
        emb.append(embed(np.stack(buf)))
        cemb.append(embed(np.stack(cbuf)))
    if not times:
        raise RuntimeError(f"no frames could be read from {src}")
    out = {"times": np.array(times, dtype=np.float32),
           "appearance": np.concatenate(emb).astype(np.float16),
           "crop": np.concatenate(cemb).astype(np.float16),
           "motion": np.stack(motion).astype(np.float32),
           "fps": np.float32(FPS), "version": np.int32(FEATURE_VERSION),
           "preview_bytes": np.int64(os.path.getsize(src))}
    os.makedirs(FEATURES, exist_ok=True)
    np.savez_compressed(cache_path(name), **out)
    print(f"  {name}: {len(times)} samples in {time.time() - t0:.0f} s")
    return out


def load(name: str, fresh: bool = False, video: Optional[str] = None) -> dict:
    """The features of one video, from the cache when it is up to date."""
    path = cache_path(name)
    if not fresh and os.path.exists(path):
        d = dict(np.load(path))
        preview = os.path.join(OUTPUTS, name, "preview.mp4")
        same_video = (not os.path.exists(preview) or "preview_bytes" not in d
                      or int(d["preview_bytes"]) == os.path.getsize(preview))
        if int(d.get("version", -1)) == FEATURE_VERSION and same_video:
            return d
    print(f"Describing the frames of {name} (once per video; cached in features/)...")
    return compute(name, video)


if __name__ == "__main__":
    import glob
    names = sys.argv[1:] or [os.path.splitext(os.path.basename(v))[0]
                             for v in sorted(glob.glob(os.path.join(VIDEOS, "*.mp4")))]
    for n in names:
        d = load(n, fresh="--fresh" in sys.argv)
        print(n, d["appearance"].shape, d["motion"].shape)
