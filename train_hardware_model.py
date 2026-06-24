"""
train_hardware_model.py
-----------------------
Fine-tune a small YOLOv8 detector on labelled assembly frames to produce a
custom `hardware_model.pt`. This is the detector the main pipeline loads with
`--detector yolo --model hardware_model.pt` (see TRAINING_GUIDE.md for the full
workflow: frame extraction -> labelling -> data.yaml -> training).

The script is a thin, documented wrapper around the Ultralytics training API so
the training run is reproducible from one command.

Example
-------
python train_hardware_model.py --data dataset/data.yaml --epochs 100 \
    --base yolov8n.pt --imgsz 960 --name hardware
# Best weights are copied to ./hardware_model.pt on success.
"""
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description="Train a custom YOLO hardware detector on labelled frames."
    )
    ap.add_argument(
        "--data",
        required=True,
        help="Path to the YOLO dataset config (data.yaml).",
    )
    ap.add_argument(
        "--base",
        default="yolov8n.pt",
        help="Base model to fine-tune (yolov8n.pt is fast; yolov8s.pt is stronger).",
    )
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument(
        "--name",
        default="hardware",
        help="Run name under runs/detect/<name>.",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="Training device, e.g. 0 for the first GPU, or cpu. Default: auto.",
    )
    ap.add_argument(
        "--out",
        default="hardware_model.pt",
        help="Where to copy the best weights after training.",
    )
    args = ap.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(
            f"Dataset config not found: {args.data}\n"
            "Create it as described in TRAINING_GUIDE.md (a data.yaml pointing at "
            "your train/val images and listing the class names)."
        )

    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "ultralytics is required to train. Install it with:\n"
            "    python -m pip install --upgrade ultralytics\n"
            f"(import error: {exc})"
        )

    print(f"Fine-tuning {args.base} on {args.data} for {args.epochs} epochs...")
    model = YOLO(args.base)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device,
        verbose=True,
    )

    # Locate and copy the best checkpoint to a predictable path.
    save_dir = Path(getattr(results, "save_dir", f"runs/detect/{args.name}"))
    best = save_dir / "weights" / "best.pt"

    if best.exists():
        shutil.copy(best, args.out)
        print(f"\nTraining complete. Best weights copied to: {args.out}")
        print(
            "Use it with the pipeline:\n"
            f"    python main.py --video <clip>.mp4 --detector yolo --model {args.out} ..."
        )
    else:
        print(
            "\nTraining finished, but best.pt was not found at "
            f"{best}. Check the run folder: {save_dir}"
        )


if __name__ == "__main__":
    main()
