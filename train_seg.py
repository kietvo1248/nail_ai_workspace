"""
================================================================================
TRAIN_SEG.PY - Train YOLO-Seg model on Nail.v1i.yolov11 dataset
================================================================================
Trains a YOLOv11n-Seg model on the nail segmentation dataset and exports
the best checkpoint for downstream conversion (ONNX / TFLite).

Usage:
    python train_seg.py
    python train_seg.py --epochs 300 --imgsz 640 --batch 16

NOTE: This script is meant to be run manually by the user. Training 300
epochs on a single GPU can take several hours. Checkpoints are saved
to runs/seg/<name>/weights/best.pt and can then be exported with
``export_all.py``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE_ROOT: Path = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO-Seg on Nail.v1i.yolov11 dataset"
    )
    parser.add_argument("--data", type=str,
                        default=str(WORKSPACE_ROOT / "data.yaml"),
                        help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolo11n-seg.pt",
                        help="Base model weights (default: yolo11n-seg.pt)")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Number of training epochs (default: 300)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (default: 640)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device id, 'cpu', or '' for auto (default: 0)")
    parser.add_argument("--project", type=str, default="runs/seg",
                        help="Project directory (default: runs/seg)")
    parser.add_argument("--name", type=str, default="nail_seg",
                        help="Run name (default: nail_seg)")
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience (default: 50)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Dataloader workers (default: 8)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as e:
        print(f"ERROR: ultralytics not installed ({e}).")
        print("Run: pip install ultralytics")
        return 1

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: data.yaml not found at {data_path}")
        return 1

    print("=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"  Base model : {args.model}")
    print(f"  Data yaml  : {args.data}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Image size : {args.imgsz}")
    print(f"  Batch size : {args.batch}")
    print(f"  Device     : {args.device or 'auto'}")
    print(f"  Project    : {args.project}")
    print(f"  Name       : {args.name}")
    print(f"  Patience   : {args.patience}")
    print("=" * 70)

    model = YOLO(args.model)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        plots=True,
        verbose=True,
    )

    print("\nTraining complete.")
    print(f"Best checkpoint: {args.project}/{args.name}/weights/best.pt")
    print("Next step: run ``export_all.py`` to produce ONNX + TFLite models.")
    return 0


if __name__ == "__main__":
    sys.exit(main())