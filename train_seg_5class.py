"""
================================================================================
TRAIN_SEG_5CLASS.PY - Train YOLO11-Seg 5-class model for finger labelling
================================================================================
Trains a YOLOv11n-Seg model that predicts BOTH segmentation mask AND finger class
(index / middle / pinky / ring / thumb). Use this when you need the model's
``cls_name`` directly in the desktop app, instead of the heuristic in
``overlay.assign_finger_names``.

The training pipeline is similar to ``train_seg.py``, but adds the constraint
that the dataset must have **5 classes** (the standard nail finger set) and uses
the same augmentation parameters as ``train_obb_5class.py`` to learn rotations.

Augmentation choices are tuned for the 5-finger problem:
    - ``degrees=180.0`` (on) - learn nail orientation at any angle.
    - ``fliplr=0.0`` (off)  - flipping a hand image swaps thumb/pinky which
      would teach the model the wrong association between image-x and finger
      identity.
    - ``copy_paste=0.3``    - increase data variance by copying nails.
    - ``cls=2.0``           - heavier loss weight on classification.

Usage:
    python train_seg_5class.py --data ../Nail_Detection_ThanhDT.v1i.yolov11/data.yaml
    python train_seg_5class.py --data ../Nail_Detection_ThanhDT.v1i.yolov11/data.yaml --epochs 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE_ROOT: Path = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO11-Seg 5-class (per-finger) model.",
    )
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data.yaml (must have 5 classes).")
    parser.add_argument("--model", type=str, default="yolo11n-seg.pt",
                        help="Base model weights (default: yolo11n-seg.pt).")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Number of training epochs (default: 300).")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (default: 640).")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (default: 16).")
    parser.add_argument("--patience", type=int, default=50,
                        help="Early-stopping patience (default: 50).")
    parser.add_argument("--workers", type=int, default=8,
                        help="Dataloader workers (default: 8).")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device id, 'cpu', or '' for auto (default: '0').")
    parser.add_argument("--project", type=str, default="runs/seg",
                        help="Project directory (default: runs/seg).")
    parser.add_argument("--name", type=str, default=None,
                        help="Run name (default: <dataset>_seg_5class_<timestamp>).")
    parser.add_argument("--degrees", type=float, default=180.0,
                        help="Rotation augmentation in degrees (default: 180).")
    parser.add_argument("--no-strict-5class", action="store_true",
                        help="Allow datasets with class count != 5.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as e:
        print(f"ERROR: ultralytics not installed ({e}).")
        print("Run: pip install ultralytics")
        return 1

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        print(f"ERROR: data.yaml not found: {data_yaml}")
        return 1

    print("=" * 70)
    print("TRAIN_SEG_5CLASS - YOLO11 Seg 5-class per-finger")
    print("=" * 70)

    # Verify 5 classes if strict mode is on.
    try:
        import yaml
        with open(data_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        nc = data.get("nc")
        names = data.get("names", [])
        if nc != 5 and not args.no_strict_5class:
            print(f"WARN: dataset has nc={nc} (expected 5). Continuing anyway.")
            print(f"      Classes: {names}")
            print("      Use --no-strict-5class to silence this warning.")
    except Exception as e:
        print(f"WARN: could not verify class count: {e}")

    name = args.name
    if not name:
        from datetime import datetime
        stem = data_yaml.parent.name.replace(" ", "_").replace(".", "_")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{stem}_seg_5class_{stamp}"

    print()
    print(f"Base model : {args.model}")
    print(f"Data yaml  : {data_yaml}")
    print(f"Epochs     : {args.epochs}")
    print(f"Image size : {args.imgsz}")
    print(f"Batch size : {args.batch}")
    print(f"Patience   : {args.patience}")
    print(f"Workers    : {args.workers}")
    print(f"Device     : {args.device or 'auto'}")
    print(f"Project    : {args.project}")
    print(f"Name       : {name}")
    print(f"Augment    : degrees={args.degrees}, fliplr=0.0 (locked for 5-class)")
    print("=" * 70)

    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=name,
        patience=args.patience,
        workers=args.workers,
        plots=True,
        verbose=True,
        # Augmentation parameters identical to OBB for maximum robustness
        degrees=args.degrees,
        translate=0.2,
        scale=0.9,
        shear=10.0,
        perspective=0.001,
        fliplr=0.0,            # hard-locked off for 5-class to keep thumb vs pinky distinct
        flipud=0.0,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        erasing=0.4,
        cls=2.0,               # heavily penalize wrong finger class
    )

    print("\nTraining complete.")
    print(f"Best checkpoint: {args.project}/{name}/weights/best.pt")
    print("Next step: run ``export_all.py --run {name}`` to produce ONNX.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
