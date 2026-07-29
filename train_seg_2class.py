"""
================================================================================
TRAIN_SEG_2CLASS.PY - Train YOLO11-Seg 2-class model (nail_bed + full_nail)
================================================================================
Trains a YOLOv11n-Seg model that predicts TWO segmentation classes:
    - class 0: nail_bed   (portion near cuticle, where design is applied)
    - class 1: full_nail  (entire nail plate, for size measurement)

Finger names (index/middle/ring/pinky/thumb) are NOT in the model.
They are assigned at inference time using MediaPipe hand landmarks.

The training pipeline:
    1. Run convert_polygon_to_nailbed.py to generate labels_2class/ from labels/
    2. Run this script to train the 2-class model

Augmentation choices (tuned for nail segmentation):
    - degrees=180.0   (on)  - learn nail orientation at any angle
    - fliplr=0.0     (off)  - do not flip (would swap thumb/pinky)
    - copy_paste=0.3         - increase data variance
    - shear=10.0             - learn nail bed boundary deformation
    - cls=2.0                - heavier loss weight on classification

Usage:
    # First, generate 2-class labels:
    python convert_polygon_to_nailbed.py \
        --labels "D:/.../Nail_Detection_ThanhDT.v1i.yolov11/train/labels" \
        --output "D:/.../Nail_Detection_ThanhDT.v1i.yolov11/train/labels_2class" \
        --bed-ratio 0.75 \
        --valid "D:/.../Nail_Detection_ThanhDT.v1i.yolov11/valid/labels" \
        --valid-output "D:/.../Nail_Detection_ThanhDT.v1i.yolov11/valid/labels_2class"

    # Then train:
    python train_seg_2class.py \
        --data ../Nail_Detection_ThanhDT.v1i.yolov11/data_2class.yaml \
        --epochs 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE_ROOT: Path = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO11-Seg 2-class (nail_bed + full_nail) model.",
    )
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data_2class.yaml (must have 2 classes).")
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
                        help="Run name (default: <dataset>_seg_2class_<timestamp>).")
    parser.add_argument("--degrees", type=float, default=180.0,
                        help="Rotation augmentation in degrees (default: 180).")
    parser.add_argument("--no-strict-2class", action="store_true",
                        help="Allow datasets with class count != 2.")
    return parser.parse_args()


def verify_labels_exist(data_yaml: Path) -> bool:
    """Check that labels_2class directories exist for train and val."""
    dataset_root = data_yaml.parent
    train_labels = dataset_root / "train" / "labels_2class"
    valid_labels = dataset_root / "valid" / "labels_2class"

    issues = []
    if not train_labels.exists():
        issues.append(f"  MISSING: {train_labels}")
    if not valid_labels.exists():
        issues.append(f"  MISSING: {valid_labels}")

    if issues:
        print("ERROR: Required label directories not found.")
        print("Run convert_polygon_to_nailbed.py first to generate labels_2class/")
        print("\nIssues:")
        for issue in issues:
            print(issue)
        return False

    train_count = len(list(train_labels.glob("*.txt")))
    valid_count = len(list(valid_labels.glob("*.txt")))
    print(f"  Train labels: {train_count} files")
    print(f"  Valid labels: {valid_count} files")
    return True


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
    print("TRAIN_SEG_2CLASS - YOLO11 Seg 2-class (nail_bed + full_nail)")
    print("=" * 70)

    # Verify 2 classes if strict mode is on.
    try:
        import yaml
        with open(data_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        nc = data.get("nc")
        names = data.get("names", [])
        if nc != 2 and not args.no_strict_2class:
            print(f"WARN: dataset has nc={nc} (expected 2). Continuing anyway.")
            print(f"      Classes: {names}")
            print("      Use --no-strict-2class to silence this warning.")
        else:
            print(f"  Classes: {names}")
    except Exception as e:
        print(f"WARN: could not verify class count: {e}")

    # Verify labels_2class directories exist
    print("\nVerifying labels_2class directories...")
    if not verify_labels_exist(data_yaml):
        return 1

    name = args.name
    if not name:
        from datetime import datetime
        stem = data_yaml.parent.name.replace(" ", "_").replace(".", "_")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{stem}_seg_2class_{stamp}"

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
    print(f"Augment    : degrees={args.degrees}, fliplr=0.0, copy_paste=0.3, cls=2.0")
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
        amp=False,             # Disable AMP to prevent CUDA misaligned address error
        # Augmentation tuned for nail bed boundary segmentation
        degrees=args.degrees,
        translate=0.2,
        scale=0.9,
        shear=10.0,
        perspective=0.001,
        fliplr=0.0,            # hard-locked off: flipping swaps thumb/pinky
        flipud=0.0,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,         # helps learn nail bed boundary variance
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        erasing=0.4,
        cls=2.0,               # heavier loss on classification for 2 classes
    )

    print("\nTraining complete.")
    print(f"Best checkpoint: {args.project}/{name}/weights/best.pt")
    print("\nNext steps:")
    print("  1. Run export_all.py to produce ONNX")
    print("  2. Update nail_desktop_app/onnx_inference.py for 2-class parsing")
    print("  3. Update nail_desktop_app/overlay.py to use nail_bed_polygon as clip mask")
    return 0


if __name__ == "__main__":
    sys.exit(main())
