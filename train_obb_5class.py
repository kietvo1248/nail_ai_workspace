"""
================================================================================
TRAIN_OBB_5CLASS.PY - Train YOLO11-OBB 5-class model for finger labelling
================================================================================
Trains a YOLOv11n-OBB model that predicts BOTH orientation AND finger class
(index / middle / pinky / ring / thumb). Use this when you need the model's
``cls_name`` directly in the desktop app, instead of the heuristic in
``overlay.assign_finger_names``.

The training pipeline is identical to ``train_obb.py`` (dataset-agnostic), but
adds the constraint that the dataset must have **5 classes** (the standard
nail finger set). Smoke-test datasets that still carry the legacy ``Index``
typo class will fall back gracefully (the model will learn 6 classes if the
dataset has 6).

Augmentation choices are tuned for the 5-finger problem:

    - ``fliplr=0.0`` (off)  - flipping a hand image swaps thumb/pinky which
      would teach the model the wrong association between image-x and finger
      identity.

Usage:
    python train_obb_5class.py --data ../nail-segmentation.v1i.yolov11_10501/data.yaml
    python train_obb_5class.py --data ../Nail_Detection_ThanhDT.v1i.yolov11/data.yaml --epochs 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE_ROOT: Path = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE_ROOT))

import train_obb as _train_obb  # reuse auto-tune + ensure_obb_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO11-OBB 5-class (per-finger) model.",
    )
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data.yaml (must have 5 classes).")
    parser.add_argument("--model", type=str, default="yolo11n-obb.pt",
                        help="Base model weights (default: yolo11n-obb.pt).")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs (default: auto-tuned).")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Override image size (default: auto-tuned).")
    parser.add_argument("--batch", type=int, default=None,
                        help="Override batch size (default: auto-tuned).")
    parser.add_argument("--patience", type=int, default=None,
                        help="Override early-stopping patience (default: auto-tuned).")
    parser.add_argument("--workers", type=int, default=None,
                        help="Override dataloader workers (default: auto-tuned).")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device id, 'cpu', or '' for auto (default: '0').")
    parser.add_argument("--project", type=str, default="runs/obb",
                        help="Project directory (default: runs/obb).")
    parser.add_argument("--name", type=str, default=None,
                        help="Run name (default: <dataset>_5class_<timestamp>).")
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
    print("TRAIN_OBB_5CLASS - YOLO11 OBB 5-class per-finger")
    print("=" * 70)

    obb_data_yaml = _train_obb.ensure_obb_labels(data_yaml)
    print(f"Data yaml (OBB): {obb_data_yaml}")

    # Verify 5 classes if strict mode is on.
    try:
        import yaml
        with open(obb_data_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        nc = data.get("nc")
        names = data.get("names", [])
        if nc != 5 and not args.no_strict_5class:
            print(f"WARN: dataset has nc={nc} (expected 5). Continuing anyway.")
            print(f"      Classes: {names}")
            print("      Use --no-strict-5class to silence this warning.")
    except Exception as e:
        print(f"WARN: could not verify class count: {e}")

    dataset_size = _train_obb.count_train_images(obb_data_yaml)
    tuned = _train_obb.auto_tune(dataset_size)
    epochs = args.epochs if args.epochs is not None else tuned["epochs"]
    imgsz = args.imgsz if args.imgsz is not None else tuned["imgsz"]
    batch = args.batch if args.batch is not None else tuned["batch"]
    patience = args.patience if args.patience is not None else tuned["patience"]
    workers = args.workers if args.workers is not None else tuned["workers"]

    name = args.name
    if not name:
        from datetime import datetime
        stem = data_yaml.parent.name.replace(" ", "_").replace(".", "_")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{stem}_5class_{stamp}"

    print()
    print(f"Base model : {args.model}")
    print(f"Data yaml  : {obb_data_yaml}")
    print(f"Dataset sz : {dataset_size}")
    print(f"Epochs     : {epochs}")
    print(f"Image size : {imgsz}")
    print(f"Batch size : {batch}")
    print(f"Patience   : {patience}")
    print(f"Workers    : {workers}")
    print(f"Device     : {args.device or 'auto'}")
    print(f"Project    : {args.project}")
    print(f"Name       : {name}")
    print(f"Augment    : degrees={args.degrees}, fliplr=0.0 (locked for 5-class)")
    print("=" * 70)

    model = YOLO(args.model)
    results = model.train(
        data=str(obb_data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=args.device,
        project=args.project,
        name=name,
        patience=patience,
        workers=workers,
        plots=True,
        verbose=True,
        degrees=args.degrees,
        translate=0.2,
        scale=0.9,
        shear=10.0,
        perspective=0.001,
        fliplr=0.0,            # hard-locked off for 5-class
        flipud=0.0,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        erasing=0.4,
        cls=2.0,
    )

    print("\nTraining complete.")
    print(f"Best checkpoint: {args.project}/{name}/weights/best.pt")
    print("Next step: run ``export_all.py --obb --run {name}`` to produce ONNX.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
