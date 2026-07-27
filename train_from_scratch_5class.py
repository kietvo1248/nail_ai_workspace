"""
===============================================================================
TRAIN_FROM_SCRATCH_5CLASS.PY - Train tu yolo11n-seg.pt goc
===============================================================================
Muc tieu: Train hoan toan moi tu base model, voi hyperparameter dung.
Chi chay khi fine-tune khong cai thien classification.

Hyperparameter quan trong:
  - cls=2.0          : Tang class loss weight -> phan biet 5 ngon
  - fliplr=0.0       : TAT flip ngang (tranh dao index<->ring, thumb<->pinky)
  - mosaic=0.8       : Giam tu 1.0 -> 0.8
  - degrees=30.0     : Xoay +-30 do
  - perspective=0.0005 : Them perspective distortion

Su dung:
    python train_from_scratch_5class.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE_ROOT: Path = Path(__file__).resolve().parent


def find_data_yaml() -> Path | None:
    candidates = [
        WORKSPACE_ROOT / "data.yaml",
        WORKSPACE_ROOT / "data_Nail_v1i_yolov11.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p
    parent = WORKSPACE_ROOT.parent
    for ds in parent.iterdir():
        if ds.is_dir():
            dy = ds / "data.yaml"
            if dy.exists():
                return dy
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO11-Seg tu dau (5 classes).")
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--model", type=str, default="yolo11n-seg.pt",
                   help="Base model (se download tu Ultralytics neu khong co)")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--name", type=str, default="nail-segmentation_scratch_v2")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    data_yaml = Path(args.data) if args.data else find_data_yaml()
    if data_yaml is None or not data_yaml.exists():
        print(f"[ERROR] Khong tim thay data.yaml. Hay dat dataset vao parent folder hoac chi dinh --data.")
        return 1
    data_yaml = data_yaml.resolve()

    print("=" * 70)
    print("FROM-SCRATCH TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"  Base model : {args.model}")
    print(f"  Data yaml  : {data_yaml}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Image size : {args.imgsz}")
    print(f"  Batch size : {args.batch}")
    print(f"  Device     : {args.device}")
    print(f"  Run name   : {args.name}")
    print("=" * 70)
    print()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics chua duoc cai dat. Chay: python setup_env.py")
        return 1

    print("Loading yolo11n-seg.pt as base model...")
    model = YOLO(args.model)

    print("Starting training from scratch...")
    print("  Goal: Train fresh model with correct hyperparameters")

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,

        # Classification
        cls=2.0,
        cls_pw=0.0,

        # Augmentation
        degrees=30.0,
        shear=5.0,
        perspective=0.0005,
        fliplr=0.0,
        mosaic=0.8,
        mixup=0.1,

        # Training strategy
        patience=30,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3.0,
        weight_decay=0.0005,
        momentum=0.937,

        # Output
        name=args.name,
        project=str(WORKSPACE_ROOT / "runs"),
        exist_ok=True,
    )

    print("Training from scratch complete.")
    print(f"  Results: {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
