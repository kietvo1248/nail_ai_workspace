"""
===============================================================================
TRAIN_FINETUNE_5CLASS.PY - Fine-tune tu best.pt cua run truoc
===============================================================================
Muc tieu: Sua phan classification (model da segmentation tot, gan class kem)
Dataset: 5 classes (index/middle/pinky/ring/thumb)

Hyperparameter quan trong:
  - cls=2.0          : Tang class loss weight -> phan biet 5 ngon
  - fliplr=0.0       : TAT flip ngang (tranh dao Left<->Right -> index<->ring, thumb<->pinky)
  - mosaic=0.8       : Giam tu 1.0 -> 0.8 (tranh class confusion tu mosaic)
  - degrees=30.0     : Xoay +-30 do -> cover du huong mong da dang
  - perspective=0.0005 : Them perspective distortion

Su dung:
    python train_finetune_5class.py
    python train_finetune_5class.py --base runs/<ten_run>/weights/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE_ROOT: Path = Path(__file__).resolve().parent


def find_latest_base_model() -> Path:
    """Tim best.pt moi nhat trong runs/ de fine-tune tu no."""
    runs_dir = WORKSPACE_ROOT / "runs"
    if not runs_dir.exists():
        return None
    candidates = list(runs_dir.glob("**/weights/best.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_data_yaml() -> Path | None:
    """Tim data.yaml trong workspace (uu tien patched Seg, sau do dataset ben ngoai)."""
    candidates = [
        WORKSPACE_ROOT / "data.yaml",
        WORKSPACE_ROOT / "data_Nail_v1i_yolov11.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Tim trong parent (cung cap)
    parent = WORKSPACE_ROOT.parent
    for ds in parent.iterdir():
        if ds.is_dir():
            dy = ds / "data.yaml"
            if dy.exists():
                return dy
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune YOLO11-Seg tu best.pt cu.")
    p.add_argument("--base", type=str, default=None,
                   help="Duong dan best.pt goc (mac dinh: tu dong tim moi nhat trong runs/)")
    p.add_argument("--data", type=str, default=None,
                   help="Duong dan data.yaml (mac dinh: tu dong tim)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="0",
                   help="GPU id (0/1/...), 'cpu', hoac 'auto'")
    p.add_argument("--name", type=str, default="nail-segmentation_finetune_v2")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    base_model = Path(args.base) if args.base else find_latest_base_model()
    if base_model is None or not base_model.exists():
        print(f"[ERROR] Khong tim thay best.pt nao trong runs/.")
        print(f"        Hay chi dinh: python train_finetune_5class.py --base <path/best.pt>")
        print(f"        Hoac chay:   python train.py de train tu dau truoc.")
        return 1
    base_model = base_model.resolve()

    data_yaml = Path(args.data) if args.data else find_data_yaml()
    if data_yaml is None or not data_yaml.exists():
        print(f"[ERROR] Khong tim thay data.yaml. Hay dat dataset vao parent folder hoac chi dinh --data.")
        return 1
    data_yaml = data_yaml.resolve()

    print("=" * 70)
    print("FINE-TUNE CONFIGURATION")
    print("=" * 70)
    print(f"  Base model  : {base_model}")
    print(f"  Data yaml   : {data_yaml}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Image size  : {args.imgsz}")
    print(f"  Batch size  : {args.batch}")
    print(f"  Device      : {args.device}")
    print(f"  Run name    : {args.name}")
    print("=" * 70)
    print()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics chua duoc cai dat.")
        print("        Chay: python setup_env.py")
        return 1

    print("Loading base model for fine-tuning...")
    model = YOLO(str(base_model))

    print("Starting fine-tune training...")
    print("  Goal: Fix finger classification (index/middle/pinky/ring/thumb)")
    print("  Key changes: cls=2.0, fliplr=0.0, degrees=30, mosaic=0.8")

    results = model.train(
        # Dataset
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,

        # Classification: fix pinky bias
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
        patience=25,
        lr0=0.005,
        lrf=0.01,
        warmup_epochs=2.0,

        # Output
        name=args.name,
        project=str(WORKSPACE_ROOT / "runs"),
        exist_ok=True,
    )

    print("Fine-tune complete.")
    print(f"  Results: {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
