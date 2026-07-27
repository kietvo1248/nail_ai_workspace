"""
================================================================================
TRAIN_OBB.PY - Train YOLO11-OBB on any YOLO dataset (auto-converts from Seg)
================================================================================
Trains a YOLOv11n-OBB model on the given dataset. The script is
**dataset-agnostic**:

    - Pass ``--data <path/to/data.yaml>`` for any YOLO-Seg / YOLO-OBB dataset.
    - If the dataset has Seg labels but no OBB labels, this script will call
      ``convert_seg_to_obb.py`` to produce ``labels_obb/`` automatically.
    - Hyperparameters are **auto-adjusted** based on dataset size: smoke-test
      datasets (small image count) get fewer epochs and smaller batch.

The augmentation profile is heavy to cover all nail orientations (the source
data has rotated hands and thumbs pointing every direction):

    degrees=180        # full 360-degree rotation
    translate=0.2
    scale=0.9
    shear=10
    perspective=0.001
    fliplr=0.0         # off - flipping would invert thumb/pinky
    flipud=0.0
    mosaic=1.0
    mixup=0.15
    copy_paste=0.3
    hsv_h=0.02 hsv_s=0.8 hsv_v=0.5
    erasing=0.4
    cls=2.0

Usage:
    # Smoke test (139 images) - auto-adjusts hyperparams to 20 epochs / batch 8
    python train_obb.py --data ../Nail_Detection_ThanhDT.v1i.yolov11/data.yaml --epochs 20 --imgsz 416

    # Production (10,501 images) - default 150 epochs / batch 16
    python train_obb.py --data ../nail-segmentation.v1i.yolov11_10501/data.yaml --epochs 150 --imgsz 640
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

WORKSPACE_ROOT: Path = Path(__file__).resolve().parent


# =============================================================================
# AUTO-TUNE HYPERPARAMS
# =============================================================================
def auto_tune(dataset_size: int) -> dict:
    """Pick sensible defaults based on the number of training images.

    Returns a dict that callers can use to fill in any arg the user did not
    specify on the CLI.
    """
    if dataset_size < 500:
        # Smoke test regime: small dataset -> fewer epochs, smaller batch.
        return {
            "epochs": 30,
            "imgsz": 416,
            "batch": 8,
            "patience": 10,
            "workers": 2,
        }
    if dataset_size < 3000:
        return {
            "epochs": 80,
            "imgsz": 640,
            "batch": 16,
            "patience": 25,
            "workers": 4,
        }
    # Production regime.
    return {
        "epochs": 150,
        "imgsz": 640,
        "batch": 16,
        "patience": 30,
        "workers": 8,
    }


# =============================================================================
# DATA AUTO-CONVERT
# =============================================================================
def _ensure_val_split(data_yaml: Path) -> None:
    """Run auto_split.py if the dataset has no usable validation split.

    Ultralytics refuses to train when ``val`` points at a missing or empty
    directory. Smoke-test datasets typically ship without one.
    """
    import yaml as _yaml
    try:
        with open(data_yaml, "r", encoding="utf-8") as f:
            data = _yaml.safe_load(f) or {}
    except Exception:
        return

    val_rel = data.get("val")
    if not val_rel:
        return

    dataset_root = data.get("path")
    if dataset_root:
        dataset_root = Path(dataset_root).expanduser().resolve()
    else:
        dataset_root = data_yaml.parent.resolve()

    val_path = Path(val_rel)
    if not val_path.is_absolute():
        cand1 = (data_yaml.parent / val_path).resolve()
        cand2 = (dataset_root / val_path).resolve()
        # Try to match Roboflow's fallback: dataset_root/val/images.
        val_path = cand1 if cand1.exists() else cand2

    val_has_files = False
    if val_path.exists():
        for ext in ("jpg", "jpeg", "png", "bmp", "JPG", "JPEG", "PNG", "BMP"):
            if any(val_path.glob(f"*.{ext}")):
                val_has_files = True
                break

    if val_has_files:
        return

    print(f"[auto-split] No val/ files found - running auto_split.py")
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location(
        "auto_split",
        WORKSPACE_ROOT / "auto_split.py",
    )
    mod = _ilu.module_from_spec(spec)
    sys.modules[spec.name] = mod
    # Patch sys.argv so auto_split.main() picks up our --data.
    saved_argv = sys.argv
    sys.argv = ["auto_split.py", "--data", str(data_yaml), "--val-ratio", "0.2"]
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    finally:
        sys.argv = saved_argv


def count_train_images(data_yaml: Path) -> int:
    """Count how many images are in the train/ split of the given data.yaml."""
    try:
        import yaml as _yaml
        with open(data_yaml, "r", encoding="utf-8") as f:
            data = _yaml.safe_load(f) or {}
    except Exception:
        return -1

    dataset_root = data.get("path")
    if dataset_root:
        dataset_root = Path(dataset_root).expanduser().resolve()
    else:
        dataset_root = data_yaml.parent.resolve()

    train_rel = data.get("train", "train/images")
    train_path = Path(train_rel)
    if not train_path.is_absolute():
        cand1 = (data_yaml.parent / train_path).resolve()
        cand2 = (dataset_root / train_path).resolve()
        train_path = cand1 if cand1.exists() else cand2
    train_path = train_path.resolve()
    if not train_path.exists():
        # Fallback: dataset_root/train/images
        train_path = dataset_root / "train" / "images"

    if not train_path.exists():
        return -1
    n = 0
    for ext in ("jpg", "jpeg", "png", "bmp", "JPG", "JPEG", "PNG", "BMP"):
        n += len(list(train_path.glob(f"*.{ext}")))
    return n


def ensure_obb_labels(data_yaml: Path) -> Path:
    """If the dataset has Seg labels but no OBB labels, run the converter.

    Returns the data.yaml path to use for OBB training (may be the patched
    data_obb.yaml if conversion happened).
    """
    import yaml as _yaml

    try:
        with open(data_yaml, "r", encoding="utf-8") as f:
            data = _yaml.safe_load(f) or {}
    except Exception:
        return data_yaml

    # Detect dataset_root and train images path.
    dataset_root = data.get("path")
    if dataset_root:
        dataset_root = Path(dataset_root).expanduser().resolve()
    else:
        dataset_root = data_yaml.parent.resolve()
    train_rel = data.get("train", "train/images")
    train_path = Path(train_rel)
    if not train_path.is_absolute():
        cand1 = (data_yaml.parent / train_path).resolve()
        cand2 = (dataset_root / train_path).resolve()
        train_path = cand1 if cand1.exists() else cand2

    labels_dir = train_path.parent / "labels"
    obb_labels_dir = train_path.parent / "labels_obb"

    # If labels_obb/ is missing or empty, run the converter.
    need_convert = (not obb_labels_dir.exists()) or (
        not any(obb_labels_dir.glob("*.txt"))
    )
    if need_convert:
        if not labels_dir.exists() or not any(labels_dir.glob("*.txt")):
            # Dataset has no Seg labels at all - assume it's already OBB.
            return data_yaml
        print(f"[auto-convert] No labels_obb/ found - running convert_seg_to_obb.py")
        # Run the converter as a module call (avoid subprocess overhead).
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "convert_seg_to_obb",
            WORKSPACE_ROOT / "convert_seg_to_obb.py",
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        try:
            spec.loader.exec_module(mod)
        except SystemExit:
            pass
        # Look for the produced data_obb.yaml next to the input data.yaml.
        patched = data_yaml.parent / "data_obb.yaml"
        if patched.exists():
            return patched
        print("WARN: conversion finished but data_obb.yaml was not produced.")
        return data_yaml
    # labels_obb/ exists - look for a sibling data_obb.yaml; otherwise build
    # one on the fly.
    patched = data_yaml.parent / "data_obb.yaml"
    if patched.exists():
        return patched
    # Build the patched yaml directly.
    nc = data.get("nc", 0)
    names = data.get("names", [])
    print(f"[auto-convert] labels_obb/ exists - writing data_obb.yaml")
    from convert_seg_to_obb import write_patched_data_yaml
    val_rel = data.get("val")
    splits = [("train", train_path)]
    if val_rel:
        from convert_seg_to_obb import _resolve_split_path
        val_path = _resolve_split_path(data_yaml, dataset_root, val_rel)
        if val_path.exists():
            splits.append(("val", val_path))
    write_patched_data_yaml(
        source_data_yaml=data_yaml,
        dataset_root=dataset_root,
        splits=splits,
        nc=nc,
        names=names,
        out_path=patched,
    )
    return patched


# =============================================================================
# MAIN
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO11-OBB on a (possibly Seg) YOLO dataset.",
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to data.yaml. If Seg-only labels, auto-converted to OBB.",
    )
    parser.add_argument(
        "--model", type=str, default="yolo11n-obb.pt",
        help="Base model weights (default: yolo11n-obb.pt).",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override training epochs (default: auto-tuned).",
    )
    parser.add_argument(
        "--imgsz", type=int, default=None,
        help="Override image size (default: auto-tuned).",
    )
    parser.add_argument(
        "--batch", type=int, default=None,
        help="Override batch size (default: auto-tuned).",
    )
    parser.add_argument(
        "--patience", type=int, default=None,
        help="Override early-stopping patience (default: auto-tuned).",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Override dataloader workers (default: auto-tuned).",
    )
    parser.add_argument(
        "--device", type=str, default="0",
        help="CUDA device id, 'cpu', or '' for auto (default: '0').",
    )
    parser.add_argument(
        "--project", type=str, default="runs/obb",
        help="Project directory (default: runs/obb).",
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Run name (default: <dataset>_obb_<timestamp>).",
    )
    parser.add_argument(
        "--fliplr", type=float, default=0.0,
        help="Horizontal flip augmentation (default: 0.0 for nail dataset).",
    )
    parser.add_argument(
        "--degrees", type=float, default=180.0,
        help="Rotation augmentation in degrees (default: 180).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from latest checkpoint in <project>/<name>/.",
    )
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

    # Auto-convert to OBB labels if needed.
    print("=" * 70)
    print("TRAIN_OBB - YOLO11 Oriented Bounding Box")
    print("=" * 70)
    obb_data_yaml = ensure_obb_labels(data_yaml)
    print(f"Data yaml (OBB): {obb_data_yaml}")

    # Auto-split if val/ is missing (Ultralytics requires non-empty val/).
    _ensure_val_split(obb_data_yaml)

    # Auto-tune hyperparameters based on dataset size.
    dataset_size = count_train_images(obb_data_yaml)
    tuned = auto_tune(dataset_size)
    epochs = args.epochs if args.epochs is not None else tuned["epochs"]
    imgsz = args.imgsz if args.imgsz is not None else tuned["imgsz"]
    batch = args.batch if args.batch is not None else tuned["batch"]
    patience = args.patience if args.patience is not None else tuned["patience"]
    workers = args.workers if args.workers is not None else tuned["workers"]

    # Default name = <dataset>_obb_<timestamp>
    name = args.name
    if not name:
        from datetime import datetime
        stem = data_yaml.parent.name.replace(" ", "_").replace(".", "_")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{stem}_obb_{stamp}"

    print()
    print(f"Base model : {args.model}")
    print(f"Data yaml  : {obb_data_yaml}")
    print(f"Dataset sz : {dataset_size} (auto-tuned)" if dataset_size > 0 else "Dataset sz : unknown")
    print(f"Epochs     : {epochs}")
    print(f"Image size : {imgsz}")
    print(f"Batch size : {batch}")
    print(f"Patience   : {patience}")
    print(f"Workers    : {workers}")
    print(f"Device     : {args.device or 'auto'}")
    print(f"Project    : {args.project}")
    print(f"Name       : {name}")
    print(f"Augment    : degrees={args.degrees}, fliplr={args.fliplr}")
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
        # Heavy augmentation to cover all nail orientations.
        degrees=args.degrees,
        translate=0.2,
        scale=0.9,
        shear=10.0,
        perspective=0.001,
        fliplr=args.fliplr,
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
    print("Next step: run ``export_all.py --obb`` to produce ONNX + TFLite.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
