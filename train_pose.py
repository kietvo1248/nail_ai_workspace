"""
================================================================================
TRAIN_POSE.PY - YOLOv11-Pose Training Script for Nail AR
================================================================================
Script huấn luyện mô hình YOLOv11-Pose với 4 keypoints mỗi móng:
    - Top    (đỉnh móng)
    - Bottom (gốc móng)
    - Left
    - Right

Model output (sau khi export ONNX):
    output0: [1, 4 + 5 + 4*3 = 21, N]
        - 4 bbox: cx, cy, w, h
        - 5 scores: obj, cls (1 cls + 4 dummy cho classes)
        - 4 keypoints * 3 (x, y, visibility)

Yêu cầu dataset:
    - Có dữ liệu keypoints (4 điểm / móng) đã được annotate
    - data.yaml có kpt_shape: [4, 3]

Sử dụng:
    python train_pose.py
    python train_pose.py --epochs 100 --imgsz 416
================================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


# =============================================================================
# ANSI COLORS for Terminal
# =============================================================================
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(text: str) -> None:
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}{Colors.END}\n")


def print_success(text: str) -> None:
    print(f"{Colors.GREEN}[OK] {text}{Colors.END}")


def print_error(text: str) -> None:
    print(f"{Colors.RED}[ERROR] {text}{Colors.END}")


def print_warning(text: str) -> None:
    print(f"{Colors.YELLOW}[WARNING] {text}{Colors.END}")


def print_info(text: str) -> None:
    print(f"{Colors.BLUE}[INFO] {text}{Colors.END}")


# =============================================================================
# CONSTANTS
# =============================================================================
NUM_KEYPOINTS = 21    # 21 hand keypoints (MediaPipe skeleton)
KEYPOINT_DIM = 3      # x, y, visibility
# Dùng .pt để dùng pretrained backbone, giúp mô hình hội tụ nhanh trên dataset nhỏ.
DEFAULT_POSE_WEIGHTS = "yolo11n-pose.pt"
NUM_CLASSES = 1       # hand
CLASS_NAME = "hand"


# =============================================================================
# DATA.YAML GENERATION FOR POSE
# =============================================================================
def create_pose_data_yaml(dataset_path: Path) -> Path:
    """Create a Pose data.yaml beside the dataset folder.

    The Pose training format requires ``kpt_shape``. This file is a sibling
    of the dataset so the original dataset is never modified.
    """
    dataset_path = Path(dataset_path).resolve()
    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        print_error(f"Dataset data.yaml not found: {data_yaml}")
        sys.exit(1)

    raw = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) if yaml else {}
    if not isinstance(raw, dict):
        raw = {}

    pose_cfg = dict(raw)
    pose_cfg.pop("kpt_shape", None)
    pose_cfg.pop("flip_idx", None)
    pose_cfg["path"] = str(dataset_path)
    pose_cfg["kpt_shape"] = [NUM_KEYPOINTS, KEYPOINT_DIM]

    # Use train/valid Roboflow convention when present, else fall back to yolo.
    if (dataset_path / "train" / "images").exists():
        pose_cfg["train"] = "train/images"
        pose_cfg["val"] = "valid/images" if (dataset_path / "valid" / "images").exists() else "train/images"
    elif (dataset_path / "images" / "train").exists():
        pose_cfg["train"] = "images/train"
        pose_cfg["val"] = "images/val" if (dataset_path / "images" / "val").exists() else "images/train"
    else:
        print_warning("Could not detect train/val folders; leaving existing paths in data.yaml.")

    pose_cfg["nc"] = NUM_CLASSES
    pose_cfg["names"] = {0: CLASS_NAME}

    out_path = dataset_path.parent / f"data_pose_{dataset_path.name.replace(' ', '_').replace('.', '_')}.yaml"
    if yaml is None:
        print_error("PyYAML not installed. Run: pip install PyYAML")
        sys.exit(1)
    out_path.write_text(yaml.safe_dump(pose_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print_success(f"Pose data.yaml created: {out_path}")
    return out_path


# =============================================================================
# GPU DETECTION
# =============================================================================
def scan_gpus() -> str:
    """Detect available GPUs and return the device string ("0", "1", ...)."""
    try:
        import torch
    except ImportError:
        print_warning("torch not installed - falling back to CPU")
        return "cpu"

    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}")
    print(f"  GPU DETECTION (Pose)")
    print(f"{'='*60}{Colors.END}\n")

    if not torch.cuda.is_available():
        print_error("No GPU detected - training will be very slow on CPU.")
        return "cpu"

    count = torch.cuda.device_count()
    print(f"  {Colors.GREEN}Found {count} GPU(s):{Colors.END}\n")
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        print(f"  {Colors.CYAN}[{i}]{Colors.END} {props.name}  "
              f"({props.total_memory / (1024**3):.1f} GB)")

    if count == 1:
        return "0"

    raw = input(f"\nSelect GPU [0-{count-1}], default 0: ").strip()
    if not raw:
        return "0"
    if raw.lower() == "c":
        return "cpu"
    try:
        idx = int(raw)
        if 0 <= idx < count:
            return str(idx)
    except ValueError:
        pass
    print_warning("Invalid selection, using GPU 0.")
    return "0"


# =============================================================================
# DATASET PATH
# =============================================================================
def select_dataset() -> Path:
    """Pick a dataset that already has a data.yaml."""
    workspace_root = Path(__file__).parent.resolve()
    parent_root = workspace_root.parent

    candidates = []
    for item in parent_root.iterdir():
        if item.is_dir() and (item / "data.yaml").exists():
            candidates.append(item)

    if not candidates:
        print_warning("No datasets with data.yaml found in workspace root.")
        raw = input("Enter dataset path: ").strip().strip('"')
        ds = Path(raw).resolve()
        if not (ds / "data.yaml").exists():
            print_error(f"data.yaml missing in: {ds}")
            sys.exit(1)
        return ds

    if len(candidates) == 1:
        print_success(f"Using dataset: {candidates[0]}")
        return candidates[0]

    print(f"\n{Colors.YELLOW}Available datasets:{Colors.END}")
    for i, d in enumerate(candidates, 1):
        print(f"  {Colors.CYAN}{i}.{Colors.END} {d.name}")
    while True:
        raw = input(f"\nSelect dataset (1-{len(candidates)}): ").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(candidates):
                return candidates[idx]
        except ValueError:
            pass
        print_warning("Invalid selection.")


# =============================================================================
# TRAINING
# =============================================================================
def train_pose(
    dataset: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    weights: str,
    project_name: str,
) -> None:
    """Run YOLO11-Pose training."""
    try:
        from ultralytics import YOLO
    except ImportError as e:
        print_error(f"ultralytics not installed: {e}")
        print_info("Install with: pip install ultralytics")
        sys.exit(1)

    pose_yaml = create_pose_data_yaml(dataset)
    workspace_root = Path(__file__).parent.resolve()
    runs_dir = workspace_root / "runs"
    runs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{project_name}_{timestamp}"

    print_header("POSE TRAINING CONFIG")
    print(f"  Dataset      : {dataset}")
    print(f"  Pose yaml    : {pose_yaml}")
    print(f"  Weights      : {weights}")
    print(f"  Epochs       : {epochs}")
    print(f"  Image size   : {imgsz}")
    print(f"  Batch size   : {batch}")
    print(f"  Device       : {device}")
    print(f"  Run name     : {run_name}")
    print(f"  Keypoints    : {NUM_KEYPOINTS} ({KEYPOINT_DIM} dims each)")
    print()

    weights_path = Path(weights)
    if not weights_path.exists():
        print_warning(f"Weights not found at {weights_path}; Ultralytics will download.")

    model = YOLO(str(weights_path))
    model.train(
        data=str(pose_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(runs_dir),
        name=run_name,
        device=device,
        # ── OPTIMIZER: Sử dụng default cho fine-tuning ──────────────
        optimizer="auto",
        # ── LOSS WEIGHTS ──────────────────────
        pose=15.0,         # Siết chặt hơn (default 12) để ép mô hình bắt điểm thật sát
        kobj=2.0,          # Ép học độ tin cậy của 21 điểm
        box=5.0,           # Giảm bớt box để dồn toàn bộ 'sự chú ý' vào keypoints
        # ── WARMUP ─────────────────────
        warmup_epochs=3.0,
        warmup_bias_lr=0.1,
        # ── AUGMENTATION: Khôi phục lại độ quay và lật theo yêu cầu thực tế
        degrees=180.0,     # Bàn tay có thể xoay đủ mọi hướng 360 độ (±180)
        translate=0.05,    # Giảm dịch chuyển để bàn tay luôn ở giữa khung hình
        scale=0.2,         # Giảm phóng to/thu nhỏ quá đà để giữ nguyên form tay
        shear=3.0,
        perspective=0.0,
        flipud=0.5,        # Bật lại lộn ngược vì camera có thể chĩa từ trên xuống
        fliplr=0.5,
        mosaic=0.5,        # Giảm mosaic (trước=1.0) để ảnh đơn chiếm 50% để học chính xác
        mixup=0.0,
        copy_paste=0.0,
        # ── GENERAL ──────────────────────────────────────────────────────────
        patience=80,       # Kiên nhẫn hơn vì keypoint cần nhiều epoch để hội tụ
        amp=False,
        workers=2,
        save=True,
        save_period=10,
        verbose=True,
        close_mosaic=20,   # Tắt mosaic ở 20 epoch cuối để fine-tune chính xác
    )

    print_header("POSE TRAINING COMPLETE")
    run_dir = runs_dir / run_name
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    if best.exists():
        print_success(f"Best weights: {best} ({best.stat().st_size / (1024**2):.1f} MB)")
    if last.exists():
        print_info(f"Last weights: {last}")
    print_info("Export to ONNX with: python export_all.py --pose")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO11-Pose for nail AR")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=416, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detect if None)")
    parser.add_argument("--weights", type=str, default=DEFAULT_POSE_WEIGHTS,
                        help="Pretrained pose weights")
    parser.add_argument("--data", type=str, default=None, help="Path to dataset directory")
    parser.add_argument("--name", type=str, default="nail_pose",
                        help="Run name prefix")
    return parser.parse_args()


def main() -> int:
    print_header("YOLOV11-POSE NAIL AR - TRAINING")
    args = parse_args()

    dataset = Path(args.data).resolve() if args.data else select_dataset()
    device = args.device if args.device is not None else scan_gpus()

    train_pose(
        dataset=dataset,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        weights=args.weights,
        project_name=args.name,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
