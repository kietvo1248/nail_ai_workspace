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
    1. Run create_2class_dataset.py to clone the dataset and generate 2-class labels
    2. Run this script to train the 2-class model on the new dataset

Augmentation choices (tuned for nail segmentation):
    - degrees=180.0   (on)  - learn nail orientation at any angle
    - fliplr=0.0     (off)  - do not flip (would swap thumb/pinky)
    - copy_paste=0.3         - increase data variance
    - shear=10.0             - learn nail bed boundary deformation
    - cls=2.0                - heavier loss weight on classification

Usage:
    # First, generate the 2-class dataset:
    python create_2class_dataset.py \
        --input "../nail-segmentation.v2i.yolov11" \
        --output "../nail-segmentation.v2i.yolov11_2class" \
        --bed-ratio 0.75

    # Then train on the new dataset:
    python train_seg_2class.py \
        --data ../nail-segmentation.v2i.yolov11_2class/data.yaml \
        --epochs 300
"""
from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

# Make CUDA errors synchronous so the real assert message appears in the right place
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

WORKSPACE_ROOT: Path = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE_ROOT))


# =============================================================================
# MONKEY-PATCH: Fix TWO bugs in ultralytics 8.4.x tal.py (TaskAlignedAssigner)
#
# Bug 1 — get_box_metrics (line ~193):
#   pd_scores[batch_ind, :, gt_labels.squeeze(-1).long()]
#   → crashes when any gt_label >= nc (no upper clamp)
#
# Bug 2 — get_targets (line ~277):
#   target_labels.clamp_(0)   ← only clamps min, not max
#   target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
#   → crashes when any target_label >= nc
#
# Root cause: mosaic + copy_paste augmentation in ultralytics 8.4.x can
# produce GT class IDs outside [0, nc-1] for segmentation tasks.
# Both patches add .clamp(0, nc-1) to prevent CUDA index out-of-bounds.
# =============================================================================
def _patch_tal():
    try:
        import torch
        import ultralytics.utils.tal as tal_module

        # ---- Patch 1: get_box_metrics ----------------------------------------
        def _safe_get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
            """Patched: clamps gt_labels to [0, nc-1] before indexing pd_scores."""
            na = pd_bboxes.shape[-2]
            mask_gt = mask_gt.bool()
            overlaps = torch.zeros(
                [self.bs, self.n_max_boxes, na],
                dtype=pd_bboxes.dtype, device=pd_bboxes.device
            )
            bbox_scores = torch.zeros(
                [self.bs, self.n_max_boxes, na],
                dtype=pd_scores.dtype, device=pd_scores.device
            )
            batch_ind = torch.arange(self.bs, device=pd_scores.device)[:, None]
            nc = pd_scores.shape[-1]
            gt_cls_idx = gt_labels.squeeze(-1).long().clamp(0, nc - 1)  # PATCH
            bbox_scores[mask_gt] = pd_scores[batch_ind, :, gt_cls_idx][mask_gt]
            pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
            gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
            overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)
            align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
            return align_metric, overlaps

        tal_module.TaskAlignedAssigner.get_box_metrics = _safe_get_box_metrics

        # ---- Patch 2: get_targets --------------------------------------------
        def _safe_get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
            """Patched: clamps target_labels to [0, nc-1] before scatter_.
            Original code: target_labels.clamp_(0)  ← missing max bound!
            """
            batch_ind = torch.arange(
                end=self.bs, dtype=torch.int64, device=gt_labels.device
            )[..., None]
            target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
            target_labels = gt_labels.long().flatten()[target_gt_idx]
            target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]
            nc = self.num_classes
            target_labels.clamp_(0, nc - 1)  # PATCH: was clamp_(0) — missing max!
            target_scores = torch.zeros(
                (target_labels.shape[0], target_labels.shape[1], nc),
                dtype=torch.int8,
                device=target_labels.device,
            )
            target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
            target_scores = target_scores * (fg_mask[:, :, None] > 0)
            return target_labels, target_bboxes, target_scores

        tal_module.TaskAlignedAssigner.get_targets = _safe_get_targets

        print("[PATCH] ultralytics tal.py: get_box_metrics + get_targets patched (clamp to [0, nc-1])")
    except Exception as e:
        print(f"[PATCH] WARNING: Could not patch tal.py: {e}")


_patch_tal()
# =============================================================================


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
    parser.add_argument("--workers", type=int, default=4,
                        help="Dataloader workers (default: 4; keep <=4 on Windows).")
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
        copy_paste=0.3,        # safe after tal.py patch
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
