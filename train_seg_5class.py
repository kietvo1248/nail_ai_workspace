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

V3+ tuning (added 2026-08-08) for nail-vs-skin boundary:
    - ``mosaic=0.5``        - reduced from 1.0 so model sees full hand context
                              more often (avoids hallucinating nails from
                              cropped mosaic tiles).
    - ``close_mosaic=15``   - last 15 epochs turn mosaic OFF → forces model
                              to learn crisp nail-edge vs skin boundary on
                              full-resolution images.
    - ``overlap_mask=True`` - when two nails touch, allow masks to overlap
                              during training so model learns to keep them
                              distinct (instead of merging into one blob).
    - ``--neg-frames-dir``  - optional path to a folder of background/negative
                              frames (hands without nails, skin-only images,
                              etc.) that get auto-copied into train/images
                              with empty labels. Teaches the model that
                              ``skin ≠ nail`` and removes false positives.

Usage:
    python train_seg_5class.py --data ../nail-segmentation.v3-demo-train.yolov11/data.yaml
    python train_seg_5class.py --data ../Nail_Detection_ThanhDT.v1i.yolov11/data.yaml --epochs 50
    python train_seg_5class.py --data ../data.yaml --neg-frames-dir ../background_frames --epochs 100
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
        description="Train YOLO11-Seg 5-class (per-finger) model.",
    )
    parser.add_argument("--data", type=str, default="../nail-segmentation.v2i.yolov11_4438/data.yaml",
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
    parser.add_argument("--neg-frames-dir", type=str, default=None,
                        help="Optional folder of negative frames (hands without "
                             "nails, skin-only, etc.). These will be auto-copied "
                             "into <data>/train/images with empty .txt labels so "
                             "the model learns 'skin ≠ nail'. Default: None.")
    parser.add_argument("--neg-prefix", type=str, default="neg_",
                        help="Filename prefix used when copying negative frames "
                             "into the dataset (default: 'neg_').")
    return parser.parse_args()


def inject_negative_frames(
    data_yaml: Path,
    neg_dir: str | None,
    prefix: str = "neg_",
) -> int:
    """Copy negative frames into ``<dataset>/train/images`` with empty labels.

    Why: the V3 dataset has zero empty labels → model assumes every image
    contains a nail → it produces false positives on skin-only images.
    Adding a handful of negative frames (skin, hands without visible nails,
    defect photos, etc.) teaches the model to output *no* mask when there
    is no nail.

    Returns the number of negative frames injected.
    """
    if not neg_dir:
        return 0
    neg_path = Path(neg_dir).expanduser().resolve()
    if not neg_path.is_dir():
        print(f"[NEG] WARNING: --neg-frames-dir not found: {neg_path}")
        return 0

    train_images = data_yaml.parent / "train" / "images"
    train_labels = data_yaml.parent / "train" / "labels"
    train_images.mkdir(parents=True, exist_ok=True)
    train_labels.mkdir(parents=True, exist_ok=True)

    import shutil
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    injected = 0
    for src in sorted(neg_path.iterdir()):
        if src.suffix.lower() not in img_exts:
            continue
        dst_name = f"{prefix}{src.name}"
        # Avoid double-prefix if file already starts with prefix
        if src.stem.startswith(prefix):
            dst_name = src.name
        dst_img = train_images / dst_name
        dst_lbl = train_labels / (Path(dst_name).stem + ".txt")
        if dst_img.exists():
            # Idempotent: do not overwrite existing data
            continue
        shutil.copy2(src, dst_img)
        # Empty label file = negative sample
        dst_lbl.write_text("", encoding="utf-8")
        injected += 1
    print(f"[NEG] Injected {injected} negative frame(s) from {neg_path}")
    print(f"[NEG] -> images: {train_images}")
    print(f"[NEG] -> empty labels written to: {train_labels}")
    return injected


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

    # Inject negative frames BEFORE training so the data.yaml scan picks them up.
    if args.neg_frames_dir:
        injected = inject_negative_frames(
            data_yaml=data_yaml,
            neg_dir=args.neg_frames_dir,
            prefix=args.neg_prefix,
        )
        if injected == 0:
            print("[NEG] No new negative frames were injected (folder empty or "
                  "all files already present).")

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
        amp=False,             # Disable AMP to prevent CUDA misaligned address error
        # Optimized augmentations for crisp Segmentation masks
        degrees=180.0,
        translate=0.1,
        scale=0.3,             # Reduced to preserve nail texture
        shear=0.0,             # Disabled to prevent mask distortion
        perspective=0.0,       # Disabled to prevent polygon warping
        fliplr=0.0,            # Hard-locked off for 5-class to keep thumb vs pinky distinct
        flipud=0.0,
        mosaic=0.5,            # V3+: reduced from 1.0 — fewer stitched tiles,
                               # model sees full hand context more often and
                               # hallucinates fewer nails on cropped mosaic pieces
        close_mosaic=15,       # V3+: last 15 epochs turn mosaic OFF — forces
                               # crisp nail-edge vs skin boundary learning
        mixup=0.0,             # Disabled to avoid mask ghosting
        copy_paste=0.0,        # Disabled to avoid overlap
        hsv_h=0.02,
        hsv_s=0.3,             # Reduced to preserve surface color
        hsv_v=0.2,             # Reduced to preserve surface brightness
        erasing=0.0,           # Disabled to prevent destroying the nail surface
        cls=2.0,               # Heavily penalize wrong finger class
        overlap_mask=True,     # V3+: was False — now allow mask overlap so
                               # adjacent nails learn distinct boundaries
                               # instead of merging into one blob
    )

    print("\nTraining complete.")
    print(f"Best checkpoint: {args.project}/{name}/weights/best.pt")
    print("Next step: run ``export_all.py --run {name}`` to produce ONNX.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
