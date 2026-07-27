"""
================================================================================
CONVERT_SEG_TO_OBB.PY - Convert YOLO-Seg labels to YOLO-OBB labels
================================================================================
Convert polygon labels (YOLO-Seg format) into oriented-bounding-box labels
(YOLO-OBB format) so they can be used to train a YOLO11-OBB model.

The conversion is **dataset-agnostic**: pass any ``data.yaml`` that points at a
Roboflow-style or YOLO-style dataset, and the script will:

    1. Discover all image/label pairs under ``train/`` (and ``valid/`` if any).
    2. For each polygon, fit a ``cv2.minAreaRect`` and sort the 4 corners
       according to the **Ultralytics OBB convention** (verified empirically
       for ultralytics 8.4.x).
    3. Write side-by-side labels to ``labels_obb/`` next to the original
       ``labels/`` directory. The original labels are never modified.
    4. Emit a patched ``data_obb.yaml`` next to the source ``data.yaml`` so
       Ultralytics can train directly from the new labels without touching the
       original dataset.

**OBB corner convention used by Ultralytics YOLO11-OBB:**

The model exports 4 corners in the order ``[BR, TR, TL, BL]`` (verified by
calling ``ultralytics.utils.ops.xywhr2xyxyxyxy`` at ``angle=0``). The
``xywhr`` layout is ``[cx, cy, w, h, angle_rad]`` where ``angle`` is the CCW
rotation of the rectangle in radians, measured from the +x axis. We sort
the 4 corners of ``cv2.minAreaRect`` into the same ``[BR, TR, TL, BL]`` order
so the model learns the right angle convention.

Usage:
    # Smoke test dataset (139 images, 6-class - 1 garbage class)
    python convert_seg_to_obb.py --data ../Nail_Detection_ThanhDT.v1i.yolov11/data.yaml

    # Production dataset (10,501 images, 5-class)
    python convert_seg_to_obb.py --data ../nail-segmentation.v1i.yolov11_10501/data.yaml

    # Use an existing data.yaml that already points at labels_obb/ (skip convert)
    python convert_seg_to_obb.py --data ../Nail_Detection_ThanhDT.v1i.yolov11/data.yaml --skip-if-exists
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

try:
    import yaml  # PyYAML
except ImportError:  # pragma: no cover
    yaml = None

import cv2
import numpy as np


WORKSPACE_ROOT: Path = Path(__file__).resolve().parent


# =============================================================================
# DATA.YAML RESOLUTION
# =============================================================================
def load_data_yaml(data_yaml: Path) -> dict:
    """Read a data.yaml file. Returns a dict (empty on failure)."""
    if yaml is None:
        raise RuntimeError("PyYAML is required - install with: pip install PyYAML")
    try:
        with open(data_yaml, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        raise RuntimeError(f"Failed to read data.yaml {data_yaml}: {e}") from e


def resolve_dataset_paths(data_yaml: Path, data: dict) -> Tuple[Path, Path, Path]:
    """Return absolute paths for (dataset_root, splits_dir, train_images_dir).

    The Ultralytics data.yaml convention is:
        path: <absolute dataset root>
        train: <relative or absolute path to train images>
        val:   <relative or absolute path to val images>

    ``path`` may be omitted (then the dataset root is the data.yaml's parent).

    ``train`` may be a path RELATIVE TO data.yaml (typical for Roboflow exports
    that hard-code ``../train/images``) OR relative to ``path``. We try both,
    plus a fallback to ``dataset_root/train/images`` because Roboflow exports
    ship with the data.yaml pointing one level above the real files.
    """
    dataset_root = data.get("path")
    if dataset_root:
        dataset_root = Path(dataset_root).expanduser().resolve()
    else:
        dataset_root = data_yaml.parent.resolve()

    train_rel = data.get("train", "train/images")
    train_path = _resolve_split_path(data_yaml, dataset_root, train_rel)

    return dataset_root, train_path.parent, train_path


def _resolve_split_path(data_yaml: Path, dataset_root: Path, raw_path: str) -> Path:
    """Resolve a ``train``/``val`` path from data.yaml to an absolute Path.

    Mirrors Ultralytics' behaviour: try relative-to-data.yaml first
    (Roboflow convention), then relative-to-dataset-root. If neither exists,
    fall back to ``dataset_root/<stem>/images`` which is the actual on-disk
    layout (Roboflow exports typically ship with the data.yaml pointing one
    level above the real files).
    """
    p = Path(raw_path)
    if p.is_absolute():
        return p.resolve()

    cand1 = (data_yaml.parent / p).resolve()
    if cand1.exists():
        return cand1
    cand2 = (dataset_root / p).resolve()
    if cand2.exists():
        return cand2

    # Fallback: dataset_root/train/images (or val/images).
    # Take the last segment of raw_path (e.g. "train/images" or "val/images")
    # and rebuild under dataset_root.
    parts = Path(raw_path).parts
    if len(parts) >= 2 and parts[-1] == "images":
        candidate = dataset_root / parts[-2] / "images"
        if candidate.exists():
            return candidate.resolve()
    # Last resort: just return the relative-to-dataset-root candidate so the
    # caller can decide what to do (it'll fail gracefully).
    return cand2


# =============================================================================
# CORNER SORTING (BR, TR, TL, BL)
# =============================================================================
def sort_corners_ultralytics_obb(rotated_rect) -> np.ndarray:
    """Sort 4 corners from cv2.minAreaRect into Ultralytics OBB order.

    Args:
        rotated_rect: tuple ``((cx, cy), (w, h), angle_deg)`` as returned by
            ``cv2.minAreaRect``.

    Returns:
        np.ndarray of shape ``(4, 2)`` with corner order ``[BR, TR, TL, BL]``.
    """
    box = cv2.boxPoints(rotated_rect)  # (4, 2) float32
    box = np.asarray(box, dtype=np.float32)

    # Find the corner with the largest (x + y) sum -> Bottom-Right
    sums = box.sum(axis=1)
    br_idx = int(np.argmax(sums))
    br = box[br_idx]

    # Top-Left is opposite: smallest (x + y)
    tl_idx = int(np.argmin(sums))
    tl = box[tl_idx]

    # Two remaining corners are TR and BL.
    # TR has larger x, BL has smaller x. (Ties: use y.)
    remaining = [i for i in range(4) if i not in (br_idx, tl_idx)]
    a, b = box[remaining[0]], box[remaining[1]]
    if a[0] >= b[0]:
        tr, bl = a, b
    else:
        tr, bl = b, a

    return np.stack([br, tr, tl, bl], axis=0)


# =============================================================================
# SINGLE-POLYGON CONVERSION
# =============================================================================
def polygon_to_obb_line(cls: int, polygon: np.ndarray, img_w: int, img_h: int) -> Optional[str]:
    """Convert a single YOLO-Seg polygon (normalized 0-1) to one YOLO-OBB line.

    Returns the line in ``cls x1 y1 x2 y2 x3 y3 x4 y4`` format (normalized),
    or ``None`` if the polygon is degenerate.
    """
    if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
        return None

    # Scale polygon up for minAreaRect so the rotation is well-conditioned.
    poly_px = polygon.copy()
    poly_px[:, 0] *= img_w
    poly_px[:, 1] *= img_h
    poly_px = poly_px.astype(np.float32)

    if poly_px.shape[0] < 3:
        return None

    # minAreaRect requires a float32 ndarray of shape (N, 1, 2) or (N, 2).
    rect = cv2.minAreaRect(poly_px)
    corners_px = sort_corners_ultralytics_obb(rect)  # (4, 2) in pixel space

    # Normalize back to 0-1 using actual image dimensions (NOT input size).
    corners_norm = corners_px.copy()
    corners_norm[:, 0] /= float(img_w)
    corners_norm[:, 1] /= float(img_h)

    # Clamp tiny numerical errors.
    corners_norm = np.clip(corners_norm, 0.0, 1.0)

    # Build the line: "cls x1 y1 x2 y2 x3 y3 x4 y4"
    coords = " ".join(f"{c:.6f}" for c in corners_norm.flatten())
    return f"{cls} {coords}"


def read_image_size(image_path: Path) -> Optional[Tuple[int, int]]:
    """Read image dimensions (width, height). Returns None on failure."""
    try:
        # Use cv2.IMREAD_UNCHANGED so we can detect single-channel masks.
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        h, w = img.shape[:2]
        return w, h
    except Exception:
        return None


# =============================================================================
# BATCH CONVERSION
# =============================================================================
def convert_split(
    images_dir: Path,
    labels_dir: Path,
    out_labels_dir: Path,
    img_exts: Iterable[str] = ("jpg", "jpeg", "png", "bmp"),
) -> Tuple[int, int, int]:
    """Convert all ``labels_dir/*.txt`` files under ``images_dir``.

    Returns ``(converted_files, skipped_files, total_lines_written)``.
    """
    if not images_dir.exists():
        print(f"  [skip] images dir not found: {images_dir}")
        return 0, 0, 0
    if not labels_dir.exists():
        print(f"  [skip] labels dir not found: {labels_dir}")
        return 0, 0, 0

    out_labels_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0
    total_lines = 0

    label_files = sorted(labels_dir.glob("*.txt"))
    for lbl_path in label_files:
        stem = lbl_path.stem
        # Find matching image (any supported extension).
        img_path: Optional[Path] = None
        for ext in img_exts:
            candidate = images_dir / f"{stem}.{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            skipped += 1
            continue

        wh = read_image_size(img_path)
        if wh is None:
            skipped += 1
            continue
        img_w, img_h = wh

        out_lines: List[str] = []
        with open(lbl_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 7:  # need at least cls + 3 pts (x,y)
                    continue
                try:
                    cls = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                except ValueError:
                    continue
                if len(coords) % 2 != 0:
                    continue
                polygon = np.array(coords, dtype=np.float32).reshape(-1, 2)
                line_out = polygon_to_obb_line(cls, polygon, img_w, img_h)
                if line_out is not None:
                    out_lines.append(line_out)

        out_path = out_labels_dir / lbl_path.name
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + ("\n" if out_lines else ""))

        if out_lines:
            converted += 1
            total_lines += len(out_lines)
        else:
            # Still wrote an empty file - that means we kept the stem but
            # produced nothing; count as skipped so the user notices.
            skipped += 1

    return converted, skipped, total_lines


# =============================================================================
# PATCHED DATA.YAML
# =============================================================================
def write_patched_data_yaml(
    source_data_yaml: Path,
    dataset_root: Path,
    splits: List[Tuple[str, Path]],
    nc: int,
    names: List[str],
    out_path: Optional[Path] = None,
) -> Path:
    """Write a workspace-friendly data_obb.yaml that points at ``labels_obb/``.

    Args:
        source_data_yaml: original data.yaml
        dataset_root: absolute dataset root (== ``data['path']`` or fallback)
        splits: list of ``(key, image_dir)`` such as ``[("train", Path), ("val", Path)]``.
        nc: number of classes (preserved from the source).
        names: class names (preserved from the source).
        out_path: where to write. If None, write next to source_data_yaml.

    Returns:
        The path to the patched yaml.
    """
    if out_path is None:
        out_path = source_data_yaml.parent / "data_obb.yaml"

    lines = [
        "# Auto-generated by convert_seg_to_obb.py",
        f"# Source: {source_data_yaml.name}",
        "",
        f"path: {str(dataset_root)}",
    ]

    for key, img_dir in splits:
        try:
            rel = img_dir.relative_to(dataset_root).as_posix()
        except ValueError:
            # Image dir lives outside dataset_root (Roboflow quirk); use absolute.
            rel = str(img_dir)
        lines.append(f"{key}: {rel}")
        # Replace "images" segment in path with "labels_obb" for labels.
        labels_rel = rel.replace("/images", "/labels_obb", 1).replace(
            "\\images", "\\labels_obb", 1
        )
        if labels_rel == rel:
            # Fallback if the path doesn't contain /images.
            labels_rel = str(Path(rel).parent / "labels_obb")
        lines.append(
            f"# {key[:-len('images')] if key.endswith('images') else key}_labels: {labels_rel}"
        )

    lines.extend([
        "",
        f"nc: {nc}",
        "names:",
    ])
    for i, n in enumerate(names):
        lines.append(f"  {i}: {n}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# =============================================================================
# MAIN
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert YOLO-Seg polygon labels to YOLO-OBB oriented-bbox labels.",
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to dataset's data.yaml (any YOLO-Seg or YOLO-OBB dataset).",
    )
    parser.add_argument(
        "--skip-if-exists", action="store_true",
        help="If labels_obb/ already contains a non-empty label file, skip conversion for that split.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    source_yaml = Path(args.data).resolve()
    if not source_yaml.exists():
        print(f"ERROR: data.yaml not found: {source_yaml}")
        return 1

    try:
        data = load_data_yaml(source_yaml)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1

    nc = data.get("nc")
    names = data.get("names", [])
    if not isinstance(nc, int) or not isinstance(names, list):
        print(f"ERROR: data.yaml is missing 'nc' or 'names' fields: {source_yaml}")
        return 1

    dataset_root, splits_dir, train_images = resolve_dataset_paths(source_yaml, data)
    print(f"Dataset root : {dataset_root}")
    print(f"Train images : {train_images}")
    print(f"Classes      : nc={nc}, names={names}")

    # Process train + (optional) valid splits.
    splits = [("train", train_images)]
    val_images = data.get("val")
    if val_images:
        val_path = _resolve_split_path(source_yaml, dataset_root, val_images)
        if val_path.exists():
            splits.append(("val", val_path))

    print()
    print(f"Converting {len(splits)} split(s)...")
    grand_total_converted = 0
    grand_total_skipped = 0
    grand_total_lines = 0

    for split_key, img_dir in splits:
        labels_dir = img_dir.parent / "labels"
        out_labels_dir = img_dir.parent / "labels_obb"

        if args.skip_if_exists and any(out_labels_dir.glob("*.txt")):
            print(f"[{split_key}] labels_obb/ already has files - skipping (use --force to override).")
            continue

        print(f"[{split_key}] {img_dir} -> labels_obb/")
        converted, skipped, lines = convert_split(img_dir, labels_dir, out_labels_dir)
        print(f"  -> {converted} converted, {skipped} skipped, {lines} polygons")
        grand_total_converted += converted
        grand_total_skipped += skipped
        grand_total_lines += lines

    # Write patched data_obb.yaml next to the source yaml.
    patched = write_patched_data_yaml(
        source_yaml,
        dataset_root=dataset_root,
        splits=[(k, v) for k, v in splits],
        nc=nc,
        names=names,
    )
    print()
    print(f"Patched data.yaml: {patched}")

    # Summary
    print()
    print("=" * 70)
    print("CONVERSION SUMMARY")
    print("=" * 70)
    print(f"Files converted : {grand_total_converted}")
    print(f"Files skipped   : {grand_total_skipped}")
    print(f"OBB lines total : {grand_total_lines}")
    print(f"OBB label dir   : {splits_dir / 'labels_obb'}")
    print(f"Patched YAML    : {patched}")
    print()
    print("Next steps:")
    print(f"  python auto_split.py --data {patched}    # (only if you need a val split)")
    print(f"  python train_obb.py --data {patched}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
