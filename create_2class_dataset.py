"""
================================================================================
CREATE_2CLASS_DATASET.PY
================================================================================
Converts a 5-class YOLO Seg dataset into a 2-class dataset (nail_bed + full_nail).

YOLO strictly requires the label folder to be named "labels" alongside "images".
Therefore, we cannot just output "labels_2class" in the same dataset folder.
This script clones the dataset structure, copies the images, and generates the
new 2-class labels into the new dataset.

Usage:
    python create_2class_dataset.py \
        --input "../Nail_Detection_ThanhDT.v1i.yolov11" \
        --output "../Nail_Detection_ThanhDT.v1i.yolov11_2class" \
        --bed-ratio 0.75
"""
from __future__ import annotations

import argparse
import sys
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np


# ============================================================================
# CORE ALGORITHMS (From convert_polygon_to_nailbed.py)
# ============================================================================

def parse_yolo_seg_label(line: str) -> Tuple[int, List[Tuple[float, float]]]:
    parts = line.strip().split()
    if len(parts) < 7:
        raise ValueError(f"Invalid label line (need >= 3 points): {line!r}")
    class_id = int(parts[0])
    coords = [float(x) for x in parts[1:]]
    if len(coords) % 2 != 0:
        raise ValueError(f"Invalid coordinate count (must be even): {len(coords)}")
    polygon = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
    return class_id, polygon


def polygon_to_coords(polygon: List[Tuple[float, float]]) -> np.ndarray:
    return np.array(polygon, dtype=np.float64)


def get_direction_from_polygon_pca(polygon: List[Tuple[float, float]]) -> Tuple[float, float]:
    pts = polygon_to_coords(polygon)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    pc1 = eigenvectors[:, np.argmax(eigenvalues)]
    
    projections = centered @ pc1
    tip_idx = np.argmax(projections)
    base_idx = np.argmin(projections)
    tip = pts[tip_idx]
    base = pts[base_idx]
    
    direction = tip - base
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return (0.0, -1.0)
    direction = direction / norm
    return (float(direction[0]), float(direction[1]))


def find_cutoff_edge_intersections(
    pts: np.ndarray, direction: Tuple[float, float], base_proj: float, cutoff_proj: float
) -> List[Tuple[float, float]]:
    intersections = []
    n = len(pts)
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        proj1 = p1[0] * direction[0] + p1[1] * direction[1]
        proj2 = p2[0] * direction[0] + p2[1] * direction[1]
        
        if (proj1 <= cutoff_proj) != (proj2 <= cutoff_proj):
            t = (cutoff_proj - proj1) / (proj2 - proj1 + 1e-9)
            t = max(0.0, min(1.0, t))
            ix = p1[0] + t * (p2[0] - p1[0])
            iy = p1[1] + t * (p2[1] - p1[1])
            intersections.append((float(ix), float(iy)))
    return intersections


def cut_polygon_at_ratio(
    polygon: List[Tuple[float, float]], direction: Tuple[float, float], bed_ratio: float = 0.75
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    if bed_ratio <= 0.0 or bed_ratio >= 1.0:
        raise ValueError(f"bed_ratio must be in (0.0, 1.0), got {bed_ratio}")
    pts = polygon_to_coords(polygon)
    dx, dy = direction
    projections = (pts[:, 0] * dx + pts[:, 1] * dy)
    sorted_indices = np.argsort(projections)
    base_proj = projections[sorted_indices[0]]
    tip_proj = projections[sorted_indices[-1]]
    total_length = tip_proj - base_proj
    if total_length < 1e-9:
        return polygon, polygon
        
    cutoff_proj = base_proj + total_length * bed_ratio
    nail_bed_pts = []
    for idx in sorted_indices:
        proj = projections[idx]
        if proj <= cutoff_proj + 1e-9:
            nail_bed_pts.append((float(pts[idx, 0]), float(pts[idx, 1])))
            
    cutoff_pts = find_cutoff_edge_intersections(pts, direction, base_proj, cutoff_proj)
    if len(cutoff_pts) >= 2:
        perp_dir = np.array([-dy, dx])
        cutoff_pts = sorted(cutoff_pts, key=lambda p: float(p[0]) * perp_dir[0] + float(p[1]) * perp_dir[1])
        nail_bed_pts.extend(cutoff_pts[:2])
        
    if len(nail_bed_pts) < 3:
        mid = len(sorted_indices) // 2
        nail_bed_pts = [tuple(pts[idx]) for idx in sorted_indices[:mid + 1]]
        
    return nail_bed_pts, polygon


def coords_to_yolo_line(class_id: int, polygon: List[Tuple[float, float]]) -> str:
    coords = []
    for x, y in polygon:
        coords.append(f"{x:.10f}")
        coords.append(f"{y:.10f}")
    return f"{class_id} {' '.join(coords)}"


# ============================================================================
# DATASET CLONING & CONVERSION
# ============================================================================

def process_split(input_dir: Path, output_dir: Path, split: str, bed_ratio: float):
    print(f"\nProcessing split: {split}")
    in_split = input_dir / split
    out_split = output_dir / split
    if not in_split.exists():
        print(f"  WARNING: Split {split} not found at {in_split}")
        return

    # Copy images
    in_images = in_split / "images"
    out_images = out_split / "images"
    if in_images.exists():
        if out_images.exists():
            shutil.rmtree(out_images)
        shutil.copytree(in_images, out_images)
        print(f"  Copied {len(list(in_images.glob('*.jpg')))} images.")
    else:
        print(f"  WARNING: No images found at {in_images}")

    # Process labels
    in_labels = in_split / "labels"
    out_labels = out_split / "labels"
    out_labels.mkdir(parents=True, exist_ok=True)
    
    label_files = sorted(in_labels.glob("*.txt"))
    if not label_files:
        print(f"  WARNING: No labels found in {in_labels}")
        return

    processed = 0
    errors = 0
    for label_path in label_files:
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            output_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    class_id, polygon = parse_yolo_seg_label(line)
                    direction = get_direction_from_polygon_pca(polygon)
                    nail_bed, full_nail = cut_polygon_at_ratio(polygon, direction, bed_ratio)
                    output_lines.append(coords_to_yolo_line(0, nail_bed))
                    output_lines.append(coords_to_yolo_line(1, full_nail))
                except Exception as e:
                    print(f"    ERROR in {label_path.name}: {e}")
                    errors += 1
                    
            with open(out_labels / label_path.name, "w", encoding="utf-8") as f:
                f.write("\n".join(output_lines) + "\n")
            processed += 1
        except Exception as e:
            print(f"    ERROR reading {label_path.name}: {e}")
            errors += 1
            
    print(f"  Generated {processed} 2-class label files. Errors: {errors}")


def create_data_yaml(output_dir: Path):
    yaml_content = f"""path: {output_dir.absolute().as_posix()}
train: train/images
val: valid/images

nc: 2
names: ['nail_bed', 'full_nail']
"""
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"\nGenerated data.yaml at {yaml_path}")
    return yaml_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create 2-class YOLO dataset from 5-class dataset.")
    parser.add_argument("--input", required=True, help="Original dataset directory (containing train/, valid/)")
    parser.add_argument("--output", required=True, help="New 2-class dataset directory")
    parser.add_argument("--bed-ratio", type=float, default=0.75, help="Nail bed ratio (default: 0.75)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    
    if not input_dir.exists():
        print(f"ERROR: Input dataset not found: {input_dir}")
        return 1
        
    print("=" * 70)
    print("CREATE 2-CLASS DATASET")
    print("=" * 70)
    print(f"Input  : {input_dir}")
    print(f"Output : {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ["train", "valid", "test"]:
        process_split(input_dir, output_dir, split, args.bed_ratio)
        
    yaml_path = create_data_yaml(output_dir)
    
    print("\nDataset generation complete!")
    print("\nTo train the model, run:")
    print(f"python train_seg_2class.py --data \"{yaml_path}\" --epochs 100")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
