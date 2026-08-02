"""
================================================================================
CONVERT_POLYGON_TO_NAILBED.PY
================================================================================
Convert 5-class per-finger labels → 2-class (nail_bed + full_nail) labels.

This script auto-generates nail_bed polygons from the original nail polygons
without requiring manual re-annotation. Each original polygon is split into:
    - class 0: nail_bed (portion near cuticle, adjustable ratio)
    - class 1: full_nail (original polygon)

The direction vector (base→tip) is automatically inferred from polygon geometry
using PCA (Principal Component Analysis).

Usage:
    python convert_polygon_to_nailbed.py \
        --labels "../Nail_Detection_ThanhDT.v1i.yolov11/train/labels" \
        --output "../Nail_Detection_ThanhDT.v1i.yolov11/train/labels_2class" \
        --bed-ratio 0.75

    # Process both train and valid:
    python convert_polygon_to_nailbed.py \
        --labels "../Nail_Detection_ThanhDT.v1i.yolov11/train/labels" \
        --output "../Nail_Detection_ThanhDT.v1i.yolov11/train/labels_2class" \
        --bed-ratio 0.75 \
        --valid "../Nail_Detection_ThanhDT.v1i.yolov11/valid/labels" \
        --valid-output "../Nail_Detection_ThanhDT.v1i.yolov11/valid/labels_2class"

Output format (YOLO Seg):
    class_id x1 y1 x2 y2 ... xN yN
    0 0.258 0.604 0.270 0.610 ...   # nail_bed polygon
    1 0.258 0.604 0.270 0.610 ...   # full_nail polygon (same vertices)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


# ============================================================================
# CORE ALGORITHMS
# ============================================================================

def parse_yolo_seg_label(line: str) -> Tuple[int, List[Tuple[float, float]]]:
    """Parse a single line from YOLO segmentation label.

    Args:
        line: e.g. "0 0.258 0.604 0.270 0.610 ..."

    Returns:
        (class_id, list of (x, y) normalized coordinates)
    """
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
    """Convert polygon to numpy array (N, 2)."""
    return np.array(polygon, dtype=np.float64)


def get_direction_from_polygon_pca(
    polygon: List[Tuple[float, float]],
) -> Tuple[float, float]:
    """Infer direction vector (base→tip) from polygon using PCA.

    The primary axis (PC1) from PCA represents the long axis of the nail.
    We determine which end is the tip vs base by checking which has smaller Y
    (assuming standard image coordinates where Y increases downward).

    Args:
        polygon: List of (x, y) normalized coordinates in CCW order.

    Returns:
        Normalized direction vector (dx, dy) from base to tip.
    """
    pts = polygon_to_coords(polygon)

    # Center the points
    centroid = pts.mean(axis=0)

    # PCA to find principal axis
    centered = pts - centroid
    cov = centered.T @ centered  # 2x2 covariance
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # PC1 is the eigenvector with the largest eigenvalue
    pc1 = eigenvectors[:, np.argmax(eigenvalues)]

    # Determine tip direction: smaller Y = farther from wrist (tip)
    # Project all points onto PC1 to find extents
    projections = centered @ pc1
    tip_idx = np.argmax(projections)  # farthest along PC1 direction
    base_idx = np.argmin(projections)

    tip = pts[tip_idx]
    base = pts[base_idx]

    direction = tip - base
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        # Degenerate case: fallback to vertical
        return (0.0, -1.0)

    direction = direction / norm
    return (float(direction[0]), float(direction[1]))


def cut_polygon_at_ratio(
    polygon: List[Tuple[float, float]],
    direction: Tuple[float, float],
    bed_ratio: float = 0.75,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Cut polygon at a given ratio along the direction vector.

    Args:
        polygon: Original nail polygon (CCW or CW order).
        direction: Normalized direction vector (base→tip).
        bed_ratio: Fraction of nail length to keep as nail_bed (0.0-1.0).
                   0.75 means nail_bed = 75%, free_edge = 25%.

    Returns:
        (nail_bed_polygon, full_nail_polygon)
        nail_bed contains vertices from base to cutoff point.
        full_nail is the original polygon unchanged.
    """
    if bed_ratio <= 0.0 or bed_ratio >= 1.0:
        raise ValueError(f"bed_ratio must be in (0.0, 1.0), got {bed_ratio}")

    pts = polygon_to_coords(polygon)
    dx, dy = direction

    # Project all points onto direction vector
    projections = (pts[:, 0] * dx + pts[:, 1] * dy)

    # Sort points by projection to find base (min) and tip (max)
    sorted_indices = np.argsort(projections)
    base_proj = projections[sorted_indices[0]]
    tip_proj = projections[sorted_indices[-1]]

    total_length = tip_proj - base_proj
    if total_length < 1e-9:
        # Degenerate polygon, return as-is
        return polygon, polygon

    # Cutoff projection value
    cutoff_proj = base_proj + total_length * bed_ratio

    # Build nail_bed polygon
    nail_bed_pts = []

    # Add all vertices before the cutoff
    for idx in sorted_indices:
        proj = projections[idx]
        if proj <= cutoff_proj + 1e-9:
            nail_bed_pts.append((float(pts[idx, 0]), float(pts[idx, 1])))

    # Add two cutoff edge intersection points
    # Find edges that cross the cutoff line
    cutoff_pts = find_cutoff_edge_intersections(pts, direction, base_proj, cutoff_proj)

    if len(cutoff_pts) >= 2:
        # Sort cutoff points by their projection onto perpendicular direction
        # to maintain proper polygon order
        perp_dir = np.array([-dy, dx])  # perpendicular to direction
        cutoff_pts = sorted(cutoff_pts, key=lambda p: float(p[0]) * perp_dir[0] + float(p[1]) * perp_dir[1])

        nail_bed_pts.extend(cutoff_pts[:2])

    # Ensure we have enough points for a valid polygon
    if len(nail_bed_pts) < 3:
        # Fallback: just use the base half of the polygon
        mid = len(sorted_indices) // 2
        nail_bed_pts = [tuple(pts[idx]) for idx in sorted_indices[:mid + 1]]

    nail_bed = nail_bed_pts
    full_nail = polygon

    return nail_bed, full_nail


def find_cutoff_edge_intersections(
    pts: np.ndarray,
    direction: Tuple[float, float],
    base_proj: float,
    cutoff_proj: float,
) -> List[Tuple[float, float]]:
    """Find intersection points of cutoff line with polygon edges.

    Args:
        pts: Polygon vertices as (N, 2) array.
        direction: Normalized direction vector (base→tip).
        base_proj: Projection value of base point.
        cutoff_proj: Projection value of cutoff line.

    Returns:
        List of intersection points (max 2).
    """
    intersections = []
    n = len(pts)

    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]

        proj1 = p1[0] * direction[0] + p1[1] * direction[1]
        proj2 = p2[0] * direction[0] + p2[1] * direction[1]

        # Check if edge crosses cutoff
        if (proj1 <= cutoff_proj) != (proj2 <= cutoff_proj):
            # Linear interpolation to find exact intersection
            t = (cutoff_proj - proj1) / (proj2 - proj1 + 1e-9)
            t = max(0.0, min(1.0, t))
            ix = p1[0] + t * (p2[0] - p1[0])
            iy = p1[1] + t * (p2[1] - p1[1])
            intersections.append((float(ix), float(iy)))

    return intersections


def coords_to_yolo_line(
    class_id: int,
    polygon: List[Tuple[float, float]],
) -> str:
    """Convert polygon to YOLO segmentation line format."""
    coords = []
    for x, y in polygon:
        coords.append(f"{x:.10f}")
        coords.append(f"{y:.10f}")
    return f"{class_id} {' '.join(coords)}"


# ============================================================================
# BATCH CONVERSION
# ============================================================================

def convert_labels(
    input_dir: Path,
    output_dir: Path,
    bed_ratio: float = 0.75,
    verbose: bool = True,
) -> dict:
    """Convert all labels in a directory from 5-class to 2-class format.

    Args:
        input_dir: Directory containing original 5-class labels.
        output_dir: Output directory for 2-class labels.
        bed_ratio: Nail bed ratio (0.0-1.0).
        verbose: Print progress.

    Returns:
        Statistics dict with counts of processed files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(input_dir.glob("*.txt"))
    if not label_files:
        print(f"WARNING: No .txt label files found in {input_dir}")
        return {"processed": 0, "total_polygons": 0, "errors": 0}

    stats = {"processed": 0, "total_polygons": 0, "errors": 0}

    for label_path in label_files:
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            output_lines = []
            polygon_count = 0

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                try:
                    class_id, polygon = parse_yolo_seg_label(line)
                    polygon_count += 1

                    # Infer direction from polygon shape
                    direction = get_direction_from_polygon_pca(polygon)

                    # Cut polygon into nail_bed and full_nail
                    nail_bed, full_nail = cut_polygon_at_ratio(
                        polygon, direction, bed_ratio
                    )

                    # Write both polygons
                    # class 0 = nail_bed, class 1 = full_nail
                    output_lines.append(coords_to_yolo_line(0, nail_bed))
                    output_lines.append(coords_to_yolo_line(1, full_nail))

                except Exception as e:
                    print(f"  ERROR in {label_path.name}, line {polygon_count}: {e}")
                    stats["errors"] += 1

            # Write output file
            output_path = output_dir / label_path.name
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(output_lines) + "\n")

            stats["processed"] += 1
            stats["total_polygons"] += polygon_count

        except Exception as e:
            print(f"  ERROR processing {label_path.name}: {e}")
            stats["errors"] += 1

    if verbose:
        print(f"\nConversion complete:")
        print(f"  Processed : {stats['processed']} files")
        print(f"  Polygons  : {stats['total_polygons']} original → {stats['total_polygons'] * 2} new")
        print(f"  Errors    : {stats['errors']}")
        print(f"  Output    : {output_dir}")

    return stats


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert 5-class per-finger labels to 2-class (nail_bed + full_nail).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert train labels
  python convert_polygon_to_nailbed.py \\
      --labels train/labels \\
      --output train/labels_2class \\
      --bed-ratio 0.75

  # Convert both train and valid
  python convert_polygon_to_nailbed.py \\
      --labels train/labels \\
      --output train/labels_2class \\
      --bed-ratio 0.75 \\
      --valid valid/labels \\
      --valid-output valid/labels_2class
        """,
    )
    parser.add_argument(
        "--labels", "-l",
        type=str,
        required=True,
        help="Path to input labels directory (train/labels).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output labels directory (train/labels_2class).",
    )
    parser.add_argument(
        "--bed-ratio", "-r",
        type=float,
        default=0.75,
        help="Nail bed ratio (0.0-1.0). Default: 0.75 (75%% nail bed, 25%% free edge).",
    )
    parser.add_argument(
        "--valid",
        type=str,
        default=None,
        help="Optional: path to validation labels directory.",
    )
    parser.add_argument(
        "--valid-output",
        type=str,
        default=None,
        help="Optional: path to output validation labels directory.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    labels_path = Path(args.labels).resolve()
    output_path = Path(args.output).resolve()

    if not labels_path.exists():
        print(f"ERROR: Labels directory not found: {labels_path}")
        return 1

    if args.bed_ratio <= 0.0 or args.bed_ratio >= 1.0:
        print(f"ERROR: bed-ratio must be in (0.0, 1.0), got {args.bed_ratio}")
        return 1

    print("=" * 60)
    print("CONVERT_POLYGON_TO_NAILBED - 5-class → 2-class conversion")
    print("=" * 60)
    print(f"Input dir  : {labels_path}")
    print(f"Output dir : {output_path}")
    print(f"Bed ratio  : {args.bed_ratio}")
    print("=" * 60)

    # Convert train labels
    stats = convert_labels(labels_path, output_path, args.bed_ratio, verbose=not args.quiet)

    # Convert valid labels if provided
    if args.valid and args.valid_output:
        valid_input = Path(args.valid).resolve()
        valid_output = Path(args.valid_output).resolve()

        if valid_input.exists():
            print("\n" + "=" * 60)
            print("Processing validation set...")
            print("=" * 60)
            valid_stats = convert_labels(valid_input, valid_output, args.bed_ratio, verbose=not args.quiet)
            stats["processed"] += valid_stats["processed"]
            stats["total_polygons"] += valid_stats["total_polygons"]
            stats["errors"] += valid_stats["errors"]
        else:
            print(f"WARNING: Validation labels directory not found: {valid_input}")

    print("\n" + "=" * 60)
    print("ALL DONE")
    print(f"Total processed: {stats['processed']} files")
    print(f"Total errors   : {stats['errors']}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
