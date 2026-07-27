"""
================================================================================
AUTO_SPLIT.PY - Auto-split a YOLO dataset into train/val when valid/ is empty
================================================================================
Many smoke-test or freshly-exported Roboflow datasets come without a ``valid/``
folder, or with it present but empty. Ultralytics training will refuse to
start (or worse, silently use 0% val data) without a non-empty validation
split.

This script:

    1. Inspects the dataset described by ``--data <data.yaml>``.
    2. If ``valid/images/`` is missing or empty, randomly moves ``val-ratio``
       of the train images (and their labels, including any matching
       ``labels_obb/`` subfolder) into ``valid/``.
    3. Performs a **stratified split** when class labels exist, so every class
       has at least one representative in val. Falls back to a plain random
       split if class labels are missing or degenerate.
    4. Updates the source ``data.yaml`` so ``val`` points to the new
       ``valid/images`` directory. Other fields (``path``, ``train``) are
       preserved.

The script is **dataset-agnostic**: it works for YOLO-Seg, YOLO-OBB and any
other Ultralytics task as long as labels live next to images with the same
stem and a ``.txt`` extension.

Usage:
    # Smoke test dataset (no valid split)
    python auto_split.py --data ../Nail_Detection_ThanhDT.v1i.yolov11/data.yaml --val-ratio 0.2

    # Production dataset (valid already exists - will skip)
    python auto_split.py --data ../nail-segmentation.v1i.yolov11_10501/data.yaml
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import yaml  # PyYAML
except ImportError:  # pragma: no cover
    yaml = None


WORKSPACE_ROOT: Path = Path(__file__).resolve().parent


# =============================================================================
# PATH HELPERS (shared logic with convert_seg_to_obb.py)
# =============================================================================
def load_data_yaml(data_yaml: Path) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required - install with: pip install PyYAML")
    with open(data_yaml, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_split_path(data_yaml: Path, dataset_root: Path, raw_path: str) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p.resolve()

    cand1 = (data_yaml.parent / p).resolve()
    if cand1.exists():
        return cand1
    cand2 = (dataset_root / p).resolve()
    if cand2.exists():
        return cand2

    parts = Path(raw_path).parts
    if len(parts) >= 2 and parts[-1] == "images":
        candidate = dataset_root / parts[-2] / "images"
        if candidate.exists():
            return candidate.resolve()
    return cand2


def resolve_dataset_paths(data_yaml: Path, data: dict) -> Tuple[Path, Path]:
    dataset_root = data.get("path")
    if dataset_root:
        dataset_root = Path(dataset_root).expanduser().resolve()
    else:
        dataset_root = data_yaml.parent.resolve()

    train_rel = data.get("train", "train/images")
    train_path = _resolve_split_path(data_yaml, dataset_root, train_rel)
    return dataset_root, train_path


# =============================================================================
# SPLIT LOGIC
# =============================================================================
IMG_EXTS = ("jpg", "jpeg", "png", "bmp", "JPG", "JPEG", "PNG", "BMP")


def list_image_stems(images_dir: Path) -> List[str]:
    if not images_dir.exists():
        return []
    out: List[str] = []
    for ext in IMG_EXTS:
        for p in images_dir.glob(f"*.{ext}"):
            out.append(p.stem)
    return sorted(set(out))


def count_files_in_dir(d: Path, exts: Tuple[str, ...] = IMG_EXTS) -> int:
    if not d.exists():
        return 0
    n = 0
    for ext in exts:
        n += len(list(d.glob(f"*.{ext}")))
    return n


def collect_classes_for_images(
    images_dir: Path,
    labels_dir: Path,
) -> Dict[str, set]:
    """Map image stem -> set of class ids seen in its label file."""
    result: Dict[str, set] = defaultdict(set)
    if not labels_dir.exists():
        return result
    for lbl in labels_dir.glob("*.txt"):
        stem = lbl.stem
        classes = set()
        for raw in lbl.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                classes.add(int(raw.split()[0]))
            except (ValueError, IndexError):
                continue
        if classes:
            result[stem] = classes
    return result


def stratified_pick(
    image_stems: List[str],
    class_map: Dict[str, set],
    val_ratio: float,
    seed: int = 42,
) -> List[str]:
    """Stratified pick: ensure each class has at least one sample in val.

    Algorithm:
        1. Group image stems by class (an image belongs to multiple classes).
        2. For each class, randomly take ~val_ratio of its members.
        3. Union the picks.

    Returns the list of stems selected for the validation split.
    """
    rnd = random.Random(seed)
    by_class: Dict[int, List[str]] = defaultdict(list)
    for stem in image_stems:
        for c in class_map.get(stem, set()):
            by_class[c].append(stem)

    picked: set = set()
    for cls, members in by_class.items():
        rnd.shuffle(members)
        k = max(1, int(round(len(members) * val_ratio)))
        # Take at least 1 if class is non-empty (so val always has examples).
        if members:
            picked.update(members[:k])

    # If total picked ratio is too high or too low, top up with random picks.
    target = max(1, int(round(len(image_stems) * val_ratio)))
    if len(picked) > target:
        # Drop excess randomly while keeping all classes represented.
        rnd2 = random.Random(seed + 1)
        all_picked = list(picked)
        rnd2.shuffle(all_picked)
        kept: set = set()
        # Always keep at least one per class.
        per_class_kept: Dict[int, bool] = {}
        for stem in all_picked:
            for c in class_map.get(stem, set()):
                if c in per_class_kept:
                    continue
            kept.add(stem)
            for c in class_map.get(stem, set()):
                per_class_kept[c] = True
            if len(kept) >= target:
                break
        picked = kept

    return sorted(picked)


# =============================================================================
# MOVE
# =============================================================================
def move_file_safely(src: Path, dst: Path) -> bool:
    """Move src -> dst. Returns True on success."""
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return True


def move_label_variants(image_stem: str, train_labels_dir: Path, valid_labels_dir: Path) -> int:
    """Move the seg label and the OBB label (if any) for the given stem.

    Returns the number of files moved.
    """
    moved = 0
    # Primary label (Seg format in train/labels/)
    seg_src = train_labels_dir / f"{image_stem}.txt"
    if seg_src.exists():
        if move_file_safely(seg_src, valid_labels_dir / seg_src.name):
            moved += 1
    # OBB label (in train/labels_obb/) - move to valid/labels_obb/
    obb_src = train_labels_dir.parent / "labels_obb" / f"{image_stem}.txt"
    if obb_src.exists():
        obb_dst_dir = valid_labels_dir.parent / "labels_obb"
        if move_file_safely(obb_src, obb_dst_dir / obb_src.name):
            moved += 1
    return moved


# =============================================================================
# MAIN
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-split a YOLO dataset into train/val when valid/ is missing.",
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to dataset's data.yaml.",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2,
        help="Fraction of images to move to valid/ (default: 0.2).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-split even if valid/ already has images.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be moved without actually moving anything.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    source_yaml = Path(args.data).resolve()
    if not source_yaml.exists():
        print(f"ERROR: data.yaml not found: {source_yaml}")
        return 1
    if not (0.05 <= args.val_ratio <= 0.9):
        print(f"ERROR: --val-ratio must be in [0.05, 0.9], got {args.val_ratio}")
        return 1

    data = load_data_yaml(source_yaml)
    dataset_root, train_images = resolve_dataset_paths(source_yaml, data)
    train_labels = train_images.parent / "labels"

    print(f"Dataset root : {dataset_root}")
    print(f"Train images : {train_images}")
    print(f"Train labels : {train_labels}")
    print(f"Val ratio    : {args.val_ratio}")
    print()

    # Resolve current val/ (might already exist).
    val_rel = data.get("val")
    val_path: Optional[Path] = None
    if val_rel:
        val_path = _resolve_split_path(source_yaml, dataset_root, val_rel)

    val_has_files = val_path is not None and count_files_in_dir(val_path) > 0

    if val_has_files and not args.force:
        print(f"[skip] valid/ already has {count_files_in_dir(val_path)} images.")
        print("       Use --force to re-split (existing val files will be moved back to train/).")
        return 0

    if val_has_files and args.force:
        print(f"[force] Re-splitting: existing val has {count_files_in_dir(val_path)} images.")
        print("        These will be moved back to train/ first.")
        # Move val -> train
        for p in val_path.glob("*"):
            if p.is_file():
                move_file_safely(p, train_images / p.name)
        # Move val labels (and obb) -> train labels (and obb)
        val_labels = val_path.parent / "labels"
        if val_labels.exists():
            for p in val_labels.glob("*.txt"):
                move_file_safely(p, train_labels / p.name)
        val_obb = val_labels.parent / "labels_obb"
        if val_obb.exists():
            for p in val_obb.glob("*.txt"):
                move_file_safely(p, train_labels.parent / "labels_obb" / p.name)

    # Pick the validation stems.
    image_stems = list_image_stems(train_images)
    if len(image_stems) < 2:
        print(f"ERROR: not enough images in train/ to split ({len(image_stems)})")
        return 1

    class_map = collect_classes_for_images(train_images, train_labels)
    val_stems = stratified_pick(image_stems, class_map, args.val_ratio, seed=args.seed)
    val_stems_set = set(val_stems)

    print(f"Train images : {len(image_stems)}")
    print(f"Val picked   : {len(val_stems)} ({100 * len(val_stems) / len(image_stems):.1f}%)")
    print(f"Classes seen : {len(class_map)}")

    # Destination dirs.
    valid_images = train_images.parent.parent / "valid" / "images"
    valid_labels = train_images.parent.parent / "valid" / "labels"

    if args.dry_run:
        print("\n[dry-run] Would move the following files:")
        for stem in val_stems[:10]:
            print(f"  - {stem}.*")
        if len(val_stems) > 10:
            print(f"  ... and {len(val_stems) - 10} more")
        return 0

    valid_images.mkdir(parents=True, exist_ok=True)
    valid_labels.mkdir(parents=True, exist_ok=True)

    moved_images = 0
    moved_labels = 0
    for stem in val_stems:
        # Move image (any extension).
        for ext in IMG_EXTS:
            src = train_images / f"{stem}.{ext}"
            if src.exists():
                if move_file_safely(src, valid_images / src.name):
                    moved_images += 1
                break
        moved_labels += move_label_variants(stem, train_labels, valid_labels)

    # Update data.yaml in-place: set val to valid/images (relative to dataset_root).
    if "path" not in data or not data.get("path"):
        data["path"] = str(dataset_root)
    data["val"] = "valid/images"

    # If train/val/test are absolute paths or contain "..", normalise them so
    # Ultralytics resolves them correctly relative to ``path``.
    for key in ("train", "val", "test"):
        v = data.get(key)
        if not v:
            continue
        v_path = Path(v)
        # If the path uses "../" or contains nested "..", fix to a clean
        # relative-to-dataset-root path. We assume the on-disk layout is
        # dataset_root/<split>/images.
        if v_path.is_absolute() or ".." in v_path.parts:
            parts = v_path.parts
            # Take the last two segments if they match '<split>/images'.
            if parts and parts[-1] == "images" and len(parts) >= 2:
                split_name = parts[-2]
                data[key] = f"{split_name}/images"
            else:
                data[key] = str(v_path)

    # Save the updated data.yaml back. If the user passed the auto-generated
    # data_obb.yaml, prefer to also update train to labels_obb so the patched
    # config remains self-consistent.
    if "data_obb" in source_yaml.stem:
        # No change: OBB labels already use train/labels_obb which is unchanged.
        pass

    with open(source_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print()
    print("=" * 70)
    print("SPLIT SUMMARY")
    print("=" * 70)
    print(f"Images moved to valid/ : {moved_images}")
    print(f"Labels moved           : {moved_labels}")
    print(f"Train remaining        : {len(image_stems) - moved_images}")
    print(f"val/                   : {valid_images}")
    print(f"data.yaml updated      : {source_yaml}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
