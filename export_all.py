"""
================================================================================
EXPORT_ALL.PY - Unified export script: ONNX (Desktop) + TFLite (Mobile)
================================================================================
Workflow:
    1. Detect available training runs in ``runs/`` (each contains best.pt).
    2. Pick a run interactively (or accept --run <name>).
    3. Export best.pt -> ONNX (imgsz=416, simplified) for the Desktop app.
    4. Attempt TFLite/LiteRT export (platform-dependent; see below).
    5. Auto-copy the ONNX file into ``nail_desktop_app/assets/`` (override via --no-copy).

For Pose models (Phase 1 + Phase 7 + 21-point Hand Pose):
    Use --pose to export a YOLO11-Pose run. Pass ``--kpt-shape 21 3`` so the
    exported ONNX contains 21 keypoints (Hand Skeleton).

For OBB models (Phase 8 - OBB refactor):
    Use --obb to export a YOLO11-OBB run. The output is ``nail_obb.onnx`` or
    ``nail_obb_5class.onnx`` depending on whether the run name suggests a
    5-class model. The 4 corner coordinates + 1 angle per detection are
    exported as a [1, C, N] tensor where C = 4 (4 corners * 2) + 1 (angle) +
    num_classes + 1 (objectness). The desktop app parses this via
    ``onnx_inference._parse_obb``.

Platform notes:
  - ONNX: works everywhere (recommended for mobile - ONNX Runtime runs on Android/iOS).
  - LiteRT/TFLite: requires Linux or macOS. On Windows these are skipped gracefully.
  - On Windows, ONNX Runtime on mobile is the primary path.

Usage:
    python export_all.py                       # interactive (Seg)
    python export_all.py --run <run_name>      # non-interactive
    python export_all.py --no-tflite           # skip TFLite
    python export_all.py --no-copy             # don't copy to desktop app
    python export_all.py --pose                # export a Pose model (4 keypoints)
    python export_all.py --pose --kpt-shape 4 3
    python export_all.py --obb                 # export an OBB model
    python export_all.py --obb --run nail_obb_smoke_20260727
    python export_all.py --seg-2class         # export a Seg 2-class model (nail_bed + full_nail)
    python export_all.py --seg-2class --run Nail_Detection_ThanhDT_seg_2class_20260729
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import List, Optional

WORKSPACE_ROOT: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = WORKSPACE_ROOT.parent
RUNS_DIR: Path = WORKSPACE_ROOT / "runs"
DESKTOP_ASSETS: Path = PROJECT_ROOT / "nail_desktop_app" / "assets"
MOBILE_DIR: Path = WORKSPACE_ROOT / "mobile_model"


def list_runs() -> List[Path]:
    """Return sorted list of run directories containing best.pt (recursive)."""
    if not RUNS_DIR.exists():
        return []
    runs: List[Path] = []
    for best_pt in RUNS_DIR.rglob("weights/best.pt"):
        run_dir = best_pt.parent.parent  # weights/ -> run root
        if run_dir not in runs:
            runs.append(run_dir)
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def pick_run(runs: List[Path]) -> Path:
    """Interactive selection of a training run."""
    print("\nAvailable training runs:")
    print("=" * 70)
    for i, run in enumerate(runs):
        best_pt = run / "weights" / "best.pt"
        size_mb = best_pt.stat().st_size / (1024 * 1024)
        print(f"  [{i + 1}] {run.name:<55s}  best.pt: {size_mb:.1f} MB")
    print("=" * 70)
    while True:
        try:
            raw = input(f"Select run [1-{len(runs)}]: ").strip()
            idx = int(raw) - 1
            if 0 <= idx < len(runs):
                return runs[idx]
            print(f"Please enter a number between 1 and {len(runs)}.")
        except (ValueError, EOFError):
            print("Invalid input, try again.")


def export_onnx(
    best_pt: Path,
    out_dir: Path,
    imgsz: int = 416,
    pose: bool = False,
    obb: bool = False,
    seg_2class: bool = False,
    kpt_shape: tuple[int, int] = (4, 3),
) -> Optional[Path]:
    """Export best.pt to ONNX format. Returns the path to the ONNX file.

    Args:
        best_pt:    Path to the trained weights.
        out_dir:    Directory where the ONNX will be saved.
        imgsz:      Input image size.
        pose:       If True, exports as a Pose model (with keypoints).
        obb:        If True, exports as an OBB model (oriented bounding boxes).
        seg_2class: If True, exports as a Seg 2-class model (nail_bed + full_nail).
        kpt_shape:  (num_keypoints, keypoint_dim) for Pose export. Ignored for OBB.
    """
    try:
        from ultralytics import YOLO
    except ImportError as e:
        print(f"ERROR: ultralytics not installed ({e}).")
        print("Run: pip install ultralytics")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    if obb:
        task = "OBB"
    elif pose:
        task = "Pose"
    else:
        task = "Seg"
    print(f"\n[ONNX] Exporting {best_pt.name} -> ONNX (task={task}, imgsz={imgsz})...")
    model = YOLO(str(best_pt))

    if pose:
        target_name = out_dir / "nail_pose.onnx"
        try:
            exported_path = model.export(
                format="onnx",
                imgsz=imgsz,
                simplify=False,
                opset=20,
                kpt_shape=list(kpt_shape),
            )
        except (TypeError, SyntaxError):
            # Ultralytics: kpt_shape auto-detected from the .pt.
            exported_path = model.export(
                format="onnx", imgsz=imgsz, simplify=False, opset=20
            )
    elif seg_2class:
        target_name = out_dir / "nail_seg_2class.onnx"
        exported_path = model.export(format="onnx", imgsz=imgsz, simplify=False, opset=20)
    else:
        # Default Seg
        suffix = "_5class" if obb and best_pt.parent.parent.name.endswith("5class") else ""
        target_name = out_dir / ("nail_obb" + suffix + ".onnx" if obb else "nail_seg.onnx")
        exported_path = model.export(format="onnx", imgsz=imgsz, simplify=False, opset=20)

    exported_path = Path(exported_path)
    if exported_path.resolve() != target_name.resolve():
        shutil.move(str(exported_path), str(target_name))
    print(f"[ONNX] Saved to: {target_name} ({target_name.stat().st_size / (1024 * 1024):.1f} MB)")
    return target_name


def export_tflite(
    best_pt: Path,
    out_dir: Path,
    imgsz: int = 416,
    pose: bool = False,
    obb: bool = False,
    seg_2class: bool = False,
) -> Optional[Path]:
    """Export best.pt to LiteRT/TFLite format.

    - Linux / macOS: uses Ultralytics' built-in ``format='litert'`` export.
    - Windows: uses onnx2tf via the two-step ONNX->TFLite pipeline.
    """
    system = platform.system()

    if system == "Windows":
        return _export_tflite_windows(best_pt, out_dir, imgsz=imgsz, pose=pose, obb=obb, seg_2class=seg_2class)

    # Linux / macOS: use Ultralytics native path.
    out_dir.mkdir(parents=True, exist_ok=True)
    if obb:
        task = "OBB"
    elif pose:
        task = "Pose"
    else:
        task = "Seg"
    print(f"\n[TFLite] Exporting {best_pt.name} -> LiteRT/TFLite FP16 "
          f"(task={task}, imgsz={imgsz})...")
    try:
        from ultralytics import YOLO
        model = YOLO(str(best_pt))
        exported_path = model.export(format="litert", imgsz=imgsz, half=True)
        exported_path = Path(exported_path)
        if obb:
            target = out_dir / "nail_obb_float16.tflite"
        elif pose:
            target = out_dir / "nail_pose_float16.tflite"
        elif seg_2class:
            target = out_dir / "nail_seg_2class_float16.tflite"
        else:
            target = out_dir / "nail_seg_float16.tflite"
        if exported_path.resolve() != target.resolve():
            shutil.move(str(exported_path), str(target))
        size_mb = target.stat().st_size / (1024 * 1024)
        print(f"[TFLite] Saved to: {target} ({size_mb:.1f} MB)")
        return target
    except Exception as e:
        print(f"[TFLite] Export failed: {e}")
        return None


def _export_tflite_windows(
    best_pt: Path,
    out_dir: Path,
    imgsz: int = 416,
    pose: bool = False,
    obb: bool = False,
    seg_2class: bool = False,
    kpt_shape: tuple[int, int] = (21, 3),
) -> Optional[Path]:
    """Windows-only: ONNX -> TFLite via onnx2tf."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import onnx2tf  # noqa: F401
    except ImportError:
        print("ERROR: onnx2tf not installed. Run:")
        print("  pip install onnx2tf tf_keras onnx-graphsurgeon")
        return None

    if obb:
        task = "OBB"
        target = out_dir / "nail_obb_float16.tflite"
    elif pose:
        task = "Pose"
        target = out_dir / "nail_pose_float16.tflite"
    elif seg_2class:
        task = "Seg 2-class"
        target = out_dir / "nail_seg_2class_float16.tflite"
    else:
        task = "Seg"
        target = out_dir / "nail_seg_float16.tflite"
    print(f"\n[TFLite] Windows pipeline: ONNX (static) -> onnx2tf -> TFLite "
          f"(task={task}, imgsz={imgsz})...")

    # --- Step 1: re-export ONNX with static shape ---
    if obb:
        static_onnx = out_dir / "nail_obb_static.onnx"
    elif pose:
        static_onnx = out_dir / "nail_pose_static.onnx"
    elif seg_2class:
        static_onnx = out_dir / "nail_seg_2class_static.onnx"
    else:
        static_onnx = out_dir / "nail_seg_static.onnx"
    try:
        from ultralytics import YOLO
        model = YOLO(str(best_pt))
        
        export_kwargs = {
            "format": "onnx",
            "imgsz": imgsz,
            "simplify": True,
            "opset": 20,
            "batch": 1,
        }
        if pose:
            export_kwargs["kpt_shape"] = list(kpt_shape)
            
        try:
            export_target = model.export(**export_kwargs)
        except (TypeError, SyntaxError):
            # Fallback for older Ultralytics versions
            export_kwargs.pop("kpt_shape", None)
            export_target = model.export(**export_kwargs)
        export_target = Path(export_target)
        if export_target.resolve() != static_onnx.resolve():
            shutil.move(str(export_target), str(static_onnx))
        print(f"[TFLite] Static ONNX: {static_onnx} "
              f"({static_onnx.stat().st_size / (1024*1024):.1f} MB)")
    except Exception as e:
        print(f"[TFLite] Static ONNX export failed: {e}")
        return None

    # --- Step 2: ONNX -> TFLite via onnx2tf ---
    if obb:
        target = out_dir / "nail_obb_float16.tflite"
    elif pose:
        target = out_dir / "nail_pose_float16.tflite"
    else:
        target = out_dir / "nail_seg_float16.tflite"
    static_onnx_abs = static_onnx.resolve()
    out_dir_abs = out_dir.resolve()
    try:
        import numpy as _np
        work = out_dir / "_onnx2tf_work"
        work.mkdir(parents=True, exist_ok=True)
        _calib_name = "calibration_image_sample_data_20x128x128x3_float32.npy"
        _calib_path = work / _calib_name
        if not _calib_path.exists():
            _np.save(str(_calib_path), _np.random.rand(20, 128, 128, 3).astype(_np.float32), allow_pickle=False)

        cwd = Path.cwd()
        try:
            run_dir_under = os.path.relpath(out_dir_abs, work)
        except ValueError:
            run_dir_under = str(out_dir_abs)
        try:
            os.chdir(work)
            onnx2tf.convert(
                input_onnx_file_path=str(static_onnx_abs),
                output_folder_path=run_dir_under,
                output_signaturedefs=True,
                non_verbose=True,
            )
        finally:
            os.chdir(cwd)

        produced = out_dir / f"{static_onnx.stem}_float16.tflite"
        if not produced.exists():
            produced = out_dir / f"{static_onnx.stem}_float32.tflite"
        if produced.exists():
            if produced.resolve() != target.resolve():
                shutil.move(str(produced), str(target))
            for junk in out_dir.glob(f"{static_onnx.stem}*.tflite"):
                if junk.resolve() != target.resolve():
                    junk.unlink()
            for junk in out_dir.glob(f"{static_onnx.stem}*.json"):
                try:
                    junk.unlink()
                except OSError:
                    pass
        else:
            print("[TFLite] onnx2tf finished but no .tflite file was found.")
            return None
        try:
            static_onnx.unlink()
        except OSError:
            pass
        try:
            shutil.rmtree(work, ignore_errors=True)
        except OSError:
            pass
        if target.exists():
            print(f"[TFLite] Saved to: {target} "
                  f"({target.stat().st_size / (1024*1024):.1f} MB)")
            return target
        return None
    except Exception as e:
        print(f"[TFLite] onnx2tf conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def copy_to_desktop(onnx_path: Path, pose: bool = False, obb: bool = False, seg_2class: bool = False) -> Optional[Path]:
    """Copy the ONNX file to nail_desktop_app/assets/."""
    if not DESKTOP_ASSETS.exists():
        print(f"[Copy] Desktop assets dir not found: {DESKTOP_ASSETS}")
        print("  Skipping copy. Place the ONNX file manually.")
        return None
    DESKTOP_ASSETS.mkdir(parents=True, exist_ok=True)
    target = DESKTOP_ASSETS / onnx_path.name
    shutil.copy2(str(onnx_path), str(target))
    if obb:
        label = "OBB"
    elif pose:
        label = "Pose"
    elif seg_2class:
        label = "Seg 2-class"
    else:
        label = "Seg"
    print(f"[Copy] Copied {label} ONNX to: {target}")
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified ONNX + TFLite exporter for nail-seg / nail-pose / nail-obb YOLO models."
    )
    parser.add_argument("--run", type=str, default=None,
                        help="Specific run directory name under runs/ (skip interactive pick)")
    parser.add_argument("--imgsz", type=int, default=416,
                        help="Input image size for export (default: 416)")
    parser.add_argument("--no-tflite", action="store_true",
                        help="Skip TFLite export (ONNX only)")
    parser.add_argument("--no-copy", action="store_true",
                        help="Don't copy ONNX to nail_desktop_app/assets/")
    parser.add_argument("--pose", action="store_true",
                        help="Export a Pose model (expects kpt_shape in data.yaml)")
    parser.add_argument("--obb", action="store_true",
                        help="Export an OBB (oriented bounding box) model")
    parser.add_argument("--seg", action="store_true",
                        help="Export a Seg model (default when neither --pose nor --obb is set)")
    parser.add_argument("--seg-2class", action="store_true",
                        help="Export a Seg 2-class model (nail_bed + full_nail)")
    parser.add_argument("--kpt-shape", type=int, nargs=2, default=[21, 3],
                        help="Pose keypoint shape: <num_keypoints> <keypoint_dim>")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    runs = list_runs()
    if not runs:
        print("ERROR: No training runs with best.pt found in runs/")
        print("Train a model first: python train.py")
        if args.pose:
            print("Or train a Pose model: python train_pose.py")
        elif args.obb:
            print("Or train an OBB model: python train_obb.py")
        return 1

    if args.run:
        run_dir = RUNS_DIR / args.run
        if not run_dir.exists() or not (run_dir / "weights" / "best.pt").exists():
            print(f"ERROR: Run '{args.run}' not found or missing best.pt")
            print("Available runs:")
            for r in runs:
                print(f"  - {r.name}")
            return 1
    else:
        run_dir = pick_run(runs)

    best_pt = run_dir / "weights" / "best.pt"
    run_name_lower = run_dir.name.lower()
    if args.obb or "obb" in run_name_lower:
        args.obb = True
        mode_label = "OBB"
    elif args.pose or "pose" in run_name_lower:
        args.pose = True
        mode_label = "Pose"
    elif args.seg_2class or "2class" in run_name_lower:
        args.seg_2class = True
        mode_label = "Seg 2-class"
    else:
        mode_label = "Seg"
    print(f"\nSelected run: {run_dir.name}")
    print(f"Source model: {best_pt}")
    print(f"Mode        : {mode_label}")

    mobile_out = MOBILE_DIR / run_dir.name
    mobile_out.mkdir(parents=True, exist_ok=True)

    # --- ONNX ---
    onnx_path = export_onnx(
        best_pt, mobile_out,
        imgsz=args.imgsz,
        pose=args.pose,
        obb=args.obb,
        seg_2class=args.seg_2class,
        kpt_shape=tuple(args.kpt_shape),
    )
    if onnx_path is None:
        return 1

    # --- TFLite ---
    tflite_path: Optional[Path] = None
    if not args.no_tflite:
        tflite_path = export_tflite(best_pt, mobile_out, imgsz=args.imgsz, pose=args.pose, obb=args.obb, seg_2class=args.seg_2class)
        if tflite_path is None:
            sysname = platform.system()
            if sysname == "Windows":
                print("[TFLite] NOTE: TFLite export is not supported on Windows natively.")
                print("         To get TFLite, run this script on Linux/macOS, or use --no-tflite.")
            else:
                print("[TFLite] Export failed — see error messages above.")

    # --- Copy to desktop ---
    if not args.no_copy:
        copy_to_desktop(onnx_path, pose=args.pose, obb=args.obb, seg_2class=args.seg_2class)
        # Also copy TFLite if it was produced.
        if tflite_path and tflite_path.exists():
            tflite_dest = DESKTOP_ASSETS / tflite_path.name
            shutil.copy2(str(tflite_path), str(tflite_dest))
            print(f"[Copy] Copied TFLite to: {tflite_dest}")
        elif not args.no_tflite:
            # Show TFLite status even when it was skipped/failed.
            sysname = platform.system()
            if sysname == "Windows":
                print("[TFLite] SKIPPED on Windows (requires --no-tflite or Linux/macOS)")
            elif sysname == "Linux" or sysname == "Darwin":
                print("[TFLite] Export failed — see messages above")

    # --- Summary ---
    print("\n" + "=" * 70)
    print(f"EXPORT SUMMARY  ({mode_label})")
    print("=" * 70)
    print(f"Run            : {run_dir.name}")
    print(f"ONNX           : {onnx_path}  "
          f"({onnx_path.stat().st_size / (1024 * 1024):.2f} MB)")
    if tflite_path and tflite_path.exists():
        print(f"TFLite FP16    : {tflite_path}  "
              f"({tflite_path.stat().st_size / (1024 * 1024):.2f} MB)")
    elif not args.no_tflite:
        sysname = platform.system()
        if sysname == "Windows":
            print("TFLite FP16    : SKIPPED on Windows "
                  "(use --no-tflite or run on Linux/macOS)")
        else:
            print("TFLite FP16    : FAILED — see messages above")
    if not args.no_copy:
        print(f"Desktop ONNX   : {DESKTOP_ASSETS / onnx_path.name}")
        if tflite_path and tflite_path.exists():
            print(f"Desktop TFLite : {DESKTOP_ASSETS / tflite_path.name}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())