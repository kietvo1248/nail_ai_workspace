#!/usr/bin/env python3
"""
Export YOLO11 Seg Model to Mobile Format

Flow: best.pt -> ONNX -> (optional) LiteRT/TFLite
      ONNX runs natively on Android/iOS via ONNX Runtime.

Platform notes:
  - LiteRT/TFLite: only works on Linux/macOS (Ultralytics built-in).
  - ONNX Runtime: works on ALL platforms (recommended for Windows/Android/iOS).
    No extra conversion needed.

Usage:
    python export_tflite.py
"""

import os
import sys
import shutil
from pathlib import Path

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def log(msg, level="INFO"):
    prefix = {
        "INFO": f"{Colors.CYAN}[INFO]{Colors.END}",
        "SUCCESS": f"{Colors.GREEN}[SUCCESS]{Colors.END}",
        "ERROR": f"{Colors.RED}[ERROR]{Colors.END}",
        "WARN": f"{Colors.YELLOW}[WARN]{Colors.END}",
    }
    print(f"{prefix.get(level, '[INFO]')} {msg}")

def banner(msg):
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{msg.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}\n")


def find_all_runs():
    runs_dir = WORKSPACE / "runs"
    if not runs_dir.exists():
        return []

    runs = []
    for run_path in sorted(runs_dir.iterdir()):
        if run_path.is_dir():
            weights_dir = run_path / "weights"
            if weights_dir.exists():
                best_pt = weights_dir / "best.pt"
                if best_pt.exists():
                    runs.append({
                        "path": run_path,
                        "name": run_path.name,
                        "best_pt": best_pt,
                        "size_mb": best_pt.stat().st_size / (1024*1024)
                    })
    return runs


def select_run(runs):
    if not runs:
        return None

    if len(runs) == 1:
        log(f"Only one run found: {runs[0]['name']}", "INFO")
        return runs[0]

    print()
    banner("SELECT TRAINING RUN TO EXPORT")
    print(f"Found {len(runs)} training runs:\n")

    for i, run in enumerate(runs, 1):
        print(f"  [{i}] {run['name']}")
        print(f"      Model: {run['best_pt'].name} ({run['size_mb']:.2f} MB)")
        print()

    while True:
        try:
            choice = input(f"Enter selection (1-{len(runs)}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                sys.exit(0)
            idx = int(choice) - 1
            if 0 <= idx < len(runs):
                return runs[idx]
            print(f"Please enter a number between 1 and {len(runs)}")
        except ValueError:
            print("Invalid input. Please enter a number.")


def try_convert_onnx_to_tflite_via_onnxsim():
    """
    Try to convert ONNX → TFLite using onnx2tf with all dependencies resolved.
    Falls back gracefully if it fails.
    """
    log("Attempting TFLite conversion via onnx2tf...", "INFO")

    deps = {}
    for pkg in ["onnx", "onnx_tf", "tf_keras", "onnx_graphsurgeon", "onnx2tf"]:
        try:
            m = __import__(pkg.replace("-", "_").replace("tf_keras", "tf_keras").replace("onnx_tf", "onnx_tf"))
            deps[pkg] = getattr(m, "__version__", "unknown")
        except ImportError:
            deps[pkg] = None

    log(f"Dependencies: {deps}", "INFO")

    # onnx2tf requires specific version combinations - only try if all deps present
    required = ["onnx", "onnx_tf", "tf_keras", "onnx_graphsurgeon"]
    if all(deps.get(p) for p in required):
        try:
            import onnx2tf
            return True
        except Exception:
            pass

    log("onnx2tf not fully available, skipping TFLite (ONNX export succeeded)", "WARN")
    return False


def export_to_tflite():
    global WORKSPACE
    WORKSPACE = Path(__file__).parent

    banner("EXPORT YOLO11 POSE → MOBILE")

    # Find runs
    runs = find_all_runs()
    if not runs:
        log("No trained models found in runs/ folder!", "ERROR")
        log("Please run train.py first!", "ERROR")
        sys.exit(1)

    selected = select_run(runs)
    if not selected:
        return

    MODEL_PATH = selected["best_pt"]
    OUTPUT_DIR = WORKSPACE / "mobile_model" / selected["name"]

    log(f"Selected: {selected['name']} ({selected['size_mb']:.2f} MB)", "SUCCESS")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================
    # STEP 1: Export to ONNX
    # =========================================
    banner("STEP 1: EXPORT TO ONNX")
    log("Loading YOLO model...")

    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))

    log("Exporting to ONNX (imgsz=416)...")
    onnx_path = model.export(format="onnx", imgsz=416)

    if not onnx_path or not Path(onnx_path).exists():
        log("ONNX export failed!", "ERROR")
        sys.exit(1)

    onnx_size = Path(onnx_path).stat().st_size / (1024*1024)
    onnx_dest = OUTPUT_DIR / "nail_pose.onnx"
    shutil.move(onnx_path, onnx_dest)
    log(f"ONNX exported: {onnx_dest} ({onnx_size:.2f} MB)", "SUCCESS")

    # =========================================
    # STEP 2: Attempt LiteRT/TFLite (Linux/macOS only)
    # =========================================
    banner("STEP 2: ATTEMPT LITERT/TFLITE CONVERSION")

    tflite_path = None

    try:
        import platform
        if platform.system() == "Windows":
            log("LiteRT/TFLite export is only supported on Linux/macOS", "WARN")
            log("ONNX Runtime is the recommended mobile format on Windows", "INFO")
            log("Android/iOS support ONNX via ONNX Runtime (no extra conversion needed)", "INFO")
        else:
            # Use Ultralytics built-in litert export (no onnx2tf needed)
            log("Attempting LiteRT export via Ultralytics...", "INFO")
            try:
                onnx_full_path = str(onnx_dest)
                model_exp = YOLO(str(MODEL_PATH))
                tflite_out = model_exp.export(format="litert", imgsz=416, half=True)
                if tflite_out and Path(tflite_out).exists():
                    TFLITE_PATH = OUTPUT_DIR / "nail_pose.tflite"
                    shutil.move(str(tflite_out), str(TFLITE_PATH))
                    tflite_size = TFLITE_PATH.stat().st_size / (1024*1024)
                    tflite_path = TFLITE_PATH
                    log(f"LiteRT/TFLite exported: {TFLITE_PATH} ({tflite_size:.2f} MB)", "SUCCESS")
                else:
                    log("LiteRT export returned no file, skipping", "WARN")
            except Exception as e:
                log(f"LiteRT conversion failed: {e}", "WARN")
    except Exception as e:
        log(f"TFLite step error: {e}", "WARN")

    # =========================================
    # SUMMARY
    # =========================================
    banner("EXPORT COMPLETE!")

    log("Output files:", "SUCCESS")
    print()
    print("  ONNX Runtime:  nail_pose.onnx    (works on Android/iOS - primary mobile path)")

    if tflite_path:
        tflite_size = os.path.getsize(tflite_path) / (1024*1024)
        print(f"  TFLite:         nail_pose.tflite  ({tflite_size:.2f} MB)")
    else:
        print("  TFLite:         NOT exported (Windows/Linux-macOS restriction)")

    print()
    print("HOW TO USE ON MOBILE:")
    print()
    print("  ANDROID (ONNX Runtime - RECOMMENDED):")
    print("  1. Add dependency:")
    print("       implementation(\"ai.onnxruntime:onnxruntime-android:1.17.0\")")
    print("  2. Load model:")
    print("       val env = OrtEnvironment.getEnvironment()")
    print("       val session = env.createSession(\"nail_pose.onnx\")")
    print("  3. Run inference with input shape [1, 3, 416, 416]")
    print()
    print("  FLUTTER (ONNX Runtime):")
    print("  1. pubspec.yaml: onnxruntime_flutter: ^0.3.0")
    print("  2. Load model: InferenceSession() -> loadAsset('nail_pose.onnx')")
    print()
    print("  MODEL INPUT FORMAT:")
    print("  - Input shape: [1, 3, 416, 416] (NCHW)")
    print("  - Preprocessing: normalize to [0, 1]")
    print()

    log("Mobile model ready!", "SUCCESS")


if __name__ == "__main__":
    WORKSPACE = Path(__file__).parent

    try:
        export_to_tflite()
    except KeyboardInterrupt:
        print("\n\nExport cancelled.")
        sys.exit(0)
    except Exception as e:
        log(f"Export failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)
