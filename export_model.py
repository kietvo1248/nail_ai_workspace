"""
================================================================================
EXPORT_MODEL.PY - Convert YOLOv11-Pose to TensorFlow Lite for Mobile
================================================================================
Script chuyển đổi file best.pt sang định dạng TFLite phục vụ deployment mobile.

**Nguyên tắc SOLID - Single Responsibility:**
  - Script này chỉ đảm nhiệm một nhiệm vụ: export model

**Tối ưu cho Mobile:**
  - format="tflite": Định dạng chuẩn của TensorFlow cho Android/iOS
  - half=True: Float16 quantization giảm ~50% dung lượng
  - imgsz=320: Kích thước input đồng nhất với training
  - nms=True: Nhúng Non-Maximum Suppression vào model (giảm post-processing)

**Hỗ trợ External Dataset:**
  - model: Đường dẫn đến best.pt (hoặc tự động tìm trong runs/)
================================================================================
"""

from ultralytics import YOLO
import os
import sys
import glob
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export YOLOv11-Pose model to TensorFlow Lite"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to best.pt (None = auto detect)"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        default="tflite",
        choices=["tflite", "onnx", "torchscript", "openvino"],
        help="Export format (default: tflite)"
    )
    parser.add_argument(
        "--imgsz", "-i",
        type=int,
        default=320,
        help="Input size (default: 320)"
    )
    parser.add_argument(
        "--half",
        action="store_true",
        default=True,
        help="FP16 quantization (default: True)"
    )
    parser.add_argument(
        "--no-half",
        action="store_true",
        help="Disable FP16 quantization"
    )
    parser.add_argument(
        "--nms",
        action="store_true",
        default=True,
        help="Embed NMS in model (default: True)"
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="INT8 quantization (lower quality, smaller size)"
    )
    parser.add_argument(
        "--batch", "-b",
        type=int,
        default=1,
        help="Batch size (default: 1)"
    )
    parser.add_argument(
        "--project", "-o",
        type=str,
        default="runs",
        help="Project folder to search for best.pt (default: runs)"
    )
    return parser.parse_args()


def find_latest_weights(project: str = "runs", name: str = "pose/train") -> str:
    """
    Tìm file best.pt mới nhất trong thư mục runs.
    
    Args:
        project: Thư mục project chứa weights
        name: Tên experiment
    
    Returns:
        str: Đường dẫn đến best.pt
    """
    # Thử tìm theo đường dẫn mặc định trước
    default_path = os.path.join(project, name, "weights", "best.pt")
    if os.path.exists(default_path):
        return default_path
    
    # Nếu không có, tìm kiếm tất cả best.pt trong thư mục runs
    search_pattern = os.path.join(project, "**", "weights", "best.pt")
    weights_files = glob.glob(search_pattern, recursive=True)
    
    if not weights_files:
        raise FileNotFoundError(
            f"[ERROR] No best.pt found in '{project}' directory.\n"
            f"[HINT] Please run train.py first to generate the model."
        )
    
    # Sắp xếp theo thời gian sửa đổi, lấy mới nhất
    latest = max(weights_files, key=os.path.getmtime)
    print(f"[INFO] Found latest weights: {latest}")
    return latest


def export_to_tflite(
    model_path: str = None,
    format: str = "tflite",
    imgsz: int = 320,
    half: bool = True,
    nms: bool = True,
    int8: bool = False,
    batch: int = 1
) -> str:
    """
    Export YOLOv11-Pose model sang TensorFlow Lite.
    
    Args:
        model_path: Đường dẫn đến best.pt (None = auto detect)
        format: Định dạng export ("tflite")
        imgsz: Kích thước input (320 tối ưu cho mobile)
        half: FP16 quantization (giảm dung lượng ~50%)
        nms: Nhúng NMS vào model (giảm post-processing code)
        int8: INT8 quantization (cần calibration data, chất lượng thấp hơn)
        batch: Batch size cho inference
    
    Returns:
        str: Đường dẫn đến file TFLite đã export
    """
    # Auto-detect model path nếu không provided
    if model_path is None:
        print("[INFO] Auto-detecting best.pt...")
        model_path = find_latest_weights()
    
    # Verify model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Model not found: {model_path}")
    
    # Load model
    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Cấu hình export
    print(f"\n{'='*60}")
    print(f"EXPORT CONFIGURATION")
    print(f"{'='*60}")
    print(f"  - Input model:    {model_path}")
    print(f"  - Output format:  {format}")
    print(f"  - Image size:     {imgsz}x{imgsz}")
    print(f"  - Half precision: {half} (FP16 quantization)")
    print(f"  - NMS embedded:   {nms}")
    print(f"  - INT8 quantized: {int8}")
    print(f"  - Batch size:     {batch}")
    print(f"{'='*60}\n")
    
    # Export model
    print("[INFO] Starting export to TensorFlow Lite...")
    export_path = model.export(
        format=format,
        imgsz=imgsz,
        half=half,
        nms=nms,
        int8=int8,
        batch=batch
    )
    
    return export_path


def get_file_size_human(path: str) -> str:
    """
    Chuyển đổi dung lượng file sang format dễ đọc.
    
    Args:
        path: Đường dẫn file
    
    Returns:
        str: Dung lượng file (VD: "4.5 MB")
    """
    size_bytes = os.path.getsize(path)
    
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def main():
    """
    Main function - Export best.pt to TFLite.
    """
    args = parse_args()

    print("="*60)
    print(" YOLOv11-Pose -> TensorFlow Lite Exporter")
    print("="*60)

    # Override half precision if --no-half is specified
    half = args.half and not args.no_half

    try:
        # Export với cấu hình từ command line
        tflite_path = export_to_tflite(
            model_path=args.model,
            format=args.format,
            imgsz=args.imgsz,
            half=half,
            nms=args.nms,
            int8=args.int8,
            batch=args.batch
        )
        
        # Kiểm tra kết quả
        if tflite_path and os.path.exists(tflite_path):
            file_size = get_file_size_human(tflite_path)
            
            print(f"\n{'='*60}")
            print(f" EXPORT SUCCESSFUL!")
            print(f"{'='*60}")
            print(f"[SUCCESS] TFLite model saved at:")
            print(f"          {os.path.abspath(tflite_path)}")
            print(f"[INFO]    File size: {file_size}")
            print(f"{'='*60}")
            print(f"\n[NEXT STEPS]")
            print(f"  1. Copy '{os.path.basename(tflite_path)}' to mobile project")
            print(f"  2. Use TensorFlow Lite interpreter for inference")
            print(f"  3. Input shape: [1, 320, 320, 3]")
            print(f"  4. Output: Bounding boxes + 4 keypoints (x, y, visible)")
            print(f"{'='*60}")
        else:
            print("[ERROR] Export failed - no output file generated")
            sys.exit(1)
            
    except FileNotFoundError as e:
        print(f"\n{str(e)}")
        print("\n[HINT] Run 'python train.py' first to train the model.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
