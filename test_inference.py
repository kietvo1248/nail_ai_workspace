"""
================================================================================
TEST_INFERENCE.PY - Visual Debug Tool for Nail Keypoint / OBB Detection
================================================================================
Script debug nội bộ dùng OpenCV để visualize kết quả inference.

**Công dụng:**
  - Load một ảnh test bất kỳ
  - Chạy inference với best.pt
  - Vẽ Bounding Box + 4 Keypoints (Pose) HOẶC Oriented Rectangle + angle arrow (OBB)
  - Lưu ảnh kết quả để kỹ sư kiểm tra bằng mắt (Visual Inspection)
  - Hỗ trợ team Mobile xác minh model hoạt động đúng trước khi deploy

**Visual Output theo task:**
  - ``pose``: Bounding Box + 4 Keypoints (Top, Bottom, Left, Right)
  - ``obb``:  Oriented rectangle (4 corners) + rotation arrow từ bbox center
              theo góc ``angle_deg``.
  - ``seg``:  (legacy) Bounding Box + segmentation mask polygon

**Hỗ trợ External Dataset & Model:**
  - --model: Đường dẫn đến best.pt
  - --image: Đường dẫn ảnh test cụ thể
  - --task:  pose (default) | obb | seg
================================================================================
"""

import argparse
import glob
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

# Màu sắc cho visualization (BGR format cho OpenCV)
COLORS = {
    "bbox": (0, 255, 0),          # Xanh lá - Bounding Box
    "obb": (0, 255, 255),         # Vàng - Oriented rectangle
    "obb_corner": (0, 200, 255),  # Cam nhạt - OBB corner
    "obb_arrow": (255, 100, 100), # Đỏ nhạt - rotation arrow
    "keypoint_visible": (0, 0, 255),  # Đỏ - Keypoint nhìn thấy
    "keypoint_occluded": (128, 128, 128),  # Xám - Keypoint bị che
    "text": (255, 255, 255),      # Trắng - Text
    "line": (255, 255, 0),        # Cyan - Đường nối keypoints
}

# Tên các keypoints (theo thứ tự trong model output)
KEYPOINT_NAMES = ["Top", "Bottom", "Left", "Right"]

# Ngưỡng confidence tối thiểu để hiển thị kết quả
CONFIDENCE_THRESHOLD = 0.25

# Ngưỡng visibility tối thiểu
VISIBILITY_THRESHOLD = 0.5


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test YOLO inference (Pose/OBB/Seg) with visual output"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to best.pt (None = auto detect)"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Path to test image (None = auto detect)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {CONFIDENCE_THRESHOLD})"
    )
    parser.add_argument(
        "--imgsz", "-s",
        type=int,
        default=320,
        help="Input size (default: 320)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output",
        help="Output directory (default: output)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save output image"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs",
        help="Project folder to search for best.pt (default: runs)"
    )
    parser.add_argument(
        "--task",
        choices=["pose", "obb", "seg"],
        default="pose",
        help="Inference task: pose (4 keypoints), obb (oriented bbox), seg (legacy). Default: pose.",
    )
    return parser.parse_args()


def find_best_model(project: str = "runs") -> str:
    """
    Tự động tìm file best.pt mới nhất.

    Args:
        project: Thư mục project để tìm model

    Returns:
        str: Đường dẫn đến best.pt
    """
    search_pattern = os.path.join(project, "**", "weights", "best.pt")
    weights_files = glob.glob(search_pattern, recursive=True)

    if not weights_files:
        raise FileNotFoundError(
            f"[ERROR] No best.pt found in '{project}' directory.\n"
            f"[HINT] Please run train.py first to train the model."
        )

    # Lấy file mới nhất
    latest = max(weights_files, key=os.path.getmtime)
    print(f"[INFO] Using model: {latest}")
    return latest


def find_test_image(default_path: str = "dataset/images/val") -> str:
    """
    Tìm một ảnh test trong thư mục validation.
    
    Args:
        default_path: Thư mục chứa ảnh test
    
    Returns:
        str: Đường dẫn đến ảnh test đầu tiên tìm được
    """
    # Các extension ảnh phổ biến
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    
    for ext in extensions:
        pattern = os.path.join(default_path, ext)
        images = glob.glob(pattern)
        if images:
            # Lấy ảnh đầu tiên
            test_img = max(images, key=os.path.getmtime)
            print(f"[INFO] Found test image: {test_img}")
            return test_img
    
    raise FileNotFoundError(
        f"[ERROR] No test images found in '{default_path}'"
    )


def draw_bbox(frame: np.ndarray, box: np.ndarray, conf: float) -> None:
    """
    Vẽ Bounding Box lên ảnh.

    Args:
        frame: Ảnh numpy (OpenCV format BGR)
        box: Bounding box [x1, y1, x2, y2]
        conf: Confidence score
    """
    x1, y1, x2, y2 = map(int, box)

    # Vẽ rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS["bbox"], 2)

    # Vẽ nhãn với background
    label = f"nail {conf:.2f}"
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

    # Background cho text
    cv2.rectangle(
        frame,
        (x1, y1 - label_size[1] - 10),
        (x1 + label_size[0] + 10, y1),
        COLORS["bbox"],
        -1
    )

    # Text
    cv2.putText(
        frame, label,
        (x1 + 5, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["text"], 2
    )


def draw_obb(
    frame: np.ndarray,
    xyxyxyxy: np.ndarray,
    xywhr: np.ndarray,
    cls_id: int,
    cls_name: str,
    conf: float,
) -> None:
    """Draw oriented bounding box: 4 corners + rotation arrow + label.

    Args:
        frame: BGR image
        xyxyxyxy: shape (4, 2) - 4 (x, y) corners in image coords. Order is
            whatever Ultralytics gives (BR, TR, TL, BL in 8.4.x).
        xywhr: shape (5,) - (cx, cy, w, h, angle_rad)
        cls_id / cls_name: classification info (for the label text)
        conf: detection confidence
    """
    if xyxyxyxy is None or len(xyxyxyxy) < 4:
        return

    # Draw the oriented rectangle.
    pts = np.asarray(xyxyxyxy, dtype=np.float32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts.astype(np.int32)], isClosed=True, color=COLORS["obb"], thickness=2)

    # Draw each corner with a small filled circle.
    for px, py in xyxyxyxy:
        cv2.circle(frame, (int(px), int(py)), 4, COLORS["obb_corner"], -1, cv2.LINE_AA)
        cv2.circle(frame, (int(px), int(py)), 4, (50, 50, 50), 1, cv2.LINE_AA)

    # Rotation arrow from bbox center along angle_deg.
    cx, cy, w, h, angle_rad = xywhr
    if abs(angle_rad) > 0.01:
        length = max(w, h) * 0.55
        rad = float(angle_rad)
        # Image-space arrow direction. Ultralytics' angle convention for OBB
        # is a CCW rotation in radians; convert to (dx, dy) for cv2.arrowedLine.
        dx = math.cos(rad) * length
        dy = math.sin(rad) * length
        ex = cx + dx
        ey = cy + dy
        cv2.arrowedLine(
            frame,
            (int(round(cx)), int(round(cy))),
            (int(round(ex)), int(round(ey))),
            COLORS["obb_arrow"],
            2,
            tipLength=0.25,
        )

    # Label at first corner (BR).
    x0 = int(round(float(xyxyxyxy[0][0])))
    y0 = int(round(float(xyxyxyxy[0][1])))
    label = f"{cls_name} {conf:.2f} ang={math.degrees(angle_rad):.1f}"
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    # Background box above the label anchor.
    cv2.rectangle(
        frame,
        (x0, max(0, y0 - label_size[1] - 8)),
        (x0 + label_size[0] + 8, y0),
        COLORS["obb"],
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x0 + 4, max(12, y0 - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),  # Black text on yellow background
        1,
        cv2.LINE_AA,
    )


def draw_keypoints(frame: np.ndarray, keypoints: np.ndarray, box: np.ndarray) -> None:
    """
    Vẽ 4 keypoints và đường nối lên ảnh.
    
    Keypoints: Top, Bottom, Left, Right của móng tay
    
    Args:
        frame: Ảnh numpy (OpenCV format BGR)
        keypoints: Array shape [4, 3] = [[x, y, visible], ...]
        box: Bounding box để scale keypoints về đúng tọa độ ảnh gốc
    """
    # Chuyển đổi keypoints từ normalized (0-1) sang pixel coordinates
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    
    points = []
    for i, (name, kp) in enumerate(zip(KEYPOINT_NAMES, keypoints)):
        x_norm, y_norm, visible = kp
        
        # Scale về tọa độ trong ảnh gốc
        x = int(x_norm * w)
        y = int(y_norm * h)
        
        # Kiểm tra visibility
        is_visible = visible >= VISIBILITY_THRESHOLD
        color = COLORS["keypoint_visible"] if is_visible else COLORS["keypoint_occluded"]
        
        # Vẽ điểm
        radius = 6 if is_visible else 4
        cv2.circle(frame, (x, y), radius, color, -1)
        cv2.circle(frame, (x, y), radius, (255, 255, 255), 2)  # Viền trắng
        
        # Vẽ nhãn
        offset_y = -15 if i % 2 == 0 else 25
        cv2.putText(
            frame, name,
            (x + 8, y + offset_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )
        
        points.append((x, y, is_visible))
    
    # Vẽ đường nối giữa các keypoints (hình chữ nhật của móng)
    if len(points) == 4:
        # Kết nối: Top -> Right -> Bottom -> Left -> Top
        indices = [0, 3, 1, 2, 0]  # Top, Left, Bottom, Right, Top
        for i in range(len(indices) - 1):
            pt1 = points[indices[i]]
            pt2 = points[indices[i + 1]]
            
            # Chỉ vẽ đường nếu cả 2 điểm đều visible
            if pt1[2] and pt2[2]:
                cv2.line(frame, (pt1[0], pt1[1]), (pt2[0], pt2[1]), COLORS["line"], 2)


def run_inference(
    model_path: str,
    image_path: str,
    conf_threshold: float = CONFIDENCE_THRESHOLD,
    save_output: bool = True,
    output_dir: str = "output",
    imgsz: int = 320,
    task: str = "pose",
) -> None:
    """
    Chạy inference và visualize kết quả.

    Args:
        model_path: Đường dẫn đến best.pt
        image_path: Đường dẫn ảnh test
        conf_threshold: Ngưỡng confidence tối thiểu
        save_output: Lưu ảnh kết quả
        output_dir: Thư mục lưu ảnh output
        imgsz: Kích thước input model
        task: ``pose`` | ``obb`` | ``seg``
    """
    # Load model
    print(f"[INFO] Loading model from: {model_path}")
    model = YOLO(model_path)

    # Load ảnh
    print(f"[INFO] Loading image: {image_path}")
    frame = cv2.imread(image_path)

    if frame is None:
        raise ValueError(f"[ERROR] Cannot load image: {image_path}")

    original_h, original_w = frame.shape[:2]

    # Chạy inference
    print(f"[INFO] Running inference (task={task})...")
    results = model.predict(
        image_path,
        conf=conf_threshold,
        imgsz=imgsz,
        verbose=False,
    )

    # Tạo thư mục output nếu cần
    if save_output:
        os.makedirs(output_dir, exist_ok=True)

    # Xử lý kết quả
    result = results[0]
    boxes = result.boxes
    boxes_len = 0 if boxes is None else len(boxes)
    print(f"[INFO] Detections: {boxes_len}")

    if boxes_len > 0:
        if task == "obb" and hasattr(result, "obb") and result.obb is not None:
            for i, (box, obb) in enumerate(zip(boxes, result.obb)):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy()) if hasattr(box, "cls") else 0
                cls_name = result.names.get(cls_id, f"class_{cls_id}") if hasattr(result, "names") else f"class_{cls_id}"

                # Draw the conventional axis-aligned bbox first for context.
                draw_bbox(frame, (x1, y1, x2, y2), conf)

                # Draw the oriented rectangle on top.
                corners = obb.xyxyxyxy.cpu().numpy().reshape(-1, 2)
                xywhr = obb.xywhr.cpu().numpy().reshape(-1)
                print(f"[DETECTION {i + 1}] cls={cls_name} conf={conf:.3f} "
                      f"angle={math.degrees(float(xywhr[4])):.2f}deg "
                      f"size=({float(xywhr[2]):.1f}, {float(xywhr[3]):.1f})")
                draw_obb(frame, corners, xywhr, cls_id, cls_name, conf)
        elif task == "pose" and result.keypoints is not None:
            for i, (box, kp) in enumerate(zip(boxes, result.keypoints)):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                print(f"[DETECTION {i + 1}]")
                print(f"  - Bounding Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                print(f"  - Confidence:   {conf:.3f}")
                draw_bbox(frame, (x1, y1, x2, y2), conf)
                kp_data = kp.data[0].cpu().numpy()  # Shape: [4, 3]
                print(f"  - Keypoints:")
                for j, (name, kp_vals) in enumerate(zip(KEYPOINT_NAMES, kp_data)):
                    x, y, visible = kp_vals
                    vis_status = "visible" if visible >= VISIBILITY_THRESHOLD else "occluded"
                    print(f"    [{j}] {name}: x={x:.3f}, y={y:.3f}, visible={visible:.3f} ({vis_status})")
                draw_keypoints(frame, kp_data, (x1, y1, x2, y2))
                info_text = f"Top=({kp_data[0][0]:.2f}, {kp_data[0][1]:.2f})"
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # Legacy / Seg path: draw bbox only (seg mask was the previous default).
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                print(f"[DETECTION {i + 1}]")
                print(f"  - Bounding Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                print(f"  - Confidence:   {conf:.3f}")
                draw_bbox(frame, (x1, y1, x2, y2), conf)
    else:
        print("[WARNING] No detections found in the image!")
        cv2.putText(
            frame, "No nail detected",
            (frame.shape[1] // 2 - 100, frame.shape[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
        )

    # Lưu ảnh kết quả
    if save_output:
        # Tạo tên file output
        base_name = os.path.basename(image_path)
        name_without_ext = os.path.splitext(base_name)[0]
        output_path = os.path.join(output_dir, f"{name_without_ext}_{task}_result.jpg")

        cv2.imwrite(output_path, frame)
        print(f"\n[SUCCESS] Result saved to: {os.path.abspath(output_path)}")
    
    # Hiển thị ảnh (nếu có display)
    # cv2.imshow("Nail Detection Result", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def main():
    """
    Main function - Run inference on test image.
    """
    args = parse_args()

    print("="*60)
    print(" NAIL KEYPOINT DETECTION - Visual Debug Tool")
    print("="*60)

    try:
        # Tìm model
        if args.model:
            model_path = args.model if os.path.isfile(args.model) else find_best_model(args.project)
        else:
            model_path = find_best_model(args.project)

        # Tìm ảnh test
        if args.image:
            if os.path.isfile(args.image):
                image_path = args.image
            else:
                print(f"[ERROR] Image not found: {args.image}")
                sys.exit(1)
        else:
            # Thử nhiều đường dẫn để tìm ảnh test
            workspace_root = Path(__file__).parent.resolve()
            test_paths = [
                workspace_root / "test_image.jpg",
                workspace_root / "test_image.png",
                workspace_root / "data",
                workspace_root.parent / "test_image.jpg",
            ]

            image_path = None
            for path in test_paths:
                if path.isfile():
                    image_path = str(path)
                    break
                elif path.is_dir():
                    try:
                        image_path = find_test_image(str(path))
                        break
                    except:
                        continue

            if image_path is None:
                print("[ERROR] No test image found!")
                print("[HINT] Place a test image or use --image <path>")
                sys.exit(1)

        # Chạy inference
        run_inference(
            model_path=model_path,
            image_path=image_path,
            conf_threshold=args.conf,
            save_output=not args.no_save,
            output_dir=args.output,
            imgsz=args.imgsz,
            task=args.task,
        )

        print("\n" + "=" * 60)
        print(" VISUAL INSPECTION CHECKLIST")
        print("=" * 60)
        if args.task == "obb":
            print(" [ ] Oriented rectangle bao quanh móng đúng hướng?")
            print(" [ ] Rotation arrow (yellow) chỉ theo chiều móng?")
            print(" [ ] Góc angle_deg hợp lý (-90 .. 90)?")
            print(" [ ] 4 corners của OBB không bị sort ngược (so với minAreaRect)?")
        elif args.task == "pose":
            print(" [ ] Bounding Box bao quanh móng tay chính xác?")
            print(" [ ] 4 Keypoints (Top, Bottom, Left, Right) đúng vị trí?")
            print(" [ ] Keypoints không bị lệch khỏi móng?")
            print(" [ ] Visibility của các điểm đúng (visible vs occluded)?")
        else:
            print(" [ ] Bounding Box bao quanh móng đúng vị trí?")
            print(" [ ] Mask polygon khớp biên móng?")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\n{str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
