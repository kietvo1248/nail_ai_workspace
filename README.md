# Nail AI Workspace (YOLO11 - Seg / Pose / OBB)

Workspace chứa toàn bộ script training & export mô hình YOLO11 dùng cho
**Tryon-Nail Detector** (phát hiện & phân vùng móng tay, 5 classes: index /
middle / pinky / ring / thumb).

> **Pipeline OBB (khuyến nghị cho AR nail try-on):** Mô hình mới dùng
> **YOLO11-OBB** (Oriented Bounding Box) để đo được **góc xoay chính xác**
> của móng. Khi train xong, ứng dụng desktop sẽ dùng góc này để xoay móng
> fake theo hướng ngón tay thật — thay vì dùng `cv2.minAreaRect` (dễ sai).

> **Lưu ý:** Repo này **CHỈ chứa source code**. Các thư lớn như
> `venv311/`, `runs/`, `mobile_model/`, file `*.pt` / `*.onnx` / `*.tflite`
> đã được `.gitignore` loại trừ để repo nhẹ. Khi clone về, chỉ cần chạy
> `python setup_env.py` là mọi thứ tự khởi tạo.

---

## 1. Cấu trúc workspace

```
nail_ai_workspace/
├── setup_env.py                  # Auto tạo venv311 + cài requirements.txt
├── requirements.txt              # Danh sách thư viện cần thiết
├── .gitignore                    # Loại trừ file lớn (venv, runs, *.pt,...)
├── README.md                     # File này
│
├── data.yaml                     # Config dataset (1 class: nail, dùng cho detect/seg)
├── data_Nail_v1i_yolov11.yaml    # Config dataset 5 classes (dùng cho fine-tune 5 ngón)
│
├── train.py                      # Menu training (Seg / OBB / Pose / 5-class / Resume)
├── train_seg.py                  # Train YOLO11-Seg (CLI mode, dễ chạy trên server)
├── train_pose.py                 # Train YOLO11-Pose (4 keypoints: top/bottom/left/right)
├── train_finetune_5class.py      # Fine-tune từ best.pt cũ với 5 classes (cls=2.0, fliplr=0)
├── train_from_scratch_5class.py  # Train từ đầu với 5 classes
├── train_obb.py                  # Train YOLO11-OBB (1 class) - khuyến nghị cho AR
├── train_obb_5class.py           # Train YOLO11-OBB 5-class (label ngón tay)
│
├── convert_seg_to_obb.py         # Convert YOLO-Seg polygon -> YOLO-OBB 4 corners
├── auto_split.py                 # Tự tách train/val 80/20 nếu dataset chỉ có train/
│
├── export_model.py               # Export best.pt -> TFLite (legacy, dùng cho Pose)
├── export_all.py                 # Export ONNX + TFLite (khuyến nghị dùng, có --obb)
├── export_tflite.py              # Export ONNX (LiteRT cố gắng thêm nếu Linux/macOS)
├── test_inference.py             # Test model trên ảnh (pose / obb / seg), vẽ bbox + keypoint/OBB
│
├── TFLITE_WINDOWS.md             # Ghi chú khi export TFLite trên Windows
│
├── runs/                         # (KHÔNG push lên git) Output training: weights/, plots, csv
├── mobile_model/                 # (KHÔNG push lên git) ONNX/TFLite đã export
└── venv311/                      # (KHÔNG push lên git) Virtual env Python 3.11
```

> Dataset (`Nail.v1i.yolov11*`) thường đặt **NGOÀI workspace**, ngang hàng
> với thư mục này (xem phần [3. Chuẩn bị dataset](#3-chuẩn-bị-dataset)).

---

## 2. Hướng dẫn cho người mới clone về

### 2.1. Yêu cầu hệ thống

| Thành phần | Phiên bản | Ghi chú |
|---|---|---|
| OS | Windows 10/11, Linux, macOS | Mọi OS đều OK |
| Python | **3.11 hoặc 3.12** | TensorFlow **không hỗ trợ** 3.13+ |
| GPU (khuyến nghị) | NVIDIA ≥ 4GB VRAM | Train 5-class chạy được trên RTX 3050 |
| RAM | ≥ 8GB | |
| Disk | ≥ 5GB trống | Sau khi cài venv311 ~3GB + dataset |

Nếu chưa có Python 3.11:

- Tải từ: <https://www.python.org/downloads/release/python-3119/>
- Hoặc dùng `pyenv` / `conda`:
  ```bash
  conda create -n nail_ai python=3.11 -y
  ```

### 2.2. Cài đặt (một lần duy nhất)

```bash
cd nail_ai_workspace
python setup_env.py
```

Script sẽ **tự động**:

1. Tìm Python 3.11/3.12 trong hệ thống.
2. Tạo `venv311/` (nếu chưa có).
3. Cài đặt tất cả package từ `requirements.txt`.
4. In hướng dẫn bước tiếp theo.

Các flag tùy chọn:

```bash
python setup_env.py --recreate     # Xóa venv311 cũ, tạo mới
python setup_env.py --no-venv      # Cài thẳng vào Python hiện tại (không khuyến nghị)
python setup_env.py --python 3.12  # Ép dùng Python 3.12
```

### 2.3. Kích hoạt môi trường (mỗi lần mở terminal mới)

**Windows (PowerShell):**
```powershell
.\venv311\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
venv311\Scripts\activate.bat
```

**Linux / macOS:**
```bash
source venv311/bin/activate
```

> Nếu PowerShell báo lỗi "running scripts is disabled", chạy:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

---

## 3. Chuẩn bị dataset

Đặt dataset ở **parent folder** của workspace (cùng cấp với `nail_ai_workspace/`).
Cấu trúc chuẩn Roboflow:

```
Tryon-Nail Detector (YOLO)/
├── nail_ai_workspace/        ← workspace này
├── Nail.v1i.yolov11/         ← dataset 1 class
│   ├── data.yaml
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
└── nail-segmentation.v1i.yolov11_10501/   ← dataset 5 classes
    └── data.yaml
```

Nếu đặt dataset chỗ khác, truyền `--data <path/data.yaml>` khi train.

---

## 4. Workflow training

### 4.0. Quick-start (khuyến nghị — YOLO11-OBB cho AR nail try-on)

Mặc định `python train.py` sẽ hiện **menu 6 chế độ**:

```
[1] Train Seg (legacy, 1-class)
[2] Train OBB (smoke test)        ← 139 ảnh, nhanh, verify pipeline
[3] Train OBB (production)        ← 10,501 ảnh
[4] Train OBB 5-class             ← 5-class OBB cho label ngón
[5] Train Pose (legacy, 4 kp)
[6] Resume training               ← tự động phát hiện last.pt khi chạy
```

Các script OBB ([2]–[4]) tự động:
- Convert YOLO-Seg polygon → YOLO-OBB 4 corners (`convert_seg_to_obb.py`).
- Tách train/val 80/20 nếu dataset chỉ có `train/` (`auto_split.py`).

Nên dùng chế độ `[2]` để smoke test pipeline với 139 ảnh trước, sau đó
chuyển sang `[3]` cho production.

### 4.1. Train YOLO11-OBB (smoke test, 139 ảnh)

```bash
# Cách 1: dùng menu
python train.py            # rồi chọn [2]

# Cách 2: CLI trực tiếp
python train_obb.py --data ../Nail_Detection_ThanhDT.v1i.yolov11/data.yaml --epochs 20 --imgsz 416
```

Augmentation đã được tinh chỉnh cho OBB:

| Aug | Value | Lý do |
|---|---|---|
| `degrees` | **180** | Xoay đủ 360° - OBB model đã học invariance với rotation |
| `fliplr` | **0.0** | Tắt flip ngang - tránh thumb↔pinky confusion |
| `mosaic` | 1.0 | OBB model train mosaic tốt hơn Seg |
| `mixup` | 0.15 | Đa dạng ảnh |
| `copy_paste` | 0.3 | Augmentation copy ngón từ ảnh khác → đa dạng góc |
| `erasing` | 0.4 | Cutout chống overfit |
| `cls` | 2.0 | Tăng weight classification loss |

### 4.2. Train YOLO11-OBB (production, 10,501 ảnh)

```bash
python train.py            # rồi chọn [3]
# hoặc:
python train_obb.py --data ../nail-segmentation.v1i.yolov11_10501/data.yaml --epochs 150 --imgsz 640
```

### 4.3. Train YOLO11-OBB 5-class (label tên ngón)

```bash
python train.py            # rồi chọn [4]
# hoặc:
python train_obb_5class.py --data ../nail-segmentation.v1i.yolov11_10501/data.yaml --epochs 150 --imgsz 640
```

Cùng augmentation như 4.1, nhưng `fliplr=0.0` được hard-lock để tránh
nhầm ngón (thumb ↔ pinky khi flip ngang).

### 4.4. Train YOLO11-Seg (legacy)

```bash
python train.py            # rồi chọn [1] - interactive
python train_seg.py --epochs 100 --imgsz 416 --batch 16 --device 0   # CLI
```

Giữ lại để backward-compat với pipeline cũ (overlay cũ dùng `minAreaRect`).

### 4.5. Train YOLO11-Pose (4 keypoints, legacy)

```bash
python train.py            # rồi chọn [5]
python train_pose.py --epochs 100 --imgsz 416 --batch 16
```

Pose model là option dự phòng. Khuyến nghị dùng **OBB** thay vì Pose vì
OBB cho angle chính xác hơn (Pose 4 keypoints Top/Bottom/Left/Right rất
nhạy với nhiễu).

### 4.6. Fine-tune 5 classes từ run cũ

```bash
python train_finetune_5class.py --base runs/<ten_run>/weights/best.pt
```

Hyperparameter: `cls=2.0`, `fliplr=0.0`, `degrees=30`, `mosaic=0.8`.

### 4.7. Train 5 classes từ đầu

```bash
python train_from_scratch_5class.py --epochs 150 --imgsz 640
```

### 4.8. Resume training sau khi crash

Chạy lại `python train.py` - script tự phát hiện `last.pt` và hỏi resume.

---

## 5. Export model cho mobile / desktop

### 5.1. Export ONNX + TFLite (khuyến nghị)

```bash
python export_all.py
# Hoặc chỉ định run cụ thể:
python export_all.py --run nail-segmentation_v1i_yolov11_10501_20260726_194607
# Pose model:
python export_all.py --pose --kpt-shape 4 3
# OBB model (cho AR nail try-on):
python export_all.py --obb --run <ten_run_obb>
```

Output nằm ở `mobile_model/<ten_run>/`:
- `nail_seg.onnx` / `nail_pose.onnx` / `nail_obb.onnx` — cho Desktop + ONNX Runtime
- `nail_seg_float16.tflite` / `nail_pose_float16.tflite` — cho mobile (Linux/macOS)
- `nail_obb_float16.tflite` — OBB model cho mobile (Linux/macOS)

> **Trên Windows:** LiteRT/TFLite sẽ tự fallback sang pipeline ONNX → onnx2tf
> (xem `TFLITE_WINDOWS.md`). Nếu chỉ cần ONNX, dùng `--no-tflite`.

### 5.2. Test inference nhanh

```bash
# Pose model:
python test_inference.py --model runs/<ten_run>/weights/best.pt --task pose --image <path/to/test.jpg>
# OBB model:
python test_inference.py --model runs/<ten_run>/weights/best.pt --task obb --image <path/to/test.jpg>
# Seg model:
python test_inference.py --model runs/<ten_run>/weights/best.pt --task seg --image <path/to/test.jpg>
```

Sẽ vẽ:
- **pose**: bbox + 4 keypoint (Top/Bottom/Left/Right), lưu vào `output/`.
- **obb**:  oriented rectangle (4 corners) + rotation arrow từ bbox center
  theo góc `angle_deg` - kiểm tra được model đoán góc đúng chưa.
- **seg**:  bbox + mask polygon (legacy).

Output file: `output/<image>_<task>_result.jpg`.

---

## 6. Cấu hình đề xuất theo GPU

| GPU | VRAM | imgsz | batch | epochs | Ghi chú |
|---|---|---|---|---|---|
| RTX 3050 / 4050 | 4GB | 416 | 8–16 | 30–80 | Bình thường, không chạy 640/32 |
| RTX 3060 / 4060 | 8–12GB | 640 | 16–32 | 100–200 | Khuyến nghị |
| RTX 3080 / 4080 | 10–16GB | 640 | 32 | 200+ | Tốt cho từ scratch |
| RTX 3090 / 4090 | 24GB | 640 | 64 | 300+ | Full speed |
| CPU | — | 320 | 4 | 30 | **Rất chậm**, chỉ để test pipeline |
| Colab T4 | 16GB | 640 | 16 | 100 | OK |

**Công thức nhanh khi máy yếu (4GB VRAM):**
- `imgsz=320`, `batch=8` → chắc chắn chạy được
- `imgsz=416`, `batch=8` → khuyến nghị tối đa
- `imgsz=416`, `batch=16` → có thể OOM, cần `cache=False` + `workers=0`

**Triệu chứng hết RAM/GPU:**
- Bị `CUDA out of memory` → giảm batch, hoặc `cache=False`
- Bị `Killed` không log gì → CPU RAM hết, giảm `workers=2` hoặc `imgsz=320`
- Train chạy 1 epoch mất >30 phút → CPU mode, nên chuyển máy có GPU

---

## 7. Troubleshooting

### Lỗi `ModuleNotFoundError: ultralytics`
→ Quên activate venv, hoặc chưa chạy `python setup_env.py`.

### Lỗi `CUDA out of memory`
→ Giảm `batch` xuống 8 hoặc 4, hoặc dùng `imgsz=320`.

### Lỗi khi export TFLite trên Windows
→ Xem `TFLITE_WINDOWS.md`. Nếu chỉ cần ONNX, dùng `python export_all.py --no-tflite`.

### Train chậm / bị sập
- Kiểm tra GPU có đang được dùng không (script `train.py` có in `device`).
- Nếu máy không có GPU NVIDIA, training sẽ chạy trên CPU (rất chậm).
- Cân nhắc train trên Google Colab / Kaggle (free GPU) hoặc cloud GPU.

### `data.yaml` không tìm thấy
→ Đặt dataset ở parent folder, hoặc truyền `--data <đường dẫn>`.

### Resume training sau khi crash
→ Chạy lại `python train.py` — script tự phát hiện `last.pt` và hỏi resume.

### OBB model predict sai góc xoay (fake nail xoay bậy)
**Triệu chứng:** overlay xoay 180° ngược chiều, hoặc góc xoay bị giật.

**Nguyên nhân phổ biến:**
1. **Sai thứ tự 4 corners OBB trong label.** Ultralytics OBB dùng
   convention `[BR, TR, TL, BL]`. Nếu `convert_seg_to_obb.py` sort sai,
   model sẽ học góc nghịch đảo.
2. **Cls 0 (`Index` viết hoa)** trong smoke test dataset có 1 mẫu - đây
   là class rác từ Roboflow. Khi train production với dataset 5-class
   sạch (`nail-segmentation.v1i.yolov11_10501`), class này không còn.
3. **Augmentation xoay quá yếu** (`degrees < 90°`) - model chưa học
   invariance với rotation. Khuyến nghị `degrees=180` cho OBB.
4. **Top/Bottom ambiguity:** OBB 4 góc không phân biệt được "đầu móng"
   vs "gốc móng". `NailOverlayRenderer._paste_overlay` đã có heuristic
   fallback: nếu `|model_angle - minAreaRect_angle| > 90°` thì flip 180°.

**Cách debug:**
- Chạy `python test_inference.py --task obb --model runs/.../best.pt --image test.jpg`.
- Kiểm tra rotation arrow (yellow) có chỉ đúng chiều móng không.
- Nếu sai, kiểm tra lại `convert_seg_to_obb.py` có sort đúng `[BR, TR, TL, BL]` không.
- Kiểm tra ảnh `runs/<run>/train_batch*.jpg` để xem label OBB có đúng chiều sau augmentation không.

### So sánh augmentation Seg vs OBB

| Aug | Seg (legacy) | OBB (mới) | Lý do |
|---|---|---|---|
| `degrees` | 15–30 | **180** | OBB model học rotation invariance tốt hơn với xoay mạnh |
| `fliplr` | 0.5 | **0.0** | Tránh thumb↔pinky confusion khi flip ngang |
| `mosaic` | 0.8 | **1.0** | OBB model train mosaic tốt hơn Seg |
| `mixup` | 0.0–0.1 | **0.15** | Cân bằng dataset |
| `copy_paste` | — | **0.3** | Đa dạng góc xoay ngón tay |
| `erasing` | — | **0.4** | Cutout chống overfit pixel cụ thể |
| `cls` | 1.0 (mặc định) | **2.0** | Tăng weight classification loss |

---

## 8. Quy ước tên file & folder

| File / folder | Vai trò |
|---|---|
| `best.pt` | Weights tốt nhất (theo metric val) |
| `last.pt` | Weights của epoch cuối (để resume) |
| `args.yaml` | Hyperparameter đã dùng (cho run đó) |
| `results.csv` | Log loss/metric từng epoch |
| `results.png` | Đồ thị loss/metric |
| `confusion_matrix.png` | Confusion matrix trên tập val |
| `BoxPR_curve.png` | Precision-Recall curve |

---

## 9. License

Source code: nội bộ dự án **SEP490 - Tryon-Nail Detector**.
Model weights: tuân theo license của Ultralytics (AGPL-3.0 cho repo public,
Enterprise license cho dùng thương mại).
