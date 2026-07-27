# Nail AI Workspace (YOLO11 - Segmentation / Pose)

Workspace chứa toàn bộ script training & export mô hình YOLO11 dùng cho
**Tryon-Nail Detector** (phát hiện & phân vùng móng tay, 5 classes: index /
middle / pinky / ring / thumb).

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
├── train.py                      # Train YOLO11-Seg (interactive, tự quét GPU & dataset)
├── train_seg.py                  # Train YOLO11-Seg (CLI mode, dễ chạy trên server)
├── train_pose.py                 # Train YOLO11-Pose (4 keypoints: top/bottom/left/right)
├── train_finetune_5class.py      # Fine-tune từ best.pt cũ với 5 classes (cls=2.0, fliplr=0)
├── train_from_scratch_5class.py  # Train từ đầu với 5 classes
│
├── export_model.py               # Export best.pt -> TFLite (legacy, dùng cho Pose)
├── export_all.py                 # Export ONNX + TFLite (khuyến nghị dùng)
├── export_tflite.py              # Export ONNX (LiteRT cố gắng thêm nếu Linux/macOS)
├── test_inference.py             # Test model trên ảnh, vẽ bbox + keypoint
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

### 4.1. Train YOLO11-Seg (interactive, 1 class)

```bash
python train.py
```

Script sẽ hỏi: chọn dataset → chọn epochs/imgsz/batch → chọn GPU → train.
Phù hợp khi chạy trên máy local, muốn tương tác.

### 4.2. Train YOLO11-Seg (CLI, dễ chạy trên server)

```bash
python train_seg.py --epochs 100 --imgsz 416 --batch 16 --device 0
```

### 4.3. Train YOLO11-Pose (4 keypoints)

```bash
python train_pose.py --epochs 100 --imgsz 416 --batch 16
```

### 4.4. Fine-tune 5 classes (sửa classification từ run cũ)

```bash
python train_finetune_5class.py
```

Mặc định sẽ tự tìm `best.pt` mới nhất trong `runs/`. Để chỉ định:

```bash
python train_finetune_5class.py --base runs/<ten_run>/weights/best.pt
```

Hyperparameter đã được tinh chỉnh cho 5 ngón tay:
- `cls=2.0` — tăng weight class loss
- `fliplr=0.0` — tắt flip ngang (tránh đảo thumb↔pinky)
- `degrees=30` — xoay đa hướng
- `mosaic=0.8`, `mixup=0.1` — augmentation cân bằng

### 4.5. Train 5 classes từ đầu

```bash
python train_from_scratch_5class.py --epochs 150 --imgsz 640
```

---

## 5. Export model cho mobile / desktop

### 5.1. Export ONNX + TFLite (khuyến nghị)

```bash
python export_all.py
# Hoặc chỉ định run cụ thể:
python export_all.py --run nail-segmentation_v1i_yolov11_10501_20260726_194607
# Pose model:
python export_all.py --pose --kpt-shape 4 3
```

Output nằm ở `mobile_model/<ten_run>/`:
- `nail_seg.onnx` / `nail_pose.onnx` — cho Desktop + ONNX Runtime
- `nail_seg_float16.tflite` / `nail_pose_float16.tflite` — cho mobile (Linux/macOS)

> **Trên Windows:** LiteRT/TFLite sẽ tự fallback sang pipeline ONNX → onnx2tf
> (xem `TFLITE_WINDOWS.md`). Nếu chỉ cần ONNX, dùng `--no-tflite`.

### 5.2. Test inference nhanh

```bash
python test_inference.py --model runs/<ten_run>/weights/best.pt --image <path/to/test.jpg>
```

Sẽ vẽ bbox + 4 keypoint, lưu ảnh output vào `output/`.

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
