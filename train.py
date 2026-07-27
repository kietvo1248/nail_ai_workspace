"""
================================================================================
TRAIN.PY - YOLOv11-Seg Training Script for Nail Segmentation
================================================================================
Script huấn luyện mô hình YOLOv11-Seg để phân vùng móng tay (nail segmentation).

**Tính năng:**
  - Interactive input: Hỏi vị trí dataset, validate, thực thi training
  - Multi-dataset support: Mỗi dataset có thư mục results riêng với timestamp
  - Tự động tạo train/val split nếu chưa có
  - Lưu kết quả có tổ chức: runs/<dataset_name>/<timestamp>/

**Sử dụng:**
  python train.py
  (Script sẽ hỏi từng bước qua terminal)
================================================================================
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

try:
    import yaml  # PyYAML - used to inspect / patch data.yaml
except ImportError:  # pragma: no cover
    yaml = None


# =============================================================================
# ANSI COLORS for Terminal
# =============================================================================
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(text):
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}{Colors.END}\n")


def print_success(text):
    print(f"{Colors.GREEN}[OK] {text}{Colors.END}")


def print_error(text):
    print(f"{Colors.RED}[ERROR] {text}{Colors.END}")


def print_warning(text):
    print(f"{Colors.YELLOW}[WARNING] {text}{Colors.END}")


def print_info(text):
    print(f"{Colors.BLUE}[INFO] {text}{Colors.END}")


def input_path(prompt_text):
    """Interactive input for path with color."""
    return input(f"{Colors.CYAN}{prompt_text}: {Colors.END}").strip()


def input_choice(prompt_text, options, default=None):
    """Interactive choice selection."""
    print(f"\n{Colors.YELLOW}{prompt_text}{Colors.END}")
    for i, opt in enumerate(options, 1):
        marker = " (default)" if opt == default else ""
        print(f"  {Colors.CYAN}{i}.{Colors.END} {opt}{marker}")

    while True:
        choice = input(f"\n{Colors.CYAN}Chọn (Enter = default): {Colors.END}").strip()
        if choice == "" and default:
            return default
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
            print_warning("Vui lòng chọn số trong danh sách!")
        except ValueError:
            print_warning("Vui lòng nhập số!")


# =============================================================================
# GPU DETECTION
# =============================================================================

def scan_gpus():
    """
    Scan and display all available GPUs, let user select one.
    Returns the selected device string ('0', '1', 'cpu').
    """
    try:
        import torch
    except ImportError:
        print_warning("torch chua duoc cai dat - khong the quet GPU")
        return "cpu"

    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}")
    print(f"  GPU DETECTION")
    print(f"{'='*60}{Colors.END}\n")

    if not torch.cuda.is_available():
        print_error("Khong co GPU nao duoc phat hien! Training se chay tren CPU.")
        print(f"  {Colors.YELLOW}Luu y: CPU rat cham, khuyen nghi cai dat GPU.{Colors.END}\n")
        return "cpu"

    gpu_count = torch.cuda.device_count()
    print(f"  {Colors.GREEN}Phat hien {gpu_count} GPU(s):{Colors.END}\n")

    gpu_list = []
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        total_mem_gb = props.total_memory / (1024**3)
        gpu_list.append({
            "index": i,
            "name": props.name,
            "total_mem_gb": total_mem_gb,
        })
        print(f"  {Colors.CYAN}[{i}]{Colors.END} {props.name}")
        print(f"       VRAM: {total_mem_gb:.1f} GB")
        print(f"       Compute: {props.major}.{props.minor}")
        print()

    # Neu chi co 1 GPU, dung no luon
    if gpu_count == 1:
        print_success(f"Tu dong chon GPU 0: {gpu_list[0]['name']}\n")
        return "0"

    # Neu co nhieu hon 1 GPU, hoi user chon
    print(f"{Colors.YELLOW}Co {gpu_count} GPU. Ban muon dung GPU nao?{Colors.END}")
    print(f"  {Colors.CYAN}[0-{gpu_count-1}]{Colors.END} Chon GPU theo so index")
    print(f"  {Colors.CYAN}[c]{Colors.END} CPU (khong khuyen khich)")

    while True:
        choice = input(f"\n{Colors.CYAN}Chon GPU (Enter = 0): {Colors.END}").strip()
        if choice == "":
            choice = "0"

        if choice.lower() == "c":
            print_warning("Ban da chon CPU - training se rat cham!")
            return "cpu"

        try:
            idx = int(choice)
            if 0 <= idx < gpu_count:
                gpu = gpu_list[idx]
                print_success(f"Dang su dung GPU {idx}: {gpu['name']} ({gpu['total_mem_gb']:.1f} GB)\n")
                return str(idx)
            else:
                print_warning(f"Chi so khong hop le. Vui long nhap 0-{gpu_count-1} hoac 'c'.")
        except ValueError:
            print_warning("Vui long nhap so (0-{}) hoac 'c'.".format(gpu_count - 1))


# =============================================================================
# UTILITIES
# =============================================================================

def get_python_executable():
    """
    Get the correct Python executable to use.
    Priority: venv > conda > current
    """
    # Check if we're in a venv
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        # We're in a venv
        if os.name == "nt":
            return Path(sys.prefix) / "Scripts" / "python.exe"
        else:
            return Path(sys.prefix) / "bin" / "python"

    # Check for venv in workspace
    workspace_root = Path(__file__).parent.resolve()
    venv_python = workspace_root / "venv" / "Scripts" / "python.exe"

    if venv_python.exists():
        print_info(f"Tìm thấy virtual environment: {venv_python}")
        return str(venv_python)

    # Check for conda
    conda_python = os.environ.get("CONDA_PYTHON_EXE")
    if conda_python:
        print_info(f"Sử dụng Conda Python: {conda_python}")
        return conda_python

    # Fall back to current Python
    print_warning(f"Sử dụng Python mặc định: {sys.executable}")
    return sys.executable


def check_ultralytics_installed(python_path):
    """Check if ultralytics is installed for the given Python."""
    try:
        result = subprocess.run(
            [python_path, "-c", "import ultralytics; print('OK')"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0 and "OK" in result.stdout
    except Exception:
        return False


def ensure_dependencies(python_path):
    """Ensure required dependencies are installed."""
    print_info("Kiểm tra dependencies...")

    if not check_ultralytics_installed(python_path):
        print_warning("ultralytics chưa được cài đặt!")

        # Try to install
        print_info("Đang cài đặt ultralytics...")
        try:
            subprocess.run([python_path, "-m", "pip", "install", "ultralytics"], check=True)
            print_success("ultralytics đã được cài đặt!")
        except Exception as e:
            print_error(f"Không thể cài đặt ultralytics: {e}")
            print_info("Vui lòng chạy lệnh sau để cài đặt thủ công:")
            print(f"  {python_path} -m pip install ultralytics")
            return False

    return True
# =============================================================================
def validate_dataset(dataset_path):
    """
    Validate dataset structure.

    Returns:
        dict: Validation result with details
    """
    result = {
        "valid": False,
        "train_images": None,
        "train_labels": None,
        "valid_images": None,
        "valid_labels": None,
        "total_images": 0,
        "format": None,
        "errors": []
    }

    # Resolve path
    dataset_path = Path(dataset_path).resolve()

    if not dataset_path.exists():
        result["errors"].append(f"Dataset path không tồn tại: {dataset_path}")
        return result

    # Check for Roboflow format (train/images, train/labels)
    train_images = dataset_path / "train" / "images"
    train_labels = dataset_path / "train" / "labels"

    if train_images.exists() and train_labels.exists():
        result["format"] = "roboflow"
        result["train_images"] = train_images
        result["train_labels"] = train_labels

        # Count images
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        image_count = 0
        for ext in extensions:
            image_count += len(list(train_images.glob(ext)))

        result["total_images"] = image_count

        # Check validation folder
        valid_images = dataset_path / "valid" / "images"
        valid_labels = dataset_path / "valid" / "labels"

        if valid_images.exists() and valid_labels.exists():
            result["valid_images"] = valid_images
            result["valid_labels"] = valid_labels
            val_count = sum(len(list(valid_images.glob(ext))) for ext in extensions)
            result["total_images"] += val_count
            print_success(f"Validation folder tồn tại ({val_count} ảnh)")
        else:
            print_warning("Không tìm thấy validation folder - sẽ tự động chia")

        result["valid"] = True
        return result

    # Check for YOLO format (images/train, labels/train)
    images_dir = dataset_path / "images" / "train"
    labels_dir = dataset_path / "labels" / "train"

    if images_dir.exists() and labels_dir.exists():
        result["format"] = "yolo"
        result["train_images"] = images_dir
        result["train_labels"] = labels_dir

        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        image_count = 0
        for ext in extensions:
            image_count += len(list(images_dir.glob(ext)))

        result["total_images"] = image_count
        result["valid"] = True
        return result

    # Check for data.yaml
    data_yaml = dataset_path / "data.yaml"
    if data_yaml.exists():
        result["format"] = "custom"
        print_info(f"Found data.yaml - dataset có thể đã được cấu hình")
        result["valid"] = True
        return result

    # Not valid
    result["errors"].append("Dataset không đúng format. Cần có cấu trúc:")
    result["errors"].append("  - Format Roboflow: train/images/, train/labels/")
    result["errors"].append("  - Format YOLO: images/train/, labels/train/")
    result["errors"].append("  - Hoặc có file data.yaml")

    return result


def split_dataset_auto(dataset_path, val_ratio=0.2):
    """
    Automatically split dataset into train/val (80/20).

    Args:
        dataset_path: Path to dataset root
        val_ratio: Validation ratio (default 0.2 = 20%)
    """
    import random

    dataset_path = Path(dataset_path)
    train_images = dataset_path / "train" / "images"
    train_labels = dataset_path / "train" / "labels"

    # Check if already split
    if (dataset_path / "valid").exists():
        print_success("Dataset đã được chia train/val")
        return

    print_info(f"Đang chia dataset (train={int((1-val_ratio)*100)}%, val={int(val_ratio*100)}%)...")

    # Get all images
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    train_files = []
    for ext in extensions:
        train_files.extend(list(train_images.glob(ext)))

    if len(train_files) == 0:
        raise ValueError("Không tìm thấy ảnh trong train/images!")

    # Shuffle and split
    random.seed(42)
    random.shuffle(train_files)

    val_count = max(1, int(len(train_files) * val_ratio))
    val_files = train_files[:val_count]

    # Create validation folders
    valid_images = dataset_path / "valid" / "images"
    valid_labels = dataset_path / "valid" / "labels"
    valid_images.mkdir(parents=True, exist_ok=True)
    valid_labels.mkdir(parents=True, exist_ok=True)

    # Move files with simple progress
    moved = 0
    for img_file in val_files:
        # Move image
        shutil.move(str(img_file), str(valid_images / img_file.name))

        # Move label
        label_file = train_labels / (img_file.stem + ".txt")
        if label_file.exists():
            shutil.move(str(label_file), str(valid_labels / label_file.name))

        moved += 1
        if moved % 10 == 0 or moved == len(val_files):
            percent = int(moved / len(val_files) * 100)
            print(f"\r    Đang di chuyển: {moved}/{len(val_files)} ({percent}%)", end="", flush=True)

    print()  # New line after progress

    print_success(f"Đã chia xong: {len(train_files) - val_count} train, {val_count} val")


def create_data_yaml(dataset_path, format_type="roboflow"):
    """
    Create data.yaml for dataset.

    Args:
        dataset_path: Path to dataset root
        format_type: Dataset format (roboflow/yolo)
    """
    dataset_path = Path(dataset_path)

    if format_type == "roboflow":
        train_path = "train/images"
        val_path = "valid/images"
    else:
        train_path = "images/train"
        val_path = "images/val"

    yaml_content = f"""# =============================================================================
# DATA CONFIGURATION - Auto-generated by train.py (YOLO-Seg)
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# =============================================================================

# Dataset root path
path: {dataset_path.absolute()}

# Train and validation paths
train: {train_path}
val: {val_path}

# =============================================================================
# Number of classes
nc: 1
names:
  0: nail
"""

    yaml_path = dataset_path / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print_success(f"Đã tạo data.yaml tại: {yaml_path}")
    return yaml_path


def ensure_seg_data_yaml(dataset_data_yaml: Path, workspace_root: Path) -> Path:
    """Return a data.yaml path suitable for YOLO Seg training.

    Resolution order:
        1. If the workspace already has a ``data.yaml`` whose ``path`` matches the
           dataset root and that file does not contain ``kpt_shape``, prefer it.
        2. If the dataset's own ``data.yaml`` is already clean (no ``kpt_shape``),
           use it as-is.
        3. Otherwise, strip any ``kpt_shape`` / ``flip_idx`` fields and write a
           workspace-local patched copy. The original dataset file is never modified.
    """
    dataset_data_yaml = Path(dataset_data_yaml)
    if not dataset_data_yaml.exists():
        return dataset_data_yaml  # Caller will surface the missing-file error.

    if yaml is None:
        print_warning(
            "PyYAML not installed - cannot verify Seg data.yaml. "
            "Install with: pip install PyYAML"
        )
        return dataset_data_yaml

    def _load(p: Path) -> dict:
        try:
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    # --- Step 1: workspace's own data.yaml (when it points to the same dataset)
    workspace_yaml = workspace_root / "data.yaml"
    ws_data = _load(workspace_yaml) if workspace_yaml.exists() else {}
    if "kpt_shape" not in ws_data and ws_data.get("path"):
        ws_path = Path(ws_data["path"])
        try:
            if ws_path.resolve() == dataset_data_yaml.parent.resolve():
                print_success(f"Using workspace data.yaml: {workspace_yaml}")
                return workspace_yaml
        except Exception:
            pass

    # --- Step 2: dataset's data.yaml already clean
    data = _load(dataset_data_yaml)
    if "kpt_shape" not in data:
        return dataset_data_yaml

    # --- Step 3: strip kpt fields + write workspace-local copy
    print_warning(
        f"data.yaml chứa 'kpt_shape' - đang tạo bản patch Seg trong workspace."
    )
    data.pop("kpt_shape", None)
    data.pop("flip_idx", None)

    # CRITICAL: Set ``path`` to the dataset's absolute root so that
    # ``train`` / ``val`` / ``test`` relative paths are resolved correctly
    # regardless of where the patched YAML file lives on disk.
    data["path"] = str(dataset_data_yaml.parent.resolve())

    dataset_stem = dataset_data_yaml.parent.name.replace(" ", "_").replace(".", "_")
    patched_path = workspace_root / f"data_{dataset_stem}_seg.yaml"
    with open(patched_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print_success(f"Đã tạo patched Seg data.yaml tại: {patched_path}")
    print_info(f"path set to: {data['path']}")
    print_info("Bản gốc trong dataset không bị thay đổi.")
    return patched_path


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
def get_training_config(dataset_name):
    """
    Get training configuration interactively.

    Returns:
        dict: Training configuration
    """
    config = {}

    print_header("CẤU HÌNH TRAINING")

    # Epochs
    print("\nChọn số epochs:")
    print(f"  {Colors.CYAN}1.{Colors.END} 50 epochs (nhanh, demo)")
    print(f"  {Colors.CYAN}2.{Colors.END} 100 epochs (mặc định)")
    print(f"  {Colors.CYAN}3.{Colors.END} 200 epochs (chất lượng cao)")
    print(f"  {Colors.CYAN}4.{Colors.END} Tự nhập")

    choice = input(f"\n{Colors.CYAN}Chọn (1-4): {Colors.END}").strip()
    epoch_map = {"1": 50, "2": 100, "3": 200}

    if choice in epoch_map:
        config["epochs"] = epoch_map[choice]
    elif choice == "4":
        config["epochs"] = int(input("Nhập số epochs: ").strip())
    else:
        config["epochs"] = 100

    print_success(f"Epochs: {config['epochs']}")

    # Image size
    print("\nChọn kích thước ảnh:")
    print(f"  {Colors.CYAN}1.{Colors.END} 320 (nhanh, mobile)")
    print(f"  {Colors.CYAN}2.{Colors.END} 416 (cân bằng)")
    print(f"  {Colors.CYAN}3.{Colors.END} 640 (chất lượng cao)")

    choice = input(f"\n{Colors.CYAN}Chọn (1-3): {Colors.END}").strip()
    size_map = {"1": 320, "2": 416, "3": 640}

    config["imgsz"] = size_map.get(choice, 320)
    print_success(f"Image size: {config['imgsz']}")

    # Batch size
    print("\nChọn batch size:")
    print(f"  {Colors.CYAN}1.{Colors.END} 8 (ít RAM)")
    print(f"  {Colors.CYAN}2.{Colors.END} 16 (mặc định)")
    print(f"  {Colors.CYAN}3.{Colors.END} 32 (nhiều RAM)")

    choice = input(f"\n{Colors.CYAN}Chọn (1-3): {Colors.END}").strip()
    batch_map = {"1": 8, "2": 16, "3": 32}

    config["batch"] = batch_map.get(choice, 16)
    print_success(f"Batch size: {config['batch']}")

    # Generate run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["run_name"] = f"{dataset_name}_{timestamp}"

    return config


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    print_header("YOLOV11-SEG NAIL SEGMENTATION - TRAINING")
    print(f"{Colors.YELLOW}Script huấn luyện mô hình với interactive prompts{Colors.END}\n")

    workspace_root = Path(__file__).parent.resolve()
    runs_dir = workspace_root / "runs"

    # --- KHÔI PHỤC TRAINING (RESUME) TỰ ĐỘNG ---
    if runs_dir.exists():
        recent_lasts = list(runs_dir.glob("*/weights/last.pt"))
        if recent_lasts:
            recent_lasts.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest_last_pt = recent_lasts[0]
            print_header("TÌM THẤY CHECKPOINT CHƯA HOÀN THÀNH")
            print(f"{Colors.YELLOW}Phát hiện quá trình training có thể đã bị gián đoạn (mất điện).{Colors.END}")
            print(f"  {Colors.YELLOW}Checkpoint gần nhất:{Colors.END}")
            try:
                ckpt_disp = latest_last_pt.relative_to(workspace_root)
            except ValueError:
                ckpt_disp = latest_last_pt
            print(f"  {Colors.CYAN}{ckpt_disp}{Colors.END}")
            
            choice = input(f"\n{Colors.YELLOW}Bạn có muốn tiếp tục (resume) quá trình train này không? (y/n, Enter để bỏ qua): {Colors.END}").strip().lower()
            if choice in ['y', 'yes']:
                python_exec = get_python_executable()
                if not ensure_dependencies(python_exec):
                    sys.exit(1)
                    
                print_header("BẮT ĐẦU PHỤC HỒI TRAINING")
                try:
                    from ultralytics import YOLO
                    print_info(f"Đang tải bản lưu trạng thái từ: {latest_last_pt.name}")
                    model = YOLO(str(latest_last_pt))
                    # resume=True sẽ tự động lấy mọi cấu hình cũ (data.yaml, device, epochs...)
                    model.train(resume=True)
                    
                    print_header("TRAINING HOÀN TẤT!")
                    run_dir = latest_last_pt.parent.parent
                    best_model = run_dir / "weights" / "best.pt"
                    print_success("Quá trình huấn luyện phục hồi đã kết thúc.")
                    print(f"\n{Colors.GREEN}KẾT QUẢ:{Colors.END}")
                    print(f"  Run directory: {run_dir}")
                    if best_model.exists():
                        print(f"  Best model:   {best_model}")
                    sys.exit(0)
                except KeyboardInterrupt:
                    print("\n")
                    print_warning("Training đã bị dừng bởi user.")
                    sys.exit(0)
                except Exception as e:
                    print_error(f"Lỗi khi resume: {e}")
                    import traceback
                    traceback.print_exc()
                    sys.exit(1)

    # Step 0: Scan GPUs
    print_header("BƯỚC 0: QUÉT GPU")
    selected_device = scan_gpus()

    # Step 1: Get dataset path
    print_header("BƯỚC 1: CHỌN DATASET")

    # Try to find existing datasets
    workspace_root = Path(__file__).parent.resolve()
    parent_root = workspace_root.parent

    # Scan for potential datasets
    potential_datasets = []

    # Scan workspace root
    for item in parent_root.iterdir():
        if item.is_dir():
            # Check if it looks like a dataset
            if (item / "train" / "images").exists() or (item / "images").exists():
                potential_datasets.append({
                    "name": item.name,
                    "path": str(item),
                    "type": "folder"
                })

    # Add manual input option
    potential_datasets.append({
        "name": "Nhập đường dẫn khác...",
        "path": "manual",
        "type": "manual"
    })

    if len(potential_datasets) > 1:
        print(f"{Colors.YELLOW}Tìm thấy các dataset trong workspace:{Colors.END}\n")
        for i, ds in enumerate(potential_datasets[:-1], 1):
            print(f"  {Colors.CYAN}{i}.{Colors.END} {ds['name']}")
            print(f"      {Colors.BLUE}→{Colors.END} {ds['path']}")

        choice = input(f"\n{Colors.CYAN}Chọn dataset (số) hoặc Enter để nhập đường dẫn: {Colors.END}").strip()

        if choice and choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(potential_datasets) - 1:
                dataset_path = potential_datasets[idx]["path"]
            elif idx == len(potential_datasets) - 1:
                dataset_path = None
            else:
                dataset_path = None
        else:
            dataset_path = None
    else:
        dataset_path = None

    # Manual input if not selected
    while not dataset_path or not Path(dataset_path).exists():
        dataset_path = input_path("Nhập đường dẫn đến dataset (hoặc Enter để quét workspace)")

        if not dataset_path:
            # Scan workspace
            if potential_datasets:
                dataset_path = potential_datasets[0]["path"]
            else:
                print_error("Không tìm thấy dataset nào trong workspace!")
                dataset_path = input_path("Nhập đường dẫn dataset: ")

        if Path(dataset_path).exists():
            break

        print_error(f"Đường dẫn không tồn tại: {dataset_path}")
        dataset_path = None

    dataset_path = Path(dataset_path).resolve()
    print_success(f"Dataset: {dataset_path}")

    # Step 2: Validate dataset
    print_header("BƯỚC 2: VALIDATE DATASET")

    validation = validate_dataset(dataset_path)

    if not validation["valid"]:
        print_error("Dataset không hợp lệ!")
        for err in validation["errors"]:
            print(f"  {Colors.RED}•{Colors.END} {err}")
        sys.exit(1)

    print_success(f"Dataset hợp lệ!")
    print(f"  Format: {validation['format']}")
    print(f"  Tổng ảnh: {validation['total_images']}")

    if not validation["valid_images"]:
        print_warning("Dataset chưa có validation folder")

        choice = input(f"\n{Colors.YELLOW}Chia train/val tự động (80/20)? (y/n): {Colors.END}").strip().lower()
        if choice in ["y", "yes", ""]:
            try:
                split_dataset_auto(dataset_path, val_ratio=0.2)
            except Exception as e:
                print_error(f"Lỗi khi chia dataset: {e}")
                sys.exit(1)

    # Step 3: Create data.yaml
    print_header("BƯỚC 3: TẠO DATA.YAML")

    data_yaml_path = dataset_path / "data.yaml"
    if not data_yaml_path.exists():
        create_data_yaml(dataset_path, validation["format"])
    else:
        print_success(f"data.yaml đã tồn tại: {data_yaml_path}")

    # If the dataset's data.yaml has Pose/Seg fields (kpt_shape), build a
    # workspace-local copy with those fields stripped so Ultralytics Detect
    # training does not crash on a format mismatch. The original dataset file
    # is never modified.
    workspace_root = Path(__file__).parent.resolve()
    data_yaml_path = ensure_seg_data_yaml(data_yaml_path, workspace_root)
    print_info(f"data.yaml dùng cho training: {data_yaml_path}")

    # Step 4: Get training config
    dataset_name = dataset_path.name.replace(" ", "_").replace(".", "_")
    config = get_training_config(dataset_name)

    # Step 5: Summary
    print_header("TỔNG KẾT CẤU HÌNH")
    print(f"  Dataset:     {dataset_path}")
    device_display = f"GPU {selected_device}" if selected_device != "cpu" else "CPU"
    print(f"  Device:      {device_display}")
    print(f"  data.yaml:   {data_yaml_path}")
    print(f"  Epochs:      {config['epochs']}")
    print(f"  Image size:  {config['imgsz']}")
    print(f"  Batch size:  {config['batch']}")
    print(f"  Run name:    {config['run_name']}")

    choice = input(f"\n{Colors.YELLOW}Bắt đầu training? (y/n): {Colors.END}").strip().lower()

    if choice not in ["y", "yes"]:
        print_info("Đã hủy training")
        sys.exit(0)

    # Step 6: Get Python executable and check dependencies
    print_header("BƯỚC 4: KIỂM TRA MÔI TRƯỜNG")

    python_exec = get_python_executable()
    print_info(f"Python executable: {python_exec}")

    # Check dependencies
    if not ensure_dependencies(python_exec):
        print_error("Không thể tiếp tục do thiếu dependencies!")
        sys.exit(1)

    # Step 7: Execute training
    print_header("BƯỚC 5: BẮT ĐẦU TRAINING")

    runs_dir = workspace_root / "runs"
    runs_dir.mkdir(exist_ok=True)

    print_info("Bắt đầu training... (Ctrl+C để dừng)")
    print("-" * 60)

    try:
        # Import YOLO and run training directly
        from ultralytics import YOLO
        import torch

        device = selected_device
        print_info(f"Training device: {'GPU ' + device + ' (' + torch.cuda.get_device_name(int(device)) + ')' if device != 'cpu' else 'CPU'}")

        model = YOLO("yolo11n-seg.pt")

        results = model.train(
            data=str(data_yaml_path),
            epochs=config['epochs'],
            imgsz=config['imgsz'],
            batch=config['batch'],
            project=str(runs_dir),
            name=config['run_name'],
            device=device,
            degrees=15.0,
            translate=0.1,
            scale=0.5,
            shear=5.0,
            flipud=0.0,
            fliplr=0.5,
            patience=20,
            save=True,
            save_period=10,
            verbose=True,
        )

        print_header("TRAINING HOÀN TẤT!")

        # Find output
        run_dir = runs_dir / config["run_name"]
        best_model = run_dir / "weights" / "best.pt"
        last_model = run_dir / "weights" / "last.pt"

        print_success("Training đã hoàn tất!")
        print()
        print(f"{Colors.GREEN}KẾT QUẢ:{Colors.END}")
        print(f"  Run directory: {run_dir}")

        if best_model.exists():
            print(f"  Best model:   {best_model}")
            print(f"  File size:    {best_model.stat().st_size / (1024*1024):.2f} MB")

        # Show results location
        print()
        print_success(f"Đường dẫn kết quả: {run_dir}")
        print()
        print(f"{Colors.CYAN}Các file quan trọng:{Colors.END}")
        for item in run_dir.glob("**/*"):
            if item.is_file() and item.suffix in [".pt", ".png", ".jpg"]:
                rel_path = item.relative_to(run_dir)
                print(f"  - {rel_path}")

    except KeyboardInterrupt:
        print()
        print_warning("Training đã bị dừng bởi user")
        sys.exit(0)

    except Exception as e:
        print_error(f"Lỗi khi chạy training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
