"""
================================================================================
TRAIN.PY - Interactive Training Menu for YOLOv11 (Seg / OBB / Pose)
================================================================================
Script điều phối huấn luyện nhiều loại mô hình YOLOv11 cho nhận diện móng.

**Tính năng:**
  - Interactive menu: chọn Seg / OBB / Pose / 5-class OBB / Resume
  - Multi-dataset support: Mỗi dataset có thư mục results riêng với timestamp
  - Tự động tạo train/val split nếu chưa có
  - Lưu kết quả có tổ chức: runs/<dataset_name>/<timestamp>/

**Menu (chạy ``python train.py`` rồi chọn):**

  [1] Train Seg (legacy)        - YOLOv11-Seg, 1-class, segmentation mask
  [2] Train OBB (smoke test)    - YOLO11-OBB, 139 ảnh (Nail_Detection_ThanhDT)
  [3] Train OBB (production)    - YOLO11-OBB, 10,501 ảnh (nail-segmentation)
  [4] Train OBB 5-class         - YOLO11-OBB, classification index/middle/...
  [5] Train Pose (legacy)       - YOLOv11-Pose, 4 keypoints
  [6] Resume training           - tiếp tục last.pt

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
# MENU / DISPATCH
# =============================================================================

# Smoke test dataset (139 ảnh) - hard-coded default paths to make the menu
# flow effortless. Both files live in the parent directory of nail_ai_workspace.
_SMOKE_DATASET_DATA_YAML = "..\\Nail_Detection_ThanhDT.v1i.yolov11\\data.yaml"
_PRODUCTION_DATASET_DATA_YAML = "..\\nail-segmentation.v1i.yolov11_10501\\data.yaml"


def _run_subprocess(script_relpath: str, *extra_args: str) -> int:
    """Run another script in the same workspace with the same venv.

    Returns the subprocess exit code.
    """
    python_exec = get_python_executable()
    script_path = Path(__file__).parent.resolve() / script_relpath
    cmd = [str(python_exec), str(script_path), *extra_args]
    print_info(f"Exec: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.resolve())
    return result.returncode


def _run_seg_legacy(dataset_path: Path) -> None:
    """Run the legacy Seg training flow inline.

    Reproduces the original ``train.py`` behavior: validate, patch
    ``data.yaml`` if it has pose fields, then train ``yolo11n-seg.pt``.
    """
    print_header("TRAIN SEG (LEGACY)")

    # Validate
    validation = validate_dataset(dataset_path)
    if not validation["valid"]:
        print_error("Dataset không hợp lệ!")
        for err in validation["errors"]:
            print(f"  {Colors.RED}•{Colors.END} {err}")
        sys.exit(1)

    # Split if needed
    if not validation["valid_images"]:
        print_warning("Dataset chưa có validation folder")
        choice = input(
            f"{Colors.YELLOW}Chia train/val tự động (80/20)? (y/n): {Colors.END}"
        ).strip().lower()
        if choice in ["y", "yes", ""]:
            try:
                split_dataset_auto(dataset_path, val_ratio=0.2)
            except Exception as e:
                print_error(f"Lỗi khi chia dataset: {e}")
                sys.exit(1)

    # data.yaml
    data_yaml_path = dataset_path / "data.yaml"
    if not data_yaml_path.exists():
        create_data_yaml(dataset_path, validation["format"])
    else:
        print_success(f"data.yaml đã tồn tại: {data_yaml_path}")

    workspace_root = Path(__file__).parent.resolve()
    data_yaml_path = ensure_seg_data_yaml(data_yaml_path, workspace_root)
    print_info(f"data.yaml dùng cho training: {data_yaml_path}")

    # Config
    dataset_name = dataset_path.name.replace(" ", "_").replace(".", "_")
    config = get_training_config(dataset_name)

    # Summary
    print_header("TỔNG KẾT CẤU HÌNH (SEG)")
    print(f"  Dataset:     {dataset_path}")
    print(f"  data.yaml:   {data_yaml_path}")
    print(f"  Epochs:      {config['epochs']}")
    print(f"  Image size:  {config['imgsz']}")
    print(f"  Batch size:  {config['batch']}")
    print(f"  Run name:    {config['run_name']}")

    choice = input(f"\n{Colors.YELLOW}Bắt đầu training? (y/n): {Colors.END}").strip().lower()
    if choice not in ["y", "yes"]:
        print_info("Đã hủy training")
        return

    python_exec = get_python_executable()
    if not ensure_dependencies(python_exec):
        print_error("Không thể tiếp tục do thiếu dependencies!")
        sys.exit(1)

    print_header("BẮT ĐẦU TRAINING SEG")

    from ultralytics import YOLO

    runs_dir = workspace_root / "runs"
    runs_dir.mkdir(exist_ok=True)

    model = YOLO("yolo11n-seg.pt")
    model.train(
        data=str(data_yaml_path),
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        project=str(runs_dir),
        name=config["run_name"],
        device="0",
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

    run_dir = runs_dir / config["run_name"]
    best_model = run_dir / "weights" / "best.pt"
    print_header("TRAINING HOÀN TẤT!")
    print_success(f"Run directory: {run_dir}")
    if best_model.exists():
        print_success(f"Best model:    {best_model}")


def _run_pose_legacy() -> None:
    """Run the legacy Pose training script via subprocess."""
    print_header("TRAIN POSE (LEGACY)")
    rc = _run_subprocess("train_pose.py")
    if rc != 0:
        print_error(f"train_pose.py exited with code {rc}")


def _pick_dataset_path() -> Path:
    """Helper to ask the user to pick a dataset path.

    Same logic as before - scans workspace parent for ``train/images`` or
    ``images`` folders, lets user pick one or type a manual path.
    """
    workspace_root = Path(__file__).parent.resolve()
    parent_root = workspace_root.parent

    potential_datasets = []
    for item in parent_root.iterdir():
        if item.is_dir() and (
            (item / "train" / "images").exists() or (item / "images").exists()
        ):
            potential_datasets.append({"name": item.name, "path": str(item)})

    if len(potential_datasets) > 1:
        print(f"{Colors.YELLOW}Tìm thấy các dataset trong workspace:{Colors.END}\n")
        for i, ds in enumerate(potential_datasets, 1):
            print(f"  {Colors.CYAN}{i}.{Colors.END} {ds['name']}")
            print(f"      {Colors.BLUE}→{Colors.END} {ds['path']}")

        choice = input(
            f"\n{Colors.CYAN}Chọn dataset (số) hoặc Enter để nhập đường dẫn: {Colors.END}"
        ).strip()
        if choice and choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(potential_datasets):
                return Path(potential_datasets[idx]["path"]).resolve()

    while True:
        dataset_path = input_path(
            "Nhập đường dẫn đến dataset (hoặc Enter để quét workspace)"
        )
        if not dataset_path and potential_datasets:
            dataset_path = potential_datasets[0]["path"]
        if dataset_path and Path(dataset_path).exists():
            return Path(dataset_path).resolve()
        print_error(f"Đường dẫn không tồn tại: {dataset_path}")


def _resolve_obb_data_yaml(dataset_path: Path) -> Path:
    """Return the ``data.yaml`` inside the dataset folder.

    For OBB scripts the convention is to point ``--data`` directly at the
    dataset's ``data.yaml`` (the scripts themselves handle Seg->OBB conversion
    and auto-split). We fall back to a typed path if the dataset doesn't ship
    one.
    """
    candidate = dataset_path / "data.yaml"
    if candidate.exists():
        return candidate
    typed = input_path("Dataset không có data.yaml. Nhập đường dẫn data.yaml: ")
    p = Path(typed)
    if not p.exists():
        print_error(f"Không tìm thấy: {p}")
        sys.exit(1)
    return p


def _prompt_train_params(default_epochs: int, default_imgsz: int, default_batch: int) -> tuple:
    """ chọn epochs, imgsz, batch size, và device.

    Returns:
        (epochs_str, imgsz_str, batch_str, device_str)
    """
    print(f"\n{Colors.YELLOW}--- Cấu hình training ---{Colors.END}")

    # --- Epochs ---
    epochs_input = input(
        f"  {Colors.CYAN}Số epochs {Colors.END}"
        f"{Colors.YELLOW}(Enter = {default_epochs}){Colors.END}: "
    ).strip()
    try:
        epochs = int(epochs_input) if epochs_input else default_epochs
    except ValueError:
        print_warning(f"Giá trị không hợp lệ, dùng mặc định: {default_epochs}")
        epochs = default_epochs

    # --- Image size ---
    print(f"\n  Chọn image size:")
    imgsz_options = {
        "1": (320, "320 (nhanh, mobile)"),
        "2": (416, "416 (cân bằng)"),
        "3": (640, "640 (chất lượng cao, mặc định)"),
    }
    for k, (v, label) in imgsz_options.items():
        marker = f"{Colors.YELLOW} ← default{Colors.END}" if v == default_imgsz else ""
        print(f"    {Colors.CYAN}{k}.{Colors.END} {label}{marker}")
    imgsz_choice = input(f"  {Colors.CYAN}Chọn (1-3, Enter = default): {Colors.END}").strip()
    imgsz = imgsz_options.get(imgsz_choice, (default_imgsz, ""))[0]

    # --- Batch size ---
    print(f"\n  Chọn batch size:")
    batch_options = {
        "1": (4, "4 (ít VRAM, ~4 GB)"),
        "2": (8, "8 (RTX 2060 an toàn)"),
        "3": (16, "16 (mặc định, cần ~8 GB)"),
        "4": (32, "32 (nhiều VRAM)"),
    }
    for k, (v, label) in batch_options.items():
        marker = f"{Colors.YELLOW} ← default{Colors.END}" if v == default_batch else ""
        print(f"    {Colors.CYAN}{k}.{Colors.END} {label}{marker}")
    batch_choice = input(f"  {Colors.CYAN}Chọn (1-4, Enter = default): {Colors.END}").strip()
    batch = batch_options.get(batch_choice, (default_batch, ""))[0]

    # --- Device ---
    print(f"\n  Chọn thiết bị training:")
    device_str = scan_gpus()
    if device_str != "cpu":
        print(f"  {Colors.GREEN}GPU sẵn sàng: device={device_str}{Colors.END}")
    else:
        print(f"  {Colors.YELLOW}Sẽ train trên CPU (chậm hơn).{Colors.END}")

    print(f"\n{Colors.GREEN}  ✔ epochs={epochs}  imgsz={imgsz}  batch={batch}  device={device_str}{Colors.END}\n")
    return str(epochs), str(imgsz), str(batch), device_str


def _run_obb_smoke() -> None:
    """[2] OBB smoke test (139 ảnh)."""
    print_header("TRAIN OBB - SMOKE TEST (139 ảnh)")
    print_info(f"Default data.yaml: {_SMOKE_DATASET_DATA_YAML}")
    override = input_path("Nhập data.yaml khác (Enter = default)")
    data_yaml = override if override else _SMOKE_DATASET_DATA_YAML

    epochs, imgsz, batch, device = _prompt_train_params(
        default_epochs=20, default_imgsz=416, default_batch=8
    )
    rc = _run_subprocess(
        "train_obb.py",
        "--data", data_yaml,
        "--epochs", epochs,
        "--imgsz", imgsz,
        "--batch", batch,
        "--device", device,
    )
    if rc != 0:
        print_error(f"train_obb.py exited with code {rc}")


def _run_obb_production() -> None:
    """[3] OBB production (10,501 ảnh)."""
    print_header("TRAIN OBB - PRODUCTION (10,501 ảnh)")
    print_info(f"Default data.yaml: {_PRODUCTION_DATASET_DATA_YAML}")
    override = input_path("Nhập data.yaml khác (Enter = default)")
    data_yaml = override if override else _PRODUCTION_DATASET_DATA_YAML

    epochs, imgsz, batch, device = _prompt_train_params(
        default_epochs=150, default_imgsz=640, default_batch=8
    )
    rc = _run_subprocess(
        "train_obb.py",
        "--data", data_yaml,
        "--epochs", epochs,
        "--imgsz", imgsz,
        "--batch", batch,
        "--device", device,
    )
    if rc != 0:
        print_error(f"train_obb.py exited with code {rc}")


def _run_obb_5class() -> None:
    """[4] OBB 5-class (finger classification)."""
    print_header("TRAIN OBB 5-CLASS")
    print_info(f"Default data.yaml: {_PRODUCTION_DATASET_DATA_YAML}")
    override = input_path("Nhập data.yaml khác (Enter = default)")
    data_yaml = override if override else _PRODUCTION_DATASET_DATA_YAML

    epochs, imgsz, batch, device = _prompt_train_params(
        default_epochs=150, default_imgsz=640, default_batch=8
    )
    rc = _run_subprocess(
        "train_obb_5class.py",
        "--data", data_yaml,
        "--epochs", epochs,
        "--imgsz", imgsz,
        "--batch", batch,
        "--device", device,
    )
    if rc != 0:
        print_error(f"train_obb_5class.py exited with code {rc}")


def _run_seg_5class() -> None:
    """[7] Seg 5-class (finger classification)."""
    print_header("TRAIN SEG 5-CLASS")
    print_info(f"Default data.yaml: {_PRODUCTION_DATASET_DATA_YAML}")
    override = input_path("Nhập data.yaml khác (Enter = default)")
    data_yaml = override if override else _PRODUCTION_DATASET_DATA_YAML

    epochs, imgsz, batch, device = _prompt_train_params(
        default_epochs=150, default_imgsz=640, default_batch=8
    )
    rc = _run_subprocess(
        "train_seg_5class.py",
        "--data", data_yaml,
        "--epochs", epochs,
        "--imgsz", imgsz,
        "--batch", batch,
        "--device", device,
    )
    if rc != 0:
        print_error(f"train_seg_5class.py exited with code {rc}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    print_header("YOLOV11 NAIL DETECTION - TRAINING MENU")
    print(
        f"{Colors.YELLOW}Script điều phối huấn luyện nhiều loại model "
        f"(Seg / OBB / Pose).{Colors.END}\n"
    )

    workspace_root = Path(__file__).parent.resolve()
    runs_dir = workspace_root / "runs"

    # --- RESUME (option 6) ---
    if runs_dir.exists():
        recent_lasts = list(runs_dir.glob("*/weights/last.pt"))
        if recent_lasts:
            recent_lasts.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest_last_pt = recent_lasts[0]
            print_header("TÌM THẤY CHECKPOINT CHƯA HOÀN THÀNH")
            print(f"{Colors.YELLOW}Phát hiện quá trình training có thể đã bị gián đoạn.{Colors.END}")
            try:
                ckpt_disp = latest_last_pt.relative_to(workspace_root)
            except ValueError:
                ckpt_disp = latest_last_pt
            print(f"  {Colors.CYAN}{ckpt_disp}{Colors.END}")

            choice = input(
                f"\n{Colors.YELLOW}Bạn có muốn resume training này không? "
                f"(y/n, Enter = bỏ qua): {Colors.END}"
            ).strip().lower()
            if choice in ["y", "yes"]:
                python_exec = get_python_executable()
                if not ensure_dependencies(python_exec):
                    sys.exit(1)

                print_header("BẮT ĐẦU PHỤC HỒI TRAINING")
                try:
                    from ultralytics import YOLO
                    print_info(f"Đang tải checkpoint: {latest_last_pt.name}")
                    model = YOLO(str(latest_last_pt))
                    model.train(resume=True)
                    print_header("TRAINING HOÀN TẤT (RESUME)")
                    sys.exit(0)
                except KeyboardInterrupt:
                    print()
                    print_warning("Training đã bị dừng bởi user.")
                    sys.exit(0)
                except Exception as e:
                    print_error(f"Lỗi khi resume: {e}")
                    import traceback
                    traceback.print_exc()
                    sys.exit(1)

    # --- MAIN MENU ---
    print_header("CHỌN CHẾ ĐỘ TRAINING")
    menu_options = [
        "Train Seg (legacy, 1-class)",          # 1
        "Train OBB (smoke test) - 139 ảnh",     # 2
        "Train OBB (production) - 10,501 ảnh",  # 3
        "Train OBB 5-class (finger names)",     # 4
        "Train Pose (legacy, 4 keypoints)",     # 5
        "Resume training (đã có last.pt)",       # 6
        "Train Seg 5-class (finger names & rotation)", # 7
    ]
    for i, opt in enumerate(menu_options, 1):
        print(f"  {Colors.CYAN}{i}.{Colors.END} {opt}")

    choice = input(
        f"\n{Colors.CYAN}Chọn (1-7, Enter = 1): {Colors.END}"
    ).strip() or "1"

    if choice not in {"1", "2", "3", "4", "5", "6", "7"}:
        print_warning("Lựa chọn không hợp lệ, mặc định về [1] Seg.")
        choice = "1"

    # --- DISPATCH ---
    if choice == "1":
        print_header("BƯỚC 0: QUÉT GPU")
        scan_gpus()  # Informational; legacy flow defaults device="0"
        dataset_path = _pick_dataset_path()
        _run_seg_legacy(dataset_path)

    elif choice == "2":
        _run_obb_smoke()

    elif choice == "3":
        _run_obb_production()

    elif choice == "4":
        _run_obb_5class()

    elif choice == "5":
        _run_pose_legacy()

    elif choice == "6":
        # Re-trigger the resume block if user picks it from the menu.
        print_info("Hãy chạy lại script để kích hoạt resume prompt.")
        print_info("Hoặc đặt last.pt vào runs/<name>/weights/last.pt rồi chạy lại.")

    elif choice == "7":
        _run_seg_5class()


if __name__ == "__main__":
    main()
