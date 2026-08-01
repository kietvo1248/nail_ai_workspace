import os
import random
import shutil
from pathlib import Path

def split_yolo_dataset(dataset_dir: str, val_ratio: float = 0.2):
    base_dir = Path(dataset_dir)
    train_images = base_dir / "train" / "images"
    train_labels = base_dir / "train" / "labels"
    
    val_images = base_dir / "valid" / "images"
    val_labels = base_dir / "valid" / "labels"
    
    # Tạo thư mục valid nếu chưa có
    val_images.mkdir(parents=True, exist_ok=True)
    val_labels.mkdir(parents=True, exist_ok=True)
    
    # Lấy danh sách ảnh trong thư mục train
    all_images = list(train_images.glob("*.jpg"))
    if not all_images:
        print("Không tìm thấy ảnh nào trong train/images!")
        return
        
    print(f"Tổng số ảnh ban đầu trong train: {len(all_images)}")
    
    # Tính số lượng ảnh cần chuyển sang valid
    val_count = int(len(all_images) * val_ratio)
    
    # Trộn ngẫu nhiên và chọn ảnh
    random.seed(42) # Để kết quả luôn cố định
    val_selected = random.sample(all_images, val_count)
    
    print(f"Đang chuyển {val_count} ảnh sang thư mục valid...")
    
    moved_count = 0
    for img_path in val_selected:
        # Tên file ảnh và label
        img_name = img_path.name
        label_name = img_path.stem + ".txt"
        
        lbl_path = train_labels / label_name
        
        # Di chuyển ảnh
        shutil.move(str(img_path), str(val_images / img_name))
        
        # Di chuyển label (nếu có)
        if lbl_path.exists():
            shutil.move(str(lbl_path), str(val_labels / label_name))
            moved_count += 1
            
    print(f"Xong! Đã tạo tập Validation với {val_count} ảnh và {moved_count} file nhãn.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Sử dụng: python split_dataset.py <đường_dẫn_tới_thư_mục_dataset>")
    else:
        split_yolo_dataset(sys.argv[1])
