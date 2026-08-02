import numpy as np
from typing import List, Tuple

def polygon_to_coords(polygon: List[Tuple[float, float]]) -> np.ndarray:
    return np.array(polygon, dtype=np.float64)

def get_direction_from_polygon_pca(polygon: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Tự động tính toán direction vector (hướng trục dọc) của polygon.
    Dùng PCA (Principal Component Analysis) để tìm trục chính yếu của hình dạng móng.
    """
    pts = polygon_to_coords(polygon)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    
    # Tính Covariance matrix
    cov = centered.T @ centered
    
    # Tính Eigenvalues và Eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # PC1 là eigenvector tương ứng với eigenvalue lớn nhất (trục dọc của móng)
    pc1 = eigenvectors[:, np.argmax(eigenvalues)]
    
    # Chiếu tất cả các điểm lên trục PC1
    projections = centered @ pc1
    
    # Tìm 2 điểm xa nhất trên trục (base và tip)
    tip_idx = np.argmax(projections)
    base_idx = np.argmin(projections)
    tip = pts[tip_idx]
    base = pts[base_idx]
    
    # Tính direction vector (từ base -> tip)
    direction = tip - base
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return (0.0, -1.0) # Tránh chia cho 0
        
    direction = direction / norm
    
    # Để an toàn cho móng tay (thường hướng lên hoặc ngang, hiếm khi chĩa thẳng xuống đất
    # trong khung hình try-on), ta có thể ép hướng chính luôn đi lên trên (y âm trong toạ độ ảnh).
    # Mặc dù PCA tìm ra trục nhưng hướng của trục có thể ngược.
    if direction[1] > 0:
        direction = -direction
        
    return (float(direction[0]), float(direction[1]))


def find_cutoff_edge_intersections(
    pts: np.ndarray, direction: Tuple[float, float], base_proj: float, cutoff_proj: float
) -> List[Tuple[float, float]]:
    """Tìm 2 điểm giao cắt của đường thẳng cắt (cutoff) với các cạnh của polygon"""
    intersections = []
    n = len(pts)
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        
        # Chiếu p1, p2 lên trục direction
        proj1 = p1[0] * direction[0] + p1[1] * direction[1]
        proj2 = p2[0] * direction[0] + p2[1] * direction[1]
        
        # Nếu đường thẳng (p1, p2) cắt ngang đường thẳng cutoff_proj
        if (proj1 <= cutoff_proj) != (proj2 <= cutoff_proj):
            t = (cutoff_proj - proj1) / (proj2 - proj1 + 1e-9)
            t = max(0.0, min(1.0, t))
            ix = p1[0] + t * (p2[0] - p1[0])
            iy = p1[1] + t * (p2[1] - p1[1])
            intersections.append((float(ix), float(iy)))
            
    return intersections


def cut_polygon_at_ratio(
    polygon: List[Tuple[float, float]], direction: Tuple[float, float], bed_ratio: float = 0.75
) -> List[Tuple[float, float]]:
    """
    Cắt bỏ phần ngọn móng (free edge), giữ lại phần thân móng (nail bed).
    
    Args:
        polygon: Chùm điểm viền móng (full nail) do YOLO dự đoán.
        direction: Vector hướng dọc của móng (tính bằng PCA).
        bed_ratio: Tỉ lệ phần thân móng muốn giữ lại (ví dụ 0.75 = 75%).
        
    Returns:
        nail_bed_polygon: Đa giác của phần thân móng.
    """
    if bed_ratio <= 0.0 or bed_ratio >= 1.0:
        raise ValueError(f"bed_ratio must be in (0.0, 1.0), got {bed_ratio}")
        
    pts = polygon_to_coords(polygon)
    dx, dy = direction
    
    # Chiếu tất cả điểm lên trục direction
    projections = (pts[:, 0] * dx + pts[:, 1] * dy)
    
    # Tìm gốc móng (nhỏ nhất) và ngọn móng (lớn nhất)
    base_proj = np.min(projections)
    tip_proj = np.max(projections)
    total_length = tip_proj - base_proj
    
    if total_length < 1e-9:
        return polygon # Không thể cắt
        
    # Tính vị trí cắt (75% từ gốc)
    cutoff_proj = base_proj + total_length * bed_ratio
    
    nail_bed_pts = []
    
    # Giữ lại các điểm thuộc về nail bed (từ gốc đến vị trí cắt)
    for i in range(len(pts)):
        if projections[i] <= cutoff_proj + 1e-9:
            nail_bed_pts.append((float(pts[i, 0]), float(pts[i, 1])))
            
    # Tìm 2 điểm giao cắt để đóng polygon mượt mà
    cutoff_pts = find_cutoff_edge_intersections(pts, direction, base_proj, cutoff_proj)
    
    if len(cutoff_pts) >= 2:
        # Sắp xếp 2 điểm cắt theo chiều ngang so với hướng móng để polygon không bị chéo góc
        perp_dir = np.array([-dy, dx])
        cutoff_pts = sorted(cutoff_pts, key=lambda p: float(p[0]) * perp_dir[0] + float(p[1]) * perp_dir[1])
        nail_bed_pts.extend(cutoff_pts[:2])
        
    # Sắp xếp lại polygon (convex hull hoặc sắp xếp theo góc) để đảm bảo vẽ đúng
    # (Vì cắt ngang xong, thứ tự các đỉnh có thể bị lộn xộn)
    if len(nail_bed_pts) >= 3:
        nail_bed_pts = _sort_polygon_points(nail_bed_pts)
        
    return nail_bed_pts

def _sort_polygon_points(pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Sắp xếp các điểm theo thứ tự ngược chiều kim đồng hồ quanh trọng tâm"""
    pts_array = np.array(pts)
    centroid = pts_array.mean(axis=0)
    angles = np.arctan2(pts_array[:, 1] - centroid[1], pts_array[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    return [tuple(pts_array[i]) for i in sorted_indices]
