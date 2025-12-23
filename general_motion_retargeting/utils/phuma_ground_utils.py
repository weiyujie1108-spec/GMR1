"""
PHUMA Ground Detection Utilities for GMR1
集成PHUMA的鲁棒地面检测功能到GMR1代码库

主要功能：
1. find_robust_ground: 使用统计方法找到最合理的地面高度
2. get_foot_contact: 计算脚部接触地面的分数
"""

import numpy as np
from scipy.spatial import ConvexHull, Delaunay

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def find_robust_ground(vertices, foot_contact_vertex_indices, ground_thresh=0.05, heel_offset=0.005):
    """
    找到鲁棒的地面高度，使用PHUMA的方法。
    
    该方法通过统计脚部接触点的高度分布，找到最密集的高度区间作为地面高度。
    这比简单找最低点更鲁棒，能处理floating和penetration问题。
    
    Args:
        vertices: (N, V, 3) numpy array, 所有帧的顶点位置
        foot_contact_vertex_indices: dict, 包含以下key:
            - "left_toe_indices": 左脚趾顶点索引列表
            - "left_heel_indices": 左脚跟顶点索引列表
            - "right_toe_indices": 右脚趾顶点索引列表
            - "right_heel_indices": 右脚跟顶点索引列表
        ground_thresh: float, 地面高度阈值（米），默认0.05
        heel_offset: float, 脚跟偏移量（米），默认0.005
        
    Returns:
        float: 鲁棒的地面高度（Y坐标）
    """
    toe_contact_indices = list(foot_contact_vertex_indices["left_toe_indices"]) + list(foot_contact_vertex_indices["right_toe_indices"])
    heel_contact_indices = list(foot_contact_vertex_indices["left_heel_indices"]) + list(foot_contact_vertex_indices["right_heel_indices"])
    
    # 获取每帧脚趾和脚跟的最低高度
    toe_contact_heights = np.min(vertices[:, toe_contact_indices, 1], axis=1)
    heel_contact_heights = np.min(vertices[:, heel_contact_indices, 1], axis=1) + heel_offset
    
    # 计算每帧的平均接触高度
    contact_heights = np.mean(np.stack([toe_contact_heights, heel_contact_heights], axis=1), axis=1)
    contact_heights.sort()

    max_count = 0
    best_h_stars = []
    
    # 使用滑动窗口找到最密集的高度区间
    j = 0
    for i in range(len(contact_heights)):
        while j < len(contact_heights) and contact_heights[j] <= contact_heights[i] + ground_thresh:
            j += 1
        current_count = j - i
        current_h_star = contact_heights[i] + ground_thresh / 2
        
        if current_count > max_count:
            max_count = current_count
            best_h_stars = [current_h_star]
        elif current_count == max_count:
            best_h_stars.append(current_h_star)

    return np.median(best_h_stars)


def get_foot_contact(vertices, foot_contact_vertex_indices, robust_ground_y, ground_thresh=0.05, heel_offset=0.005):
    """
    计算每帧脚部接触地面的分数。
    
    Args:
        vertices: (N, V, 3) numpy array, 所有帧的顶点位置
        foot_contact_vertex_indices: dict, 脚部接触顶点索引
        robust_ground_y: float, 鲁棒的地面高度
        ground_thresh: float, 地面高度阈值（米）
        heel_offset: float, 脚跟偏移量（米）
        
    Returns:
        (N, 4) numpy array: 每帧的脚部接触分数 [left_toe, left_heel, right_toe, right_heel]
    """
    num_frames = vertices.shape[0]
    foot_contacts = np.zeros((num_frames, len(foot_contact_vertex_indices)), dtype=np.float32)
    contact_range = [robust_ground_y - ground_thresh / 2.0, robust_ground_y + ground_thresh / 2.0]

    for i in range(num_frames):
        for j, (contact_type, indices) in enumerate(foot_contact_vertex_indices.items()):
            heights = vertices[i, indices, 1]
            offset = heel_offset if "heel" in contact_type else 0.0
            contact_count = np.sum((heights >= contact_range[0] + offset) & (heights <= contact_range[1] + offset))
            foot_contacts[i, j] = contact_count / len(indices)

    return foot_contacts


def find_robust_ground_from_joints(joints, foot_joint_names, ground_thresh=0.05):
    """
    从关节位置找到鲁棒的地面高度（适用于没有顶点数据的情况）。
    
    Args:
        joints: (N, J, 3) numpy array, 所有帧的关节位置
        foot_joint_names: list, 脚部关节名称列表，例如 ["LeftFoot", "RightFoot"]
        ground_thresh: float, 地面高度阈值（米）
        
    Returns:
        float: 鲁棒的地面高度（Y坐标）
    """
    # 找到脚部关节的索引（需要根据实际的关节名称映射）
    # 这里假设joints是按照固定顺序排列的
    foot_heights = []
    for foot_name in foot_joint_names:
        # 这里需要根据实际的关节索引来获取
        # 假设foot_joint_names对应joints中的某些索引
        pass
    
    # 简化版本：直接使用所有关节的最低点
    all_heights = joints[:, :, 1].flatten()  # Y坐标
    all_heights.sort()
    
    max_count = 0
    best_h_stars = []
    
    j = 0
    for i in range(len(all_heights)):
        while j < len(all_heights) and all_heights[j] <= all_heights[i] + ground_thresh:
            j += 1
        current_count = j - i
        current_h_star = all_heights[i] + ground_thresh / 2
        
        if current_count > max_count:
            max_count = current_count
            best_h_stars = [current_h_star]
        elif current_count == max_count:
            best_h_stars.append(current_h_star)
    
    return np.median(best_h_stars)


def calculate_bos_distance(joints, target_joint_id=0):
    """
    计算目标关节到支撑基座（Base of Support）的距离。
    
    Args:
        joints: (N, J, 3) numpy array, 所有帧的关节位置
        target_joint_id: int, 目标关节索引（0通常是骨盆）
        
    Returns:
        float: 平均距离到支撑基座的距离
    """
    num_frames = joints.shape[0]
    target_projection = joints[:, target_joint_id, [0, 2]]  # XZ平面投影
    frame_distances = []
    
    for i in range(num_frames):
        # 获取脚踝和脚部关节位置（需要根据实际索引调整）
        # 假设索引7,8是脚踝，10,11是脚部
        foot_ankle_joints = joints[i, [7, 8, 10, 11], :]
        foot_ankle_points_xz = foot_ankle_joints[:, [0, 2]]
        pelvis_pt = target_projection[i]
        
        unique_points = np.unique(foot_ankle_points_xz, axis=0)
        
        if len(unique_points) < 3:
            frame_distances.append(0.0)
            continue
            
        hull = ConvexHull(unique_points)
        if Delaunay(hull.points[hull.vertices]).find_simplex(pelvis_pt) >= 0:
            dist = 0.0
        else:
            hull_vertices = hull.points[hull.vertices]
            min_dist_to_edge = float('inf')
            for j in range(len(hull_vertices)):
                p1 = hull_vertices[j]
                p2 = hull_vertices[(j + 1) % len(hull_vertices)]
                seg_dist = _point_to_segment_dist(pelvis_pt, p1, p2)
                if seg_dist < min_dist_to_edge:
                    min_dist_to_edge = seg_dist
            dist = min_dist_to_edge
                
        frame_distances.append(dist)
        
    return np.mean(frame_distances) if sum(frame_distances) > 0.0 else 0.0


def _point_to_segment_dist(p, a, b):
    """计算点到线段的距离"""
    ab = b - a
    ap = p - a
    
    dot_ab_ab = np.dot(ab, ab)
    if dot_ab_ab == 0:
        return np.linalg.norm(ap)

    t = np.dot(ap, ab) / dot_ab_ab
    
    if 0.0 <= t <= 1.0:
        projection = a + t * ab
        return np.linalg.norm(p - projection)
    else:
        return min(np.linalg.norm(p - a), np.linalg.norm(p - b))


def find_robust_ground_from_human_data(human_data, ground_thresh=0.05, ground_offset=0.0):
    """
    从human_data格式（字典）中找到鲁棒的地面高度，使用PHUMA的方法。
    
    该方法通过统计脚部body位置的高度分布，找到最密集的高度区间作为地面高度。
    这比简单找最低点更鲁棒，能处理floating和penetration问题。
    
    Args:
        human_data: dict, 格式为 {body_name: [position, quaternion], ...}
            position是3D numpy array (x, y, z)，其中z是高度方向
        ground_thresh: float, 地面高度阈值（米），默认0.05（5cm）
        ground_offset: float, 地面偏移量（米），默认0.0（不推荐使用，应在调用函数中应用）
        
    Returns:
        float: 鲁棒的地面高度（Z坐标）
    """
    # 识别脚部相关的body名称
    foot_body_names = []
    toe_body_names = []
    heel_body_names = []
    ankle_body_names = []
    
    for body_name in human_data.keys():
        body_name_lower = body_name.lower()
        
        # 识别脚趾（toe）
        if "toe" in body_name_lower:
            toe_body_names.append(body_name)
            foot_body_names.append(body_name)
        
        # 识别脚跟（heel），通常脚跟就是脚本身
        elif "heel" in body_name_lower:
            heel_body_names.append(body_name)
            foot_body_names.append(body_name)
        
        # 识别脚（foot）
        elif "foot" in body_name_lower:
            # 如果已经有toe和heel分类，foot可以作为heel
            if "toe" not in body_name_lower:
                heel_body_names.append(body_name)
            foot_body_names.append(body_name)
        
        # 识别脚踝（ankle），也可以作为参考点
        elif "ankle" in body_name_lower:
            ankle_body_names.append(body_name)
            foot_body_names.append(body_name)
    
    if not foot_body_names:
        # 如果没有找到脚部body，返回默认值
        return 0.0
    
    # 收集所有脚部位置的高度
    all_heights = []
    
    for body_name in foot_body_names:
        if body_name in human_data:
            pos, _ = human_data[body_name]
            # 假设position是(x, y, z)格式，z是高度
            # 注意：这里需要确认坐标系，PHUMA使用y作为高度，GMR1可能使用z
            height = pos[2]  # 使用z坐标作为高度
            
            # 根据body类型应用偏移（类似于PHUMA的heel_offset）
            offset = 0.0
            if body_name in heel_body_names:
                offset = 0.005  # 脚跟偏移5mm（类似PHUMA）
            
            all_heights.append(height + offset)
    
    if not all_heights:
        return 0.0
    
    # 转换为numpy数组并排序
    all_heights = np.array(all_heights)
    if len(all_heights) == 0:
        return 0.0
    
    all_heights.sort()
    
    # 对于单帧数据，使用滑动窗口方法找到最密集的高度区间（类似PHUMA的find_robust_ground）
    # 这样可以处理单个frame中多个脚部body点的情况
    max_count = 0
    best_h_stars = []
    
    j = 0
    for i in range(len(all_heights)):
        while j < len(all_heights) and all_heights[j] <= all_heights[i] + ground_thresh:
            j += 1
        current_count = j - i
        current_h_star = all_heights[i] + ground_thresh / 2
        
        if current_count > max_count:
            max_count = current_count
            best_h_stars = [current_h_star]
        elif current_count == max_count:
            best_h_stars.append(current_h_star)
    
    # 返回最密集区间的中位数作为地面高度
    # 如果没有找到密集区间，返回所有高度的中位数
    if best_h_stars:
        robust_ground = np.median(best_h_stars)
    else:
        robust_ground = np.median(all_heights)
    
    return robust_ground


def find_robust_ground_from_body_positions(
    body_positions,
    body_names=None,
    foot_body_indices=None,
    ground_thresh=0.05,
    heel_offset=0.005,
    use_robust_statistics=True,
):
    """
    从 robot body positions 中找到鲁棒的地面高度（Z），尽量对齐 PHUMA 的思路：
    - 先按帧构造 “接触候选高度”(contact_heights)：每帧取左右脚的接触候选，再取更低的一侧
    - 再在 contact_heights 上做密度滑窗，找最密集的高度区间作为地面高度

    为什么要这样做：
    - 如果直接把所有 foot/toe/ankle 的高度混在一起统计，摆动脚高度会主导分布，导致地面被估高 → 脚穿地

    Args:
        body_positions: (T, N, 3) numpy / torch, 所有帧的 body 位置（z 为高度）
        body_names: list[str] or None, body 名称（用于区分 left/right 以及 toe/heel/foot）
        foot_body_indices: list[int] or None, 指定用于脚部判定的 body 索引（可选）
        ground_thresh: float, 密度窗口阈值（米），默认 0.05
        heel_offset: float, heel 偏移（米），默认 0.005（与 PHUMA 一致）
        use_robust_statistics: bool, True 使用密度统计；False 直接返回 contact_heights 的最小值（更保守）

    Returns:
        float: 估计的地面高度（Z）
    """
    # 转换为 numpy
    if HAS_TORCH and torch.is_tensor(body_positions):
        body_positions_np = body_positions.detach().cpu().numpy()
    else:
        body_positions_np = np.asarray(body_positions)

    if body_positions_np.ndim != 3 or body_positions_np.shape[-1] != 3:
        raise ValueError(f"body_positions must be (T, N, 3), got {body_positions_np.shape}")

    T, N, _ = body_positions_np.shape
    if T == 0 or N == 0:
        return 0.0

    # 默认选择“最可能是脚”的一部分 body（用于没有 body_names 的兜底）
    if foot_body_indices is None:
        min_heights_per_body = np.min(body_positions_np[:, :, 2], axis=0)  # (N,)
        num_foot_bodies = max(4, int(N * 0.2))
        foot_body_indices = np.argsort(min_heights_per_body)[:num_foot_bodies].tolist()

    foot_body_indices = [i for i in foot_body_indices if 0 <= int(i) < N]
    if not foot_body_indices:
        return 0.0

    # 构造 contact_heights：按帧，只取“更像接触地面”的候选（类似 PHUMA 的 per-frame contact_heights）
    contact_heights = []

    if body_names is not None and len(body_names) == N:
        # 先分类索引：left/right + toe/heel/foot/ankle
        def _is_left(name: str) -> bool:
            n = name.lower()
            return ("left" in n) or n.startswith("l_") or n.endswith("_l") or ("_l_" in n)

        def _is_right(name: str) -> bool:
            n = name.lower()
            return ("right" in n) or n.startswith("r_") or n.endswith("_r") or ("_r_" in n)

        def _is_toe(name: str) -> bool:
            return "toe" in name.lower()

        def _is_heel(name: str) -> bool:
            n = name.lower()
            # 很多模型没有 heel body；foot 往往更接近 heel 的候选
            return ("heel" in n) or ("foot" in n)

        left_toe = []
        left_heel = []
        right_toe = []
        right_heel = []

        for idx in foot_body_indices:
            name = body_names[idx]
            if _is_left(name):
                if _is_toe(name):
                    left_toe.append(idx)
                if _is_heel(name):
                    left_heel.append(idx)
            elif _is_right(name):
                if _is_toe(name):
                    right_toe.append(idx)
                if _is_heel(name):
                    right_heel.append(idx)

        # 允许某些模型不含 toe/heel：退化到 “左右脚所有 foot_body_indices 的 min”
        left_any = [i for i in foot_body_indices if _is_left(body_names[i])]
        right_any = [i for i in foot_body_indices if _is_right(body_names[i])]

        for t in range(T):
            candidates = []

            # left contact candidate
            left_vals = []
            if left_toe:
                left_vals.append(float(np.min(body_positions_np[t, left_toe, 2])))
            if left_heel:
                left_vals.append(float(np.min(body_positions_np[t, left_heel, 2]) + heel_offset))
            if not left_vals and left_any:
                left_vals.append(float(np.min(body_positions_np[t, left_any, 2])))
            if left_vals:
                candidates.append(float(np.mean(left_vals)))  # toe/heel 平均（对齐 PHUMA）

            # right contact candidate
            right_vals = []
            if right_toe:
                right_vals.append(float(np.min(body_positions_np[t, right_toe, 2])))
            if right_heel:
                right_vals.append(float(np.min(body_positions_np[t, right_heel, 2]) + heel_offset))
            if not right_vals and right_any:
                right_vals.append(float(np.min(body_positions_np[t, right_any, 2])))
            if right_vals:
                candidates.append(float(np.mean(right_vals)))

            if candidates:
                contact_heights.append(min(candidates))  # 每帧取更低的一侧（更像支撑脚）
    else:
        # 兜底：没有 body_names 就按帧取 foot_body_indices 的 min，当作 contact height
        for t in range(T):
            contact_heights.append(float(np.min(body_positions_np[t, foot_body_indices, 2])))

    if not contact_heights:
        return 0.0

    contact_heights = np.asarray(contact_heights, dtype=np.float64)
    if contact_heights.size == 0:
        return 0.0

    if not use_robust_statistics:
        return float(np.min(contact_heights))

    # PHUMA-style：在 contact_heights 上做密度滑窗
    hs = np.sort(contact_heights)
    max_count = 0
    best_h_stars = []
    j = 0
    for i in range(len(hs)):
        while j < len(hs) and hs[j] <= hs[i] + ground_thresh:
            j += 1
        current_count = j - i
        current_h_star = hs[i] + ground_thresh / 2.0
        if current_count > max_count:
            max_count = current_count
            best_h_stars = [current_h_star]
        elif current_count == max_count:
            best_h_stars.append(current_h_star)

    return float(np.median(best_h_stars)) if best_h_stars else float(np.median(hs))

