#!/usr/bin/env python3
"""
从现有motion数据中提取12个关键点，进行retargeting并可视化
用于debug这个方法是否可行
"""
import math
import argparse
import pathlib
import sys
import time
import numpy as np
import json
import threading
import asyncio
import torch
import scipy.ndimage.filters as filters
from scipy.spatial.transform import Rotation as R
from aiohttp import web

# 确保可以导入项目根目录下的模块
HERE = pathlib.Path(__file__).parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import websocket

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
from smplx.joint_names import JOINT_NAMES
from websocket_client import RobotDataClient
from general_motion_retargeting.kinematics_model import KinematicsModel
# === 辅助类：CPU 数据包装器 ===
class CPUOutput:
    def __init__(self, output):
        self.global_orient = output.global_orient.cpu()
        self.full_pose = output.full_pose.cpu()
        self.joints = output.joints.cpu()

# === 核心：One Euro Filter (平滑器，消除视觉抖动) ===
class OneEuroSmoother:
    def __init__(self, min_cutoff=0.01, beta=0.05, d_cutoff=1.0, freq=30):
        """
        min_cutoff: 最小截止频率，越小静止时越稳。
        beta: 速度系数，越大跟随越快（但也越容易抖）。
        """
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def process(self, qpos):
        current_qpos = qpos.copy()
        t = time.time()
        
        if self.x_prev is None:
            self.x_prev = current_qpos
            self.dx_prev = np.zeros_like(current_qpos)
            self.t_prev = t
            return current_qpos

        t_e = t - self.t_prev
        if t_e <= 0: return self.x_prev 
        
        # 四元数反转检查 (防止旋转突变)
        curr_quat = current_qpos[3:7]
        prev_quat = self.x_prev[3:7]
        if np.dot(curr_quat, prev_quat) < 0:
            current_qpos[3:7] = -curr_quat

        # 滤波计算
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (current_qpos - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, current_qpos, self.x_prev)
        
        # 归一化四元数
        x_hat[3:7] /= np.linalg.norm(x_hat[3:7])

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat

def to_tensor(data_dict, device='cuda'):
    tensor_dict = {}
    for k, v in data_dict.items():
        if isinstance(v, np.ndarray):
            if v.dtype == np.float64: v = v.astype(np.float32)
            tensor_dict[k] = torch.from_numpy(v).to(device)
        else:
            tensor_dict[k] = v
    return tensor_dict

# === 全局变量：存储机器人数据（用于 WebSocket 推送）===
# 注意：虽然变量名是 local_body_pos，但实际存储的是全局位置（global_body_pos）
robot_local_body_pos = np.zeros([12, 3])
robot_body_vel = np.zeros([12, 3])
robot_timestamp = 0.0

def compute_velocity(p, time_delta, guassian_filter=True):
    """计算速度"""
    velocity = np.gradient(p.numpy(), axis=-3) / time_delta
    if guassian_filter:
        velocity = torch.from_numpy(filters.gaussian_filter1d(velocity, 2, axis=-3, mode="nearest")).to(p)
    else:
        velocity = torch.from_numpy(velocity).to(p)
    return velocity

def update_robot_data(local_body_pos, ref_body_vel):
    """
    更新全局机器人数据
    注意：参数名是 local_body_pos，但实际传入的是全局位置（global_body_pos）
    为了与 mocap_to_robot.py 保持一致，保持这个命名
    """
    global robot_local_body_pos, robot_body_vel, robot_timestamp
    
    # 转换torch tensor为numpy array
    if isinstance(local_body_pos, torch.Tensor):
        local_body_pos = local_body_pos.cpu().numpy()
    if isinstance(ref_body_vel, torch.Tensor):
        ref_body_vel = ref_body_vel.cpu().numpy()
    
    robot_local_body_pos = local_body_pos
    robot_body_vel = ref_body_vel
    robot_timestamp = time.time()

# === WebSocket 服务器：用于向机器人端推送数据 ===
class RobotDataServer:
    """机器人数据Web服务器"""
    
    def __init__(self, port=8080):
        self.port = port
        self.app = None
        self.runner = None
        self.site = None
        self.active_websockets = set()  # 存储所有活跃的WebSocket连接
    
    async def websocket_handler(self, request):
        """WebSocket处理器 - 实时推送模式"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        # 添加到活跃连接列表
        self.active_websockets.add(ws)
        print(f'WebSocket连接建立，当前连接数: {len(self.active_websockets)}')
        
        try:
            # 循环推送数据
            while True:
                await asyncio.sleep(0.01)  # 控制推送频率，约100Hz
                
                # 读取全局变量
                global robot_local_body_pos, robot_body_vel, robot_timestamp
                data = {
                    "local_body_pos": robot_local_body_pos.tolist(),
                    "body_vel": robot_body_vel.tolist(),
                    "timestamp": robot_timestamp
                }
                
                # 推送数据
                try:
                    await ws.send_json(data)
                except Exception as e:
                    # 如果发送失败，说明连接已断开
                    print(f'WebSocket推送失败: {e}')
                    break
                    
        except Exception as e:
            print(f'WebSocket错误: {e}')
        finally:
            # 从活跃连接列表移除
            self.active_websockets.discard(ws)
            print(f'WebSocket连接关闭，当前连接数: {len(self.active_websockets)}')
        
        return ws
    
    async def start_server(self):
        """启动Web服务器"""
        self.app = web.Application()
        
        # 添加路由
        self.app.router.add_route('GET', '/ws', self.websocket_handler)
        
        # 启动服务器
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, '0.0.0.0', self.port)
        await self.site.start()
        print(f'机器人数据服务器已启动: http://localhost:{self.port}')
        print(f'  - WebSocket: ws://localhost:{self.port}/ws')
    
    def start(self):
        """在新线程中启动Web服务器"""
        def run_server():
            # 创建新的事件循环并设置为当前线程的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # 启动服务器并运行事件循环
            loop.run_until_complete(self.start_server())
            loop.run_forever()
        
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        # 等待服务器启动
        time.sleep(0.5)
# Metrab 关节顺序（SMPL_BONE_ORDER_NAMES）
METRAB_JOINT_NAMES = [
    "Pelvis",      # 0
    "L_Hip",       # 1
    "R_Hip",       # 2
    "Torso",       # 3
    "L_Knee",      # 4
    "R_Knee",      # 5
    "Spine",       # 6
    "L_Ankle",     # 7
    "R_Ankle",     # 8
    "Chest",       # 9
    "L_Toe",       # 10
    "R_Toe",       # 11
    "Neck",        # 12
    "L_Thorax",    # 13
    "R_Thorax",    # 14
    "Head",        # 15
    "L_Shoulder",  # 16
    "R_Shoulder",  # 17
    "L_Elbow",     # 18
    "R_Elbow",     # 19
    "L_Wrist",     # 20
    "R_Wrist",     # 21
    "L_Hand",      # 22
    "R_Hand",      # 23
]

# Metrab 到 SMPLX 的关节映射（索引 -> smplx关节名）
METRAB_TO_SMPLX_MAP = {
    0: "pelvis",           # Pelvis
    1: "left_hip",         # L_Hip
    2: "right_hip",       # R_Hip
    3: "torso",          # Torso
    4: "left_knee",        # L_Knee
    5: "right_knee",      # R_Knee
    6: "spine2",          # Spine
    7: "left_ankle",      # L_Ankle
    8: "right_ankle",     # R_Ankle
    9: "spine3",          # Chest
    10: "left_foot",      # L_Toe
    11: "right_foot",     # R_Toe
    12: "neck",           # Neck
    13: "left_collar",    # L_Thorax
    14: "right_collar",   # R_Thorax
    15: "head",           # Head
    16: "left_shoulder",  # L_Shoulder
    17: "right_shoulder", # R_Shoulder
    18: "left_elbow",     # L_Elbow
    19: "right_elbow",    # R_Elbow
    20: "left_wrist",     # L_Wrist
    21: "right_wrist",    # R_Wrist
    22: "left_wrist",     # L_Hand (使用left_wrist)
    23: "right_wrist",    # R_Hand (使用right_wrist)
}



def process_metrab_data(j3d):
    """
    处理metrab数据到smplx所需关节点
    j3d: shape [num_people, 24, 3] - 多个人的24个关节3D位置
    返回: smplx格式的字典，包含位置和旋转信息（旋转使用单位四元数）
    只返回 human_scale_table 中需要的关节
    """
    # 取第一个人的数据
    j3d = j3d[0]  # shape: [24, 3]
    
    # 坐标系转换矩阵：从metrab坐标系转换到smplx/机器人坐标系
    # metrab坐标可能需要绕X轴旋转90度（向上转90度）来匹配smplx坐标系
    # 绕X轴旋转90度：Y轴变成Z轴，Z轴变成-Y轴
    rotation_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    
    
    # 单位四元数（wxyz格式，scalar-first）
    default_quat = np.array([1.0, 0.0, 0.0, 0.0]) 
    
    # 只保留 human_scale_table 中需要的关节
    required_joints = {
        "pelvis", "torso",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_foot", "right_foot",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist"
    }
    
    # 存储所需关节的位置和旋转
    smplx_data = {}
    for idx, smplx_name in METRAB_TO_SMPLX_MAP.items():
        if smplx_name in required_joints:
            pos = j3d[idx]
            # 应用坐标系转换到位置
            pos_transformed = pos @ rotation_matrix.T
            # 所有关节使用单位四元数（旋转已经在配置文件的offset中处理）
            smplx_data[smplx_name] = (pos_transformed, default_quat.copy())
    
    return smplx_data
if __name__ == "__main__":
    #启动websocket客户端

    
    
    
    HERE = pathlib.Path(__file__).parent
    parser = argparse.ArgumentParser(description="metrab to robot")
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1", 
                 "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", 
                 "berkeley_humanoid_lite", "booster_k1", "pnd_adam_lite", "tienkung",
                 "unitree_g1_fixed_wrist", "unitree_g1_fixed_wrist_metrabs"],
        default="unitree_g1_fixed_wrist",
    )

    parser.add_argument(
        "--record_video",
        action="store_true",
        help="Record video",
    )
    parser.add_argument(
        "--rate_limit",
        action="store_true",
        help="Limit frame rate",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="192.168.3.9",
        help="Metrab server address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Metrab server port",
    )
    args = parser.parse_args()
    
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 初始化retargeting系统
    actual_human_height = 1.6
    print(f"Initializing GMR for {args.robot}...")
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )
    
    # 初始化前向运动学模型
    print(f"Initializing KinematicsModel for {args.robot}...")
    kinematics_model = KinematicsModel(retarget.xml_file, device=device, extend_hand=True, extend_head=True)
    
    # 定义需要发送的关节名称（与 mocap_to_robot.py 保持一致）
    body_names = kinematics_model.body_names
    sub_local_body_pos_names = ['pelvis', 'left_knee_link', 'left_ankle_roll_link', 'right_knee_link', 'right_ankle_roll_link', 
                                'left_shoulder_pitch_link', 'left_elbow_link', 'right_shoulder_pitch_link', 'right_elbow_link', 
                                'left_hand_link', 'right_hand_link', 'head_link']
    sub_local_body_pos_id = [body_names.index(name) for name in sub_local_body_pos_names]
    print(f"Selected body parts: {sub_local_body_pos_names}")
    
    # 初始化 Viewer
    viewer = RobotMotionViewer(robot_type=args.robot, motion_fps=30, transparent_robot=0, record_video=args.record_video)
    
    # 初始化滤波器
    # beta: 推荐 0.2-0.5。如果感觉移动有延迟，改大；如果感觉机器人抖，改小。
    smoother = OneEuroSmoother(min_cutoff=0.01, beta=0.3, freq=30)
    
    # 初始化 WebSocket 服务器（用于向机器人端推送数据）
    web_server = RobotDataServer(port=8080)  # 使用不同的端口避免冲突
    web_server.start()
    
    # 初始化用于速度计算的变量
    prev_global_body_pos = None
    prev_get_data_time = None
    
    # 启动 metrab 客户端
    client = RobotDataClient(server=args.server, port=args.port)
    thread = client.start()
    
    try:
        while True:
            data = client.get_latest_data()
            
            if data is not None:
                # 获取接收到的数据到j3d
                j3d = np.array(data.get('j3d', []))
                if j3d.size == 0:
                    time.sleep(0.01)
                    continue
                
                # 处理数据从metrab到smplx所需关节点
                smplx_data = process_metrab_data(j3d)
                
                # 执行 Retarget (IK 解算)
                qpos = retarget.retarget(smplx_data)
                
                # 滤波平滑 (One Euro Filter)
                qpos = smoother.process(qpos)
                
                # --- 前向运动学计算（FK）---
                # 准备 qpos 数据用于前向运动学
                qpos_list = []
                qpos_list.append(qpos.copy())
                qpos_list = np.array(qpos_list)
                root_pos = qpos_list[:, :3]
                root_rot = qpos_list[:, 3:7]
                # 转换四元数格式：wxyz -> xyzw
                root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]
                dof_pos = qpos_list[:, 7:]
                
                # 使用单位根位置和旋转进行前向运动学计算
                identity_root_pos = torch.zeros((1, 3), device=device)
                identity_root_rot = torch.zeros((1, 4), device=device)
                identity_root_rot[:, -1] = 1.0
                
                # 执行前向运动学
                local_body_pos, _ = kinematics_model.forward_kinematics(
                    identity_root_pos, 
                    torch.from_numpy(root_rot).to(device=device, dtype=torch.float), 
                    torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
                )
                local_body_pos_np = local_body_pos[0].cpu().numpy() if isinstance(local_body_pos, torch.Tensor) else local_body_pos[0]
                
                # 提取所需关节的局部位置
                sub_local_body_pos = local_body_pos_np[sub_local_body_pos_id]
                # 计算全局位置
                global_body_pos = sub_local_body_pos + root_pos[0]
                
                # --- 将最低点对齐到地面 ---
                # 找到所有关节位置中的最低 Z 值
                ankle_height = 0.02  # 脚踝高度，保持脚踝在地面上方 0.02 米
                lowest_height = np.min(global_body_pos[:, 2])  # 找到最低的 Z 坐标
                
                # 调整所有关节的 Z 坐标，使得最低点（脚踝）保持在地面上方 ankle_height 米
                height_adjustment = lowest_height - ankle_height
                global_body_pos[:, 2] -= height_adjustment
                
                # 同时调整 root_pos，以便后续计算保持一致
                root_pos[0, 2] -= height_adjustment
                
                # 同步更新 qpos 的高度，确保可视化时使用调整后的高度
                qpos[2] = root_pos[0, 2]
                
                # --- 计算速度 ---
                current_get_data_time = time.time()
                if prev_global_body_pos is not None and prev_get_data_time is not None:
                    # 计算调用间隔作为时间差
                    s_dt = current_get_data_time - prev_get_data_time
                    
                    # 将numpy数组转换为torch tensor
                    prev_ref_body_pos_tensor = torch.from_numpy(prev_global_body_pos).to(device=device, dtype=torch.float).cpu()
                    ref_body_tensor = torch.from_numpy(global_body_pos).to(device=device, dtype=torch.float).cpu()
                    
                    # 计算速度: shape [12, 3] -> [1, 2, 12, 3]
                    stacked_pos = torch.stack([prev_ref_body_pos_tensor[None, ...], ref_body_tensor[None, ...]], dim=1)
                    ref_body_vel = compute_velocity(
                        stacked_pos, 
                        time_delta=s_dt, 
                        guassian_filter=False
                    )[0, 1]  # 提取当前时间点的速度: [12, 3]
                    ref_body_vel = ref_body_vel.cpu().numpy()
                else:
                    # 第一次调用，速度为零
                    ref_body_vel = np.zeros_like(sub_local_body_pos)
                
                # 更新上一次的global_body_pos和调用时间
                prev_global_body_pos = global_body_pos.copy()
                prev_get_data_time = current_get_data_time
                
                # --- 更新机器人数据（用于 WebSocket 推送）---
                update_robot_data(global_body_pos, ref_body_vel)
                
                # --- 驱动可视化 ---
                viewer.step(
                    root_pos=qpos[:3],
                    root_rot=qpos[3:7],
                    dof_pos=qpos[7:],
                    human_motion_data=retarget.scaled_human_data,
                    show_human_body_name=False,
                    rate_limit=args.rate_limit,
                )
            
            time.sleep(0.01)  # 控制循环频率（100Hz）
            
    except KeyboardInterrupt:
        print("\n正在关闭客户端...")
        client.stop()
        viewer.close()
    

    
    