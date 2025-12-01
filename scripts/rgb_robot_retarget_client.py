### 运行指令: python robot_retarget_client_lqh.py --robot unitree_g1
import asyncio
import websockets
import pickle
import numpy as np
import argparse
import torch
import smplx
import os
import pathlib
import traceback
import time
import math
import threading
import scipy.ndimage.filters as filters
from scipy.spatial.transform import Rotation as R
from aiohttp import web

# === 引入 GMR 依赖 ===
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import get_smplx_data_offline_fast
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

async def run_client(uri, robot_name, model_root_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # === 1. 初始化 SMPL-X ===
    smplx_path = os.path.join(model_root_path, "smplx")
    print(f"Loading SMPL-X model from: {smplx_path}")
    try:
        body_model = smplx.SMPLX(
            model_path=smplx_path, gender='neutral', use_pca=False, batch_size=2
        ).to(device)
        print("✅ Body Model Loaded.")
    except Exception as e:
        print(f"❌ Error loading SMPL-X: {e}")
        return

    # === 2. 初始化 GMR ===
    print(f"Initializing GMR for {robot_name}...")
    retarget = GMR(actual_human_height=1.75, src_human="smplx", tgt_robot=robot_name)
    
    # === 3. 初始化前向运动学模型 ===
    print(f"Initializing KinematicsModel for {robot_name}...")
    kinematics_model = KinematicsModel(retarget.xml_file, device=device, extend_hand=True, extend_head=True)
    
    # 定义需要发送的关节名称（与 mocap_to_robot.py 保持一致）
    body_names = kinematics_model.body_names
    sub_local_body_pos_names = ['pelvis', 'left_knee_link', 'left_ankle_roll_link', 'right_knee_link', 'right_ankle_roll_link', 
                                'left_shoulder_pitch_link', 'left_elbow_link', 'right_shoulder_pitch_link', 'right_elbow_link', 
                                'left_hand_link', 'right_hand_link', 'head_link']
    sub_local_body_pos_id = [body_names.index(name) for name in sub_local_body_pos_names]
    print(f"Selected body parts: {sub_local_body_pos_names}")
    
    # === 4. 初始化 Viewer ===
    viewer = RobotMotionViewer(robot_type=robot_name, motion_fps=30, transparent_robot=0, record_video=False)
    
    # === 5. 初始化滤波器 ===
    # beta: 推荐 0.2-0.5。如果感觉移动有延迟，改大；如果感觉机器人抖，改小。
    smoother = OneEuroSmoother(min_cutoff=0.01, beta=0.3, freq=30)
    
    # === 6. 初始化 WebSocket 服务器（用于向机器人端推送数据）===
    web_server = RobotDataServer(port=8080)
    web_server.start()
    
    # 初始化用于速度计算的变量
    prev_global_body_pos = None
    prev_get_data_time = None
    
    # === 5. 初始化坐标系转换 ===
    # 定义从发送端坐标系到接收端期望坐标系的旋转
    # 发送端坐标系 (RealSense/HybrIK): X=Right, Y=Down, Z=Forward
    # 默认转换：从相机坐标系到标准坐标系 (X前Y左Z上)
    # 可以通过参数调整，例如: coord_transform_euler=[-90, 0, 90]
    coord_transform_euler = [0, 0, 0]  # 可以根据实际情况调整
    coord_transform_rot = R.from_euler('xyz', coord_transform_euler, degrees=True)
    print(f"坐标系转换: 欧拉角 {coord_transform_euler} 度 (绕X/Y/Z轴)")

    print(f"Connecting to {uri}...")
    async with websockets.connect(uri) as websocket:
        print("✅ Connected! Streaming motion...")
        
        try:
            while True:
                # --- A. 接收 ---
                data = await websocket.recv()
                params_np = pickle.loads(data)
                params_np['mocap_frame_rate'] = np.array(30.0)

                # === 坐标系转换：在传入 SMPL-X 之前统一坐标系 ===
                # 发送端坐标系 (RealSense/HybrIK): X=Right, Y=Down, Z=Forward
                # 接收端期望坐标系 (SMPL-X/GMR): 需要根据实际情况定义
                # 
                # 使用函数开始时定义的坐标系转换旋转
                
                # 1. 转换 root_orient (根节点旋转)
                if 'root_orient' in params_np and params_np['root_orient'].size > 0:
                    root_orient_rodrigues = params_np['root_orient'][0]  # (3,) 轴角表示
                    root_orient_rot = R.from_rotvec(root_orient_rodrigues)
                    # 应用坐标系转换：新旋转 = 坐标系转换旋转 * 原始旋转
                    root_orient_new = coord_transform_rot * root_orient_rot
                    params_np['root_orient'][0] = root_orient_new.as_rotvec()
                
                # 2. 转换 trans (平移向量)
                if 'trans' in params_np and params_np['trans'].size > 0:
                    trans_original = params_np['trans'][0]  # (3,)
                    # 平移向量也需要在正确的坐标系下：应用旋转矩阵
                    trans_new = coord_transform_rot.apply(trans_original)
                    params_np['trans'][0] = trans_new
                
                # Batch Size Hack (GMR 内部需要 batch>1 避免维度压缩 bug)
                real_batch_size = 2 
                keys_to_expand = ['betas', 'root_orient', 'pose_body', 'trans']
                for key in keys_to_expand:
                    if key in params_np:
                        params_np[key] = np.concatenate([params_np[key], params_np[key]], axis=0)

                missing_keys_dims = [('pose_jaw', 3), ('pose_eye', 6), ('pose_hand', 90), ('expression', 10)]
                for key, dim in missing_keys_dims:
                    if key not in params_np:
                        params_np[key] = np.zeros((real_batch_size, dim), dtype=np.float32)
                    else:
                        params_np[key] = np.concatenate([params_np[key], params_np[key]], axis=0)
                
                # --- B. 计算 SMPL (GPU) ---
                smplx_data_batch = to_tensor(params_np, device=device)
                with torch.no_grad():
                    smplx_output = body_model(
                        betas=smplx_data_batch['betas'],
                        global_orient=smplx_data_batch['root_orient'],
                        body_pose=smplx_data_batch['pose_body'],
                        transl=smplx_data_batch['trans'], # HybrIK 原始 Camera 坐标 (Z=Depth)
                        jaw_pose=smplx_data_batch['pose_jaw'],
                        leye_pose=smplx_data_batch['pose_eye'][:, :3],
                        reye_pose=smplx_data_batch['pose_eye'][:, 3:],
                        left_hand_pose=smplx_data_batch['pose_hand'][:, :45],
                        right_hand_pose=smplx_data_batch['pose_hand'][:, 45:],
                        expression=smplx_data_batch['expression'],
                        return_full_pose=True 
                    )

                # --- C. 提取 Targets (CPU) ---
                smplx_output_cpu = CPUOutput(smplx_output)
                
                target_frames, _ = get_smplx_data_offline_fast(
                    smplx_data_batch, body_model, smplx_output_cpu, tgt_fps=30         
                )
                current_frame_target = target_frames[0]
                
                # --- D. 执行 Retarget (IK 解算) ---
                # 注意：此时 IK 算出来的 qpos[:3] 是基于 HybrIK 原始坐标的，方向是错的
                qpos = retarget.retarget(current_frame_target)
                
                # --- E. Client 端后处理 (关键修正逻辑) ---
                # 注意：前面的坐标系转换已经将发送端的 root_orient 和 trans 转换到了 SMPL-X 坐标系
                # 这里需要进一步将 SMPL-X/GMR 坐标系转换到机器人坐标系

                # 1. 姿态 (Rotation) 修正：从 SMPL-X 坐标系转换到机器人坐标系
                # 将 SMPL 躺着的姿态转正（如果需要的话，根据实际情况调整）
                r_robot = R.from_quat(qpos[3:7][[1, 2, 3, 0]]) # wxyz -> xyzw
                r_fix = R.from_euler('x', -90, degrees=True) 
                r_new = r_fix * r_robot
                qpos[3:7] = r_new.as_quat()[[3, 0, 1, 2]]      # xyzw -> wxyz
                
                # 2. 位移 (Translation) 驱动：从 SMPL-X 坐标系转换到机器人坐标系
                # 注意：trans 已经在前面进行了坐标系转换，这里使用的是转换后的值
                # Robot World Frame:   X=Forward, Y=Left, Z=Up
                
                raw_trans = smplx_data_batch['trans'][0].cpu().numpy()  # 已经是转换后的坐标系
                
                # [调试参数]
                depth_offset = 2.0   # 原点设定：假设人站在离摄像头 2.0 米处时，机器人在 X=0
                scale_forward = 1.5  # 前后移动系数
                scale_side = 1.5     # 左右移动系数
                
                # [映射 1] Robot X (前后) <--- HybrIK Z (深度)
                # 逻辑: (原始深度 - 2.0米) * 系数
                # 假设人靠近(Z变小)，Robot X 应该变大(向前) -> 取反: (offset - z)
                # 假设人走远(Z变大)，Robot X 应该变小(向后)
                # 如果发现方向反了，把下面的式子改成: (raw_trans[2] - depth_offset) * scale_forward
                qpos[1] = -(depth_offset - raw_trans[2]) * scale_forward
                
                # [映射 2] Robot Y (左右) <--- HybrIK X (左右)
                # 逻辑: HybrIK 向右是 +X，Robot 向左是 +Y。
                # 所以通常取负号： -X * scale
                qpos[0] = -raw_trans[0] * scale_side

                # [映射 3] Robot Z (高度) <--- 固定值
                # 强制锁死高度，防止深度估计的误差被映射为“飞天/钻地”
                qpos[2] = 0.9-raw_trans[1]*1  # 根据场景调整，0.75 是一个比较常规的数值 

                # 3. 滤波平滑 (One Euro Filter)
                # 平滑 HybrIK 的深度噪声
                qpos = smoother.process(qpos)

                # --- F. 前向运动学计算（FK）---
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
                
                # --- F.1. 将最低点对齐到地面 ---
                # 找到所有关节位置中的最低 Z 值（假设没有跳跃，最低点通常是脚部）
                ankle_height = 0.02  # 脚踝高度，保持脚踝在地面上方 0.02 米
                lowest_height = np.min(global_body_pos[:, 2])  # 找到最低的 Z 坐标
                
                # 调整所有关节的 Z 坐标，使得最低点（脚踝）保持在地面上方 ankle_height 米
                # 方法：将所有关节向下平移 (lowest_height - ankle_height)
                height_adjustment = lowest_height - ankle_height
                global_body_pos[:, 2] -= height_adjustment
                
                # 同时调整 root_pos，以便后续计算保持一致
                root_pos[0, 2] -= height_adjustment
                
                # 同步更新 qpos 的高度，确保可视化时使用调整后的高度
                qpos[2] = root_pos[0, 2]
                
                # --- G. 计算速度 ---
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
                
                # --- H. 更新机器人数据（用于 WebSocket 推送）---
                update_robot_data(global_body_pos, ref_body_vel)

                # --- I. 驱动可视化 ---
                viewer.step(
                    root_pos=qpos[:3], # 使用修正后的 X,Y,Z
                    root_rot=qpos[3:7],
                    dof_pos=qpos[7:],
                    human_motion_data=retarget.scaled_human_data,
                    show_human_body_name=False,
                    rate_limit=False 
                )
                
                await asyncio.sleep(0.001)

        except websockets.exceptions.ConnectionClosed:
            print("❌ Server disconnected")
        except KeyboardInterrupt:
            print("Stopping...")
        except Exception:
            traceback.print_exc()
        finally:
            viewer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("--robot", type=str, default="unitree_g1")
    
    current_dir = pathlib.Path(__file__).parent.resolve()
    default_model_path = current_dir.parent / "assets" / "body_models"
    
    parser.add_argument("--model_path", type=str, 
                        default=str(default_model_path),
                        help="Path to the folder containing 'smplx' directory")
    
    args = parser.parse_args()
    print(f"Using SMPL-X model path: {args.model_path}")
    print(f"Connecting to WebSocket server at: ws://{args.ip}:8765")
    
    asyncio.run(run_client(f"ws://{args.ip}:8765", args.robot, args.model_path))