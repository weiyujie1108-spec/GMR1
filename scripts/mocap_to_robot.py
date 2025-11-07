from general_motion_retargeting.optitrack_vendor.NatNetClient import setup_optitrack
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
import threading
import argparse
import time
from mocap_api import *
import numpy as np
import general_motion_retargeting.utils.lafan_vendor.utils as utils
from scipy.spatial.transform import Rotation as R
import torch
import asyncio
import aiohttp
from aiohttp import web
import json
import scipy.ndimage.filters as filters
from lpf import ActionFilterButter, ActionFilterExp, ActionFilterButterTorch

def get_event_type_name(event_type_value):
    """
    Convert event type value to corresponding enum name
    
    Args:
        event_type_value: The numeric event type value
        
    Returns:
        str: The corresponding event type name
    """
    event_type_map = {
        MCPEventType.InvalidEvent: 'InvalidEvent',
        MCPEventType.AvatarUpdated: 'AvatarUpdated',
        MCPEventType.TrackerUpdated: 'TrackerUpdated',
        MCPEventType.AliceIMUUpdated: 'AliceIMUUpdated',
        MCPEventType.AliceRigidbodyUpdated: 'AliceRigidbodyUpdated',
        MCPEventType.AliceTrackerUpdated: 'AliceTrackerUpdated',
        MCPEventType.AliceMarkerUpdated: 'AliceMarkerUpdated',
    }
    return event_type_map.get(event_type_value, f'Unknown({event_type_value})')

robot_local_body_pos = np.zeros([12, 3])
robot_timestamp = 0.0

async def get_robot_data(request):
    """
    HTTP GET处理器
    """
    global robot_local_body_pos, robot_body_vel, robot_timestamp
    data = {
        "local_body_pos": robot_local_body_pos.tolist(),
        "body_vel": robot_body_vel.tolist(),
        "timestamp": robot_timestamp
    }
    return web.json_response(data)

def update_robot_data(local_body_pos, ref_body_vel):
    global robot_local_body_pos, robot_body_vel, robot_timestamp
    
    # 转换torch tensor为nrime array
    if isinstance(local_body_pos, torch.Tensor):
        local_body_pos = local_body_pos.cpu().numpy()
    if isinstance(ref_body_vel, torch.Tensor):
        ref_body_vel = ref_body_vel.cpu().numpy()
    
    robot_local_body_pos = local_body_pos
    robot_body_vel = ref_body_vel
    robot_timestamp = time.time()

class RobotDataServer:
    """
    机器人数据Web服务器
    """
    
    def __init__(self, port=8080):
        self.port = port
        self.app = None
        self.runner = None
        self.site = None
        self.active_websockets = set()  # 存储所有活跃的WebSocket连接
    
    async def websocket_handler(self, request):
        """
        WebSocket处理器 - 实时推送模式
        自动向所有连接的客户端推送最新的机器人数据
        """
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
        """
        启动Web服务器
        """
        self.app = web.Application()
        
        # 添加路由
        self.app.router.add_route('GET', '/get_robot_data', get_robot_data)
        self.app.router.add_route('GET', '/ws', self.websocket_handler)
        
        # 启动服务器
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, '0.0.0.0', self.port)
        await self.site.start()
        print(f'机器人数据服务器已启动: http://localhost:{self.port}')
        print(f'  - HTTP GET: http://localhost:{self.port}/get_robot_data')
        print(f'  - WebSocket: ws://localhost:{self.port}/ws')
    
    def start(self):
        """
        在新线程中启动Web服务器
        """
        def run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.start_server())
            loop.run_forever()
        
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        # 等待服务器启动
        time.sleep(0.5)


class MocapStream:
    """
    Mocap Hybrid Data Server Demo class for demonstrating how to get Hybrid Data Server data through Mocap API
    """
    
    def __init__(self):
        """
        Initialize Mocap HDS Demo instance
        """
        self.app = None
        self.running = False
    
    def start(self, udp_port=7012):
        """
        Start Mocap application and handle event loop
        
        Args:
            udp_port: UDP port number, default is 7012
        """
        # Initialize Mocap application
        self.app = MCPApplication()
        settings = MCPSettings()
        settings.set_udp(udp_port)
        settings.set_bvh_rotation(MCPBvhRotation.XYZ)
        self.app.set_settings(settings)
        self.app.open()
        print(f"Mocap application initialized, UDP port: {udp_port}")
        
        self.running = True
        try:
            while self.running:
                evts = self.app.poll_next_event()
                for evt in evts:
                    if evt.event_type == MCPEventType.AvatarUpdated: # avatar bvh
                        self._handle_avatar_data(evt)
                    elif evt.event_type == MCPEventType.AliceTrackerUpdated: # tracker
                        self._handle_tracker_data()
                    elif evt.event_type == MCPEventType.AliceMarkerUpdated: # marker
                        self._handle_marker_data()
                    else:
                        print('Other events:', get_event_type_name(evt.event_type))
                time.sleep(0.001)
        except KeyboardInterrupt:
            print("Program interrupted by user")
        finally:
            self.stop()

    def _handle_marker_data(self):
        """
        Handle marker data
        """
        alicehub = MCPAliceHub()
        recv, count = alicehub.get_marker_list()
        if count > 0:
            recv, count1 = alicehub.get_marker_list(count)
            timestamp = alicehub.get_marker_timestamp()
            for i in range(count):
                marker_handle = recv[i]
                marker = MCPMarker(marker_handle)
                marker_x, marker_y, marker_z = marker.get_marker_position()
                print(f'marker data : timestamp: {timestamp}, position: {marker_x}, {marker_y}, {marker_z}')
    
    def _handle_tracker_data(self):
        """
        Handle tracker data
        """
        alicehub = MCPAliceHub()
        recv, count = alicehub.get_PWR_list()
        if count > 0:
            recv, count1 = alicehub.get_PWR_list(count)
            timestamp = alicehub.get_PWR_timestamp()
            for i in range(count):
                PWRHandle = recv[i]
                MCPPWRH = MCPPWR(PWRHandle)
                id = MCPPWRH.get_PWR_id()
                status = MCPPWRH.get_PWR_status()
                position = MCPPWRH.get_PWR_position()
                quaternion = MCPPWRH.get_PWR_quaternion()
                print('tracker data : timestamp',timestamp,'id:', id, 'status:', status, 'position:', position, 'quaternion:', quaternion)

    def _handle_avatar_data(self, evt):
        """
        Handle avatar data
        """
        avatar = MCPAvatar(evt.event_data.avatar_handle)
        self.avatar = avatar
        # # Get and print timecode information
        # second, nanosecond = avatar.get_avatar_posture_ptp_time()
        # print(f" avatar posture ptp time : {second},{nanosecond}")
        # joints = avatar.get_joints()  # Get all joint data
        # for joint in joints:
        #     link_name = joint.get_name()  # Get joint name
        #     position = joint.get_local_position()  # Get joint position
        #     rotation = joint.get_local_rotation()  # Get joint rotation
        #     print(f"avatar data : joint: {link_name}, position: {position}, rotation: {rotation}")
    
    def get_lafan1_data_from_avatar_data(self):
        """
        Get lafan1 data from avatar data
        """
        # mocap: bones names and parents (hardcoded)
        bones = ['Hips', 'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Neck1', 'Head', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightHandIndex', 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightHandMiddle', 'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 'RightHandRing', 'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightHandPinky', 'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3', 'LeftHandIndex', 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3', 'LeftHandMiddle', 'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 'LeftHandRing', 'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 'LeftHandPinky', 'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3']
        parents = np.array([-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  9, 10, 11,  9, 13, 14, 15,
                            16, 17, 18, 16, 20, 21, 22, 16, 24, 25, 26, 16, 28, 29, 30, 16, 32,
                            33, 34,  9, 36, 37, 38, 39, 40, 41, 39, 43, 44, 45, 39, 47, 48, 49,
                            39, 51, 52, 53, 39, 55, 56, 57])
        
        # mocap realtime data
        joints = self.avatar.get_joints()  # Get all joint data
        pos = np.zeros((len(joints), 3))
        quat = np.zeros((len(joints), 4))
        for i, joint in enumerate(joints):
            pos[i] = joint.get_local_position()  # Get joint position
            quat[i] = joint.get_local_rotation()  # Get joint rotation
    
        global_data = utils.quat_fk(quat, pos, parents)

        rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

        result = {}
        for i, bone in enumerate(bones):
            orientation = utils.quat_mul(rotation_quat, global_data[0][i])
            position = global_data[1][i] @ rotation_matrix.T / 100  # cm to m
            result[bone] = [position, orientation]
        result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftFoot"][1]]
        result["RightFootMod"] = [result["RightFoot"][0], result["RightFoot"][1]]
        return result
            
   
    def stop(self):
        """
        Close Mocap application
        """
        self.running = False
        if self.app:
            self.app.close()
            print("Mocap application closed")

def compute_velocity(p, time_delta, guassian_filter=True):
    velocity = np.gradient(p.numpy(), axis=-3) / time_delta
    if guassian_filter:
        velocity = torch.from_numpy(filters.gaussian_filter1d(velocity, 2, axis=-3, mode="nearest")).to(p)
    else:
        velocity = torch.from_numpy(velocity).to(p)

    return velocity


def main(args):
    # Check if firewall is disabled on this machine
    print("Make sure to disable firewall on both machines:")
    print("On OptiTrack computer: Disable Windows Firewall")
    print("On this computer: sudo ufw disable")

    # 初始化并启动Web服务器
    web_server = RobotDataServer(port=8080)
    web_server.start()

    mocap_stream = MocapStream()

    # start a thread to client.run()
    thread = threading.Thread(target=mocap_stream.start)
    thread.start()

    if not mocap_stream:
        print("Failed to setup Mocap Stream")
        exit(1)

    print(f"Mocap Stream connected: {mocap_stream.running}")
    print("Starting motion retargeting...")

    retarget = GMR(
            src_human="bvh_nokov",
            tgt_robot=args.robot,
            actual_human_height=1.75,
        )

    viewer = RobotMotionViewer(robot_type=args.robot)

    # env = RobotMotionViewer(robot_type=robot_type,
    #                     motion_fps=60.,
    #                     camera_follow=False,
    #                     record_video=args.record_video, video_path=args.video_path)

    # Initialize the forward kinematics
    device = "cuda:0"
    kinematics_model = KinematicsModel(retarget.xml_file, device=device, extend_hand=True, extend_head=True)

    # 初始化统计变量（局部变量）
    update_count = 0
    update_start_time = time.time()
    last_update_time = update_start_time
    
    # 用于统计各部分耗时
    time_get_data = []
    time_retarget = []
    time_forward_kinematics = []
    time_update_data = []
    # ["Pelvis", "L_Knee", "L_Ankle", "R_Knee", "R_Ankle", "L_Shoulder", "L_Elbow", "R_Shoulder", "R_Elbow", "L_Hand", "R_Hand", "Head"]
    body_names = kinematics_model.body_names
    sub_local_body_pos_names = ['pelvis', 'left_knee_link', 'left_ankle_roll_link', 'right_knee_link', 'right_ankle_roll_link', 
    'left_shoulder_pitch_link',  'left_elbow_link', 'right_shoulder_pitch_link', 'right_elbow_link', 'left_hand_link', 'right_hand_link', 'head_link']
    sub_local_body_pos_id = [body_names.index(name) for name in sub_local_body_pos_names]
    
    # 初始化用于速度计算的变量
    prev_global_body_pos = None
    prev_get_data_time = None
    pose_filt = False
    pose_cutfreq = 15
    if pose_filt:
        pose_filter = ActionFilterButterTorch(
        lowcut=np.zeros(12*3),  
        highcut=np.ones(12*3) * pose_cutfreq,  
        sampling_rate=30, 
        num_joints=12*3,
        device="cpu"
        )
    while True:
        # 1. 获取Mocap数据
        current_get_data_time = time.time()
        t0 = time.time()
        lafan1_data = mocap_stream.get_lafan1_data_from_avatar_data()
        t1 = time.time()
        time_get_data.append((t1 - t0) * 1000)  # 转换为毫秒
        
        # 2. 运动重定向
        t0 = time.time()
        qpos = retarget.retarget(lafan1_data)
        t1 = time.time()
        time_retarget.append((t1 - t0) * 1000)
        
        viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            rate_limit=False,
        )
        qpos_list = []
        qpos_list.append(qpos.copy())
        qpos_list = np.array(qpos_list)
        root_pos = qpos_list[:, :3]
        root_rot = qpos_list[:, 3:7]
        root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]
        dof_pos = qpos_list[:, 7:]
        identity_root_pos = torch.zeros((1, 3), device=device)
        identity_root_rot = torch.zeros((1, 4), device=device)
        identity_root_rot[:, -1] = 1.0
        
        # 3. 前向运动学计算
        t0 = time.time()
        local_body_pos, _ = kinematics_model.forward_kinematics(
                identity_root_pos, 
                 torch.from_numpy(root_rot).to(device=device, dtype=torch.float), 
                torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
            )
        local_body_pos_np = local_body_pos[0].cpu().numpy() if isinstance(local_body_pos, torch.Tensor) else local_body_pos[0]
        t1 = time.time()
        time_forward_kinematics.append((t1 - t0) * 1000)
        
        sub_local_body_pos = local_body_pos_np[sub_local_body_pos_id]
        global_body_pos = sub_local_body_pos + root_pos
        if pose_filt:
            global_body_pos = np.array(pose_filter.filter(torch.from_numpy(global_body_pos).reshape(12*3)).reshape(12, 3))
        # 计算每个关节点的速度
        if prev_global_body_pos is not None and prev_get_data_time is not None:
            # 计算调用间隔作为时间差
            s_dt = current_get_data_time - prev_get_data_time
            
            # 将numpy数组转换为torch tensor
            prev_ref_body_pos_tensor = torch.from_numpy(prev_global_body_pos).to(device=device, dtype=torch.float).cpu()
            ref_body_tensor = torch.from_numpy(global_body_pos).to(device=device, dtype=torch.float).cpu()
            
            # 计算速度: shape [12, 3] -> [1, 2, 12, 3]
            # 堆叠上一帧和当前帧的位置: [1, 12, 3] + [1, 12, 3] -> [1, 2, 12, 3]
            stacked_pos = torch.stack([prev_ref_body_pos_tensor[None, ...], ref_body_tensor[None, ...]], dim=1)
            ref_body_vel = compute_velocity(
                stacked_pos, 
                time_delta=s_dt, 
                guassian_filter=False
            )[0, 1]  # 提取当前时间点的速度: [12, 3]
        else:
            # 第一次调用，速度为零或未定义
            ref_body_vel = np.zeros_like(sub_local_body_pos)
        # print(ref_body_vel[0])
        # 更新上一次的global_body_pos和调用时间
        prev_global_body_pos = global_body_pos.copy()
        prev_get_data_time = current_get_data_time

        # 4. 更新机器人数据
        t0 = time.time()
        update_robot_data(
            global_body_pos,
            ref_body_vel,
        )
        t1 = time.time()
        time_update_data.append((t1 - t0) * 1000)
        
        # 统计更新频率
        update_count += 1
        current_time = time.time()
        
        if last_update_time is not None:
            interval = current_time - last_update_time
            elapsed_time = current_time - update_start_time
            
            # 每0.2秒打印一次频率和耗时统计
            if elapsed_time >= 0.2:
                update_freq = update_count / elapsed_time
                avg_interval = elapsed_time / update_count
                
                # 计算各部分的平均、最大、最小耗时（使用最近 update_count 次的数据）
                current_count = update_count  # 保存当前计数
                
                if len(time_get_data) >= current_count and current_count > 0:
                    avg_get_data = np.mean(time_get_data[-current_count:])
                    max_get_data = np.max(time_get_data[-current_count:])
                    min_get_data = np.min(time_get_data[-current_count:])
                else:
                    avg_get_data = max_get_data = min_get_data = 0
                    
                if len(time_retarget) >= current_count and current_count > 0:
                    avg_retarget = np.mean(time_retarget[-current_count:])
                    max_retarget = np.max(time_retarget[-current_count:])
                    min_retarget = np.min(time_retarget[-current_count:])
                else:
                    avg_retarget = max_retarget = min_retarget = 0
                    
                if len(time_forward_kinematics) >= current_count and current_count > 0:
                    avg_fk = np.mean(time_forward_kinematics[-current_count:])
                    max_fk = np.max(time_forward_kinematics[-current_count:])
                    min_fk = np.min(time_forward_kinematics[-current_count:])
                else:
                    avg_fk = max_fk = min_fk = 0
                    
                if len(time_update_data) >= current_count and current_count > 0:
                    avg_update = np.mean(time_update_data[-current_count:])
                    max_update = np.max(time_update_data[-current_count:])
                    min_update = np.min(time_update_data[-current_count:])
                else:
                    avg_update = max_update = min_update = 0
                
                total_time = avg_get_data + avg_retarget + avg_fk + avg_update
                
                print(f"\n=== 性能统计 (频率: {update_freq:.2f} Hz, 总耗时: {total_time:.2f} ms) ===")
                # print(f"1. 获取Mocap数据:    平均 {avg_get_data:.3f} ms  (最大 {max_get_data:.3f}, 最小 {min_get_data:.3f})")
                # print(f"2. 运动重定向:        平均 {avg_retarget:.3f} ms  (最大 {max_retarget:.3f}, 最小 {min_retarget:.3f})")
                # print(f"3. 前向运动学计算:    平均 {avg_fk:.3f} ms  (最大 {max_fk:.3f}, 最小 {min_fk:.3f})")
                # print(f"4. 更新机器人数据:    平均 {avg_update:.3f} ms  (最大 {max_update:.3f}, 最小 {min_update:.3f})")
                # print(f"   总循环时间:        {avg_interval*1000:.2f} ms")
                
                update_count = 0
                update_start_time = current_time
        
        last_update_time = current_time
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--server_ip", type=str, default="192.168.200.160")
    # parser.add_argument("--client_ip", type=str, default="192.168.200.117")
    # parser.add_argument("--use_multicast", type=bool, default=False)
    # # parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--robot", type=str, default="unitree_g1_fixed_wrist")
    args = parser.parse_args()
    main(args)
    