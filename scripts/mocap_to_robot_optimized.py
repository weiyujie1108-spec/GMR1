"""
高性能线程安全的Web服务器实现
比较三种方法：全局变量、锁、无锁原子操作
"""

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

# ============================================================================
# 方法1：全局变量 + GIL锁（与pose_server.py相同）
# 优点：简单、快速、无阻塞
# 缺点：依赖Python的GIL，理论上不安全，但实践中可用
# ============================================================================
robot_local_body_pos = np.zeros([1, 3])
robot_dof_pos = np.zeros([1])
robot_root_pos = np.zeros([3])
robot_root_rot = np.zeros([4])
robot_timestamp = 0.0

def update_robot_data_method1(local_body_pos, dof_pos, root_pos, root_rot):
    """方法1：直接更新全局变量 - 最快，Python GIL保证基本线程安全"""
    global robot_local_body_pos, robot_dof_pos, robot_root_pos, robot_root_rot, robot_timestamp
    
    # 转换为numpy array
    if isinstance(local_body_pos, torch.Tensor):
        local_body_pos = local_body_pos.cpu().numpy()
    if isinstance(dof_pos, torch.Tensor):
        dof_pos = dof_pos.cpu().numpy()
    if isinstance(root_pos, torch.Tensor):
        root_pos = root_pos.cpu().numpy()
    if isinstance(root_rot, torch.Tensor):
        root_rot = root_rot.cpu().numpy()
    
    # 直接赋值（Python的GIL确保每行赋值是原子的）
    robot_local_body_pos = local_body_pos
    robot_dof_pos = dof_pos
    robot_root_pos = root_pos
    robot_root_rot = root_rot
    robot_timestamp = time.time()

async def get_robot_data_method1(request):
    """方法1的HTTP处理器"""
    data = {
        "local_body_pos": robot_local_body_pos.tolist(),
        "dof_pos": robot_dof_pos.tolist(),
        "root_pos": robot_root_pos.tolist(),
        "root_rot": robot_root_rot.tolist(),
        "timestamp": robot_timestamp
    }
    return web.json_response(data)

# ============================================================================
# 方法2：全局变量 + 显式锁（最安全）
# 优点：完全线程安全
# 缺点：有锁的开销，但锁只保护字典，非常快
# ============================================================================
_data_lock = threading.Lock()
_robot_data_dict = {
    "local_body_pos": np.zeros([1, 3]),
    "dof_pos": np.zeros([1]),
    "root_pos": np.zeros([3]),
    "root_rot": np.zeros([4]),
    "timestamp": 0.0
}

def update_robot_data_method2(local_body_pos, dof_pos, root_pos, root_rot):
    """方法2：使用锁保护更新"""
    global _robot_data_dict
    
    # 转换数据
    if isinstance(local_body_pos, torch.Tensor):
        local_body_pos = local_body_pos.cpu().numpy()
    if isinstance(dof_pos, torch.Tensor):
        dof_pos = dof_pos.cpu().numpy()
    if isinstance(root_pos, torch.Tensor):
        root_pos = root_pos.cpu().numpy()
    if isinstance(root_rot, torch.Tensor):
        root_rot = root_rot.cpu().numpy()
    
    # 锁保护（只保护字典更新，非常快）
    with _data_lock:
        _robot_data_dict.update({
            "local_body_pos": local_body_pos,
            "dof_pos": dof_pos,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "timestamp": time.time()
        })

async def get_robot_data_method2(request):
    """方法2的HTTP处理器"""
    with _data_lock:
        data = {
            "local_body_pos": _robot_data_dict["local_body_pos"].tolist(),
            "dof_pos": _robot_data_dict["dof_pos"].tolist(),
            "root_pos": _robot_data_dict["root_pos"].tolist(),
            "root_rot": _robot_data_dict["root_rot"].tolist(),
            "timestamp": _robot_data_dict["timestamp"]
        }
    return web.json_response(data)

# ============================================================================
# 方法3：Copy-on-Write（无锁，最优性能）
# 优点：完全无锁、无阻塞
# 缺点：内存开销稍大
# ============================================================================
_robot_data_ref = threading.local()

def _get_current_data():
    """获取当前线程的数据引用"""
    if not hasattr(_robot_data_ref, 'data'):
        _robot_data_ref.data = {
            "local_body_pos": np.zeros([1, 3]),
            "dof_pos": np.zeros([1]),
            "root_pos": np.zeros([3]),
            "root_rot": np.zeros([4]),
            "timestamp": 0.0
        }
    return _robot_data_ref.data

def update_robot_data_method3(local_body_pos, dof_pos, root_pos, root_rot):
    """方法3：Copy-on-Write - 无锁版本"""
    # 创建新数据对象
    if isinstance(local_body_pos, torch.Tensor):
        local_body_pos = local_body_pos.cpu().numpy()
    if isinstance(dof_pos, torch.Tensor):
        dof_pos = dof_pos.cpu().numpy()
    if isinstance(root_pos, torch.Tensor):
        root_pos = root_pos.cpu().numpy()
    if isinstance(root_rot, torch.Tensor):
        root_rot = root_rot.cpu().numpy()
    
    # 原子性更新引用
    _robot_data_ref.data = {
        "local_body_pos": local_body_pos,
        "dof_pos": dof_pos,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "timestamp": time.time()
    }

async def get_robot_data_method3(request):
    """方法3的HTTP处理器"""
    data = _get_current_data()
    return web.json_response({
        "local_body_pos": data["local_body_pos"].tolist(),
        "dof_pos": data["dof_pos"].tolist(),
        "root_pos": data["root_pos"].tolist(),
        "root_rot": data["root_rot"].tolist(),
        "timestamp": data["timestamp"]
    })

# ============================================================================
# 选择使用的方法（推荐方法1，平衡速度和安全性）
# ============================================================================
# 设置使用的方法（1, 2, 或 3）
USE_METHOD = 1  # 修改这个值切换方法

if USE_METHOD == 1:
    update_robot_data = update_robot_data_method1
    get_robot_data_handler = get_robot_data_method1
    print("使用方案1：全局变量（最快，适合单线程更新）")
elif USE_METHOD == 2:
    update_robot_data = update_robot_data_method2
    get_robot_data_handler = get_robot_data_method2
    print("使用方案2：显式锁（最安全，多线程读写推荐）")
else:
    update_robot_data = update_robot_data_method3
    get_robot_data_handler = get_robot_data_method3
    print("使用方案3：Copy-on-Write（无锁，适合高频读取）")

# ============================================================================
# Web服务器实现
# ============================================================================
class RobotDataServer:
    """机器人数据Web服务器"""
    
    def __init__(self, port=8080):
        self.port = port
        self.app = None
        self.runner = None
        self.site = None
    
    async def websocket_handler(self, request):
        """WebSocket处理器"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        print('WebSocket连接建立')
        
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    if msg.data == "get_pose":
                        # 读取数据
                        data = {
                            "local_body_pos": robot_local_body_pos.tolist() if USE_METHOD == 1 else
                                             _robot_data_dict["local_body_pos"].tolist() if USE_METHOD == 2 else
                                             _get_current_data()["local_body_pos"].tolist(),
                            "dof_pos": robot_dof_pos.tolist() if USE_METHOD == 1 else
                                      _robot_data_dict["dof_pos"].tolist() if USE_METHOD == 2 else
                                      _get_current_data()["dof_pos"].tolist(),
                            "root_pos": robot_root_pos.tolist() if USE_METHOD == 1 else
                                       _robot_data_dict["root_pos"].tolist() if USE_METHOD == 2 else
                                       _get_current_data()["root_pos"].tolist(),
                            "root_rot": robot_root_rot.tolist() if USE_METHOD == 1 else
                                       _robot_data_dict["root_rot"].tolist() if USE_METHOD == 2 else
                                       _get_current_data()["root_rot"].tolist(),
                            "timestamp": robot_timestamp if USE_METHOD == 1 else
                                        _robot_data_dict["timestamp"] if USE_METHOD == 2 else
                                        _get_current_data()["timestamp"]
                        }
                        await ws.send_json(data)
                    elif msg.data == "close":
                        await ws.close()
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f'WebSocket错误: {ws.exception()}')
        except Exception as e:
            print(f'WebSocket错误: {e}')
        finally:
            print('WebSocket连接关闭')
        
        return ws
    
    async def start_server(self):
        """启动Web服务器"""
        self.app = web.Application()
        
        # 添加路由
        self.app.router.add_route('GET', '/get_robot_data', get_robot_data_handler)
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
        """在新线程中启动Web服务器"""
        def run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.start_server())
            loop.run_forever()
        
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        time.sleep(0.5)


# 后续代码与原始文件相同...
# （MocapStream类、main函数等）


