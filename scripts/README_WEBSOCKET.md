# 实时数据传输使用指南

## 概述

该系统实现了实时人体姿态数据传输，支持两种方式：
1. **HTTP GET**：简单的请求-响应模式，适合现有代码
2. **WebSocket**：实时推送模式，适合高性能场景

两种方式都支持多个客户端同时连接，让多个机器人同步模仿同一人体姿态。

## 架构说明

```
人体姿态数据 → [服务器] → HTTP GET   → [客户端1: 机器人1] 
                   ↓         ↓
              WebSocket 广播 → [客户端2: 机器人2] 
                                  ↓ 
                           [客户端3: 机器人3]
```

## 服务器端

### 启动服务器
```bash
python scripts/mocap_to_robot.py
```

服务器将启动在 `http://localhost:8080`，提供两个接口：
- **HTTP GET**: `http://localhost:8080/get_robot_data` - 请求-响应模式，每次请求返回最新数据
- **WebSocket**: `ws://localhost:8080/ws` - 实时推送模式，自动推送最新数据

### 工作原理

1. 服务器通过 Mocap 接收人体姿态数据
2. 将人体姿态转换为机器人关节角度
3. 以约 100Hz 的频率自动推送数据给所有连接的客户端
4. 支持多个客户端同时连接

## 客户端

### 方式1：HTTP GET（推荐用于现有代码）

使用 `requests` 库，这是您之前一直使用的方式：

```python
import requests
import time

while True:
    url = f"http://localhost:8080/get_robot_data"
    res = requests.get(url)
    data = res.json()
    
    # 使用数据进行机器人控制
    robot.set_dof_positions(data['dof_pos'])
    robot.set_root_pose(data['root_pos'], data['root_rot'])
    
    time.sleep(0.02)  # 50Hz
```

**优点**：
- 使用简单，无需安装额外依赖
- 与现有代码兼容
- 每次请求获取最新数据

**缺点**：
- 需要定期轮询
- 相比 WebSocket 有更高的延迟和开销

### 方式2：WebSocket（推荐用于高性能场景）

#### 安装依赖
```bash
pip install websocket-client
```

#### 使用方法
```python
from scripts.websocket_client import RobotDataClient

# 创建客户端
client = RobotDataClient(server="localhost", port=8080)

# 启动
client.start()

# 获取数据
while True:
    data = client.get_latest_data()
    if data:
        # 使用数据进行机器人控制
        joint_positions = data['dof_pos']
        root_pos = data['root_pos']
        root_rot = data['root_rot']
        
        # 控制您的机器人
        robot.set_joint_positions(joint_positions)
    
    time.sleep(0.01)
```

#### 方式2：在机器人控制程序中使用
```python
from scripts.websocket_client import RobotDataClient
import time

class RobotController:
    def __init__(self):
        self.client = RobotDataClient(server="your_server_ip", port=8080)
        self.client.start()
    
    def update(self):
        """机器人控制循环"""
        data = self.client.get_latest_data()
        
        if data is not None:
            # 获取关节位置
            q = np.array(data['dof_pos'])
            
            # 获取根节点位姿
            root_pos = np.array(data['root_pos'])
            root_rot = np.array(data['root_rot'])
            
            # 应用到机器人
            self.robot.set_dof_positions(q)
            self.robot.set_root_pose(root_pos, root_rot)
    
    def run(self):
        while True:
            self.update()
            time.sleep(0.01)  # 100Hz
```

## 数据格式

### 接收到的数据格式
```json
{
    "local_body_pos": [...],      // 局部身体位置
    "dof_pos": [...],             // 机器人自由度位置（关节角度）
    "root_pos": [x, y, z],        // 根节点位置
    "root_rot": [w, x, y, z],     // 根节点旋转（四元数）
    "timestamp": 123456789.123    // 时间戳
}
```

## 多个客户端同时连接

系统支持多个机器人同时连接到同一个服务器：

```python
# 机器人1
client1 = RobotDataClient(server="192.168.1.100", port=8080)
client1.start()

# 机器人2
client2 = RobotDataClient(server="192.168.1.100", port=8080)
client2.start()

# 它们将同时接收到相同的人体姿态数据
```

## 性能参数

### HTTP GET 方式
- **延迟**: 每次请求约 10-50ms
- **开销**: 每次请求建立新的TCP连接
- **适用场景**: 低频查询、现有代码兼容

### WebSocket 方式
- **推送频率**: 约 100Hz (每 10ms 推送一次)
- **延迟**: 通常 < 50ms
- **开销**: 保持持久连接
- **适用场景**: 高频更新、高性能需求

### 两者共同特点
- **并发**: 支持多个客户端同时连接
- **带宽**: 取决于 DOF 数量，一般 < 10KB/s

## 故障排除

### 连接失败
- 检查服务器是否已启动
- 检查防火墙设置
- 确认 IP 地址和端口正确

### 数据延迟
- 降低推送频率（修改 `await asyncio.sleep(0.01)` 中的值）
- 检查网络延迟

### 数据丢失
- 确保网络稳定
- 检查客户端处理速度

## 示例：两个机器人模仿同一人体

```python
# 服务器端（在一台机器上运行）
python scripts/mocap_to_robot.py

# 客户端1（在第一台机器人上运行）
python scripts/robot1_client.py  # 连接到服务器

# 客户端2（在第二台机器人上运行）
python scripts/robot2_client.py  # 连接到服务器
```

两台机器人将实时同步模仿同一个人的动作！

## 选择哪种方式？

### 使用 HTTP GET 如果：
- ✅ 您已有使用 requests 的代码
- ✅ 不需要很高的更新频率（< 50Hz）
- ✅ 希望保持简单，无需额外依赖
- ✅ 只是偶尔查询数据

### 使用 WebSocket 如果：
- ✅ 需要高频率更新（> 50Hz）
- ✅ 需要最低延迟
- ✅ 多个机器人需要实时同步
- ✅ 能接受额外的 websocket-client 依赖

### 混合使用：
您也可以让不同的机器人使用不同的方式：
```python
# 机器人1使用HTTP GET
robot1_client = HTTPClient(server=server, port=8080)

# 机器人2使用WebSocket
robot2_client = RobotDataClient(server=server, port=8080)
```

两种方式都会从同一个服务器获取相同的数据！

