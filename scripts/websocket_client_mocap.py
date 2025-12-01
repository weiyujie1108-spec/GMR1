#!/usr/bin/env python3
"""
WebSocket客户端 - 实时接收机器人数据
用于机器人模仿任务
"""

import websocket
import json
import numpy as np
import threading
import time

class RobotDataClient:
    """机器人数据客户端"""
    
    def __init__(self, server="localhost", port=8080):
        self.server = server
        self.port = port
        self.url = f"ws://{server}:{port}/ws"
        self.ws = None
        self.running = False
        
        # 最新接收到的数据
        self.latest_data = None
        self.data_lock = threading.Lock()
    
    def on_message(self, ws, message):
        """接收到消息时的回调函数"""
        try:
            data = json.loads(message)
            
            # 更新最新数据
            with self.data_lock:
                self.latest_data = data
            
            # 可选：打印接收信息（注释掉以减少打印开销）
            # print(f"[{time.time():.3f}] 收到数据 - 时间戳: {data['timestamp']:.3f}")
            
        except Exception as e:
            print(f"解析数据错误: {e}")
    
    def on_error(self, ws, error):
        """错误回调"""
        print(f"WebSocket错误: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """连接关闭回调"""
        self.running = False
        print("WebSocket连接已关闭")
    
    def on_open(self, ws):
        """连接建立后的回调"""
        print(f"WebSocket连接已建立: {self.url}")
        print("开始接收实时数据...")
    
    def connect(self):
        """连接到WebSocket服务器"""
        print(f"连接到: {self.url}")
        
        self.ws = websocket.WebSocketApp(
            self.url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        
        self.running = True
        # 在新线程中运行WebSocket
        self.ws.run_forever()
    
    def start(self):
        """在后台线程中启动客户端"""
        thread = threading.Thread(target=self.connect, daemon=True)
        thread.start()
        # 等待连接建立
        time.sleep(0.5)
        return thread
    
    def get_latest_data(self):
        """获取最新的数据（线程安全）"""
        with self.data_lock:
            return self.latest_data
    
    def stop(self):
        """停止客户端"""
        self.running = False
        if self.ws:
            self.ws.close()


def main():
    """主函数 - 示例用法"""
    # 创建客户端
    client = RobotDataClient(server="localhost", port=8080)
    
    # 在后台启动
    thread = client.start()
    
    try:
        # 模拟机器人控制循环
        while True:
            data = client.get_latest_data()
            
            if data is not None:
                # 获取接收到的数据
                # 数据格式: {
                #     "local_body_pos": [[x, y, z], ...],  # 12个身体部位的局部位置
                #     "body_vel": [[vx, vy, vz], ...],      # 12个身体部位的速度
                #     "timestamp": 123456789.123            # 时间戳
                # }
                
                local_body_pos = np.array(data['local_body_pos'])  # shape: [12, 3]
                body_vel = np.array(data['body_vel'])              # shape: [12, 3]
                timestamp = data['timestamp']
                
                # 打印数据（可选）
                print(f"[{time.time():.3f}] 收到数据 - 时间戳: {timestamp:.3f}")
                print(f"  身体位置: {local_body_pos[0]}")  # 打印第一个部位的位置
                print(f"  身体速度: {body_vel[0]}")        # 打印第一个部位的速度
                
                # TODO: 在这里添加您的机器人控制代码
                # 例如:
                # robot.set_body_positions(local_body_pos)
                # robot.set_body_velocities(body_vel)
            
            time.sleep(0.01)  # 控制循环频率（100Hz）
            
    except KeyboardInterrupt:
        print("\n正在关闭客户端...")
        client.stop()


if __name__ == "__main__":
    main()

