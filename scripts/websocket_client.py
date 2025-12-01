#!/usr/bin/env python3
import websocket
import json
import numpy as np
import threading
import time

class RobotDataClient:
    
    def __init__(self, server="localhost", port=8080, enable_freq_stats=True):
        self.server = server
        self.port = port
        self.url = f"ws://{server}:{port}/ws"
        self.ws = None
        self.running = False
        
        # 最新接收到的数据
        self.latest_data = None
        self.data_lock = threading.Lock()
        
        # 服务端数据发送频率统计（在on_message中记录，反映服务端实际发送频率）
        self.enable_freq_stats = enable_freq_stats
        self.receive_times = []  # 记录每次接收数据的时间戳（服务端发送时间）
        self.receive_count = 0  # 接收数据计数
        self.max_receive_times = 1000  # 最多保留1000条记录
        self.stats_lock = threading.Lock()
    
    def on_message(self, ws, message):
        """接收消息回调（服务端发送数据时触发）"""
        try:
            data = json.loads(message)
            receive_time = time.time()  # 记录服务端发送数据的时间
            
            # 更新最新数据
            with self.data_lock:
                self.latest_data = data
            
            # 记录接收时间（用于统计服务端实际发送数据的频率）
            if self.enable_freq_stats:
                with self.stats_lock:
                    self.receive_times.append(receive_time)
                    self.receive_count += 1
                    # 限制列表大小
                    if len(self.receive_times) > self.max_receive_times:
                        self.receive_times.pop(0)
            
            # 可选：打印接收信息（注释掉以减少打印开销）
            # print(f"[{time.time():.3f}] 收到数据 - 时间戳: {data['timestamp']:.3f}")
            
        except Exception as e:
            print(f"解析数据错误: {e}")
    
    def on_error(self, ws, error):
        print(f"WebSocket错误: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        self.running = False
        print("WebSocket连接已关闭")
    
    def on_open(self, ws):
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
        """获取最新的数据（线程安全）
        
        注意：这个方法只是读取数据，不会触发频率统计。
        频率统计是在 on_message 回调中记录的，反映的是服务端实际发送数据的频率。
        """
        with self.data_lock:
            return self.latest_data
    
    def get_send_freq_stats(self):
        """获取服务端数据发送频率统计信息
        
        注意：这是服务端实际发送数据的频率，不是客户端读取数据的频率。
        即使客户端多次调用 get_latest_data()，只要没有新消息到达，频率统计不会变化。
        
        Returns:
            dict: 包含频率统计信息的字典，包括：
                - instant_freq: 瞬时频率（Hz），基于最近两次接收间隔
                - avg_freq: 平均频率（Hz），基于所有记录
                - recent_avg_freq: 最近1秒的平均频率（Hz）
                - total_count: 总接收次数（服务端发送次数）
                - last_10_avg_freq: 最近10次接收的平均频率（Hz）
        """
        if not self.enable_freq_stats or len(self.receive_times) < 2:
            return {
                "instant_freq": 0.0,
                "avg_freq": 0.0,
                "recent_avg_freq": 0.0,
                "total_count": self.receive_count,
                "last_10_avg_freq": 0.0
            }
        
        with self.stats_lock:
            current_time = time.time()
            
            # 计算瞬时频率（最近两次接收间隔）
            if len(self.receive_times) >= 2:
                instant_interval = self.receive_times[-1] - self.receive_times[-2]
                instant_freq = 1.0 / instant_interval if instant_interval > 0 else 0.0
            else:
                instant_freq = 0.0
            
            # 计算平均频率（基于所有记录）
            if len(self.receive_times) >= 2:
                total_time = self.receive_times[-1] - self.receive_times[0]
                avg_freq = (len(self.receive_times) - 1) / total_time if total_time > 0 else 0.0
            else:
                avg_freq = 0.0
            
            # 计算最近1秒的平均频率
            recent_times = [t for t in self.receive_times if current_time - t <= 1.0]
            if len(recent_times) >= 2:
                recent_avg_freq = (len(recent_times) - 1) / (recent_times[-1] - recent_times[0])
            else:
                recent_avg_freq = 0.0
            
            # 计算最近10次接收的平均频率
            if len(self.receive_times) >= 10:
                last_10_intervals = [self.receive_times[i] - self.receive_times[i-1] 
                                    for i in range(len(self.receive_times)-9, len(self.receive_times))]
                avg_interval_last_10 = sum(last_10_intervals) / len(last_10_intervals)
                last_10_avg_freq = 1.0 / avg_interval_last_10 if avg_interval_last_10 > 0 else 0.0
            else:
                last_10_avg_freq = 0.0
            
            return {
                "instant_freq": instant_freq,
                "avg_freq": avg_freq,
                "recent_avg_freq": recent_avg_freq,
                "total_count": self.receive_count,
                "last_10_avg_freq": last_10_avg_freq
            }
    
    def print_send_freq_stats(self):
        """打印服务端数据发送频率统计信息"""
        stats = self.get_send_freq_stats()
        print(f"[WebSocket服务端数据发送频率统计]")
        print(f"  总接收次数（服务端发送次数）: {stats['total_count']}")
        print(f"  瞬时频率: {stats['instant_freq']:.2f} Hz")
        print(f"  平均频率: {stats['avg_freq']:.2f} Hz")
        print(f"  最近1秒平均频率: {stats['recent_avg_freq']:.2f} Hz")
        print(f"  最近10次平均频率: {stats['last_10_avg_freq']:.2f} Hz")
    
    def stop(self):
        """停止客户端"""
        self.running = False
        if self.ws:
            self.ws.close()


def main():
    # 创建客户端
    client = RobotDataClient(server="192.168.3.9", port=8080)
    
    # 在后台启动
    thread = client.start()
    
    try:
        # 模拟机器人控制循环
        while True:
            data = client.get_latest_data()
            
            if data is not None:
                # 获取接收到的数据
                # 数据格式: {
                #     "j3d": [[[x, y, z], ...], ...],  # [5, 24, 3] 5个人的24个关节3D位置
                #     "dt": 0.033,                        # 时间间隔
                # }
                
                j3d = np.array(data.get('j3d', []))
                print(j3d.shape)# shape: [5, 24, 3]
                dt = data.get('dt', 0.0)                 # 时间间隔
                
                # 打印数据（可选）
                print(f"[{time.time():.3f}] 收到数据 - dt: {dt:.4f}")
                # if len(j3d) > 0:
                    # print(f"  第一个人根关节位置: {j3d[0, 0]}")  # 打印第一个人的根关节位置
                
                # TODO: 在这里添加您的机器人控制代码
                # 例如:
                # robot.set_joint_positions(j3d[0])  # 使用第一个人的关节位置
                # 或者根据您的需求处理多个人的数据
            
            time.sleep(0.01)  # 控制循环频率（100Hz）
            
    except KeyboardInterrupt:
        print("\n正在关闭客户端...")
        client.stop()


if __name__ == "__main__":
    main()

