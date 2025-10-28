#!/usr/bin/env python3
"""
HTTP客户端示例 - 使用 requests 拉取数据
这是您之前一直使用的方式
"""

import requests
import time
import json

def get_robot_data(server="localhost", port=8080):
    """
    通过HTTP GET方式拉取机器人数据
    
    参数:
        server: 服务器地址
        port: 服务器端口
    
    返回:
        dict: 机器人数据字典，如果失败返回None
    """
    url = f"http://{server}:{port}/get_robot_data"
    
    try:
        res = requests.get(url, timeout=1.0)
        res.raise_for_status()  # 检查HTTP错误
        data = res.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return None


def main():
    """主函数 - 示例用法"""
    server = "localhost"
    port = 8080
    
    print(f"连接到服务器: http://{server}:{port}/get_robot_data")
    print("开始拉取数据...\n")
    
    try:
        while True:
            # 拉取数据
            data = get_robot_data(server, port)
            
            if data is not None:
                print(f"时间戳: {data['timestamp']:.3f}")
                print(f"DOF数量: {len(data['dof_pos'])}")
                print(f"根节点位置: {data['root_pos']}")
                print(f"根节点旋转: {data['root_rot']}")
                print("-" * 50)
                
                # TODO: 在这里添加您的机器人控制代码
                # 例如:
                # robot.set_dof_positions(data['dof_pos'])
                # robot.set_root_pose(data['root_pos'], data['root_rot'])
            
            # 控制拉取频率（Hz）
            time.sleep(0.02)  # 50Hz
            
    except KeyboardInterrupt:
        print("\n正在退出...")


if __name__ == "__main__":
    main()

