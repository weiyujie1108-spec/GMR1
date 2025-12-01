#!/usr/bin/env python3
"""
合并 retargeted_data 目录下的所有 pkl 文件到一个文件中

"""
import os
import pickle
import joblib
import argparse
from pathlib import Path
from tqdm import tqdm


def get_key_name(file_path, base_dir="retargeted_data"):
    """
    将文件路径转换为 key 名称
    例如: retargeted_data/AMASS/GRAB/s1/airplane_fly_1_stageii.pkl 
    -> AMASS_GRAB_s1_airplane_fly_1_stageii
    """
    # 获取相对路径
    rel_path = str(file_path)
    
    # 去掉 base_dir 前缀
    if rel_path.startswith(base_dir + "/"):
        rel_path = rel_path[len(base_dir) + 1:]
    elif rel_path.startswith(base_dir):
        rel_path = rel_path[len(base_dir):]
    
    # 去掉 .pkl 后缀
    if rel_path.endswith(".pkl"):
        rel_path = rel_path[:-4]
    
    # 将 / 替换为 _
    key_name = rel_path.replace("/", "_")
    
    return key_name


def find_all_pkl_files(root_dir):
    """
    递归查找所有 pkl 文件
    """
    pkl_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.pkl'):
                pkl_files.append(os.path.join(root, file))
    return sorted(pkl_files)


def merge_pkl_files(input_dir, output_file):
    """
    合并所有 pkl 文件到一个字典中
    """
    # 查找所有 pkl 文件
    print(f"正在查找 {input_dir} 目录下的所有 pkl 文件...")
    pkl_files = find_all_pkl_files(input_dir)
    print(f"找到 {len(pkl_files)} 个 pkl 文件")
    
    # 合并所有数据
    merged_data = {}
    failed_files = []
    
    print("正在合并文件...")
    for pkl_file in tqdm(pkl_files):
        try:
            # 读取 pkl 文件 (使用 joblib)
            data = joblib.load(pkl_file)
            
            # 生成 key 名称
            key_name = get_key_name(pkl_file, input_dir)
            
            # 如果data是字典且只有一个key，提取内层数据
            if isinstance(data, dict) and len(data) == 1:
                inner_key = list(data.keys())[0]
                data = data[inner_key]
            
            # 存储到合并的字典中
            if key_name in merged_data:
                print(f"\n警告: key '{key_name}' 已存在，将被覆盖")
            
            merged_data[key_name] = data
            
        except Exception as e:
            print(f"\n错误: 无法读取文件 {pkl_file}: {e}")
            failed_files.append(pkl_file)
    
    # 保存合并后的数据
    print(f"\n正在保存合并后的数据到 {output_file}...")
    joblib.dump(merged_data, output_file)
    
    print(f"\n合并完成!")
    print(f"- 成功合并: {len(merged_data)} 个文件")
    print(f"- 失败: {len(failed_files)} 个文件")
    
    if failed_files:
        print("\n失败的文件列表:")
        for f in failed_files:
            print(f"  - {f}")
    
    # 显示一些示例 key
    print("\n示例 key 名称:")
    for i, key in enumerate(list(merged_data.keys())[:5]):
        print(f"  - {key}")
    
    return merged_data, failed_files


def main():

    default_input = "retargeted_data/mocap_1127"
    default_output = "merged_motions/merged_retargeted_data_mocap_1127.pkl"
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='合并 pkl 文件到一个文件中')
    parser.add_argument('-i', '--input', type=str, default=default_input,
                        help=f'输入目录路径，包含所有要合并的 pkl 文件 (默认: {default_input})')
    parser.add_argument('-o', '--output', type=str, default=default_output,
                        help=f'输出文件路径，合并后的 pkl 文件保存位置 (默认: {default_output})')
    
    args = parser.parse_args()
    
    input_dir = os.path.abspath(args.input)
    output_file = os.path.abspath(args.output)
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录 {input_dir} 不存在")
        return
    
    print(f"输入目录: {input_dir}")
    print(f"输出文件: {output_file}")
    print()
    
    # 合并文件
    merged_data, failed_files = merge_pkl_files(input_dir, output_file)
    
    # 输出统计信息
    print(f"\n最终统计:")
    print(f"- 输出文件: {output_file}")
    print(f"- 文件大小: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    print(f"- 包含条目数: {len(merged_data)}")


if __name__ == "__main__":
    main()

