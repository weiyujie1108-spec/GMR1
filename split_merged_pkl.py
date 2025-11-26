#!/usr/bin/env python3
"""
从合并的 pkl 文件中按照标签列表分割成多个文件
"""
import os
import joblib
import argparse
from tqdm import tqdm


def path_to_key(file_path, base_dir="retargeted_data"):
    """
    将文件路径转换为 key 名称
    例如: retargeted_data/AMASS/GRAB/s1/airplane_fly_1_stageii.pkl 
    -> AMASS_GRAB_s1_airplane_fly_1_stageii
    """
    # 获取相对路径
    rel_path = str(file_path).strip()
    
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


def load_path_list(txt_file):
    """
    从文本文件中加载路径列表，并转换为 key 列表
    """
    keys = []
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                key = path_to_key(line)
                keys.append(key)
    return keys


def split_merged_pkl(merged_file, label_files, output_files, output_dir=None):
    """
    根据标签文件将合并的pkl分割成多个文件
    
    Args:
        merged_file: 合并的pkl文件路径
        label_files: 标签文件列表 [(名称, 文件路径), ...]
        output_files: 输出文件名列表 [文件名, ...]
        output_dir: 输出目录，如果为None则使用merged_file所在目录
    """
    # 加载合并的数据
    print(f"正在加载合并文件: {merged_file}")
    merged_data = joblib.load(merged_file)
    print(f"总共包含 {len(merged_data)} 条数据")
    print()
    
    # 如果没有指定输出目录，使用merged_file所在目录
    if output_dir is None:
        output_dir = os.path.dirname(merged_file)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个标签文件
    results = {}
    for (label_name, label_file), output_file in zip(label_files, output_files):
        print(f"处理标签: {label_name}")
        print(f"标签文件: {label_file}")
        
        # 加载标签对应的key列表
        keys = load_path_list(label_file)
        print(f"标签包含 {len(keys)} 个条目")
        
        # 提取对应的数据
        split_data = {}
        missing_keys = []
        
        for key in tqdm(keys, desc=f"提取 {label_name} 数据"):
            if key in merged_data:
                split_data[key] = merged_data[key]
            else:
                missing_keys.append(key)
        
        # 保存分割后的数据
        output_path = os.path.join(output_dir, output_file)
        print(f"保存到: {output_path}")
        joblib.dump(split_data, output_path)
        
        # 统计信息
        results[label_name] = {
            'total': len(keys),
            'found': len(split_data),
            'missing': len(missing_keys),
            'output_file': output_path,
            'file_size': os.path.getsize(output_path) / (1024*1024)
        }
        
        if missing_keys:
            print(f"警告: 有 {len(missing_keys)} 个key在合并文件中未找到")
            if len(missing_keys) <= 10:
                print("缺失的key:")
                for k in missing_keys:
                    print(f"  - {k}")
            else:
                print("前10个缺失的key:")
                for k in missing_keys[:10]:
                    print(f"  - {k}")
        
        print(f"成功: {len(split_data)}/{len(keys)} 条数据")
        print()
    
    # 输出总结
    print("=" * 60)
    print("分割完成！统计信息:")
    print("=" * 60)
    for label_name, info in results.items():
        print(f"\n{label_name}:")
        print(f"  - 标签条目数: {info['total']}")
        print(f"  - 找到数据: {info['found']}")
        print(f"  - 缺失数据: {info['missing']}")
        print(f"  - 输出文件: {info['output_file']}")
        print(f"  - 文件大小: {info['file_size']:.2f} MB")
    
    return results


def main():
    # 默认路径
    default_merged = "merged_motions/merged_retargeted_data.pkl"
    default_normal = "motion_paths_normal.txt"
    default_abnormal = "motion_paths_abnormal.txt"
    default_difficult = "motion_paths_difficult.txt"
    default_output_dir = "merged_motions/"
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='根据标签列表分割合并的pkl文件')
    parser.add_argument('-i', '--input', type=str, default=default_merged,
                        help=f'输入的合并pkl文件路径 (默认: {default_merged})')
    parser.add_argument('--normal', type=str, default=default_normal,
                        help=f'normal标签文件路径 (默认: {default_normal})')
    parser.add_argument('--abnormal', type=str, default=default_abnormal,
                        help=f'abnormal标签文件路径 (默认: {default_abnormal})')
    parser.add_argument('--difficult', type=str, default=default_difficult,
                        help=f'difficult标签文件路径 (默认: {default_difficult})')
    parser.add_argument('-o', '--output-dir', type=str, default=default_output_dir,
                        help=f'输出目录 (默认: {default_output_dir})')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件 {args.input} 不存在")
        return
    
    # 准备标签文件和输出文件
    label_files = [
        ("normal", args.normal),
        ("abnormal", args.abnormal),
        ("difficult", args.difficult)
    ]
    
    output_files = [
        "motions_normal.pkl",
        "motions_abnormal.pkl",
        "motions_difficult.pkl"
    ]
    
    # 检查标签文件是否都存在
    for label_name, label_file in label_files:
        if not os.path.exists(label_file):
            print(f"错误: 标签文件 {label_file} 不存在")
            return
    
    # 执行分割
    split_merged_pkl(args.input, label_files, output_files, args.output_dir)


if __name__ == "__main__":
    main()

