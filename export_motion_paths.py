#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出不同标签的motion_path列表脚本
"""

import json
from collections import defaultdict

def load_annotations(file_path):
    """加载标注文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def export_motion_paths(annotations, output_prefix='motion_paths'):
    """按标签导出motion_path列表"""
    
    # 按标签分类
    label_paths = defaultdict(list)
    
    for key, item in annotations.items():
        label = item['label']
        motion_path = item['motion_path']
        
        # 替换路径：data/out/ -> data/AMASS/
        new_path = motion_path.replace('data/out/', 'retargeted_data/AMASS/')
        
        label_paths[label].append(new_path)
    
    # 为每个标签保存文件
    for label, paths in label_paths.items():
        output_file = f"{output_prefix}_{label}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for path in paths:
                f.write(path + '\n')
        
        print(f"✓ 已导出 {label} 标签的 {len(paths)} 条路径到: {output_file}")
    
    return label_paths

def main():
    # 文件路径
    annotation_file = 'annotations.json'
    output_prefix = 'motion_paths'
    
    print(f"正在加载标注文件: {annotation_file}")
    annotations = load_annotations(annotation_file)
    
    print(f"正在导出motion_path列表...")
    label_paths = export_motion_paths(annotations, output_prefix)
    
    print("\n" + "=" * 60)
    print("导出完成！统计信息：")
    print("=" * 60)
    for label, paths in sorted(label_paths.items()):
        print(f"  {label}: {len(paths)} 条路径")
    print("=" * 60)

if __name__ == '__main__':
    main()

