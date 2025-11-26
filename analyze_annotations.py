#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标注数据统计分析脚本
"""

import json
from collections import defaultdict, Counter
from datetime import datetime
import os

def load_annotations(file_path):
    """加载标注文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_annotations(annotations):
    """分析标注数据"""
    
    # 统计所有标签类型
    label_counter = Counter()
    for item in annotations.values():
        label_counter[item['label']] += 1
    
    # 基础统计
    total_count = len(annotations)
    normal_count = label_counter.get('normal', 0)
    abnormal_count = label_counter.get('abnormal', 0)
    
    # 异常原因统计
    reason_counter = Counter()
    for item in annotations.values():
        if item['label'] == 'abnormal' and item['reason']:
            reason_counter[item['reason']] += 1
    
    # 按数据集统计
    dataset_stats = defaultdict(lambda: defaultdict(int))
    for key, item in annotations.items():
        dataset = key.split('/')[0]  # 例如 "GRAB"
        dataset_stats[dataset][item['label']] += 1
    
    # 按主体ID统计（如 s1, s2 等）
    subject_stats = defaultdict(lambda: defaultdict(int))
    for key, item in annotations.items():
        parts = key.split('/')
        if len(parts) >= 2:
            subject = parts[1]  # 例如 "s1"
            subject_stats[subject][item['label']] += 1
    
    # 按对象类型统计（如 airplane, apple 等）
    object_stats = defaultdict(lambda: defaultdict(int))
    for key, item in annotations.items():
        parts = key.split('/')
        if len(parts) >= 3:
            filename = parts[2]
            obj_name = filename.split('_')[0]  # 获取对象名称
            object_stats[obj_name][item['label']] += 1
    
    # 按动作类型统计（如 fly, eat, pass 等）
    action_stats = defaultdict(lambda: defaultdict(int))
    for key, item in annotations.items():
        parts = key.split('/')
        if len(parts) >= 3:
            filename = parts[2].replace('_stageii.pkl', '')
            # 提取动作类型（对象名称之后的部分）
            parts_name = filename.split('_')
            if len(parts_name) >= 2:
                action = parts_name[1]  # 获取动作
                action_stats[action][item['label']] += 1
    
    # 按日期统计
    date_stats = defaultdict(lambda: defaultdict(int))
    for item in annotations.values():
        if item['timestamp']:
            date = item['timestamp'].split('T')[0]  # 获取日期部分
            date_stats[date][item['label']] += 1
    
    return {
        'total': total_count,
        'normal': normal_count,
        'abnormal': abnormal_count,
        'label_distribution': label_counter,
        'reasons': reason_counter,
        'datasets': dataset_stats,
        'subjects': subject_stats,
        'objects': object_stats,
        'actions': action_stats,
        'dates': date_stats
    }

def print_statistics(stats):
    """打印统计结果"""
    
    print("=" * 80)
    print("标注数据统计分析报告")
    print("=" * 80)
    print()
    
    # 1. 总体统计
    print("【1. 总体统计】")
    print(f"  总标注数量: {stats['total']}")
    print(f"  标签分布:")
    for label, count in sorted(stats['label_distribution'].items(), key=lambda x: x[1], reverse=True):
        print(f"    {label}: {count} ({count/stats['total']*100:.2f}%)")
    print()
    
    # 2. 异常原因分布
    print("【2. 异常原因分布】")
    if stats['reasons']:
        for reason, count in sorted(stats['reasons'].items(), key=lambda x: x[1], reverse=True):
            percentage = count / stats['abnormal'] * 100
            print(f"  {reason}: {count} ({percentage:.2f}%)")
    else:
        print("  无异常原因记录")
    print()
    
    # 3. 按数据集统计
    print("【3. 按数据集统计】")
    for dataset in sorted(stats['datasets'].keys()):
        data = stats['datasets'][dataset]
        total = sum(data.values())
        print(f"  {dataset}:")
        print(f"    总数: {total}")
        for label in sorted(data.keys()):
            count = data[label]
            print(f"    {label}: {count} ({count/total*100:.2f}%)")
    print()
    
    # 4. 按主体ID统计（Top 10）
    print("【4. 按主体ID统计 (Top 10)】")
    subject_list = []
    for subject, data in stats['subjects'].items():
        total = sum(data.values())
        subject_list.append((subject, total, data))
    subject_list.sort(key=lambda x: x[1], reverse=True)
    
    for subject, total, data in subject_list[:10]:
        print(f"  {subject}:")
        label_str = ", ".join([f"{label}: {count} ({count/total*100:.2f}%)" 
                               for label, count in sorted(data.items())])
        print(f"    总数: {total}, {label_str}")
    print()
    
    # 5. 按对象类型统计（Top 10）
    print("【5. 按对象类型统计 (Top 10)】")
    object_list = []
    for obj, data in stats['objects'].items():
        total = sum(data.values())
        object_list.append((obj, total, data))
    object_list.sort(key=lambda x: x[1], reverse=True)
    
    for obj, total, data in object_list[:10]:
        print(f"  {obj}:")
        label_str = ", ".join([f"{label}: {count} ({count/total*100:.2f}%)" 
                               for label, count in sorted(data.items())])
        print(f"    总数: {total}, {label_str}")
    print()
    
    # 6. 按动作类型统计（Top 15）
    print("【6. 按动作类型统计 (Top 15)】")
    action_list = []
    for action, data in stats['actions'].items():
        total = sum(data.values())
        action_list.append((action, total, data))
    action_list.sort(key=lambda x: x[1], reverse=True)
    
    for action, total, data in action_list[:15]:
        print(f"  {action}:")
        label_str = ", ".join([f"{label}: {count} ({count/total*100:.2f}%)" 
                               for label, count in sorted(data.items())])
        print(f"    总数: {total}, {label_str}")
    print()
    
    # 7. 按日期统计
    print("【7. 按日期统计】")
    for date in sorted(stats['dates'].keys()):
        data = stats['dates'][date]
        total = sum(data.values())
        print(f"  {date}:")
        label_str = ", ".join([f"{label}: {count}" for label, count in sorted(data.items())])
        print(f"    总数: {total}, {label_str}")
    print()
    
    print("=" * 80)

def save_statistics_to_file(stats, output_file):
    """将统计结果保存到文件"""
    import sys
    from io import StringIO
    
    # 捕获打印输出
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    print_statistics(stats)
    
    output = sys.stdout.getvalue()
    sys.stdout = old_stdout
    
    # 保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output)
    
    print(f"统计结果已保存到: {output_file}")

def main():
    # 文件路径
    annotation_file = 'annotations.json'
    output_file = 'annotation_statistics.txt'
    
    print(f"正在加载标注文件: {annotation_file}")
    annotations = load_annotations(annotation_file)
    
    print(f"正在分析数据...")
    stats = analyze_annotations(annotations)
    
    print(f"正在生成统计报告...")
    print()
    
    # 打印统计结果
    print_statistics(stats)
    
    # 保存到文件
    save_statistics_to_file(stats, output_file)

if __name__ == '__main__':
    main()

