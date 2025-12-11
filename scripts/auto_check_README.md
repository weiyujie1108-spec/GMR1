# Auto Check Robot Motions

自动检测机器人运动数据中的异常情况，包括浮动、关节限制违反、关节不连续和自碰撞等问题。

## 功能概述

`auto_check.py` 是一个自动化工具，用于批量检测机器人运动数据（`.pkl` 文件）中的异常情况。它可以检测以下问题：

1. **浮动检测 (Floating Detection)**: 检测双脚同时离地的情况
2. **关节限制违反 (Joint Limit Violation)**: 检测关节角度是否超出机器人限制（当前已禁用）
3. **关节不连续 (Joint Discontinuity)**: 检测关节角度是否存在异常跳变
4. **自碰撞检测 (Self-Collision Detection)**: 检测机器人自身部件之间的碰撞
5. **抖动检测 (Jitter Detection)**: 检测关节角度或根位置的来回抖动（基于总变分和净位移）



## 使用方法
### 基本用法
```bash
# 检测单个文件夹中的所有运动文件
python scripts/auto_check.py --motion_folder data/out --robot_type unitree_g1
```

### 检测单个文件
```bash
# 检测单个运动文件（只输出异常结果）
python scripts/auto_check.py --motion_file data/out/GRAB/s1/airplane_fly_1_stageii.pkl --robot_type unitree_g1
```

### 完整参数示例

```bash
python scripts/auto_check.py \
    --motion_folder data/out \
    --robot_type unitree_g1 \
    --height_threshold 0.12 \
    --min_seconds 1.0 \
    --min_ratio 0.25 \
    --output_json auto_check_reports.json
```

## 命令行参数
### 必需参数

- `--motion_folder`: 包含机器人运动 `.pkl` 文件的文件夹路径（与 `--motion_file` 二选一，但在文件夹模式下必需）
- `--robot_type`: 机器人类型（必需）
- `--motion_file`: 单个运动文件路径（与 `--motion_folder` 二选一，单文件模式下使用）

### 可选参数

#### 机器人相关
- `--robot_type`: 机器人类型（必需参数）
  - 支持的机器人类型取决于 `assets/` 目录中的模型
  - 常见类型: `unitree_g1`, `unitree_h1` 等

#### 浮动检测参数
- `--height_threshold`: 离地高度阈值（米，默认: `0.12`，即 12cm）
- `--min_seconds`: 最小连续离地时间（秒，默认: `1.0`）
- `--min_ratio`: 最小离地帧数比例（默认: `0.25`，即 25%）
- `--foot_names`: 脚部链接名称列表（可选，默认使用预设列表）

#### 关节检测参数
- `--joint_limit_tolerance`: 关节限制违反容差（弧度，默认: `0.005`，约 1.15 度）
- `--min_joint_violation_ratio`: 最小关节违反帧数比例（默认: `0.05`，即 5%）
- `--max_joint_jump`: 最大允许关节跳变（弧度，默认: `0.5`，约 29 度）
- `--min_jump_ratio`: 最小跳变帧数比例（默认: `0.02`，即 2%）

**注意**: 关节不连续检测已改进，能够区分真正的跳跃和正常的大幅度快速动作（如跑跳）。改进包括：
- 加速度检查：真正的跳跃有突然的加速度变化
- 连续序列检查：跑跳动作通常有连续的大变化序列
- Jerk（加速度变化率）检查：真正的跳跃有突然的加速度变化

#### 抖动检测参数
- `--min_total_variation`: 最小总变分阈值（弧度，默认: `0.20`）
  - 窗口内总移动距离的最小值，低于此值不认为是抖动
- `--max_net_displacement`: 最大净位移阈值（弧度，默认: `0.02`）
  - 窗口内净位移的最大值，超过此值不认为是抖动（可能是单向运动）
- `--min_displacement_ratio`: 最小位移比例阈值（默认: `5.0`）
  - TV/D 的最小值，低于此值不认为是抖动
- `--max_step_size`: 最大单步位移阈值（弧度，默认: `0.04`）
  - 窗口内相邻帧之间最大位移，超过此值不认为是抖动（可能是正常快速运动）
- `--min_jitter_ratio`: 最小抖动帧比例（默认: `0.05`，即 5%）
  - 抖动帧数占总帧数的比例，低于此值不标记为异常
- `--oscillation_window_size`: 振荡检测窗口大小（帧数，默认: `10`）
  - 滑动窗口的大小，用于计算 TV、D 和 max_step

**注意**: 抖动检测已改进，增加了两个额外的判断条件以避免误判大幅度动作为抖动：
- **平均步长检查** (`MAX_AVG_STEP_SIZE = 0.025 rad`): 抖动应该有很小的平均步长
- **方向变化频率检查** (`MIN_DIRECTION_CHANGES = 3`): 抖动应该有频繁的方向变化（振荡模式）

这些阈值经过调优，能够有效区分抖动和正常快速运动（包括大幅度动作）。如果调整阈值，建议：
- 降低 `min_total_variation` 或 `min_displacement_ratio` → 更敏感，可能误检正常运动
- 提高 `max_net_displacement` 或 `max_step_size` → 更宽松，可能漏检轻微抖动

## 如何调整检测阈值

### 方法 1: 通过命令行参数（推荐）

大部分阈值可以通过命令行参数直接调整，无需修改代码：

```bash
# 调整浮动检测阈值
python scripts/auto_check.py \
    --motion_folder data/out \
    --robot_type unitree_g1 \
    --height_threshold 0.10 \        # 降低阈值，更敏感
    --min_seconds 0.8 \               # 降低最小连续时间
    --min_ratio 0.20                  # 降低最小比例

# 调整关节不连续检测阈值
python scripts/auto_check.py \
    --motion_folder data/out \
    --robot_type unitree_g1 \
    --max_joint_jump 0.4 \            # 降低阈值，更敏感
    --min_jump_ratio 0.03             # 降低最小比例

# 调整抖动检测阈值
python scripts/auto_check.py \
    --motion_folder data/out \
    --robot_type unitree_g1 \
    --min_total_variation 0.15 \      # 降低阈值，更敏感
    --max_net_displacement 0.015 \    # 降低阈值，更敏感
    --min_displacement_ratio 4.0 \    # 降低阈值，更敏感
    --max_step_size 0.03 \            # 降低阈值，更敏感
    --min_jitter_ratio 0.08           # 降低最小比例
```

### 方法 2: 修改代码中的常量

如果命令行参数不满足需求，可以直接修改 `scripts/auto_check.py` 文件中的常量定义：

#### 浮动检测阈值（第 40-42 行）

```python
OFF_GROUND_THRESHOLD = 0.12  # 离地高度阈值（米），默认 12cm
MIN_CONTINUOUS_SECONDS = 1.0  # 最小连续离地时间（秒）
MIN_CONTINUOUS_RATIO = 0.25  # 最小离地帧数比例（25%）
```

**调整建议**：
- 降低 `OFF_GROUND_THRESHOLD` → 更敏感，可能误检轻微离地
- 降低 `MIN_CONTINUOUS_SECONDS` → 更敏感，检测更短的离地时间
- 降低 `MIN_CONTINUOUS_RATIO` → 更敏感，检测更低的离地比例

#### 自碰撞检测阈值（第 44-54 行）

```python
SELF_COLLISION_MIN_SECONDS = 0.6  # 最小连续碰撞时间（秒）
SELF_COLLISION_MIN_RATIO = 0.10  # 最小碰撞帧数比例（10%）

SELF_COLLISION_CONTACT_DICT = {
    "default": {
        "pairs": [],  # 碰撞对白名单（空列表表示检测所有碰撞）
        "min_penetration": 0.003,  # 最小穿透深度（米），默认 3mm
    },
}
```

**调整建议**：
- 降低 `SELF_COLLISION_MIN_SECONDS` → 更敏感，检测更短的碰撞时间
- 降低 `SELF_COLLISION_MIN_RATIO` → 更敏感，检测更低的碰撞比例
- 降低 `min_penetration` → 更敏感，检测更轻微的碰撞

#### 关节不连续检测阈值（第 68-74 行）

```python
MAX_JOINT_JUMP = 0.5  # 最大允许关节跳变（弧度），约 29 度
MIN_JOINT_JUMP_RATIO = 0.02  # 最小跳变帧数比例（2%）
SEVERE_JOINT_JUMP = 1.0  # 严重跳变阈值（弧度），约 57 度（总是标记为异常）
MAX_JOINT_ACCELERATION = 20.0  # 最大允许关节加速度（rad/s²），用于区分跳跃和快速动作
MIN_CONSECUTIVE_LARGE_CHANGES = 3  # 最小连续大变化帧数，用于识别跑跳动作
MAX_ACCELERATION_CHANGE = 15.0  # 最大允许加速度变化率（jerk, rad/s³），真正的跳跃有突然的加速度变化
```

**调整建议**：
- 降低 `MAX_JOINT_JUMP` → 更敏感，检测更小的跳变
- 降低 `MIN_JOINT_JUMP_RATIO` → 更敏感，检测更低的跳变比例
- 降低 `SEVERE_JOINT_JUMP` → 更敏感，更早标记为严重跳变
- 提高 `MAX_JOINT_ACCELERATION` → 更宽松，允许更大的加速度（可能误判更多跑跳动作为正常）
- 降低 `MIN_CONSECUTIVE_LARGE_CHANGES` → 更敏感，更短的连续序列也会被认为是正常动作
- 提高 `MAX_ACCELERATION_CHANGE` → 更宽松，允许更大的加速度变化（可能误判更多跳跃为正常）

#### 抖动检测阈值

抖动检测阈值通过命令行参数传递，默认值在函数调用时设置。如果需要修改默认值，可以：

1. **通过命令行参数调整**（推荐）：
   ```bash
   --min_total_variation 0.15 \
   --max_net_displacement 0.015 \
   --min_displacement_ratio 4.0 \
   --max_step_size 0.03 \
   --min_jitter_ratio 0.08 \
   --oscillation_window_size 10
   ```

2. **修改代码中的默认值**：
   在 `auto_check.py` 的 `check_motion()` 或相关函数调用处，找到抖动检测函数的调用，修改默认参数值：
   ```python
   # 在函数调用处修改默认值
   detect_jitter(
       ...,
       min_total_variation=0.15,  # 修改默认值
       max_net_displacement=0.015,
       min_displacement_ratio=4.0,
       max_step_size=0.03,
       min_jitter_ratio=0.08,
       oscillation_window_size=10,
   )
   ```

**调整建议**：
- **更敏感（检测更多抖动）**：
  - 降低 `MIN_TOTAL_VARIATION`（如 0.15）
  - 降低 `MAX_NET_DISPLACEMENT`（如 0.015）
  - 降低 `MIN_DISPLACEMENT_RATIO`（如 4.0）
  - 降低 `MAX_STEP_SIZE`（如 0.03）
  - 降低 `MIN_JITTER_RATIO`（如 0.08）

- **更宽松（减少误检）**：
  - 提高 `MIN_TOTAL_VARIATION`（如 0.25）
  - 提高 `MAX_NET_DISPLACEMENT`（如 0.03）
  - 提高 `MIN_DISPLACEMENT_RATIO`（如 6.0）
  - 提高 `MAX_STEP_SIZE`（如 0.05）
  - 提高 `MIN_JITTER_RATIO`（如 0.15）

### 调整后的验证

修改阈值后，建议：

1. **在小数据集上测试**：先用少量文件测试，观察检测结果是否合理
2. **对比前后结果**：记录修改前后的检测结果，评估调整效果
3. **逐步调整**：不要一次性大幅修改，建议每次调整 10-20%
4. **关注误检和漏检**：
   - 如果误检太多（正常运动被标记为异常）→ 提高阈值
   - 如果漏检太多（异常运动未被检测）→ 降低阈值

### 阈值调优示例

```python
# 示例：更严格的抖动检测（检测更多抖动）
MIN_TOTAL_VARIATION = 0.15  # 从 0.20 降低到 0.15
MAX_NET_DISPLACEMENT = 0.015  # 从 0.02 降低到 0.015
MIN_DISPLACEMENT_RATIO = 4.0  # 从 5.0 降低到 4.0
MAX_STEP_SIZE = 0.03  # 从 0.04 降低到 0.03
MIN_JITTER_RATIO = 0.08  # 从 0.10 降低到 0.08

# 示例：更宽松的浮动检测（减少误检）
OFF_GROUND_THRESHOLD = 0.15  # 从 0.12 提高到 0.15
MIN_CONTINUOUS_SECONDS = 1.5  # 从 1.0 提高到 1.5
MIN_CONTINUOUS_RATIO = 0.30  # 从 0.25 提高到 0.30
```

#### 批量处理和断点续传参数
- `--batch_size`: 批量处理大小（默认: `100`）
  - 每批处理的文件数，处理完一批后会清理内存
  - 如果内存不足，可以减小此值（如 50 或 20）
- `--no_save_progress`: 禁用进度保存（默认: 启用）
  - 如果禁用，程序中断后无法从断点继续
- `--progress_file`: 进度文件路径（默认: `auto_check_progress.json`）
  - 保存处理进度的文件，用于断点续传

#### 输出参数
- `--output_json`: 输出 JSON 报告文件路径（默认: `auto_check_reports.json`，仅包含异常数据）

## 检测标准

### 浮动检测

运动被标记为异常（浮动）如果满足以下任一条件：

1. **连续离地时间过长**: 双脚同时离地（高度 > `height_threshold`）的连续时间 > `min_seconds` 秒
2. **离地比例过高**: 双脚同时离地的帧数比例 > `min_ratio`

### 关节不连续检测

关节不连续检测已改进，能够区分真正的跳跃和正常的大幅度快速动作（如跑跳）。

#### 检测原理

对于每个大角度变化（> `max_joint_jump`），系统会检查：

1. **加速度检查**: 真正的跳跃有突然的加速度变化（> `MAX_JOINT_ACCELERATION = 20.0 rad/s²`）
2. **连续序列检查**: 如果连续 ≥ 3 帧都有大变化，更可能是正常快速动作（跑跳）
3. **Jerk 检查**: 真正的跳跃有突然的加速度变化（jerk > `MAX_ACCELERATION_CHANGE = 15.0 rad/s³`）
4. **速度连续性检查**: 正常快速动作的速度变化应该是平滑的

#### 检测条件

运动被标记为异常（关节不连续）如果满足以下任一条件：

1. **跳变比例过高**: 真正的跳跃帧数比例 > `min_jump_ratio`（默认 2%）
2. **严重跳变**: 存在严重跳变（> `severe_joint_jump`，默认 1.0 弧度，约 57 度）

**注意**: 正常的大幅度快速动作（如跑跳）不会被误判为异常，因为：
- 它们有平滑的加速度变化
- 它们通常是连续的大变化序列
- 它们的速度变化是连续的

### 自碰撞检测

运动被标记为异常（自碰撞）如果满足以下任一条件：

1. **连续碰撞时间过长**: 自碰撞的连续时间 > 0.6 秒
2. **碰撞比例过高**: 自碰撞的帧数比例 > 10%

### 抖动检测

抖动检测用于识别关节角度或根位置的**来回振荡**（back-and-forth oscillation）模式。这种模式的特点是：总移动距离大，但净位移小，且单步位移小。

#### 检测原理

抖动检测基于**四条件联合判断**，使用滑动窗口分析：

1. **总变分（Total Variation, TV）**: 窗口内所有相邻帧之间位移的绝对值之和
   ```
   TV = Σ|q[t+1] - q[t]|  (t 在窗口内)
   ```
   表示关节在窗口内的总移动距离。

2. **净位移（Net Displacement, D）**: 窗口起始和结束位置的绝对差值
   ```
   D = |q[end] - q[start]|
   ```
   表示关节的实际位移。

3. **位移比例（TV/D Ratio）**: 总变分与净位移的比值
   ```
   Ratio = TV / (D + eps)
   ```
   高比例表示来回振荡（总移动大但净位移小）。

4. **最大单步位移（Max Step）**: 窗口内相邻帧之间的最大位移
   ```
   max_step = max(|q[t+1] - q[t]|)  (t 在窗口内)
   ```
   用于区分抖动和正常快速运动。

#### 检测条件

一个窗口被标记为抖动，必须**同时满足**以下六个条件：

1. **TV > min_total_variation** (默认: 0.20 rad)
   - 必须有显著的总变分，确保是真正的振荡

2. **D < max_net_displacement** (默认: 0.02 rad)
   - 净位移必须很小，确保是来回运动而非单向运动

3. **TV / (D + eps) > min_displacement_ratio** (默认: 5.0)
   - 位移比例必须超过阈值，确保是振荡模式

4. **max_step < max_step_size** (默认: 0.04 rad)
   - 单步位移必须小，确保是高频率小幅度抖动（而非正常快速运动）

5. **avg_step < max_avg_step_size** (默认: 0.025 rad) ← **新增**
   - 平均步长必须小，用于区分抖动和大幅度动作
   - 抖动：平均步长很小
   - 大幅度动作：平均步长较大

6. **direction_changes >= min_direction_changes** (默认: 3) ← **新增**
   - 方向变化必须频繁，确保是振荡模式
   - 抖动：频繁的方向变化（振荡模式）
   - 大幅度动作：方向变化较少（通常是单向或简单来回）

#### 算法实现

```python
def compute_jitter_score_stable(q, W, min_tv, max_d, min_ratio, max_step):
    """
    计算抖动分数（滑动窗口方法）
    
    参数:
        q: 关节角度序列 (T,)
        W: 窗口大小 (默认: 10 帧)
        min_tv: 最小总变分阈值
        max_d: 最大净位移阈值
        min_ratio: 最小位移比例阈值
        max_step: 最大单步位移阈值
    
    返回:
        scores: (T,) 二进制分数 (1.0 表示抖动, 0.0 表示正常)
    """
    scores = np.zeros(len(q))
    
    for t in range(len(q) - W + 1):
        window = q[t:t+W]
        
        # 计算总变分 (TV)
        tv = np.sum(np.abs(np.diff(window)))
        
        # 计算净位移 (D)
        d = np.abs(window[-1] - window[0])
        
        # 计算最大单步位移
        max_step_actual = np.max(np.abs(np.diff(window)))
        
        # 四条件联合判断
        is_jitter = (
            tv > min_tv and           # 条件1: 总变分足够大
            d < max_d and             # 条件2: 净位移足够小
            tv / (d + eps) > min_ratio and  # 条件3: 比例足够高
            max_step_actual < max_step     # 条件4: 单步位移足够小
        )
        
        # 将结果标记到窗口中心帧
        scores[t + W // 2] = 1.0 if is_jitter else 0.0
    
    return scores
```

#### 全局判断标准

运动被标记为异常（抖动）如果：

- **抖动帧比例 >= min_jitter_ratio** (默认: 5%)
  - 即：`抖动帧数 / 总帧数 >= 0.05`

#### 为什么需要六个条件？

- **条件1 (TV > min_tv)**: 确保有足够的振荡幅度
- **条件2 (D < max_d)**: 确保是来回运动而非单向运动
- **条件3 (TV/D > min_ratio)**: 确保振荡模式明显
- **条件4 (max_step < max_step_size)**: 确保单步位移小（高频率小幅度）
- **条件5 (avg_step < max_avg_step_size)**: **新增**，用于区分抖动和大幅度动作
  - **抖动**: 平均步长很小
  - **大幅度动作**: 平均步长较大
- **条件6 (direction_changes >= min_direction_changes)**: **新增**，确保是振荡模式
  - **抖动**: 频繁的方向变化（振荡模式）
  - **大幅度动作**: 方向变化较少（通常是单向或简单来回）

#### 示例

**抖动示例**（会被检测到）:
```
帧:  0    1    2    3    4    5    6    7    8    9
角度: 0.0  0.02 0.0  0.02 0.0  0.02 0.0  0.02 0.0  0.02
TV = 0.10, D = 0.02, Ratio = 5.0, max_step = 0.02
✓ 满足所有条件 → 抖动
```

**正常快速运动示例**（不会被误检）:
```
帧:  0    1    2    3    4    5    6    7    8    9
角度: 0.0  0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9
TV = 0.9, D = 0.9, Ratio = 1.0, max_step = 0.1
✗ max_step 太大 → 正常运动
```

#### 检测范围

抖动检测应用于：
- **所有关节角度** (dof_pos)
- **根位置** (root_pos) - 可选

对于每个关节，独立计算抖动分数，然后汇总统计。



## 输出文件

### 1. `auto_check_reports.json`

包含所有异常运动的详细检测报告（数组格式）：

```json
[
  {
    "motion_file": "GRAB/s1/airplane_fly_1_stageii.pkl",
    "is_abnormal": true,
    "floating": {
      "is_floating": true,
      "total_frames": 300,
      "fps": 30,
      "violation_reason": "Continuous off-ground time (2.5s) exceeds threshold (1.0s)"
    },
    "joint_limits": {...},
    "joint_discontinuity": {...},
    "self_collision": {...}
  }
]
```

### 2. `auto_check_annotations.json`

包含所有运动文件的标注信息（字典格式，与 `annotations.json` 相同）：

```json
{
  "GRAB/s1/airplane_fly_1_stageii.pkl": {
    "label": "abnormal",
    "reason": "1. 与地形/场景存在交互, 3. 动作衔接异常或抖动",
    "timestamp": "2025-01-XX...",
    "motion_path": "data/out/GRAB/s1/airplane_fly_1_stageii.pkl"
  },
  "GRAB/s1/airplane_offhand_1_stageii.pkl": {
    "label": "normal",
    "reason": null,
    "timestamp": "2025-01-XX...",
    "motion_path": "data/out/GRAB/s1/airplane_offhand_1_stageii.pkl"
  }
}
```

## 异常原因映射

检测到的问题会自动映射到对应的异常原因：

| 检测问题 | 异常原因 |
|---------|---------|
| `floating` (浮动) | `"1. 与地形/场景存在交互"` |
| `self_collision` (自碰撞) | `"2. 穿模或自碰撞"` |
| `joint_discontinuity` (关节不连续) | `"3. 动作衔接异常或抖动"` |
| `jitter` (抖动) | `"3. 动作衔接异常或抖动"` |
| `joint_limits` (关节限制) | `"4. 重定向失败（奇怪姿势）"` |
| 多个问题 | 用逗号连接所有原因 |

## 使用示例

### 示例 1: 检测整个数据集

```bash
# 检测 data/out 目录下的所有运动文件
python scripts/auto_check.py \
    --motion_folder data/out \
    --robot_type unitree_g1
```

输出：
```
Found 1234 motion files, scanning...
================================================================================
Check Results
================================================================================
Total motions scanned: 1234
Abnormal motions detected: 156
Percentage: 12.64%

Floating issues: 45
Joint limit violations: 0
Joint discontinuities: 78
Self-collisions: 33
Jitter issues: 12

Abnormal reports (156 motions) saved to: auto_check_reports.json
Annotations (1234 motions) saved to: auto_check_annotations.json
```

### 示例 2: 调整检测阈值

```bash
# 使用更严格的浮动检测阈值
python scripts/auto_check.py \
    --motion_folder data/out \
    --robot_type unitree_g1 \
    --height_threshold 0.08 \
    --min_seconds 0.5 \
    --min_ratio 0.15
```

### 示例 3: 检测单个文件

```bash
# 检测单个文件（只输出异常结果）
python scripts/auto_check.py \
    --motion_file data/out/GRAB/s1/airplane_fly_1_stageii.pkl \
    --robot_type unitree_g1
```

## 技术细节

### 检测流程

1. **加载运动数据**: 从 `.pkl` 文件加载机器人运动数据
2. **MuJoCo 渲染**: 通过 MuJoCo 渲染每一帧，获取实际关节角度和身体位置
3. **浮动检测**: 使用 MuJoCo 渲染的世界坐标检测脚部高度
4. **关节检测**: 分析关节角度的连续性和限制违反
5. **碰撞检测**: 使用 MuJoCo 的碰撞检测系统检测自碰撞
6. **抖动检测**: 基于总变分和净位移分析关节角度或根位置的来回抖动
7. **生成报告**: 汇总所有检测结果并生成报告

### 脚部链接名称

默认检测以下脚部链接（可通过 `--foot_names` 自定义）：

- `left_ankle_roll_link`
- `right_ankle_roll_link`
- `left_toe_link`
- `right_toe_link`
- `left_foot_link`
- `right_foot_link`

### 自碰撞过滤

自碰撞检测会过滤掉：
- 与地面/环境的碰撞
- 穿透深度 < 3mm 的轻微碰撞

## 批量处理和断点续传

### 批量处理

`auto_check.py` 支持批量处理大量文件，避免内存不足：

- **默认批量大小**: 100 个文件/批
- **内存管理**: 每批处理完后会自动清理内存
- **可调整**: 通过 `--batch_size` 参数调整批量大小

```bash
# 使用较小的批量大小（如果内存不足）
python scripts/auto_check.py \
    --motion_folder data/out \
    --robot_type unitree_g1 \
    --batch_size 50
```

### 断点续传

如果程序中断，可以从中断点继续：

- **自动保存**: 每批处理完后自动保存进度到 `auto_check_progress.json`
- **自动恢复**: 重新运行时会自动加载进度，跳过已处理文件
- **自定义进度文件**: 通过 `--progress_file` 指定进度文件路径

```bash
# 使用自定义进度文件
python scripts/auto_check.py \
    --motion_folder data/out \
    --robot_type unitree_g1 \
    --progress_file my_progress.json

# 禁用进度保存（不推荐，除非确定不会中断）
python scripts/auto_check.py \
    --motion_folder data/out \
    --robot_type unitree_g1 \
    --no_save_progress
```

### 处理大量数据（如 10,000 条）

```bash
# 建议配置
python scripts/auto_check.py \
    --motion_folder data/out \
    --robot_type unitree_g1 \
    --batch_size 50 \          # 根据可用内存调整（建议 50-100）
    --progress_file auto_check_progress.json
```

**建议**：
- 批量大小: 50-100（根据可用内存调整）
- 确保有足够内存: 建议至少 16GB RAM
- 如果内存不足，减小 `batch_size`（如 20-50）
- 启用进度保存，避免中断后重新开始

## 注意事项

1. **内存使用**: 每个文件的检测需要一定内存，建议在内存充足的环境下运行
   - 使用批量处理可以降低内存峰值
   - 如果内存不足，减小 `--batch_size`
2. **处理时间**: 检测时间取决于文件数量和帧数，可能需要较长时间
   - 使用断点续传可以避免中断后重新开始
3. **文件格式**: 只支持 `.pkl` 格式的机器人运动数据
4. **机器人模型**: 确保指定的机器人类型在 `assets/` 目录中有对应的模型文件
5. **路径格式**: `auto_check_annotations.json` 中的 `motion_path` 是相对于项目根目录的路径
6. **检测改进**: 
   - 抖动检测已改进，避免误判大幅度动作为抖动
   - 关节不连续检测已改进，避免误判跑跳等大幅度动作为异常

## 故障排除

### 问题: 找不到机器人模型

```
ValueError: Unknown robot type: xxx
```

**解决方案**: 检查 `assets/` 目录中是否有对应的机器人模型，或使用 `--robot_type` 指定正确的机器人类型。

### 问题: 无法加载运动文件

```
[WARN] Failed to load xxx.pkl: ...
```

**解决方案**: 检查文件格式是否正确，确保是有效的机器人运动数据文件。

### 问题: MuJoCo 渲染失败

```
[WARN] Failed to render through MuJoCo for xxx.pkl: ...
```

**解决方案**: 脚本会自动回退到使用原始数据，但检测精度可能降低。检查 MuJoCo 安装和机器人模型文件。

## 相关文件

- `auto_check_reports.json`: 详细的异常检测报告
- `auto_check_annotations.json`: 自动生成的标注文件


