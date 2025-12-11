#!/usr/bin/env python3
"""
Auto check robot motions for abnormalities:
1. Floating detection: Both feet off ground for extended periods
2. Joint discontinuity: Sudden jumps in joint angles
3. Self-collision: Collisions between robot body parts
4. Jitter detection: Rapid oscillations in joint angles and root position

Judgment criteria:
- Floating: Both feet simultaneously off ground (height > threshold) for > 3s OR > 50% of frames
- Joint discontinuity: Sudden jumps in joint angles between consecutive frames
- Self-collision: Collisions between robot body parts exceeding thresholds
- Jitter: High joint angular acceleration or root position acceleration exceeding thresholds
"""

import argparse
import json
import os
import gc
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import torch
import mujoco as mj

# Try to import psutil for memory monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from general_motion_retargeting.data_loader import load_robot_motion
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting.params import ROBOT_XML_DICT, ROBOT_BASE_DICT

DEFAULT_FOOT_KEYS = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_toe_link",
    "right_toe_link",
    "left_foot_link",
    "right_foot_link",
]

OFF_GROUND_THRESHOLD = 0.12  # 12cm above ground (stricter threshold)
MIN_CONTINUOUS_SECONDS = 1.0
MIN_CONTINUOUS_RATIO = 0.25

SELF_COLLISION_MIN_SECONDS = 0.6
SELF_COLLISION_MIN_RATIO = 0.10

# contact_dict-style filters to keep only meaningful self-collisions
# `pairs`: optional whitelist of (body_a, body_b) tuples (sorted names)
# `min_penetration`: minimum penetration depth (in meters) to count a collision
SELF_COLLISION_CONTACT_DICT = {
    "default": {
        "pairs": [],
        "min_penetration": 0.003,
    },
}

# Joint discontinuity thresholds
MAX_JOINT_JUMP = 0.8 # Maximum allowed joint angle change between consecutive frames (radians, ~29 degrees)
MIN_JOINT_JUMP_RATIO = 0.02  # Minimum ratio of frames with jumps to flag as abnormal (2%)
SEVERE_JOINT_JUMP = 1.0  # Severe jump threshold in radians (~57 degrees): always flag even if ratio is low
MAX_JOINT_ACCELERATION = 20.0  # Maximum allowed joint acceleration (rad/s^2) to distinguish jumps from smooth fast motions (increased for running/jumping)
MIN_SMOOTH_VELOCITY_RATIO = 0.4  # Minimum ratio of frames with smooth velocity changes to consider as normal fast motion (not a jump)
MIN_CONSECUTIVE_LARGE_CHANGES = 3  # Minimum consecutive frames with large changes to consider as normal fast motion (running/jumping)
MAX_ACCELERATION_CHANGE = 15.0  # Maximum allowed change in acceleration (jerk, rad/s^3) - true jumps have sudden acceleration changes

# Jitter detection thresholds (for oscillating jitter)
# Multiple conditions must be satisfied simultaneously to avoid false positives from large-amplitude motions:
# 1. TV > min_tv: Must have significant total variation (really oscillating)
# 2. D < max_d: Net displacement must be small (not moving far)
# 3. TV / (D + eps) > min_ratio: Ratio must exceed threshold
# 4. max_step < max_step_threshold: Each step should be small (true jitter is high-frequency small-amplitude)
# 5. avg_step < avg_step_threshold: Average step size should be small (distinguish from large-amplitude motions)
# 6. direction_changes >= min_direction_changes: Must have frequent direction changes (oscillating pattern)
MIN_TOTAL_VARIATION = 0.20  # Minimum total variation (TV) to consider as jitter (rad) - increased to avoid false positives from normal fast motion
MAX_NET_DISPLACEMENT = 0.02  # Maximum net displacement (D) to consider as jitter (rad)
MIN_DISPLACEMENT_RATIO = 5.0  # Minimum ratio of TV/D to consider as jitter - increased to be more strict
MAX_STEP_SIZE = 0.04  # Maximum single-step amplitude for jitter (rad) - true jitter has small steps (high-frequency small-amplitude)
MAX_AVG_STEP_SIZE = 0.025  # Maximum average step size for jitter (rad) - distinguish from large-amplitude motions
MIN_DIRECTION_CHANGES = 3  # Minimum number of direction changes in window (oscillating pattern)
MIN_JITTER_RATIO = 0.05  # Minimum ratio of frames with jitter to flag as abnormal (5%, lowered to detect more jitter cases)
OSCILLATION_WINDOW_SIZE = 10  # Window size for detecting oscillations (frames)


@dataclass
class FloatingInfo:
    is_floating: bool
    total_frames: int
    fps: float
    total_duration: float
    max_continuous_off_ground_frames: int
    max_continuous_off_ground_seconds: float
    max_continuous_off_ground_ratio: float
    total_off_ground_frames: int
    total_off_ground_ratio: float
    ground_level: float
    foot_heights_min: float
    foot_heights_max: float
    foot_heights_mean: float
    violation_reason: Optional[str] = None


@dataclass
class JointDiscontinuityInfo:
    has_discontinuity: bool
    total_frames: int
    num_discontinuity_frames: int
    discontinuity_ratio: float
    max_jump_rad: float
    max_jump_deg: float
    max_jump_frame: int
    max_jump_joint: int
    jumped_joints: List[int] = field(default_factory=list)
    discontinuity_details: Optional[str] = None


@dataclass
class CheckReport:
    motion_file: str
    is_abnormal: bool  # True if floating, joint discontinuity, self-collision, or jitter
    floating: FloatingInfo
    joint_discontinuity: JointDiscontinuityInfo
    self_collision: "SelfCollisionInfo"
    jitter: "JitterInfo"


@dataclass
class SelfCollisionInfo:
    has_self_collision: bool
    total_frames: int
    collision_frames: int
    collision_ratio: float
    max_continuous_collision_frames: int
    max_continuous_collision_seconds: float
    max_continuous_collision_ratio: float
    total_collision_events: int
    top_collision_pairs: List[str]
    violation_reason: Optional[str] = None


@dataclass
class JitterInfo:
    has_jitter: bool
    total_frames: int
    fps: float
    jitter_frames: int
    jitter_ratio: float
    max_joint_acceleration_rad: float
    max_joint_acceleration_deg: float
    max_joint_acceleration_frame: int
    max_joint_acceleration_joint: int
    max_root_vel_change: float
    max_root_acceleration: float
    max_root_acceleration_frame: int
    jittery_joints: List[int] = field(default_factory=list)
    violation_reason: Optional[str] = None


def get_mujoco_rendered_data(
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    dof_pos: np.ndarray,
    robot_type: str,
) -> tuple:
    """Render motion data through MuJoCo and get actual joint positions, body positions, and collisions.
    
    Returns:
        actual_dof_pos: (T, N) actual joint angles from MuJoCo (may be clipped)
        actual_body_pos: (T, M, 3) actual world positions of all bodies from MuJoCo
        body_names: List of body names
        collisions_by_frame: List[List[Tuple[str, str, float]]] collisions per frame
    """
    if robot_type not in ROBOT_XML_DICT:
        raise ValueError(f"Unknown robot type: {robot_type}")
    
    xml_file = ROBOT_XML_DICT[robot_type]
    model = mj.MjModel.from_xml_path(str(xml_file))
    data = mj.MjData(model)
    
    total_frames = len(root_pos)
    num_dof = len(dof_pos[0]) if len(dof_pos.shape) > 1 else 0
    num_bodies = model.nbody
    num_geoms = model.ngeom
    
    actual_dof_pos = np.zeros((total_frames, num_dof))
    actual_body_pos = np.zeros((total_frames, num_bodies, 3))
    body_names = []
    geom_names = []
    collisions_by_frame: List[List[Tuple[str, str, float]]] = [[] for _ in range(total_frames)]
    
    filter_cfg = SELF_COLLISION_CONTACT_DICT.get(
        robot_type, SELF_COLLISION_CONTACT_DICT.get("default", {})
    )
    pair_whitelist = {
        tuple(sorted(pair)) for pair in filter_cfg.get("pairs", []) if len(pair) == 2
    }
    min_penetration = filter_cfg.get("min_penetration", 0.0)
    
    # Get body and geom names
    for i in range(num_bodies):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
        body_names.append(name if name is not None else f"body_{i}")
    for i in range(num_geoms):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i)
        geom_names.append(name if name is not None else f"geom_{i}")
    
    # Render each frame through MuJoCo
    for frame_idx in range(total_frames):
        # Set joint positions
        data.qpos[:3] = root_pos[frame_idx]
        data.qpos[3:7] = root_rot[frame_idx]  # quat scalar first for MuJoCo
        data.qpos[7:] = dof_pos[frame_idx]
        
        # Forward kinematics + collision detection
        mj.mj_forward(model, data)
        
        # Get actual joint angles (after potential clipping)
        actual_dof_pos[frame_idx] = data.qpos[7:].copy()
        
        # Get actual body positions in world frame
        actual_body_pos[frame_idx] = data.xpos.copy()
        
        # Collect self-collisions (exclude world/environment geoms)
        if data.ncon > 0:
            frame_collisions = collisions_by_frame[frame_idx]
            for c_idx in range(data.ncon):
                contact = data.contact[c_idx]
                geom1 = contact.geom1
                geom2 = contact.geom2
                if geom1 < 0 or geom2 < 0:
                    continue
                body1 = model.geom_bodyid[geom1]
                body2 = model.geom_bodyid[geom2]
                # Skip contacts involving world/ground (body id 0)
                if body1 <= 0 or body2 <= 0:
                    continue
                body_pair = tuple(sorted((body_names[body1], body_names[body2])))
                if pair_whitelist and body_pair not in pair_whitelist:
                    continue
                penetration = max(0.0, -float(contact.dist))
                if penetration < min_penetration:
                    continue
                frame_collisions.append(
                    (body_pair[0], body_pair[1], penetration)
                )
    
    return actual_dof_pos, actual_body_pos, body_names, collisions_by_frame


def calculate_foot_world_heights(
    root_pos: np.ndarray,
    body_pos: np.ndarray,
    link_names: List[str],
    foot_names: List[str],
    is_world_positions: bool = False,
) -> dict:
    """Calculate world heights for each foot link.
    
    Args:
        root_pos: (T, 3) root position (used if body_pos is local)
        body_pos: (T, N, 3) body positions (either local or world frame)
        link_names: List of link/body names
        foot_names: List of foot link names to check
        is_world_positions: If True, body_pos is in world frame (from MuJoCo data.xpos)
    """
    foot_heights = {}
    FOOT_BOTTOM_OFFSET = -0.03
    
    for foot_name in foot_names:
        if foot_name not in link_names:
            continue
        foot_idx = link_names.index(foot_name)
        
        if is_world_positions:
            # body_pos is already in world frame (from MuJoCo data.xpos)
            world_heights = body_pos[:, foot_idx, 2]
        else:
            # body_pos is in local frame, need to add root_pos
            link_center_height = root_pos[:, 2] + body_pos[:, foot_idx, 2]
            world_heights = link_center_height
        
        # Apply foot bottom offset for ankle_roll_link
        if "ankle_roll_link" in foot_name:
            world_heights = world_heights + FOOT_BOTTOM_OFFSET
        
        foot_heights[foot_name] = world_heights
    return foot_heights


def find_max_continuous_off_ground(mask: np.ndarray) -> tuple:
    """Find maximum continuous sequence of True values in mask."""
    if mask.size == 0:
        return (0, -1, -1)
    
    max_length = 0
    max_start = -1
    max_end = -1
    
    current_length = 0
    current_start = -1
    
    for i, value in enumerate(mask):
        if value:
            if current_length == 0:
                current_start = i
            current_length += 1
            if current_length > max_length:
                max_length = current_length
                max_start = current_start
                max_end = i
        else:
            current_length = 0
            current_start = -1
    
    return (max_length, max_start, max_end)


def detect_floating(
    root_pos: np.ndarray,
    body_pos: np.ndarray,
    link_body_list: List[str],
    fps: float,
    foot_names: List[str] = None,
    height_threshold: float = OFF_GROUND_THRESHOLD,
    min_seconds: float = MIN_CONTINUOUS_SECONDS,
    min_ratio: float = MIN_CONTINUOUS_RATIO,
    is_world_positions: bool = False,
) -> FloatingInfo:
    """Detect if robot is floating.
    
    Args:
        root_pos: (T, 3) root position
        body_pos: (T, N, 3) body positions (local or world frame)
        link_body_list: List of body names
        fps: Frames per second
        foot_names: List of foot link names to check
        height_threshold: Height threshold for off-ground detection
        min_seconds: Minimum continuous seconds off-ground
        min_ratio: Minimum ratio of frames off-ground
        is_world_positions: If True, body_pos is in world frame (from MuJoCo)
    """
    if foot_names is None:
        foot_names = DEFAULT_FOOT_KEYS
    
    foot_heights = calculate_foot_world_heights(
        root_pos, body_pos, link_body_list, foot_names, is_world_positions=is_world_positions
    )
    
    if not foot_heights:
        # Return default info if no feet found
        return FloatingInfo(
            is_floating=False,
            total_frames=len(root_pos),
            fps=fps,
            total_duration=len(root_pos) / fps if fps > 0 else 0.0,
            max_continuous_off_ground_frames=0,
            max_continuous_off_ground_seconds=0.0,
            max_continuous_off_ground_ratio=0.0,
            total_off_ground_frames=0,
            total_off_ground_ratio=0.0,
            ground_level=0.0,
            foot_heights_min=0.0,
            foot_heights_max=0.0,
            foot_heights_mean=0.0,
            violation_reason="No valid foot links found",
        )
    
    # Find left and right foot
    left_foot = None
    right_foot = None
    
    for name in ["left_ankle_roll_link", "left_toe_link", "left_foot_link"]:
        if name in foot_heights:
            left_foot = foot_heights[name]
            break
    
    for name in ["right_ankle_roll_link", "right_toe_link", "right_foot_link"]:
        if name in foot_heights:
            right_foot = foot_heights[name]
            break
    
    if left_foot is None or right_foot is None:
        available_feet = list(foot_heights.values())
        if len(available_feet) < 2:
            return FloatingInfo(
                is_floating=False,
                total_frames=len(root_pos),
                fps=fps,
                total_duration=len(root_pos) / fps if fps > 0 else 0.0,
                max_continuous_off_ground_frames=0,
                max_continuous_off_ground_seconds=0.0,
                max_continuous_off_ground_ratio=0.0,
                total_off_ground_frames=0,
                total_off_ground_ratio=0.0,
                ground_level=0.0,
                foot_heights_min=0.0,
                foot_heights_max=0.0,
                foot_heights_mean=0.0,
                violation_reason=f"Need at least 2 feet, found {len(available_feet)}",
            )
        left_foot = available_feet[0]
        right_foot = available_feet[1]
    
    ground_level = 0.0
    
    left_off_ground = left_foot > (ground_level + height_threshold)
    right_off_ground = right_foot > (ground_level + height_threshold)
    both_feet_off_ground = left_off_ground & right_off_ground
    
    total_frames = len(both_feet_off_ground)
    total_duration = total_frames / fps if fps > 0 else 0.0
    
    max_continuous_frames, start_idx, end_idx = find_max_continuous_off_ground(
        both_feet_off_ground
    )
    max_continuous_seconds = max_continuous_frames / fps if fps > 0 else 0.0
    max_continuous_ratio = max_continuous_frames / total_frames if total_frames > 0 else 0.0
    
    total_off_ground_frames = int(np.sum(both_feet_off_ground))
    total_off_ground_ratio = total_off_ground_frames / total_frames if total_frames > 0 else 0.0
    
    all_foot_heights = np.concatenate([left_foot, right_foot])
    foot_heights_min = float(np.min(all_foot_heights))
    foot_heights_max = float(np.max(all_foot_heights))
    foot_heights_mean = float(np.mean(all_foot_heights))
    
    is_floating = False
    violation_reason = None
    
    if max_continuous_seconds >= min_seconds:
        is_floating = True
        violation_reason = f"Continuous {max_continuous_seconds:.2f}s >= {min_seconds}s"
    elif max_continuous_ratio >= min_ratio:
        is_floating = True
        violation_reason = f"Continuous ratio {max_continuous_ratio:.2%} >= {min_ratio:.2%}"
    
    return FloatingInfo(
        is_floating=is_floating,
        total_frames=total_frames,
        fps=fps,
        total_duration=total_duration,
        max_continuous_off_ground_frames=max_continuous_frames,
        max_continuous_off_ground_seconds=max_continuous_seconds,
        max_continuous_off_ground_ratio=max_continuous_ratio,
        total_off_ground_frames=total_off_ground_frames,
        total_off_ground_ratio=total_off_ground_ratio,
        ground_level=float(ground_level),
        foot_heights_min=foot_heights_min,
        foot_heights_max=foot_heights_max,
        foot_heights_mean=foot_heights_mean,
        violation_reason=violation_reason,
    )


def detect_joint_discontinuity(
    dof_pos: np.ndarray,
    fps: float,
    max_jump: float = MAX_JOINT_JUMP,
    min_jump_ratio: float = MIN_JOINT_JUMP_RATIO,
    severe_jump: float = SEVERE_JOINT_JUMP,
    max_acceleration: float = MAX_JOINT_ACCELERATION,
    min_smooth_velocity_ratio: float = MIN_SMOOTH_VELOCITY_RATIO,
    min_consecutive_large_changes: int = MIN_CONSECUTIVE_LARGE_CHANGES,
    max_acceleration_change: float = MAX_ACCELERATION_CHANGE,
) -> JointDiscontinuityInfo:
    """Detect sudden jumps/discontinuities in joint angles between consecutive frames.
    
    This function distinguishes true jumps from smooth fast motions (like running/jumping) by checking:
    1. Frame-to-frame angle changes (large changes may indicate jumps)
    2. Acceleration (sudden acceleration changes indicate jumps, smooth acceleration indicates normal fast motion)
    3. Velocity continuity (smooth velocity changes indicate normal fast motion, not jumps)
    4. Consecutive large changes (running/jumping have continuous sequences, true jumps are isolated)
    5. Acceleration change rate (jerk) - true jumps have sudden acceleration changes
    
    Args:
        dof_pos: (T, N) joint angles
        fps: Frames per second
        max_jump: Maximum allowed joint angle change between consecutive frames (rad)
        min_jump_ratio: Minimum ratio of frames with jumps to flag as abnormal
        severe_jump: Severe jump threshold - always flag even if ratio is low (rad)
        max_acceleration: Maximum allowed joint acceleration (rad/s^2) to distinguish jumps from smooth fast motions
        min_smooth_velocity_ratio: Minimum ratio of frames with smooth velocity changes to consider as normal fast motion
        min_consecutive_large_changes: Minimum consecutive frames with large changes to consider as normal fast motion
        max_acceleration_change: Maximum allowed change in acceleration (jerk, rad/s^3) - true jumps have sudden acceleration changes
    """
    if len(dof_pos.shape) != 2:
        raise ValueError(f"Expected dof_pos shape (T, N), got {dof_pos.shape}")
    
    total_frames, num_joints = dof_pos.shape
    
    if total_frames < 4:  # Need at least 4 frames for acceleration and jerk calculation
        return JointDiscontinuityInfo(
            has_discontinuity=False,
            total_frames=total_frames,
            num_discontinuity_frames=0,
            discontinuity_ratio=0.0,
            max_jump_rad=0.0,
            max_jump_deg=0.0,
            max_jump_frame=-1,
            max_jump_joint=-1,
            jumped_joints=[],
            discontinuity_details="Insufficient frames for discontinuity detection",
        )
    
    dt = 1.0 / fps if fps > 0 else 1.0 / 30.0
    
    # Calculate frame-to-frame changes (velocities)
    joint_changes = np.abs(np.diff(dof_pos, axis=0))  # (T-1, N)
    joint_velocities = joint_changes / dt  # (T-1, N) in rad/s
    
    # Calculate accelerations (second derivative)
    joint_accelerations = np.abs(np.diff(joint_velocities, axis=0)) / dt  # (T-2, N) in rad/s^2
    
    # Calculate jerk (third derivative, acceleration change rate)
    joint_jerks = np.abs(np.diff(joint_accelerations, axis=0)) / dt  # (T-3, N) in rad/s^3
    
    # Find frames with large angle changes
    large_changes = joint_changes > max_jump  # (T-1, N)
    
    # For frames with large changes, check if they are smooth (normal fast motion) or jumps
    # A jump is characterized by:
    # 1. Large angle change AND
    # 2. Large acceleration change (jerk) OR
    # 3. Not part of a smooth velocity sequence OR
    # 4. Isolated (not part of a consecutive sequence)
    
    # First, identify consecutive sequences of large changes (likely running/jumping)
    # For each joint, find sequences of consecutive large changes
    true_jumps = np.zeros_like(large_changes, dtype=bool)  # (T-1, N)
    
    for j in range(num_joints):
        # Find consecutive sequences of large changes
        consecutive_sequences = []
        in_sequence = False
        sequence_start = -1
        
        for i in range(len(large_changes)):
            if large_changes[i, j]:
                if not in_sequence:
                    in_sequence = True
                    sequence_start = i
            else:
                if in_sequence:
                    # End of sequence
                    sequence_length = i - sequence_start
                    if sequence_length >= min_consecutive_large_changes:
                        consecutive_sequences.append((sequence_start, i))
                    in_sequence = False
        
        # Handle sequence that extends to the end
        if in_sequence:
            sequence_length = len(large_changes) - sequence_start
            if sequence_length >= min_consecutive_large_changes:
                consecutive_sequences.append((sequence_start, len(large_changes)))
        
        # Mark all frames in consecutive sequences as potential normal fast motion
        for seq_start, seq_end in consecutive_sequences:
            # Check if this sequence has smooth acceleration (not sudden jumps)
            is_smooth_sequence = True
            for i in range(seq_start, min(seq_end, len(large_changes))):
                if i > 0 and i <= len(joint_accelerations):
                    accel_idx = i - 1
                    if accel_idx >= 0 and accel_idx < len(joint_accelerations):
                        accel = joint_accelerations[accel_idx, j]
                        # Check jerk (acceleration change)
                        if i > 1 and i <= len(joint_jerks) + 1:
                            jerk_idx = i - 2
                            if jerk_idx >= 0 and jerk_idx < len(joint_jerks):
                                jerk = joint_jerks[jerk_idx, j]
                                if jerk > max_acceleration_change:
                                    # Sudden acceleration change - might be a jump
                                    is_smooth_sequence = False
                                    break
            
            # If sequence is smooth, don't mark as jumps
            if is_smooth_sequence:
                for i in range(seq_start, min(seq_end, len(large_changes))):
                    true_jumps[i, j] = False
            else:
                # Mark as potential jumps (will be further checked)
                for i in range(seq_start, min(seq_end, len(large_changes))):
                    true_jumps[i, j] = True
    
    # Now check isolated large changes and sequences that weren't smooth
    for i in range(len(large_changes)):
        for j in range(num_joints):
            if large_changes[i, j] and not true_jumps[i, j]:
                # This is an isolated large change or part of a non-smooth sequence
                # Check if it's a true jump
                is_jump = True
                
                # Check if it's part of a consecutive sequence (already handled above)
                if i > 0 and i < len(large_changes) - 1:
                    if large_changes[i-1, j] or large_changes[i+1, j]:
                        # Part of a sequence, but sequence wasn't long enough or wasn't smooth
                        # Check acceleration and jerk more carefully
                        if i > 0 and i <= len(joint_accelerations):
                            accel_idx = i - 1
                            if accel_idx >= 0 and accel_idx < len(joint_accelerations):
                                accel = joint_accelerations[accel_idx, j]
                                # Check jerk
                                if i > 1 and i <= len(joint_jerks) + 1:
                                    jerk_idx = i - 2
                                    if jerk_idx >= 0 and jerk_idx < len(joint_jerks):
                                        jerk = joint_jerks[jerk_idx, j]
                                        # If acceleration is reasonable and jerk is low, might be normal fast motion
                                        if accel < max_acceleration and jerk < max_acceleration_change:
                                            # Check velocity continuity in a longer window
                                            window_start = max(0, i - 3)
                                            window_end = min(len(joint_velocities), i + 4)
                                            if window_end > window_start + 1:
                                                window_velocities = joint_velocities[window_start:window_end, j]
                                                if len(window_velocities) > 1:
                                                    velocity_changes = np.abs(np.diff(window_velocities))
                                                    smooth_ratio = np.sum(velocity_changes < max_acceleration * dt) / len(velocity_changes) if len(velocity_changes) > 0 else 0.0
                                                    if smooth_ratio >= min_smooth_velocity_ratio:
                                                        is_jump = False
                
                # Check isolated large changes (not part of any sequence)
                if is_jump and (i == 0 or not large_changes[i-1, j]) and (i == len(large_changes) - 1 or not large_changes[i+1, j]):
                    # Isolated large change - check if it's a sudden jump
                    if i > 0 and i <= len(joint_accelerations):
                        accel_idx = i - 1
                        if accel_idx >= 0 and accel_idx < len(joint_accelerations):
                            accel = joint_accelerations[accel_idx, j]
                            # Check jerk for sudden acceleration change
                            if i > 1 and i <= len(joint_jerks) + 1:
                                jerk_idx = i - 2
                                if jerk_idx >= 0 and jerk_idx < len(joint_jerks):
                                    jerk = joint_jerks[jerk_idx, j]
                                    # If acceleration and jerk are both reasonable, might be normal
                                    if accel < max_acceleration and jerk < max_acceleration_change:
                                        is_jump = False
                
                true_jumps[i, j] = is_jump
    
    # Per-frame discontinuity mask (any joint has true jump)
    frame_discontinuities = np.any(true_jumps, axis=1)  # (T-1,)
    num_discontinuity_frames = int(np.sum(frame_discontinuities))
    discontinuity_ratio = num_discontinuity_frames / (total_frames - 1) if total_frames > 1 else 0.0
    
    # Find joints with jumps
    joint_has_jumps = np.any(true_jumps, axis=0)  # (N,)
    jumped_joints = [int(i) for i in np.where(joint_has_jumps)[0]]
    
    # Find maximum jump (always, not just when exceeding threshold)
    max_jump_idx_1d = np.argmax(joint_changes)
    max_jump_frame_idx = max_jump_idx_1d // num_joints
    max_jump_joint_idx = max_jump_idx_1d % num_joints
    max_jump_value = joint_changes[max_jump_frame_idx, max_jump_joint_idx]
    
    # Frame index where jump occurs (jump is between frame and frame+1)
    max_jump_frame = int(max_jump_frame_idx)
    max_jump_joint = int(max_jump_joint_idx)
    max_jump_rad = float(max_jump_value)
    max_jump_deg = float(max_jump_value * 180.0 / np.pi)
    
    # Flag as discontinuity if:
    # 1. Ratio exceeds minimum threshold, OR
    # 2. Maximum jump exceeds severe threshold (always flag severe jumps)
    # Note: We only count true jumps (after filtering smooth fast motions)
    has_discontinuity = (discontinuity_ratio >= min_jump_ratio) or (max_jump_rad >= severe_jump)
    
    discontinuity_details = None
    if has_discontinuity:
        if max_jump_rad >= severe_jump:
            discontinuity_details = (
                f"SEVERE jump detected: {max_jump_deg:.2f} deg at frame {max_jump_frame+1} (joint {max_jump_joint}) "
                f"(exceeds severe threshold {severe_jump*180/np.pi:.1f} deg)"
            )
        else:
            discontinuity_details = (
                f"{num_discontinuity_frames} frame transitions ({discontinuity_ratio:.2%}) have true jumps > {max_jump:.3f} rad. "
                f"Max jump: {max_jump_deg:.2f} deg at frame {max_jump_frame+1} (joint {max_jump_joint})"
            )
    elif max_jump_rad > 0:
        discontinuity_details = (
            f"Found jump of {max_jump_deg:.2f} deg at frame {max_jump_frame+1} (joint {max_jump_joint}) "
            f"(below threshold ratio {min_jump_ratio:.2%} or filtered as smooth fast motion)"
        )
    
    return JointDiscontinuityInfo(
        has_discontinuity=has_discontinuity,
        total_frames=total_frames,
        num_discontinuity_frames=num_discontinuity_frames,
        discontinuity_ratio=discontinuity_ratio,
        max_jump_rad=max_jump_rad,
        max_jump_deg=max_jump_deg,
        max_jump_frame=max_jump_frame + 1 if max_jump_frame >= 0 else -1,  # Report as frame where jump occurs
        max_jump_joint=max_jump_joint,
        jumped_joints=jumped_joints,
        discontinuity_details=discontinuity_details,
    )


def compute_jitter_score_stable(q, W, min_tv=0.20, max_d=0.02, min_ratio=5.0, max_step=0.04, max_avg_step=0.025, min_direction_changes=3, eps=1e-6):
    """Compute jitter score for a single joint angle sequence (robust version).
    
    Jitter is detected using multiple conditions simultaneously to avoid false positives:
    1. TV > min_tv: Must have significant total variation (really oscillating)
    2. D < max_d: Net displacement must be small (not moving far)
    3. TV / (D + eps) > min_ratio: Ratio must exceed threshold
    4. max_step < max_step: Each step must be small (high-frequency small-amplitude)
    5. avg_step < max_avg_step: Average step size must be small (distinguish from large-amplitude motions)
    6. direction_changes >= min_direction_changes: Must have frequent direction changes (oscillating pattern)
    
    This avoids false positives from:
    - Slow smooth motion (TV small, D small, but ratio high due to D≈0)
    - Normal movement (TV large, D large, ratio ≈ 1)
    - Static poses (TV very small, D=0, ratio explodes but TV too small)
    - Fast normal motion (TV large, but max_step also large - not jitter)
    - Large-amplitude motions (TV large, D small, but avg_step large or few direction changes - not jitter)
    
    Args:
        q: (T,) Joint angle sequence
        W: Window size
        min_tv: Minimum total variation to consider as jitter (rad)
        max_d: Maximum net displacement to consider as jitter (rad)
        min_ratio: Minimum ratio of TV/D to consider as jitter
        max_step: Maximum single-step amplitude for jitter (rad)
        max_avg_step: Maximum average step size for jitter (rad)
        min_direction_changes: Minimum number of direction changes in window
        eps: Small epsilon to avoid division by zero
        
    Returns:
        scores: (T,) Binary jitter scores (1.0 if jitter, 0.0 otherwise)
    """
    T = len(q)
    scores = np.zeros(T)
    
    for t in range(0, T - W):
        window = q[t:t+W+1]
        delta = np.diff(window)
        
        TV = np.sum(np.abs(delta))  # Total Variation
        D = np.abs(window[-1] - window[0])  # Net Displacement
        max_step_size = np.max(np.abs(delta))  # Maximum single-step amplitude
        avg_step_size = np.mean(np.abs(delta))  # Average step size
        
        # Count direction changes (sign changes in delta)
        # A direction change occurs when delta[i] and delta[i+1] have opposite signs
        signs = np.sign(delta)
        direction_changes = np.sum(np.abs(np.diff(signs)) > 0)  # Count sign changes
        
        # Multiple conditions must be satisfied simultaneously
        is_jitter = (
            (TV > min_tv) and                              # Must really oscillate
            (D < max_d) and                                # Must not move far
            (TV / (D + eps) > min_ratio) and              # Ratio must exceed threshold
            (max_step_size < max_step) and                # Each step must be small (high-frequency small-amplitude)
            (avg_step_size < max_avg_step) and            # Average step must be small (distinguish from large-amplitude motions)
            (direction_changes >= min_direction_changes)  # Must have frequent direction changes (oscillating pattern)
        )
        
        scores[t + W // 2] = 1.0 if is_jitter else 0.0
    
    return scores


def detect_jitter(
    dof_pos: np.ndarray,
    root_pos: np.ndarray,
    fps: float,
    min_tv: float = MIN_TOTAL_VARIATION,
    max_d: float = MAX_NET_DISPLACEMENT,
    min_ratio: float = MIN_DISPLACEMENT_RATIO,
    max_step: float = MAX_STEP_SIZE,
    max_avg_step: float = MAX_AVG_STEP_SIZE,
    min_direction_changes: int = MIN_DIRECTION_CHANGES,
    min_jitter_ratio: float = MIN_JITTER_RATIO,
    window_size: int = OSCILLATION_WINDOW_SIZE,
) -> JitterInfo:
    """Detect oscillating jitter in joint angles (back-and-forth movement).
    
    Jitter is detected using four conditions simultaneously:
    1. TV > min_tv: Must have significant total variation (really oscillating)
    2. D < max_d: Net displacement must be small (not moving far)
    3. TV / (D + eps) > min_ratio: Ratio must exceed threshold
    4. max_step < max_step: Each step must be small (high-frequency small-amplitude)
    
    This avoids false positives from slow smooth motion, normal movement, static poses, and fast normal motion.
    
    Mathematical definition:
    We define kinematic jitter as a motion pattern that exhibits large accumulated 
    variation within a short time window while maintaining a negligible net displacement.
    A segment is classified as jitter only if it simultaneously satisfies:
        TV > τ_tv,  D < τ_d,  TV/(D+ε) > τ_r,  max_step < τ_step,  avg_step < τ_avg_step,  direction_changes >= τ_dir
    
    Args:
        dof_pos: (T, N) joint angles
        root_pos: (T, 3) root positions (not used for oscillation detection)
        fps: Frames per second
        min_tv: Minimum total variation to consider as jitter (rad)
        max_d: Maximum net displacement to consider as jitter (rad)
        min_ratio: Minimum ratio of TV/D to consider as jitter
        max_step: Maximum single-step amplitude for jitter (rad)
        max_avg_step: Maximum average step size for jitter (rad)
        min_direction_changes: Minimum number of direction changes in window
        min_jitter_ratio: Minimum ratio of frames with jitter to flag as abnormal
        window_size: Window size for detecting oscillations (frames)
    """
    total_frames = len(dof_pos)
    
    if total_frames < window_size + 1:
        return JitterInfo(
            has_jitter=False,
            total_frames=total_frames,
            fps=fps,
            jitter_frames=0,
            jitter_ratio=0.0,
            max_joint_acceleration_rad=0.0,
            max_joint_acceleration_deg=0.0,
            max_joint_acceleration_frame=-1,
            max_joint_acceleration_joint=-1,
            max_root_vel_change=0.0,
            max_root_acceleration=0.0,
            max_root_acceleration_frame=-1,
            jittery_joints=[],
            violation_reason="Insufficient frames for oscillation detection",
        )
    
    # Compute jitter scores for each joint using robust three-condition check
    frame_has_jitter = np.zeros(total_frames, dtype=bool)
    joint_jitter_count = np.zeros(dof_pos.shape[1], dtype=int)
    max_jitter_ratio = 0.0
    max_ratio_joint = -1
    max_ratio_frame = -1
    
    for joint_idx in range(dof_pos.shape[1]):
        q = dof_pos[:, joint_idx]  # (T,)
        scores = compute_jitter_score_stable(
            q, window_size, 
            min_tv=min_tv, 
            max_d=max_d, 
            min_ratio=min_ratio,
            max_step=max_step,
            max_avg_step=max_avg_step,
            min_direction_changes=min_direction_changes
        )  # (T,) binary scores (1.0 if jitter, 0.0 otherwise)
        
        # Mark frames with jitter
        jitter_frames_for_joint = np.where(scores > 0.5)[0]
        if len(jitter_frames_for_joint) > 0:
            frame_has_jitter[jitter_frames_for_joint] = True
            joint_jitter_count[joint_idx] = len(jitter_frames_for_joint)
            
            # Calculate max ratio for this joint (for reporting)
            for t in jitter_frames_for_joint:
                window_start = max(0, t - window_size // 2)
                window_end = min(total_frames, window_start + window_size + 1)
                if window_end - window_start >= window_size + 1:
                    window = q[window_start:window_end]
                    delta = np.diff(window)
                    TV = np.sum(np.abs(delta))
                    D = np.abs(window[-1] - window[0])
                    ratio = TV / (D + 1e-6)
                    if ratio > max_jitter_ratio:
                        max_jitter_ratio = ratio
                        max_ratio_joint = joint_idx
                        max_ratio_frame = t
    
    # Find joints with jitter
    jittery_joints = [int(i) for i in np.where(joint_jitter_count > 0)[0]]
    
    # Calculate statistics
    jitter_frames = int(np.sum(frame_has_jitter))
    jitter_ratio = jitter_frames / total_frames if total_frames > 0 else 0.0
    
    # Calculate joint accelerations for display (second derivative)
    dt = 1.0 / fps if fps > 0 else 1.0 / 30.0
    joint_velocities = np.diff(dof_pos, axis=0) / dt  # (T-1, N)
    joint_accelerations = np.diff(joint_velocities, axis=0) / dt  # (T-2, N)
    joint_accel_magnitudes = np.abs(joint_accelerations)  # (T-2, N)
    
    # Find maximum acceleration for display
    if joint_accel_magnitudes.size > 0:
        max_accel_idx_1d = np.argmax(joint_accel_magnitudes)
        max_accel_frame_idx = max_accel_idx_1d // joint_accel_magnitudes.shape[1]
        max_accel_joint_idx = max_accel_idx_1d % joint_accel_magnitudes.shape[1]
        max_accel_value_for_display = joint_accel_magnitudes[max_accel_frame_idx, max_accel_joint_idx]
        max_accel_frame_for_display = int(max_accel_frame_idx) + 2
    else:
        max_accel_value_for_display = 0.0
        max_accel_joint_idx = max_ratio_joint if max_ratio_joint >= 0 else -1
        max_accel_frame_for_display = max_ratio_frame if max_ratio_frame >= 0 else -1
    
    # Root position metrics (not used for oscillation detection, but kept for compatibility)
    root_velocities = np.diff(root_pos, axis=0) / dt  # (T-1, 3)
    root_vel_magnitudes = np.linalg.norm(root_velocities, axis=1)  # (T-1,)
    root_vel_changes = np.diff(root_vel_magnitudes)  # (T-2,)
    root_vel_changes_abs = np.abs(root_vel_changes)  # (T-2,)
    root_accelerations = np.diff(root_velocities, axis=0) / dt  # (T-2, 3)
    root_accel_magnitudes = np.linalg.norm(root_accelerations, axis=1)  # (T-2,)
    
    max_root_vel_change_value = float(np.max(root_vel_changes_abs)) if root_vel_changes_abs.size > 0 else 0.0
    max_root_accel_value = float(np.max(root_accel_magnitudes)) if root_accel_magnitudes.size > 0 else 0.0
    max_root_accel_frame = int(np.argmax(root_accel_magnitudes)) + 2 if root_accel_magnitudes.size > 0 else -1
    
    # Flag as jitter if ratio exceeds threshold
    has_jitter = jitter_ratio >= min_jitter_ratio
    
    violation_reason = None
    if has_jitter:
        reasons = []
        if max_jitter_ratio > 0:
            reasons.append(f"max ratio {max_jitter_ratio:.2f} (TV/D, threshold {min_ratio:.2f})")
        if len(jittery_joints) > 0:
            reasons.append(f"{len(jittery_joints)} joint(s) with jitter")
        if jitter_ratio > 0:
            reasons.append(f"jitter ratio {jitter_ratio:.2%}")
        violation_reason = "; ".join(reasons) if reasons else f"jitter ratio {jitter_ratio:.2%}"
    
    return JitterInfo(
        has_jitter=has_jitter,
        total_frames=total_frames,
        fps=fps,
        jitter_frames=jitter_frames,
        jitter_ratio=jitter_ratio,
        max_joint_acceleration_rad=float(max_accel_value_for_display),
        max_joint_acceleration_deg=float(max_accel_value_for_display * 180.0 / np.pi),
        max_joint_acceleration_frame=max_accel_frame_for_display,
        max_joint_acceleration_joint=int(max_ratio_joint) if max_ratio_joint >= 0 else int(max_accel_joint_idx),
        max_root_vel_change=max_root_vel_change_value,
        max_root_acceleration=max_root_accel_value,
        max_root_acceleration_frame=max_root_accel_frame,
        jittery_joints=jittery_joints,
        violation_reason=violation_reason,
    )


def detect_self_collisions(
    collisions_by_frame: List[List[Tuple[str, str, float]]],
    fps: float,
    min_seconds: float = SELF_COLLISION_MIN_SECONDS,
    min_ratio: float = SELF_COLLISION_MIN_RATIO,
) -> SelfCollisionInfo:
    total_frames = len(collisions_by_frame)
    if total_frames == 0:
        return SelfCollisionInfo(
            has_self_collision=False,
            total_frames=0,
            collision_frames=0,
            collision_ratio=0.0,
            max_continuous_collision_frames=0,
            max_continuous_collision_seconds=0.0,
            max_continuous_collision_ratio=0.0,
            total_collision_events=0,
            top_collision_pairs=[],
            violation_reason=None,
        )
    
    collision_mask = np.array([len(frame) > 0 for frame in collisions_by_frame], dtype=bool)
    collision_frames = int(collision_mask.sum())
    collision_ratio = collision_frames / total_frames if total_frames > 0 else 0.0
    
    max_frames, start_idx, end_idx = find_max_continuous_off_ground(collision_mask)
    max_seconds = max_frames / fps if fps > 0 else 0.0
    max_ratio = max_frames / total_frames if total_frames > 0 else 0.0
    
    # Count collision pairs
    pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    total_events = 0
    for frame in collisions_by_frame:
        for geom1, geom2, _ in frame:
            key = tuple(sorted((geom1, geom2)))
            pair_counts[key] += 1
            total_events += 1
    
    top_pairs = [
        f"{a} <-> {b} ({count} frames)"
        for (a, b), count in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    ]
    
    has_self_collision = (collision_ratio >= min_ratio) or (max_seconds >= min_seconds)
    violation_reason = None
    if has_self_collision:
        reason_parts = []
        if collision_ratio >= min_ratio:
            reason_parts.append(f"collision ratio {collision_ratio:.2%} ≥ {min_ratio:.2%}")
        if max_seconds >= min_seconds:
            reason_parts.append(f"continuous {max_seconds:.2f}s ≥ {min_seconds:.2f}s")
        violation_reason = "; ".join(reason_parts)
    
    return SelfCollisionInfo(
        has_self_collision=has_self_collision,
        total_frames=total_frames,
        collision_frames=collision_frames,
        collision_ratio=collision_ratio,
        max_continuous_collision_frames=max_frames,
        max_continuous_collision_seconds=max_seconds,
        max_continuous_collision_ratio=max_ratio,
        total_collision_events=total_events,
        top_collision_pairs=top_pairs,
        violation_reason=violation_reason,
    )


def check_motion(
    motion_file: str,
    robot_type: str,
    foot_names: List[str] = None,
    height_threshold: float = OFF_GROUND_THRESHOLD,
    min_seconds: float = MIN_CONTINUOUS_SECONDS,
    min_ratio: float = MIN_CONTINUOUS_RATIO,
    max_joint_jump: float = MAX_JOINT_JUMP,
    min_jump_ratio: float = MIN_JOINT_JUMP_RATIO,
    severe_joint_jump: float = SEVERE_JOINT_JUMP,
) -> Optional[CheckReport]:
    """Check a single motion file for floating and joint limit violations."""
    if foot_names is None:
        foot_names = DEFAULT_FOOT_KEYS
    
    # Load motion data
    try:
        (
            motion_data,
            fps,
            root_pos,
            root_rot,
            dof_pos,
            local_body_pos,
            link_body_list,
        ) = load_robot_motion(motion_file)
    except Exception as e:
        print(f"[WARN] Failed to load {motion_file}: {e}")
        return None
    
    # Render through MuJoCo to get actual data (joint angles may be clipped, body positions from MuJoCo)
    try:
        (
            actual_dof_pos,
            actual_body_pos,
            mujoco_body_names,
            collisions_by_frame,
        ) = get_mujoco_rendered_data(root_pos, root_rot, dof_pos, robot_type)
        
        # For floating detection, use MuJoCo world body positions directly
        # calculate_foot_world_heights will detect if it's world positions and use them directly
        actual_body_pos_for_floating = actual_body_pos  # (T, M, 3) world positions from MuJoCo
    except Exception as e:
        print(f"[WARN] Failed to render through MuJoCo for {motion_file}: {e}")
        # Fallback to original data
        actual_dof_pos = dof_pos
        actual_body_pos_for_floating = local_body_pos  # Local positions
        mujoco_body_names = link_body_list
        use_world_positions = False
        collisions_by_frame = [[] for _ in range(len(dof_pos))]
    else:
        use_world_positions = True
    
    # Detect floating using MuJoCo rendered body positions (world frame from data.xpos)
    floating_info = detect_floating(
        root_pos,
        actual_body_pos_for_floating,  # MuJoCo world positions (data.xpos) or local positions (fallback)
        mujoco_body_names,
        fps,
        foot_names=foot_names,
        height_threshold=height_threshold,
        min_seconds=min_seconds,
        min_ratio=min_ratio,
        is_world_positions=use_world_positions,  # True if from MuJoCo, False if fallback
    )
    
    # Detect joint discontinuities using MuJoCo rendered joint angles
    # Detect joint discontinuities using MuJoCo rendered joint angles
    # Uses acceleration and velocity continuity checks to distinguish true jumps from smooth fast motions
    # Detect joint discontinuities using MuJoCo rendered joint angles
    # Uses acceleration, jerk, and velocity continuity checks to distinguish true jumps from smooth fast motions (running/jumping)
    joint_discontinuity_info = detect_joint_discontinuity(
        actual_dof_pos,
        fps,
        max_jump=max_joint_jump,
        min_jump_ratio=min_jump_ratio,
        severe_jump=severe_joint_jump,
        max_acceleration=MAX_JOINT_ACCELERATION,
        min_smooth_velocity_ratio=MIN_SMOOTH_VELOCITY_RATIO,
        min_consecutive_large_changes=MIN_CONSECUTIVE_LARGE_CHANGES,
        max_acceleration_change=MAX_ACCELERATION_CHANGE,
    )
    
    # Detect self-collisions using collision data
    self_collision_info = detect_self_collisions(
        collisions_by_frame,
        fps,
        min_seconds=SELF_COLLISION_MIN_SECONDS,
        min_ratio=SELF_COLLISION_MIN_RATIO,
    )
    
    # Detect jitter (oscillating back-and-forth movement) in joint angles
    # Uses multiple-condition check to avoid false positives from large-amplitude motions
    jitter_info = detect_jitter(
        actual_dof_pos,
        root_pos,
        fps,
        min_tv=MIN_TOTAL_VARIATION,
        max_d=MAX_NET_DISPLACEMENT,
        min_ratio=MIN_DISPLACEMENT_RATIO,
        max_step=MAX_STEP_SIZE,
        max_avg_step=MAX_AVG_STEP_SIZE,
        min_direction_changes=MIN_DIRECTION_CHANGES,
        min_jitter_ratio=MIN_JITTER_RATIO,
        window_size=OSCILLATION_WINDOW_SIZE,
    )
    
    # Determine if motion is abnormal
    is_abnormal = (
        floating_info.is_floating
        or joint_discontinuity_info.has_discontinuity
        or self_collision_info.has_self_collision
        or jitter_info.has_jitter
    )
    
    # Use relative path for motion_file
    return CheckReport(
        motion_file=motion_file,
        is_abnormal=is_abnormal,
        floating=floating_info,
        joint_discontinuity=joint_discontinuity_info,
        self_collision=self_collision_info,
        jitter=jitter_info,
    )


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 2)
    return 0.0


def check_memory_threshold(warning_threshold_mb: float = 8000, critical_threshold_mb: float = 12000) -> bool:
    """Check if memory usage exceeds thresholds. Returns True if critical."""
    if not HAS_PSUTIL:
        return False
    
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024 ** 2)
    
    if memory_mb > critical_threshold_mb:
        print(f"[CRITICAL] Memory usage: {memory_mb:.1f} MB exceeds critical threshold ({critical_threshold_mb:.1f} MB)")
        return True
    elif memory_mb > warning_threshold_mb:
        print(f"[WARNING] Memory usage: {memory_mb:.1f} MB exceeds warning threshold ({warning_threshold_mb:.1f} MB)")
    
    return False


def _save_progress(progress_path: Path, processed_files: set, reports: List[CheckReport]):
    """Save progress to JSON file."""
    try:
        progress_data = {
            "processed_files": list(processed_files),
            "reports": [asdict(r) for r in reports],
            "timestamp": datetime.now().isoformat(),
        }
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to save progress: {e}")


def scan_folder(
    motion_folder: str,
    robot_type: str,
    height_threshold: float = OFF_GROUND_THRESHOLD,
    min_seconds: float = MIN_CONTINUOUS_SECONDS,
    min_ratio: float = MIN_CONTINUOUS_RATIO,
    foot_names: List[str] = None,
    max_joint_jump: float = MAX_JOINT_JUMP,
    min_jump_ratio: float = MIN_JOINT_JUMP_RATIO,
    batch_size: int = 100,
    save_progress: bool = True,
    progress_file: str = None,
) -> List[CheckReport]:
    """Scan folder for abnormal motions with memory-efficient batch processing.
    
    Args:
        motion_folder: Folder containing motion files
        robot_type: Robot type
        batch_size: Number of files to process before clearing memory (default: 100)
        save_progress: Whether to save progress periodically (default: True)
        progress_file: Path to save progress JSON (default: auto_check_progress.json)
    """
    motion_folder = Path(motion_folder)
    if not motion_folder.exists():
        raise FileNotFoundError(f"Motion folder does not exist: {motion_folder}")
    
    reports = []
    motion_files = []
    
    # Collect all pkl files
    for root, _, files in os.walk(motion_folder):
        for file in files:
            if file.endswith(".pkl"):
                motion_files.append(Path(root) / file)
    
    total_files = len(motion_files)
    print(f"Found {total_files} motion files, scanning...")
    
    if HAS_PSUTIL:
        initial_memory = get_memory_usage_mb()
        print(f"Initial memory usage: {initial_memory:.1f} MB")
    
    # Load existing progress if available
    if progress_file is None:
        progress_file = "auto_check_progress.json"
    progress_path = Path(progress_file)
    processed_files = set()
    
    if progress_path.exists() and save_progress:
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                progress_data = json.load(f)
                processed_files = set(progress_data.get("processed_files", []))
                # Load existing reports
                existing_reports = progress_data.get("reports", [])
                reports = []
                for r in existing_reports:
                    try:
                        # Convert dict to CheckReport, handling missing fields
                        if isinstance(r, dict):
                            # Ensure all required fields are present
                            if "jitter" not in r:
                                # Create default JitterInfo for old progress files
                                r["jitter"] = {
                                    "has_jitter": False,
                                    "total_frames": r.get("floating", {}).get("total_frames", 0),
                                    "fps": r.get("floating", {}).get("fps", 30.0),
                                    "jitter_frames": 0,
                                    "jitter_ratio": 0.0,
                                    "max_joint_acceleration_rad": 0.0,
                                    "max_joint_acceleration_deg": 0.0,
                                    "max_joint_acceleration_frame": -1,
                                    "max_joint_acceleration_joint": -1,
                                    "max_root_vel_change": 0.0,
                                    "max_root_acceleration": 0.0,
                                    "max_root_acceleration_frame": -1,
                                    "jittery_joints": [],
                                    "violation_reason": None,
                                }
                            # Convert nested dicts to dataclass instances
                            if "floating" in r and isinstance(r["floating"], dict):
                                r["floating"] = FloatingInfo(**r["floating"])
                            if "joint_discontinuity" in r and isinstance(r["joint_discontinuity"], dict):
                                r["joint_discontinuity"] = JointDiscontinuityInfo(**r["joint_discontinuity"])
                            if "self_collision" in r and isinstance(r["self_collision"], dict):
                                r["self_collision"] = SelfCollisionInfo(**r["self_collision"])
                            if "jitter" in r and isinstance(r["jitter"], dict):
                                r["jitter"] = JitterInfo(**r["jitter"])
                            reports.append(CheckReport(**r))
                        elif isinstance(r, CheckReport):
                            reports.append(r)
                    except Exception as e:
                        print(f"[WARN] Failed to load report from progress file: {e}")
                        continue
                print(f"Loaded {len(processed_files)} processed files from progress file")
        except Exception as e:
            print(f"[WARN] Failed to load progress file: {e}")
    
    # Filter out already processed files
    remaining_files = [f for f in motion_files if str(f.relative_to(motion_folder)) not in processed_files]
    print(f"Remaining files to process: {len(remaining_files)}")
    
    # Process files in batches
    for batch_idx in range(0, len(remaining_files), batch_size):
        batch_files = remaining_files[batch_idx:batch_idx + batch_size]
        batch_start = batch_idx + 1
        batch_end = min(batch_idx + batch_size, len(remaining_files))
        
        print(f"\n[Batch {batch_idx // batch_size + 1}] Processing files {batch_start}-{batch_end} of {len(remaining_files)}")
        
        batch_reports = []
        for file_idx, motion_file in enumerate(batch_files):
            rel_path = str(motion_file.relative_to(motion_folder))
            
            try:
                report = check_motion(
                    str(motion_file),
                    robot_type,
                    foot_names=foot_names,
                    height_threshold=height_threshold,
                    min_seconds=min_seconds,
                    min_ratio=min_ratio,
                    max_joint_jump=max_joint_jump,
                    min_jump_ratio=min_jump_ratio,
                )
                if report:
                    report.motion_file = rel_path
                    batch_reports.append(report)
                    reports.append(report)
                    processed_files.add(rel_path)
            except MemoryError as e:
                print(f"\n[ERROR] Memory error processing {rel_path}: {e}")
                print("[ERROR] Try reducing batch_size or processing fewer files at once")
                # Save progress before exiting
                if save_progress:
                    _save_progress(progress_path, processed_files, reports)
                raise
            except Exception as e:
                print(f"[WARN] Failed to process {rel_path}: {e}")
                # Still mark as processed to avoid retrying
                processed_files.add(rel_path)
            
            # Print progress every 10 files
            if (file_idx + 1) % 10 == 0:
                current_memory = get_memory_usage_mb() if HAS_PSUTIL else 0
                print(f"  Processed {file_idx + 1}/{len(batch_files)} files"
                      + (f", Memory: {current_memory:.1f} MB" if HAS_PSUTIL else ""))
        
        # Memory cleanup after each batch
        if HAS_PSUTIL:
            current_memory = get_memory_usage_mb()
            print(f"Batch complete. Memory usage: {current_memory:.1f} MB")
            
            # Force garbage collection
            gc.collect()
            
            # Check memory threshold
            if check_memory_threshold():
                print("[WARNING] High memory usage detected. Consider reducing batch_size.")
        
        # Save progress after each batch
        if save_progress:
            _save_progress(progress_path, processed_files, reports)
            print(f"Progress saved: {len(processed_files)}/{total_files} files processed")
    
    # Final cleanup
    gc.collect()
    
    if HAS_PSUTIL:
        final_memory = get_memory_usage_mb()
        print(f"\nFinal memory usage: {final_memory:.1f} MB")
        if initial_memory > 0:
            print(f"Memory increase: {final_memory - initial_memory:.1f} MB")
    
    return reports


def main():
    parser = argparse.ArgumentParser(
        description="Auto check robot motions for floating and joint limit violations"
    )
    parser.add_argument(
        "--motion_folder",
        type=str,
        required=True,
        help="Folder containing robot *.pkl files",
    )
    parser.add_argument(
        "--robot_type",
        type=str,
        required=True,
        help=f"Robot type. Available: {list(ROBOT_XML_DICT.keys())}",
    )
    parser.add_argument(
        "--height_threshold",
        type=float,
        default=OFF_GROUND_THRESHOLD,
        help=f"Height above ground to consider off-ground (meters, default: {OFF_GROUND_THRESHOLD})",
    )
    parser.add_argument(
        "--min_seconds",
        type=float,
        default=MIN_CONTINUOUS_SECONDS,
        help=f"Minimum continuous seconds off-ground to flag (default: {MIN_CONTINUOUS_SECONDS})",
    )
    parser.add_argument(
        "--min_ratio",
        type=float,
        default=MIN_CONTINUOUS_RATIO,
        help=f"Minimum ratio of total frames off-ground to flag (default: {MIN_CONTINUOUS_RATIO})",
    )
    parser.add_argument(
        "--foot_names",
        type=str,
        nargs="*",
        default=None,
        help=f"Foot link names to check (default: {DEFAULT_FOOT_KEYS})",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save check reports as JSON (default: auto_check_reports.json)",
    )
    parser.add_argument(
        "--motion_file",
        type=str,
        default=None,
        help="Check single motion file instead of folder",
    )
    parser.add_argument(
        "--max_joint_jump",
        type=float,
        default=MAX_JOINT_JUMP,
        help=f"Maximum allowed joint angle change between consecutive frames in radians (default: {MAX_JOINT_JUMP})",
    )
    parser.add_argument(
        "--min_jump_ratio",
        type=float,
        default=MIN_JOINT_JUMP_RATIO,
        help=f"Minimum ratio of frames with joint jumps to flag as abnormal (default: {MIN_JOINT_JUMP_RATIO})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Number of files to process before clearing memory (default: 100). Reduce if memory issues occur.",
    )
    parser.add_argument(
        "--no_save_progress",
        action="store_true",
        help="Disable periodic progress saving (default: enabled)",
    )
    parser.add_argument(
        "--progress_file",
        type=str,
        default=None,
        help="Path to save/load progress JSON (default: auto_check_progress.json)",
    )
    
    args = parser.parse_args()
    
    foot_names = args.foot_names if args.foot_names else DEFAULT_FOOT_KEYS
    
    if args.motion_file:
        # Single file mode - only output if abnormal
        report = check_motion(
            args.motion_file,
            args.robot_type,
            foot_names=foot_names,
            height_threshold=args.height_threshold,
            min_seconds=args.min_seconds,
            min_ratio=args.min_ratio,
            max_joint_jump=args.max_joint_jump,
            min_jump_ratio=args.min_jump_ratio,
        )
        if report:
            if report.is_abnormal:
                print(f"\n[ABNORMAL] Motion: {report.motion_file}")
                if report.floating.is_floating:
                    print(f"  Floating: {report.floating.is_floating}")
                    print(f"    Violation: {report.floating.violation_reason}")
                if report.joint_discontinuity.has_discontinuity:
                    print(f"  Joint discontinuity: {report.joint_discontinuity.has_discontinuity}")
                    print(f"    Max jump: {report.joint_discontinuity.max_jump_deg:.2f} deg at frame {report.joint_discontinuity.max_jump_frame} (joint {report.joint_discontinuity.max_jump_joint})")
                    print(f"    Details: {report.joint_discontinuity.discontinuity_details}")
            if report.self_collision.has_self_collision:
                print(f"  Self-collision: {report.self_collision.has_self_collision}")
                print(f"    Collision ratio: {report.self_collision.collision_ratio:.2%}")
                print(f"    Max continuous: {report.self_collision.max_continuous_collision_seconds:.2f}s "
                      f"({report.self_collision.max_continuous_collision_ratio:.2%})")
                if report.self_collision.top_collision_pairs:
                    print("    Top pairs:")
                    for pair in report.self_collision.top_collision_pairs:
                        print(f"      - {pair}")
                if report.self_collision.violation_reason:
                    print(f"    Reason: {report.self_collision.violation_reason}")
            if report.jitter.has_jitter:
                print(f"  Jitter: {report.jitter.has_jitter}")
                print(f"    Jitter ratio: {report.jitter.jitter_ratio:.2%}")
                print(f"    Max joint acceleration: {report.jitter.max_joint_acceleration_deg:.2f} deg/s^2 "
                      f"at frame {report.jitter.max_joint_acceleration_frame} (joint {report.jitter.max_joint_acceleration_joint})")
                print(f"    Max root acceleration: {report.jitter.max_root_acceleration:.2f} m/s^2 "
                      f"at frame {report.jitter.max_root_acceleration_frame}")
                if report.jitter.violation_reason:
                    print(f"    Reason: {report.jitter.violation_reason}")
            else:
                # Normal motion, no output
                pass
        else:
            print(f"Failed to analyze {args.motion_file}")
    else:
        # Folder mode
        reports = scan_folder(
            args.motion_folder,
            args.robot_type,
            height_threshold=args.height_threshold,
            min_seconds=args.min_seconds,
            min_ratio=args.min_ratio,
            foot_names=foot_names,
            batch_size=args.batch_size,
            save_progress=not args.no_save_progress,
            progress_file=args.progress_file,
        )
        
        abnormal_reports = [r for r in reports if r.is_abnormal]
        
        print(f"\n{'='*80}")
        print(f"Check Results")
        print(f"{'='*80}")
        print(f"Total motions scanned: {len(reports)}")
        print(f"Abnormal motions detected: {len(abnormal_reports)}")
        print(f"Percentage: {len(abnormal_reports)/len(reports)*100:.2f}%\n")
        
        # Count by issue type
        floating_count = sum(1 for r in abnormal_reports if r.floating.is_floating)
        discontinuity_count = sum(1 for r in abnormal_reports if r.joint_discontinuity.has_discontinuity)
        self_collision_count = sum(1 for r in abnormal_reports if r.self_collision.has_self_collision)
        jitter_count = sum(1 for r in abnormal_reports if r.jitter.has_jitter)
        
        print(f"Floating issues: {floating_count}")
        print(f"Joint discontinuities: {discontinuity_count}")
        print(f"Self-collisions: {self_collision_count}")
        print(f"Jitter: {jitter_count}\n")
        
        if abnormal_reports:
            print("Abnormal Motions:")
            print("-" * 80)
            for report in sorted(
                abnormal_reports,
                key=lambda r: (
                    r.floating.is_floating,
                    r.joint_discontinuity.has_discontinuity,
                    r.self_collision.has_self_collision,
                    r.jitter.has_jitter,
                ),
                reverse=True,
            ):
                issues = []
                if report.floating.is_floating:
                    issues.append("floating")
                if report.joint_discontinuity.has_discontinuity:
                    issues.append("discontinuity")
                if report.self_collision.has_self_collision:
                    issues.append("self_collision")
                if report.jitter.has_jitter:
                    issues.append("jitter")
                print(f"{report.motion_file}: {', '.join(issues)}")
        else:
            print("No abnormal motions detected.")
        
        # Save abnormal reports to JSON
        output_json = args.output_json
        if output_json is None and abnormal_reports:
            output_json = "auto_check_reports.json"
        
        if output_json:
            output_path = Path(output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save only abnormal reports
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    [asdict(r) for r in abnormal_reports],
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"\nAbnormal reports ({len(abnormal_reports)} motions) saved to: {output_path}")
        
        # Save annotations format (only abnormal motions)
        annotations_path = Path("auto_check_annotations.json")
        annotations = {}
        
        # Get project root (assuming script is in scripts/ directory)
        project_root = Path(__file__).parent.parent.resolve()
        
        # Get base path for motion_path (resolve to absolute path)
        motion_folder_path = (project_root / args.motion_folder).resolve()
        
        # Only annotate motions that have detected problems
        for report in reports:
            # Skip normal motions (no annotation)
            if not report.is_abnormal:
                continue
            
            # Generate reason string for abnormal motions
            issues = []
            if report.floating.is_floating:
                issues.append("1. 与地形/场景存在交互")
            if report.joint_discontinuity.has_discontinuity:
                issues.append("3. 动作衔接异常或抖动")
            if report.self_collision.has_self_collision:
                issues.append("2. 穿模或自碰撞")
            if report.jitter.has_jitter:
                issues.append("3. 动作衔接异常或抖动")  # Jitter is also considered as jitter/shake
            reason = ", ".join(issues) if issues else "其他"
            
            # Get motion_path (relative to project root)
            motion_file_path = motion_folder_path / report.motion_file
            # Convert to relative path from project root
            try:
                motion_path = str(motion_file_path.relative_to(project_root))
            except ValueError:
                # If not relative to project root, use absolute path or keep as is
                motion_path = str(motion_file_path)
            
            # Use motion_file as key (same format as annotations.json)
            key = report.motion_file
            
            annotations[key] = {
                "label": "abnormal",
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
                "motion_path": motion_path,
            }
        
        # Load existing annotations if they exist (to merge/update)
        if annotations_path.exists():
            try:
                with open(annotations_path, "r", encoding="utf-8") as f:
                    existing_annotations = json.load(f)
                # Update existing annotations with new abnormal results
                # Only update entries for motions that were checked and found abnormal
                existing_annotations.update(annotations)
                annotations = existing_annotations
            except Exception as e:
                print(f"[WARN] Failed to load existing annotations: {e}")
        
        # Save annotations (only abnormal motions are annotated)
        with open(annotations_path, "w", encoding="utf-8") as f:
            json.dump(
                annotations,
                f,
                ensure_ascii=False,
                indent=2,
            )
        abnormal_count = sum(1 for v in annotations.values() if v.get("label") == "abnormal")
        print(f"\nAnnotations ({abnormal_count} abnormal motions) saved to: {annotations_path}")


if __name__ == "__main__":
    main()