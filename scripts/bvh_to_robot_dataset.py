# python scripts/bvh_to_robot_dataset.py --src_folder data/lafan --tgt_folder retargeted_data/lafan/ --robot unitree_g1_fixed_wrist --num_cpus 20 --memory_threshold 30
import argparse
import pathlib
import os
import sys
import multiprocessing as mp
import mujoco as mj
import numpy as np
from tqdm import tqdm
import torch
import pickle
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from rich import print
import joblib
import gc
import time
import psutil
import tracemalloc
G1_ROTATION_AXIS = torch.tensor([[
    [0, 1, 0], # l_hip_pitch 
    [1, 0, 0], # l_hip_roll
    [0, 0, 1], # l_hip_yaw
    
    [0, 1, 0], # l_knee
    [0, 1, 0], # l_ankle_pitch
    [1, 0, 0], # l_ankle_roll
    
    [0, 1, 0], # r_hip_pitch
    [1, 0, 0], # r_hip_roll
    [0, 0, 1], # r_hip_yaw
    
    [0, 1, 0], # r_knee
    [0, 1, 0], # r_ankle_pitch
    [1, 0, 0], # r_ankle_roll
    
    [0, 0, 1], # waist_yaw_joint
    [1, 0, 0], # waist_roll_joint
    [0, 1, 0], # waist_pitch_joint
   
    [0, 1, 0], # l_shoulder_pitch
    [1, 0, 0], # l_shoulder_roll
    [0, 0, 1], # l_shoulder_yaw
    
    [0, 1, 0], # l_elbow
    
    [0, 1, 0], # r_shoulder_pitch
    [1, 0, 0], # r_shoulder_roll
    [0, 0, 1], # r_shoulder_yaw
    
    [0, 1, 0], # r_elbow
    ]])

HERE = pathlib.Path(__file__).parent

def check_memory(threshold_gb):
    """检查内存使用情况"""
    mem = psutil.virtual_memory()
    used_memory_gb = (mem.total - mem.available) / (1024 ** 3)
    available_memory_gb = mem.available / (1024 ** 3)
    if available_memory_gb < threshold_gb:
        print(f"[WARNING] Memory usage:{used_memory_gb:.2f} GB, available:{available_memory_gb:.2f} GB, exceeding the threshold of {threshold_gb} GB.")
        return True
    return False


def process_file(bvh_file_path, tgt_file_path, robot, src_folder, tgt_folder, total_files, memory_threshold, verbose=False):
    """处理单个 BVH 文件并保存"""
    def log_memory(message):
        if verbose:
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / (1024 ** 3)
            print(f"[MEMORY] {message}: {memory_usage:.2f} GB")
    
    # Start memory tracking if verbose
    if verbose:
        tracemalloc.start()
    
    log_memory("Initial memory usage")
    
    # Check memory before processing
    num_pause = 0
    while check_memory(memory_threshold):
        print(f"[PAUSE] Paused processing {bvh_file_path} to prevent memory overflow. num_pause: {num_pause}")
        time.sleep(60 * 2)
        num_pause += 1
        if num_pause > 10:
            print(f"[ERROR] Memory usage is still high after 10 pauses. Skipping file.")
            return
    
    # Load BVH file
    try:
        lafan1_data_frames, actual_human_height = load_bvh_file(bvh_file_path, format="nokov")
        src_fps = 30  # LAFAN1 data is typically 30 FPS
        log_memory("After loading BVH data")
    except Exception as e:
        print(f"Error loading {bvh_file_path}: {e}")
        return
    
    # Initialize the retargeting system
    try:
        retarget = GMR(
            src_human="bvh_nokov",
            tgt_robot=robot,
            actual_human_height=actual_human_height,
        )
        model = mj.MjModel.from_xml_path(retarget.xml_file)
        data = mj.MjData(model)
    except Exception as e:
        print(f"Error initializing retargeting for {bvh_file_path}: {e}")
        return
    
    # Retarget to get all qpos
    qpos_list = []
    for curr_frame in range(len(lafan1_data_frames)):
        smplx_data = lafan1_data_frames[curr_frame]
        qpos = retarget.retarget(smplx_data)
        qpos_list.append(qpos.copy())
    
    qpos_list = np.array(qpos_list)
    log_memory("After retargeting")
    
    root_pos = qpos_list[:, :3]
    root_rot = qpos_list[:, 3:7]
    root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]
    dof_pos = qpos_list[:, 7:]
    num_frames = root_pos.shape[0]
    
    # Initialize the forward kinematics
    device = "cuda:0"
    kinematics_model = KinematicsModel(retarget.xml_file, device=device)
    
    # Height adjustment
    HEIGHT_ADJUST = True
    PERFRAME_ADJUST = False
    if HEIGHT_ADJUST:
        body_pos, _ = kinematics_model.forward_kinematics(
            torch.from_numpy(root_pos).to(device=device, dtype=torch.float),
            torch.from_numpy(root_rot).to(device=device, dtype=torch.float),
            torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
        )
        ground_offset = 0.00
        if not PERFRAME_ADJUST:
            lowest_height = torch.min(body_pos[..., 2]).item()
            root_pos[:, 2] = root_pos[:, 2] - lowest_height + ground_offset
        else:
            for i in range(root_pos.shape[0]):
                lowest_body_part = torch.min(body_pos[i, :, 2])
                root_pos[i, 2] = root_pos[i, 2] - lowest_body_part + ground_offset
    
    # Convert rotation to axis-angle
    rot_vec_all = []
    for frame_idx in range(num_frames):
        rotation = R.from_quat(root_rot[frame_idx])
        rotvec = rotation.as_rotvec()
        rotvec = torch.from_numpy(rotvec)
        rot_vec_all.append(rotvec)
    
    device = "cpu"
    rot_vec_all = torch.cat(rot_vec_all, dim=0).view(-1, 3).to(device=device, dtype=torch.float)
    dof_pos_all = torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
    pose_aa = torch.cat([rot_vec_all[None, :, None], G1_ROTATION_AXIS * dof_pos_all[None,:,:,None], torch.zeros((1, num_frames, 3, 3), device=device)], axis=2)
    
    # Remove the first frame
    root_pos = root_pos[1:]
    root_rot = root_rot[1:]
    dof_pos = dof_pos[1:]
    pose_aa = pose_aa[:, 1:]
    num_frames = num_frames - 1
    
    motion_data = {
        "root_trans_offset": root_pos,
        "pose_aa": pose_aa.squeeze().cpu().detach().numpy(),
        "dof": dof_pos,
        "root_rot": root_rot,
        "fps": 30,
    }
    
    # Generate motion name
    rel_path = os.path.relpath(bvh_file_path, src_folder)
    motion_name = rel_path.replace(".bvh", "").replace(os.sep, "_")
    
    # Save to individual pkl file
    data_dump = {}
    data_dump[motion_name] = motion_data
    os.makedirs(os.path.dirname(tgt_file_path), exist_ok=True)
    with open(tgt_file_path, "wb") as f:
        joblib.dump(data_dump, tgt_file_path)
    
    # Progress print based on tgt_folder
    done = 0
    for root, _, files in os.walk(tgt_folder):
        done += len([f for f in files if f.endswith('.pkl')])
    print(f"Processed {done}/{total_files}: {tgt_file_path}")
    
    log_memory("After processing complete")
    
    if verbose:
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        print("\nTop 10 memory-consuming lines:")
        for stat in top_stats[:10]:
            print(stat)
        tracemalloc.stop()
    
    # Clean cache
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_folder",
        help="Folder containing BVH motion files to load.",
        required=True,
        type=str,
    )
    
    parser.add_argument(
        "--tgt_folder",
        help="Folder to save the retargeted motion files.",
        default="../../motion_data/LAFAN1_g1_gmr"
    )
    
    parser.add_argument(
        "--robot",
        default="unitree_g1_fixed_wrist",
    )
    
    parser.add_argument(
        "--override",
        default=False,
        action="store_true",
    )
    
    parser.add_argument(
        "--target_fps",
        default=30,
        type=int,
    )
    
    
    parser.add_argument(
        "--num_cpus",
        default=4,
        type=int,
        help="Number of CPUs to use for parallel processing"
    )
    
    parser.add_argument(
        "--memory_threshold",
        default=10,
        type=float,
        help="Memory threshold in GB for pausing processing (default: 10)"
    )

    args = parser.parse_args()
    
    # Print CPU information
    print(f"Total CPUs: {mp.cpu_count()}")
    print(f"Using {args.num_cpus} CPUs.")
    
    src_folder = args.src_folder
    tgt_folder = args.tgt_folder
    
    verbose = False
    
    # Collect all BVH files and prepare arguments
    args_list = []
    for dirpath, _, filenames in os.walk(src_folder):
        for filename in sorted(filenames):
            if filename.endswith(".bvh"):
                bvh_file_path = os.path.join(dirpath, filename)
                # Generate target file path
                rel_path = os.path.relpath(bvh_file_path, src_folder)
                tgt_file_path = os.path.join(tgt_folder, rel_path.replace(".bvh", ".pkl"))
                
                # Skip if file already exists and not override
                if not os.path.exists(tgt_file_path) or args.override:
                    args_list.append((bvh_file_path, tgt_file_path, args.robot, src_folder, tgt_folder))
    
    total_files = len(args_list)
    print(f"Total number of BVH files to process: {total_files}")
    print(f"Memory threshold: {args.memory_threshold} GB")
    
    # Process files in parallel
    # Each process will save its own file and print progress
    with mp.Pool(args.num_cpus) as pool:
        pool.starmap(process_file, [args_i + (total_files, args.memory_threshold, verbose) for args_i in args_list])

    print(f"Done. Saved to {tgt_folder}")
