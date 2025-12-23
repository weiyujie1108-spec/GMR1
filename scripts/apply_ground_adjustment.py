#!/usr/bin/env python3
"""
Apply ground adjustment to existing pkl motion files.
This is a post-processing script that modifies root_pos without re-running retargeting.

Usage:
    python scripts/apply_ground_adjustment.py --src_folder data/pkl_raw --tgt_folder data/pkl_phuma --ground_method phuma
    python scripts/apply_ground_adjustment.py --src_folder data/pkl_raw --tgt_folder data/pkl_min --ground_method min
    python scripts/apply_ground_adjustment.py --src_folder data/pkl_raw --tgt_folder data/pkl_phuma_lift --ground_method phuma+lift
"""

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting.params import ROBOT_XML_DICT
from general_motion_retargeting.utils.ground_adjustment import adjust_root_pos_to_ground


def process_pkl_file(
    src_path: str,
    tgt_path: str,
    kinematics_model: KinematicsModel,
    ground_method: str,
    ground_offset: float = 0.0,
) -> bool:
    """
    Apply ground adjustment to a single pkl file.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load the pkl file
        data = joblib.load(src_path)
        
        # Handle both formats: {motion_name: motion_data} and direct motion_data
        if isinstance(data, dict):
            # Check if it's {motion_name: motion_data} format
            keys = list(data.keys())
            if len(keys) == 1 and isinstance(data[keys[0]], dict):
                # Nested format: {motion_name: motion_data}
                motion_name = keys[0]
                motion_data = data[motion_name]
                is_nested = True
            else:
                # Direct format: motion_data dict
                motion_data = data
                motion_name = None
                is_nested = False
        else:
            print(f"[WARN] Unexpected data format in {src_path}")
            return False
        
        # Extract required fields (use explicit None checks to avoid array boolean ambiguity)
        root_pos = motion_data.get("root_pos")
        if root_pos is None:
            root_pos = motion_data.get("root_trans_offset")
        root_rot = motion_data.get("root_rot")
        dof_pos = motion_data.get("dof_pos")
        if dof_pos is None:
            dof_pos = motion_data.get("dof")
        
        # Check for None values (use 'is None' to avoid array boolean ambiguity)
        if root_pos is None or root_rot is None or dof_pos is None:
            print(f"[WARN] Missing required fields in {src_path}")
            return False
        
        # Ensure numpy arrays and validate shapes
        try:
            root_pos = np.asarray(root_pos, dtype=np.float64)
            root_rot = np.asarray(root_rot, dtype=np.float64)
            dof_pos = np.asarray(dof_pos, dtype=np.float64)
            
            # Validate that arrays are not empty
            if root_pos.size == 0 or root_rot.size == 0 or dof_pos.size == 0:
                print(f"[WARN] Empty arrays in {src_path}")
                return False
            
            # Validate dimensions
            if root_pos.ndim != 2 or root_pos.shape[1] != 3:
                print(f"[WARN] Invalid root_pos shape in {src_path}: {root_pos.shape}, expected (T, 3)")
                return False
        except (ValueError, TypeError) as e:
            print(f"[WARN] Failed to convert arrays in {src_path}: {e}")
            return False
        
        # Apply ground adjustment
        root_pos_adjusted, debug_info = adjust_root_pos_to_ground(
            root_pos=root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
            kinematics_model=kinematics_model,
            ground_method=ground_method,
            ground_offset=ground_offset,
        )
        
        # Update motion data
        motion_data["root_pos"] = root_pos_adjusted
        if "root_trans_offset" in motion_data:
            motion_data["root_trans_offset"] = root_pos_adjusted
        
        # Save to target path
        os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
        
        if is_nested:
            joblib.dump({motion_name: motion_data}, tgt_path)
        else:
            joblib.dump(motion_data, tgt_path)
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to process {src_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Apply ground adjustment to existing pkl motion files"
    )
    parser.add_argument(
        "--src_folder",
        type=str,
        required=True,
        help="Source folder containing pkl files",
    )
    parser.add_argument(
        "--tgt_folder",
        type=str,
        required=True,
        help="Target folder to save adjusted pkl files",
    )
    parser.add_argument(
        "--ground_method",
        type=str,
        default="phuma",
        choices=["min", "phuma", "phuma+lift"],
        help="Ground adjustment method (default: phuma)",
    )
    parser.add_argument(
        "--ground_offset",
        type=float,
        default=0.0,
        help="Additional ground offset in meters (default: 0.0)",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="unitree_g1_fixed_wrist",
        choices=list(ROBOT_XML_DICT.keys()),
        help="Robot type for kinematics model",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Override existing files in target folder",
    )
    
    args = parser.parse_args()
    
    # Initialize kinematics model
    print(f"Initializing kinematics model for {args.robot}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xml_file = str(ROBOT_XML_DICT[args.robot])
    kinematics_model = KinematicsModel(xml_file, device=device)
    
    # Collect all pkl files
    pkl_files = []
    for root, _, files in os.walk(args.src_folder):
        for filename in files:
            if filename.endswith(".pkl"):
                src_path = os.path.join(root, filename)
                rel_path = os.path.relpath(src_path, args.src_folder)
                tgt_path = os.path.join(args.tgt_folder, rel_path)
                
                if not args.override and os.path.exists(tgt_path):
                    continue
                    
                pkl_files.append((src_path, tgt_path))
    
    if not pkl_files:
        print("No pkl files to process.")
        return
    
    print(f"Processing {len(pkl_files)} pkl files...")
    print(f"  Source: {args.src_folder}")
    print(f"  Target: {args.tgt_folder}")
    print(f"  Method: {args.ground_method}")
    print(f"  Offset: {args.ground_offset}m")
    
    success_count = 0
    fail_count = 0
    
    for src_path, tgt_path in tqdm(pkl_files, desc="Applying ground adjustment"):
        if process_pkl_file(
            src_path, tgt_path, kinematics_model, args.ground_method, args.ground_offset
        ):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\nDone!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Output: {args.tgt_folder}")


if __name__ == "__main__":
    main()

