import argparse
import pickle
import os

import numpy as np
import joblib
# python scripts/batch_gmr_pkl_to_csv.py --input_folder retargeted_data/ground_29dof --output_folder retargeted_data/ground_29dof_csv
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GMR pickle files to CSV (for beyondmimic)")
    parser.add_argument(
        "--input_folder", default="retargeted_data/ground_29dof", type=str, help="Path to the folder containing pickle files from GMR",
    )
    parser.add_argument(
        "--output_folder", default="retargeted_data/ground_29dof_csv", type=str, help="Path to the folder to save the csv files",
    )
    args = parser.parse_args()

    out_folder = os.path.join(args.output_folder)
    os.makedirs(out_folder, exist_ok=True)

    # Get all pkl files first for accurate counting
    pkl_files = [f for f in os.listdir(args.input_folder) if f.endswith(".pkl")]
    total_files = len(pkl_files)
    
    if total_files == 0:
        print(f"No .pkl files found in {args.input_folder}")
        exit(1)

    for i, file in enumerate(pkl_files):
        try:
            file_path = os.path.join(args.input_folder, file)
            try:
                with open(file_path, "rb") as f:
                    motion_data = joblib.load(f)
            except:
                with open(file_path, "rb") as f:
                    motion_data = pickle.load(f)
            
            # Handle nested dictionary format: {motion_name: motion_data}
            if isinstance(motion_data, dict) and len(motion_data) == 1:
                inner_key = list(motion_data.keys())[0]
                motion_data = motion_data[inner_key]
            
            # Validate required keys
            required_keys = ["dof_pos", "fps", "root_pos", "root_rot"]
            missing_keys = [key for key in required_keys if key not in motion_data]
            if missing_keys:
                print(f"Warning: {file} missing keys: {missing_keys}, skipping...")
                continue
            
            dof_pos = motion_data["dof_pos"]
            frame_rate = motion_data["fps"]            
            motion = np.zeros((dof_pos.shape[0], dof_pos.shape[1] + 7), dtype=np.float32)
            motion[:, :3] = motion_data["root_pos"]
            motion[:, 3:7] = motion_data["root_rot"]
            motion[:, 7:] = dof_pos
            
            if frame_rate > 30:
                # downsample to 30 fps
                downsample_factor = frame_rate / 30.0
                indices = np.arange(0, motion.shape[0], downsample_factor).astype(int)
                # Ensure indices don't exceed array bounds
                indices = indices[indices < motion.shape[0]]
                old_length = motion.shape[0]
                motion = motion[indices]
                print(f"Downsampled {file} from {old_length} to {motion.shape[0]} frames")
            
            output_path = os.path.join(out_folder, file.replace(".pkl", ".csv"))
            np.savetxt(output_path, motion, delimiter=",")
            print(f"({i+1}/{total_files}) Saved to {output_path}")
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
