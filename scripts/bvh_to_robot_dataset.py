import argparse
import pathlib
import os
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

if __name__ == "__main__":
    HERE = pathlib.Path(__file__).parent

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
        default="unitree_g1",
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

    args = parser.parse_args()
    
    src_folder = args.src_folder
    tgt_folder = args.tgt_folder

   
   
        
    # walk over all files in src_folder
    for dirpath, _, filenames in os.walk(src_folder):
        for filename in tqdm(sorted(filenames), desc="Retargeting files"):
            if not filename.endswith(".bvh"):
                continue
                
            # get the bvh file path
            bvh_file_path = os.path.join(dirpath, filename)
            
            # get the target file path
            tgt_file_path = bvh_file_path.replace(src_folder, tgt_folder).replace(".bvh", ".pkl")

            if os.path.exists(tgt_file_path) and not args.override:
                print(f"Skipping {bvh_file_path} because {tgt_file_path} exists")
                continue
            
            # Load LAFAN1 trajectory
            try:
                lafan1_data_frames, actual_human_height = load_bvh_file(bvh_file_path, format="nokov")
                src_fps = 30  # LAFAN1 data is typically 30 FPS
            except Exception as e:
                print(f"Error loading {bvh_file_path}: {e}")
                continue

            
            # Initialize the retargeting system
            retarget = GMR(
                src_human="bvh_nokov",
                tgt_robot=args.robot,
                actual_human_height=actual_human_height,
            )
            model = mj.MjModel.from_xml_path(retarget.xml_file)
            data = mj.MjData(model)

            

            # retarget to get all qpos
            qpos_list = []
            for curr_frame in range(len(lafan1_data_frames)):
                smplx_data = lafan1_data_frames[curr_frame]
                
                # Retarget till convergence
                qpos = retarget.retarget(smplx_data)
                
                qpos_list.append(qpos.copy())
            
            qpos_list = np.array(qpos_list)


            root_pos = qpos_list[:, :3]
            root_rot = qpos_list[:, 3:7]
            root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]
            dof_pos = qpos_list[:, 7:]
            num_frames = root_pos.shape[0]


            # Initialize the forward kinematics
            device = "cuda:0"
            kinematics_model = KinematicsModel(retarget.xml_file, device=device)
            

            
            # obtain local body pos
            # identity_root_pos = torch.zeros((num_frames, 3), device=device)
            # identity_root_rot = torch.zeros((num_frames, 4), device=device)
            # identity_root_rot[:, -1] = 1.0
            # local_body_pos, _ = kinematics_model.forward_kinematics(
            #     identity_root_pos, 
            #     identity_root_rot, 
            #     torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
            # )
            # body_names = kinematics_model.body_names

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
            
            
            rot_vec_all = []
            for frame_idx in range(num_frames):
                rotation = R.from_quat(root_rot[frame_idx])
                rotvec = rotation.as_rotvec()
                rotvec = torch.from_numpy(rotvec)
                rot_vec_all.append(rotvec)
            device = "cpu"
            rot_vec_all = torch.cat(rot_vec_all, dim=0).view(-1, 3).to(device=device, dtype=torch.float)
            dof_pos_all = torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
            pose_aa = torch.cat([rot_vec_all[None, :, None], G1_ROTATION_AXIS * dof_pos_all[None,:,:,None], torch.zeros((1, num_frames, 3, 3),device=device)], axis = 2)
                    
            # remove the first frame
            root_pos = root_pos[1:]
            root_rot = root_rot[1:]
            dof_pos = dof_pos[1:]
            pose_aa = pose_aa[:,1:]
            num_frames = num_frames - 1
            
            motion_data = {
                "root_trans_offset": root_pos,
                "pose_aa": pose_aa.squeeze().cpu().detach().numpy(),
                "dof": dof_pos,
                "root_rot": root_rot,
                "fps": 30,
            }
            # motion_data = {
            #     "root_pos": root_pos,
            #     "root_rot": root_rot,
            #     "dof_pos": dof_pos,
            #     "local_body_pos": local_body_pos.detach().cpu().numpy(),
            #     "fps": src_fps,
            #     "link_body_list": body_names,
            # }
            
            data_dump = {}
            motion_name = tgt_file_path.split("/")[-1].split(".")[0]
            data_dump[motion_name] = motion_data
            os.makedirs(os.path.dirname(tgt_file_path), exist_ok=True)
            with open(tgt_file_path, "wb") as f:
                # pickle.dump(motion_data, f)
                joblib.dump(data_dump, tgt_file_path)

    print("Done. saved to ", tgt_folder)
