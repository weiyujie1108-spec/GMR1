import argparse
import pathlib
import os
import time

import numpy as np
import mujoco as mj
from smplx.joint_names import JOINT_NAMES

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
from general_motion_retargeting.ik_utils import extract_foot_sticking_sequence, analyze_and_plot_robot_motion, analyze_and_plot_human_motion, fix_feet_sliding

from rich import print

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smplx_file",
        help="SMPLX motion file to load.",
        type=str,
        # required=True,
        default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male1General_c3d/General_A1_-_Stand_stageii.npz",
        # default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male2MartialArtsKicks_c3d/G8_-__roundhouse_left_stageii.npz"
        # default="/home/yanjieze/projects/g1_wbc/TWIST-dev/motion_data/AMASS/KIT_572_dance_chacha11_stageii.npz"
        # default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male2MartialArtsPunches_c3d/E1_-__Jab_left_stageii.npz",
        # default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male1Running_c3d/Run_C24_-_quick_side_step_left_stageii.npz",
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_fixed_wrist", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1", 
                "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", "berkeley_humanoid_lite", "booster_k1",
                "pnd_adam_lite", "openloong", "tienkung"],
        default="unitree_g1",
    )
    
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )

    parser.add_argument(
        "--record_video",
        default=False,
        action="store_true",
        help="Record the video.",
    )

    parser.add_argument(
        "--rate_limit",
        default=False,
        action="store_true",
        help="Limit the rate of the retargeted robot motion to keep the same as the human motion.",
    )

    parser.add_argument(
        "--fix_foot_sliding",
        default=False,
        action="store_true",
        help="Fix the foot sliding.",
    )

    parser.add_argument(
        "--interaction_mesh",
        default=False,
        action="store_true",
        help="Consider interaction mesh task.",
    )

    args = parser.parse_args()


    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
    
    
    # Load SMPLX trajectory
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER
    )
    
    # align fps
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
    if args.fix_foot_sliding:
        # extract foot sticking sequence
        joint_names = JOINT_NAMES[: len(body_model.parents)]

        smplx_data_joints = np.zeros((len(smplx_data_frames), len(joint_names), 3))
        for i, frame in enumerate(smplx_data_frames):
            for j, name in enumerate(joint_names):
                smplx_data_joints[i, j] = frame[name][0]
        contact_threshold = 0.01
        velocity_threshold = 0.05

        foot_sticking_sequence = extract_foot_sticking_sequence(
            smplx_data_joints,
            joint_names,
            ["left_foot", "right_foot"],
            smpl_contact_threshold_relative=contact_threshold,
            velocity_threshold=velocity_threshold,
            fps=aligned_fps
        )

        # Visualize foot sticking sequence and feet analysis
        analyze_and_plot_human_motion(
            foot_sticking_sequence,
            smplx_data_joints,
            joint_names,
            aligned_fps,
            contact_threshold=contact_threshold,
            velocity_threshold=velocity_threshold
        )
        
    # Initialize the retargeting system
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
        use_interaction_mesh=args.interaction_mesh,
    )
    
    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=aligned_fps,
                                            transparent_robot=1 if args.interaction_mesh else 0,
                                            record_video=args.record_video,
                                            video_path=f"videos/{args.robot}_{args.smplx_file.split('/')[-1].split('.')[0]}.mp4",)
    

    curr_frame = 0
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
    
    # Lists to store robot motion data
    qpos_list = []
    robot_left_foot_pos_list = []
    robot_right_foot_pos_list = []

    # Start the viewer
    i = 0

    while True:
        if args.loop:
            i = (i + 1) % len(smplx_data_frames)
        else:
            i += 1
            if i >= len(smplx_data_frames):
                break
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
        
        # Update task targets.
        smplx_data = smplx_data_frames[i]
        if args.fix_foot_sliding:
            foot_sticking = foot_sticking_sequence[i]
            retarget.set_foot_sticking(foot_sticking)
            # Store robot motion data
            left_foot_name = retarget.foot_stick_limit.left_name
            right_foot_name = retarget.foot_stick_limit.right_name
            left_id = mj.mj_name2id(retarget.model, mj.mjtObj.mjOBJ_BODY, left_foot_name)
            right_id = mj.mj_name2id(retarget.model, mj.mjtObj.mjOBJ_BODY, right_foot_name)
            robot_left_foot_pos_list.append(retarget.configuration.data.xpos[left_id].copy())
            robot_right_foot_pos_list.append(retarget.configuration.data.xpos[right_id].copy())
        # retarget
        qpos = retarget.retarget(smplx_data)

        # visualization for interaction mesh
        if args.interaction_mesh:
            robot_keypoints = {}
            correspondence_lines = []
            ground_pts = []
            x_range = np.linspace(-2, 2, 10)
            y_range = np.linspace(-2, 2, 10)
            for x in x_range:
                for y in y_range:
                    ground_pts.append(np.array([x, y, 0.0]))
            extra_points = [{
                'pos': ground_pts,
                'color': [1, 0, 0, 0.5],
                'size': [0.02, 0.02, 0.02]
            }]

            for robot_link_name, entry in retarget.ik_match_table1.items():
                human_joint_name = entry[0]

                robot_pos = None
                try:
                    bid = mj.mj_name2id(retarget.model, mj.mjtObj.mjOBJ_BODY, robot_link_name)
                    if bid != -1:
                        robot_pos = retarget.configuration.data.xpos[bid].copy()
                        robot_keypoints[robot_link_name] = robot_pos
                except Exception as e:
                    pass

                human_pos = None
                if human_joint_name in retarget.scaled_human_data:
                    human_pos = retarget.scaled_human_data[human_joint_name][0] + np.array([0.0, 0.0, 0.0]) # Add offset if needed

                if robot_pos is not None and human_pos is not None:
                    correspondence_lines.append((human_pos, robot_pos))
        else:
            robot_keypoints = None
            extra_points = None
            correspondence_lines = None

        # visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retarget.scaled_human_data,
            # human_motion_data=smplx_data,
            human_pos_offset=np.array([0.0, 0.0, 0.0]),
            show_human_body_name=False,
            rate_limit=args.rate_limit,
            follow_camera=True,
            robot_keypoints=robot_keypoints,
            extra_points=extra_points,
            lines=correspondence_lines
        )
        if args.save_path is not None:
            qpos_list.append(qpos)
            
    # Fix feet sliding
    if args.fix_foot_sliding:
        qpos_list, robot_left_foot_pos_list, robot_right_foot_pos_list = fix_feet_sliding(
            qpos_list,
            robot_left_foot_pos_list,
            robot_right_foot_pos_list,
            foot_sticking_sequence
        )
    if args.save_path is not None:
        import pickle
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        local_body_pos = None
        body_names = None
        
        motion_data = {
            "fps": aligned_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    if args.fix_foot_sliding:
        # Robot Foot Analysis
        robot_left_foot_pos = np.array(robot_left_foot_pos_list)
        robot_right_foot_pos = np.array(robot_right_foot_pos_list)

        left_foot_stick = [frame["left_foot"] for frame in foot_sticking_sequence]
        right_foot_stick = [frame["right_foot"] for frame in foot_sticking_sequence]
        analyze_and_plot_robot_motion(
            robot_left_foot_pos,
            robot_right_foot_pos,
            qpos_list,
            retarget,
            aligned_fps,
            contact_threshold=contact_threshold,
            velocity_threshold=velocity_threshold,
            human_left_foot_stick=left_foot_stick,
            human_right_foot_stick=right_foot_stick,
        )
    
    robot_motion_viewer.close()
