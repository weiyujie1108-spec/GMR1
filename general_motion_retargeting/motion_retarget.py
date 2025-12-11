
import mink
import mujoco as mj
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from .params import ROBOT_XML_DICT, IK_CONFIG_DICT
from rich import print
from .ik_utils import FootStickLimit

class GeneralMotionRetargeting:
    """General Motion Retargeting (GMR).
    """
    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        actual_human_height: float = None,
        use_segment_length_scaling: bool=True,
        solver: str="daqp", # change from "quadprog" to "daqp".
        damping: float=5e-1, # change from 1e-1 to 1e-2.
        verbose: bool=True,
        use_velocity_limit: bool=False,
    ) -> None:

        # load the robot model
        self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
        if verbose:
            print("Use robot model: ", self.xml_file)
        self.model = mj.MjModel.from_xml_path(self.xml_file)
        
        # Print DoF names in order
        print("[GMR] Robot Degrees of Freedom (DoF) names and their order:")
        self.robot_dof_names = {}
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            self.robot_dof_names[dof_name] = i
            if verbose:
                print(f"DoF {i}: {dof_name}")
            
            
        print("[GMR] Robot Body names and their IDs:")
        self.robot_body_names = {}
        for i in range(self.model.nbody):  # 'nbody' is the number of bodies
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i
            if verbose:
                print(f"Body ID {i}: {body_name}")
        
        print("[GMR] Robot Motor (Actuator) names and their IDs:")
        self.robot_motor_names = {}
        for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            self.robot_motor_names[motor_name] = i
            if verbose:
                print(f"Motor ID {i}: {motor_name}")

        # Load the IK config
        with open(IK_CONFIG_DICT[src_human][tgt_robot]) as f:
            ik_config = json.load(f)
        if verbose:
            print("Use IK config: ", IK_CONFIG_DICT[src_human][tgt_robot])
        
        # Flag to enable segment-length-based scaling
        self.use_segment_length_scaling = use_segment_length_scaling  # Set to False to use height-based scaling
        
        # Store original scale table for reference
        self.original_scale_table = ik_config["human_scale_table"].copy()
        
        # compute the scale ratio based on given human height and the assumption in the IK config
        if actual_human_height is not None:
            ratio = actual_human_height / ik_config["human_height_assumption"]
        else:
            ratio = 1.0
            
        # If segment-length-based scaling is disabled, use improved height-based scaling
        if not self.use_segment_length_scaling:
            # Improved scaling strategy: use different scaling for leg joints to prevent knee bending
            # When human is shorter than assumption (ratio < 1), leg joints need less scaling reduction
            # to maintain proper leg length ratio and prevent knee bending
            leg_joint_keywords = ["hip", "knee", "foot", "ankle", "thigh", "shank", "calf", "leg"]
            
            # For leg joints: use a less aggressive scaling when ratio < 1
            # This helps maintain proper leg length ratio and prevents knee bending
            if ratio < 1.0:
                # When human is shorter, reduce leg scaling less aggressively
                # Use a square root function to make the scaling less linear
                leg_ratio = np.sqrt(ratio)  # e.g., ratio=0.837 -> leg_ratio=0.915
                # Or use a linear interpolation between ratio and 1.0
                # leg_ratio = 0.7 * ratio + 0.3 * 1.0  # 70% of ratio + 30% of 1.0
            else:
                # When human is taller, use normal scaling
                leg_ratio = ratio
            
            # adjust the human scale table with improved leg scaling
            for key in ik_config["human_scale_table"].keys():
                key_lower = key.lower()
                # Check if this is a leg joint
                is_leg_joint = any(keyword in key_lower for keyword in leg_joint_keywords)
                
                if is_leg_joint:
                    # Use leg-specific scaling ratio
                    ik_config["human_scale_table"][key] = ik_config["human_scale_table"][key] * leg_ratio
                else:
                    # Use normal scaling ratio for non-leg joints
                    ik_config["human_scale_table"][key] = ik_config["human_scale_table"][key] * ratio
        else:
            # Segment-length-based scaling will be computed on first retarget call
            # For now, keep original scale table
            pass
    

        # used for retargeting
        self.ik_match_table1 = ik_config["ik_match_table1"]
        self.ik_match_table2 = ik_config["ik_match_table2"]
        self.human_root_name = ik_config["human_root_name"]
        self.robot_root_name = ik_config["robot_root_name"]
        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
        self.human_scale_table = ik_config["human_scale_table"]
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])

        self.max_iter = 10

        self.solver = solver
        self.damping = damping

        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}

        self.task_errors1 = {}
        self.task_errors2 = {}
        
        # Store verbose flag
        self.verbose = verbose
        
        # Flag to track if segment lengths have been computed
        self.segment_lengths_computed = False
        self.human_segment_lengths = {}
        self.robot_segment_lengths = {}

        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        if use_velocity_limit:
            VELOCITY_LIMITS = {k: 3*np.pi for k in self.robot_motor_names.keys()}
            self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS)) 
        # Add foot stick limit not used for now
        # self.foot_stick_limit = FootStickLimit(
        #     model=self.model,
        #     left_foot_frame_name="left_ankle_roll_link",
        #     right_foot_frame_name="right_ankle_roll_link",
        #     frame_type="body",
        #     tolerance=1e-5
        # )
        # self.ik_limits.append(self.foot_stick_limit)
        self.setup_retarget_configuration()
        
        self.ground_offset = 0.0
        self.init_root_pose = True  # Initialize root pose to target before IK solving

    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)
    
        self.tasks1 = []
        self.tasks2 = []
        
        for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task1[body_name] = task
                self.pos_offsets1[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets1[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks1.append(task)
                self.task_errors1[task] = []
        
        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task2[body_name] = task
                self.pos_offsets2[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets2[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks2.append(task)
                self.task_errors2[task] = []

  
    def compute_human_segment_lengths(self, human_data):
        """Compute actual segment lengths from human data (first frame)."""
        human_data = self.to_numpy(human_data)
        root_pos = human_data[self.human_root_name][0]
        
        segment_lengths = {}
        
        # Define segment pairs: (parent, child) -> segment_name
        # Based on SMPL-X joint hierarchy
        segment_pairs = {
            # Leg segments
            ("pelvis", "left_hip"): "left_thigh",
            ("left_hip", "left_knee"): "left_shank",
            ("left_knee", "left_foot"): "left_foot_segment",
            ("pelvis", "right_hip"): "right_thigh",
            ("right_hip", "right_knee"): "right_shank",
            ("right_knee", "right_foot"): "right_foot_segment",
            # Arm segments
            ("spine3", "left_shoulder"): "left_upper_arm",
            ("left_shoulder", "left_elbow"): "left_forearm",
            ("spine3", "right_shoulder"): "right_upper_arm",
            ("right_shoulder", "right_elbow"): "right_forearm",
            # Torso
            ("pelvis", "spine3"): "torso",
        }
        
        for (parent, child), seg_name in segment_pairs.items():
            if parent in human_data and child in human_data:
                parent_pos = human_data[parent][0]
                child_pos = human_data[child][0]
                length = np.linalg.norm(child_pos - parent_pos)
                segment_lengths[seg_name] = length
                # Also store for individual joints (used for scaling)
                if child not in segment_lengths:
                    segment_lengths[child] = length
        
        return segment_lengths
    
    def compute_robot_segment_lengths(self):
        """Compute actual segment lengths from robot model (zero pose)."""
        # Use mink.Configuration to get frame positions
        configuration = mink.Configuration(self.model)
        
        # Set to zero pose (all joints at 0)
        configuration.data.qpos[:] = 0.0
        mj.mj_forward(self.model, configuration.data)
        
        segment_lengths = {}
        
        # Get frame positions using mink FrameTask
        frame_positions = {}
        robot_joint_mapping = {}
        
        # Extract from ik_match_table1 to map human joints to robot frames
        for robot_frame, (human_body, _, _, _, _) in self.ik_match_table1.items():
            robot_joint_mapping[human_body] = robot_frame
            try:
                # Create a temporary FrameTask to get frame position
                # Try site first (common for foot frames)
                temp_task = mink.FrameTask(
                    frame_name=robot_frame,
                    frame_type="site",
                    position_cost=1.0,
                    orientation_cost=0.0,
                )
                # Set target to zero to get current frame position
                temp_task.set_target(mink.SE3.identity())
                # Get current frame transform (inverse of error gives current pose)
                # Actually, we need to get the frame's current SE3 transform
                # Use get_frame_jacobian to get frame info, or compute from error
                # For now, use a simpler approach: get from task's internal state
                # The error is target - current, so if target is identity, error = -current
                error_se3 = temp_task.compute_error(configuration)
                # Current position = -error (since target is identity)
                frame_positions[robot_frame] = -error_se3.translation()
            except:
                try:
                    # Try body type
                    temp_task = mink.FrameTask(
                        frame_name=robot_frame,
                        frame_type="body",
                        position_cost=1.0,
                        orientation_cost=0.0,
                    )
                    temp_task.set_target(mink.SE3.identity())
                    error_se3 = temp_task.compute_error(configuration)
                    frame_positions[robot_frame] = -error_se3.translation()
                except:
                    if self.verbose:
                        print(f"Warning: Could not get position for frame {robot_frame}")
        
        # Define segment pairs based on robot structure
        segment_pairs = {}
        
        # Left leg: pelvis -> left_hip -> left_knee -> left_foot
        if "left_hip" in robot_joint_mapping and "pelvis" in robot_joint_mapping:
            hip_frame = robot_joint_mapping["left_hip"]
            pelvis_frame = robot_joint_mapping["pelvis"]
            if hip_frame in frame_positions and pelvis_frame in frame_positions:
                segment_pairs["left_thigh"] = (pelvis_frame, hip_frame)
        
        if "left_knee" in robot_joint_mapping and "left_hip" in robot_joint_mapping:
            knee_frame = robot_joint_mapping["left_knee"]
            hip_frame = robot_joint_mapping["left_hip"]
            if knee_frame in frame_positions and hip_frame in frame_positions:
                segment_pairs["left_shank"] = (hip_frame, knee_frame)
        
        if "left_foot" in robot_joint_mapping and "left_knee" in robot_joint_mapping:
            foot_frame = robot_joint_mapping["left_foot"]
            knee_frame = robot_joint_mapping["left_knee"]
            if foot_frame in frame_positions and knee_frame in frame_positions:
                segment_pairs["left_foot_segment"] = (knee_frame, foot_frame)
        
        # Right leg
        if "right_hip" in robot_joint_mapping and "pelvis" in robot_joint_mapping:
            hip_frame = robot_joint_mapping["right_hip"]
            pelvis_frame = robot_joint_mapping["pelvis"]
            if hip_frame in frame_positions and pelvis_frame in frame_positions:
                segment_pairs["right_thigh"] = (pelvis_frame, hip_frame)
        
        if "right_knee" in robot_joint_mapping and "right_hip" in robot_joint_mapping:
            knee_frame = robot_joint_mapping["right_knee"]
            hip_frame = robot_joint_mapping["right_hip"]
            if knee_frame in frame_positions and hip_frame in frame_positions:
                segment_pairs["right_shank"] = (hip_frame, knee_frame)
        
        if "right_foot" in robot_joint_mapping and "right_knee" in robot_joint_mapping:
            foot_frame = robot_joint_mapping["right_foot"]
            knee_frame = robot_joint_mapping["right_knee"]
            if foot_frame in frame_positions and knee_frame in frame_positions:
                segment_pairs["right_foot_segment"] = (knee_frame, foot_frame)
        
        # Compute lengths
        for seg_name, (parent_frame, child_frame) in segment_pairs.items():
            if parent_frame in frame_positions and child_frame in frame_positions:
                length = np.linalg.norm(frame_positions[child_frame] - frame_positions[parent_frame])
                segment_lengths[seg_name] = length
        
        return segment_lengths
    
    def update_scale_table_from_segment_lengths(self, human_data):
        """Update human_scale_table based on actual segment length ratios."""
        if self.segment_lengths_computed:
            return  # Already computed
        
        # Compute segment lengths
        self.human_segment_lengths = self.compute_human_segment_lengths(human_data)
        self.robot_segment_lengths = self.compute_robot_segment_lengths()
        
        if self.verbose:
            print("\n[Segment Length Scaling]")
            print("Human segment lengths:", self.human_segment_lengths)
            print("Robot segment lengths:", self.robot_segment_lengths)
        
        # Map human joints to segments for scaling
        joint_to_segment = {
            "left_hip": "left_thigh",
            "left_knee": "left_shank",
            "left_foot": "left_foot_segment",
            "right_hip": "right_thigh",
            "right_knee": "right_shank",
            "right_foot": "right_foot_segment",
            "left_shoulder": "left_upper_arm",
            "left_elbow": "left_forearm",
            "right_shoulder": "right_upper_arm",
            "right_elbow": "right_forearm",
            "spine3": "torso",
        }
        
        # Update scale table based on segment length ratios
        for joint_name in self.human_scale_table.keys():
            if joint_name in joint_to_segment:
                seg_name = joint_to_segment[joint_name]
                if seg_name in self.human_segment_lengths and seg_name in self.robot_segment_lengths:
                    human_len = self.human_segment_lengths[seg_name]
                    robot_len = self.robot_segment_lengths[seg_name]
                    
                    if human_len > 1e-6:  # Avoid division by zero
                        # Compute ratio: robot_length / human_length
                        # Then multiply by original scale to get final scale
                        length_ratio = robot_len / human_len
                        original_scale = self.original_scale_table.get(joint_name, 1.0)
                        self.human_scale_table[joint_name] = original_scale * length_ratio
                        
                        if self.verbose:
                            print(f"  {joint_name}: human={human_len:.4f}m, robot={robot_len:.4f}m, "
                                  f"ratio={length_ratio:.4f}, scale={self.human_scale_table[joint_name]:.4f}")
        
        self.segment_lengths_computed = True
        
        if self.verbose:
            print("\nUpdated scale table:", self.human_scale_table)
  
    def update_targets(self, human_data, offset_to_ground=False):
        # If using segment-length-based scaling, compute and update scale table on first call
        if self.use_segment_length_scaling and not self.segment_lengths_computed:
            self.update_scale_table_from_segment_lengths(human_data)
        
        # scale human data in local frame
        human_data = self.to_numpy(human_data)
        human_data = self.scale_human_data(human_data, self.human_root_name, self.human_scale_table)
        human_data = self.offset_human_data(human_data, self.pos_offsets1, self.rot_offsets1)
        human_data = self.apply_ground_offset(human_data)
        if offset_to_ground:
            human_data = self.offset_human_data_to_ground(human_data)
        self.scaled_human_data = human_data

        if self.use_ik_match_table1:
            for body_name in self.human_body_to_task1.keys():
                task = self.human_body_to_task1[body_name]
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
        
        if self.use_ik_match_table2:
            for body_name in self.human_body_to_task2.keys():
                task = self.human_body_to_task2[body_name]
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
    
    def set_foot_sticking(self, foot_sticking):
        self.foot_stick_limit.set_state(
            left_stick=foot_sticking['left_foot'],
            right_stick=foot_sticking['right_foot'],
        )     
            
    def retarget(self, human_data, offset_to_ground=False):
        # Update the task targets
        self.update_targets(human_data, offset_to_ground)

        # Initialize root pose to target pose for better IK convergence
        # This is especially important when the initial pose is far from the target
        # (e.g., when target rotation is opposite to initial rotation)
        if self.init_root_pose and self.robot_root_name in self.scaled_human_data:
            root_pos, root_rot = self.scaled_human_data[self.robot_root_name]
            # Set root position (first 3 elements of qpos)
            self.configuration.data.qpos[:3] = root_pos
            # Set root rotation (next 4 elements: quaternion in w,x,y,z format for MuJoCo)
            # root_rot is already in scalar-first format (w,x,y,z) from update_targets
            # Normalize quaternion to ensure it's valid
            root_rot = np.asarray(root_rot)
            root_rot = root_rot / (np.linalg.norm(root_rot) + 1e-8)
            self.configuration.data.qpos[3:7] = root_rot
            # Forward kinematics to update the configuration
            mj.mj_forward(self.model, self.configuration.data)
            self.init_root_pose = False

        if self.use_ik_match_table1:
            # Solve the IK problem
            curr_error = self.error1()
            dt = self.configuration.model.opt.timestep
            vel1 = mink.solve_ik(
                self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits
            )
            self.configuration.integrate_inplace(vel1, dt)
            next_error = self.error1()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                dt = self.configuration.model.opt.timestep
                vel1 = mink.solve_ik(
                    self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits
                )
                self.configuration.integrate_inplace(vel1, dt)
                next_error = self.error1()
                num_iter += 1

        if self.use_ik_match_table2:
            curr_error = self.error2()
            dt = self.configuration.model.opt.timestep
            vel2 = mink.solve_ik(
                self.configuration, self.tasks2, dt, self.solver, self.damping, self.ik_limits
            )
            self.configuration.integrate_inplace(vel2, dt)
            next_error = self.error2()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                # Solve the IK problem with the second task
                dt = self.configuration.model.opt.timestep
                vel2 = mink.solve_ik(
                    self.configuration, self.tasks2, dt, self.solver, self.damping, self.ik_limits
                )
                self.configuration.integrate_inplace(vel2, dt)
                
                next_error = self.error2()
                num_iter += 1
                
            
        return self.configuration.data.qpos.copy()


    def error1(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks1]
            )
        )
    
    def error2(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks2]
            )
        )


    def to_numpy(self, human_data):
        for body_name in human_data.keys():
            human_data[body_name] = [np.asarray(human_data[body_name][0]), np.asarray(human_data[body_name][1])]
        return human_data


    def scale_human_data(self, human_data, human_root_name, human_scale_table):
        
        human_data_local = {}
        root_pos, root_quat = human_data[human_root_name]
        
        # scale root
        scaled_root_pos = human_scale_table[human_root_name] * root_pos
        
        # scale other body parts in local frame
        for body_name in human_data.keys():
            if body_name not in human_scale_table:
                continue
            if body_name == human_root_name:
                continue
            else:
                # transform to local frame (only position)
                human_data_local[body_name] = (human_data[body_name][0] - root_pos) * human_scale_table[body_name]
            
        # transform the human data back to the global frame
        human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data_local.keys():
            human_data_global[body_name] = (human_data_local[body_name] + scaled_root_pos, human_data[body_name][1])

        return human_data_global
    
    def offset_human_data(self, human_data, pos_offsets, rot_offsets):
        """the pos offsets are applied in the local frame"""
        offset_human_data = {}
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            
            # 只处理存在于 offsets 中的关节，如果不存在则直接使用原始数据
            if body_name in rot_offsets and body_name in pos_offsets:
                offset_human_data[body_name] = [pos, quat]
                # apply rotation offset first
                updated_quat = (R.from_quat(quat, scalar_first=True) * rot_offsets[body_name]).as_quat(scalar_first=True)
                offset_human_data[body_name][1] = updated_quat
                
                local_offset = pos_offsets[body_name]
                # compute the global position offset using the updated rotation
                global_pos_offset = R.from_quat(updated_quat, scalar_first=True).apply(local_offset)
                
                offset_human_data[body_name][0] = pos + global_pos_offset
            else:
                # 如果关节不在 offsets 中，直接使用原始数据
                offset_human_data[body_name] = [pos, quat]
            
        return offset_human_data
            
    def offset_human_data_to_ground(self, human_data):
        """find the lowest point of the human data and offset the human data to the ground"""
        offset_human_data = {}
        ground_offset = 0.1
        lowest_pos = np.inf

        for body_name in human_data.keys():
            # only consider the foot/Foot
            if "Foot" not in body_name and "foot" not in body_name:
                continue
            pos, quat = human_data[body_name]
            if pos[2] < lowest_pos:
                lowest_pos = pos[2]
                lowest_body_name = body_name
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            offset_human_data[body_name][0] = pos - np.array([0, 0, lowest_pos]) + np.array([0, 0, ground_offset])
        return offset_human_data

    def set_ground_offset(self, ground_offset):
        self.ground_offset = ground_offset

    def set_init_root_pose(self, init_root_pose: bool):
        """Set whether to initialize root pose to target before IK solving.
        
        Args:
            init_root_pose: If True, root position and rotation will be initialized
                           to target values before IK solving. This helps when
                           initial pose is far from target (e.g., opposite rotation).
        """
        self.init_root_pose = init_root_pose

    def apply_ground_offset(self, human_data):
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            human_data[body_name][0] = pos - np.array([0, 0, self.ground_offset])
        return human_data