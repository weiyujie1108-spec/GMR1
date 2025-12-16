import mujoco
import numpy as np
import mink
from mink.limits.limit import Limit, Constraint
from mink.configuration import Configuration
from mink.tasks.task import Task

import matplotlib.pyplot as plt


class InteractionMeshTask(Task):
    def __init__(
        self,
        target_laplacian: np.ndarray,
        adj_matrix: np.ndarray,
        frame_names: list[str],
        fixed_points: list[np.ndarray] | None = None,
        cost: np.ndarray | float = 1.0,
        gain: float = 1.0,
        lm_damping: float = 2.0,
        frame_type: str = "body",
    ):
        """
        Args:
            target_laplacian: (V, 3) target laplacian coordinates delta_target
            adj_matrix: (V, V) adjacency matrix or precomputed laplacian matrix L
            frame_names: list of frame names corresponding to the first N vertices of the robot
            fixed_points: list of fixed points coordinates for the last M vertices
            cost: task weight
            gain: task gain (0~1)
            lm_damping: damping coefficient
            frame_type: Frame type ("body", "site", "geom")
        """
        self.frame_names = frame_names
        self.fixed_points = fixed_points if fixed_points is not None else []
        self.num_dynamic = len(frame_names)
        self.num_fixed = len(self.fixed_points)
        self.V = self.num_dynamic + self.num_fixed
        self.frame_type = frame_type

        # precompute laplacian matrix L
        if np.allclose(np.sum(adj_matrix, axis=1), 0):
            self.L = adj_matrix
        else:
            degree = np.sum(adj_matrix, axis=1)
            D = np.diag(degree)
            self.L = D - adj_matrix

        # build Kronecker product form of L_kron (3V, 3V)
        self.L_kron = np.kron(self.L, np.eye(3))

        # store target and flatten it to (3V,)
        self.target_lap_vec = target_laplacian.reshape(-1)

        if isinstance(cost, float):
            cost_vec = np.ones(3 * self.V) * cost
        else:
            cost_vec = cost

        super().__init__(cost=cost_vec, gain=gain, lm_damping=lm_damping)

    def set_laplacian_target(self, target_laplacian: np.ndarray):
        self.target_lap_vec = target_laplacian.reshape(-1)

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        curr_pts = [] # (keypts, 3)

        # Dynamic points from frames
        for frame in self.frame_names:
            transform = configuration.get_transform_frame_to_world(frame, self.frame_type)
            curr_pts.append(transform.translation())

        # Fixed points
        for pt in self.fixed_points:
            curr_pts.append(pt)

        curr_pts_flat = np.hstack(curr_pts) # (3V, )
        curr_lap_vec = self.L_kron @ curr_pts_flat

        return curr_lap_vec - self.target_lap_vec

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        J_list = []
        for name in self.frame_names:
            J_body = configuration.get_frame_jacobian(name, self.frame_type)
            transform = configuration.get_transform_frame_to_world(name, self.frame_type)
            R_wf = transform.rotation().as_matrix()
            J_linear_body = J_body[:3, :]
            J_linear_world = R_wf @ J_linear_body
            J_list.append(J_linear_world)

        # Fixed points have zero Jacobian
        if self.num_fixed > 0:
            nv = configuration.model.nv
            J_fixed = np.zeros((3 * self.num_fixed, nv))
            J_list.append(J_fixed)

        J_stacked = np.vstack(J_list) # (3V, nv)

        # Chain rule: J_task = d(Lp)/dq = L * dp/dq = L * J_stacked
        J_task = self.L_kron @ J_stacked # (3V, 3V) @ (3V, nv) -> (3V, nv)

        return J_task


class FootStickLimit(Limit):
    """
    Constrains specified foot frames to have zero velocity (stick to ground).

    Implements inequality constraints:
        J * dq <= tolerance
       -J * dq <= tolerance
    which effectively forces |v| <= tolerance.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        left_foot_frame_name: str,
        right_foot_frame_name: str,
        frame_type: str = "site",
        tolerance: float = 1e-4,
    ):
        """Initialize foot stick limits.

        Args:
            model: MuJoCo model (used for validation).
            left_foot_frame_name: Name of the left foot frame.
            right_foot_frame_name: Name of the right foot frame.
            frame_type: Type of the frame ('site', 'body', 'geom').
            tolerance: Maximum allowed velocity magnitude (effectively zero).
        """
        self.left_name = left_foot_frame_name
        self.right_name = right_foot_frame_name
        self.frame_type = frame_type
        self.tolerance = tolerance

        # State flags
        self.left_stick = False
        self.right_stick = False

        # Validation: Check if frames exist in the model to fail fast
        self._validate_frames(model)

        # Pre-allocate constant tolerance vectors for efficiency.
        # A spatial frame has 6 DoF (3 pos + 3 rot).
        # For one foot, we need constraints for +J and -J, so 12 rows.
        self._h_single_foot = np.full(12, tolerance)

        # Pre-allocate combined h if both feet are stuck (24 rows)
        self._h_both_feet = np.full(24, tolerance)

        print("feet limit initialized")

    def _validate_frames(self, model: mujoco.MjModel):
        """Validate that the specified frames exist in the model."""
        obj_type = getattr(mujoco.mjtObj, f"mjOBJ_{self.frame_type.upper()}")
        if mujoco.mj_name2id(model, obj_type, self.left_name) < 0:
            raise ValueError(
                f"Frame '{self.left_name}' of type '{self.frame_type}' not found."
            )
        if mujoco.mj_name2id(model, obj_type, self.right_name) < 0:
            raise ValueError(
                f"Frame '{self.right_name}' of type '{self.frame_type}' not found."
            )

    def compute_qp_inequalities(
        self,
        configuration: mink.Configuration,
        dt: float,
    ) -> Constraint:
        """Compute the velocity constraints for stuck feet."""
        del dt  # Unused for pure velocity constraint (unless we add drift compensation)

        # Early exit if nothing is sticking
        if not (self.left_stick or self.right_stick):
            return Constraint()

        G_list = []
        h_use = None

        # Logic:
        # We want J * dq = 0.
        # In QP (G * dq <= h), we express this as:
        #  J * dq <= tol
        # -J * dq <= tol

        # Case 1: Both feet sticking (Optimized path for common case)
        if self.left_stick and self.right_stick:
            J_left = configuration.get_frame_jacobian(self.left_name, self.frame_type)
            J_right = configuration.get_frame_jacobian(self.right_name, self.frame_type)

            # Stack layout:
            # [ J_left  ]
            # [ -J_left ]
            # [ J_right ]
            # [ -J_right]
            G = np.vstack([J_left, -J_left, J_right, -J_right])
            return Constraint(G=G, h=self._h_both_feet)

        # Case 2: Only one foot sticking
        target_name = self.left_name if self.left_stick else self.right_name
        J = configuration.get_frame_jacobian(target_name, self.frame_type)

        G = np.vstack([J, -J])
        return Constraint(G=G, h=self._h_single_foot)

    def set_state(self, left_stick: bool, right_stick: bool):
        """Update the stick state."""
        self.left_stick = left_stick
        self.right_stick = right_stick


def extract_foot_sticking_from_trajectories(
    left_foot_pos,
    right_foot_pos,
    contact_threshold=0.01,
    velocity_threshold=0.1,
    fps=30.0,
):
    """
    Extract contact sequence from foot trajectories.

    Args:
        left_foot_pos (np.ndarray): Left foot positions (N, 3).
        right_foot_pos (np.ndarray): Right foot positions (N, 3).
        contact_threshold (float): Height threshold relative to min z.
        velocity_threshold (float): Velocity threshold.
        fps (float): Frame rate.

    Returns:
        list: List of contact dictionaries for each frame.
    """
    z_L_min = left_foot_pos[:, 2].min()
    z_R_min = right_foot_pos[:, 2].min()

    dt = 1.0 / fps

    left_foot_vel = np.diff(left_foot_pos, axis=0, prepend=left_foot_pos[0:1]) / dt
    right_foot_vel = np.diff(right_foot_pos, axis=0, prepend=right_foot_pos[0:1]) / dt

    left_foot_vel_xy = np.linalg.norm(left_foot_vel[:, :2], axis=1)
    right_foot_vel_xy = np.linalg.norm(right_foot_vel[:, :2], axis=1)

    return [
        {
            "left_foot": (left_foot_pos[i, 2] <= z_L_min + contact_threshold)
            and (left_foot_vel_xy[i] < velocity_threshold),
            "right_foot": (right_foot_pos[i, 2] <= z_R_min + contact_threshold)
            and (right_foot_vel_xy[i] < velocity_threshold),
            # "left_foot": (left_foot_vel_xy[i] < velocity_threshold),
            # "right_foot": (right_foot_vel_xy[i] < velocity_threshold),
        }
        for i in range(len(left_foot_pos))
    ]


def extract_foot_sticking_sequence(
    smpl_joints,
    demo_joints,
    foot_names,
    smpl_contact_threshold_relative=0.01,
    velocity_threshold=0.1,
    fps=30.0,
):
    """
    Wrapper for backward compatibility or ease of use with SMPL data.
    """
    left_foot_idx = demo_joints.index(foot_names[0])
    right_foot_idx = demo_joints.index(foot_names[1])

    left_foot_pos = smpl_joints[:, left_foot_idx]
    right_foot_pos = smpl_joints[:, right_foot_idx]

    return extract_foot_sticking_from_trajectories(
        left_foot_pos,
        right_foot_pos,
        contact_threshold=smpl_contact_threshold_relative,
        velocity_threshold=velocity_threshold,
        fps=fps,
    )


def analyze_and_plot_robot_motion(
    robot_left_foot_pos,
    robot_right_foot_pos,
    qpos_list,
    retarget,
    aligned_fps,
    contact_threshold=0.01,
    velocity_threshold=0.1,
    human_left_foot_stick=None,
    human_right_foot_stick=None,
    save_path_prefix="",
):
    """
    Analyze and plot robot motion (foot sticking and joint analysis).
    """
    # Robot Foot Analysis
    robot_foot_sticking_sequence = extract_foot_sticking_from_trajectories(
        robot_left_foot_pos,
        robot_right_foot_pos,
        contact_threshold=contact_threshold,
        velocity_threshold=velocity_threshold,
        fps=aligned_fps,
    )

    robot_left_foot_stick = [
        frame["left_foot"] for frame in robot_foot_sticking_sequence
    ]
    robot_right_foot_stick = [
        frame["right_foot"] for frame in robot_foot_sticking_sequence
    ]

    dt = 1.0 / aligned_fps
    robot_left_foot_vel = (
        np.diff(robot_left_foot_pos, axis=0, prepend=robot_left_foot_pos[0:1]) / dt
    )
    robot_right_foot_vel = (
        np.diff(robot_right_foot_pos, axis=0, prepend=robot_right_foot_pos[0:1]) / dt
    )

    robot_left_foot_vel_xy = np.linalg.norm(robot_left_foot_vel[:, :2], axis=1)
    robot_right_foot_vel_xy = np.linalg.norm(robot_right_foot_vel[:, :2], axis=1)

    robot_z_L_min = robot_left_foot_pos[:, 2].min()
    robot_z_R_min = robot_right_foot_pos[:, 2].min()

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Sticking
    axes[0].plot(robot_left_foot_stick, label="Left Stick", alpha=0.7)
    axes[0].fill_between(
        range(len(robot_left_foot_stick)),
        robot_left_foot_stick,
        alpha=0.3,
        color="blue",
    )

    axes[0].plot(robot_right_foot_stick, label="Right Stick", alpha=0.7)
    axes[0].fill_between(
        range(len(robot_right_foot_stick)),
        robot_right_foot_stick,
        alpha=0.3,
        color="orange",
    )

    axes[0].set_title("Robot Foot Sticking Sequence")
    axes[0].legend()

    # Position (Z only)
    axes[1].plot(robot_left_foot_pos[:, 2], label="Left Z")
    axes[1].plot(robot_right_foot_pos[:, 2], label="Right Z")
    axes[1].axhline(
        y=robot_z_L_min + contact_threshold,
        color="blue",
        linestyle="--",
        alpha=0.5,
        label="L Threshold",
    )
    axes[1].axhline(
        y=robot_z_R_min + contact_threshold,
        color="orange",
        linestyle="--",
        alpha=0.5,
        label="R Threshold",
    )

    # Add shading for robot sticking regions
    axes[1].fill_between(
        range(len(robot_left_foot_stick)),
        0,
        1,
        where=robot_left_foot_stick,
        transform=axes[1].get_xaxis_transform(),
        alpha=0.1,
        color="blue",
        label="Robot Left Stick",
    )
    axes[1].fill_between(
        range(len(robot_right_foot_stick)),
        0,
        1,
        where=robot_right_foot_stick,
        transform=axes[1].get_xaxis_transform(),
        alpha=0.1,
        color="orange",
        label="Robot Right Stick",
    )

    # Overlay Human Sticking on Robot Plots if provided
    if human_left_foot_stick is not None:
        axes[1].fill_between(
            range(len(human_left_foot_stick)),
            0,
            1,
            where=human_left_foot_stick,
            transform=axes[1].get_xaxis_transform(),
            alpha=0.1,
            color="green",
            label="Human Left Stick",
        )
    if human_right_foot_stick is not None:
        axes[1].fill_between(
            range(len(human_right_foot_stick)),
            0,
            1,
            where=human_right_foot_stick,
            transform=axes[1].get_xaxis_transform(),
            alpha=0.1,
            color="lime",
            label="Human Right Stick",
        )

    axes[1].set_title("Robot Foot Z Position (with Human Sticking Overlay)")
    axes[1].legend()

    # Velocity (XY Magnitude)
    axes[2].plot(robot_left_foot_vel_xy, label="Left XY Vel")
    axes[2].plot(robot_right_foot_vel_xy, label="Right XY Vel")
    axes[2].axhline(
        y=velocity_threshold,
        color="red",
        linestyle="--",
        alpha=0.5,
        label="Vel Threshold",
    )

    # Add shading for robot sticking regions
    axes[2].fill_between(
        range(len(robot_left_foot_stick)),
        0,
        1,
        where=robot_left_foot_stick,
        transform=axes[2].get_xaxis_transform(),
        alpha=0.1,
        color="blue",
    )
    axes[2].fill_between(
        range(len(robot_right_foot_stick)),
        0,
        1,
        where=robot_right_foot_stick,
        transform=axes[2].get_xaxis_transform(),
        alpha=0.1,
        color="orange",
    )

    # Overlay Human Sticking on Robot Plots if provided
    if human_left_foot_stick is not None:
        axes[2].fill_between(
            range(len(human_left_foot_stick)),
            0,
            1,
            where=human_left_foot_stick,
            transform=axes[2].get_xaxis_transform(),
            alpha=0.1,
            color="green",
            label="Human Left Stick",
        )
    if human_right_foot_stick is not None:
        axes[2].fill_between(
            range(len(human_right_foot_stick)),
            0,
            1,
            where=human_right_foot_stick,
            transform=axes[2].get_xaxis_transform(),
            alpha=0.1,
            color="lime",
            label="Human Right Stick",
        )

    axes[2].set_title("Robot Foot XY Velocity Magnitude (with Human Sticking Overlay)")
    axes[2].legend()

    plt.tight_layout()
    save_name = f"{save_path_prefix}robot_foot_analysis.png"
    plt.savefig(save_name)
    print(f"Saved visualization to {save_name}")

    # Robot Joint Analysis
    dof_pos = np.array([qpos[7:] for qpos in qpos_list])
    dof_vel = np.diff(dof_pos, axis=0, prepend=dof_pos[0:1]) / dt

    # Get joint names sorted by index
    dof_names = [
        name
        for name, idx in sorted(
            retarget.robot_dof_names.items(), key=lambda item: item[1]
        )
    ]
    # Filter dof_names to match the size of dof_pos (excluding root)
    dof_names = dof_names[-dof_pos.shape[1] :]
    num_dofs = len(dof_names)

    fig_joints, axes_joints = plt.subplots(
        num_dofs, 2, figsize=(12, 2 * num_dofs), sharex=True
    )

    for i in range(num_dofs):
        # Position
        axes_joints[i, 0].plot(dof_pos[:, i])
        axes_joints[i, 0].set_ylabel(dof_names[i], rotation=0, labelpad=20, fontsize=8)
        axes_joints[i, 0].grid(True, alpha=0.3)

        # Velocity
        axes_joints[i, 1].plot(dof_vel[:, i])
        axes_joints[i, 1].grid(True, alpha=0.3)

        if i == 0:
            axes_joints[i, 0].set_title("Position")
            axes_joints[i, 1].set_title("Velocity")

    plt.tight_layout()
    save_name_joints = f"{save_path_prefix}robot_joint_analysis.png"
    plt.savefig(save_name_joints)
    print(f"Saved visualization to {save_name_joints}")


def analyze_and_plot_human_motion(
    foot_sticking_sequence,
    smplx_data_joints,
    joint_names,
    aligned_fps,
    contact_threshold=0.01,
    velocity_threshold=0.1,
    save_path_prefix="",
):
    """
    Analyze and plot human motion (foot sticking).
    """
    left_foot_stick = [frame["left_foot"] for frame in foot_sticking_sequence]
    right_foot_stick = [frame["right_foot"] for frame in foot_sticking_sequence]

    # Get indices
    left_foot_idx = joint_names.index("left_foot")
    right_foot_idx = joint_names.index("right_foot")

    # Extract positions (frames, 3)
    left_foot_pos = smplx_data_joints[:, left_foot_idx]
    right_foot_pos = smplx_data_joints[:, right_foot_idx]

    # Calculate velocities
    dt = 1.0 / aligned_fps
    left_foot_vel = np.diff(left_foot_pos, axis=0, prepend=left_foot_pos[0:1]) / dt
    right_foot_vel = np.diff(right_foot_pos, axis=0, prepend=right_foot_pos[0:1]) / dt

    left_foot_vel_xy = np.linalg.norm(left_foot_vel[:, :2], axis=1)
    right_foot_vel_xy = np.linalg.norm(right_foot_vel[:, :2], axis=1)

    # Calculate Z thresholds for plotting
    z_L_min = smplx_data_joints[:, left_foot_idx, 2].min()
    z_R_min = smplx_data_joints[:, right_foot_idx, 2].min()

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Sticking
    axes[0].plot(left_foot_stick, label="Left Stick", alpha=0.7)
    axes[0].fill_between(
        range(len(left_foot_stick)), left_foot_stick, alpha=0.3, color="blue"
    )

    axes[0].plot(right_foot_stick, label="Right Stick", alpha=0.7)
    axes[0].fill_between(
        range(len(right_foot_stick)), right_foot_stick, alpha=0.3, color="orange"
    )

    axes[0].set_title("Foot Sticking Sequence")
    axes[0].legend()

    # Position (Z only)
    axes[1].plot(left_foot_pos[:, 2], label="Left Z")
    axes[1].plot(right_foot_pos[:, 2], label="Right Z")
    axes[1].axhline(
        y=z_L_min + contact_threshold,
        color="blue",
        linestyle="--",
        alpha=0.5,
        label="L Threshold",
    )
    axes[1].axhline(
        y=z_R_min + contact_threshold,
        color="orange",
        linestyle="--",
        alpha=0.5,
        label="R Threshold",
    )

    # Add shading for sticking regions
    axes[1].fill_between(
        range(len(left_foot_stick)),
        0,
        1,
        where=left_foot_stick,
        transform=axes[1].get_xaxis_transform(),
        alpha=0.1,
        color="blue",
    )
    axes[1].fill_between(
        range(len(right_foot_stick)),
        0,
        1,
        where=right_foot_stick,
        transform=axes[1].get_xaxis_transform(),
        alpha=0.1,
        color="orange",
    )

    axes[1].set_title("Foot Z Position")
    axes[1].legend()

    # Velocity (XY Magnitude)
    axes[2].plot(left_foot_vel_xy, label="Left XY Vel")
    axes[2].plot(right_foot_vel_xy, label="Right XY Vel")
    axes[2].axhline(
        y=velocity_threshold,
        color="red",
        linestyle="--",
        alpha=0.5,
        label="Vel Threshold",
    )

    # Add shading for sticking regions
    axes[2].fill_between(
        range(len(left_foot_stick)),
        0,
        1,
        where=left_foot_stick,
        transform=axes[2].get_xaxis_transform(),
        alpha=0.1,
        color="blue",
    )
    axes[2].fill_between(
        range(len(right_foot_stick)),
        0,
        1,
        where=right_foot_stick,
        transform=axes[2].get_xaxis_transform(),
        alpha=0.1,
        color="orange",
    )

    axes[2].set_title("Foot XY Velocity Magnitude")
    axes[2].legend()

    plt.tight_layout()
    save_name = f"{save_path_prefix}foot_analysis.png"
    plt.savefig(save_name)
    print(f"Saved visualization to {save_name}")


def fix_feet_sliding(
    qpos_list,
    robot_left_foot_pos_list,
    robot_right_foot_pos_list,
    foot_sticking_sequence,
):
    """
    Post-process qpos to fix feet sliding.

    Args:
        qpos_list: List of qpos (numpy arrays).
        robot_left_foot_pos_list: List of left foot positions (numpy arrays).
        robot_right_foot_pos_list: List of right foot positions (numpy arrays).
        foot_sticking_sequence: List of dicts with 'left_foot' and 'right_foot' booleans.

    Returns:
        fixed_qpos_list: List of fixed qpos.
        fixed_robot_left_foot_pos_list: List of fixed left foot positions.
        fixed_robot_right_foot_pos_list: List of fixed right foot positions.
    """
    fixed_qpos_list = [qpos.copy() for qpos in qpos_list]
    fixed_robot_left_foot_pos_list = [pos.copy() for pos in robot_left_foot_pos_list]
    fixed_robot_right_foot_pos_list = [pos.copy() for pos in robot_right_foot_pos_list]

    accumulated_delta = np.zeros(2)

    # Iterate from the second frame
    for i in range(1, len(qpos_list)):
        left_contact = foot_sticking_sequence[i]["left_foot"]
        right_contact = foot_sticking_sequence[i]["right_foot"]

        prev_left_contact = foot_sticking_sequence[i - 1]["left_foot"]
        prev_right_contact = foot_sticking_sequence[i - 1]["right_foot"]

        is_both_feet_contact = left_contact and right_contact

        if (left_contact or right_contact) and not is_both_feet_contact:
            prev_left_pos = robot_left_foot_pos_list[i - 1]
            prev_right_pos = robot_right_foot_pos_list[i - 1]

            curr_left_pos = robot_left_foot_pos_list[i]
            curr_right_pos = robot_right_foot_pos_list[i]

            if left_contact and not right_contact:
                if prev_left_contact:
                    # Calculate delta based on what the foot MOVED, to counteract it.
                    # If foot moved +dx, we want to move root -dx so foot stays put?
                    # Wait, if foot moved +dx in world, it slid.
                    # We want foot to be at prev_pos.
                    # So we need to shift EVERYTHING by (prev_pos - curr_pos).
                    delta_xy = prev_left_pos[:2] - curr_left_pos[:2]
                    accumulated_delta += delta_xy
            elif right_contact and not left_contact:
                if prev_right_contact:
                    delta_xy = prev_right_pos[:2] - curr_right_pos[:2]
                    accumulated_delta += delta_xy

        # Apply accumulated delta to root position (indices 0 and 1 for x and y)
        fixed_qpos_list[i][:2] += accumulated_delta

        # Also update foot positions for verification/plotting purposes
        fixed_robot_left_foot_pos_list[i][:2] += accumulated_delta
        fixed_robot_right_foot_pos_list[i][:2] += accumulated_delta

    return (
        fixed_qpos_list,
        fixed_robot_left_foot_pos_list,
        fixed_robot_right_foot_pos_list,
    )