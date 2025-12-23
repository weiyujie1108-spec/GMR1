from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .phuma_ground_utils import find_robust_ground_from_body_positions


def adjust_root_pos_to_ground(
    *,
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    dof_pos: np.ndarray,
    kinematics_model,
    ground_method: str = "phuma",
    ground_offset: float = 0.0,
    ground_thresh: float = 0.05,
    use_robust_statistics: bool = True,
    lift_epsilon: float = 0.004,
    foot_name_keywords: Tuple[str, ...] = ("foot", "toe", "ankle"),
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Ground adjustment helper used by dataset generation scripts.

    Supports:
    - "min": global minimum z across all bodies/frames
    - "phuma": PHUMA-style robust ground from foot bodies, then global align
    - "phuma+lift": PHUMA ground + per-frame lift for penetrated frames only

    Returns:
        root_pos_adjusted, debug_info
    """
    raw_method = (ground_method or "phuma").lower().strip()
    # Keep '+' for explicit combined modes, but normalize '_' / '-' to simplify CLI usage.
    method = raw_method.replace("_", "").replace("-", "")
    phuma_plus_lift = method in ("phuma+lift", "phumapluslift", "phumalift")
    phuma = method.startswith("phuma")
    min_method = method == "min"

    if not (phuma or min_method):
        raise ValueError(f"Unknown ground_method: {ground_method!r}. Use 'phuma', 'min', or 'phuma+lift'.")

    if root_pos is None or root_rot is None or dof_pos is None:
        raise ValueError("root_pos/root_rot/dof_pos must not be None")

    root_pos = np.asarray(root_pos, dtype=np.float64)
    root_rot = np.asarray(root_rot, dtype=np.float64)
    dof_pos = np.asarray(dof_pos, dtype=np.float64)
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_pos must be (T,3), got {root_pos.shape}")

    device = getattr(kinematics_model, "_device", "cuda:0")
    body_names: Optional[List[str]] = getattr(kinematics_model, "body_names", None)

    # FK before adjustment
    body_pos, _ = kinematics_model.forward_kinematics(
        torch.from_numpy(root_pos).to(device=device, dtype=torch.float),
        torch.from_numpy(root_rot).to(device=device, dtype=torch.float),
        torch.from_numpy(dof_pos).to(device=device, dtype=torch.float),
    )  # (T,N,3)

    lowest_height: float
    robust_ground: Optional[float] = None
    foot_body_indices: List[int] = []

    if phuma:
        if body_names is not None:
            for i, body_name in enumerate(body_names):
                n = str(body_name).lower()
                if any(k in n for k in foot_name_keywords):
                    foot_body_indices.append(i)

        if foot_body_indices:
            robust_ground = find_robust_ground_from_body_positions(
                body_pos,
                body_names=body_names,
                foot_body_indices=foot_body_indices,
                ground_thresh=ground_thresh,
                use_robust_statistics=use_robust_statistics,
            )
            lowest_height = float(robust_ground)
        else:
            lowest_height = float(torch.min(body_pos[..., 2]).item())
    else:
        lowest_height = float(torch.min(body_pos[..., 2]).item())

    root_pos_adjusted = root_pos.copy()
    root_pos_adjusted[:, 2] = root_pos_adjusted[:, 2] - lowest_height + float(ground_offset)

    per_frame_shift = None
    if phuma_plus_lift:
        # Safety margin to avoid tiny penetrations due to:
        # - body frame z not representing the lowest collision geometry point
        # - FK/float precision and coordinate mismatch vs downstream checks
        eps = float(lift_epsilon)
        body_pos_after, _ = kinematics_model.forward_kinematics(
            torch.from_numpy(root_pos_adjusted).to(device=device, dtype=torch.float),
            torch.from_numpy(root_rot).to(device=device, dtype=torch.float),
            torch.from_numpy(dof_pos).to(device=device, dtype=torch.float),
        )
        per_frame_min_z = torch.min(body_pos_after[..., 2], dim=1).values  # (T,)
        per_frame_shift_t = torch.clamp((float(ground_offset) + eps) - per_frame_min_z, min=0.0)
        if torch.any(per_frame_shift_t > 0):
            per_frame_shift = per_frame_shift_t.detach().cpu().numpy()
            root_pos_adjusted[:, 2] = root_pos_adjusted[:, 2] + per_frame_shift
        else:
            per_frame_shift = np.zeros((root_pos_adjusted.shape[0],), dtype=np.float64)

    debug = {
        "ground_method_normalized": method,
        "lowest_height": lowest_height,
        "robust_ground": robust_ground,
        "ground_offset": float(ground_offset),
        "ground_thresh": float(ground_thresh),
        "use_robust_statistics": bool(use_robust_statistics),
        "lift_epsilon": float(lift_epsilon),
        "num_foot_bodies": len(foot_body_indices),
        "per_frame_shift": per_frame_shift,
    }
    return root_pos_adjusted, debug


