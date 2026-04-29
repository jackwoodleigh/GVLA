"""
normalization_utils.py

Canonical home for all state/action space conversions used in the Franka deployment pipeline.

Three distinct spaces to keep straight:
  1. Raw robot space   — physical units from the Franka driver
                         EEF pos in meters (robot base frame), quaternion orientation,
                         gripper joint positions in meters
  2. LIBERO model space — representation the VLA was trained on
                          EEF pos in meters (LIBERO world frame), axis-angle orientation,
                          2D gripper fingers in meters; all normalized to [-1, 1] via q01/q99
  3. LIBERO OSC space  — unnormalized action commands output by the model
                          delta EEF pos (OSC units), delta rotation (OSC units), gripper 0/1
                          OSC units are scaled internally by the sim controller before moving the robot

Conversion chain for inference:
    raw robot state
        → build_libero_proprio()       # frame shift + axis-angle + gripper format
        → normalize_proprio()          # q01/q99 → [-1, 1]
        → model
        → unnormalize_action()         # [-1, 1] → OSC units
        → scale_action_for_robot()     # OSC units → real robot delta commands (requires calibration)
"""

import math
from typing import Any, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Constants derived from LIBERO source
# ---------------------------------------------------------------------------

# Offset to convert Franka EEF position (robot base frame) into LIBERO sim world frame.
#   x: MountedPanda.base_xpos_offset["table"](table_length=0.8) = -0.16 - 0.8/2 = -0.56
#   y: 0.0 (robot centered on table y-axis)
#   z: TableArena.table_offset z (0.8) + RethinkMount.top_offset z (-0.01) = 0.79
LIBERO_EEF_FRAME_OFFSET = np.array([-0.56, 0.0, 0.79], dtype=np.float64)

# Default OSC controller translation gain in robosuite / LIBERO sim.
# Maps OSC action units → approximate EEF displacement in meters per control step.
# This is a SIM-ONLY constant. Real robot deployment requires empirical calibration.
LIBERO_OSC_TRANSLATION_SCALE = 0.05   # meters per OSC unit (robosuite default)
LIBERO_OSC_ROTATION_SCALE    = 0.15   # radians per OSC unit (robosuite default, approximate)


# ---------------------------------------------------------------------------
# State conversion: raw Franka → LIBERO proprio
# ---------------------------------------------------------------------------

def quat_xyzw_to_axisangle(quat: np.ndarray) -> np.ndarray:
    """
    Convert a (x, y, z, w) unit quaternion to an axis-angle vector.

    Matches the convention used in LIBERO training data exactly.
    Returns a zero vector for near-identity rotations.
    """
    quat = quat.copy().astype(np.float64)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(max(0.0, 1.0 - quat[3] ** 2))
    if math.isclose(den, 0.0, abs_tol=1e-9):
        return np.zeros(3, dtype=np.float64)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def build_libero_proprio(
    eef_pos_robot_frame: np.ndarray,
    eef_quat_xyzw: np.ndarray,
    gripper_qpos: float,
) -> np.ndarray:
    """
    Build the 8-dim proprio vector expected by the LIBERO-trained VLA.

    Layout: [eef_pos(3) | axis_angle(3) | finger1(1) | finger2(1)]

    Args:
        eef_pos_robot_frame : (3,) EEF position in robot base frame, meters
        eef_quat_xyzw       : (4,) EEF orientation as (x, y, z, w) quaternion
        gripper_qpos        : scalar in [0, 1] — 0 = fully closed, 1 = fully open
                              (matches Franka gripper_qpos convention)

    Returns:
        (8,) float64 proprio in LIBERO world frame, raw (not yet normalized)
    """
    eef_pos   = eef_pos_robot_frame + LIBERO_EEF_FRAME_OFFSET
    axis_angle = quat_xyzw_to_axisangle(eef_quat_xyzw)

    # LIBERO gripper state = two opposing finger positions in meters
    # Franka finger range: ~0.002 m (closed) to 0.040 m (open) per finger
    finger1 =  0.002 + gripper_qpos * 0.038   # positive, grows with opening
    finger2 = -(0.002 + gripper_qpos * 0.038)  # negative mirror

    return np.concatenate([eef_pos, axis_angle, [finger1, finger2]]).astype(np.float64)


# ---------------------------------------------------------------------------
# Normalization: raw values → [-1, 1] for model input
# ---------------------------------------------------------------------------

def normalize_proprio(proprio: np.ndarray, norm_stats: Dict[str, Any]) -> np.ndarray:
    """
    Normalize a raw LIBERO-format proprio vector to [-1, 1] using q01/q99 stats.

    Args:
        proprio    : (8,) raw proprio from build_libero_proprio()
        norm_stats : dict with keys 'q01', 'q99' (and optionally 'mask')
                     from dataset_statistics.json["<unnorm_key>"]["proprio"]

    Returns:
        (8,) clipped to [-1, 1]
    """
    q01  = np.array(norm_stats["q01"], dtype=np.float64)
    q99  = np.array(norm_stats["q99"], dtype=np.float64)
    mask = np.array(norm_stats.get("mask", np.ones(len(q01), dtype=bool)), dtype=bool)

    normalized = np.where(
        mask,
        2.0 * (proprio - q01) / (q99 - q01 + 1e-8) - 1.0,
        proprio,
    )
    return np.clip(normalized, -1.0, 1.0).astype(np.float64)


def unnormalize_action(action_normalized: np.ndarray, norm_stats: Dict[str, Any]) -> np.ndarray:
    """
    Map a normalized model action output back to LIBERO OSC space.

    The last dim (gripper) is masked — it is kept as-is (model outputs 0 or 1 directly).

    Args:
        action_normalized : (7,) model output in [-1, 1] per dim (except gripper)
        norm_stats        : dict with keys 'q01', 'q99', 'mask'
                            from dataset_statistics.json["<unnorm_key>"]["action"]

    Returns:
        (7,) action in LIBERO OSC units: [dx, dy, dz, droll, dpitch, dyaw, gripper]
    """
    q01  = np.array(norm_stats["q01"], dtype=np.float64)
    q99  = np.array(norm_stats["q99"], dtype=np.float64)
    mask = np.array(norm_stats.get("mask", np.ones(len(q01), dtype=bool)), dtype=bool)

    return np.where(
        mask,
        (action_normalized + 1.0) / 2.0 * (q99 - q01) + q01,
        action_normalized,
    ).astype(np.float64)


# ---------------------------------------------------------------------------
# Gripper convention
# ---------------------------------------------------------------------------

def flip_gripper(action: np.ndarray) -> np.ndarray:
    """
    Flip gripper convention: model outputs 0=open, 1=closed; robot expects 0=closed, 1=open.
    Operates on the last dim of the action vector. Keeps values in [0, 1].
    """
    action = action.copy()
    action[..., -1] = 1.0 - action[..., -1]
    return action


# ---------------------------------------------------------------------------
# OSC → meters (sim reference, not for real deployment)
# ---------------------------------------------------------------------------

def libero_osc_to_meters(
    action_osc: np.ndarray,
    translation_scale: float = LIBERO_OSC_TRANSLATION_SCALE,
    rotation_scale: float = LIBERO_OSC_ROTATION_SCALE,
) -> np.ndarray:
    """
    Convert a LIBERO OSC action to approximate physical displacements.

    THIS IS A SIM-ONLY APPROXIMATION based on robosuite's default OSC controller
    gains. Do not use this for real robot commands without calibration.

    Args:
        action_osc        : (7,) action in OSC units [dx,dy,dz,droll,dpitch,dyaw,gripper]
        translation_scale : meters per OSC unit (robosuite default: 0.05)
        rotation_scale    : radians per OSC unit (robosuite default: ~0.15)

    Returns:
        (7,) [dx_m, dy_m, dz_m, droll_rad, dpitch_rad, dyaw_rad, gripper]
    """
    scale = np.array([
        translation_scale, translation_scale, translation_scale,
        rotation_scale,    rotation_scale,    rotation_scale,
        1.0,   # gripper passed through unchanged
    ], dtype=np.float64)
    return action_osc * scale


# ---------------------------------------------------------------------------
# OSC → real robot commands (requires calibration)
# ---------------------------------------------------------------------------

def scale_action_for_robot(
    action_osc: np.ndarray,
    translation_gain: float,
    rotation_gain: float,
    gripper_open_threshold: float = 0.5,
) -> np.ndarray:
    """
    Scale LIBERO OSC action commands for real Franka deployment.

    The gains here are NOT the same as the sim OSC controller gains — they must
    be determined empirically by moving the real robot and measuring EEF displacement.
    Start conservatively (translation_gain ~ 0.01) and increase until response matches.

    Args:
        action_osc            : (7,) unnormalized model output [dx,dy,dz,droll,dpitch,dyaw,gripper]
        translation_gain      : meters per OSC unit on the real robot (NEEDS CALIBRATION)
        rotation_gain         : radians per OSC unit on the real robot (NEEDS CALIBRATION)
        gripper_open_threshold: model gripper output above this → open command

    Returns:
        dict with keys:
            'delta_pos'   : (3,) EEF position delta in meters (robot base frame)
            'delta_rot'   : (3,) EEF rotation delta in radians (axis-angle)
            'gripper_open': bool — True = open, False = close
    """
    delta_pos = action_osc[:3] * translation_gain
    delta_rot = action_osc[3:6] * rotation_gain
    gripper_open = bool(action_osc[6] > gripper_open_threshold)

    return {
        "delta_pos":    delta_pos,
        "delta_rot":    delta_rot,
        "gripper_open": gripper_open,
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def check_proprio_coverage(
    proprio_raw: np.ndarray,
    norm_stats: Dict[str, Any],
    label: str = "proprio",
) -> None:
    """
    Print how many proprio dims fall outside the training q01/q99 range.
    Useful for catching frame/convention mismatches before they silently degrade policy.
    """
    q01 = np.array(norm_stats["q01"], dtype=np.float64)
    q99 = np.array(norm_stats["q99"], dtype=np.float64)
    raw_norm = 2.0 * (proprio_raw - q01) / (q99 - q01 + 1e-8) - 1.0
    n_saturated = int(np.sum(np.abs(raw_norm) > 1.0))

    dim_names = ["eef_x", "eef_y", "eef_z", "aa_x", "aa_y", "aa_z", "finger1", "finger2"]
    print(f"[{label}] {n_saturated}/{len(proprio_raw)} dims outside training range:")
    for i, (name, val, norm) in enumerate(zip(dim_names, proprio_raw, raw_norm)):
        flag = " ← SATURATED" if abs(norm) > 1.0 else ""
        print(f"  dim{i} {name:8s}  raw={val:+.4f}  normalized={norm:+.4f}{flag}")
