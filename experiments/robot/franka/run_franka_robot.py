"""
run_franka_robot.py

Closed-loop inference on the real Franka Panda robot.
Runs entirely locally (no server needed) — load the model on the lab GPU,
pull live camera + proprioception, push EEF delta actions.

Usage:
    # Zero-shot with pre-trained GVLA checkpoint (run prepare_gvla_checkpoint.py first):
    python experiments/robot/franka/run_franka_robot.py --use_vggt

    # Baseline VLA-Adapter (LIBERO-Spatial-Pro, no finetuning):
    python experiments/robot/franka/run_franka_robot.py \
        --pretrained_checkpoint output/LIBERO-Spatial-Pro

    # After finetuning:
    python experiments/robot/franka/run_franka_robot.py \
        --pretrained_checkpoint output/LIBERO-Spatial-Pro-FrankaFT

    python experiments/robot/franka/run_franka_robot.py \
        --pretrained_checkpoint output/FINAL-GVLA-v2-FrankaFT --use_vggt

Observations the robot must provide (fill in TODO sections below):
    full_image  : np.ndarray [H, W, 3] uint8  — scene camera (cam3)
    wrist_image : np.ndarray [H, W, 3] uint8  — wrist camera (cam4)
    state       : np.ndarray [8] float64      — raw EEF pos(3)+quat(4) + gripper(1)
                  → build with build_libero_proprio() from normalization_utils.py

Actions the robot receives:
    7-dim numpy array: [dx, dy, dz, droll, dpitch, dyaw, gripper]
    In LIBERO OSC units (×LIBERO_OSC_TRANSLATION_SCALE to get metres).
    Gripper: 0 = open, 1 = closed (after flip_gripper).
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.robot.franka.normalization_utils import (
    LIBERO_OSC_ROTATION_SCALE,
    LIBERO_OSC_TRANSLATION_SCALE,
    build_libero_proprio,
    flip_gripper,
)
from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    get_action_head,
    get_processor,
    get_proprio_projector,
    update_auto_map,
)
from experiments.robot.robot_utils import get_action, get_model, set_seed_everywhere
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RobotConfig:
    # --- Model ---
    pretrained_checkpoint: str = "output/FINAL-GVLA-v2-merged"
    model_family:         str  = "openvla"
    use_l1_regression:    bool = True
    use_minivlm:          bool = True
    use_pro_version:      bool = True
    use_proprio:          bool = True
    use_film:             bool = False
    use_vggt:             bool = False
    num_images_in_input:  int  = 2       # cam3 (scene) + cam4 (wrist)
    unnorm_key:           str  = "libero_spatial_no_noops"
    save_version:         str  = "vla-adapter"
    phase:                str  = "Inference"
    num_open_loop_steps:  int  = NUM_ACTIONS_CHUNK  # requery every 8 steps
    center_crop:          bool = True
    load_in_8bit:         bool = False
    load_in_4bit:         bool = False
    seed:                 int  = 42

    # --- Task ---
    task_instruction: str = "pick up the blue bowl next to the plate and place it on the plate"

    # --- Trial control ---
    num_trials:         int = 10
    max_steps_per_trial: int = 200     # hard cutoff per trial

    # --- Logging ---
    log_dir: str = "experiments/logs/franka_robot"

    # Set at runtime (not CLI args)
    num_task_tokens: Optional[int] = field(default=None, repr=False)


def _parse_args() -> RobotConfig:
    parser = argparse.ArgumentParser(description="Franka Panda closed-loop VLA inference")
    parser.add_argument("--pretrained_checkpoint", type=str)
    parser.add_argument("--use_vggt",      action="store_true")
    parser.add_argument("--unnorm_key",    type=str)
    parser.add_argument("--task_instruction", type=str)
    parser.add_argument("--num_trials",    type=int)
    parser.add_argument("--max_steps_per_trial", type=int)
    parser.add_argument("--log_dir",       type=str)
    parser.add_argument("--num_open_loop_steps", type=int)
    args = parser.parse_args()

    cfg = RobotConfig()
    for k, v in vars(args).items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg


# =============================================================================
# Model loading
# =============================================================================

def load_model(cfg: RobotConfig):
    """Load VLA + action head + proprio projector (+ VGGT if requested)."""
    print(f"\nLoading model from {cfg.pretrained_checkpoint} ...")
    model = get_model(cfg)
    model.set_version(cfg.save_version)

    num_patches = (
        model.vision_backbone.get_num_patches()
        * model.vision_backbone.get_num_images_in_input()
    )
    if cfg.use_proprio:
        num_patches += 1
    cfg.num_task_tokens = num_patches

    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=PROPRIO_DIM)
    action_head       = get_action_head(cfg, model.llm_dim)
    processor         = get_processor(cfg)

    vggt_query_module = None
    if cfg.use_vggt:
        vggt_query_module = _load_vggt(cfg, model)

    print("Model ready.\n")
    return model, processor, action_head, proprio_projector, vggt_query_module


def _load_vggt(cfg: RobotConfig, vla):
    """Load the VGGT backbone and query module, attach backbone to vla."""
    from vggt.models.vggt import VGGT
    from vggt.vggt_action_queries import VGGTActionQueryModule

    device = next(vla.parameters()).device

    print("Loading VGGT backbone (facebook/vggt-1b) ...")
    vggt = VGGT.from_pretrained("facebook/vggt-1b").to(torch.bfloat16).to(device)
    vggt.eval()
    vla.vggt = vggt          # get_vla_action checks hasattr(vla, 'vggt')

    print("Loading VGGT query module ...")
    query_module = VGGTActionQueryModule(
        num_queries=64,
        vggt_dim=2048,
        llm_dim=vla.llm_dim,
        num_feature_layers=24,
        num_ca_layers=6,
        num_heads=8,
        dropout=0.0,
        stride=2,
    ).to(torch.bfloat16).to(device)

    ckpt_dir = Path(cfg.pretrained_checkpoint)
    ckpt_files = [f for f in os.listdir(ckpt_dir) if "vggt_query_module" in f and "checkpoint" in f]
    assert len(ckpt_files) == 1, f"Expected 1 vggt_query_module checkpoint, found: {ckpt_files}"
    state_dict = torch.load(ckpt_dir / ckpt_files[0], map_location=device, weights_only=True)
    state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    query_module.load_state_dict(state_dict)
    query_module.eval()

    print("VGGT loaded and attached.")
    return query_module


# =============================================================================
# Robot interface  ← FILL IN THESE SECTIONS FOR YOUR SETUP
# =============================================================================

def connect_robot():
    """
    TODO: Initialize connection to the Franka robot.

    Return whatever handle / client object your robot SDK uses.
    Example libraries: polymetis, frankx, franka-interface, ROS, etc.

    Should configure:
      - Camera streams (cam3 scene, cam4 wrist)
      - Proprioception feed (EEF pose + gripper)
      - Command interface for delta-EEF actions
    """
    raise NotImplementedError(
        "TODO: connect_robot() — wire up your Franka SDK / ROS node here"
    )


def get_observation(robot_handle) -> dict:
    """
    TODO: Read one synchronized observation from the robot.

    Returns a dict with:
        full_image  : np.ndarray [H, W, 3] uint8   — scene camera (cam3)
        wrist_image : np.ndarray [H, W, 3] uint8   — wrist camera (cam4)
        state       : np.ndarray [8] float64        — build_libero_proprio(xyz, quat_xyzw, gripper_qpos)

    Example:
        xyz          = robot.get_eef_position()         # (3,)
        quat_xyzw    = robot.get_eef_quaternion_xyzw()  # (4,)
        gripper_qpos = robot.get_gripper_qpos()         # scalar [0,1]
        state = build_libero_proprio(xyz, quat_xyzw, gripper_qpos)
        scene_img = cam3.grab_frame()   # numpy [H, W, 3]
        wrist_img = cam4.grab_frame()   # numpy [H, W, 3]
        return {"full_image": scene_img, "wrist_image": wrist_img, "state": state}
    """
    raise NotImplementedError(
        "TODO: get_observation() — read cameras + proprio from robot here"
    )


def send_action(robot_handle, action: np.ndarray):
    """
    TODO: Send a 7-dim EEF delta action to the robot.

    action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        - dx/dy/dz in LIBERO OSC units; multiply by LIBERO_OSC_TRANSLATION_SCALE (0.05) for metres
        - droll/dpitch/dyaw in LIBERO OSC units; multiply by LIBERO_OSC_ROTATION_SCALE (0.2) for radians
        - gripper: 0.0 = open, 1.0 = closed (already flipped by flip_gripper)

    Example (pseudo-code):
        delta_xyz  = action[:3]  * LIBERO_OSC_TRANSLATION_SCALE   # metres
        delta_rpy  = action[3:6] * LIBERO_OSC_ROTATION_SCALE       # radians
        gripper    = action[6]                                      # 0=open, 1=closed
        robot.move_eef_delta(delta_xyz, delta_rpy)
        robot.set_gripper(gripper)
    """
    raise NotImplementedError(
        "TODO: send_action() — convert action to robot commands here"
    )


def reset_robot(robot_handle):
    """
    TODO: Move robot to home/ready position between trials.
    """
    raise NotImplementedError(
        "TODO: reset_robot() — move to home pose here"
    )


# =============================================================================
# Inference loop
# =============================================================================

def run_trial(cfg, robot_handle, model, processor, action_head, proprio_projector,
              vggt_query_module, trial_idx: int) -> dict:
    """Run one closed-loop trial. Returns a stats dict."""
    print(f"\n{'='*60}")
    print(f"Trial {trial_idx + 1}/{cfg.num_trials}")
    print(f"Task: {cfg.task_instruction}")
    print(f"{'='*60}")
    input("Press Enter to start trial (robot should be at home position)...")

    action_queue: List[np.ndarray] = []
    step = 0
    t_start = time.time()
    total_query_time = 0.0

    try:
        while step < cfg.max_steps_per_trial:
            obs = get_observation(robot_handle)

            if len(action_queue) == 0:
                # Requery the policy for a new chunk of 8 actions
                t_q = time.time()
                actions = get_action(
                    cfg,
                    model,
                    obs,
                    cfg.task_instruction,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    use_minivlm=cfg.use_minivlm,
                    vggt_query_module=vggt_query_module,
                )
                total_query_time += time.time() - t_q

                # Flip gripper convention: model outputs 0=open,1=closed
                # → Franka expects the same, but verify and adjust if needed
                actions = [flip_gripper(a) for a in actions]
                action_queue.extend(actions)
                print(f"  step {step:3d}: requeried policy, "
                      f"chunk[0][:3]={actions[0][:3].round(4)}")

            action = action_queue.pop(0)
            send_action(robot_handle, action)
            step += 1

    except KeyboardInterrupt:
        print("\n[Interrupted by user]")

    duration = time.time() - t_start
    num_queries = max(1, step // cfg.num_open_loop_steps)
    avg_query_time = total_query_time / num_queries

    success_input = input("\nSuccess? (y/n): ").strip().lower()
    success = success_input == "y"

    print(f"  steps={step}, duration={duration:.1f}s, "
          f"avg_query_time={avg_query_time:.3f}s, success={success}")

    return {
        "trial": trial_idx + 1,
        "steps": step,
        "duration": duration,
        "avg_query_time": avg_query_time,
        "success": success,
        "checkpoint": cfg.pretrained_checkpoint,
        "use_vggt": cfg.use_vggt,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    cfg = _parse_args()
    set_seed_everywhere(cfg.seed)
    os.makedirs(cfg.log_dir, exist_ok=True)

    print("="*60)
    print("Franka Panda closed-loop VLA inference")
    print(f"Checkpoint : {cfg.pretrained_checkpoint}")
    print(f"VGGT       : {cfg.use_vggt}")
    print(f"Task       : {cfg.task_instruction}")
    print(f"Trials     : {cfg.num_trials}")
    print("="*60)

    model, processor, action_head, proprio_projector, vggt_query_module = load_model(cfg)

    robot = connect_robot()

    all_results = []
    total_successes = 0

    for trial_idx in range(cfg.num_trials):
        if trial_idx > 0:
            reset_robot(robot)

        stats = run_trial(
            cfg, robot, model, processor, action_head, proprio_projector,
            vggt_query_module, trial_idx,
        )
        all_results.append(stats)
        if stats["success"]:
            total_successes += 1

        success_rate = total_successes / (trial_idx + 1)
        print(f"  Running success rate: {total_successes}/{trial_idx+1} "
              f"({success_rate*100:.1f}%)")

    # Save results
    tag = "gvla" if cfg.use_vggt else "baseline"
    ts = time.strftime("%Y%m%d_%H%M%S")
    results_path = Path(cfg.log_dir) / f"results_{tag}_{ts}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    final_rate = total_successes / len(all_results)
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS ({tag})")
    print(f"  Trials:       {len(all_results)}")
    print(f"  Successes:    {total_successes}")
    print(f"  Success rate: {final_rate*100:.1f}%")
    print(f"  Saved to:     {results_path}")
    print("="*60)


if __name__ == "__main__":
    main()
