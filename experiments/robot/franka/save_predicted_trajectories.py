"""
save_predicted_trajectories.py

For each demo episode, runs the VLA-Adapter model at every timestep and saves
the predicted EEF trajectory as a numpy array for visualization in the local
point cloud viewer.

Output per episode:
    predicted_trajectory.npy  →  shape (T, N, 4)
        T = timesteps in episode
        N = rollout length (NUM_ACTIONS_CHUNK, default 8)
        4 = [x, y, z, gripper]
            x, y, z  : absolute EEF position in robot base frame (meters)
                        integrated from pose_wrt_world[t, :3] + cumsum(delta_xyz)
            gripper  : 0.0 = closed, 1.0 = open

Run with:
    sbatch experiments/robot/franka/run_save_trajectories.sh
(from the GVLA root directory)
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.robot.franka.normalization_utils import (
    build_libero_proprio,
    flip_gripper,
    LIBERO_OSC_TRANSLATION_SCALE,
)
from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector
from experiments.robot.robot_utils import get_action, get_model, set_seed_everywhere
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class Config:
    # --- Model (must match the checkpoint) ---
    pretrained_checkpoint: str = "output/LIBERO-Spatial-Pro-FrankaFT"
    model_family: str = "openvla"
    use_l1_regression: bool = True
    use_minivlm: bool = True
    use_film: bool = False
    use_vggt: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    unnorm_key: str = "libero_spatial_no_noops"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_pro_version: bool = True
    save_version: str = "vla-adapter"
    phase: str = "Inference"
    num_open_loop_steps: int = NUM_ACTIONS_CHUNK
    seed: int = 42

    # --- Task ---
    task_instruction: str = "pick up the blue bowl next to the plate and place it on the plate"

    # --- Paths ---
    episodes_dir: str = "/users/mfenner1/workspace/csci2951K/final_project/episodes_bowl_pickplace"


# ==============================================================================
# Per-episode inference
# ==============================================================================

def process_episode(cfg, episode_path, model, processor, action_head, proprio_projector):
    """
    Query the model at every timestep and integrate delta actions into absolute positions.

    Returns:
        predicted_trajectory : (T, N, 4) — [x, y, z, gripper] in robot base frame
    """
    state_dir = episode_path / "state"
    pose         = np.load(state_dir / "pose_wrt_world.npy")   # (T, 7): xyz + quat xyzw
    gripper_qpos = np.load(state_dir / "gripper_qpos.npy")     # (T,)

    cam3_frames = sorted((episode_path / "cam3" / "rgb").glob("*.png"))
    cam4_frames = sorted((episode_path / "cam4" / "rgb").glob("*.png"))

    T = len(cam3_frames)
    assert len(cam3_frames) == len(cam4_frames) == len(pose), (
        f"Frame/state count mismatch in {episode_path.name}: "
        f"cam3={len(cam3_frames)}, cam4={len(cam4_frames)}, state={len(pose)}"
    )

    N = cfg.num_open_loop_steps
    predicted_trajectory = np.zeros((T, N, 4), dtype=np.float32)

    import time
    t_start = time.time()

    for t in range(T):
        proprio = build_libero_proprio(pose[t, :3], pose[t, 3:7], gripper_qpos[t])

        scene_img = np.array(Image.open(cam3_frames[t]).convert("RGB"))
        wrist_img = np.array(Image.open(cam4_frames[t]).convert("RGB"))

        obs = {
            "full_image":  scene_img,
            "wrist_image": wrist_img,
            "state":       proprio,
        }

        # Query model — returns list of N action arrays, each (7,) in LIBERO OSC units
        actions = get_action(
            cfg, model, obs, cfg.task_instruction,
            processor=processor,
            action_head=action_head,
            proprio_projector=proprio_projector,
            use_minivlm=cfg.use_minivlm,
        )
        actions = [flip_gripper(a) for a in actions]   # 0=closed, 1=open
        actions = np.stack(actions)                     # (N, 7)

        # Convert OSC translation units → meters, then integrate from current EEF pos
        delta_xyz = actions[:, :3] * LIBERO_OSC_TRANSLATION_SCALE  # (N, 3)
        positions = np.cumsum(delta_xyz, axis=0) + pose[t, :3]     # (N, 3)
        gripper   = actions[:, 6:7]                                 # (N, 1)

        predicted_trajectory[t] = np.hstack([positions, gripper])  # (N, 4)

        elapsed = time.time() - t_start
        avg_s   = elapsed / (t + 1)
        eta_s   = avg_s * (T - t - 1)
        print(f"  [{t+1:3d}/{T}] {avg_s:.2f}s/step  ETA {eta_s/60:.1f}min", flush=True)

    elapsed_total = time.time() - t_start
    print(f"  Episode done in {elapsed_total/60:.1f}min  ({elapsed_total/T:.2f}s/step avg)")

    return predicted_trajectory


# ==============================================================================
# Main
# ==============================================================================

def main():
    cfg = Config()
    set_seed_everywhere(cfg.seed)

    print("=" * 60)
    print("Saving predicted trajectories for point cloud visualization")
    print(f"Checkpoint : {cfg.pretrained_checkpoint}")
    print(f"Instruction: {cfg.task_instruction}")
    print(f"Episodes   : {cfg.episodes_dir}")
    print(f"Rollout N  : {cfg.num_open_loop_steps}")
    print("=" * 60)

    print("\nLoading model...")
    model = get_model(cfg)
    model.set_version(cfg.save_version)

    NUM_PATCHES = (
        model.vision_backbone.get_num_patches()
        * model.vision_backbone.get_num_images_in_input()
    )
    if cfg.use_proprio:
        NUM_PATCHES += 1
    cfg.num_task_tokens = NUM_PATCHES

    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
    action_head       = get_action_head(cfg, model.llm_dim)
    processor         = get_processor(cfg)
    print("Model loaded.\n")

    episodes = sorted(Path(cfg.episodes_dir).iterdir())
    print(f"Found {len(episodes)} episodes\n")

    for ep_path in episodes:
        if not ep_path.is_dir():
            continue

        out_path = ep_path / "predicted_trajectory_finetuned.npy"
        if out_path.exists():
            print(f"--- {ep_path.name}  [skipping, already exists]")
            continue

        print(f"--- {ep_path.name}")
        traj = process_episode(cfg, ep_path, model, processor, action_head, proprio_projector)
        np.save(out_path, traj)
        print(f"  Saved {traj.shape} → {out_path}\n")

    print("Done.")


if __name__ == "__main__":
    main()
