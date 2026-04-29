"""
finetune_franka_gvla.py

LoRA fine-tuning of GVLA (VLA-Adapter + VGGT) on collected Franka demo episodes.

Starts from output/FINAL-GVLA-v2-merged (run prepare_gvla_checkpoint.py first).
Trains: new LoRA on the VLM + action_head + proprio_projector + vggt_query_module.
The VGGT backbone (facebook/vggt-1b) is frozen throughout.

Key architectural difference from finetune_franka.py:
  VGGT features replace vision patches as task context in the multi-layer hidden
  states fed to the action head (matching the forward pass in vla-scripts/finetune.py).
  action_head.predict_action(..., num_task_tokens=64) instead of num_patches.

Episode data layout (same as baseline):
  state/pose_wrt_world.npy   (T, 7)
  state/grasp.npy            (T,)
  state/gripper_qpos.npy     (T,)
  cam3/rgb/*.png             scene camera
  cam4/rgb/*.png             wrist camera

Run with:
    sbatch experiments/robot/franka/run_finetune_gvla.sh
"""

import copy
import json
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TVF
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.robot.franka.normalization_utils import (
    LIBERO_OSC_ROTATION_SCALE,
    LIBERO_OSC_TRANSLATION_SCALE,
    build_libero_proprio,
    quat_xyzw_to_axisangle,
)
from experiments.robot.openvla_utils import check_model_logic_mismatch, update_auto_map
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.projectors import ProprioProjector
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    NUM_TOKENS,
    PROPRIO_DIM,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FinetuneGVLAConfig:
    # --- Model ---
    pretrained_checkpoint: str = "output/FINAL-GVLA-v2-merged"
    model_family: str = "openvla"
    use_l1_regression: bool = True
    use_minivlm: bool = True
    use_pro_version: bool = True
    use_proprio: bool = True
    num_images_in_input: int = 2        # cam3 (scene) + cam4 (wrist)
    unnorm_key: str = "libero_spatial_no_noops"
    save_version: str = "vla-adapter"
    phase: str = "Training"
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # --- Dataset ---
    episodes_dir: str = "/users/mfenner1/workspace/csci2951K/final_project/episodes_bowl_pickplace"
    task_instruction: str = "pick up the blue bowl next to the plate and place it on the plate"

    # --- Output ---
    output_dir: str = "output/FINAL-GVLA-v2-FrankaFT"

    # --- LoRA ---
    lora_rank: int = 8
    lora_dropout: float = 0.05

    # --- Training ---
    batch_size: int = 2
    learning_rate: float = 2e-5
    lr_warmup_steps: int = 20
    max_steps: int = 300
    grad_accumulation_steps: int = 1
    save_freq: int = 100
    seed: int = 42

    # --- VGGT ---
    vggt_dropout: float = 0.1   # probability of zeroing VGGT features per sample


# ---------------------------------------------------------------------------
# VGGT image preprocessing
# ---------------------------------------------------------------------------

def _preprocess_for_vggt(img_path: Path) -> torch.Tensor:
    """
    Preprocess a scene image for VGGT: resize to 224, normalize to [0,1].
    Returns float32 tensor [3, 224, 224] on CPU.
    """
    img = Image.open(img_path).convert("RGB").resize((224, 224), Image.BILINEAR)
    return TVF.to_tensor(img)          # float32 [3,224,224] in [0,1]


# ---------------------------------------------------------------------------
# Action computation helpers  (identical to finetune_franka.py)
# ---------------------------------------------------------------------------

def _quat_multiply(q1_xyzw: np.ndarray, q2_xyzw: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1_xyzw
    x2, y2, z2, w2 = q2_xyzw
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dtype=np.float64)


def _quat_conjugate(q_xyzw: np.ndarray) -> np.ndarray:
    return np.array([-q_xyzw[0], -q_xyzw[1], -q_xyzw[2], q_xyzw[3]], dtype=np.float64)


def _compute_actions_osc(pose: np.ndarray, grasp: np.ndarray) -> np.ndarray:
    T = len(pose)
    actions = np.zeros((T, 7), dtype=np.float64)
    for t in range(T - 1):
        delta_xyz = pose[t + 1, :3] - pose[t, :3]
        q_rel = _quat_multiply(pose[t + 1, 3:7], _quat_conjugate(pose[t, 3:7]))
        delta_rot = quat_xyzw_to_axisangle(q_rel)
        actions[t, :3] = delta_xyz / LIBERO_OSC_TRANSLATION_SCALE
        actions[t, 3:6] = delta_rot / LIBERO_OSC_ROTATION_SCALE
        actions[t, 6] = float(grasp[t])
    if T > 1:
        actions[-1] = actions[-2]
    return actions


def _normalize_action(action_osc: np.ndarray, norm_stats: Dict) -> np.ndarray:
    q01 = np.array(norm_stats["action"]["q01"])
    q99 = np.array(norm_stats["action"]["q99"])
    mask = np.array(norm_stats["action"].get("mask", np.ones(ACTION_DIM, dtype=bool)), dtype=bool)
    normalized = np.where(
        mask,
        2.0 * (action_osc - q01) / (q99 - q01 + 1e-8) - 1.0,
        action_osc,
    )
    normalized = np.where(mask, np.clip(normalized, -1.0, 1.0), normalized)
    return normalized.astype(np.float32)


def _normalize_proprio(proprio_raw: np.ndarray, norm_stats: Dict) -> np.ndarray:
    q01 = np.array(norm_stats["proprio"]["q01"])
    q99 = np.array(norm_stats["proprio"]["q99"])
    normalized = 2.0 * (proprio_raw - q01) / (q99 - q01 + 1e-8) - 1.0
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FrankaGVLADataset(Dataset):
    """
    Identical to FrankaEpisodeDataset but adds vggt_pixel_values:
    the scene image (cam3) preprocessed for VGGT (float32 [3,224,224]).
    """

    def __init__(
        self,
        episodes_dir: str,
        processor,
        norm_stats: Dict[str, Any],
        task_instruction: str,
        chunk_size: int = NUM_ACTIONS_CHUNK,
    ):
        self.image_transform = processor.image_processor.apply_transform
        self.base_tokenizer = processor.tokenizer
        self.action_tokenizer = ActionTokenizer(processor.tokenizer)
        self.norm_stats = norm_stats
        self.task_instruction = task_instruction
        self.chunk_size = chunk_size

        self.samples: List[tuple] = []
        self._ep_cache: Dict[Path, Dict] = {}

        for ep_path in sorted(Path(episodes_dir).iterdir()):
            if not ep_path.is_dir():
                continue
            state_dir = ep_path / "state"
            pose         = np.load(state_dir / "pose_wrt_world.npy")
            grasp        = np.load(state_dir / "grasp.npy")
            gripper_qpos = np.load(state_dir / "gripper_qpos.npy")
            T = len(pose)

            cam3 = sorted((ep_path / "cam3" / "rgb").glob("*.png"))
            cam4 = sorted((ep_path / "cam4" / "rgb").glob("*.png"))

            proprio_raw = np.stack([
                build_libero_proprio(pose[t, :3], pose[t, 3:7], gripper_qpos[t])
                for t in range(T)
            ])
            actions_osc = _compute_actions_osc(pose, grasp)

            self._ep_cache[ep_path] = {
                "proprio_raw": proprio_raw,
                "actions_osc": actions_osc,
                "cam3": cam3,
                "cam4": cam4,
            }
            for t in range(T - chunk_size):
                self.samples.append((ep_path, t))

        print(
            f"FrankaGVLADataset: {len(self.samples)} samples "
            f"from {len(self._ep_cache)} episodes"
        )

    def _build_tokens(self, actions_chunk_norm: np.ndarray):
        lang = self.task_instruction.lower()
        prompt = (
            "<|im_start|>system\n"
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
            "<|im_end|>\n"
            f"<|im_start|>user\nWhat action should the robot take to {lang}?"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        input_ids: List[int] = self.base_tokenizer(prompt, add_special_tokens=True).input_ids
        if len(input_ids) >= 3:
            del input_ids[-1]
            del input_ids[-1]
            del input_ids[-1]

        current_tok: List[int] = self.action_tokenizer(actions_chunk_norm[0], use_minivlm=True)
        future_tok: List[List[int]] = [
            self.action_tokenizer(actions_chunk_norm[i], use_minivlm=True)
            for i in range(1, self.chunk_size)
        ]
        flat_tok: List[int] = current_tok + [x for sub in future_tok for x in sub]

        if len(flat_tok) >= NUM_TOKENS:
            flat_tok = flat_tok[:NUM_TOKENS]
        else:
            extra = random.choices(flat_tok, k=NUM_TOKENS - len(flat_tok))
            flat_tok = flat_tok + extra

        input_ids = input_ids + flat_tok
        labels = list(input_ids)
        input_ids_t = torch.tensor(input_ids)
        labels_t = torch.tensor(labels)
        labels_t[: -(NUM_TOKENS + 1)] = IGNORE_INDEX
        return input_ids_t, labels_t

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep_path, t = self.samples[idx]
        ep = self._ep_cache[ep_path]

        scene_img = Image.open(ep["cam3"][t]).convert("RGB")
        wrist_img = Image.open(ep["cam4"][t]).convert("RGB")
        pixel_values       = self.image_transform(scene_img)
        pixel_values_wrist = self.image_transform(wrist_img)

        # VGGT input: scene image preprocessed to float32 [3,224,224]
        vggt_pixel_values = _preprocess_for_vggt(ep["cam3"][t])

        proprio = _normalize_proprio(ep["proprio_raw"][t], self.norm_stats)

        actions_chunk = np.stack([
            _normalize_action(ep["actions_osc"][t + i], self.norm_stats)
            for i in range(self.chunk_size)
        ])

        input_ids, labels = self._build_tokens(actions_chunk)

        return {
            "pixel_values":       pixel_values,
            "pixel_values_wrist": pixel_values_wrist,
            "vggt_pixel_values":  vggt_pixel_values,     # [3,224,224] float32
            "input_ids":          input_ids,
            "labels":             labels,
            "actions":            actions_chunk,          # (chunk_size, ACTION_DIM)
            "proprio":            proprio,                # (PROPRIO_DIM,)
            "dataset_name":       "franka_bowl_pickplace",
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _register_hf_classes():
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


def train(cfg: FinetuneGVLAConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load norm stats ---
    stats_path = Path(cfg.pretrained_checkpoint) / "dataset_statistics.json"
    with open(stats_path) as f:
        all_stats = json.load(f)
    norm_stats = all_stats[cfg.unnorm_key]

    # --- Processor ---
    _register_hf_classes()
    update_auto_map(cfg.pretrained_checkpoint)
    check_model_logic_mismatch(cfg.pretrained_checkpoint)
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)

    # --- Load base VLA (merged GVLA weights) ---
    print("Loading GVLA base model ...")
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        trust_remote_code=False,
    )
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    llm_dim = vla.llm_dim
    num_patches = (
        vla.vision_backbone.get_num_patches()
        * vla.vision_backbone.get_num_images_in_input()
    )

    # --- Apply LoRA ---
    print(f"Applying LoRA (rank={cfg.lora_rank}) ...")
    lora_cfg = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=2 * cfg.lora_rank,
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora_cfg)
    vla.print_trainable_parameters()
    vla = vla.to(device)

    # --- Action head + proprio projector ---
    print("Loading action head and proprio projector ...")
    action_head = L1RegressionActionHead(
        input_dim=llm_dim,
        hidden_dim=llm_dim,
        action_dim=ACTION_DIM,
        use_pro_version=cfg.use_pro_version,
    ).to(torch.bfloat16).to(device)

    ah_files = [
        f for f in os.listdir(cfg.pretrained_checkpoint)
        if "action_head" in f and "checkpoint" in f
    ]
    assert len(ah_files) == 1, f"Expected 1 action_head file, found: {ah_files}"
    ah_state = torch.load(
        Path(cfg.pretrained_checkpoint) / ah_files[0],
        map_location=device, weights_only=True,
    )
    ah_state = {k[7:] if k.startswith("module.") else k: v for k, v in ah_state.items()}
    action_head.load_state_dict(ah_state)

    proprio_projector = ProprioProjector(
        llm_dim=llm_dim,
        proprio_dim=PROPRIO_DIM,
    ).to(torch.bfloat16).to(device)

    pp_files = [
        f for f in os.listdir(cfg.pretrained_checkpoint)
        if "proprio_projector" in f and "checkpoint" in f
    ]
    assert len(pp_files) == 1, f"Expected 1 proprio_projector file, found: {pp_files}"
    pp_state = torch.load(
        Path(cfg.pretrained_checkpoint) / pp_files[0],
        map_location=device, weights_only=True,
    )
    pp_state = {k[7:] if k.startswith("module.") else k: v for k, v in pp_state.items()}
    proprio_projector.load_state_dict(pp_state)

    # --- VGGT backbone (frozen) + query module (trainable) ---
    from vggt.models.vggt import VGGT
    from vggt.vggt_action_queries import VGGTActionQueryModule

    print("Loading VGGT backbone (facebook/vggt-1b) — frozen ...")
    vggt = VGGT.from_pretrained("facebook/vggt-1b").to(torch.bfloat16).to(device)
    vggt.eval()
    for p in vggt.parameters():
        p.requires_grad = False

    print("Loading VGGT query module ...")
    vggt_query_module = VGGTActionQueryModule(
        num_queries=64,
        vggt_dim=2048,
        llm_dim=llm_dim,
        num_feature_layers=24,
        num_ca_layers=6,
        num_heads=8,
        dropout=0.0,
        stride=2,
    ).to(torch.bfloat16).to(device)

    qm_files = [
        f for f in os.listdir(cfg.pretrained_checkpoint)
        if "vggt_query_module" in f and "checkpoint" in f
    ]
    assert len(qm_files) == 1, f"Expected 1 vggt_query_module file, found: {qm_files}"
    qm_state = torch.load(
        Path(cfg.pretrained_checkpoint) / qm_files[0],
        map_location=device, weights_only=True,
    )
    qm_state = {k[7:] if k.startswith("module.") else k: v for k, v in qm_state.items()}
    vggt_query_module.load_state_dict(qm_state)

    # --- Dataset + dataloader ---
    dataset = FrankaGVLADataset(
        cfg.episodes_dir, processor, norm_stats, cfg.task_instruction,
    )

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # --- Optimizer: LoRA params + action_head + proprio_projector + vggt_query_module ---
    trainable_params = [p for p in vla.parameters() if p.requires_grad]
    trainable_params += list(action_head.parameters())
    trainable_params += list(proprio_projector.parameters())
    trainable_params += list(vggt_query_module.parameters())
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=1e-4)

    def _lr_lambda(step: int) -> float:
        if step < cfg.lr_warmup_steps:
            return 0.1 + 0.9 * step / max(1, cfg.lr_warmup_steps)
        progress = (step - cfg.lr_warmup_steps) / max(1, cfg.max_steps - cfg.lr_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # --- Training loop ---
    vla.train()
    action_head.train()
    proprio_projector.train()
    vggt_query_module.train()

    step = 0
    data_iter = iter(dataloader)

    print(f"\nStarting GVLA training: max_steps={cfg.max_steps}, batch_size={cfg.batch_size}")
    print(f"Dataset size: {len(dataset)} samples  |  Steps per epoch: {len(dataloader)}\n")

    while step < cfg.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        ground_truth_actions = batch["actions"].to(device, dtype=torch.bfloat16)
        batch_size = batch["input_ids"].shape[0]

        # --- VGGT forward (frozen backbone, trainable query module) ---
        with torch.no_grad():
            vggt_input = (
                batch["vggt_pixel_values"]
                .to(device, dtype=torch.bfloat16)
                .unsqueeze(1)        # [B, 1, 3, 224, 224]
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vggt_tokens_list, patch_start_idx = vggt.aggregator(vggt_input)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            vggt_query_features = vggt_query_module(vggt_tokens_list, patch_start_idx)
        # shape: [B, num_feature_layers, num_queries, llm_dim]
        vggt_num_task    = vggt_query_features.shape[2]   # 64
        num_vggt_layers  = vggt_query_features.shape[1]   # 24

        # Optional dropout on VGGT features (regularization)
        if cfg.vggt_dropout > 0.0:
            drop_mask = (torch.rand(batch_size) < cfg.vggt_dropout).to(vggt_query_features.device)
            vggt_query_features = vggt_query_features * (
                1.0 - drop_mask[:, None, None, None].to(vggt_query_features.dtype)
            )

        # --- VLA forward pass ---
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = vla(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                pixel_values=batch["pixel_values"].to(device, dtype=torch.bfloat16),
                labels=batch["labels"].to(device),
                output_hidden_states=True,
                proprio=batch["proprio"].to(device, dtype=torch.bfloat16) if cfg.use_proprio else None,
                proprio_projector=proprio_projector if cfg.use_proprio else None,
                noisy_actions=None,
                noisy_action_projector=None,
                diffusion_timestep_embeddings=None,
                use_film=False,
            )

        # --- Build multi-layer hidden states with VGGT task context ---
        ground_truth_token_ids = batch["labels"][:, 1:].to(device)
        curr_mask = get_current_action_mask(ground_truth_token_ids)
        next_mask = get_next_actions_mask(ground_truth_token_ids)
        action_mask = curr_mask | next_mask

        multi_layer_hidden = []
        for layer_idx, layer_hidden in enumerate(output.hidden_states):
            text_hidden = layer_hidden[:, num_patches:-1]
            action_hidden = (
                text_hidden[action_mask]
                .reshape(batch_size, 1, NUM_TOKENS, -1)
                .to(torch.bfloat16)
            )

            # VGGT features replace vision patches as task context
            vggt_idx = min(layer_idx, num_vggt_layers - 1)
            task_hidden = vggt_query_features[:, vggt_idx:vggt_idx+1, :, :]  # [B,1,64,D]

            multi_layer_hidden.append(torch.cat([task_hidden, action_hidden], dim=2))

        multi_layer_hidden = torch.cat(multi_layer_hidden, dim=1)

        predicted_actions = action_head.predict_action(
            multi_layer_hidden,
            proprio=batch["proprio"].to(device, dtype=torch.bfloat16) if cfg.use_proprio else None,
            proprio_projector=proprio_projector if cfg.use_proprio else None,
            phase=cfg.phase,
            num_task_tokens=vggt_num_task,    # 64 — tells action head how many task tokens
        )

        loss = nn.functional.l1_loss(predicted_actions, ground_truth_actions)

        (loss / cfg.grad_accumulation_steps).backward()

        if (step + 1) % cfg.grad_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  step {step + 1:4d}/{cfg.max_steps}  loss={loss.item():.4f}  lr={lr_now:.2e}")

        step += 1

        if step % cfg.save_freq == 0 or step == cfg.max_steps:
            _save_checkpoint(cfg, vla, action_head, proprio_projector, vggt_query_module,
                             processor, step, device)

    print("\nGVLA training complete.")


def _save_checkpoint(cfg, vla, action_head, proprio_projector, vggt_query_module,
                     processor, step, device):
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Saving checkpoint at step {step} → {out_dir}")

    merged_vla = copy.deepcopy(vla).merge_and_unload()
    merged_vla.save_pretrained(str(out_dir))
    processor.save_pretrained(str(out_dir))

    torch.save(action_head.state_dict(),       out_dir / "action_head--checkpoint.pt")
    torch.save(proprio_projector.state_dict(), out_dir / "proprio_projector--checkpoint.pt")
    torch.save(vggt_query_module.state_dict(), out_dir / "vggt_query_module--checkpoint.pt")

    shutil.copy2(
        Path(cfg.pretrained_checkpoint) / "dataset_statistics.json",
        out_dir / "dataset_statistics.json",
    )
    print(f"  Checkpoint saved to {out_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = FinetuneGVLAConfig()
    train(cfg)
