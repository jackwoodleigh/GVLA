"""
prepare_gvla_checkpoint.py

Unzip the FINAL-GVLA LoRA checkpoint and merge its LoRA adapter into the
LIBERO-Spatial-Pro base model, producing a self-contained checkpoint directory
that deploy_franka.py (and get_vla / get_action_head / etc.) can load directly.

Run once before your first deployment:
    sbatch experiments/robot/franka/run_prepare_gvla.sh
    # or directly (needs GPU for bfloat16 model load):
    python experiments/robot/franka/prepare_gvla_checkpoint.py

Output: output/FINAL-GVLA-v2-merged/
    config.json, modeling_prismatic.py, model.safetensors  ← merged weights
    action_head--checkpoint.pt
    vggt_query_module--checkpoint.pt
    proprio_projector--checkpoint.pt
    dataset_statistics.json
    tokenizer / processor files
"""

import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.robot.openvla_utils import check_model_logic_mismatch, update_auto_map
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


GVLA_ROOT       = Path(__file__).resolve().parents[3]
GVLA_CKPT_DIR   = Path("/users/mfenner1/scratch/GVLA_checkpoints/FINAL-GVLA-v2(6-layer-stride2)--110000_chkpt")
BASE_CHECKPOINT = GVLA_ROOT / "output" / "LIBERO-Spatial-Pro"
OUT_DIR         = GVLA_ROOT / "output" / "FINAL-GVLA-v2-merged"

# Component filenames inside the unzipped GVLA checkpoint directory
COMPONENT_SRCS = {
    "action_head--checkpoint.pt":       "action_head--110000_checkpoint.pt",
    "vggt_query_module--checkpoint.pt": "vggt_query_module--110000_checkpoint.pt",
    "proprio_projector--checkpoint.pt": "proprio_projector--110000_checkpoint.pt",
}

# Tokenizer / processor files to carry over from the GVLA checkpoint
TOKENIZER_FILES = [
    "added_tokens.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "preprocessor_config.json",
    "processing_prismatic.py",
    "processor_config.json",
]


def _register_hf_classes():
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


def main():
    # -----------------------------------------------------------------------
    # 1. Locate pre-extracted GVLA checkpoint
    # -----------------------------------------------------------------------
    gvla_dir = GVLA_CKPT_DIR
    assert gvla_dir.is_dir(), f"GVLA checkpoint dir not found: {gvla_dir}"
    print(f"Using GVLA checkpoint from {gvla_dir}")

    lora_adapter_dir = gvla_dir / "lora_adapter"
    assert lora_adapter_dir.is_dir(), "lora_adapter/ not found inside extracted checkpoint"

    # -----------------------------------------------------------------------
    # 2. Load the base VLA model
    # -----------------------------------------------------------------------
    print(f"\nLoading base model from {BASE_CHECKPOINT} ...")
    _register_hf_classes()
    update_auto_map(str(BASE_CHECKPOINT))
    check_model_logic_mismatch(str(BASE_CHECKPOINT))

    base_vla = AutoModelForVision2Seq.from_pretrained(
        str(BASE_CHECKPOINT),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        trust_remote_code=False,
    )
    print("Base model loaded.")

    # -----------------------------------------------------------------------
    # 3. Apply and merge the LoRA adapter
    # -----------------------------------------------------------------------
    print(f"\nApplying LoRA from {lora_adapter_dir} ...")
    peft_model = PeftModel.from_pretrained(base_vla, str(lora_adapter_dir))
    print("Merging LoRA weights into base model ...")
    merged_vla = peft_model.merge_and_unload()
    print("Merge complete.")

    # -----------------------------------------------------------------------
    # 4. Save the merged model
    # -----------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving merged model to {OUT_DIR} ...")
    merged_vla.save_pretrained(str(OUT_DIR))

    # Copy processor / tokenizer from base checkpoint (safe_serialization writes model only)
    processor = AutoProcessor.from_pretrained(str(BASE_CHECKPOINT), trust_remote_code=True)
    processor.save_pretrained(str(OUT_DIR))
    print("Processor saved.")

    # -----------------------------------------------------------------------
    # 5. Copy component checkpoints
    # -----------------------------------------------------------------------
    print("\nCopying component checkpoints ...")
    for dst_name, src_name in COMPONENT_SRCS.items():
        src = gvla_dir / src_name
        dst = OUT_DIR / dst_name
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  {src_name} → {dst_name}")
        else:
            print(f"  WARNING: {src_name} not found in {gvla_dir}, skipping")

    # -----------------------------------------------------------------------
    # 6. Copy dataset statistics
    # -----------------------------------------------------------------------
    stats_src = gvla_dir / "dataset_statistics.json"
    if stats_src.exists():
        shutil.copy2(stats_src, OUT_DIR / "dataset_statistics.json")
        print("  dataset_statistics.json copied from GVLA checkpoint")
    else:
        # Fall back to base checkpoint stats
        shutil.copy2(BASE_CHECKPOINT / "dataset_statistics.json", OUT_DIR / "dataset_statistics.json")
        print("  dataset_statistics.json copied from LIBERO-Spatial-Pro (GVLA version missing)")

    print(f"\nDone. Merged GVLA checkpoint ready at:\n  {OUT_DIR}")
    print("\nFiles in output directory:")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
