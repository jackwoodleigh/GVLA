#!/bin/bash
#SBATCH --job-name=gvla-prep
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=48G
#SBATCH -t 0:30:00
#SBATCH -o /users/mfenner1/workspace/csci2951K/final_project/GVLA/logs/prepare_gvla_%j.out
#SBATCH -e /users/mfenner1/workspace/csci2951K/final_project/GVLA/logs/prepare_gvla_%j.err

GVLA_ROOT="/users/mfenner1/workspace/csci2951K/final_project/GVLA"
cd "$GVLA_ROOT"

source /users/mfenner1/workspace/envs/vla-adapter/bin/activate

export PYTHONPATH="$GVLA_ROOT:$PYTHONPATH"

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Merging GVLA LoRA into base checkpoint..."

python experiments/robot/franka/prepare_gvla_checkpoint.py

echo "Done. Output: output/FINAL-GVLA-v2-merged/"
