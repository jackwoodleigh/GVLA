#!/bin/bash
#SBATCH --job-name=vla-eval
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH -t 0:10:00
#SBATCH -o /users/mfenner1/workspace/csci2951K/final_project/GVLA/logs/eval_%j.out
#SBATCH -e /users/mfenner1/workspace/csci2951K/final_project/GVLA/logs/eval_%j.err

GVLA_ROOT="/users/mfenner1/workspace/csci2951K/final_project/GVLA"
cd "$GVLA_ROOT"

source /users/mfenner1/workspace/envs/vla-adapter/bin/activate

export PYTHONPATH="$GVLA_ROOT:$PYTHONPATH"

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Starting eval..."

python experiments/robot/franka/eval_open_loop.py
