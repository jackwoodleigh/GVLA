#!/bin/bash
#SBATCH --job-name=split-dataset
#SBATCH -p batch
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH -t 0:05:00
#SBATCH -o /users/mfenner1/workspace/csci2951K/final_project/GVLA/logs/split_dataset_%j.out
#SBATCH -e /users/mfenner1/workspace/csci2951K/final_project/GVLA/logs/split_dataset_%j.err

GVLA_ROOT="/users/mfenner1/workspace/csci2951K/final_project/GVLA"
cd "$GVLA_ROOT"

source /users/mfenner1/workspace/envs/vla-adapter/bin/activate

export PYTHONPATH="$GVLA_ROOT:$PYTHONPATH"

echo "Creating train/test split..."
python experiments/robot/franka/split_dataset.py

echo "Done."
