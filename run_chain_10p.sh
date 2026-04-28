#!/bin/bash
#SBATCH --job-name=run
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail
shopt -s nullglob

# =====================================================================
# User config
# =====================================================================
RUN_ID="FINAL-GVLA-v2(6-layer-stride2)moddrop10"
OUTDIR="outputs"
DATA_NAME="libero_spatial_no_noops"
WALL_TIME_LIMIT_SECONDS=3000        # emergency save at 50:00 (before the 1:00:00 walltime kill)
STOP_FILE="${OUTDIR}/STOP_CHAIN"    # touch this to halt the chain cleanly

# =====================================================================
# Self-resubmit with afterany dependency (runs the next copy of this
# script the moment the current one exits -- clean, crash, or walltime).
# =====================================================================
NEXT_JOB_FILE=""
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    mkdir -p "${OUTDIR}/.slurm_chain"
    NEXT_JOB_FILE="${OUTDIR}/.slurm_chain/next_job_${SLURM_JOB_ID}.txt"
    if [[ ! -f "$STOP_FILE" ]]; then
        NEXT_JOB_ID=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} "$0")
        echo "$NEXT_JOB_ID" > "$NEXT_JOB_FILE"
        echo "[chain] Submitted successor job: $NEXT_JOB_ID"
    else
        echo "[chain] STOP_CHAIN exists; not submitting successor."
    fi
fi

# =====================================================================
# Original environment (unchanged from your launch.sh)
# =====================================================================
source /users/jwoodlei/scratch/miniconda3/etc/profile.d/conda.sh
conda activate vla-adapter
export WANDB_API_KEY=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

# =====================================================================
# Pick the best resume source.
#   Priority: highest step number wins. Emergency and regular compete
#   directly. If a regular checkpoint is newer than the emergency, we
#   also remove the stale emergency dir.
# =====================================================================
RUN_DIR="${OUTDIR}/${RUN_ID}"
EMERGENCY_DIR="${RUN_DIR}--emergency_chkpt"
EMERGENCY_META="${EMERGENCY_DIR}/emergency_meta.json"

best_step=""
best_dir=""
best_from_emergency=0

# Candidate: emergency
if [[ -f "$EMERGENCY_META" ]]; then
    e_step=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['step'])" "$EMERGENCY_META" 2>/dev/null || echo "")
    if [[ "$e_step" =~ ^[0-9]+$ ]]; then
        best_step=$e_step
        best_dir=$EMERGENCY_DIR
        best_from_emergency=1
    fi
fi

# Candidates: regular stepped checkpoints
for d in "${RUN_DIR}"--*_chkpt; do
    [[ "$d" == *"--emergency_chkpt" ]] && continue
    base=$(basename "$d")
    step=$(echo "$base" | sed -E "s/^.*--([0-9]+)_chkpt$/\\1/")
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    # Also verify the merged model actually landed (merge_lora_during_training can be slow and we don't
    # want to resume from a half-written dir if a previous job died mid-save).
    [[ -f "${d}/config.json" ]] || continue
    if [[ -z "$best_step" ]] || (( step > best_step )); then
        best_step=$step
        best_dir=$d
        best_from_emergency=0
    fi
done

RESUME_ARGS=()
if [[ -n "$best_step" ]]; then
    if [[ $best_from_emergency -eq 1 ]]; then
        echo "[chain] Resuming from EMERGENCY checkpoint at step $best_step ($best_dir)"
        RESUME_ARGS=(
            --resume True
            --resume_step "$best_step"
            --resum_vla_path "$best_dir"
            --config_file_path "$best_dir"
            --resume_from_emergency True
        )
    else
        echo "[chain] Resuming from REGULAR checkpoint at step $best_step ($best_dir)"
        RESUME_ARGS=(
            --resume True
            --resume_step "$best_step"
            --resum_vla_path "$best_dir"
            --config_file_path "$best_dir"
        )
        # Clean up a stale emergency dir that's older than our chosen regular checkpoint.
        if [[ -d "$EMERGENCY_DIR" ]]; then
            echo "[chain] Removing stale emergency checkpoint (regular is newer)."
            rm -rf "$EMERGENCY_DIR"
        fi
    fi
else
    echo "[chain] No checkpoint found; starting fresh."
fi

# =====================================================================
# Launch training. We pass the ORIGINAL config (with pretrained paths)
# as defaults; RESUME_ARGS at the end overrides config_file_path when
# resuming, which is correct (draccus uses last-value-wins).
# =====================================================================
torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune_chained.py \
    --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
    --config_file_path pretrained_models/configs \
    --data_root_dir data/libero \
    --dataset_name "$DATA_NAME" \
    --run_root_dir "$OUTDIR" \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --use_lora True \
    --use_fz False \
    --use_minivlm True \
    --image_aug True \
    --max_steps 150005 \
    --num_steps_before_decay 150000 \
    --save_freq 10000 \
    --shuffle_buffer_size 25000 \
    --save_latest_checkpoint_only False \
    --merge_lora_during_training False \
    --batch_size 8 \
    --grad_accumulation_steps 2 \
    --learning_rate 2e-4 \
    --lora_rank 64 \
    --use_pro_version True \
    --wandb_entity "CollaborativeRobotics" \
    --wandb_project "GVLA" \
    --run_id_override "$RUN_ID" \
    --use_vggt True \
    --modality_drop 0.1 \
    --wall_time_limit_seconds "$WALL_TIME_LIMIT_SECONDS" \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
    &
APP_PID=$!

# Forward signals so a scancel or walltime SIGTERM kills torchrun cleanly instead of orphaning it.
handle_signal() {
    echo "[chain] Caught signal, forwarding to training process..."
    if kill -0 "$APP_PID" 2>/dev/null; then
        kill -TERM "$APP_PID" || true
        wait "$APP_PID" || true
    fi
    exit 0
}
trap handle_signal TERM INT

wait "$APP_PID"
STATUS=$?
echo "[chain] Training exited with status $STATUS"

# If training finished cleanly (hit max_steps), stop the chain and cancel the queued successor.
if [[ $STATUS -eq 0 && -n "${SLURM_JOB_ID:-}" ]]; then
    touch "$STOP_FILE"
    if [[ -n "$NEXT_JOB_FILE" && -f "$NEXT_JOB_FILE" ]]; then
        scancel "$(cat "$NEXT_JOB_FILE")" 2>/dev/null || true
    fi
fi

exit "$STATUS"
