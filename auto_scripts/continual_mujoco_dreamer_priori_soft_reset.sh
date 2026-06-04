#!/bin/bash

# ============= Configuration =============
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Available CUDA devices for this experiment.
CUDA_DEVICES=(0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7)

# Maximum concurrent runs launched by this script on each GPU.
MAX_RUNS_PER_GPU=1

# Task string (same for all settings)
TASK_STRING="walker_run|hopper_hop|fish_swim"

# Prefix for log directories
PREFIX="continual_dreamer_soft_reset"

# Model configuration
MODEL_SIZE="size1m"  # Options: size0.5m, size1m, size12m, size50m, etc.

# Base log directory
BASE_LOGDIR_ROOT="logdir"

# Training configuration
TRAIN_RATIO=1024
TASK_INTERVAL=1000000
RESET_FREQUENCY=50000
RESET_MECHANISM="sandp"  # Options: hard, sandp, merge
RESET_ALPHA=0.8

# Set REVIVE_EPOCH=0 to disable revive while still keeping periodic reset.
REVIVE_EPOCH=0
REVIVE_STRATEGY="threshold"

# Other hyper-parameters
AGENT_IMAG_LENGTH=15
REDO_ENABLED=True
GRAD_REDO_ENABLED=True
ACT_LOG_ITEM="log+erank+srank"
GRAD_LOG_ITEM="log+erank+srank"

# Optional extra CLI args, for example:
# EXTRA_ARGS="--run.log_every 5000 --batch_size 8"
EXTRA_ARGS=""

RESET_FREQUENCY_K="$((RESET_FREQUENCY / 1000))k"
RESET_ALPHA_TAG="${RESET_ALPHA//./p}"
if (( REVIVE_EPOCH > 0 )); then
    RESET_FREQUENCY_TAG="${RESET_FREQUENCY_K}_revive_${REVIVE_EPOCH}_${REVIVE_STRATEGY}"
else
    RESET_FREQUENCY_TAG="${RESET_FREQUENCY_K}_no_revive"
fi
# ============= Settings Definition =============
# Format: "reset_target|seed"
declare -a SETTINGS=(
    # "no_reset|1000"
    # "no_reset|2000"
    # "no_reset|3000"
    # "no_reset|4000"
    # "no_reset|5000"

    "agent_head|1000"
    "agent_head|2000"
    "agent_head|3000"
    "agent_head|4000"
    "agent_head|5000"

    "wm_head|1000"
    "wm_head|2000"
    "wm_head|3000"
    "wm_head|4000"
    "wm_head|5000"

    "only_rssm|1000"
    "only_rssm|2000"
    "only_rssm|3000"
    "only_rssm|4000"
    "only_rssm|5000"

    "all_head|1000"
    "all_head|2000"
    "all_head|3000"
    "all_head|4000"
    "all_head|5000"

    "all|1000"
    "all|2000"
    "all|3000"
    "all|4000"
    "all|5000"
)

# ============= Initialize =============
run_counter=0
TOTAL_RUNS=${#SETTINGS[@]}
declare -a FAILED_DEPLOYMENTS=()

# ============= Run Experiments =============
echo "============================================"
echo "Starting Continual DreamerV3 Priori Soft Reset Experiments"
echo "============================================"
echo "Total runs requested: $TOTAL_RUNS"
echo "Using GPUs: ${CUDA_DEVICES[@]}"
echo "Max runs per GPU launched by this script: $MAX_RUNS_PER_GPU"
echo "Model size: $MODEL_SIZE"
echo "Tasks: $TASK_STRING"
echo "Reset mechanism: $RESET_MECHANISM"
if (( RESET_FREQUENCY == TASK_INTERVAL )); then
    echo "Reset schedule: reset once at each task boundary"
else
    echo "Reset schedule: reset every $RESET_FREQUENCY env steps"
fi
if (( REVIVE_EPOCH > 0 )); then
    echo "Revive schedule: up to $REVIVE_EPOCH updates after each reset ($REVIVE_STRATEGY)"
else
    echo "Revive schedule: disabled (REVIVE_EPOCH=0)"
fi
echo ""

# Iterate over all settings
for setting_spec in "${SETTINGS[@]}"; do
    IFS='|' read -r reset_target seed <<< "$setting_spec"

    if [[ "$reset_target" == "no_reset" ]]; then
        reset_tag="no_reset"
    elif [[ "$RESET_MECHANISM" == "hard" ]]; then
        reset_tag="hard_${reset_target}"
    else
        reset_tag="${RESET_MECHANISM}_${reset_target}_a${RESET_ALPHA_TAG}"
    fi

    # Create log directory
    logdir="$BASE_LOGDIR_ROOT/${PREFIX}_${MODEL_SIZE}/${reset_tag}_${RESET_FREQUENCY_TAG}/seed_$seed"
    if [ -s "$logdir/train.log" ]; then
        FAILED_DEPLOYMENTS+=("$setting_spec")
        echo "SKIPPED: $reset_target with seed $seed (existing log: $logdir/train.log)"
        continue
    fi
    mkdir -p "$logdir"

    # Assign GPU by sequential index (no waiting).
    device_num="${CUDA_DEVICES[$run_counter % ${#CUDA_DEVICES[@]}]}"

    # Use array to avoid leading-space issues from multi-line string quoting.
    cmd_args=(
        python dreamerv3/main.py
        --configs continual_dmc_priori "$MODEL_SIZE"
        --task "$TASK_STRING"
        --logdir "$logdir"
        --run.train_ratio "$TRAIN_RATIO"
        --run.task_interval "$TASK_INTERVAL"
        --run.reset_frequency "$RESET_FREQUENCY"
        --run.reset_mechanism "$RESET_MECHANISM"
        --run.reset_target "$reset_target"
        --run.reset_alpha "$RESET_ALPHA"
        --run.revive_epoch "$REVIVE_EPOCH"
        --run.revive_strategy "$REVIVE_STRATEGY"
        --seed "$seed"
        --egl_device "$device_num"
        --agent.imag_length "$AGENT_IMAG_LENGTH"
        --agent.redo.redo_enabled "$REDO_ENABLED"
        --agent.redo.grad_redo_enabled "$GRAD_REDO_ENABLED"
        --agent.redo.act_log_item "$ACT_LOG_ITEM"
        --agent.redo.grad_log_item "$GRAD_LOG_ITEM"
    )

    if [[ -n "$EXTRA_ARGS" ]]; then
        read -r -a extra_args_array <<< "$EXTRA_ARGS"
        cmd_args+=("${extra_args_array[@]}")
    fi

    # Execute
    echo "[$((run_counter + 1))/$TOTAL_RUNS] Launching: $reset_target seed $seed -> GPU $device_num"
    echo "   Task order: $TASK_STRING"
    echo "   Config: continual_dmc_priori $MODEL_SIZE reset_mechanism=$RESET_MECHANISM reset_target=$reset_target reset_alpha=$RESET_ALPHA reset_frequency=$RESET_FREQUENCY revive_epoch=$REVIVE_EPOCH"
    echo "   ReDo: act_log_item=$ACT_LOG_ITEM grad_log_item=$GRAD_LOG_ITEM"
    if [[ -n "$EXTRA_ARGS" ]]; then
        echo "   Extra args: $EXTRA_ARGS"
    fi
    echo "   Logdir: $logdir"
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$device_num "${cmd_args[@]}" > "$logdir/train.log" 2>&1 &

    run_counter=$((run_counter + 1))
    sleep 50
    echo ""
done

echo ""
echo "============================================"
echo "Deployment Summary"
echo "============================================"
echo "Successfully deployed: $run_counter / $TOTAL_RUNS"

if [ ${#FAILED_DEPLOYMENTS[@]} -gt 0 ]; then
    echo "Failed to deploy: ${#FAILED_DEPLOYMENTS[@]} experiments"
    for failed in "${FAILED_DEPLOYMENTS[@]}"; do
        IFS='|' read -r reset_target seed <<< "$failed"
        echo "   - Reset Target: $reset_target, Seed: $seed"
    done
else
    echo "All experiments deployed successfully!"
fi

echo ""
echo "============================================"
echo "Monitoring Commands"
echo "============================================"
echo "  tail -f $BASE_LOGDIR_ROOT/${PREFIX}_${MODEL_SIZE}/*/seed_*/train.log"
echo "  ps aux | grep 'dreamerv3/main.py'"
echo "  watch -n 1 nvidia-smi"
echo ""
