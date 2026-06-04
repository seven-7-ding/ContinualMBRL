#!/bin/bash

# ============= Configuration =============
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Available CUDA devices for this experiment.
CUDA_DEVICES=(2 2 3 3 3 4 4 4 5 5 6 6 7 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7)

# Maximum concurrent runs launched by this script on each GPU.
MAX_RUNS_PER_GPU=1

# Task string (same for all settings)
TASK_STRING="walker_run|hopper_hop|fish_swim"

# Prefix for log directories
PREFIX="continual_dreamer_reset_partial"

# Model configuration
MODEL_SIZE="size1m"  # Options: size0.5m, size1m, size12m, size50m, etc.

# Base log directory
BASE_LOGDIR_ROOT="logdir"

# Training configuration
TRAIN_RATIO=1024
TASK_INTERVAL=1000000
RESET_FREQUENCY=250000
# Set REVIVE_EPOCH=0 to disable revive while still keeping periodic reset.
REVIVE_EPOCH=0
REVIVE_STRATEGY=threshold
RESET_FREQUENCY_K="$((RESET_FREQUENCY / 1000))k"
if (( REVIVE_EPOCH > 0 )); then
    RESET_FREQUENCY_TAG="${RESET_FREQUENCY_K}_1w_revive"
else
    RESET_FREQUENCY_TAG="${RESET_FREQUENCY_K}_1w_no_revive"
fi

# ============= Settings Definition =============
# Format: "task_type|seed"
declare -a SETTINGS=(
    # "no_reset|1000"
    # "no_reset|2000"
    # "no_reset|3000"
    # "no_reset|4000"
    # "no_reset|5000"
    # # "no_reset|6000"
    
    "reset_only_agent|1000"
    "reset_only_agent|2000"
    "reset_only_agent|3000"
    "reset_only_agent|4000"
    "reset_only_agent|5000"
    # "reset_only_agent|6000"

    "reset_only_wm|1000"
    "reset_only_wm|2000"
    "reset_only_wm|3000"
    "reset_only_wm|4000"
    "reset_only_wm|5000"
    # "reset_only_wm|6000"

    "reset_only_rssm|1000"
    "reset_only_rssm|2000"
    "reset_only_rssm|3000"
    "reset_only_rssm|4000"
    "reset_only_rssm|5000"
    # "reset_only_rssm|6000"

    "reset_all_heads|1000"
    "reset_all_heads|2000"
    "reset_all_heads|3000"
    "reset_all_heads|4000"
    "reset_all_heads|5000"
    # "reset_all_heads|6000"

    "reset_agent_heads|1000"
    "reset_agent_heads|2000"
    "reset_agent_heads|3000"
    "reset_agent_heads|4000"
    "reset_agent_heads|5000"
    # "reset_agent_heads|6000"

    "reset_wm_heads|1000"
    "reset_wm_heads|2000"
    "reset_wm_heads|3000"
    "reset_wm_heads|4000"
    "reset_wm_heads|5000"
    # "reset_wm_heads|6000"

    # "reset_all|1000"
    # "reset_all|2000"
    # "reset_all|3000"
    # "reset_all|4000"
    # "reset_all|5000"
    # "reset_all|6000"
)

# ============= Initialize =============
run_counter=0
TOTAL_RUNS=${#SETTINGS[@]}

declare -a FAILED_DEPLOYMENTS=()

# ============= Run Experiments =============
echo "============================================"
echo "Starting Continual DreamerV3 Priori Data Diversity Experiments"
echo "============================================"
echo "Total runs requested: $TOTAL_RUNS"
echo "Using GPUs: ${CUDA_DEVICES[@]}"
echo "Max runs per GPU launched by this script: $MAX_RUNS_PER_GPU"
echo "Model size: $MODEL_SIZE"
echo "Tasks: $TASK_STRING"
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
    IFS='|' read -r task_type seed <<< "$setting_spec"

    # Create log directory
    logdir="$BASE_LOGDIR_ROOT/${PREFIX}_${MODEL_SIZE}/${task_type}_${RESET_FREQUENCY_TAG}/seed_$seed"
    if [ -s "$logdir/train.log" ]; then
        FAILED_DEPLOYMENTS+=("$setting_spec")
        echo "SKIPPED: $task_type with seed $seed (existing log: $logdir/train.log)"
        continue
    fi
    mkdir -p "$logdir"

    # Assign GPU by sequential index (no waiting).
    device_num="${CUDA_DEVICES[$run_counter % ${#CUDA_DEVICES[@]}]}"

    # Periodic reset configuration.
    reset_mode="$task_type"
    if [[ "$task_type" == "no_reset" ]]; then
        reset_mode="no_reset"
    fi

    # ReDo analysis is always enabled. The per-analyser log item controls
    # whether that analyser performs layer reset or only logs analysis metrics.
    act_log_item="log+erank+srank"
    grad_log_item="log+erank+srank"
    if [[ "$task_type" == "noreset_actredo_no_gradredo" ]]; then
        act_log_item="reset+erank+srank"
    fi
    if [[ "$task_type" == "noreset_no_actredo_gradredo" ]]; then
        grad_log_item="reset+erank+srank"
    fi

    # Use array to avoid leading-space issues from multi-line string quoting.
    cmd_args=(
        python dreamerv3/main.py
        --configs continual_dmc_priori "$MODEL_SIZE"
        --task "$TASK_STRING"
        --logdir "$logdir"
        --run.train_ratio "$TRAIN_RATIO"
        --run.task_interval "$TASK_INTERVAL"
        --run.reset_mode "$reset_mode"
        --run.reset_frequency "$RESET_FREQUENCY"
        --run.revive_epoch "$REVIVE_EPOCH"
        --run.revive_strategy "$REVIVE_STRATEGY"
        --seed "$seed"
        --egl_device "$device_num"
        --agent.imag_length 15
        --agent.redo.redo_enabled True
        --agent.redo.grad_redo_enabled True
        --agent.redo.act_log_item "$act_log_item"
        --agent.redo.grad_log_item "$grad_log_item"
    )

    # Execute
    echo "[$((run_counter + 1))/$TOTAL_RUNS] Launching: $task_type seed $seed -> GPU $device_num"
    echo "   Task order: $TASK_STRING"
    echo "   Config: continual_dmc_priori $MODEL_SIZE  reset_mode=$reset_mode reset_frequency=$RESET_FREQUENCY revive_epoch=$REVIVE_EPOCH"
    echo "   ReDo: act_log_item=$act_log_item grad_log_item=$grad_log_item"
    echo "   Logdir: $logdir"
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$device_num "${cmd_args[@]}" > "$logdir/train.log" 2>&1 &
    pid=$!

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
        IFS='|' read -r task_type seed <<< "$failed"
        echo "   - Task Type: $task_type, Seed: $seed"
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
