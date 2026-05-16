#!/bin/bash

# ============= Configuration =============
cd /home/jiale/MBRL/ContinualMBRL-full-loop

# Available CUDA devices for this experiment.
CUDA_DEVICES=(0 1 2 3 4 5 0 1 2 3 4 5)

# Maximum concurrent runs launched by this script on each GPU.
MAX_RUNS_PER_GPU=1

# Task string (same for all settings)
TASK_STRING="finger_spin|walker_walk|cheetah_run|reacher_easy"

# Prefix for log directories
PREFIX="continual_dreamer_full_loop"

# Model configuration
MODEL_SIZE="size1m"  # Options: size0.5m, size1m, size12m, size50m, etc.

# Base log directory
BASE_LOGDIR_ROOT="logdir"

# Training configuration
TRAIN_RATIO=1024
TASK_INTERVAL=200000  # Match VDRL task_steps=200000

# ============= Settings Definition =============
# Format: "task_type|seed"
declare -a SETTINGS=(
    "noreset_no_actredo_no_gradredo|1000"
    "noreset_no_actredo_no_gradredo|2000"
    "noreset_no_actredo_no_gradredo|3000"

    "reset_all_no_actredo_no_gradredo|1000"
    "reset_all_no_actredo_no_gradredo|2000"
    "reset_all_no_actredo_no_gradredo|3000"

    "noreset_actredo_no_gradredo|1000"
    "noreset_actredo_no_gradredo|2000"
    "noreset_actredo_no_gradredo|3000"

    "noreset_no_actredo_gradredo|1000"
    "noreset_no_actredo_gradredo|2000"
    "noreset_no_actredo_gradredo|3000"
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
echo ""

# Iterate over all settings
for setting_spec in "${SETTINGS[@]}"; do
    IFS='|' read -r task_type seed <<< "$setting_spec"

    # Create log directory
    logdir="$BASE_LOGDIR_ROOT/${PREFIX}_${MODEL_SIZE}/${task_type}/seed_$seed"
    if [ -s "$logdir/train.log" ]; then
        FAILED_DEPLOYMENTS+=("$setting_spec")
        echo "SKIPPED: $task_type with seed $seed (existing log: $logdir/train.log)"
        continue
    fi
    mkdir -p "$logdir"

    # Assign GPU by sequential index (no waiting).
    device_num="${CUDA_DEVICES[$run_counter % ${#CUDA_DEVICES[@]}]}"

    # Task-switch reset configuration.
    reset_flag="False"
    reset_mode="none"
    if [[ "$task_type" == "reset_all_"* ]]; then
        reset_flag="True"
        reset_mode="all"
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
        --run.reset_on_switch "$reset_flag"
        --run.reset_mode "$reset_mode"
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
    echo "   Config: continual_dmc_priori $MODEL_SIZE  reset_on_switch=$reset_flag reset_mode=$reset_mode"
    echo "   ReDo: act_log_item=$act_log_item grad_log_item=$grad_log_item"
    echo "   Logdir: $logdir"
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$device_num "${cmd_args[@]}" > "$logdir/train.log" 2>&1 &
    pid=$!

    run_counter=$((run_counter + 1))
    sleep 6
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
