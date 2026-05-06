#!/bin/bash

# ============= Configuration =============
cd /home/jiale/MBRL/ContinualMBRL-examine

# Available CUDA devices (modify as needed)
CUDA_DEVICES=(3 4 5)  # Modify to your available GPUs

# Maximum runs per GPU
MAX_RUNS_PER_GPU=2  # Adjust based on GPU memory

# Task string (same for all settings)
TASK_STRING="finger_spin|walker_walk|cheetah_run|reacher_easy"

# Prefix for log directories
PREFIX="continual_sac_dreamer_together"

# Model configuration
MODEL_SIZE="size1m"  # Options: size1m, size12m, size50m, etc.

# Base log directory
BASE_LOGDIR_ROOT="logdir"

# Training configuration
TRAIN_RATIO=256
TASK_INTERVAL=200000  # Match VDRL task_steps=200000

# ============= Settings Definition =============
# Format: "task_type|seed"
declare -a SETTINGS=(
    "mb_disabled|1000"
    "mb_disabled|2000"
    "mb_disabled|3000"

    "mb_disabled_reset_all|1000"
    "mb_disabled_reset_all|2000"
    "mb_disabled_reset_all|3000"
)

# ============= Initialize =============
run_counter=0
TOTAL_CAPACITY=$((${#CUDA_DEVICES[@]} * MAX_RUNS_PER_GPU))
TOTAL_RUNS=${#SETTINGS[@]}

declare -a FAILED_DEPLOYMENTS=()

# ============= Run Experiments =============
echo "============================================"
echo "Starting Continual DreamerV3 Priori Experiments"
echo "============================================"
echo "Total runs requested: $TOTAL_RUNS"
echo "Total GPU capacity: $TOTAL_CAPACITY (${#CUDA_DEVICES[@]} GPUs x $MAX_RUNS_PER_GPU runs/GPU)"
echo "Using GPUs: ${CUDA_DEVICES[@]}"
echo "Model size: $MODEL_SIZE"
echo "Tasks: finger_spin|walker_walk|cheetah_run|reacher_easy"
echo ""

if [ $TOTAL_RUNS -gt $TOTAL_CAPACITY ]; then
    echo "WARNING: Total runs ($TOTAL_RUNS) exceeds GPU capacity ($TOTAL_CAPACITY)"
    echo "Only the first $TOTAL_CAPACITY experiments will be deployed"
    echo ""
fi

# Iterate over all settings
for setting_spec in "${SETTINGS[@]}"; do
    IFS='|' read -r task_type seed <<< "$setting_spec"

    # Check if we've reached capacity
    if [ $run_counter -ge $TOTAL_CAPACITY ]; then
        FAILED_DEPLOYMENTS+=("$setting_spec")
        echo "SKIPPED: $task_type with seed $seed (GPU capacity reached)"
        run_counter=$((run_counter + 1))
        continue
    fi

    # Get the task string for this type
    # (TASK_STRING is fixed; all settings share the same task sequence)

    # Determine which GPU to use
    gpu_idx=$((run_counter / MAX_RUNS_PER_GPU))
    device_num=${CUDA_DEVICES[$gpu_idx]}

    # Create log directory
    logdir="$BASE_LOGDIR_ROOT/${PREFIX}_${MODEL_SIZE}/${task_type}/seed_$seed"
    mkdir -p "$logdir"

    # Construct command using continual_dmc_priori config
    reset_flag="False"
    if [[ "$task_type" == *"reset_all"* ]]; then
        reset_flag="True"
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
        --seed "$seed"
        --egl_device "$device_num"
    )

    # Execute
    echo "[$((run_counter + 1))/$TOTAL_RUNS] Launching: $task_type seed $seed -> GPU $device_num"
    echo "   Task order: $TASK_STRING"
    echo "   Config: continual_dmc_priori $MODEL_SIZE  reset_on_switch=$reset_flag"
    echo "   Logdir: $logdir"
    CUDA_VISIBLE_DEVICES=$device_num "${cmd_args[@]}" > "$logdir/train.log" 2>&1 &

    run_counter=$((run_counter + 1))
    sleep 3
    echo ""
done

echo ""
echo "============================================"
echo "Deployment Summary"
echo "============================================"
echo "Successfully deployed: $((run_counter < TOTAL_CAPACITY ? run_counter : TOTAL_CAPACITY)) / $TOTAL_RUNS"

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
