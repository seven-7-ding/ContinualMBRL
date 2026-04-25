#!/bin/bash

# ============= Configuration =============
cd /home/jiale/MBRL/ContinualMBRL-examine

# Available CUDA devices (modify as needed)
CUDA_DEVICES=(0 1 2 3 4 5)  # Modify to your available GPUs

# Maximum runs per GPU
MAX_RUNS_PER_GPU=1  # Adjust based on GPU memory

# Task definitions
# less_to_more: from easier to harder tasks
# more_to_less: from harder to easier tasks  
# random: random task order
declare -A TASK_STRINGS=(
    # ["less_to_more"]="finger_spin|reacher_hard|hopper_hop|fish_swim|walker_walk"
    # ["more_to_less"]="walker_walk|fish_swim|hopper_hop|reacher_hard|finger_spin"
    # ["random"]="finger_spin|hopper_hop|reacher_hard|walker_walk|fish_swim"
    ["less_to_more"]="pendulum_swingup|finger_spin|walker_walk"
    ["more_to_less"]="walker_walk|finger_spin|pendulum_swingup"
    ["random"]="finger_spin|pendulum_swingup|walker_walk|"
)
# Prefix for log directories
PREFIX="examine_dreamer_continual_simple_short"

# Model configuration
MODEL_SIZE="size50m"  # Options: size1m, size12m, size50m, etc.

# Base log directory
BASE_LOGDIR_ROOT="logdir"

# Training configuration
TRAIN_RATIO=256
TASK_INTERVAL=200000  # Modify as needed

# ============= Settings Definition =============
# Format: "task_type|seed"
# Each entry will be assigned to a GPU independently
declare -a SETTINGS=(
    "less_to_more|1000"
    "less_to_more|2000"
    "less_to_more|3000"
    
    # "more_to_less|1000"
    # "more_to_less|2000"
    # "more_to_less|3000"
    
    # "random|1000"
    # "random|2000"
    # "random|3000"
)

# ============= Initialize =============
run_counter=0
TOTAL_CAPACITY=$((${#CUDA_DEVICES[@]} * MAX_RUNS_PER_GPU))
TOTAL_RUNS=${#SETTINGS[@]}

declare -a FAILED_DEPLOYMENTS=()

# ============= Run Experiments =============
echo "============================================"
echo "Starting Continual DreamerV3 Experiments"
echo "============================================"
echo "Total runs requested: $TOTAL_RUNS"
echo "Total GPU capacity: $TOTAL_CAPACITY (${#CUDA_DEVICES[@]} GPUs × $MAX_RUNS_PER_GPU runs/GPU)"
echo "Using GPUs: ${CUDA_DEVICES[@]}"
echo "Model size: $MODEL_SIZE"
echo ""

if [ $TOTAL_RUNS -gt $TOTAL_CAPACITY ]; then
    echo "⚠️  WARNING: Total runs ($TOTAL_RUNS) exceeds GPU capacity ($TOTAL_CAPACITY)"
    echo "⚠️  Only the first $TOTAL_CAPACITY experiments will be deployed"
    echo ""
fi

# Iterate over all settings
for setting_spec in "${SETTINGS[@]}"; do
    IFS='|' read -r task_type seed <<< "$setting_spec"
    
    # Check if we've reached capacity
    if [ $run_counter -ge $TOTAL_CAPACITY ]; then
        FAILED_DEPLOYMENTS+=("$setting_spec")
        echo "❌ [$((run_counter + 1))/$TOTAL_RUNS] SKIPPED: $task_type with seed $seed (GPU capacity reached)"
        run_counter=$((run_counter + 1))
        continue
    fi
    
    # Get the task string for this type
    TASK_STRING="${TASK_STRINGS[$task_type]}"
    
    # Determine which GPU to use
    gpu_idx=$((run_counter / MAX_RUNS_PER_GPU))
    device_num=${CUDA_DEVICES[$gpu_idx]}
    
    # Create log directory
    logdir="$BASE_LOGDIR_ROOT/${PREFIX}_${MODEL_SIZE}/${task_type}/seed_$seed"
    mkdir -p "$logdir"
    
    # Construct command
    cmd="CUDA_VISIBLE_DEVICES=$device_num python dreamerv3/main.py \
        --configs continual_dmc_vision $MODEL_SIZE \
        --task \"$TASK_STRING\" \
        --logdir $logdir \
        --run.train_ratio $TRAIN_RATIO \
        --run.task_interval $TASK_INTERVAL \
        --seed $seed \
        --egl_device $device_num \
        > $logdir/train.log 2>&1 &"
    
    # Execute
    echo "✅ [$((run_counter + 1))/$TOTAL_RUNS] Launching: $task_type with seed $seed on GPU $device_num"
    echo "   Task order: $TASK_STRING"
    echo "   Logdir: $logdir"
    eval $cmd
    
    # Increment counter
    run_counter=$((run_counter + 1))
    
    # Sleep to avoid overwhelming the system
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
    echo ""
    echo "⚠️  The following experiments were NOT deployed due to GPU capacity limits:"
    echo ""
    for failed in "${FAILED_DEPLOYMENTS[@]}"; do
        IFS='|' read -r task_type seed <<< "$failed"
        echo "   - Task Type: $task_type, Seed: $seed"
    done
    echo ""
    echo "💡 Solutions:"
    echo "   1. Increase MAX_RUNS_PER_GPU (currently: $MAX_RUNS_PER_GPU)"
    echo "   2. Add more GPUs to CUDA_DEVICES (currently: ${#CUDA_DEVICES[@]})"
    echo "   3. Run the failed experiments separately"
else
    echo "All experiments deployed successfully! ✅"
fi

echo ""
echo "============================================"
echo "Monitoring Commands"
echo "============================================"
echo "Monitor all logs:"
echo "  tail -f $BASE_LOGDIR_ROOT/dreamer_continual_*/seed_*/train.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep 'dreamerv3/main.py'"
echo ""
echo "Monitor GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "============================================"
echo "Monitor all logs:"
echo "  tail -f $BASE_LOGDIR_ROOT/dreamer_continual_*/seed_*/train.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep 'dreamerv3/main.py'"
echo ""
echo "Monitor GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""