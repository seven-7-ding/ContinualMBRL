#!/bin/bash

# ============= Configuration =============
cd /home/jiale/MBRL/MF-dreamer

# Available CUDA devices (modify as needed)
CUDA_DEVICES=(6 7 6 7 6 7)  # Modify to your available GPUs

# Maximum runs per GPU
MAX_RUNS_PER_GPU=1  # Adjust based on GPU memory

# Training configuration
PREFIX="continual_dreamer_env"

# Continual task settings
TASKS_STR="finger_spin,walker_walk,cheetah_run,reacher_easy"
OBS_DIM=32
ACT_DIM=6
TASK_STEPS=200000      # individual transitions per task
TASK_REPEATS=10
START_TRAINING=10000
SIZE="size1m"          # DreamerV3 size1m: 3x64 SiLU+RMSNorm

# Parallel env workers
NUM_ENVS=16
NUM_IMAG_ENVS=64       # envs used for imagination rollouts

# ReDo settings
REDO_ENABLED=True
REDO_LOG_ITEM="log+erank+srank"
REDO_FREQUENCY=10000

# Base log directory
BASE_LOGDIR=./logdir

# UTD values to sweep
UTD=(1)

# ============= Settings Definition =============
# Format: "vd_mode|seed"
declare -a SETTINGS=(
    "disabled|1000"
    "disabled|2000"
    "disabled|3000"

    "reset_all|1000"
    "reset_all|2000"
    "reset_all|3000"
)

# ============= Initialize =============
run_counter=0
TOTAL_CAPACITY=$((${#CUDA_DEVICES[@]} * MAX_RUNS_PER_GPU))
TOTAL_RUNS=$(( ${#SETTINGS[@]} * ${#UTD[@]} ))

declare -a FAILED_DEPLOYMENTS=()

# ============= Run Experiments =============
echo "============================================"
echo "Starting Continual DreamerEnv Experiments"
echo "============================================"
echo "Total runs requested: $TOTAL_RUNS"
echo "Total GPU capacity:   $TOTAL_CAPACITY (${#CUDA_DEVICES[@]} GPUs x $MAX_RUNS_PER_GPU runs/GPU)"
echo "Using GPUs:           ${CUDA_DEVICES[@]}"
echo "Tasks:                $TASKS_STR"
echo "Network:              3x64 SiLU+RMSNorm (DreamerV3 $SIZE)"
echo "TwoHot bins:          255 (symexp-spaced +-20)"
echo "Parallel envs:        $NUM_ENVS  |  imag envs: $NUM_IMAG_ENVS"
echo "UTD values:           ${UTD[@]}"
echo "task_steps:           $TASK_STEPS  |  task_repeats: $TASK_REPEATS"
echo ""

if [ $TOTAL_RUNS -gt $TOTAL_CAPACITY ]; then
    echo "WARNING: Total runs ($TOTAL_RUNS) exceeds GPU capacity ($TOTAL_CAPACITY)"
    echo "Only the first $TOTAL_CAPACITY experiments will be deployed"
    echo ""
fi

for setting_spec in "${SETTINGS[@]}"; do
  for utd in "${UTD[@]}"; do
    IFS='|' read -r vd_mode seed <<< "$setting_spec"
    combined="${vd_mode}|${seed}|${utd}"

    # Check capacity
    if [ $run_counter -ge $TOTAL_CAPACITY ]; then
        FAILED_DEPLOYMENTS+=("$combined")
        echo "SKIPPED: $vd_mode / seed $seed / utd $utd (GPU capacity reached)"
        run_counter=$((run_counter + 1))
        continue
    fi

    # Assign GPU
    gpu_idx=$((run_counter / MAX_RUNS_PER_GPU))
    device_num=${CUDA_DEVICES[$gpu_idx]}

    # Log directory
    logdir="${BASE_LOGDIR}/${PREFIX}_${SIZE}/${vd_mode}_utd_${utd}_envs_${NUM_ENVS}/seed_${seed}"
    mkdir -p "$logdir"

    echo "[$((run_counter + 1))/$TOTAL_RUNS] Launching: $vd_mode | seed $seed | utd $utd | envs $NUM_ENVS -> GPU $device_num"
    echo "   Logdir: $logdir"

    CUDA_VISIBLE_DEVICES=$device_num \
    XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" \
    python examples/train_continual_dreamer_env.py \
        --config=./examples/configs/continual_dreamer_env.py \
        --tasks=${TASKS_STR} \
        --obs_dim=${OBS_DIM} \
        --act_dim=${ACT_DIM} \
        --task_steps=${TASK_STEPS} \
        --task_repeats=${TASK_REPEATS} \
        --start_training=${START_TRAINING} \
        --num_envs=${NUM_ENVS} \
        --num_imag_envs=${NUM_IMAG_ENVS} \
        --vd_mode=${vd_mode} \
        --save_dir=${logdir} \
        --seed=${seed} \
        --utd=${utd} \
        --config.model_size=${SIZE} \
        --config.redo.redo_enabled=${REDO_ENABLED} \
        --config.redo.log_item=${REDO_LOG_ITEM} \
        --config.redo.frequency=${REDO_FREQUENCY} \
        > ${logdir}/train.log 2>&1 &

    run_counter=$((run_counter + 1))
    sleep 10   # avoid GPU/ptxas resource contention during JIT compilation
    echo ""
  done
done

echo ""
echo "============================================"
echo "Deployment complete."
echo "Launched: $run_counter / $TOTAL_RUNS runs"
if [ ${#FAILED_DEPLOYMENTS[@]} -gt 0 ]; then
    echo ""
    echo "SKIPPED (GPU capacity exceeded):"
    for item in "${FAILED_DEPLOYMENTS[@]}"; do
        echo "  $item"
    done
fi
echo "============================================"
