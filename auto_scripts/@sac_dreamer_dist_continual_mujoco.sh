#!/bin/bash

# ============= Configuration =============
cd /home/jiale/MBRL/MF-dreamer
export PYTHONPATH="/home/jiale/MBRL/MF-dreamer:${PYTHONPATH}"

# Available CUDA devices (modify as needed)
CUDA_DEVICES=(6 7 6 7 6 7)  # Modify to your available GPUs

# Maximum runs per GPU
MAX_RUNS_PER_GPU=1  # Adjust based on GPU memory

# Training configuration
PREFIX="continual_sac_dreamer_together"

# Continual task settings
TASKS_STR="finger_spin,walker_walk,cheetah_run,reacher_easy"
OBS_DIM=32
ACT_DIM=6
TASK_STEPS=200000      # individual transitions per task (matches DreamerV3 task_interval=200000)
TASK_REPEATS=10
START_TRAINING=10000
SIZE="size1m"          # DreamerV3 size1m: 3x64 SiLU+RMSNorm

# Parallel env workers for data collection
NUM_ENVS=16
# Imagination env pool (each rollout uses up to NUM_IMAG_ENVS envs from replay states)
NUM_IMAG_ENVS=64

# Imagination horizon (replaces DreamerV3 imag_length)
IMAG_LENGTH=5

# ReDo settings (mirrors dreamerv3/configs.yaml redo defaults)
REDO_ENABLED=True
REDO_LOG_ITEM="log+erank+srank"
REDO_FREQUENCY=10000

# Base log directory
BASE_LOGDIR=./logdir

# UTD values to sweep
UTD=(1)

# ============= Settings Definition =============
# Format: "vd_mode|seed"
# vd_mode: 'disabled' for standard run; 'reset_all' to reset agent on task switch
declare -a SETTINGS=(
    "dreamer_like_disabled|1000"
    "dreamer_like_disabled|2000"
    "dreamer_like_disabled|3000"

    "dreamer_like_disabled_reset_all|1000"
    "dreamer_like_disabled_reset_all|2000"
    "dreamer_like_disabled_reset_all|3000"
)

# ============= Initialize =============
run_counter=0
TOTAL_CAPACITY=$((${#CUDA_DEVICES[@]} * MAX_RUNS_PER_GPU))
TOTAL_RUNS=$(( ${#SETTINGS[@]} * ${#UTD[@]} ))

declare -a FAILED_DEPLOYMENTS=()

# ============= Run Experiments =============
echo "============================================"
echo "Starting Continual DreamerEnvLearner Experiments"
echo "============================================"
echo "Total runs requested: $TOTAL_RUNS"
echo "Total GPU capacity:   $TOTAL_CAPACITY (${#CUDA_DEVICES[@]} GPUs x $MAX_RUNS_PER_GPU runs/GPU)"
echo "Using GPUs:           ${CUDA_DEVICES[@]}"
echo "Tasks:                $TASKS_STR"
echo "Network:              3x64 SiLU+RMSNorm (DreamerV3 $SIZE)"
echo "Value:                TwoHot distributional (255 bins), V(s) not Q(s,a)"
echo "Imagination:          real env rollouts, H=$IMAG_LENGTH, pool=$NUM_IMAG_ENVS"
echo "Collect envs:         $NUM_ENVS"
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
        --config.imag_length=${IMAG_LENGTH} \
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
echo "Deployment Summary"
echo "============================================"
echo "Successfully deployed: $((run_counter < TOTAL_CAPACITY ? run_counter : TOTAL_CAPACITY)) / $TOTAL_RUNS"

if [ ${#FAILED_DEPLOYMENTS[@]} -gt 0 ]; then
    echo "Failed to deploy: ${#FAILED_DEPLOYMENTS[@]} experiments"
    for failed in "${FAILED_DEPLOYMENTS[@]}"; do
        IFS='|' read -r f_mode f_seed f_utd <<< "$failed"
        echo "   - Mode: $f_mode, Seed: $f_seed, UTD: $f_utd"
    done
else
    echo "All experiments deployed successfully!"
fi

echo ""
echo "============================================"
echo "Monitoring Commands"
echo "============================================"
echo "  tail -f ${BASE_LOGDIR}/${PREFIX}_${SIZE}/*/seed_*/train.log"
echo "  ps aux | grep 'train_continual_dreamer_env.py'"
echo "  watch -n 1 nvidia-smi"
echo ""
