#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate dreamer

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"

PROJECT="continual_mfrl_size1m"
BASE_LOGDIR="${BASE_LOGDIR:-logdir}"
TASKS_STR="walker_run,hopper_hop,fish_swim"
OBS_DIM=24
ACT_DIM=6
TASK_STEPS=1000000
TASK_REPEATS=5
START_TRAINING="${START_TRAINING:-10000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
UTD="${UTD:-1}"
NUM_ENVS="${NUM_ENVS:-8}"
EVAL_INTERVAL="${EVAL_INTERVAL:-20000}"
LOG_INTERVAL="${LOG_INTERVAL:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
MODEL_SIZE="size1m"
DEFAULT_LR="4e-5"
SEEDS=(${SEEDS_OVERRIDE:-1000 2000 3000})
MAX_RUNS_PER_GPU="${MAX_RUNS_PER_GPU:-1}"
GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-6000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-70}"
RAM_MIN_AVAILABLE_MB="${RAM_MIN_AVAILABLE_MB:-24000}"
LAUNCH_STAGGER_SECONDS="${LAUNCH_STAGGER_SECONDS:-30}"
SCHED_SLEEP_SECONDS="${SCHED_SLEEP_SECONDS:-60}"

if [[ -n "${CUDA_DEVICES_OVERRIDE:-}" ]]; then
  CUDA_DEVICES=(${CUDA_DEVICES_OVERRIDE})
else
  mapfile -t CUDA_DEVICES < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
fi
if (( ${#CUDA_DEVICES[@]} == 0 )); then
  echo "No CUDA devices found." >&2
  exit 1
fi

JOB_SET="${JOB_SET:-priority}"
case "$JOB_SET" in
  priority)
    declare -a JOB_GROUPS=(
      "no_wsc_lr_default|disabled|${DEFAULT_LR}"
      "wsc_skip_last_layer_dout_all|wsc_skip_last_layer_dout_all|${DEFAULT_LR}"
    )
    ;;
  lr_sweep)
    declare -a JOB_GROUPS=(
      "no_wsc_lr_div10|disabled|4e-6"
      "no_wsc_lr_x10|disabled|4e-4"
    )
    ;;
  all)
    declare -a JOB_GROUPS=(
      "no_wsc_lr_default|disabled|${DEFAULT_LR}"
      "no_wsc_lr_div10|disabled|4e-6"
      "no_wsc_lr_x10|disabled|4e-4"
      "wsc_skip_last_layer_dout_all|wsc_skip_last_layer_dout_all|${DEFAULT_LR}"
    )
    ;;
  *)
    echo "Unknown JOB_SET=$JOB_SET; expected priority, lr_sweep, or all." >&2
    exit 1
    ;;
esac

declare -a PIDS=()
declare -A PID_LOGDIR=()
declare -A ACTIVE_GPU_COUNTS=()
declare -a FAILED=()
for gpu in "${CUDA_DEVICES[@]}"; do
  ACTIVE_GPU_COUNTS[$gpu]=0
done

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

ram_available_mb() {
  free -m | awk '/^Mem:/ {print $7}'
}

gpu_free_mb() {
  local gpu="$1"
  nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' '
}

gpu_util() {
  local gpu="$1"
  nvidia-smi --id="$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1 | tr -d ' '
}

gpu_active_repo_runs() {
  local gpu="$1"
  local count=0
  local pid_file
  while IFS= read -r pid_file; do
    local logdir
    logdir="$(dirname "$pid_file")"
    local gpu_file="$logdir/gpu_id"
    [[ -f "$gpu_file" ]] || continue
    [[ "$(<"$gpu_file")" == "$gpu" ]] || continue
    local pid
    pid="$(<"$pid_file")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      count=$((count + 1))
    fi
  done < <(find "${BASE_LOGDIR}/${PROJECT}" -mindepth 3 -maxdepth 3 -name pid 2>/dev/null)
  echo "$count"
}

cleanup_finished() {
  local next=()
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      next+=("$pid")
      continue
    fi
    local logdir="${PID_LOGDIR[$pid]}"
    local gpu_file="$logdir/gpu_id"
    local gpu="unknown"
    [[ -f "$gpu_file" ]] && gpu="$(<"$gpu_file")"
    if [[ "$gpu" != "unknown" && -n "${ACTIVE_GPU_COUNTS[$gpu]+x}" ]]; then
      ACTIVE_GPU_COUNTS[$gpu]=$(( ACTIVE_GPU_COUNTS[$gpu] - 1 ))
    fi
    if wait "$pid"; then
      touch "$logdir/scheduler_done"
      echo "$(timestamp) DONE $logdir"
    else
      touch "$logdir/scheduler_failed"
      FAILED+=("$logdir")
      echo "$(timestamp) FAILED $logdir"
    fi
    unset "PID_LOGDIR[$pid]"
  done
  PIDS=("${next[@]}")
}

select_gpu() {
  local best_gpu=""
  local best_free=-1
  for gpu in "${CUDA_DEVICES[@]}"; do
    local active_count
    active_count="$(gpu_active_repo_runs "$gpu")"
    if (( active_count >= MAX_RUNS_PER_GPU )); then
      continue
    fi
    local free_mb
    free_mb="$(gpu_free_mb "$gpu" || echo 0)"
    local util
    util="$(gpu_util "$gpu" || echo 100)"
    if (( free_mb >= GPU_MIN_FREE_MB && util <= GPU_MAX_UTIL && free_mb > best_free )); then
      best_free="$free_mb"
      best_gpu="$gpu"
    fi
  done
  echo "$best_gpu"
}

wait_for_capacity() {
  while true; do
    cleanup_finished
    local ram_mb
    ram_mb="$(ram_available_mb)"
    local gpu
    gpu="$(select_gpu)"
    if [[ -n "$gpu" && "$ram_mb" -ge "$RAM_MIN_AVAILABLE_MB" ]]; then
      echo "$gpu"
      return
    fi
    echo "$(timestamp) WAIT ram_available_mb=$ram_mb gpu_candidate=${gpu:-none}" >&2
    sleep "$SCHED_SLEEP_SECONDS"
  done
}

launch_one() {
  local group="$1"
  local mechanism="$2"
  local lr="$3"
  local seed="$4"
  local gpu
  gpu="$(wait_for_capacity)"
  local run="seed_${seed}"
  local logdir="${BASE_LOGDIR}/${PROJECT}/${group}/${run}"
  mkdir -p "$logdir"
  if [[ -f "$logdir/pid" ]]; then
    local old_pid
    old_pid="$(<"$logdir/pid")"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "$(timestamp) SKIP active pid=$old_pid logdir=$logdir"
      return
    fi
  fi
  if [[ -f "$logdir/scheduler_done" ]]; then
    echo "$(timestamp) SKIP completed $logdir"
    return
  fi
  if [[ -s "$logdir/train.log" ]]; then
    mv "$logdir/train.log" "$logdir/train.$(date +%Y%m%d_%H%M%S).log"
  fi
  echo "$gpu" > "$logdir/gpu_id"
  echo "$(timestamp) LAUNCH gpu=$gpu group=$group seed=$seed lr=$lr mechanism=$mechanism logdir=$logdir"
  CUDA_VISIBLE_DEVICES="$gpu" \
  python examples/train_continual_dreamer_dist.py \
    --config=./examples/configs/continual_sac_dreamer_dist.py \
    --tasks="$TASKS_STR" \
    --obs_dim="$OBS_DIM" \
    --act_dim="$ACT_DIM" \
    --task_steps="$TASK_STEPS" \
    --task_repeats="$TASK_REPEATS" \
    --start_training="$START_TRAINING" \
    --batch_size="$BATCH_SIZE" \
    --utd="$UTD" \
    --num_envs="$NUM_ENVS" \
    --eval_interval="$EVAL_INTERVAL" \
    --log_interval="$LOG_INTERVAL" \
    --eval_episodes="$EVAL_EPISODES" \
    --vd_mode=disabled \
    --save_dir="$logdir" \
    --seed="$seed" \
    --wandb=True \
    --config.model_size="$MODEL_SIZE" \
    --config.actor_lr="$lr" \
    --config.critic_lr="$lr" \
    --config.temp_lr="$lr" \
    --config.opt.optimizer=adam \
    --config.redo.redo_enabled=True \
    --config.redo.grad_redo_enabled=True \
    --config.redo.tau=0.1 \
    --config.redo.frequency=1000 \
    --config.redo.log_item=log+erank+srank \
    --config.redo.skip_last_layer=True \
    --config.wsc.mechanism="$mechanism" \
    --config.wsc.target=all \
    > "$logdir/train.log" 2>&1 &
  local pid=$!
  PIDS+=("$pid")
  PID_LOGDIR[$pid]="$logdir"
  ACTIVE_GPU_COUNTS[$gpu]=$(( ACTIVE_GPU_COUNTS[$gpu] + 1 ))
  echo "$pid" > "$logdir/pid"
  sleep "$LAUNCH_STAGGER_SECONDS"
}

echo "Scheduler started at $(timestamp)"
echo "Project: $PROJECT"
echo "Job set: $JOB_SET"
echo "Tasks: $TASKS_STR task_steps=$TASK_STEPS repeats=$TASK_REPEATS"
echo "GPUs: ${CUDA_DEVICES[*]} max_runs_per_gpu=$MAX_RUNS_PER_GPU gpu_min_free_mb=$GPU_MIN_FREE_MB gpu_max_util=$GPU_MAX_UTIL"
echo "RAM min available MB: $RAM_MIN_AVAILABLE_MB"

for spec in "${JOB_GROUPS[@]}"; do
  IFS='|' read -r group mechanism lr <<< "$spec"
  for seed in "${SEEDS[@]}"; do
    launch_one "$group" "$mechanism" "$lr" "$seed"
  done
done

while (( ${#PIDS[@]} > 0 )); do
  sleep "$SCHED_SLEEP_SECONDS"
  cleanup_finished
  echo "$(timestamp) MONITOR active=${#PIDS[@]} failed=${#FAILED[@]}"
done

if (( ${#FAILED[@]} > 0 )); then
  echo "Failed runs:"
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
echo "All scheduled runs finished at $(timestamp)"
