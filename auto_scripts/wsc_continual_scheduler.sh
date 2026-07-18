#!/bin/bash

set -euo pipefail
trap 'status=$?; echo "ERROR line=$LINENO status=$status" >&2; exit $status' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .env.wandb.local ]]; then
  set -a
  source .env.wandb.local
  set +a
fi

CUDA_DEVICES=(${CUDA_DEVICES_OVERRIDE:-0 1 2 3 4 5 6 7})
MAX_RUNS_PER_GPU="${MAX_RUNS_PER_GPU:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-$(( ${#CUDA_DEVICES[@]} * MAX_RUNS_PER_GPU ))}"
MODEL_SIZE="${MODEL_SIZE:-size1m}"
BASE_LOGDIR_ROOT="${BASE_LOGDIR_ROOT:-logdir}"
TRAIN_RATIO="${TRAIN_RATIO:-1024}"
REPLAY_CHUNKSIZE="${REPLAY_CHUNKSIZE:-4096}"
REPLAY_CACHE_CHUNKS="${REPLAY_CACHE_CHUNKS:-1024}"
AGENT_IMAG_LENGTH="${AGENT_IMAG_LENGTH:-15}"
REDO_ENABLED="${REDO_ENABLED:-True}"
GRAD_REDO_ENABLED="${GRAD_REDO_ENABLED:-True}"
ACT_LOG_ITEM="${ACT_LOG_ITEM:-log+erank+srank}"
GRAD_LOG_ITEM="${GRAD_LOG_ITEM:-log+erank+srank}"
WSC_TARGET="${WSC_TARGET:-all}"
WSC_TARGET_NORM="${WSC_TARGET_NORM:-1.0}"
WSC_SCALE_FACTOR="${WSC_SCALE_FACTOR:-0.999}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
RUN_SAVE_EVERY="${RUN_SAVE_EVERY:-}"
PHASE="${PHASE:-phase12}"
SEEDS=(${SEEDS_OVERRIDE:-1000 2000 3000})

WSC_MECHANISMS=(
  no_wsc
  WSC_nograd_scale_init
  WSC_grad_scale_init
  WSC_nograd_scale_constant
  WSC_grad_scale_constant
  WSC_nograd_scale_factor
  WSC_grad_scale_factor
)
if [[ -n "${WSC_MECHANISMS_OVERRIDE:-}" ]]; then
  WSC_MECHANISMS=(${WSC_MECHANISMS_OVERRIDE})
fi

TASK_SPECS_PHASE12=(
  "walker_run|hopper_hop|fish_swim::continual_dreamer_soft_reset_${MODEL_SIZE}::1000000"
  "dog_stand|dog_walk|dog_trot::continual_dreamer_soft_reset_dog_stand|dog_walk|dog_trot_${MODEL_SIZE}::2000000"
)

TASK_SPECS_LATER=(
  "humanoid_stand|humanoid_run::continual_dreamer_soft_reset_humanoid_stand|humanoid_run_${MODEL_SIZE}::3000000"
  "quadruped_walk|quadruped_escape|quadruped_fetch::continual_dreamer_soft_reset_quadruped_walk|quadruped_escape|quadruped_fetch_${MODEL_SIZE}::1000000"
)

case "$PHASE" in
  phase12) TASK_SPECS=("${TASK_SPECS_PHASE12[@]}") ;;
  later) TASK_SPECS=("${TASK_SPECS_LATER[@]}") ;;
  all) TASK_SPECS=("${TASK_SPECS_PHASE12[@]}" "${TASK_SPECS_LATER[@]}") ;;
  *) echo "Unknown PHASE=$PHASE; use phase12, later, or all" >&2; exit 2 ;;
esac
if [[ -n "${TASK_SPECS_OVERRIDE:-}" ]]; then
  read -r -a TASK_SPECS <<< "$TASK_SPECS_OVERRIDE"
fi

task_dims() {
  local task_string="$1"
  local dims_json="$REPO_ROOT/embodied/envs/dmc_priori_dims.json"
  python - "$dims_json" "$task_string" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
tasks = [x.strip() for x in sys.argv[2].split('|') if x.strip()]
entries = {item.get('task'): item for item in payload.get('tasks', [])}
matched = []
for task in tasks:
  item = entries.get(task)
  if item and item.get('status') == 'ok':
    matched.append(item)
if not matched:
  raise SystemExit(f'No probed dimensions found for {tasks}')
print(
  max(int(item['real_obs_dim']) for item in matched),
  max(int(item['real_act_dim']) for item in matched),
)
PY
}

declare -a PIDS=()
declare -A PID_LOGDIR=()
declare -a FAILED=()
run_counter=0

cleanup_finished() {
  local next=()
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      next+=("$pid")
    else
      if wait "$pid"; then
        touch "${PID_LOGDIR[$pid]}/scheduler_done"
      else
        FAILED+=("${PID_LOGDIR[$pid]}")
      fi
      unset "PID_LOGDIR[$pid]"
    fi
  done
  PIDS=("${next[@]}")
}

wait_for_slot() {
  cleanup_finished
  while (( ${#PIDS[@]} >= MAX_PARALLEL )); do
    sleep 60
    cleanup_finished
  done
}

launch_run() {
  local task_string="$1"
  local project="$2"
  local task_interval="$3"
  local obs_dim="$4"
  local act_dim="$5"
  local mechanism="$6"
  local seed="$7"

  wait_for_slot
  if [[ -n "${MAX_LAUNCHES:-}" ]] && (( run_counter >= MAX_LAUNCHES )); then
    return
  fi
  local gpu="${CUDA_DEVICES[$((run_counter % ${#CUDA_DEVICES[@]}))]}"
  local reset_mechanism="$mechanism"
  local reset_target="$WSC_TARGET"
  local group="wsc_${mechanism}_${WSC_TARGET}"
  if [[ "$mechanism" == "no_wsc" || "$mechanism" == "disabled" || "$mechanism" == "off" || "$mechanism" == "none" ]]; then
    reset_mechanism="disabled"
    reset_target="all"
    group="no_wsc"
  fi
  local logdir="${BASE_LOGDIR_ROOT}/${project}/${group}/seed_${seed}"
  if [[ "${FRESH_RERUN:-0}" == "1" && -e "$logdir" ]]; then
    local backup="${logdir}.failed.$(date +%Y%m%d_%H%M%S)"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "DRY_RUN FRESH_RERUN would move existing $logdir -> $backup"
    else
      echo "FRESH_RERUN moving existing $logdir -> $backup"
      mv "$logdir" "$backup"
    fi
  fi
  if [[ "${FRESH_RERUN:-0}" != "1" && -f "$logdir/scheduler_done" ]]; then
    echo "SKIP completed $logdir"
    return
  fi

  local cmd_args=(
    python dreamerv3/main.py
    --configs continual_dmc_priori "$MODEL_SIZE"
    --task "$task_string"
    --logdir "$logdir"
    --run.train_ratio "$TRAIN_RATIO"
    --run.task_interval "$task_interval"
    --run.reset_frequency 0
    --run.reset_mechanism "$reset_mechanism"
    --run.reset_target "$reset_target"
    --run.revive_epoch 0
    --env.continual_dmc_priori.obs_dim "$obs_dim"
    --env.continual_dmc_priori.task_action_space "$act_dim"
    --replay.chunksize "$REPLAY_CHUNKSIZE"
    --replay.cache_chunks "$REPLAY_CACHE_CHUNKS"
    --seed "$seed"
    --egl_device "$gpu"
    --agent.imag_length "$AGENT_IMAG_LENGTH"
    --agent.wsc.target_norm "$WSC_TARGET_NORM"
    --agent.wsc.scale_factor "$WSC_SCALE_FACTOR"
    --agent.redo.redo_enabled "$REDO_ENABLED"
    --agent.redo.grad_redo_enabled "$GRAD_REDO_ENABLED"
    --agent.redo.act_log_item "$ACT_LOG_ITEM"
    --agent.redo.grad_log_item "$GRAD_LOG_ITEM"
  )
  if [[ -n "$RUN_SAVE_EVERY" ]]; then
    cmd_args+=(--run.save_every "$RUN_SAVE_EVERY")
  fi
  if [[ -n "$EXTRA_ARGS" ]]; then
    read -r -a extra_args_array <<< "$EXTRA_ARGS"
    cmd_args+=("${extra_args_array[@]}")
  fi

  echo "LAUNCH gpu=$gpu task=$task_string mechanism=$mechanism reset_mechanism=$reset_mechanism seed=$seed logdir=$logdir"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    run_counter=$((run_counter + 1))
    return
  fi
  mkdir -p "$logdir"
  if [[ -s "$logdir/train.log" ]]; then
    mv "$logdir/train.log" "$logdir/train.$(date +%Y%m%d_%H%M%S).log"
  fi
  local env_args=(PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$gpu")
  local latest_wandb_run=""
  latest_wandb_run="$(ls -dt "$logdir"/wandb/wandb/run-* 2>/dev/null | head -n 1 || true)"
  if [[ -n "$latest_wandb_run" && -z "${WANDB_RUN_ID:-}" ]]; then
    local wandb_run_id="${latest_wandb_run##*-}"
    if [[ -n "$wandb_run_id" ]]; then
      env_args+=(WANDB_RUN_ID="$wandb_run_id" WANDB_RESUME="${WANDB_RESUME:-allow}")
    fi
  fi
  env "${env_args[@]}" "${cmd_args[@]}" > "$logdir/train.log" 2>&1 &
  local pid=$!
  PIDS+=("$pid")
  PID_LOGDIR[$pid]="$logdir"
  run_counter=$((run_counter + 1))
  sleep "${LAUNCH_STAGGER_SECONDS:-20}"
}

echo "WSC/no_wsc scheduler phase=$PHASE model=$MODEL_SIZE max_parallel=$MAX_PARALLEL gpus=${CUDA_DEVICES[*]}"
echo "Replay chunksize=$REPLAY_CHUNKSIZE cache_chunks=$REPLAY_CACHE_CHUNKS target=$WSC_TARGET"

for spec in "${TASK_SPECS[@]}"; do
  IFS='::' read -r task_string _ project _ task_interval <<< "$spec"
  # Bash IFS treats each ':' independently; parse robustly via parameter cuts.
  task_string="${spec%%::*}"
  rest="${spec#*::}"
  project="${rest%%::*}"
  task_interval="${rest##*::}"
  read -r obs_dim act_dim < <(task_dims "$task_string")
  echo "TASK task=$task_string project=$project interval=$task_interval dims=$obs_dim/$act_dim"
  for mechanism in "${WSC_MECHANISMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      launch_run "$task_string" "$project" "$task_interval" "$obs_dim" "$act_dim" "$mechanism" "$seed"
    done
  done
done

while (( ${#PIDS[@]} > 0 )); do
  sleep 120
  cleanup_finished
  echo "MONITOR active=${#PIDS[@]} failed=${#FAILED[@]}"
done

if (( ${#FAILED[@]} > 0 )); then
  echo "Failed runs:"
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
echo "All scheduled WSC/no_wsc runs finished."
