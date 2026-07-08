#!/bin/bash

# Launch the requested walker and quadruped/dog/humanoid continual sequences.
# Defaults are intentionally conservative: one process each on GPU 2 and 3.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CUDA_DEVICES_STR="${CUDA_DEVICES_STR:-2 3}"
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
MODEL_SIZE="${MODEL_SIZE:-size1m}"
PREFIX="${PREFIX:-continual_dreamer_soft_reset}"
BASE_LOGDIR_ROOT="${BASE_LOGDIR_ROOT:-logdir}"

TRAIN_RATIO="${TRAIN_RATIO:-1024}"
RESET_FREQUENCY="${RESET_FREQUENCY:-50000}"
RESET_MECHANISM="${RESET_MECHANISM:-sandp}"
RESET_ALPHA="${RESET_ALPHA:-0.8}"
REVIVE_EPOCH="${REVIVE_EPOCH:-0}"
REVIVE_STRATEGY="${REVIVE_STRATEGY:-threshold}"
REPLAY_CACHE_CHUNKS="${REPLAY_CACHE_CHUNKS:-512}"
REPLAY_CHUNKSIZE="${REPLAY_CHUNKSIZE:-1024}"

AGENT_IMAG_LENGTH="${AGENT_IMAG_LENGTH:-15}"
REDO_ENABLED="${REDO_ENABLED:-True}"
GRAD_REDO_ENABLED="${GRAD_REDO_ENABLED:-True}"
ACT_LOG_ITEM="${ACT_LOG_ITEM:-log+erank+srank}"
GRAD_LOG_ITEM="${GRAD_LOG_ITEM:-log+erank+srank}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

DIMS_JSON="$REPO_ROOT/embodied/envs/dmc_priori_dims.json"

read -r -a CUDA_DEVICES <<< "$CUDA_DEVICES_STR"
if (( ${#CUDA_DEVICES[@]} == 0 )); then
  echo "ERROR: CUDA_DEVICES_STR is empty." >&2
  exit 1
fi

RESET_FREQUENCY_K="$((RESET_FREQUENCY / 1000))k"
RESET_ALPHA_TAG="${RESET_ALPHA//./p}"
if (( REVIVE_EPOCH > 0 )); then
  RESET_FREQUENCY_TAG="${RESET_FREQUENCY_K}_revive_${REVIVE_EPOCH}_${REVIVE_STRATEGY}"
else
  RESET_FREQUENCY_TAG="${RESET_FREQUENCY_K}_no_revive"
fi

declare -a SETTINGS=(
  "no_reset|1000"
  "no_reset|2000"
  "no_reset|3000"
  "all|1000"
  "all|2000"
  "all|3000"
  "ab_wm_head|1000"
  "ab_wm_head|2000"
  "ab_wm_head|3000"
  "ab_agent_head|1000"
  "ab_agent_head|2000"
  "ab_agent_head|3000"
  "agent_head|1000"
  "agent_head|2000"
  "agent_head|3000"
  "wm_head|1000"
  "wm_head|2000"
  "wm_head|3000"
)

dims_for_tasks() {
  python - "$DIMS_JSON" "$1" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
tasks = [item.strip() for item in sys.argv[2].split('|') if item.strip()]
entries = {item.get('task'): item for item in payload.get('tasks', [])}
matched = []
missing = []
for task in tasks:
  item = entries.get(task)
  if item and item.get('status') == 'ok':
    matched.append(item)
  else:
    missing.append(task)
if missing:
  raise SystemExit(f'Missing dimension probes for {missing}')
print(
    max(int(item['real_obs_dim']) for item in matched),
    max(int(item['real_act_dim']) for item in matched),
)
PY
}

reset_tag_for() {
  local target="$1"
  if [[ "$target" == "no_reset" ]]; then
    echo "no_reset"
  elif [[ "$RESET_MECHANISM" == "hard" || "$RESET_MECHANISM" == "opt_only" ]]; then
    echo "${RESET_MECHANISM}_${target}"
  else
    echo "${RESET_MECHANISM}_${target}_a${RESET_ALPHA_TAG}"
  fi
}

launch_run() {
  local suite_name="$1"
  local task_string="$2"
  local task_intervals="$3"
  local task_interval="$4"
  local total_steps="$5"
  local reset_target="$6"
  local seed="$7"
  local gpu="$8"

  read -r obs_dim act_dim < <(dims_for_tasks "$task_string")
  local reset_tag
  reset_tag="$(reset_tag_for "$reset_target")"
  local logdir="$BASE_LOGDIR_ROOT/${PREFIX}_${suite_name}_${MODEL_SIZE}/${reset_tag}_${RESET_FREQUENCY_TAG}/seed_$seed"

  if [[ -s "$logdir/train.log" ]]; then
    echo "SKIP existing train.log: $logdir"
    return 0
  fi

  mkdir -p "$logdir"
  local -a cmd_args=(
    python dreamerv3/main.py
    --configs continual_dmc_priori "$MODEL_SIZE"
    --task "$task_string"
    --logdir "$logdir"
    --run.steps "$total_steps"
    --run.train_ratio "$TRAIN_RATIO"
    --run.task_interval "$task_interval"
    --run.task_intervals "$task_intervals"
    --run.reset_frequency "$RESET_FREQUENCY"
    --run.reset_mechanism "$RESET_MECHANISM"
    --run.reset_target "$reset_target"
    --run.reset_alpha "$RESET_ALPHA"
    --run.revive_epoch "$REVIVE_EPOCH"
    --run.revive_strategy "$REVIVE_STRATEGY"
    --env.continual_dmc_priori.obs_dim "$obs_dim"
    --env.continual_dmc_priori.task_action_space "$act_dim"
    --replay.cache_chunks "$REPLAY_CACHE_CHUNKS"
    --replay.chunksize "$REPLAY_CHUNKSIZE"
    --seed "$seed"
    --egl_device "$gpu"
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

  echo "START suite=$suite_name target=$reset_target seed=$seed gpu=$gpu steps=$total_steps intervals=$task_intervals logdir=$logdir"
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$gpu" "${cmd_args[@]}" > "$logdir/train.log" 2>&1 &
  local pid=$!
  printf 'pid\tgpu\tsuite\ttask_string\ttask_intervals\ttotal_steps\treset_target\tseed\tlog\n' > "$logdir/launch_manifest.tsv"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$pid" "$gpu" "$suite_name" "$task_string" "$task_intervals" \
    "$total_steps" "$reset_target" "$seed" "$logdir/train.log" \
    >> "$logdir/launch_manifest.tsv"
}

wait_for_slot() {
  while (( $(jobs -pr | wc -l) >= MAX_CONCURRENT )); do
    sleep 60
  done
}

echo "Sequence experiment launcher"
echo "CUDA_DEVICES_STR=$CUDA_DEVICES_STR MAX_CONCURRENT=$MAX_CONCURRENT MODEL_SIZE=$MODEL_SIZE"
echo "Reset: mechanism=$RESET_MECHANISM alpha=$RESET_ALPHA frequency=$RESET_FREQUENCY revive=$REVIVE_EPOCH/$REVIVE_STRATEGY"

job_idx=0
for setting in "${SETTINGS[@]}"; do
  IFS='|' read -r reset_target seed <<< "$setting"

  wait_for_slot
  gpu="${CUDA_DEVICES[$((job_idx % ${#CUDA_DEVICES[@]}))]}"
  launch_run \
    "walker_stand|walker_walk|walker_run" \
    "walker_stand|walker_walk|walker_run" \
    "500000|500000|500000" \
    "500000" \
    "1500000" \
    "$reset_target" \
    "$seed" \
    "$gpu"
  job_idx=$((job_idx + 1))
  sleep 45

  wait_for_slot
  gpu="${CUDA_DEVICES[$((job_idx % ${#CUDA_DEVICES[@]}))]}"
  launch_run \
    "quadruped_run|dog_stand|humanoid_stand" \
    "quadruped_run|dog_stand|humanoid_stand" \
    "1000000|2000000|4000000" \
    "1000000" \
    "7000000" \
    "$reset_target" \
    "$seed" \
    "$gpu"
  job_idx=$((job_idx + 1))
  sleep 45
done

wait
echo "All requested sequence experiments finished."
