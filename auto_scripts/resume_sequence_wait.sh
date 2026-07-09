#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODEL_SIZE="${MODEL_SIZE:-size1m}"
CUDA_DEVICES_STR="${CUDA_DEVICES_STR:-0 1 2 3 4 5 6 7}"
MIN_FREE_MB="${MIN_FREE_MB:-5200}"
MAX_CONCURRENT="${MAX_CONCURRENT:-1}"
POLL_SECONDS="${POLL_SECONDS:-300}"
WANDB_RESUME_MODE="${WANDB_RESUME_MODE:-allow}"
REPLAY_CACHE_CHUNKS="${REPLAY_CACHE_CHUNKS:-4096}"
REPLAY_CHUNKSIZE="${REPLAY_CHUNKSIZE:-1024}"
LOGDIRS="${LOGDIRS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -z "$LOGDIRS" ]]; then
  echo "ERROR: LOGDIRS must contain one or more run logdirs." >&2
  exit 1
fi

read -r -a CUDA_DEVICES <<< "$CUDA_DEVICES_STR"
read -r -a QUEUE <<< "$LOGDIRS"
timestamp="$(date +%Y%m%dT%H%M%S)"
mkdir -p logdir
supervisor_log="logdir/sequence_resume_wait_${timestamp}.log"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$supervisor_log"
}

is_running() {
  local logdir="$1"
  ps -eo args | rg -F -- "dreamerv3/main.py" | rg -F -- "--logdir $logdir" >/dev/null 2>&1
}

free_gpu() {
  python - "$MIN_FREE_MB" "${CUDA_DEVICES[@]}" <<'PY'
import sys
import subprocess

threshold = int(sys.argv[1])
allowed = set(sys.argv[2:])
best = None
output = subprocess.check_output([
    "nvidia-smi",
    "--query-gpu=index,memory.used,memory.total",
    "--format=csv,noheader,nounits",
], text=True)
for line in output.splitlines():
  idx, used, total = [item.strip() for item in line.split(",")]
  if idx not in allowed:
    continue
  free = int(total) - int(used)
  if free >= threshold and (best is None or free > best[1]):
    best = (idx, free)
if best:
  print(best[0])
PY
}

run_info() {
  local logdir="$1"
  python - "$logdir" <<'PY'
import json
import shlex
import sys
from pathlib import Path

import yaml

logdir = Path(sys.argv[1])
config = yaml.safe_load((logdir / "config.yaml").read_text())
run = config.get("run", {})
agent = config.get("agent", {})
redo = agent.get("redo", {})
env = config.get("env", {}).get("continual_dmc_priori", {})

last_step = 0
metrics = logdir / "metrics.jsonl"
if metrics.exists() and metrics.stat().st_size:
  for line in metrics.read_text(errors="replace").splitlines():
    if line.strip():
      try:
        last_step = int(float(json.loads(line).get("step", last_step)))
      except Exception:
        pass

action_space = env.get("task_action_space", "")
if isinstance(action_space, list):
  action_space = action_space[0] if action_space else ""

task_intervals = run.get("task_intervals", "")
if task_intervals is None:
  task_intervals = ""

values = {
    "task": config.get("task", ""),
    "seed": config.get("seed", ""),
    "steps": int(float(run.get("steps", 0))),
    "last_step": last_step,
    "train_ratio": run.get("train_ratio", ""),
    "task_interval": int(float(run.get("task_interval", 0))),
    "task_intervals": task_intervals,
    "reset_frequency": int(float(run.get("reset_frequency", 0))),
    "reset_mechanism": run.get("reset_mechanism", "sandp"),
    "reset_target": run.get("reset_target", "no_reset"),
    "reset_alpha": run.get("reset_alpha", ""),
    "revive_epoch": int(float(run.get("revive_epoch", 0))),
    "revive_strategy": run.get("revive_strategy", "threshold"),
    "obs_dim": env.get("obs_dim", ""),
    "action_space": action_space,
    "imag_length": agent.get("imag_length", ""),
    "redo_enabled": redo.get("redo_enabled", ""),
    "grad_redo_enabled": redo.get("grad_redo_enabled", ""),
    "act_log_item": redo.get("act_log_item", ""),
    "grad_log_item": redo.get("grad_log_item", ""),
}
for key, value in values.items():
  print(f"{key}={shlex.quote(str(value))}")
PY
}

wandb_id_for_logdir() {
  local logdir="$1"
  if [[ -s "$logdir/wandb_corrected_id.txt" ]]; then
    head -n 1 "$logdir/wandb_corrected_id.txt"
    return
  fi
  python - "$logdir" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]) / "wandb" / "wandb"
runs = sorted(root.glob("run-*"), key=lambda path: path.stat().st_mtime)
if runs:
  print(runs[-1].name.rsplit("-", 1)[-1])
PY
}

launch_one() {
  local logdir="$1" gpu="$2"
  local task seed steps last_step train_ratio task_interval task_intervals
  local reset_frequency reset_mechanism reset_target reset_alpha revive_epoch
  local revive_strategy obs_dim action_space imag_length redo_enabled
  local grad_redo_enabled act_log_item grad_log_item
  eval "$(run_info "$logdir")"

  local wandb_id
  wandb_id="$(wandb_id_for_logdir "$logdir")"
  if [[ -z "$wandb_id" ]]; then
    log "SKIP missing wandb id: $logdir"
    return 1
  fi

  local resume_log="$logdir/resume_wait_${timestamp}.log"
  local -a cmd_args=(
    python dreamerv3/main.py
    --configs continual_dmc_priori "$MODEL_SIZE"
    --task "$task"
    --logdir "$logdir"
    --run.steps "$steps"
    --run.train_ratio "$train_ratio"
    --run.task_interval "$task_interval"
    --run.reset_frequency "$reset_frequency"
    --run.reset_mechanism "$reset_mechanism"
    --run.reset_target "$reset_target"
    --run.reset_alpha "$reset_alpha"
    --run.revive_epoch "$revive_epoch"
    --run.revive_strategy "$revive_strategy"
    --env.continual_dmc_priori.obs_dim "$obs_dim"
    --env.continual_dmc_priori.task_action_space "$action_space"
    --replay.cache_chunks "$REPLAY_CACHE_CHUNKS"
    --replay.chunksize "$REPLAY_CHUNKSIZE"
    --seed "$seed"
    --egl_device "$gpu"
    --agent.imag_length "$imag_length"
    --agent.redo.redo_enabled "$redo_enabled"
    --agent.redo.grad_redo_enabled "$grad_redo_enabled"
    --agent.redo.act_log_item "$act_log_item"
    --agent.redo.grad_log_item "$grad_log_item"
  )
  if [[ -n "$task_intervals" ]]; then
    cmd_args+=(--run.task_intervals "$task_intervals")
  fi
  if [[ -n "$EXTRA_ARGS" ]]; then
    read -r -a extra_args_array <<< "$EXTRA_ARGS"
    cmd_args+=("${extra_args_array[@]}")
  fi

  log "RESUME logdir=$logdir gpu=$gpu wandb_id=$wandb_id step=$last_step/$steps log=$resume_log"
  setsid env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$gpu" \
    WANDB_RUN_ID="$wandb_id" WANDB_RESUME="$WANDB_RESUME_MODE" \
    "${cmd_args[@]}" >> "$resume_log" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date '+%F %T')" "$pid" "$gpu" "$wandb_id" "$last_step" "$resume_log" \
    >> "$logdir/resume_wait_manifest.tsv"
  return 0
}

log "sequence resume wait started min_free_mb=$MIN_FREE_MB max_concurrent=$MAX_CONCURRENT devices=${CUDA_DEVICES_STR}"
while (( ${#QUEUE[@]} > 0 )); do
  next=()
  for logdir in "${QUEUE[@]}"; do
    if [[ ! -d "$logdir" ]]; then
      log "DROP missing logdir: $logdir"
      continue
    fi
    if is_running "$logdir"; then
      log "DROP already running: $logdir"
      continue
    fi
    gpu="$(free_gpu || true)"
    if [[ -z "$gpu" ]]; then
      next+=("$logdir")
      continue
    fi
    if launch_one "$logdir" "$gpu"; then
      sleep 20
    else
      next+=("$logdir")
    fi
  done
  QUEUE=("${next[@]}")
  if (( ${#QUEUE[@]} > 0 )); then
    log "waiting for resources; queued=${#QUEUE[@]}"
    sleep "$POLL_SECONDS"
  fi
done
log "sequence resume wait queue empty"
