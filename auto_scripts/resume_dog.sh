#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DOG_ROOT="${DOG_ROOT:-logdir/continual_dreamer_soft_reset_dog_stand|dog_walk|dog_trot_size1m}"
MODEL_SIZE="${MODEL_SIZE:-size1m}"
CUDA_DEVICES_STR="${CUDA_DEVICES_STR:-2 3 4 5 6 7 0 1}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
WANDB_RESUME_MODE="${WANDB_RESUME_MODE:-allow}"
REQUIRE_WANDB_ID="${REQUIRE_WANDB_ID:-1}"
REPLAY_CACHE_CHUNKS="${REPLAY_CACHE_CHUNKS:-512}"
DRY_RUN="${DRY_RUN:-0}"
TARGETS="${TARGETS:-}"
SEEDS="${SEEDS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

read -r -a CUDA_DEVICES <<< "$CUDA_DEVICES_STR"
if (( ${#CUDA_DEVICES[@]} == 0 )); then
  echo "ERROR: CUDA_DEVICES_STR did not contain any devices." >&2
  exit 1
fi

timestamp="$(date +%Y%m%dT%H%M%S)"
active_pids=()
launch_count=0
skipped_count=0

contains_word() {
  local needle="$1"
  local haystack="$2"
  [[ -z "$haystack" ]] && return 0
  for item in $haystack; do
    [[ "$item" == "$needle" ]] && return 0
  done
  return 1
}

refresh_active() {
  local alive=()
  local pid
  for pid in "${active_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      alive+=("$pid")
    fi
  done
  active_pids=("${alive[@]}")
}

wait_for_slot() {
  refresh_active
  while (( ${#active_pids[@]} >= MAX_CONCURRENT )); do
    sleep 30
    refresh_active
  done
}

is_already_running() {
  local logdir="$1"
  ps -eo args | rg -F -- "dreamerv3/main.py" | rg -F -- "--logdir $logdir" >/dev/null 2>&1
}

run_info() {
  local logdir="$1"
  python - "$logdir" <<'PY'
import json
import sys
from pathlib import Path

import yaml

logdir = Path(sys.argv[1])
config_path = logdir / "config.yaml"
metrics_path = logdir / "metrics.jsonl"

if not config_path.exists():
    raise SystemExit(f"missing config: {config_path}")

config = yaml.safe_load(config_path.read_text())
run = config.get("run", {})
agent = config.get("agent", {})
redo = agent.get("redo", {})
env = config.get("env", {}).get("continual_dmc_priori", {})

last_step = 0
if metrics_path.exists() and metrics_path.stat().st_size:
    last = None
    with metrics_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    if last:
        last_step = int(float(json.loads(last).get("step", 0)))

task_action_space = env.get("task_action_space", "")
if isinstance(task_action_space, list):
    task_action_space = task_action_space[0] if task_action_space else ""

values = {
    "task": config.get("task", ""),
    "seed": config.get("seed", ""),
    "steps": int(float(run.get("steps", 0))),
    "last_step": last_step,
    "train_ratio": run.get("train_ratio", ""),
    "task_interval": int(float(run.get("task_interval", 0))),
    "reset_frequency": int(float(run.get("reset_frequency", 0))),
    "reset_mechanism": run.get("reset_mechanism", "hard"),
    "reset_target": run.get("reset_target", "no_reset"),
    "reset_alpha": run.get("reset_alpha", ""),
    "revive_epoch": int(float(run.get("revive_epoch", 0))),
    "revive_strategy": run.get("revive_strategy", "fixed"),
    "obs_dim": env.get("obs_dim", ""),
    "task_action_space": task_action_space,
    "imag_length": agent.get("imag_length", ""),
    "redo_enabled": redo.get("redo_enabled", ""),
    "grad_redo_enabled": redo.get("grad_redo_enabled", ""),
    "act_log_item": redo.get("act_log_item", ""),
    "grad_log_item": redo.get("grad_log_item", ""),
}
print("\t".join(str(values[key]) for key in (
    "task", "seed", "steps", "last_step", "train_ratio", "task_interval",
    "reset_frequency", "reset_mechanism", "reset_target", "reset_alpha",
    "revive_epoch", "revive_strategy", "obs_dim", "task_action_space",
    "imag_length", "redo_enabled", "grad_redo_enabled", "act_log_item",
    "grad_log_item")))
PY
}

wandb_id_for_logdir() {
  local logdir="$1"
  python - "$logdir" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]) / "wandb" / "wandb"
runs = sorted(root.glob("run-*"), key=lambda path: path.stat().st_mtime)
if runs:
    print(runs[-1].name.rsplit("-", 1)[-1])
PY
}

append_run_manifest() {
  local path="$1"
  local row="$2"
  if [[ ! -e "$path" ]]; then
    echo -e "logdir\tpid\tgpu\twandb_id\tlast_step\tsteps\treset_target\tseed\tresume_log" > "$path"
  fi
  echo -e "$row" >> "$path"
}

while IFS= read -r -d '' logdir; do
  if [[ ! -d "$logdir/ckpt" ]]; then
    echo "SKIP no checkpoint: $logdir"
    skipped_count=$((skipped_count + 1))
    continue
  fi

  info="$(run_info "$logdir")"
  IFS=$'\t' read -r task seed steps last_step train_ratio task_interval \
    reset_frequency reset_mechanism reset_target reset_alpha revive_epoch \
    revive_strategy obs_dim task_action_space imag_length redo_enabled \
    grad_redo_enabled act_log_item grad_log_item <<< "$info"

  if ! contains_word "$reset_target" "$TARGETS"; then
    echo "SKIP target filter: $logdir"
    skipped_count=$((skipped_count + 1))
    continue
  fi
  if ! contains_word "$seed" "$SEEDS"; then
    echo "SKIP seed filter: $logdir"
    skipped_count=$((skipped_count + 1))
    continue
  fi
  if (( last_step >= steps )); then
    echo "SKIP completed: $logdir (last_step=$last_step steps=$steps)"
    skipped_count=$((skipped_count + 1))
    continue
  fi
  if is_already_running "$logdir"; then
    echo "SKIP already running: $logdir"
    skipped_count=$((skipped_count + 1))
    continue
  fi

  wandb_id="$(wandb_id_for_logdir "$logdir")"
  if [[ -z "$wandb_id" && "$REQUIRE_WANDB_ID" == "1" ]]; then
    echo "SKIP missing wandb id: $logdir"
    skipped_count=$((skipped_count + 1))
    continue
  fi

  device_num="${CUDA_DEVICES[$((launch_count % ${#CUDA_DEVICES[@]}))]}"
  resume_log="$logdir/resume_${timestamp}.log"
  cmd_args=(
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
    --env.continual_dmc_priori.task_action_space "$task_action_space"
    --replay.cache_chunks "$REPLAY_CACHE_CHUNKS"
    --seed "$seed"
    --egl_device "$device_num"
    --agent.imag_length "$imag_length"
    --agent.redo.redo_enabled "$redo_enabled"
    --agent.redo.grad_redo_enabled "$grad_redo_enabled"
    --agent.redo.act_log_item "$act_log_item"
    --agent.redo.grad_log_item "$grad_log_item"
  )
  if [[ -n "$EXTRA_ARGS" ]]; then
    read -r -a extra_args_array <<< "$EXTRA_ARGS"
    cmd_args+=("${extra_args_array[@]}")
  fi

  echo "RESUME $logdir"
  echo "  step: $last_step / $steps"
  echo "  target/seed: $reset_target / $seed"
  echo "  gpu: $device_num"
  echo "  wandb: ${wandb_id:-NEW_RUN}"
  echo "  replay cache chunks: $REPLAY_CACHE_CHUNKS"
  echo "  log: $resume_log"

  if [[ "$DRY_RUN" == "1" ]]; then
    launch_count=$((launch_count + 1))
    continue
  fi

  wait_for_slot
  env_args=(
    PYTHONUNBUFFERED=1
    CUDA_VISIBLE_DEVICES="$device_num"
  )
  if [[ -n "$wandb_id" ]]; then
    env_args+=(
      WANDB_RUN_ID="$wandb_id"
      WANDB_RESUME="$WANDB_RESUME_MODE"
    )
  fi
  setsid env "${env_args[@]}" "${cmd_args[@]}" >> "$resume_log" 2>&1 < /dev/null &
  pid=$!
  active_pids+=("$pid")
  append_run_manifest \
    "$logdir/resume_manifest.tsv" \
    "$logdir\t$pid\t$device_num\t${wandb_id:-}\t$last_step\t$steps\t$reset_target\t$seed\t$resume_log"
  launch_count=$((launch_count + 1))
  sleep 5
done < <(find "$DOG_ROOT" -mindepth 2 -maxdepth 2 -type d -name 'seed_*' -print0 | sort -z)

echo "Resume deployment complete."
if [[ "$DRY_RUN" == "1" ]]; then
  echo "Planned: $launch_count"
else
  echo "Launched: $launch_count"
fi
echo "Skipped: $skipped_count"
if [[ "$DRY_RUN" != "1" ]]; then
  echo "Per-run manifests: <logdir>/resume_manifest.tsv"
fi
if (( ${#active_pids[@]} > 0 )); then
  echo "Active PIDs: ${active_pids[*]}"
fi
