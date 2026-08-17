#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PROJECT="${PROJECT:-continual_mfrl_size1m_dmc_vision}"
TASKS="${TASKS:-walker_run,hopper_hop,cheetah_run}"
TASK_STEPS="${TASK_STEPS:-1000000}"
TASK_REPEATS="${TASK_REPEATS:-5}"
SEEDS="${SEEDS:-1000 2000 3000}"
DEFAULT_LR="${DEFAULT_LR:-4e-5}"
LR_DIV10="${LR_DIV10:-4e-6}"
LR_X10="${LR_X10:-4e-4}"
AUGMENTATION_ENABLED="${AUGMENTATION_ENABLED:-False}"
MODEL_SIZE="${MODEL_SIZE:-size1m}"
CONFIG="${CONFIG:-examples/configs/continual_drq_vision.py}"
LOG_ROOT="${LOG_ROOT:-logdir/${PROJECT}}"
SCHED_LOG_DIR="${SCHED_LOG_DIR:-logdir/scheduler}"
GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-10000}"
RAM_MIN_FREE_MB="${RAM_MIN_FREE_MB:-24000}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-98}"
MAX_COMPUTE_PROCS_PER_GPU="${MAX_COMPUTE_PROCS_PER_GPU:-4}"
MAX_NEW_RUNS_PER_GPU="${MAX_NEW_RUNS_PER_GPU:-2}"
LAUNCH_SLEEP_SEC="${LAUNCH_SLEEP_SEC:-45}"
WAIT_SLEEP_SEC="${WAIT_SLEEP_SEC:-120}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-100000}"
CHECKPOINT_REPLAY="${CHECKPOINT_REPLAY:-True}"
CHECKPOINT_KEEP="${CHECKPOINT_KEEP:-1}"
RESTORE_CHECKPOINT="${RESTORE_CHECKPOINT:-}"
CHECKPOINT_LOCK="${CHECKPOINT_LOCK:-logdir/scheduler/drq_vision_checkpoint.lock}"

mkdir -p "${LOG_ROOT}" "${SCHED_LOG_DIR}"
SCHED_LOG="${SCHED_LOG_DIR}/${PROJECT}_scheduler.$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${SCHED_LOG_DIR}/${PROJECT}_scheduler.pid"
echo "$$" > "${PID_FILE}"

if [[ -f .env ]]; then
  _xtrace_on=0
  case "$-" in
    *x*) _xtrace_on=1; set +x ;;
  esac
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
  if (( _xtrace_on )); then
    set -x
  fi
fi

declare -A NEW_LAUNCH_COUNT=()

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${SCHED_LOG}"
}

free_ram_mb() {
  free -m | awk '/Mem:/ {print $7}'
}

active_gpu_procs() {
  local gpu="$1"
  nvidia-smi pmon -c 1 2>/dev/null \
    | awk -v g="${gpu}" '$1 == g && $2 ~ /^[0-9]+$/ {count++} END {print count + 0}'
}

select_gpu() {
  local best_gpu=""
  local best_score=999999
  local best_free=-1
  while IFS=',' read -r raw_gpu raw_free raw_util; do
    local gpu free_mb util active launched score
    gpu="$(echo "${raw_gpu}" | xargs)"
    case " ${GPU_IDS} " in
      *" ${gpu} "*) ;;
      *) continue ;;
    esac
    free_mb="$(echo "${raw_free}" | xargs)"
    util="$(echo "${raw_util}" | xargs)"
    active="$(active_gpu_procs "${gpu}")"
    launched="${NEW_LAUNCH_COUNT[${gpu}]:-0}"
    if (( launched >= MAX_NEW_RUNS_PER_GPU )); then
      continue
    fi
    score=$((active + launched))
    if (( free_mb < GPU_MIN_FREE_MB )); then
      continue
    fi
    if (( util > MAX_GPU_UTIL )); then
      continue
    fi
    if (( score >= MAX_COMPUTE_PROCS_PER_GPU )); then
      continue
    fi
    if (( score < best_score || (score == best_score && free_mb > best_free) )); then
      best_gpu="${gpu}"
      best_score="${score}"
      best_free="${free_mb}"
    fi
  done < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
      --format=csv,noheader,nounits)
  echo "${best_gpu}"
}

run_is_active() {
  local save_dir="$1"
  pgrep -af "examples/train_continual_drq_vision.py .*--save_dir=${save_dir}" \
    >/dev/null 2>&1
}

launch_run() {
  local group="$1"
  local seed="$2"
  local lr="$3"
  local wsc_mechanism="$4"
  local run_name="seed_${seed}"
  local save_dir="${LOG_ROOT}/${group}/${run_name}"
  local run_log="${save_dir}/stdout.log"
  mkdir -p "${save_dir}"

  if run_is_active "${save_dir}"; then
    log "skip active ${group}/${run_name}"
    return
  fi

  while (( "$(free_ram_mb)" < RAM_MIN_FREE_MB )); do
    log "waiting: free RAM below ${RAM_MIN_FREE_MB} MB"
    sleep "${WAIT_SLEEP_SEC}"
  done

  local gpu=""
  while [[ -z "${gpu}" ]]; do
    gpu="$(select_gpu)"
    if [[ -z "${gpu}" ]]; then
      log "waiting: no GPU satisfies free=${GPU_MIN_FREE_MB}MB util<=${MAX_GPU_UTIL} max_procs=${MAX_COMPUTE_PROCS_PER_GPU}"
      sleep "${WAIT_SLEEP_SEC}"
    fi
  done

  NEW_LAUNCH_COUNT["${gpu}"]=$(( ${NEW_LAUNCH_COUNT[${gpu}]:-0} + 1 ))
  log "launch ${group}/${run_name} gpu=${gpu} lr=${lr} wsc=${wsc_mechanism}"

  nohup bash -lc "
    source \"\$(conda info --base)/etc/profile.d/conda.sh\"
    conda activate dreamer
    cd '${ROOT_DIR}'
    export CUDA_VISIBLE_DEVICES='${gpu}'
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
    export XLA_FLAGS=\"\${XLA_FLAGS:-} --xla_gpu_force_compilation_parallelism=1\"
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    export EGL_DEVICE_ID='${gpu}'
    export MUJOCO_EGL_DEVICE_ID='${gpu}'
    export PYTHONPATH='${ROOT_DIR}':\"\${PYTHONPATH:-}\"
    python examples/train_continual_drq_vision.py \
      --config='${CONFIG}' \
      --tasks='${TASKS}' \
      --task_steps='${TASK_STEPS}' \
      --task_repeats='${TASK_REPEATS}' \
      --seed='${seed}' \
      --save_dir='${save_dir}' \
      --project='${PROJECT}' \
      --group='${group}' \
      --run_name='${run_name}' \
      --compile_lock_path='logdir/scheduler/drq_vision_compile.lock' \
      --diagnostics_lock_path='logdir/scheduler/drq_vision_compile.lock' \
      --egl_device_id='${gpu}' \
      --log_interval=1000 \
      --diagnostics_interval=1000 \
      --eval_interval=10000 \
      --wandb=True \
      --tqdm=False \
      --restore_checkpoint='${RESTORE_CHECKPOINT}' \
      --checkpoint_interval='${CHECKPOINT_INTERVAL}' \
      --checkpoint_replay='${CHECKPOINT_REPLAY}' \
      --checkpoint_keep='${CHECKPOINT_KEEP}' \
      --checkpoint_lock_path='${CHECKPOINT_LOCK}' \
      --config.model_size='${MODEL_SIZE}' \
      --config.actor_lr='${lr}' \
      --config.critic_lr='${lr}' \
      --config.temp_lr='${lr}' \
      --config.augmentation_enabled='${AUGMENTATION_ENABLED}' \
      --config.wsc.mechanism='${wsc_mechanism}' \
      --config.wsc.target='all' \
      --config.l2_init.enabled=False \
      --config.redo.grad_redo_enabled=True \
      --config.redo.grad_redo_frequency=1000
  " > "${run_log}" 2>&1 &
  echo "$!" > "${save_dir}/pid"
  sleep "${LAUNCH_SLEEP_SEC}"
}

main() {
  log "scheduler start project=${PROJECT} tasks=${TASKS} task_steps=${TASK_STEPS} repeats=${TASK_REPEATS} model_size=${MODEL_SIZE} augmentation=${AUGMENTATION_ENABLED}"
  for seed in ${SEEDS}; do
    launch_run "no_wsc_lr_default" "${seed}" "${DEFAULT_LR}" "disabled"
  done
  for seed in ${SEEDS}; do
    launch_run "no_wsc_lr_div10" "${seed}" "${LR_DIV10}" "disabled"
  done
  for seed in ${SEEDS}; do
    launch_run "no_wsc_lr_x10" "${seed}" "${LR_X10}" "disabled"
  done
  for seed in ${SEEDS}; do
    launch_run "wsc_skip_last_layer_dout_all" "${seed}" "${DEFAULT_LR}" "wsc_skip_last_layer_dout_all"
  done
  log "scheduler finished submitting requested runs"
}

main "$@"
