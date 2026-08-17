#!/usr/bin/env python3
"""Launch and monitor size12m continual DrQ vision experiments.

This monitor intentionally runs forever until it is explicitly terminated by the
user. It launches four batches of three runs, waits for each batch to produce
three valid local metric records per run, and continuously enforces resource
redlines for the processes it started.
"""

import json
import os
import signal
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = os.environ.get("PROJECT", "continual_mfrl_size12m_dmc_vision")
TASKS = os.environ.get("TASKS", "walker_run,hopper_hop,cheetah_run")
TASK_STEPS = os.environ.get("TASK_STEPS", "500000")
TASK_REPEATS = os.environ.get("TASK_REPEATS", "5")
SEEDS = os.environ.get("SEEDS", "1000 2000 3000").split()
MODEL_SIZE = os.environ.get("MODEL_SIZE", "size12m")
DEFAULT_LR = os.environ.get("DEFAULT_LR", "4e-5")
LOG_ROOT = ROOT / os.environ.get("LOG_ROOT", f"logdir/{PROJECT}")
SCHED_DIR = ROOT / os.environ.get("SCHED_LOG_DIR", "logdir/scheduler")
MAIN_LOG = ROOT / os.environ.get("MAIN_LOG", "mfrl_vision_12m.log")
STATE_PATH = SCHED_DIR / f"{PROJECT}_monitor_state.json"
PID_PATH = SCHED_DIR / f"{PROJECT}_monitor.pid"
COMPILE_LOCK = os.environ.get(
    "COMPILE_LOCK", "logdir/scheduler/drq_vision_size12m_compile.lock")
DIAGNOSTICS_LOCK = os.environ.get(
    "DIAGNOSTICS_LOCK", "logdir/scheduler/drq_vision_size12m_compile.lock")
POLL_SEC = int(os.environ.get("POLL_SEC", "30"))
RAM_MIN_FREE_MB = int(os.environ.get("RAM_MIN_FREE_MB", "30000"))
GPU_MIN_FREE_MB = int(os.environ.get("GPU_MIN_FREE_MB", "8000"))
GPU_EMERGENCY_FREE_MB = int(os.environ.get("GPU_EMERGENCY_FREE_MB", "512"))
MAX_NEW_RUNS_PER_GPU = int(os.environ.get("MAX_NEW_RUNS_PER_GPU", "1"))
MAX_COMPUTE_PROCS_PER_GPU = int(os.environ.get("MAX_COMPUTE_PROCS_PER_GPU", "8"))
GPU_IDS = os.environ.get("GPU_IDS", "1 2 3 4 5 6 7").split()
CHECKPOINT_INTERVAL = os.environ.get("CHECKPOINT_INTERVAL", "100000")
CHECKPOINT_REPLAY = os.environ.get("CHECKPOINT_REPLAY", "True")
CHECKPOINT_KEEP = os.environ.get("CHECKPOINT_KEEP", "1")
RESTORE_CHECKPOINT = os.environ.get("RESTORE_CHECKPOINT", "")
CHECKPOINT_LOCK = os.environ.get(
    "CHECKPOINT_LOCK", "logdir/scheduler/drq_vision_size12m_checkpoint.lock")

BATCHES = [
    {
        "group": "no_wsc",
        "augmentation": "False",
        "wsc": "disabled",
        "l2_enabled": "False",
        "l2_weight": "2e-5",
    },
    {
        "group": "l2_init_2e-4",
        "augmentation": "False",
        "wsc": "disabled",
        "l2_enabled": "True",
        "l2_weight": "2e-4",
    },
    {
        "group": "data_augmentation",
        "augmentation": "True",
        "wsc": "disabled",
        "l2_enabled": "False",
        "l2_weight": "2e-5",
    },
    {
        "group": "WSC_skip_last_layer_dout_all",
        "augmentation": "False",
        "wsc": "wsc_skip_last_layer_dout_all",
        "l2_enabled": "False",
        "l2_weight": "2e-5",
    },
]

ERROR_MARKERS = (
    "Traceback",
    "XlaRuntimeError",
    "ptxas",
    "CUDA_ERROR",
    "CUDNN_STATUS",
    "RuntimeError:",
    "ValueError:",
)

COMMON_REQUIRED_KEYS = (
    "loss/policy",
    "loss/value",
    "loss/temp",
    "train/temperature",
    "train/data_augmentation/drqv2_random_shift_active",
    "replay_buffer/ram_allocated_mb",
    "replay_buffer/ram_used_mb",
    "replay_buffer/size",
    "replay_buffer/capacity",
)
COMMON_REQUIRED_PREFIXES = (
    "act_redo/Linear_WB_FNorm/",
    "act_redo/RMSNorm_Out_L2_Mean/",
    "act_redo/erank/",
    "act_redo/srank/",
    "act_redo/Dormant_",
    "grad_redo/GradDormant_",
    "grad_redo/Grad_Mean/",
    "opt/raw_grad_mean/",
    "opt/update_mean/",
)


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message):
    MAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with MAIN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now()}] {message}\n")


def load_dotenv():
    path = ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def free_ram_mb():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return 0


def gpu_rows():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        log(f"RESOURCE nvidia-smi failed: {exc}")
        return []
    rows = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        rows.append({
            "gpu": parts[0],
            "free_mb": int(parts[1]),
            "util": int(parts[2]),
        })
    return rows


def active_gpu_procs(gpu):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "pmon", "-c", "1"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return 0
    count = 0
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == str(gpu) and parts[1].isdigit():
            count += 1
    return count


def select_gpu(launched_by_gpu):
    candidates = []
    allowed = set(GPU_IDS)
    for row in gpu_rows():
        gpu = row["gpu"]
        if gpu not in allowed:
            continue
        launched = launched_by_gpu.get(gpu, 0)
        if launched >= MAX_NEW_RUNS_PER_GPU:
            continue
        active = active_gpu_procs(gpu)
        if active + launched >= MAX_COMPUTE_PROCS_PER_GPU:
            continue
        if row["free_mb"] < GPU_MIN_FREE_MB:
            continue
        score = active + launched
        candidates.append((score, -row["free_mb"], gpu))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def _has_prefix(metrics, prefix):
    return any(key.startswith(prefix) for key in metrics)


def _wsc_skip_last_layer_matches_design(metrics):
    required = (
        "opt/wsc/controlled/actor_encoder_Conv_0",
        "opt/wsc/controlled/critic_encoder_Conv_0",
        "opt/wsc/controlled/actor_Dense_0",
        "opt/wsc/controlled/critic_Dense_0",
    )
    skipped = (
        "opt/wsc/controlled/actor_network_Dense_0",
        "opt/wsc/controlled/actor_network_Dense_1",
        "opt/wsc/controlled/critic_network_Vmap_StateActionValueSiLU_0__SiLUMLP_0_layer_3",
    )
    return (
        all(metrics.get(key) == 1.0 for key in required) and
        all(metrics.get(key) == 0.0 for key in skipped))


def _required_metrics_present(run, metrics):
    if not all(key in metrics for key in COMMON_REQUIRED_KEYS):
        return False
    if not all(_has_prefix(metrics, prefix) for prefix in COMMON_REQUIRED_PREFIXES):
        return False
    group = run.get("group", "")
    if group == "l2_init_2e-4":
        return (
            "loss/mechanism/l2_init/raw_loss" in metrics and
            "loss/mechanism/l2_init/weighted_loss" in metrics)
    if group == "data_augmentation":
        return metrics.get("train/data_augmentation/drqv2_random_shift_active") == 1.0
    if group == "WSC_skip_last_layer_dout_all":
        return _has_prefix(metrics, "opt/wsc/") and _wsc_skip_last_layer_matches_design(metrics)
    return True


def _missing_required_metrics(run, metrics):
    missing = [key for key in COMMON_REQUIRED_KEYS if key not in metrics]
    missing.extend(
        prefix for prefix in COMMON_REQUIRED_PREFIXES
        if not _has_prefix(metrics, prefix))
    group = run.get("group", "")
    if group == "l2_init_2e-4":
        for key in (
            "loss/mechanism/l2_init/raw_loss",
            "loss/mechanism/l2_init/weighted_loss",
        ):
            if key not in metrics:
                missing.append(key)
    elif group == "data_augmentation":
        if metrics.get("train/data_augmentation/drqv2_random_shift_active") != 1.0:
            missing.append("train/data_augmentation/drqv2_random_shift_active=1.0")
    elif group == "WSC_skip_last_layer_dout_all":
        if not _has_prefix(metrics, "opt/wsc/"):
            missing.append("opt/wsc/")
        elif not _wsc_skip_last_layer_matches_design(metrics):
            missing.append("opt/wsc/controlled/{encoder_Conv_0,Dense_0}=1.0")
    return missing[:16]


def valid_metric_count(run):
    save_dir = run["save_dir"]
    path = Path(save_dir) / "metrics.jsonl"
    count = 0
    required_count = 0
    last_step = None
    last_key_count = 0
    last_missing = []
    if not path.exists():
        return count, required_count, last_step, last_key_count, last_missing
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            metrics = row.get("metrics", {})
            if isinstance(metrics, dict) and metrics:
                count += 1
                last_step = row.get("step")
                last_key_count = len(metrics)
                if _required_metrics_present(run, metrics):
                    required_count += 1
                    last_missing = []
                else:
                    last_missing = _missing_required_metrics(run, metrics)
    if required_count:
        last_missing = []
    return count, required_count, last_step, last_key_count, last_missing


def _batch_by_group(group):
    for batch in BATCHES:
        if batch["group"] == group:
            return batch
    return {}


def load_existing_state():
    if not STATE_PATH.exists():
        return [], 0, False
    try:
        state = json.loads(STATE_PATH.read_text())
    except Exception as exc:
        log(f"STATE_LOAD_FAILED path={STATE_PATH} error={exc}")
        return [], 0, False
    if state.get("project") != PROJECT:
        return [], 0, False
    runs = []
    for item in state.get("runs", []):
        batch = _batch_by_group(item.get("group"))
        run = {**batch, **item}
        run["process"] = None
        run["stdout_offset"] = 0
        runs.append(run)
    if runs:
        log(
            f"STATE_RESUME path={STATE_PATH} runs={len(runs)} "
            f"batch_index={state.get('current_batch_index', 0)}")
    return (
        runs,
        int(state.get("current_batch_index", 0) or 0),
        bool(state.get("emergency_stop_triggered", False)),
    )


def save_state(runs, batch_index, emergency=False):
    SCHED_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": now(),
        "project": PROJECT,
        "tasks": TASKS,
        "task_steps": TASK_STEPS,
        "task_repeats": TASK_REPEATS,
        "model_size": MODEL_SIZE,
        "current_batch_index": batch_index,
        "emergency_stop_triggered": emergency,
        "ram_available_mb": free_ram_mb(),
        "gpus": gpu_rows(),
        "runs": [
            {key: run.get(key) for key in (
                "launch_index", "batch_index", "group", "seed", "gpu",
                "augmentation", "wsc", "l2_enabled", "l2_weight",
                "pid", "pgid", "save_dir", "run_log", "status",
                "returncode", "metric_count", "required_metric_count",
                "last_step", "last_metric_key_count", "last_missing_required")}
            for run in runs
        ],
    }
    STATE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def command_for(run):
    return f"""
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dreamer
cd '{ROOT}'
export CUDA_VISIBLE_DEVICES='{run['gpu']}'
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
export XLA_FLAGS="${{XLA_FLAGS:-}} --xla_gpu_force_compilation_parallelism=1 --xla_gpu_autotune_level=0"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID='{run['gpu']}'
export MUJOCO_EGL_DEVICE_ID='{run['gpu']}'
export PYTHONPATH='{ROOT}':"${{PYTHONPATH:-}}"
exec python examples/train_continual_drq_vision.py \\
  --config='examples/configs/continual_drq_vision.py' \\
  --tasks='{TASKS}' \\
  --task_steps='{TASK_STEPS}' \\
  --task_repeats='{TASK_REPEATS}' \\
  --seed='{run['seed']}' \\
  --save_dir='{run['save_dir']}' \\
  --project='{PROJECT}' \\
  --group='{run['group']}' \\
  --run_name='seed_{run['seed']}' \\
  --compile_lock_path='{COMPILE_LOCK}' \\
  --diagnostics_lock_path='{DIAGNOSTICS_LOCK}' \\
  --egl_device_id='{run['gpu']}' \\
  --log_interval=1000 \\
  --diagnostics_interval=1000 \\
  --eval_interval=10000 \\
  --wandb=True \\
  --tqdm=False \\
  --restore_checkpoint='{RESTORE_CHECKPOINT}' \\
  --checkpoint_interval='{CHECKPOINT_INTERVAL}' \\
  --checkpoint_replay='{CHECKPOINT_REPLAY}' \\
  --checkpoint_keep='{CHECKPOINT_KEEP}' \\
  --checkpoint_lock_path='{CHECKPOINT_LOCK}' \\
  --config.model_size='{MODEL_SIZE}' \\
  --config.actor_lr='{DEFAULT_LR}' \\
  --config.critic_lr='{DEFAULT_LR}' \\
  --config.temp_lr='{DEFAULT_LR}' \\
  --config.augmentation_enabled='{run['augmentation']}' \\
  --config.wsc.mechanism='{run['wsc']}' \\
  --config.wsc.target='all' \\
  --config.l2_init.enabled='{run['l2_enabled']}' \\
  --config.l2_init.weight='{run['l2_weight']}' \\
  --config.redo.grad_redo_enabled=True \\
  --config.redo.grad_redo_frequency=1000
"""


def launch_run(run):
    save_dir = Path(run["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    run_log = save_dir / "stdout.log"
    run["run_log"] = str(run_log)
    stdout = run_log.open("ab")
    proc = subprocess.Popen(
        ["bash", "-lc", command_for(run)],
        cwd=str(ROOT),
        stdout=stdout,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    run["pid"] = proc.pid
    run["pgid"] = os.getpgid(proc.pid)
    run["process"] = proc
    run["status"] = "running"
    run["stdout_offset"] = 0
    log(
        f"LAUNCH index={run['launch_index']} batch={run['batch_index']} "
        f"group={run['group']} seed={run['seed']} gpu={run['gpu']} "
        f"pid={run['pid']} pgid={run['pgid']} save_dir={run['save_dir']}")


def stop_run(run, reason):
    pgid = run.get("pgid")
    if not pgid:
        return
    log(
        f"EMERGENCY_STOP group={run['group']} seed={run['seed']} "
        f"pid={run.get('pid')} pgid={pgid} reason={reason}")
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception as exc:
        log(f"EMERGENCY_STOP_TERM_FAILED pgid={pgid} error={exc}")
    time.sleep(5)
    proc = run.get("process")
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(pgid, signal.SIGKILL)
            log(f"EMERGENCY_STOP_KILL pgid={pgid}")
        except ProcessLookupError:
            pass
        except Exception as exc:
            log(f"EMERGENCY_STOP_KILL_FAILED pgid={pgid} error={exc}")
    run["status"] = "emergency_stopped"


def scan_stdout(run):
    path = Path(run.get("run_log", ""))
    if not path.exists():
        return
    offset = run.get("stdout_offset", 0)
    size = path.stat().st_size
    if size < offset:
        offset = 0
    if size == offset:
        return
    with path.open("rb") as handle:
        handle.seek(offset)
        chunk = handle.read(size - offset).decode(errors="ignore")
    run["stdout_offset"] = size
    for marker in ERROR_MARKERS:
        if marker in chunk:
            log(
                f"ERROR_MARKER group={run['group']} seed={run['seed']} "
                f"marker={marker} log={path}")
            break


def update_run_status(run):
    count, required_count, last_step, key_count, missing = valid_metric_count(run)
    run["metric_count"] = count
    run["required_metric_count"] = required_count
    run["last_step"] = last_step
    run["last_metric_key_count"] = key_count
    run["last_missing_required"] = missing
    scan_stdout(run)
    proc = run.get("process")
    if proc is None:
        pid = run.get("pid")
        if pid and Path("/proc", str(pid)).exists():
            run["status"] = "running"
            run["returncode"] = None
        elif run.get("status") not in ("completed", "emergency_stopped", "exited_error"):
            run["status"] = "exited_unknown"
            run["returncode"] = None
            log(
                f"PROCESS_EXIT_UNKNOWN group={run['group']} seed={run['seed']} "
                f"pid={pid} metrics={count} required_metrics={required_count} "
                f"last_step={last_step} log={run.get('run_log')}")
        return
    code = proc.poll()
    run["returncode"] = code
    if code is None:
        run["status"] = "running"
    elif code == 0:
        run["status"] = "completed"
    elif run.get("status") != "emergency_stopped":
        run["status"] = "exited_error"
        log(
            f"PROCESS_EXIT_ERROR group={run['group']} seed={run['seed']} "
            f"pid={run.get('pid')} returncode={code} metrics={count} "
            f"required_metrics={required_count} "
            f"last_step={last_step} log={run.get('run_log')}")


def resource_redline():
    ram = free_ram_mb()
    if ram < RAM_MIN_FREE_MB:
        return f"system_ram_available_mb={ram} below {RAM_MIN_FREE_MB}"
    low_gpus = [
        row for row in gpu_rows()
        if row["gpu"] in set(GPU_IDS) and row["free_mb"] < GPU_EMERGENCY_FREE_MB
    ]
    if low_gpus:
        return f"gpu_free_mb_below_{GPU_EMERGENCY_FREE_MB}: {low_gpus}"
    return None


def wait_for_launch_resources():
    while True:
        reason = resource_redline()
        if reason:
            log(f"WAIT_RESOURCE_REDLINE {reason}")
            time.sleep(POLL_SEC)
            continue
        if free_ram_mb() >= RAM_MIN_FREE_MB:
            return
        time.sleep(POLL_SEC)


def main():
    load_dotenv()
    SCHED_DIR.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text(str(os.getpid()), encoding="utf-8")
    log(
        f"MONITOR_START project={PROJECT} tasks={TASKS} task_steps={TASK_STEPS} "
        f"task_repeats={TASK_REPEATS} model_size={MODEL_SIZE} "
        f"ram_min_free_mb={RAM_MIN_FREE_MB} gpu_min_free_mb={GPU_MIN_FREE_MB} "
        f"gpu_ids={' '.join(GPU_IDS)}")

    runs, batch_index, emergency = load_existing_state()
    launch_index = (
        max([int(run.get("launch_index", -1)) for run in runs], default=-1) + 1)

    while True:
        if not emergency and batch_index < len(BATCHES):
            if batch_index == 0:
                ready_for_next = True
            else:
                prev = [run for run in runs if run["batch_index"] == batch_index - 1]
                ready_for_next = (
                    len(prev) == len(SEEDS) and
                    all(run.get("required_metric_count", 0) >= 3 for run in prev)
                )
            if ready_for_next:
                launched_by_gpu = {}
                batch = BATCHES[batch_index]
                log(f"BATCH_READY batch={batch_index} group={batch['group']}")
                batch_runs = []
                for seed in SEEDS:
                    gpu = None
                    while gpu is None:
                        wait_for_launch_resources()
                        gpu = select_gpu(launched_by_gpu)
                        if gpu is None:
                            log(
                                "WAIT_GPU no GPU satisfies launch constraints "
                                f"gpu_min_free_mb={GPU_MIN_FREE_MB}")
                            time.sleep(POLL_SEC)
                    launched_by_gpu[gpu] = launched_by_gpu.get(gpu, 0) + 1
                    save_dir = LOG_ROOT / batch["group"] / f"seed_{seed}"
                    run = {
                        **batch,
                        "seed": seed,
                        "gpu": gpu,
                        "save_dir": str(save_dir),
                        "batch_index": batch_index,
                        "launch_index": launch_index,
                        "metric_count": 0,
                        "required_metric_count": 0,
                        "last_step": None,
                    }
                    launch_index += 1
                    batch_runs.append(run)
                for run in batch_runs:
                    launch_run(run)
                    runs.append(run)
                batch_index += 1

        for run in runs:
            update_run_status(run)

        reason = resource_redline()
        if reason and not emergency:
            emergency = True
            log(f"EMERGENCY_TRIGGER {reason}")
            for run in sorted(runs, key=lambda item: item["launch_index"], reverse=True):
                if run.get("status") == "running":
                    stop_run(run, reason)

        status_counts = {}
        for run in runs:
            status_counts[run.get("status", "unknown")] = (
                status_counts.get(run.get("status", "unknown"), 0) + 1)
        log(
            f"STATUS batch_index={batch_index}/{len(BATCHES)} "
            f"runs={len(runs)} statuses={status_counts} "
            f"required_metrics="
            f"{[(run['group'], run['seed'], run.get('required_metric_count', 0), run.get('last_step'), run.get('last_missing_required', [])) for run in runs]} "
            f"ram_available_mb={free_ram_mb()} gpus={gpu_rows()}")
        save_state(runs, batch_index, emergency)
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
