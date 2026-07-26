#!/usr/bin/env python3

import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "codex-cli-executor.log"
AGENT = ROOT / "AGENT.md"
CHECKLIST = ROOT / "task_checklist.md"
SCHED_DIR = ROOT / "logdir" / "scheduler"
STATE_PATH = SCHED_DIR / "codex_experiment_scheduler_state.json"
PID_PATH = SCHED_DIR / "codex_experiment_scheduler.pid"
OUT_PATH = SCHED_DIR / "codex_experiment_scheduler.out"

THRESH_FPS = float(os.environ.get("CODEX_SCHED_MIN_FPS", "5.8"))
CRAFTER_THRESH_FPS = float(os.environ.get("CODEX_SCHED_CRAFTER_MIN_FPS", "2.0"))
MAX_TOTAL_RUNNING = int(os.environ.get("CODEX_SCHED_MAX_TOTAL_RUNNING", "24"))
MAX_RUNS_PER_GPU = int(os.environ.get("CODEX_SCHED_MAX_RUNS_PER_GPU", "3"))
LAUNCH_BATCH = int(os.environ.get("CODEX_SCHED_LAUNCH_BATCH", "8"))
LAUNCH_STAGGER = int(os.environ.get("CODEX_SCHED_LAUNCH_STAGGER_SECONDS", "30"))
POLL_SECONDS = int(os.environ.get("CODEX_SCHED_POLL_SECONDS", "300"))
GPU_MIN_FREE_MB = int(os.environ.get("CODEX_SCHED_GPU_MIN_FREE_MB", "3500"))
RAM_MIN_FREE_MB = int(os.environ.get("CODEX_SCHED_RAM_MIN_FREE_MB", "24000"))
FRESH_GRACE_SECONDS = int(os.environ.get("CODEX_SCHED_FRESH_GRACE_SECONDS", "2700"))
CRAFTER_FRESH_GRACE_SECONDS = int(
    os.environ.get("CODEX_SCHED_CRAFTER_FRESH_GRACE_SECONDS", str(FRESH_GRACE_SECONDS))
)
MAX_ATTEMPTS = int(os.environ.get("CODEX_SCHED_MAX_ATTEMPTS", "2"))
LOG_COMPACT_SECONDS = int(os.environ.get("CODEX_SCHED_LOG_COMPACT_SECONDS", str(6 * 60 * 60)))
REMOTE_WANDB_CHECK_SECONDS = int(os.environ.get("CODEX_SCHED_REMOTE_WANDB_CHECK_SECONDS", "600"))
REMOTE_WANDB_TIMEOUT = int(os.environ.get("CODEX_SCHED_REMOTE_WANDB_TIMEOUT", "45"))

SEEDS = ("1000", "2000", "3000")
KEEP_OLD_TASKS = {
    "walker_run|hopper_hop|fish_swim",
    "dog_stand|dog_walk|dog_trot",
}
KEEP_OLD_GROUP_MARKERS = (
    "/no_wsc/",
    "/wsc_WSC_grad_scale_constant_all/",
)
ERROR_RE = (
    "Traceback",
    "RuntimeError",
    "ValueError",
    "FileNotFoundError",
    "Out of memory",
    "out of memory",
    "RESOURCE_EXHAUSTED",
    "nan detected",
)


@dataclass(frozen=True)
class Job:
    key: str
    priority: int
    label: str
    task: str
    project: str
    group: str
    run: str
    configs: tuple[str, ...]
    steps: int | None
    args: tuple[str, ...]

    @property
    def logdir(self) -> str:
        return f"logdir/{self.project}/{self.group}/{self.run}"


def stamp() -> str:
    return time.strftime("%F %T %Z")


def log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    print(f"{stamp()} | Scheduler: {message}", flush=True)
    with LOG.open("a") as handle:
        handle.write(f"{stamp()} | Scheduler: {message}\n")


def is_crafter_logdir(logdir: str) -> bool:
    return "crafter" in logdir.lower()


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {}


def save_state(state: dict) -> None:
    SCHED_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def parse_local_env() -> dict[str, str]:
    env = {}
    path = ROOT / ".env.wandb.local"
    if not path.exists():
        return env
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip("'").strip('"')
    return env


def task_dims(task_string: str) -> tuple[int, int]:
    dims_path = ROOT / "embodied" / "envs" / "dmc_priori_dims.json"
    payload = json.loads(dims_path.read_text())
    entries = {item.get("task"): item for item in payload.get("tasks", [])}
    obs = []
    act = []
    for task in [x.strip() for x in task_string.split("|") if x.strip()]:
        item = entries.get(task, {})
        if item.get("status") == "ok":
            obs.append(int(item["real_obs_dim"]))
            act.append(int(item["real_act_dim"]))
        elif "known_obs_dim_hint" in item:
            obs.append(int(item["known_obs_dim_hint"]))
    if not obs:
        raise RuntimeError(f"no probed obs dims for {task_string}")
    if not act:
        # All quadruped tasks use 12 actions in dm_control.
        act.append(12 if "quadruped" in task_string else 1)
    return max(obs), max(act)


def mechanism_group(mechanism: str) -> tuple[str, str, str]:
    if mechanism == "no_wsc":
        return "no_wsc", "disabled", "all"
    return f"wsc_{mechanism}_all", mechanism, "all"


def continual_args(
    *,
    task: str,
    interval: int,
    mechanism: str,
    train_ratio: int = 1024,
    steps: int | None = None,
    replay_chunksize: int = 4096,
    replay_cache_chunks: int = 1024,
    reset_frequency: int = 0,
    reset_alpha: float | None = None,
    revive_strategy: str = "threshold",
    save_every: int | None = 1800,
) -> tuple[str, ...]:
    obs_dim, act_dim = task_dims(task)
    _group, reset_mechanism, reset_target = mechanism_group(mechanism)
    args = [
        "--configs", "continual_dmc_priori", "size1m",
        "--task", task,
        "--run.train_ratio", str(train_ratio),
        "--run.task_interval", str(interval),
        "--run.reset_frequency", str(reset_frequency),
        "--run.reset_mechanism", reset_mechanism,
        "--run.reset_target", reset_target,
        "--run.revive_epoch", "0",
        "--run.revive_strategy", revive_strategy,
        "--env.continual_dmc_priori.obs_dim", str(obs_dim),
        "--env.continual_dmc_priori.task_action_space", str(act_dim),
        "--replay.chunksize", str(replay_chunksize),
        "--replay.cache_chunks", str(replay_cache_chunks),
        "--agent.imag_length", "15",
        "--agent.wsc.target_norm", "1.0",
        "--agent.wsc.scale_factor", "0.999",
        "--agent.redo.redo_enabled", "True",
        "--agent.redo.grad_redo_enabled", "True",
        "--agent.redo.act_log_item", "log+erank+srank",
        "--agent.redo.grad_log_item", "log+erank+srank",
    ]
    if steps is not None:
        args.extend(["--run.steps", str(steps)])
    # The soft-reset reference script has RESET_ALPHA, but this WSC repo does
    # not expose run.reset_alpha in configs.yaml. Keep the W&B group comparable
    # while avoiding an unsupported CLI flag.
    if save_every is not None:
        args.extend(["--run.save_every", str(save_every)])
    return tuple(args)


def crafter_args(mechanism: str) -> tuple[str, ...]:
    _group, reset_mechanism, reset_target = mechanism_group(mechanism)
    return (
        "--configs", "crafter", "size1m",
        "--task", "crafter_reward",
        "--run.steps", "100000000",
        "--run.task_interval", "100000000",
        "--run.reset_frequency", "0",
        "--run.reset_mechanism", reset_mechanism,
        "--run.reset_target", reset_target,
        "--run.revive_epoch", "0",
        "--agent.wsc.target_norm", "1.0",
        "--agent.wsc.scale_factor", "0.999",
        "--agent.redo.redo_enabled", "True",
        "--agent.redo.grad_redo_enabled", "True",
        "--agent.redo.act_log_item", "log+erank+srank",
        "--agent.redo.grad_log_item", "log+erank+srank",
    )


def build_jobs() -> list[Job]:
    jobs: list[Job] = []

    dmc_task = "swimmer_swimmer6|cheetah_run|reacher_hard"
    dmc_project = "continual_dreamer_soft_reset_dmcprior_swimmer_cheetah_reacher_size1m"
    for mechanism in ("no_wsc", "WSC_grad_scale_constant"):
        group, _, _ = mechanism_group(mechanism)
        for seed in SEEDS:
            jobs.append(Job(
                key=f"p1_dmc_{mechanism}_{seed}",
                priority=1,
                label=f"dmc-prior {mechanism} seed {seed}",
                task=dmc_task,
                project=dmc_project,
                group=group,
                run=f"seed_{seed}",
                configs=("continual_dmc_priori", "size1m"),
                steps=7_500_000,
                args=continual_args(
                    task=dmc_task,
                    interval=500_000,
                    mechanism=mechanism,
                    steps=7_500_000,
                    replay_cache_chunks=4096,
                ),
            ))

    crafter_project = "continual_dreamer_soft_reset_crafter_size1m"
    for mechanism in ("no_wsc", "WSC_grad_scale_constant"):
        group, _, _ = mechanism_group(mechanism)
        for seed in SEEDS:
            jobs.append(Job(
                key=f"p2_crafter_{mechanism}_{seed}",
                priority=2,
                label=f"crafter {mechanism} seed {seed}",
                task="crafter_reward",
                project=crafter_project,
                group=group,
                run=f"seed_{seed}",
                configs=("crafter", "size1m"),
                steps=100_000_000,
                args=crafter_args(mechanism),
            ))

    quad_task = "quadruped_walk|quadruped_escape|quadruped_fetch"
    quad_project = "continual_dreamer_soft_reset_quadruped_walk|quadruped_escape|quadruped_fetch_size1m"
    quad_group = "wsc_WSC_grad_scale_constant_all"
    for seed in SEEDS:
        jobs.append(Job(
            key=f"p3_quadruped_wsc_constant_{seed}",
            priority=3,
            label=f"quadruped WSC_grad_scale_constant seed {seed}",
            task=quad_task,
            project=quad_project,
            group=quad_group,
            run=f"seed_{seed}",
            configs=("continual_dmc_priori", "size1m"),
            steps=None,
            args=continual_args(
                task=quad_task,
                interval=1_000_000,
                mechanism="WSC_grad_scale_constant",
                reset_frequency=50_000,
                reset_alpha=0.8,
                replay_cache_chunks=1024,
                save_every=None,
            ),
        ))

    humanoid_task = "humanoid_stand|humanoid_run"
    humanoid_project = "continual_dreamer_soft_reset_humanoid_stand|humanoid_run_size1m"
    humanoid_group = "wsc_WSC_grad_scale_constant_all"
    for seed in SEEDS:
        jobs.append(Job(
            key=f"p4_humanoid_wsc_constant_{seed}",
            priority=4,
            label=f"humanoid WSC_grad_scale_constant seed {seed}",
            task=humanoid_task,
            project=humanoid_project,
            group=humanoid_group,
            run=f"seed_{seed}",
            configs=("continual_dmc_priori", "size1m"),
            steps=None,
            args=continual_args(
                task=humanoid_task,
                interval=3_000_000,
                mechanism="WSC_grad_scale_constant",
                replay_cache_chunks=1024,
            ),
        ))

    old_specs = (
        (
            "p5_old_walker",
            5,
            "walker_run|hopper_hop|fish_swim",
            "continual_dreamer_soft_reset_size1m",
            1_000_000,
            {"no_wsc": 4096, "WSC_grad_scale_constant": 4096},
        ),
        (
            "p6_old_dog",
            6,
            "dog_stand|dog_walk|dog_trot",
            "continual_dreamer_soft_reset_dog_stand|dog_walk|dog_trot_size1m",
            2_000_000,
            {"no_wsc": 1024, "WSC_grad_scale_constant": 4096},
        ),
    )
    for prefix, priority, task, project, interval, cache_by_mechanism in old_specs:
        for mechanism in ("no_wsc", "WSC_grad_scale_constant"):
            group, _, _ = mechanism_group(mechanism)
            for seed in SEEDS:
                jobs.append(Job(
                    key=f"{prefix}_{mechanism}_{seed}",
                    priority=priority,
                    label=f"old target {task} {mechanism} seed {seed}",
                    task=task,
                    project=project,
                    group=group,
                    run=f"seed_{seed}",
                    configs=("continual_dmc_priori", "size1m"),
                    steps=None,
                    args=continual_args(
                        task=task,
                        interval=interval,
                        mechanism=mechanism,
                        replay_cache_chunks=cache_by_mechanism[mechanism],
                    ),
                ))

    return jobs


def ps_lines() -> list[str]:
    out = subprocess.check_output(
        ["ps", "-eo", "pid,ppid,stat,etimes,cmd"],
        text=True,
        errors="replace",
    )
    return out.splitlines()[1:]


def dreamer_items() -> list[dict]:
    items = []
    for line in ps_lines():
        if "dreamerv3/main.py" not in line or "ps -eo" in line:
            continue
        parts = line.strip().split(None, 4)
        if len(parts) < 5:
            continue
        pid, ppid, stat, etimes, cmd = parts
        try:
            toks = shlex.split(cmd)
        except ValueError:
            continue
        def val(flag: str) -> str:
            if flag in toks:
                idx = toks.index(flag)
                return toks[idx + 1] if idx + 1 < len(toks) else ""
            return ""
        items.append({
            "pid": int(pid),
            "ppid": int(ppid),
            "stat": stat,
            "etimes": int(etimes),
            "cmd": cmd,
            "task": val("--task"),
            "logdir": val("--logdir"),
            "mech": val("--run.reset_mechanism"),
            "target": val("--run.reset_target"),
            "seed": val("--seed"),
            "gpu": val("--egl_device") or "?",
        })
    return items


def pgrep(pattern: str) -> list[int]:
    try:
        out = subprocess.check_output(["pgrep", "-f", pattern], text=True)
    except subprocess.CalledProcessError:
        return []
    self_pid = os.getpid()
    return [int(x) for x in out.split() if x.strip().isdigit() and int(x) != self_pid]


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def send(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass


def is_old_keep(item: dict) -> bool:
    logdir = item.get("logdir", "")
    return item.get("task") in KEEP_OLD_TASKS and any(marker in logdir for marker in KEEP_OLD_GROUP_MARKERS)


def terminate_processes(pids: list[int], reason: str) -> None:
    pids = sorted(set(pid for pid in pids if pid > 1 and pid != os.getpid()))
    if not pids:
        return
    for pid in pids:
        send(pid, signal.SIGTERM)
        send(pid, signal.SIGCONT)
    time.sleep(8)
    survivors = [pid for pid in pids if alive(pid)]
    for pid in survivors:
        send(pid, signal.SIGKILL)
    log(f"{reason}: terminated={len(pids) - len(survivors)}, killed={len(survivors)}, pids={pids}")


def bootstrap_cleanup(state: dict) -> None:
    if state.get("bootstrap_done"):
        return

    old_monitor_pids = pgrep(r"/tmp/codex_wsc_monitor.py") + pgrep(r"auto_scripts/wsc_health_monitor.py")
    terminate_processes(old_monitor_pids, "bootstrap stopped old monitors")

    old_scheduler_pids = pgrep(r"auto_scripts/wsc_continual_scheduler.sh")
    terminate_processes(old_scheduler_pids, "bootstrap stopped old queueing shells")

    items = dreamer_items()
    nonkeep = [item["pid"] for item in items if not is_old_keep(item)]
    terminate_processes(nonkeep, "bootstrap killed non-target old Dreamer runs")

    resumed = []
    resumed_at = time.time()
    for item in dreamer_items():
        if is_old_keep(item) and item["stat"].startswith("T"):
            send(item["pid"], signal.SIGCONT)
            resumed.append(item["pid"])
    if resumed:
        running_since = state.setdefault("running_since", {})
        for pid in resumed:
            running_since[str(pid)] = resumed_at
        log(f"bootstrap resumed preserved target runs pids={sorted(resumed)}")

    state["bootstrap_done"] = True
    state["bootstrap_at"] = time.time()


def gpu_info() -> dict[int, dict]:
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], text=True)
    except Exception as exc:
        log(f"gpu query failed: {exc}")
        return {}
    info = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        idx, free, total, util = parts[:4]
        info[int(idx)] = {
            "free": int(free),
            "total": int(total),
            "util": int(util),
        }
    return info


def ram_available_mb() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except Exception:
        return 0
    return 0


def latest_metrics(logdir: str) -> tuple[dict | None, float]:
    path = ROOT / logdir / "metrics.jsonl"
    if not path.exists():
        return None, 0.0
    last = None
    try:
        for line in path.read_text(errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if "fps/policy" in item or "fps/train" in item:
                last = item
    except Exception:
        return None, path.stat().st_mtime
    return last, path.stat().st_mtime


def latest_wandb_run(logdir: str) -> Path | None:
    root = ROOT / logdir / "wandb" / "wandb"
    if not root.exists():
        return None
    runs = [path for path in root.glob("run-*") if path.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda path: path.stat().st_mtime)


def wandb_health_issue(logdir: str, now: float, since: float, fresh_grace: int) -> str | None:
    run = latest_wandb_run(logdir)
    age = now - since
    if run is None:
        if age > fresh_grace:
            return f"no wandb run age={int(age)}s logdir={logdir}"
        return None

    files = [path for path in run.rglob("*") if path.is_file()]
    if files:
        latest_file_age = now - max(path.stat().st_mtime for path in files)
        if latest_file_age > fresh_grace:
            return f"stale wandb files age={int(latest_file_age)}s logdir={logdir}"

    debug = run / "logs" / "debug-internal.log"
    if not debug.exists():
        if age > fresh_grace:
            return f"missing wandb internal log age={int(age)}s logdir={logdir}"
        return None

    debug_age = now - debug.stat().st_mtime
    if debug_age > fresh_grace:
        return f"stale wandb internal log age={int(debug_age)}s logdir={logdir}"

    try:
        tail = debug.read_text(errors="ignore")[-20000:]
    except Exception:
        return None
    fatal_pos = max(tail.rfind("ERROR+4"), tail.lower().rfind("fatal error"))
    ok_pos = tail.rfind('"status":"200 OK"')
    if fatal_pos > ok_pos:
        return f"wandb filestream fatal logdir={logdir}"
    return None


def remote_wandb_repair(state: dict, items: list[dict], now: float) -> None:
    if REMOTE_WANDB_CHECK_SECONDS <= 0:
        return
    if now - float(state.get("last_remote_wandb_check", 0) or 0) < REMOTE_WANDB_CHECK_SECONDS:
        return
    state["last_remote_wandb_check"] = now

    local_env = parse_local_env()
    if not local_env.get("WANDB_API_KEY"):
        state["last_remote_wandb_issue"] = "missing local W&B API key"
        return

    try:
        import wandb
    except Exception as exc:
        state["last_remote_wandb_issue"] = f"import failed: {type(exc).__name__}: {exc}"
        return

    os.environ["WANDB_API_KEY"] = local_env["WANDB_API_KEY"]
    if local_env.get("WANDB_ENTITY"):
        os.environ["WANDB_ENTITY"] = local_env["WANDB_ENTITY"]
    os.environ.setdefault("WANDB_SILENT", "true")

    try:
        api = wandb.Api(timeout=REMOTE_WANDB_TIMEOUT)
        entity = local_env.get("WANDB_ENTITY") or api.viewer.entity
    except Exception as exc:
        state["last_remote_wandb_issue"] = f"api init failed: {type(exc).__name__}: {exc}"
        return

    repairs = []
    issues = []
    counts: dict[str, int] = {}
    for item in items:
        if item["stat"].startswith("T"):
            continue
        parts = Path(item["logdir"]).parts
        if len(parts) < 4:
            continue
        _prefix, project, group, run_name = parts[:4]
        latest = latest_wandb_run(item["logdir"])
        if latest is None:
            issues.append(f"{item['pid']} no local wandb run {item['logdir']}")
            continue
        run_id = latest.name.split("-")[-1]
        try:
            remote = api.run(f"{entity}/{project}/{run_id}")
        except Exception as exc:
            issues.append(f"{item['pid']} lookup failed {project}/{run_id}: {type(exc).__name__}")
            continue

        remote_state = str(remote.state).lower()
        counts[remote_state] = counts.get(remote_state, 0) + 1
        if remote.group != group or remote.name != run_name:
            issues.append(
                f"{item['pid']} remote naming mismatch {project}/{run_id} "
                f"group={remote.group} name={remote.name} expected={group}/{run_name}"
            )
        if remote_state in {"crashed", "failed"}:
            try:
                if remote.update_state("pending"):
                    repairs.append(f"{project}/{run_id} {remote_state}->pending")
            except Exception as exc:
                issues.append(
                    f"{item['pid']} state repair failed {project}/{run_id}: "
                    f"{type(exc).__name__}"
                )

    state["last_remote_wandb_state_counts"] = counts
    state["last_remote_wandb_repairs"] = repairs[-20:]
    state["last_remote_wandb_issues"] = issues[-20:]
    if repairs:
        log(f"remote W&B state repaired: {'; '.join(repairs[:6])}")
    if issues:
        log(f"remote W&B check issues: {'; '.join(issues[:4])}")


def tail_has_error(logdir: str) -> bool:
    path = ROOT / logdir / "train.log"
    if not path.exists():
        return False
    try:
        tail = path.read_text(errors="ignore")[-20000:]
    except Exception:
        return False
    return any(token in tail for token in ERROR_RE)


def update_running_since(state: dict, items: list[dict], now: float) -> None:
    running_since = state.setdefault("running_since", {})
    last_stat = state.setdefault("last_stat", {})
    health_below = state.setdefault("health_below", {})
    current = {str(item["pid"]) for item in items}
    for pid in list(running_since):
        if pid not in current:
            running_since.pop(pid, None)
            last_stat.pop(pid, None)
            health_below.pop(pid, None)
    for item in items:
        pid = str(item["pid"])
        stat = item["stat"]
        prev = last_stat.get(pid)
        is_running = not stat.startswith("T")
        was_running = bool(prev) and not prev.startswith("T")
        if is_running and not was_running:
            if prev is None:
                running_since[pid] = now - item.get("etimes", 0)
            else:
                running_since[pid] = now
            health_below.pop(pid, None)
        elif not is_running:
            running_since.pop(pid, None)
            health_below.pop(pid, None)
        last_stat[pid] = stat


def health_report(state: dict, items: list[dict], now: float) -> tuple[bool, list[str]]:
    running_since = state.setdefault("running_since", {})
    health_below = state.setdefault("health_below", {})
    bad = []
    summaries = []
    launch_blockers = []
    for item in items:
        if item["stat"].startswith("T"):
            continue
        metrics, mtime = latest_metrics(item["logdir"])
        since = float(running_since.get(str(item["pid"]), now - item.get("etimes", 0)))
        age = now - since
        is_crafter = is_crafter_logdir(item["logdir"])
        fresh_grace = CRAFTER_FRESH_GRACE_SECONDS if is_crafter else FRESH_GRACE_SECONDS
        wandb_issue = wandb_health_issue(item["logdir"], now, since, fresh_grace)
        if wandb_issue:
            bad.append(f"{item['pid']} {wandb_issue}")
            continue
        if metrics is None or mtime <= since:
            if age > fresh_grace:
                bad.append(f"{item['pid']} no fresh metrics age={int(age)}s logdir={item['logdir']}")
            continue
        metric_age = now - mtime
        if metric_age > fresh_grace:
            bad.append(
                f"{item['pid']} stale metrics age={int(metric_age)}s "
                f"logdir={item['logdir']}"
            )
            continue
        fps = metrics.get("fps/policy")
        step = metrics.get("step")
        if fps is None:
            continue
        try:
            fps_value = float(fps)
        except Exception:
            continue
        threshold = CRAFTER_THRESH_FPS if is_crafter else THRESH_FPS
        summaries.append(
            f"{item['pid']} step={step} fps={fps_value:.2f} "
            f"threshold={threshold:.2f} {item['logdir']}"
        )
        below = fps_value < threshold
        if below:
            launch_blockers.append(
                f"{item['pid']} fps={fps_value:.2f} threshold={threshold:.2f} "
                f"step={step} logdir={item['logdir']}"
            )
        key = str(item["pid"])
        prev = health_below.get(key, {})
        count = int(prev.get("count", 0) or 0)
        if prev.get("step") != step:
            count = count + 1 if below else 0
        health_below[key] = {
            "step": step,
            "count": count,
            "fps": fps_value,
            "threshold": threshold,
        }
        if below and count >= 2:
            bad.append(
                f"{item['pid']} fps={fps_value:.2f} threshold={threshold:.2f} "
                f"step={step} count={count} logdir={item['logdir']}"
            )
    state["last_fps_summary"] = summaries[-20:]
    state["last_low_fps_launch_blockers"] = launch_blockers[-20:]
    return not bad, bad


def mark_finished(state: dict, jobs: list[Job], items: list[dict]) -> None:
    by_logdir = {item["logdir"]: item for item in items}
    jobs_state = state.setdefault("jobs", {})
    jobs_by_logdir = {job.logdir: job for job in jobs}
    for key, record in list(jobs_state.items()):
        if record.get("status") != "running":
            continue
        logdir = record.get("logdir")
        if not logdir or logdir in by_logdir:
            continue
        job = jobs_by_logdir.get(logdir)
        metrics, _mtime = latest_metrics(logdir)
        step = int(metrics.get("step", 0)) if metrics else 0
        expected = job.steps if job else record.get("steps")
        if tail_has_error(logdir):
            attempts = int(record.get("attempts", 1))
            if attempts < MAX_ATTEMPTS:
                record["status"] = "queued"
                record["retry_after"] = time.time()
                log(f"job hit error pattern and queued for retry: {key} logdir={logdir} step={step} attempts={attempts}")
            else:
                record["status"] = "failed"
                record["failed_at"] = time.time()
                log(f"job failed with error pattern after max attempts: {key} logdir={logdir} step={step} attempts={attempts}")
        elif expected and step >= int(expected):
            record["status"] = "done"
            record["done_at"] = time.time()
            done = ROOT / logdir / "scheduler_done"
            done.parent.mkdir(parents=True, exist_ok=True)
            done.touch()
            log(f"job completed: {key} logdir={logdir} step={step}")
        else:
            attempts = int(record.get("attempts", 1))
            if attempts < MAX_ATTEMPTS:
                record["status"] = "queued"
                record["retry_after"] = time.time() + 1800
                log(f"job exited early and queued for retry: {key} logdir={logdir} step={step} attempts={attempts}")
            else:
                record["status"] = "failed"
                record["failed_at"] = time.time()
                log(f"job failed after max attempts: {key} logdir={logdir} step={step}")


def running_by_logdir(items: list[dict]) -> dict[str, dict]:
    return {item["logdir"]: item for item in items}


def active_counts(items: list[dict]) -> tuple[int, dict[int, int]]:
    total = 0
    by_gpu: dict[int, int] = {}
    for item in items:
        if item["stat"].startswith("T"):
            continue
        total += 1
        try:
            gpu = int(item["gpu"])
        except Exception:
            continue
        by_gpu[gpu] = by_gpu.get(gpu, 0) + 1
    return total, by_gpu


def choose_gpu(items: list[dict]) -> int | None:
    gpus = gpu_info()
    _total, counts = active_counts(items)
    candidates = []
    for idx, info in gpus.items():
        count = counts.get(idx, 0)
        if count >= MAX_RUNS_PER_GPU:
            continue
        if info["free"] < GPU_MIN_FREE_MB:
            continue
        used = max(0, info["total"] - info["free"])
        candidates.append((count, used / max(info["total"], 1), info["util"], idx))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][3]


def queued_jobs(state: dict, jobs: list[Job], items: list[dict], now: float) -> list[Job]:
    jobs_state = state.setdefault("jobs", {})
    by_logdir = running_by_logdir(items)
    result = []
    for job in jobs:
        done_file = ROOT / job.logdir / "scheduler_done"
        if done_file.exists():
            jobs_state.setdefault(job.key, {
                "status": "done",
                "logdir": job.logdir,
                "priority": job.priority,
                "label": job.label,
            })
            continue
        if job.logdir in by_logdir:
            jobs_state[job.key] = {
                **jobs_state.get(job.key, {}),
                "status": "running",
                "pid": by_logdir[job.logdir]["pid"],
                "logdir": job.logdir,
                "priority": job.priority,
                "label": job.label,
                "steps": job.steps,
            }
            continue
        record = jobs_state.get(job.key, {})
        status = record.get("status", "queued")
        if status == "done" or status == "failed":
            continue
        retry_after = float(record.get("retry_after", 0) or 0)
        if retry_after > now:
            continue
        result.append(job)
    order = {job.key: index for index, job in enumerate(jobs)}
    result.sort(key=lambda job: (job.priority, order[job.key]))
    return result


def protect_health(state: dict, bad: list[str]) -> None:
    jobs_state = state.setdefault("jobs", {})
    bad_pids = set()
    for item in bad:
        first = item.split(None, 1)[0]
        if first.isdigit():
            bad_pids.add(int(first))

    managed_pids = {
        int(record.get("pid") or 0)
        for record in jobs_state.values()
        if record.get("status") == "running"
    }
    unmanaged_bad = [
        pid for pid in sorted(bad_pids)
        if pid not in managed_pids and alive(pid)
    ]
    if unmanaged_bad:
        paused = []
        for pid in unmanaged_bad:
            send(pid, signal.SIGSTOP)
            paused.append(pid)
        log(f"health protection paused unmanaged low-FPS runs pids={paused}; bad={'; '.join(bad[:3])}")
        return

    candidates = []
    for key, record in jobs_state.items():
        if record.get("status") != "running":
            continue
        pid = int(record.get("pid") or 0)
        if not pid or not alive(pid):
            continue
        priority = int(record.get("priority") or 99)
        started = float(record.get("started_at") or 0)
        is_bad = pid in bad_pids
        candidates.append((0 if is_bad else 1, -priority, -started, key, record))

    if not candidates:
        return

    candidates.sort()
    _rank, _priority, _started, key, record = candidates[0]
    pid = int(record["pid"])
    terminate_processes([pid], f"health protection stopped managed job {key}")
    record["status"] = "queued"
    record["retry_after"] = time.time() + 1800
    record["stopped_for_health_at"] = time.time()
    log(f"health protection requeued {key} after FPS/freshness block; bad={'; '.join(bad[:3])}")


def launch_job(state: dict, job: Job, gpu: int) -> None:
    logdir_abs = ROOT / job.logdir
    logdir_abs.mkdir(parents=True, exist_ok=True)
    train_log = logdir_abs / "train.log"
    if train_log.exists() and train_log.stat().st_size > 0:
        backup = logdir_abs / f"train.{time.strftime('%Y%m%d_%H%M%S')}.log"
        train_log.rename(backup)

    cmd = [
        sys.executable,
        "dreamerv3/main.py",
        *job.args,
        "--logdir", job.logdir,
        "--seed", job.run.replace("seed_", ""),
        "--egl_device", str(gpu),
    ]

    env = os.environ.copy()
    env.update(parse_local_env())
    env.update({
        "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "MUJOCO_GL": "egl",
    })
    latest_wandb = sorted((logdir_abs / "wandb" / "wandb").glob("run-*"), reverse=True) if (logdir_abs / "wandb" / "wandb").exists() else []
    if latest_wandb and "WANDB_RUN_ID" not in env:
        run_id = latest_wandb[0].name.split("-")[-1]
        if run_id:
            env["WANDB_RUN_ID"] = run_id
            env["WANDB_RESUME"] = env.get("WANDB_RESUME", "must")

    with train_log.open("wb") as out:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    record = state.setdefault("jobs", {}).get(job.key, {})
    attempts = int(record.get("attempts", 0)) + 1
    state["jobs"][job.key] = {
        **record,
        "status": "running",
        "pid": proc.pid,
        "logdir": job.logdir,
        "priority": job.priority,
        "label": job.label,
        "steps": job.steps,
        "attempts": attempts,
        "started_at": time.time(),
        "gpu": gpu,
    }
    log(f"launched priority={job.priority} gpu={gpu} pid={proc.pid} {job.label} logdir={job.logdir}")


def compact_logs(state: dict) -> None:
    now = time.time()
    if not LOG.exists():
        return
    text = LOG.read_text(errors="ignore")
    lines = [line for line in text.splitlines() if line.strip()]
    if len(text) < 24000 and now - state.get("last_log_compact", 0) < LOG_COMPACT_SECONDS:
        return

    scheduler_lines = [line for line in lines if " | Scheduler:" in line]
    keep_other = [
        line for line in lines
        if " | Scheduler:" not in line and " | Background monitor:" not in line
    ]
    recent = scheduler_lines[-5:]
    jobs = state.get("jobs", {})
    counts = {}
    for record in jobs.values():
        counts[record.get("status", "queued")] = counts.get(record.get("status", "queued"), 0) + 1
    summary = (
        f"{stamp()} | Scheduler summary: active 2026-07-23 schedule; "
        f"job_status={counts}; recent details kept in last {len(recent)} scheduler records."
    )
    retained_prefix = keep_other[-20:]
    new_lines = retained_prefix + [summary] + recent
    LOG.write_text("\n".join(new_lines).rstrip() + "\n")
    state["last_log_compact"] = now


def sanity_files(state: dict) -> None:
    env_path = ROOT / ".env.wandb.local"
    gitignore = ROOT / ".gitignore"
    ignored = ".env*.local" in gitignore.read_text(errors="ignore") if gitignore.exists() else False
    state["wandb_local_config_present"] = env_path.exists()
    state["wandb_local_config_ignored"] = ignored


def loop_once(state: dict) -> None:
    now = time.time()
    sanity_files(state)
    bootstrap_cleanup(state)

    jobs = build_jobs()
    items = dreamer_items()
    update_running_since(state, items, now)
    mark_finished(state, jobs, items)

    ok, bad = health_report(state, items, now)
    if not ok:
        state["last_health_block"] = bad
        running_total, _counts = active_counts(items)
        state["last_loop_at"] = now
        state["last_running_total"] = running_total
        state["last_queue_len"] = 0
        protect_health(state, bad)
        log(f"launch blocked by FPS/freshness health: {'; '.join(bad[:4])}")
        compact_logs(state)
        return
    state.pop("last_health_block", None)
    remote_wandb_repair(state, items, now)

    low_fps_blockers = state.get("last_low_fps_launch_blockers") or []
    if low_fps_blockers:
        running_total, _counts = active_counts(items)
        queue = queued_jobs(state, build_jobs(), items, now)
        state["last_loop_at"] = now
        state["last_running_total"] = running_total
        state["last_queue_len"] = len(queue)
        log(f"launch blocked by low FPS warmup: {'; '.join(low_fps_blockers[:4])}")
        compact_logs(state)
        return

    ram_mb = ram_available_mb()
    if ram_mb and ram_mb < RAM_MIN_FREE_MB:
        state["last_ram_block"] = ram_mb
        running_total, _counts = active_counts(items)
        queue = queued_jobs(state, jobs, items, now)
        state["last_loop_at"] = now
        state["last_running_total"] = running_total
        state["last_queue_len"] = len(queue)
        log(f"launch blocked by RAM: available_mb={ram_mb}, required_mb={RAM_MIN_FREE_MB}")
        compact_logs(state)
        return

    running_total, _counts = active_counts(items)
    if running_total >= MAX_TOTAL_RUNNING:
        state["last_capacity_block"] = running_total
        queue = queued_jobs(state, jobs, items, now)
        state["last_loop_at"] = now
        state["last_running_total"] = running_total
        state["last_queue_len"] = len(queue)
        compact_logs(state)
        return

    queue = queued_jobs(state, jobs, items, now)
    launched = 0
    while queue and running_total < MAX_TOTAL_RUNNING and launched < LAUNCH_BATCH:
        gpu = choose_gpu(items)
        if gpu is None:
            state["last_gpu_block"] = gpu_info()
            log(f"launch blocked by GPU memory/count constraints: min_free_mb={GPU_MIN_FREE_MB}, max_per_gpu={MAX_RUNS_PER_GPU}")
            break
        job = queue.pop(0)
        launch_job(state, job, gpu)
        launched += 1
        time.sleep(LAUNCH_STAGGER)
        items = dreamer_items()
        running_total, _counts = active_counts(items)

    state["last_loop_at"] = now
    state["last_running_total"] = running_total
    state["last_queue_len"] = len(queue)
    if launched:
        log(f"launch loop completed launched={launched} running_total={running_total} remaining_queue={len(queue)}")
    compact_logs(state)


def main() -> None:
    SCHED_DIR.mkdir(parents=True, exist_ok=True)
    if PID_PATH.exists():
        try:
            old_pid = int(PID_PATH.read_text().strip())
        except Exception:
            old_pid = 0
        if old_pid and old_pid != os.getpid() and alive(old_pid):
            print(f"scheduler already running pid={old_pid}")
            return
    PID_PATH.write_text(str(os.getpid()) + "\n")
    state = load_state()
    state.setdefault("started_at", time.time())
    log("daemon started for 2026-07-23 schedule")
    while True:
        try:
            loop_once(state)
            save_state(state)
        except Exception as exc:
            log(f"loop error: {type(exc).__name__}: {exc}")
            save_state(state)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
