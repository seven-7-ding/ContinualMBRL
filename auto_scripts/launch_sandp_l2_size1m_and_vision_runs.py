#!/usr/bin/env python3
"""Launch and monitor the requested baseline sandp/l2 experiments.

Experiment set:
1) size1m continual fish_swim:
   sandp_all alpha=0.8 with reset frequencies 20000/50000/100000

2) cheetah vision size1m_1m_x5:
   - l2_init_2e-5
   - l2_init_2e-6
   - sandp_all alpha=0.8 with frequencies 20000/50000/100000

The script runs jobs in the foreground, keeps resources above configured safety
limits, keeps GPU0-7 balanced by memory utilization, and supports resume from
existing logdirs/checkpoints.
"""

from __future__ import annotations

import json
import os
import pathlib
import shlex
import subprocess
import time
from datetime import datetime
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
PYTHON = pathlib.Path("/home/jiale/.conda/envs/dreamer/bin/python")
ENV_FILE = ROOT / ".env.wandb.local"
STATE_FILE = ROOT / "logdir" / "scheduler" / "sandp_l2_size1m_and_vision_v1.json"
RUN_VERSION = "sandp-l2-v1"

SEEDS = (1000, 2000, 3000)
MIN_VALID_LOGS = 5
MIN_RAM_GIB = 40.0
GPU_STALE_SECONDS = 900
POLL_SECONDS = int(os.environ.get("CODEX_SANDP_L2_POLL_SECONDS", "60"))
MAX_RESTARTS = int(os.environ.get("CODEX_SANDP_L2_MAX_RESTARTS", "12"))
MAX_GPU_MEM_MB = int(os.environ.get("CODEX_SANDP_L2_MAX_GPU_MB", "16000"))

SIZE1M_PROJECT = "continual_dreamer_soft_reset_size1m"
VISION_PROJECT = (
    "continual_dreamer_soft_reset_walker_run|hopper_hop|cheetah_run_vision_size1m_1m_x5"
)


def now() -> str:
  return datetime.now().strftime("%Y-%m-%d %H:%M:%S HKT")


def load_env() -> dict[str, str]:
  env = os.environ.copy()
  env.setdefault("MUJOCO_GL", "egl")
  env.setdefault("PYOPENGL_PLATFORM", "egl")
  if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
      line = line.strip()
      if not line or line.startswith("#") or "=" not in line:
        continue
      key, value = line.split("=", 1)
      env[key.strip()] = value.strip().strip('"').strip("'")
  return env


def run_cmd(cmd: list[str], timeout: int = 15) -> str:
  return subprocess.check_output(cmd, text=True, timeout=timeout)


def query_gpus() -> list[dict[str, int]]:
  try:
    out = run_cmd([
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
  except Exception:
    return [{"index": i, "memory_used": 0, "memory_total": 1, "utilization": 0} for i in range(8)]
  rows = []
  for line in out.splitlines():
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 4:
      continue
    try:
      index, used, total, util = [int(part) for part in parts]
    except ValueError:
      continue
    if 0 <= index <= 7:
      rows.append({
          "index": index,
          "memory_used": used,
          "memory_total": total,
          "utilization": util,
      })
  if not rows:
    return [{"index": i, "memory_used": 0, "memory_total": 1, "utilization": 0} for i in range(8)]
  return rows


def gpu_order() -> list[int]:
  rows = query_gpus()
  rows.sort(key=lambda row: (row["memory_used"] / max(row["memory_total"], 1), row["utilization"], row["index"]))
  return [row["index"] for row in rows]


def gpu_memory_by_pid() -> dict[int, int]:
  try:
    out = run_cmd([
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ], timeout=15)
  except Exception:
    return {}
  mem = {}
  for line in out.splitlines():
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 2:
      continue
    try:
      pid = int(parts[0])
      used = int(parts[1])
    except ValueError:
      continue
    mem[pid] = mem.get(pid, 0) + used
  return mem


def proc_alive(pid: int | None) -> bool:
  if not pid:
    return False
  try:
    out = subprocess.check_output(["ps", "-p", str(pid), "-o", "stat="], text=True, timeout=5).strip()
  except subprocess.CalledProcessError:
    return False
  except Exception:
    return True
  return bool(out) and not out.startswith("Z")


def stop_process(job: dict[str, Any], signal: int = 15) -> None:
  pid = job.get("pid")
  if not pid:
    return
  try:
    os.killpg(int(pid), signal)
  except ProcessLookupError:
    return
  except Exception:
    try:
      os.kill(int(pid), signal)
    except ProcessLookupError:
      return


def memory_available_gib() -> float:
  try:
    meminfo = pathlib.Path("/proc/meminfo").read_text()
  except Exception:
    return 0.0
  kb = None
  for line in meminfo.splitlines():
    if line.startswith("MemAvailable:"):
      parts = line.split()
      if len(parts) >= 2:
        kb = float(parts[1])
        break
  if kb is None:
    for line in meminfo.splitlines():
      if line.startswith("MemFree:"):
        parts = line.split()
        if len(parts) >= 2:
          kb = float(parts[1])
          break
  if kb is None:
    return 0.0
  return kb / (1024 * 1024)


def has_wandb_output(logdir: pathlib.Path) -> bool:
  root = logdir / "wandb" / "wandb"
  if not root.exists():
    return False
  return any(root.glob("run-*/run-*.wandb")) or any(root.glob("run-*/logs/debug-internal.log"))


def iter_metrics(logdir: pathlib.Path):
  path = logdir / "metrics.jsonl"
  if not path.exists():
    return
  try:
    with path.open() as stream:
      for line in stream:
        line = line.strip()
        if not line:
          continue
        try:
          yield json.loads(line)
        except json.JSONDecodeError:
          continue
  except OSError:
    return


def count_valid_lines(logdir: pathlib.Path, requires_data_aug: bool) -> tuple[int, dict[str, Any] | None, float | None]:
  count = 0
  latest = None
  latest_mtime = None
  path = logdir / "metrics.jsonl"
  if path.exists():
    try:
      latest_mtime = path.stat().st_mtime
    except OSError:
      latest_mtime = None
  for metrics in iter_metrics(logdir):
    if "fps/policy" not in metrics:
      continue
    try:
      fps = float(metrics["fps/policy"])
    except (TypeError, ValueError):
      continue
    if fps < 4.0:
      continue
    has_act = any(key.startswith("act_redo/") for key in metrics)
    has_grad = any(key.startswith("grad_redo/") for key in metrics)
    if not (has_act and has_grad):
      continue
    if requires_data_aug:
      if (
          float(metrics.get("train/data_augmentation/active", 0.0)) != 1.0
          or float(metrics.get("train/data_augmentation/batch_align", 0.0)) != 1.0
      ):
        continue
    count += 1
    latest = metrics
  return count, latest, latest_mtime


def base_size1m_command(seed: int, gpu: int, logdir: pathlib.Path, mechanism: str, reset_freq: int, l2_weight: str | None, alpha: float | None) -> list[str]:
  command = [
      str(PYTHON), "-u", "dreamerv3/main.py",
      "--configs", "continual_dmc_priori", "size1m",
      "--task", "walker_run|hopper_hop|fish_swim",
      "--run.steps", "15000000",
      "--run.task_interval", "1000000",
      "--run.task_repeat", "5",
      "--run.train_ratio", "1024",
      "--run.log_every", "10000",
      "--run.save_every", "1800",
      "--run.reset_frequency", str(reset_freq),
      "--run.reset_mechanism", mechanism,
      "--run.reset_target", "all",
      "--run.revive_epoch", "0",
      "--run.revive_strategy", "threshold",
      "--env.continual_dmc_priori.obs_dim", "24",
      "--env.continual_dmc_priori.task_action_space", "6",
      "--agent.wsc.target", "all",
      "--agent.wsc.weight_decay", "0.0002",
      "--agent.wsc.l2_init_weight", "0.0002",
      "--agent.wsc.cbp_eta", "0.99",
      "--agent.wsc.cbp_maturity", "5000",
      "--agent.wsc.cbp_replacement_rate", "0.0001",
      "--agent.redo.redo_enabled", "True",
      "--agent.redo.grad_redo_enabled", "True",
      "--agent.redo.log_item", "log+erank+srank",
      "--agent.redo.act_log_item", "log+erank+srank",
      "--agent.redo.grad_log_item", "log+erank+srank",
      "--jax.memory_fraction", "0.1",
      "--jax.prealloc", "False",
      "--replay.chunksize", "4096",
      "--agent.imag_length", "15",
      "--logdir", str(logdir),
      "--seed", str(seed),
      "--egl_device", str(gpu),
  ]
  command.extend(["--agent.wsc.mechanism", mechanism])
  if mechanism in {"sandp", "sandp_wo_opt"} and alpha is not None:
    command.extend(["--run.reset_alpha", f"{alpha}"])
  if l2_weight:
    # Keep default 2e-5 and 2e-6 settings explicit for comparability.
    command[command.index("--agent.wsc.weight_decay") + 1] = l2_weight
    command[command.index("--agent.wsc.l2_init_weight") + 1] = l2_weight
  return command


def base_vision_command(seed: int, gpu: int, logdir: pathlib.Path, mechanism: str, reset_freq: int, l2_weight: str | None, alpha: float | None) -> list[str]:
  command = [
      str(PYTHON), "-u", "dreamerv3/main.py",
      "--configs", "dmc_vision", "continual_dmc_vision", "size12m",
      "--task", "walker_run|hopper_hop|cheetah_run",
      "--run.steps", "15000000",
      "--run.task_interval", "1000000",
      "--run.task_repeat", "5",
      "--run.train_ratio", "256",
      "--run.log_every", "10000",
      "--run.report_every", "20000",
      "--run.save_every", "900",
      "--run.reset_frequency", str(reset_freq),
      "--run.reset_mechanism", mechanism,
      "--run.reset_target", "all",
      "--run.revive_epoch", "0",
      "--run.revive_strategy", "fixed",
      "--agent.redo.redo_enabled", "True",
      "--agent.redo.grad_redo_enabled", "True",
      "--agent.redo.act_log_item", "log+erank+srank",
      "--agent.redo.grad_log_item", "log+erank+srank",
      "--agent.redo.log_item", "log+erank+srank",
      "--agent.data_augmentation.mode", "batch_align",
      "--agent.data_augmentation.pad", "4",
      "--env.continual_dmc.task_action_space", "6",
      "--agent.wsc.target", "all",
      "--agent.wsc.weight_decay", "0.0002",
      "--agent.wsc.l2_init_weight", "0.0002",
      "--agent.wsc.cbp_eta", "0.99",
      "--agent.wsc.cbp_maturity", "5000",
      "--agent.wsc.cbp_replacement_rate", "0.0001",
      "--jax.memory_fraction", "0.25",
      "--jax.prealloc", "False",
      "--logdir", str(logdir),
      "--seed", str(seed),
      "--egl_device", str(gpu),
  ]
  command.extend(["--agent.wsc.mechanism", mechanism])
  if l2_weight:
    command[command.index("--agent.wsc.weight_decay") + 1] = l2_weight
    command[command.index("--agent.wsc.l2_init_weight") + 1] = l2_weight
  if mechanism in {"sandp", "sandp_wo_opt"} and alpha is not None:
    command.extend(["--run.reset_alpha", f"{alpha}"])
  return command


def group_specs() -> list[dict[str, Any]]:
  specs = []
  # Size1m fish_swim: only sandp group is newly requested here.
  for reset_freq in (20000, 50000, 100000):
    specs.append({
        "project": SIZE1M_PROJECT,
        "group": f"sandp_all_a0p8_{int(reset_freq/1000)}k",
        "mechanism": "sandp",
        "reset_frequency": reset_freq,
        "reset_alpha": 0.8,
        "l2_weight": None,
        "task_type": "size1m",
        "requires_data_aug": False,
        "seeds": SEEDS,
    })
  for weight in ("2e-5", "2e-6"):
    specs.append({
        "project": VISION_PROJECT,
        "group": f"l2_init_{weight}",
        "mechanism": "l2_init",
        "reset_frequency": 0,
        "reset_alpha": None,
        "l2_weight": weight,
        "task_type": "vision",
        "requires_data_aug": True,
        "seeds": SEEDS,
    })
  for reset_freq in (20000, 50000, 100000):
    specs.append({
        "project": VISION_PROJECT,
        "group": f"sandp_all_a0p8_{int(reset_freq/1000)}k",
        "mechanism": "sandp",
        "reset_frequency": reset_freq,
        "reset_alpha": 0.8,
        "l2_weight": None,
        "task_type": "vision",
        "requires_data_aug": True,
        "seeds": SEEDS,
    })
  return specs


def build_jobs() -> list[dict[str, Any]]:
  order = gpu_order()
  jobs = []
  for spec in group_specs():
    for seed in spec["seeds"]:
      logdir = ROOT / "logdir" / spec["project"] / spec["group"] / f"seed_{seed}"
      jobs.append({
          "id": f'{spec["project"]}/{spec["group"]}/seed_{seed}',
          "project": spec["project"],
          "group": spec["group"],
          "task_type": spec["task_type"],
          "seed": seed,
          "mechanism": spec["mechanism"],
          "reset_frequency": spec["reset_frequency"],
          "reset_alpha": spec["reset_alpha"],
          "l2_weight": spec["l2_weight"],
          "requires_data_aug": spec["requires_data_aug"],
          "gpu": order[len(jobs) % len(order)],
          "logdir": str(logdir),
          "wandb_id": (
              f"baseline-{spec['project'].split('_')[-1]}-"
              f"{spec['group'].replace('_', '-')}-{seed}-{RUN_VERSION}"
          ),
          "pid": None,
          "status": "queued",
          "restarts": 0,
          "valid_logs": 0,
          "last_check": None,
          "last_detail": "queued",
          "last_valid_step": None,
          "stopped_at": None,
      })
  return jobs


def command_for(job: dict[str, Any]) -> list[str]:
  logdir = pathlib.Path(job["logdir"])
  if job["task_type"] == "size1m":
    return base_size1m_command(
        seed=job["seed"],
        gpu=job["gpu"],
        logdir=logdir,
        mechanism=job["mechanism"],
        reset_freq=job["reset_frequency"],
        l2_weight=job["l2_weight"],
        alpha=job["reset_alpha"],
    )
  if job["task_type"] == "vision":
    return base_vision_command(
        seed=job["seed"],
        gpu=job["gpu"],
        logdir=logdir,
        mechanism=job["mechanism"],
        reset_freq=job["reset_frequency"],
        l2_weight=job["l2_weight"],
        alpha=job["reset_alpha"],
    )
  raise ValueError(f'Unknown task_type={job["task_type"]}')


def save_state(state: dict[str, Any]) -> None:
  STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
  tmp = STATE_FILE.with_suffix(".tmp")
  tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
  tmp.replace(STATE_FILE)


def load_state() -> dict[str, Any]:
  if STATE_FILE.exists():
    try:
      data = json.loads(STATE_FILE.read_text())
      if isinstance(data, dict) and "jobs" in data:
        return data
    except Exception:
      pass
  jobs = build_jobs()
  return {
      "project": f"{SIZE1M_PROJECT}|{VISION_PROJECT}",
      "run_version": RUN_VERSION,
      "created_at": now(),
      "jobs": jobs,
      "events": [],
  }


def note(state: dict[str, Any], message: str) -> None:
  state.setdefault("events", []).append({"time": now(), "message": message})
  state["events"] = state["events"][-200:]
  save_state(state)
  print(f"[{now()}] {message}", flush=True)


def build_logline(job: dict[str, Any], command: list[str]) -> str:
  return " ".join(shlex.quote(part) for part in command)


def launch(job: dict[str, Any], env: dict[str, str], state: dict[str, Any]) -> None:
  logdir = pathlib.Path(job["logdir"])
  logdir.mkdir(parents=True, exist_ok=True)
  command = command_for(job)
  with (logdir / "train.log").open("ab", buffering=0) as stream:
    stream.write((f"\n[{now()}] Launch command: {build_logline(job, command)}\n").encode())
    job_env = env.copy()
    job_env.update({
        "CUDA_VISIBLE_DEVICES": str(job["gpu"]),
        "MUJOCO_EGL_DEVICE_ID": str(job["gpu"]),
        "WANDB_RUN_ID": job["wandb_id"],
        "WANDB_RESUME": "allow",
    })
    proc = subprocess.Popen(
        command,
        cwd=ROOT,
        env=job_env,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
  job.update({
      "pid": proc.pid,
      "status": "running",
      "last_launch": now(),
      "last_detail": "launched",
      "command": [str(part) for part in command],
      "stopped_at": None,
  })
  note(state, f"launched {job['id']} pid={proc.pid} gpu={job['gpu']}")


def check_resources_before_start() -> bool:
  return memory_available_gib() >= MIN_RAM_GIB


def maybe_kill_low_priority_jobs(jobs: list[dict[str, Any]], needed: int, state: dict[str, Any]) -> None:
  if memory_available_gib() >= MIN_RAM_GIB:
    return
  victims = [job for job in reversed(jobs) if job.get("status") == "running"]
  for job in victims:
    if memory_available_gib() >= MIN_RAM_GIB or needed <= 0:
      break
    stop_process(job, 15)
    job["status"] = "preempted_for_ram"
    job["stopped_at"] = now()
    job["last_detail"] = "stopped to recover memory"
    note(state, f"preempted {job['id']} pid={job.get('pid')} for RAM pressure")
    needed -= 1


def main():
  if not PYTHON.exists():
    raise SystemExit(f"Missing python at {PYTHON}")
  state = load_state()
  existing_ids = {job["id"] for job in state.get("jobs", [])}
  if state.get("jobs"):
    # Merge in any new specs after code change/restart.
    for job in build_jobs():
      if job["id"] not in existing_ids:
        state["jobs"].append(job)
  else:
    state["jobs"] = build_jobs()
  env = load_env()
  xla_flags = env.get("XLA_FLAGS", "")
  if "--xla_gpu_strict_conv_algorithm_picker=false" not in xla_flags:
    env["XLA_FLAGS"] = (xla_flags + " --xla_gpu_strict_conv_algorithm_picker=false").strip()
  save_state(state)
  note(state, f"prepared {len(state['jobs'])} jobs")

  total = len(state["jobs"])
  completed = 0
  while True:
    gpu_mem_by_pid = gpu_memory_by_pid()
    if not check_resources_before_start():
      note(state, f"RAM pressure ({memory_available_gib():.2f}GiB); preempting low-priority jobs")
      maybe_kill_low_priority_jobs(state["jobs"], len(state["jobs"]), state)

    # assign free/least-loaded GPUs to pending jobs in order when launching.
    order = gpu_order()

    # launch queued jobs while keeping at most one active job per GPU.
    running_jobs = [job for job in state["jobs"] if job["status"] == "running"]
    running_on_gpu = {int(job["gpu"]) for job in running_jobs if job.get("gpu") is not None}
    for job in state["jobs"]:
      if job["status"] in ("done", "done_with_fallback"):
        continue
      if proc_alive(job.get("pid")):
        job["status"] = "running"
      else:
        if job["status"] == "running":
          job["status"] = "stopped"
        if job["status"] == "queued":
          if not check_resources_before_start():
            continue
          if job["gpu"] in running_on_gpu:
            gpu = next((idx for idx in order if idx not in running_on_gpu), job["gpu"])
          else:
            gpu = job["gpu"]
          if gpu in running_on_gpu and len(running_jobs) >= 8:
            continue
          job["gpu"] = gpu
          running_on_gpu.add(gpu)
          launch(job, env, state)
          running_jobs.append(job)
          continue
        if job["status"] in ("stopped", "failed", "killed_fps", "preempted_for_ram"):
          if job["restarts"] < MAX_RESTARTS:
            if not check_resources_before_start():
              continue
            if int(job["restarts"]) >= 1:
              note(state, f"resuming {job['id']} restart={job['restarts']+1}")
            launch(job, env, state)
            job["restarts"] += 1
            running_jobs.append(job)
          else:
            job["status"] = "blocked"
            note(state, f"blocked {job['id']} restart limit exceeded")

    # update status and metrics for running jobs
    now_ts = time.time()
    completed = 0
    for job in state["jobs"]:
      logdir = pathlib.Path(job["logdir"])
      status = job.get("status", "queued")
      valid_count, latest, mtime = count_valid_lines(logdir, job["requires_data_aug"])
      job["valid_logs"] = valid_count
      job["last_check"] = now()
      if status == "running":
        if not proc_alive(job.get("pid")):
          if valid_count >= MIN_VALID_LOGS:
            job["status"] = "done"
            completed += 1
            note(state, f"done (valid logs reached before exit) {job['id']} pid={job.get('pid')}")
            continue
          job["status"] = "failed"
          job["stopped_at"] = now()
          job["last_detail"] = f"died before valid logs ({valid_count}/{MIN_VALID_LOGS})"
          note(state, f"failed {job['id']} pid={job.get('pid')} status={job['last_detail']}")
          continue
        if latest is None:
          job["last_detail"] = "no_valid_metrics yet"
          continue
        job["last_valid_step"] = latest.get("step")
        try:
          fps = float(latest.get("fps/policy", latest.get("fps/train", 0)))
        except (TypeError, ValueError):
          fps = 0.0
        if fps < 4.0:
          job["status"] = "killed_fps"
          job["last_detail"] = f"fps={fps:.2f} < 4.0"
          stop_process(job, 15)
          note(state, f"killed {job['id']} pid={job.get('pid')} due low fps {fps:.2f}")
          continue
        if job["requires_data_aug"] and (
            float(latest.get("train/data_augmentation/active", -1.0)) != 1.0
            or float(latest.get("train/data_augmentation/batch_align", -1.0)) != 1.0
        ):
          job["last_detail"] = "data_augmentation metrics disabled unexpectedly"
          stop_process(job, 15)
          note(state, f"killed {job['id']} for invalid DA metrics")
          continue
        if mtime and now_ts - mtime > GPU_STALE_SECONDS:
          if valid_count == 0:
            job["status"] = "failed"
            job["last_detail"] = "metrics stale before first valid entry"
            stop_process(job, 15)
            note(state, f"stale no-logging metrics for {job['id']} -> restart")
            continue
          # keep running if valid logs already exist
        if not has_wandb_output(logdir):
          job["last_detail"] = "wandb local output missing"
          stop_process(job, 15)
          note(state, f"missing wandb output, restarting {job['id']}")
          job["status"] = "failed"
          continue
        job["status"] = "running"
        if valid_count >= MIN_VALID_LOGS:
          job["status"] = "done"
          job["last_detail"] = f"valid_logs={valid_count} step={latest.get('step')} fps={fps:.2f}"
          completed += 1
          note(state, f"done {job['id']} step={latest.get('step')} fps={fps:.2f} logs={valid_count}")
        else:
          job["last_detail"] = f"valid_logs={valid_count}/{MIN_VALID_LOGS} step={latest.get('step')} fps={fps:.2f}"
        mem = gpu_mem_by_pid.get(job.get("pid"), 0)
        if mem > MAX_GPU_MEM_MB:
          job["last_detail"] += f" | gpu_mem={mem}MB"
          note(state, f"high gpu memory for {job['id']} pid={job.get('pid')} mem={mem}MB; killing")
          stop_process(job, 15)
          job["status"] = "failed"
      elif status in ("queued", "failed", "stopped", "preempted_for_ram", "killed_fps"):
        pass
      elif status == "done":
        completed += 1

    save_state(state)
    remaining = sum(1 for job in state["jobs"] if job["status"] != "done")
    if remaining == 0:
      print(f"[{now()}] all jobs reached {MIN_VALID_LOGS} valid logging entries", flush=True)
      return
    for idx, job in enumerate(state["jobs"], start=1):
      if job["status"] != "done":
        print(
            f"[{now()}] [{idx}/{total}] {job['id']} status={job['status']} "
            f"gpu={job['gpu']} pid={job.get('pid')} valid={job['valid_logs']}/{MIN_VALID_LOGS} "
            f"{job.get('last_detail')}",
            flush=True,
        )
    print(f"[{now()}] completed={completed} remaining={remaining}", flush=True)
    time.sleep(POLL_SECONDS)


if __name__ == "__main__":
  main()
