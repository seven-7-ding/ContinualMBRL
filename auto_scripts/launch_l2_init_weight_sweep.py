#!/usr/bin/env python3
"""Launch L2-init weight sweep runs and wait for first valid logging.

This script only tracks PIDs that it starts. It does not stop, pause, or alter
unrelated Dreamer processes already running on the machine.
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
RUN_VERSION = "l2init-sweep-v1"
STATE_FILE = ROOT / "logdir" / "scheduler" / f"l2_init_weight_sweep_{RUN_VERSION}.json"
WEIGHTS = ("2e-3", "2e-4", "2e-6")
SEEDS = (1000, 2000, 3000)
POLL_SECONDS = int(os.environ.get("CODEX_L2INIT_SWEEP_POLL_SECONDS", "60"))
STARTUP_STAGGER_SECONDS = float(os.environ.get("CODEX_L2INIT_SWEEP_STAGGER_SECONDS", "8"))
MAX_RESTARTS = int(os.environ.get("CODEX_L2INIT_SWEEP_MAX_RESTARTS", "2"))


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


def query_gpus() -> list[dict[str, int]]:
  cmd = [
      "nvidia-smi",
      "--query-gpu=index,memory.used,memory.total,utilization.gpu",
      "--format=csv,noheader,nounits",
  ]
  try:
    out = subprocess.check_output(cmd, text=True, timeout=15)
  except Exception:
    return [
        {"index": i, "memory_used": 0, "memory_total": 1, "utilization": 0}
        for i in range(8)
    ]
  gpus = []
  for line in out.splitlines():
    parts = [x.strip() for x in line.split(",")]
    if len(parts) != 4:
      continue
    index, memory_used, memory_total, utilization = [int(x) for x in parts]
    if 0 <= index <= 7:
      gpus.append({
          "index": index,
          "memory_used": memory_used,
          "memory_total": memory_total,
          "utilization": utilization,
      })
  return gpus or [
      {"index": i, "memory_used": 0, "memory_total": 1, "utilization": 0}
      for i in range(8)
  ]


def gpu_order() -> list[int]:
  gpus = query_gpus()
  gpus.sort(key=lambda x: (
      x["memory_used"] / max(x["memory_total"], 1),
      x["utilization"],
      x["index"],
  ))
  return [gpu["index"] for gpu in gpus]


def weight_label(weight: str) -> str:
  return weight


def wandb_weight_label(weight: str) -> str:
  return weight.replace("-", "m").replace("+", "p").replace(".", "p")


def build_jobs() -> list[dict[str, Any]]:
  order = gpu_order()
  jobs = []
  index = 0
  projects = (
      {
          "kind": "dmc",
          "project": "continual_dreamer_soft_reset_size1m",
      },
      {
          "kind": "crafter",
          "project": "continual_dreamer_soft_reset_crafter_size1m",
      },
  )
  for project in projects:
    for weight in WEIGHTS:
      group = f"l2_init_{weight_label(weight)}"
      for seed in SEEDS:
        gpu = order[index % len(order)]
        logdir = ROOT / "logdir" / project["project"] / group / f"seed_{seed}"
        jobs.append({
            "id": f"{project['kind']}/{group}/seed_{seed}",
            "kind": project["kind"],
            "project": project["project"],
            "group": group,
            "weight": weight,
            "seed": seed,
            "gpu": gpu,
            "logdir": str(logdir),
            "wandb_id": (
                f"baseline-{project['kind']}-l2-init-"
                f"{wandb_weight_label(weight)}-{seed}-{RUN_VERSION}"
            ),
            "pid": None,
            "restarts": 0,
            "status": "queued",
            "valid_logging": False,
            "last_reason": "not launched",
        })
        index += 1
  return jobs


def save_state(state: dict[str, Any]) -> None:
  STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
  tmp = STATE_FILE.with_suffix(".tmp")
  tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
  tmp.replace(STATE_FILE)


def append_event(state: dict[str, Any], message: str) -> None:
  state.setdefault("events", []).append({"time": now(), "message": message})
  state["events"] = state["events"][-80:]
  save_state(state)
  print(f"[{now()}] {message}", flush=True)


def pid_alive(pid: int | None) -> bool:
  if not pid:
    return False
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return False
  except PermissionError:
    return True
  return True


def wandb_has_output(logdir: pathlib.Path) -> bool:
  wandb_root = logdir / "wandb" / "wandb"
  if not wandb_root.exists():
    return False
  for path in wandb_root.glob("run-*/*.wandb"):
    try:
      if path.stat().st_size > 0:
        return True
    except OSError:
      continue
  return False


def last_metrics(logdir: pathlib.Path) -> dict[str, Any] | None:
  metrics = logdir / "metrics.jsonl"
  if not metrics.exists():
    return None
  last = None
  try:
    with metrics.open() as f:
      for line in f:
        line = line.strip()
        if line:
          last = json.loads(line)
  except (OSError, json.JSONDecodeError):
    return None
  return last


def has_prefix(metrics: dict[str, Any], prefix: str) -> bool:
  return any(key.startswith(prefix) for key in metrics)


def valid_logging(job: dict[str, Any]) -> tuple[bool, str]:
  logdir = pathlib.Path(job["logdir"])
  metrics = last_metrics(logdir)
  if metrics is None:
    return False, "missing metrics.jsonl"
  required_keys = (
      "opt/mechanism/l2_init/raw_loss",
      "opt/mechanism/l2_init/weighted_loss",
      "fps/policy",
  )
  missing = [key for key in required_keys if key not in metrics]
  required_prefixes = (
      "loss/mechanism/l2_init/module_delta_sq/",
  )
  missing.extend(prefix for prefix in required_prefixes if not has_prefix(metrics, prefix))
  if missing:
    step = metrics.get("step", "unknown")
    return False, f"metrics step={step} missing {missing[:3]}"
  try:
    fps_policy = float(metrics["fps/policy"])
  except Exception:
    return False, "invalid fps/policy"
  if fps_policy <= 0:
    return False, f"nonpositive fps/policy={fps_policy}"
  if not wandb_has_output(logdir):
    return False, "missing W&B local .wandb output"
  analysis = []
  for prefix in ("act_redo/", "grad_redo/", "data_diversity/"):
    if has_prefix(metrics, prefix):
      analysis.append(prefix.rstrip("/"))
  suffix = f" analysis={'+'.join(analysis)}" if analysis else " analysis=pending_by_frequency"
  return True, f"valid step={metrics.get('step')} fps_policy={fps_policy:.2f}{suffix}"


def base_command(job: dict[str, Any]) -> list[str]:
  common = [
      str(PYTHON), "-u", "dreamerv3/main.py",
      "--run.reset_frequency", "0",
      "--run.reset_mechanism", "l2_init",
      "--run.reset_target", "all",
      "--run.revive_epoch", "0",
      "--agent.wsc.mechanism", "l2_init",
      "--agent.wsc.target", "all",
      "--agent.wsc.weight_decay", job["weight"],
      "--agent.wsc.l2_init_weight", job["weight"],
      "--agent.redo.redo_enabled", "True",
      "--agent.redo.grad_redo_enabled", "True",
      "--agent.redo.act_log_item", "log+erank+srank",
      "--agent.redo.grad_log_item", "log+erank+srank",
      "--logdir", job["logdir"],
      "--seed", str(job["seed"]),
      "--egl_device", str(job["gpu"]),
  ]
  if job["kind"] == "dmc":
    return [
        str(PYTHON), "-u", "dreamerv3/main.py",
        "--configs", "continual_dmc_priori", "size1m",
        "--task", "walker_run|hopper_hop|fish_swim",
        "--run.train_ratio", "1024",
        "--run.task_interval", "1000000",
        "--run.task_repeat", "5",
        "--run.steps", "15000000",
        "--run.reset_frequency", "0",
        "--run.reset_mechanism", "l2_init",
        "--run.reset_target", "all",
        "--run.revive_epoch", "0",
        "--run.revive_strategy", "threshold",
        "--env.continual_dmc_priori.obs_dim", "24",
        "--env.continual_dmc_priori.task_action_space", "6",
        "--replay.chunksize", "4096",
        "--replay.cache_chunks", "4096",
        "--agent.imag_length", "15",
        "--agent.wsc.mechanism", "l2_init",
        "--agent.wsc.target", "all",
        "--agent.wsc.weight_decay", job["weight"],
        "--agent.wsc.l2_init_weight", job["weight"],
        "--agent.wsc.cbp_eta", "0.99",
        "--agent.wsc.cbp_maturity", "5000",
        "--agent.wsc.cbp_replacement_rate", "1e-4",
        "--agent.redo.redo_enabled", "True",
        "--agent.redo.grad_redo_enabled", "True",
        "--agent.redo.act_log_item", "log+erank+srank",
        "--agent.redo.grad_log_item", "log+erank+srank",
        "--run.save_every", "1800",
        "--logdir", job["logdir"],
        "--seed", str(job["seed"]),
        "--egl_device", str(job["gpu"]),
    ]
  if job["kind"] == "crafter":
    return [
        str(PYTHON), "-u", "dreamerv3/main.py",
        "--configs", "crafter", "size1m",
        "--run.steps", "100000000",
        "--run.train_ratio", "512",
        "--run.task_interval", "100000000",
        "--run.log_every", "10000",
        "--run.report_every", "20000",
        "--run.save_every", "900",
        "--run.envs", "1",
        "--run.revive_strategy", "fixed",
    ] + common[3:]
  raise ValueError(job["kind"])


def launch(job: dict[str, Any], base_env: dict[str, str]) -> subprocess.Popen:
  logdir = pathlib.Path(job["logdir"])
  logdir.mkdir(parents=True, exist_ok=True)
  env = base_env.copy()
  env.update({
      "CUDA_VISIBLE_DEVICES": str(job["gpu"]),
      "MUJOCO_EGL_DEVICE_ID": str(job["gpu"]),
      "WANDB_RUN_ID": job["wandb_id"],
      "WANDB_RESUME": "allow",
  })
  cmd = base_command(job)
  train_log = logdir / "train.log"
  fh = train_log.open("ab", buffering=0)
  fh.write((f"\n[{now()}] Launch: {' '.join(shlex.quote(x) for x in cmd)}\n").encode())
  proc = subprocess.Popen(
      cmd,
      cwd=ROOT,
      env=env,
      stdout=fh,
      stderr=subprocess.STDOUT,
      start_new_session=True,
  )
  job["pid"] = proc.pid
  job["status"] = "running"
  job["last_launch"] = now()
  job["last_reason"] = "launched"
  return proc


def run() -> None:
  if not PYTHON.exists():
    raise SystemExit(f"Missing required dreamer env python: {PYTHON}")
  state = {
      "run_version": RUN_VERSION,
      "started": now(),
      "initial_gpus": query_gpus(),
      "jobs": build_jobs(),
      "events": [],
  }
  save_state(state)
  base_env = load_env()
  procs: dict[str, subprocess.Popen] = {}
  append_event(state, f"launch_start jobs={len(state['jobs'])} gpu_order={gpu_order()}")
  for job in state["jobs"]:
    procs[job["id"]] = launch(job, base_env)
    append_event(
        state,
        f"launched {job['id']} pid={job['pid']} gpu={job['gpu']} weight={job['weight']}",
    )
    if STARTUP_STAGGER_SECONDS > 0:
      time.sleep(STARTUP_STAGGER_SECONDS)

  while True:
    all_valid = True
    for job in state["jobs"]:
      proc = procs.get(job["id"])
      if proc is not None and proc.poll() is not None:
        code = proc.returncode
        if job["restarts"] >= MAX_RESTARTS:
          job["status"] = "blocked"
          job["last_reason"] = f"process exited code={code}; max restarts reached"
          append_event(state, f"blocked {job['id']} code={code}")
          raise SystemExit(f"{job['id']} blocked after exit code {code}")
        job["restarts"] += 1
        append_event(state, f"restarting {job['id']} after exit code={code}")
        procs[job["id"]] = launch(job, base_env)
        all_valid = False
        continue
      if not pid_alive(job.get("pid")):
        job["status"] = "missing"
        job["last_reason"] = "pid disappeared"
        append_event(state, f"pid_missing {job['id']} pid={job.get('pid')}")
        all_valid = False
        continue
      ok, reason = valid_logging(job)
      job["valid_logging"] = ok
      job["last_reason"] = reason
      if ok:
        if job["status"] != "valid_logging":
          job["status"] = "valid_logging"
          append_event(state, f"valid_logging {job['id']} pid={job['pid']} {reason}")
      else:
        all_valid = False
    save_state(state)
    if all_valid:
      append_event(state, "all_runs_have_valid_logging")
      return
    pending = [job["id"] for job in state["jobs"] if not job.get("valid_logging")]
    print(f"[{now()}] waiting pending={len(pending)} sample={pending[:6]}", flush=True)
    time.sleep(POLL_SECONDS)


if __name__ == "__main__":
  try:
    run()
  except KeyboardInterrupt:
    print(f"[{now()}] interrupted; launched training processes remain in background", flush=True)
    raise
