#!/usr/bin/env python3
"""Launch the requested Crafter and dog-series l2_init runs.

This script only starts the six requested runs and records their PIDs. It does
not manage or signal unrelated processes.
"""

import json
import os
import pathlib
import shlex
import subprocess
from datetime import datetime


ROOT = pathlib.Path(__file__).resolve().parents[1]
PYTHON = pathlib.Path("/home/jiale/.conda/envs/dreamer/bin/python")
ENV_FILE = ROOT / ".env.wandb.local"
RUN_VERSION = "correct-project-v1"
STATE_FILE = ROOT / "logdir" / "scheduler" / f"crafter_dog_l2_init_{RUN_VERSION}.json"
SEEDS = (1000, 2000, 3000)
GPU_ASSIGNMENTS = {
    ("crafter", 1000): 0,
    ("crafter", 2000): 3,
    ("crafter", 3000): 5,
    ("dog", 1000): 7,
    ("dog", 2000): 6,
    ("dog", 3000): 4,
}


def now():
  return datetime.now().strftime("%Y-%m-%d %H:%M:%S HKT")


def load_env():
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


def command(kind, seed, logdir, gpu):
  common = [
      str(PYTHON), "dreamerv3/main.py",
      "--run.reset_frequency", "0",
      "--run.reset_mechanism", "l2_init",
      "--run.reset_target", "all",
      "--run.revive_epoch", "0",
      "--agent.wsc.mechanism", "l2_init",
      "--agent.wsc.target", "all",
      "--agent.wsc.weight_decay", "2e-5",
      "--agent.wsc.l2_init_weight", "2e-5",
      "--agent.redo.redo_enabled", "True",
      "--agent.redo.grad_redo_enabled", "True",
      "--agent.redo.act_log_item", "log+erank+srank",
      "--agent.redo.grad_log_item", "log+erank+srank",
      "--logdir", str(logdir),
      "--seed", str(seed),
      "--egl_device", str(gpu),
  ]
  if kind == "crafter":
    return [
        str(PYTHON), "dreamerv3/main.py",
        "--configs", "crafter", "size1m",
        "--run.steps", "100000000",
        "--run.train_ratio", "0.5",
        "--run.task_interval", "100000000",
        "--run.log_every", "10000",
        "--run.report_every", "20000",
        "--run.save_every", "900",
        "--run.envs", "1",
        "--run.revive_strategy", "fixed",
        *common[2:],
    ]
  if kind == "dog":
    return [
        str(PYTHON), "dreamerv3/main.py",
        "--configs", "continual_dmc_priori", "size1m",
        "--task", "dog_stand|dog_walk|dog_trot",
        "--run.train_ratio", "1024",
        "--run.task_interval", "2000000",
        "--run.steps", "6000000",
        "--run.envs", "16",
        "--run.log_every", "10000",
        "--run.report_every", "20000",
        "--run.save_every", "1800",
        "--run.revive_strategy", "threshold",
        "--env.continual_dmc_priori.obs_dim", "223",
        "--env.continual_dmc_priori.task_action_space", "38",
        "--replay.chunksize", "4096",
        "--replay.cache_chunks", "4096",
        "--agent.imag_length", "15",
        *common[2:],
    ]
  raise ValueError(kind)


def specs():
  for kind in ("crafter", "dog"):
    project = {
        "crafter": "continual_dreamer_soft_reset_crafter_size1m",
        "dog": "continual_dreamer_soft_reset_dog_stand|dog_walk|dog_trot_size1m",
    }[kind]
    group = "l2_init_2e-5"
    for seed in SEEDS:
      gpu = GPU_ASSIGNMENTS[(kind, seed)]
      logdir = ROOT / "logdir" / project / group / f"seed_{seed}"
      yield {
          "kind": kind,
          "project": project,
          "seed": seed,
          "gpu": gpu,
          "group": group,
          "logdir": str(logdir),
          "pid": None,
          "status": "queued",
          "wandb_id": f"baseline-{kind}-l2-init-{seed}-{RUN_VERSION}",
      }


def main():
  if not PYTHON.exists():
    raise SystemExit(f"Missing dreamer env python: {PYTHON}")
  base_env = load_env()
  state = {
      "projects": [
          "continual_dreamer_soft_reset_crafter_size1m",
          "continual_dreamer_soft_reset_dog_stand|dog_walk|dog_trot_size1m",
      ],
      "run_version": RUN_VERSION,
      "started_at": now(),
      "jobs": [],
  }
  for job in specs():
    logdir = pathlib.Path(job["logdir"])
    logdir.mkdir(parents=True, exist_ok=True)
    env = base_env.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(job["gpu"]),
        "MUJOCO_EGL_DEVICE_ID": str(job["gpu"]),
        "WANDB_RUN_ID": job["wandb_id"],
        "WANDB_RESUME": "allow",
    })
    cmd = command(job["kind"], job["seed"], logdir, job["gpu"])
    logfile = logdir / "train.log"
    with logfile.open("ab", buffering=0) as stream:
      stream.write((f"\n[{now()}] Launch: {' '.join(shlex.quote(x) for x in cmd)}\n").encode())
      proc = subprocess.Popen(
          cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
          start_new_session=True)
    job["pid"] = proc.pid
    job["status"] = "running"
    job["command"] = cmd
    state["jobs"].append(job)
    print(f"[{now()}] launched {job['group']}/seed_{job['seed']} pid={proc.pid} gpu={job['gpu']}", flush=True)
  STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
  STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
  print(f"[{now()}] state={STATE_FILE}", flush=True)


if __name__ == "__main__":
  main()
