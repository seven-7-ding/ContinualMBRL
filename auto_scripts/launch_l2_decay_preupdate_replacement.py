#!/usr/bin/env python3
"""Launch replacement l2_decay_preupdate runs."""

import json
import os
import pathlib
import shlex
import subprocess
from datetime import datetime


ROOT = pathlib.Path(__file__).resolve().parents[1]
PYTHON = pathlib.Path("/home/jiale/.conda/envs/dreamer/bin/python")
ENV_FILE = ROOT / ".env.wandb.local"
PROJECT = "continual_dreamer_soft_reset_size1m"
GROUP = "l2_decay_preupdate_2e-5"
RUN_VERSION = "groupfix-v1"
STATE_FILE = ROOT / "logdir" / "scheduler" / f"l2_decay_preupdate_{RUN_VERSION}.json"
TASK = "walker_run|hopper_hop|fish_swim"
SEEDS = (1000, 2000, 3000)
GPU_ASSIGNMENTS = {
    1000: 5,
    2000: 4,
    3000: 0,
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


def command(seed, logdir, gpu):
  return [
      str(PYTHON), "dreamerv3/main.py",
      "--configs", "continual_dmc_priori", "size1m",
      "--task", TASK,
      "--run.train_ratio", "1024",
      "--run.task_interval", "1000000",
      "--run.task_repeat", "5",
      "--run.steps", "15000000",
      "--run.reset_frequency", "0",
      "--run.reset_mechanism", "l2_decay_preupdate",
      "--run.reset_target", "all",
      "--run.revive_epoch", "0",
      "--run.revive_strategy", "threshold",
      "--env.continual_dmc_priori.obs_dim", "24",
      "--env.continual_dmc_priori.task_action_space", "6",
      "--replay.chunksize", "4096",
      "--replay.cache_chunks", "4096",
      "--agent.imag_length", "15",
      "--agent.wsc.mechanism", "l2_decay_preupdate",
      "--agent.wsc.target", "all",
      "--agent.wsc.weight_decay", "2e-5",
      "--agent.wsc.l2_init_weight", "2e-5",
      "--agent.wsc.cbp_eta", "0.99",
      "--agent.wsc.cbp_maturity", "5000",
      "--agent.wsc.cbp_replacement_rate", "1e-4",
      "--agent.redo.redo_enabled", "True",
      "--agent.redo.grad_redo_enabled", "True",
      "--agent.redo.act_log_item", "log+erank+srank",
      "--agent.redo.grad_log_item", "log+erank+srank",
      "--run.save_every", "1800",
      "--logdir", str(logdir),
      "--seed", str(seed),
      "--egl_device", str(gpu),
  ]


def main():
  if not PYTHON.exists():
    raise SystemExit(f"Missing dreamer env python: {PYTHON}")
  base_env = load_env()
  state = {
      "project": PROJECT,
      "group": GROUP,
      "run_version": RUN_VERSION,
      "started_at": now(),
      "jobs": [],
  }
  for seed in SEEDS:
    gpu = GPU_ASSIGNMENTS[seed]
    logdir = ROOT / "logdir" / PROJECT / GROUP / f"seed_{seed}"
    logdir.mkdir(parents=True, exist_ok=True)
    env = base_env.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "MUJOCO_EGL_DEVICE_ID": str(gpu),
        "WANDB_RUN_ID": f"baseline-l2-decay-preupdate-{seed}-{RUN_VERSION}",
        "WANDB_RESUME": "allow",
    })
    cmd = command(seed, logdir, gpu)
    logfile = logdir / "train.log"
    with logfile.open("ab", buffering=0) as stream:
      stream.write((f"\n[{now()}] Launch: {' '.join(shlex.quote(x) for x in cmd)}\n").encode())
      proc = subprocess.Popen(
          cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
          start_new_session=True)
    job = {
        "seed": seed,
        "gpu": gpu,
        "pid": proc.pid,
        "logdir": str(logdir),
        "wandb_id": env["WANDB_RUN_ID"],
        "command": cmd,
        "status": "running",
    }
    state["jobs"].append(job)
    print(f"[{now()}] launched {GROUP}/seed_{seed} pid={proc.pid} gpu={gpu}", flush=True)
  STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
  STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
  print(f"[{now()}] state={STATE_FILE}", flush=True)


if __name__ == "__main__":
  main()
