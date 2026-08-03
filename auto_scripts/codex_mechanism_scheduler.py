#!/usr/bin/env python3
"""Launch and monitor the 9 requested mechanism experiments.

This scheduler only manages PIDs it starts itself. It never kills or modifies
unrelated processes.
"""

import json
import os
import pathlib
import shlex
import subprocess
import time
from datetime import datetime


ROOT = pathlib.Path(__file__).resolve().parents[1]
PYTHON = pathlib.Path("/home/jiale/.conda/envs/dreamer/bin/python")
ENV_FILE = ROOT / ".env.wandb.local"
RUN_VERSION = "metrics-v2"
STATE_FILE = ROOT / "logdir" / "scheduler" / f"mechanism_scheduler_state_{RUN_VERSION}.json"
PROJECT = "continual_dreamer_soft_reset_size1m"
TASK = "walker_run|hopper_hop|fish_swim"
SEEDS = (1000, 2000, 3000)
MECHANISMS = (
    ("l2_decay", "l2_decay_2e-5"),
    ("l2_init", "l2_init_2e-5"),
    ("continual_backprop", "continual_backprop"),
)
TOTAL_STEPS = 15_000_000
TASK_INTERVAL = 1_000_000
POLL_SECONDS = int(os.environ.get("CODEX_MECH_POLL_SECONDS", "120"))
MAX_RESTARTS = int(os.environ.get("CODEX_MECH_MAX_RESTARTS", "100"))


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


def query_gpus():
  cmd = [
      "nvidia-smi",
      "--query-gpu=index,memory.used,memory.total,utilization.gpu",
      "--format=csv,noheader,nounits",
  ]
  try:
    out = subprocess.check_output(cmd, text=True, timeout=15)
  except Exception:
    return [(i, 0, 0, 0) for i in range(8)]
  rows = []
  for line in out.splitlines():
    parts = [x.strip() for x in line.split(",")]
    if len(parts) != 4:
      continue
    rows.append(tuple(int(x) for x in parts))
  return rows


def gpu_order():
  rows = query_gpus()
  rows.sort(key=lambda x: (x[1] / max(x[2], 1), x[3], x[0]))
  order = [row[0] for row in rows if 0 <= row[0] <= 7]
  return order or list(range(8))


def job_specs():
  order = gpu_order()
  specs = []
  index = 0
  for mechanism, group in MECHANISMS:
    for seed in SEEDS:
      gpu = order[index % len(order)]
      logdir = ROOT / "logdir" / PROJECT / group / f"seed_{seed}"
      specs.append({
          "id": f"{group}/seed_{seed}",
          "mechanism": mechanism,
          "group": group,
          "seed": seed,
          "gpu": gpu,
          "logdir": str(logdir),
          "wandb_id": f"baseline-{group.replace('_', '-')}-{seed}-{RUN_VERSION}",
          "restarts": 0,
          "pid": None,
          "status": "queued",
      })
      index += 1
  return specs


def load_state():
  if STATE_FILE.exists():
    data = json.loads(STATE_FILE.read_text())
    if data.get("project") == PROJECT:
      return data
  return {"project": PROJECT, "jobs": job_specs(), "events": []}


def save_state(state):
  STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
  tmp = STATE_FILE.with_suffix(".tmp")
  tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
  tmp.replace(STATE_FILE)


def append_event(state, message):
  state.setdefault("events", []).append({"time": now(), "message": message})
  state["events"] = state["events"][-50:]
  save_state(state)
  print(f"[{now()}] {message}", flush=True)


def is_done(job):
  metrics = pathlib.Path(job["logdir"]) / "metrics.jsonl"
  if not metrics.exists():
    return False
  try:
    with metrics.open() as f:
      last = None
      for line in f:
        if line.strip():
          last = json.loads(line)
    return bool(last and int(last.get("step", 0)) >= TOTAL_STEPS)
  except Exception:
    return False


def launch(job, base_env):
  pathlib.Path(job["logdir"]).mkdir(parents=True, exist_ok=True)
  env = base_env.copy()
  env.update({
      "CUDA_VISIBLE_DEVICES": str(job["gpu"]),
      "MUJOCO_EGL_DEVICE_ID": str(job["gpu"]),
      "WANDB_RUN_ID": job["wandb_id"],
      "WANDB_RESUME": "allow",
  })
  cmd = [
      str(PYTHON), "dreamerv3/main.py",
      "--configs", "continual_dmc_priori", "size1m",
      "--task", TASK,
      "--run.train_ratio", "1024",
      "--run.task_interval", str(TASK_INTERVAL),
      "--run.task_repeat", "5",
      "--run.steps", str(TOTAL_STEPS),
      "--run.reset_frequency", "0",
      "--run.reset_mechanism", job["mechanism"],
      "--run.reset_target", "all",
      "--run.revive_epoch", "0",
      "--run.revive_strategy", "threshold",
      "--env.continual_dmc_priori.obs_dim", "24",
      "--env.continual_dmc_priori.task_action_space", "6",
      "--replay.chunksize", "4096",
      "--replay.cache_chunks", "4096",
      "--agent.imag_length", "15",
      "--agent.wsc.mechanism", job["mechanism"],
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
      "--logdir", job["logdir"],
      "--seed", str(job["seed"]),
      "--egl_device", str(job["gpu"]),
  ]
  logfile = pathlib.Path(job["logdir"]) / "train.log"
  fh = logfile.open("ab", buffering=0)
  fh.write((f"\n[{now()}] Launch: {' '.join(shlex.quote(x) for x in cmd)}\n").encode())
  proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
  job["pid"] = proc.pid
  job["status"] = "running"
  job["last_launch"] = now()
  return proc


def main():
  if not PYTHON.exists():
    raise SystemExit(f"Missing dreamer env python: {PYTHON}")
  state = load_state()
  procs = {}
  base_env = load_env()
  append_event(state, "scheduler_start jobs=9 manages_only_own_pids=true")
  while True:
    all_done = True
    for job in state["jobs"]:
      if is_done(job):
        if job.get("status") != "done":
          job["status"] = "done"
          append_event(state, f"done {job['id']}")
        continue
      all_done = False
      proc = procs.get(job["id"])
      if proc is not None and proc.poll() is None:
        continue
      if proc is not None:
        code = proc.poll()
        job["restarts"] += 1
        job["status"] = "failed"
        append_event(state, f"exit {job['id']} code={code} restarts={job['restarts']}")
      if job["restarts"] > MAX_RESTARTS:
        job["status"] = "blocked"
        append_event(state, f"blocked {job['id']} max_restarts={MAX_RESTARTS}")
        continue
      proc = launch(job, base_env)
      procs[job["id"]] = proc
      append_event(state, f"launched {job['id']} pid={proc.pid} gpu={job['gpu']}")
    save_state(state)
    if all_done:
      append_event(state, "all_done")
      return
    live = sum(1 for proc in procs.values() if proc.poll() is None)
    print(f"[{now()}] poll live={live} done={sum(j['status'] == 'done' for j in state['jobs'])}/9", flush=True)
    time.sleep(POLL_SECONDS)


if __name__ == "__main__":
  main()
