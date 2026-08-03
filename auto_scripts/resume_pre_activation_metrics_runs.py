#!/usr/bin/env python3
"""Resume checkpointed Dreamer runs after adding preactivation metrics."""

import json
import os
import pathlib
import shlex
import subprocess
from datetime import datetime


ROOT = pathlib.Path(__file__).resolve().parents[1]
PYTHON = pathlib.Path("/home/jiale/.conda/envs/dreamer/bin/python")
ENV_FILE = ROOT / ".env.wandb.local"
STATE_FILE = ROOT / "logdir" / "scheduler" / "pre_activation_metrics_resume_20260803.json"


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


def mechanism_command(job):
  mechanism = job["mechanism"]
  return [
      str(PYTHON), "dreamerv3/main.py",
      "--configs", "continual_dmc_priori", "size1m",
      "--task", "walker_run|hopper_hop|fish_swim",
      "--run.train_ratio", "1024",
      "--run.task_interval", "1000000",
      "--run.task_repeat", "5",
      "--run.steps", "15000000",
      "--run.reset_frequency", "0",
      "--run.reset_mechanism", mechanism,
      "--run.reset_target", "all",
      "--run.revive_epoch", "0",
      "--run.revive_strategy", "threshold",
      "--env.continual_dmc_priori.obs_dim", "24",
      "--env.continual_dmc_priori.task_action_space", "6",
      "--replay.chunksize", "4096",
      "--replay.cache_chunks", "4096",
      "--agent.imag_length", "15",
      "--agent.wsc.mechanism", mechanism,
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


def load_jobs():
  jobs = []

  # Original mechanism runs: resume l2_init and continual_backprop only. The
  # old l2_decay group was intentionally superseded by l2_decay_preupdate.
  state = ROOT / "logdir" / "scheduler" / "mechanism_scheduler_state_metrics-v2.json"
  data = json.loads(state.read_text())
  for job in data["jobs"]:
    if job.get("mechanism") not in ("l2_init", "continual_backprop"):
      continue
    job = dict(job)
    job["source_state"] = str(state)
    job["command"] = mechanism_command(job)
    jobs.append(job)

  # Correct crafter/dog l2-init runs and l2_decay_preupdate runs already store
  # the exact launch command in their state files.
  for name in (
      "crafter_dog_l2_init_correct-project-v1.json",
      "l2_decay_preupdate_groupfix-v1.json"):
    state = ROOT / "logdir" / "scheduler" / name
    data = json.loads(state.read_text())
    for job in data["jobs"]:
      job = dict(job)
      job["source_state"] = str(state)
      if "command" not in job:
        raise KeyError(f"Missing command in {state}: {job}")
      if job.get("kind") == "crafter":
        job["command"] = _replace_flag(job["command"], "--run.train_ratio", "0.5")
      jobs.append(job)
  return jobs


def _replace_flag(command, flag, value):
  command = list(command)
  try:
    index = command.index(flag)
  except ValueError:
    return command
  if index + 1 >= len(command):
    raise ValueError(f"Flag {flag} has no value in command: {command}")
  command[index + 1] = value
  return command


def main():
  if not PYTHON.exists():
    raise SystemExit(f"Missing dreamer env python: {PYTHON}")
  env0 = load_env()
  launched = []
  for job in load_jobs():
    logdir = pathlib.Path(job["logdir"])
    latest = logdir / "ckpt" / "latest"
    if not latest.exists():
      print(f"[{now()}] skip missing checkpoint: {logdir}", flush=True)
      continue
    old_pid = job.get("pid")
    if old_pid and pathlib.Path(f"/proc/{old_pid}").exists():
      print(f"[{now()}] skip alive old pid={old_pid}: {logdir}", flush=True)
      continue
    env = env0.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(job["gpu"]),
        "MUJOCO_EGL_DEVICE_ID": str(job["gpu"]),
        "WANDB_RUN_ID": job["wandb_id"],
        "WANDB_RESUME": "allow",
    })
    cmd = job["command"]
    logfile = logdir / "train.log"
    with logfile.open("ab", buffering=0) as stream:
      stream.write((f"\n[{now()}] Resume after preactivation metrics: "
                    f"{' '.join(shlex.quote(x) for x in cmd)}\n").encode())
      proc = subprocess.Popen(
          cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
          start_new_session=True)
    entry = {
        "group": job.get("group") or job.get("id", "").split("/")[0],
        "kind": job.get("kind"),
        "seed": job.get("seed"),
        "gpu": job.get("gpu"),
        "pid": proc.pid,
        "old_pid": old_pid,
        "logdir": str(logdir),
        "wandb_id": job["wandb_id"],
        "source_state": job["source_state"],
        "command": cmd,
        "started_at": now(),
    }
    launched.append(entry)
    print(
        f"[{now()}] resumed {entry['group']}/seed_{entry['seed']} "
        f"pid={proc.pid} gpu={entry['gpu']}",
        flush=True)
  STATE_FILE.write_text(json.dumps({
      "created_at": now(),
      "jobs": launched,
      "reason": "resume_after_preactivation_metric_update",
  }, indent=2, sort_keys=True) + "\n")
  print(f"[{now()}] state={STATE_FILE}", flush=True)


if __name__ == "__main__":
  main()
