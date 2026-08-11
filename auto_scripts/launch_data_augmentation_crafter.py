#!/usr/bin/env python3
"""Launch and validate the requested Crafter data augmentation baselines."""

import glob
import json
import os
import pathlib
import shlex
import signal
import subprocess
import time
from datetime import datetime


ROOT = pathlib.Path(__file__).resolve().parents[1]
PYTHON = pathlib.Path("/home/jiale/.conda/envs/dreamer/bin/python")
ENV_FILE = ROOT / ".env.wandb.local"
PROJECT = "continual_dreamer_soft_reset_crafter_size1m"
RUN_VERSION = "da-crafter-trainratio512-v1"
STATE_FILE = ROOT / "logdir" / "scheduler" / f"data_augmentation_crafter_{RUN_VERSION}.json"
SEEDS = (1000, 2000, 3000)
MIN_VALID_LOGS = 10
GROUPS = {
    "data_augmentation_batch_align": "batch_align",
    "data_augmentation_batch_aug": "batch_aug",
}
REQUIRED_METRICS = (
    "loss/image",
    "loss/dyn",
    "train/data_augmentation/active",
    "train/data_augmentation/batch_align",
    "train/data_augmentation/batch_aug",
    "fps/policy",
    "fps/train",
)


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
  try:
    out = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
  except Exception as exc:
    print(f"[{now()}] GPU query failed, falling back to sequential GPUs: {exc}", flush=True)
    return list(range(8))
  gpus = []
  for line in out.splitlines():
    idx, used, total, util = [int(x.strip()) for x in line.split(",")]
    gpus.append((idx, used / max(total, 1), util))
  gpus.sort(key=lambda x: (x[1], x[2], x[0]))
  return [idx for idx, _, _ in gpus]


def command(mode, seed, logdir, gpu):
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
      "--run.reset_frequency", "0",
      "--run.reset_mechanism", "disabled",
      "--run.reset_target", "all",
      "--run.revive_epoch", "0",
      "--agent.wsc.mechanism", "disabled",
      "--agent.wsc.target", "all",
      "--agent.redo.redo_enabled", "True",
      "--agent.redo.grad_redo_enabled", "True",
      "--agent.redo.act_log_item", "log+erank+srank",
      "--agent.redo.grad_log_item", "log+erank+srank",
      "--agent.data_augmentation.mode", mode,
      "--agent.data_augmentation.pad", "4",
      "--logdir", str(logdir),
      "--seed", str(seed),
      "--egl_device", str(gpu),
  ]


def make_jobs():
  gpus = query_gpus()
  selected = gpus[:6]
  jobs = []
  i = 0
  for group, mode in GROUPS.items():
    for seed in SEEDS:
      gpu = selected[i % len(selected)]
      logdir = ROOT / "logdir" / PROJECT / group / f"seed_{seed}"
      jobs.append({
          "project": PROJECT,
          "group": group,
          "mode": mode,
          "seed": seed,
          "gpu": gpu,
          "logdir": str(logdir),
          "pid": None,
          "status": "queued",
          "wandb_id": f"baseline-crafter-da-{mode.replace('_', '-')}-{seed}-{RUN_VERSION}",
      })
      i += 1
  return jobs


def process_alive(pid):
  if not pid:
    return False
  try:
    os.kill(pid, 0)
    return True
  except ProcessLookupError:
    return False
  except PermissionError:
    return True


def valid_metrics_records(path, mode):
  metrics_path = pathlib.Path(path) / "metrics.jsonl"
  if not metrics_path.exists():
    return 0, None
  count = 0
  latest = None
  with metrics_path.open() as stream:
    for line in stream:
      line = line.strip()
      if not line:
        continue
      try:
        metrics = json.loads(line)
      except json.JSONDecodeError:
        continue
      missing = [key for key in REQUIRED_METRICS if key not in metrics]
      if missing:
        continue
      if float(metrics["train/data_augmentation/active"]) != 1.0:
        continue
      if mode == "batch_aug" and float(metrics["train/data_augmentation/batch_aug"]) != 1.0:
        continue
      if float(metrics["fps/policy"]) < 4.0:
        continue
      count += 1
      latest = metrics
  return count, latest


def wandb_has_output(path):
  pattern = str(pathlib.Path(path) / "wandb" / "wandb" / "run-*" / "run-*.wandb")
  return bool(glob.glob(pattern))


def valid_logging(job):
  count, metrics = valid_metrics_records(job["logdir"], job["mode"])
  if not metrics:
    return False, "no metrics yet"
  if count < MIN_VALID_LOGS:
    return False, (
        f"valid_logs={count}/{MIN_VALID_LOGS} "
        f"step={metrics.get('step')} fps={float(metrics['fps/policy']):.2f}")
  if not wandb_has_output(job["logdir"]):
    return False, "no wandb local output yet"
  return True, (
      f"valid_logs={count} step={metrics.get('step')} "
      f"fps={float(metrics['fps/policy']):.2f}")


def write_state(state):
  STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
  STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def launch(job, env):
  logdir = pathlib.Path(job["logdir"])
  if logdir.exists() and (logdir / "metrics.jsonl").exists():
    raise RuntimeError(f"Refusing to mix with existing metrics: {logdir}")
  logdir.mkdir(parents=True, exist_ok=True)
  job_env = env.copy()
  job_env.update({
      "CUDA_VISIBLE_DEVICES": str(job["gpu"]),
      "MUJOCO_EGL_DEVICE_ID": str(job["gpu"]),
      "WANDB_RUN_ID": job["wandb_id"],
      "WANDB_RESUME": "allow",
  })
  cmd = command(job["mode"], job["seed"], logdir, job["gpu"])
  logfile = logdir / "train.log"
  with logfile.open("ab", buffering=0) as stream:
    stream.write((f"\n[{now()}] Launch: {' '.join(shlex.quote(x) for x in cmd)}\n").encode())
    proc = subprocess.Popen(
        cmd, cwd=ROOT, env=job_env, stdout=stream, stderr=subprocess.STDOUT,
        start_new_session=True)
  job.update(pid=proc.pid, status="running", command=cmd, launched_at=now())
  print(f"[{now()}] launched {job['group']}/seed_{job['seed']} pid={proc.pid} gpu={job['gpu']}", flush=True)


def terminate_self_started(state):
  for job in state["jobs"]:
    if job.get("pid") and process_alive(job["pid"]):
      os.kill(job["pid"], signal.SIGTERM)


def main():
  if not PYTHON.exists():
    raise SystemExit(f"Missing dreamer env python: {PYTHON}")
  env = load_env()
  state = {
      "project": PROJECT,
      "run_version": RUN_VERSION,
      "started_at": now(),
      "state_file": str(STATE_FILE),
      "jobs": make_jobs(),
  }
  try:
    for job in state["jobs"]:
      launch(job, env)
      write_state(state)
  except Exception:
    terminate_self_started(state)
    raise

  print(f"[{now()}] state={STATE_FILE}", flush=True)
  pending = {i for i in range(len(state["jobs"]))}
  poll = 0
  while pending:
    poll += 1
    for idx in list(pending):
      job = state["jobs"][idx]
      alive = process_alive(job["pid"])
      valid, detail = valid_logging(job)
      job["last_check"] = now()
      job["last_detail"] = detail
      if valid and alive:
        job["status"] = "valid_logging"
        pending.remove(idx)
        print(f"[{now()}] valid {job['group']}/seed_{job['seed']} pid={job['pid']} gpu={job['gpu']} {detail}", flush=True)
      elif not alive:
        job["status"] = "exited_before_valid_logging"
        write_state(state)
        raise RuntimeError(
            f"{job['group']}/seed_{job['seed']} pid={job['pid']} exited before valid logging: {detail}")
      else:
        job["status"] = "waiting_for_logging"
    write_state(state)
    if pending:
      summary = ", ".join(
          f"{state['jobs'][i]['group']}/seed_{state['jobs'][i]['seed']}:"
          f"{state['jobs'][i].get('last_detail', 'pending')}" for i in sorted(pending))
      print(f"[{now()}] poll {poll}: waiting for {len(pending)} runs: {summary}", flush=True)
      time.sleep(60)
  print(f"[{now()}] all requested Crafter data augmentation runs have valid logging", flush=True)


if __name__ == "__main__":
  main()
