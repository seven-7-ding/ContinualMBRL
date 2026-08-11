#!/usr/bin/env python3
"""Launch and validate the requested continual DMC vision DA run."""

import glob
import json
import os
import pathlib
import shlex
import subprocess
import time
from datetime import datetime

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
PYTHON = pathlib.Path("/home/jiale/.conda/envs/dreamer/bin/python")
ENV_FILE = ROOT / ".env.wandb.local"
PROJECT = "continual_dreamer_soft_reset_walker_run|hopper_hop|fish_swim_vision_size12m_1m_x5"
GROUP = "data_augmentation_batch_align"
RUN_VERSION = "dmc-vision-size12m-da-align-1m-x5-v1"
STATE_FILE = ROOT / "logdir" / "scheduler" / f"{PROJECT}_{RUN_VERSION}.json"
SEEDS = (1000, 2000, 3000)
MIN_VALID_LOGS = 1
POLL_SECONDS = 60
MAX_GPU_MEMORY_MB = 9000


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


def process_alive(pid):
  if not pid:
    return False
  try:
    stat = subprocess.check_output(
        ["ps", "-p", str(pid), "-o", "stat="], text=True, timeout=5).strip()
    if stat.startswith("Z"):
      return False
  except subprocess.CalledProcessError:
    return False
  except Exception:
    pass
  try:
    os.kill(pid, 0)
    return True
  except ProcessLookupError:
    return False
  except PermissionError:
    return True


def query_gpus():
  try:
    out = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=15)
  except Exception as exc:
    print(f"[{now()}] GPU query failed, using GPU 0: {exc}", flush=True)
    return [0]
  gpus = []
  for line in out.splitlines():
    idx, used, total, util = [int(x.strip()) for x in line.split(",")]
    gpus.append((idx, used / max(total, 1), util))
  gpus.sort(key=lambda item: (item[1], item[2], item[0]))
  return [idx for idx, _, _ in gpus]


def command(seed, logdir, gpu):
  return [
      str(PYTHON), "-u", "dreamerv3/main.py",
      "--configs", "dmc_vision", "continual_dmc_vision", "size12m",
      "--task", "walker_run|hopper_hop|fish_swim",
      "--run.steps", "15000000",
      "--run.task_interval", "1000000",
      "--run.task_repeat", "5",
      "--run.train_ratio", "256",
      "--run.log_every", "10000",
      "--run.report_every", "20000",
      "--run.save_every", "900",
      "--run.reset_frequency", "0",
      "--run.reset_mechanism", "disabled",
      "--run.reset_target", "all",
      "--run.revive_epoch", "0",
      "--run.revive_strategy", "fixed",
      "--env.continual_dmc.task_action_space", "6",
      "--agent.wsc.mechanism", "disabled",
      "--agent.wsc.target", "all",
      "--agent.redo.redo_enabled", "True",
      "--agent.redo.grad_redo_enabled", "True",
      "--agent.redo.act_log_item", "log+erank+srank",
      "--agent.redo.grad_log_item", "log+erank+srank",
      "--agent.data_augmentation.mode", "batch_align",
      "--agent.data_augmentation.pad", "4",
      "--jax.prealloc", "False",
      "--jax.memory_fraction", "0.25",
      "--logdir", str(logdir),
      "--seed", str(seed),
      "--egl_device", str(gpu),
  ]


def make_jobs():
  gpus = query_gpus()
  jobs = []
  for index, seed in enumerate(SEEDS):
    gpu = gpus[index % len(gpus)]
    logdir = ROOT / "logdir" / PROJECT / GROUP / f"seed_{seed}"
    jobs.append({
        "project": PROJECT,
        "group": GROUP,
        "seed": seed,
        "gpu": gpu,
        "logdir": str(logdir),
        "pid": None,
        "status": "queued",
        "wandb_id": f"baseline-dmc-vision-size12m-da-align-{seed}-{RUN_VERSION}",
    })
  return jobs


def add_missing_jobs(state):
  existing = {int(job["seed"]): job for job in state.get("jobs", [])}
  used_gpus = {int(job["gpu"]) for job in existing.values() if job.get("gpu") is not None}
  available = [gpu for gpu in query_gpus() if gpu not in used_gpus] or query_gpus()
  index = 0
  for seed in SEEDS:
    if seed in existing:
      continue
    gpu = available[index % len(available)]
    index += 1
    logdir = ROOT / "logdir" / PROJECT / GROUP / f"seed_{seed}"
    state.setdefault("jobs", []).append({
        "project": PROJECT,
        "group": GROUP,
        "seed": seed,
        "gpu": gpu,
        "logdir": str(logdir),
        "pid": None,
        "status": "queued",
        "wandb_id": f"baseline-dmc-vision-size12m-da-align-{seed}-{RUN_VERSION}",
    })
  state["jobs"].sort(key=lambda job: int(job["seed"]))


def iter_metrics(logdir):
  path = pathlib.Path(logdir) / "metrics.jsonl"
  if not path.exists():
    return
  with path.open() as stream:
    for line in stream:
      line = line.strip()
      if not line:
        continue
      try:
        yield json.loads(line)
      except json.JSONDecodeError:
        continue


def config_ok(logdir):
  path = pathlib.Path(logdir) / "config.yaml"
  if not path.exists():
    return False, "missing config.yaml"
  try:
    config = yaml.safe_load(path.read_text())
  except Exception as exc:
    return False, f"config read failed: {exc}"
  checks = {
      "task": "walker_run|hopper_hop|fish_swim",
      "run.steps": 15000000.0,
      "run.task_interval": 1000000.0,
      "run.task_repeat": 5,
      "run.train_ratio": 256.0,
      "agent.data_augmentation.mode": "batch_align",
      "jax.memory_fraction": 0.25,
      "jax.prealloc": False,
  }
  values = {
      "task": config.get("task"),
      "run.steps": config.get("run", {}).get("steps"),
      "run.task_interval": config.get("run", {}).get("task_interval"),
      "run.task_repeat": config.get("run", {}).get("task_repeat"),
      "run.train_ratio": config.get("run", {}).get("train_ratio"),
      "agent.data_augmentation.mode": (
          config.get("agent", {}).get("data_augmentation", {}).get("mode")),
      "jax.memory_fraction": config.get("jax", {}).get("memory_fraction"),
      "jax.prealloc": config.get("jax", {}).get("prealloc"),
  }
  for key, expected in checks.items():
    if values.get(key) != expected:
      return False, f"{key}={values.get(key)!r}, expected {expected!r}"
  action_space = config.get("env", {}).get("continual_dmc", {}).get("task_action_space")
  if action_space not in ([6], 6):
    return False, f"task_action_space={action_space!r}, expected 6"
  if config.get("env", {}).get("continual_dmc", {}).get("proprio") is not False:
    return False, "continual_dmc proprio is not False"
  return True, "config ok"


def gpu_memory_mb(pid):
  if not pid:
    return 0
  try:
    out = subprocess.check_output([
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=15)
  except Exception:
    return 0
  total = 0
  for line in out.splitlines():
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 2:
      continue
    try:
      app_pid, used = int(parts[0]), int(parts[1])
    except ValueError:
      continue
    if app_pid == int(pid):
      total += used
  return total


def stop_process_group(pid):
  if not pid:
    return
  try:
    os.killpg(int(pid), 15)
  except ProcessLookupError:
    return
  except Exception:
    try:
      os.kill(int(pid), 15)
    except ProcessLookupError:
      return


def valid_metric(metrics):
  required = (
      "loss/image",
      "loss/dyn",
      "fps/policy",
      "fps/train",
      "train/data_augmentation/active",
      "train/data_augmentation/batch_align",
      "train/data_augmentation/batch_aug",
  )
  if any(key not in metrics for key in required):
    return False
  if float(metrics["fps/policy"]) < 4.0:
    return False
  if float(metrics["train/data_augmentation/active"]) != 1.0:
    return False
  if float(metrics["train/data_augmentation/batch_align"]) != 1.0:
    return False
  prefixes = (
      "act_redo/Linear_WB_FNorm/",
      "act_redo/Zombie_Percentage/",
      "act_redo/Saturation_Percentage/",
      "act_redo/Variation_Rank_0.9/",
      "act_redo/Variation_Rank_0.95/",
      "act_redo/Variation_Rank_0.99/",
      "grad_redo/GradDormant_0.1/",
      "data_diversity/real_data/",
      "data_diversity/imag_data/",
  )
  return all(any(key.startswith(prefix) for key in metrics) for prefix in prefixes)


def wandb_has_output(logdir):
  pattern = str(pathlib.Path(logdir) / "wandb" / "wandb" / "run-*" / "run-*.wandb")
  return bool(glob.glob(pattern))


def valid_logging(job):
  ok, detail = config_ok(job["logdir"])
  if not ok:
    return False, detail
  count = 0
  latest = None
  for metrics in iter_metrics(job["logdir"]):
    if valid_metric(metrics):
      count += 1
      latest = metrics
  job["valid_logs"] = count
  if latest is None:
    return False, "no full diagnostic metrics yet"
  if count < MIN_VALID_LOGS:
    return False, f"valid_logs={count}/{MIN_VALID_LOGS}"
  if not wandb_has_output(job["logdir"]):
    return False, "no wandb local output yet"
  return True, (
      f"valid_logs={count} step={latest.get('step')} "
      f"fps={float(latest['fps/policy']):.2f}")


def write_state(state):
  STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
  tmp = STATE_FILE.with_suffix(".tmp")
  tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
  tmp.replace(STATE_FILE)


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
  cmd = command(job["seed"], logdir, job["gpu"])
  logfile = logdir / "train.log"
  with logfile.open("ab", buffering=0) as stream:
    stream.write((f"\n[{now()}] Launch: {' '.join(shlex.quote(x) for x in cmd)}\n").encode())
    proc = subprocess.Popen(
        cmd, cwd=ROOT, env=job_env, stdout=stream, stderr=subprocess.STDOUT,
        start_new_session=True)
  job.update(pid=proc.pid, status="running", command=cmd, launched_at=now())
  print(f"[{now()}] launched {GROUP}/seed_{job['seed']} pid={proc.pid} gpu={job['gpu']}", flush=True)


def main():
  if not PYTHON.exists():
    raise SystemExit(f"Missing dreamer env python: {PYTHON}")
  if STATE_FILE.exists():
    state = json.loads(STATE_FILE.read_text())
    add_missing_jobs(state)
    print(f"[{now()}] resumed launcher state from {STATE_FILE}", flush=True)
  else:
    state = {
        "project": PROJECT,
        "group": GROUP,
        "run_version": RUN_VERSION,
        "started_at": now(),
        "jobs": make_jobs(),
    }
  env = load_env()
  for job in state["jobs"]:
    if job.get("status") == "stopped_by_user_high_cuda_memory":
      print(
          f"[{now()}] skip stopped {GROUP}/seed_{job['seed']} "
          f"pid={job.get('pid')} gpu={job['gpu']}", flush=True)
      continue
    if process_alive(job.get("pid")):
      print(
          f"[{now()}] keep existing {GROUP}/seed_{job['seed']} "
          f"pid={job['pid']} gpu={job['gpu']}", flush=True)
      continue
    if pathlib.Path(job["logdir"]).exists() and (pathlib.Path(job["logdir"]) / "metrics.jsonl").exists():
      print(
          f"[{now()}] keep existing logdir for {GROUP}/seed_{job['seed']} "
          f"without relaunch", flush=True)
      continue
    launch(job, env)
    write_state(state)
  pending = {i for i in range(len(state["jobs"]))}
  poll = 0
  while pending:
    poll += 1
    for idx in list(pending):
      job = state["jobs"][idx]
      alive = process_alive(job["pid"])
      used_mb = gpu_memory_mb(job.get("pid"))
      job["gpu_memory_mb"] = used_mb
      if alive and used_mb > MAX_GPU_MEMORY_MB:
        job["status"] = "stopped_high_cuda_memory"
        job["last_detail"] = f"gpu_memory_mb={used_mb} exceeds {MAX_GPU_MEMORY_MB}"
        stop_process_group(job["pid"])
        write_state(state)
        raise RuntimeError(
            f"{GROUP}/seed_{job['seed']} pid={job['pid']} exceeded "
            f"GPU memory guard: {used_mb} MB > {MAX_GPU_MEMORY_MB} MB")
      valid, detail = valid_logging(job)
      job["last_check"] = now()
      job["last_detail"] = detail
      if valid and alive:
        job["status"] = "valid_logging"
        pending.remove(idx)
        print(f"[{now()}] valid {GROUP}/seed_{job['seed']} pid={job['pid']} gpu={job['gpu']} {detail}", flush=True)
      elif not alive:
        job["status"] = "exited_before_valid_logging"
        write_state(state)
        raise RuntimeError(
            f"{GROUP}/seed_{job['seed']} pid={job['pid']} exited before valid logging: {detail}")
      else:
        job["status"] = "waiting_for_valid_logging"
    write_state(state)
    if pending:
      summary = ", ".join(
          f"seed_{state['jobs'][i]['seed']}:{state['jobs'][i].get('last_detail', 'pending')}"
          for i in sorted(pending))
      print(f"[{now()}] poll {poll}: waiting for {len(pending)} runs: {summary}", flush=True)
      time.sleep(POLL_SECONDS)
  print(f"[{now()}] all DMC vision DA runs have valid diagnostic logging", flush=True)


if __name__ == "__main__":
  main()
