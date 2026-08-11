#!/usr/bin/env python3
"""Launch corrected Crafter l2-init runs and monitor all corrected Crafter jobs."""

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
PROJECT = "continual_dreamer_soft_reset_crafter_size1m"
DA_STATE = ROOT / "logdir" / "scheduler" / "data_augmentation_crafter_da-crafter-trainratio512-v1.json"
RUN_VERSION = "crafter-trainratio512-correction-v1"
STATE_FILE = ROOT / "logdir" / "scheduler" / f"{RUN_VERSION}.json"
SEEDS = (1000, 2000, 3000)
L2_WEIGHTS = ("2e-4", "2e-6")
MIN_VALID_LOGS = 10
POLL_SECONDS = 60


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
    print(f"[{now()}] GPU query failed, using sequential fallback: {exc}", flush=True)
    return list(range(8))
  gpus = []
  for line in out.splitlines():
    idx, used, total, util = [int(x.strip()) for x in line.split(",")]
    gpus.append((idx, used / max(total, 1), util))
  gpus.sort(key=lambda x: (x[1], x[2], x[0]))
  return [idx for idx, _, _ in gpus]


def wandb_weight_label(weight):
  return weight.replace("-", "m").replace("+", "p").replace(".", "p")


def l2_command(weight, seed, logdir, gpu):
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
      "--run.reset_mechanism", "l2_init",
      "--run.reset_target", "all",
      "--run.revive_epoch", "0",
      "--agent.wsc.mechanism", "l2_init",
      "--agent.wsc.target", "all",
      "--agent.wsc.weight_decay", weight,
      "--agent.wsc.l2_init_weight", weight,
      "--agent.redo.redo_enabled", "True",
      "--agent.redo.grad_redo_enabled", "True",
      "--agent.redo.act_log_item", "log+erank+srank",
      "--agent.redo.grad_log_item", "log+erank+srank",
      "--logdir", str(logdir),
      "--seed", str(seed),
      "--egl_device", str(gpu),
  ]


def load_da_jobs():
  if not DA_STATE.exists():
    raise SystemExit(f"Missing DA state file: {DA_STATE}")
  state = json.loads(DA_STATE.read_text())
  jobs = []
  for job in state["jobs"]:
    jobs.append({
        "id": f"{job['group']}/seed_{job['seed']}",
        "kind": "data_augmentation",
        "mode": job["mode"],
        "group": job["group"],
        "seed": job["seed"],
        "gpu": job["gpu"],
        "pid": job["pid"],
        "logdir": job["logdir"],
        "status": "running_existing",
        "valid_logs": 0,
    })
  return jobs


def build_l2_jobs():
  order = query_gpus()
  jobs = []
  index = 0
  for weight in L2_WEIGHTS:
    group = f"l2_init_{weight}"
    for seed in SEEDS:
      gpu = order[index % len(order)]
      logdir = ROOT / "logdir" / PROJECT / group / f"seed_{seed}"
      jobs.append({
          "id": f"{group}/seed_{seed}",
          "kind": "l2_init",
          "weight": weight,
          "group": group,
          "seed": seed,
          "gpu": gpu,
          "pid": None,
          "logdir": str(logdir),
          "wandb_id": (
              f"baseline-crafter-l2-init-{wandb_weight_label(weight)}-"
              f"{seed}-{RUN_VERSION}"),
          "status": "queued",
          "valid_logs": 0,
      })
      index += 1
  return jobs


def launch_l2(job, env):
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
  cmd = l2_command(job["weight"], job["seed"], logdir, job["gpu"])
  logfile = logdir / "train.log"
  with logfile.open("ab", buffering=0) as stream:
    stream.write((f"\n[{now()}] Launch: {' '.join(shlex.quote(x) for x in cmd)}\n").encode())
    proc = subprocess.Popen(
        cmd, cwd=ROOT, env=job_env, stdout=stream, stderr=subprocess.STDOUT,
        start_new_session=True)
  job.update(pid=proc.pid, command=cmd, launched_at=now(), status="running")
  print(f"[{now()}] launched {job['id']} pid={proc.pid} gpu={job['gpu']}", flush=True)


def wandb_has_output(logdir):
  pattern = str(pathlib.Path(logdir) / "wandb" / "wandb" / "run-*" / "run-*.wandb")
  return bool(glob.glob(pattern))


def config_train_ratio(logdir):
  path = pathlib.Path(logdir) / "config.yaml"
  if not path.exists():
    return None
  try:
    return yaml.safe_load(path.read_text()).get("run", {}).get("train_ratio")
  except Exception:
    return None


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


def valid_metric(job, metrics):
  common = ("loss/image", "loss/dyn", "fps/policy", "fps/train")
  if any(key not in metrics for key in common):
    return False
  if float(metrics["fps/policy"]) < 4.0:
    return False
  required_prefixes = (
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
  for prefix in required_prefixes:
    if not any(key.startswith(prefix) for key in metrics):
      return False
  if job["kind"] == "data_augmentation":
    required = (
        "train/data_augmentation/active",
        "train/data_augmentation/batch_align",
        "train/data_augmentation/batch_aug",
    )
    if any(key not in metrics for key in required):
      return False
    if float(metrics["train/data_augmentation/active"]) != 1.0:
      return False
    if job["mode"] == "batch_aug" and float(metrics["train/data_augmentation/batch_aug"]) != 1.0:
      return False
  return True


def initial_state():
  if STATE_FILE.exists():
    state = json.loads(STATE_FILE.read_text())
    return state, False
  jobs = load_da_jobs() + build_l2_jobs()
  state = {
      "project": PROJECT,
      "run_version": RUN_VERSION,
      "started_at": now(),
      "state_file": str(STATE_FILE),
      "jobs": jobs,
  }
  return state, True


def valid_logging(job):
  ratio = config_train_ratio(job["logdir"])
  if ratio != 512.0:
    return False, f"bad train_ratio={ratio}"
  count = 0
  latest = None
  for metrics in iter_metrics(job["logdir"]):
    if valid_metric(job, metrics):
      count += 1
      latest = metrics
  job["valid_logs"] = count
  if latest is None:
    return False, "no valid metrics yet"
  if count < MIN_VALID_LOGS:
    return False, (
        f"valid_logs={count}/{MIN_VALID_LOGS} "
        f"step={latest.get('step')} fps={float(latest['fps/policy']):.2f}")
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


def main():
  if not PYTHON.exists():
    raise SystemExit(f"Missing dreamer env python: {PYTHON}")
  env = load_env()
  state, should_launch = initial_state()
  jobs = state["jobs"]
  if should_launch:
    for job in jobs:
      if job["kind"] != "l2_init":
        continue
      launch_l2(job, env)
      write_state(state)
  else:
    print(f"[{now()}] resumed monitor from {STATE_FILE}", flush=True)
  pending = {i for i in range(len(jobs))}
  poll = 0
  while pending:
    poll += 1
    for idx in list(pending):
      job = jobs[idx]
      alive = process_alive(job["pid"])
      valid, detail = valid_logging(job)
      job["last_check"] = now()
      job["last_detail"] = detail
      if valid and alive:
        job["status"] = "valid_10_logs"
        pending.remove(idx)
        print(f"[{now()}] valid {job['id']} pid={job['pid']} gpu={job['gpu']} {detail}", flush=True)
      elif not alive:
        job["status"] = "exited_before_10_logs"
        write_state(state)
        raise RuntimeError(f"{job['id']} pid={job['pid']} exited before 10 valid logs: {detail}")
      else:
        job["status"] = "waiting_for_10_logs"
    write_state(state)
    if pending:
      summary = ", ".join(
          f"{jobs[i]['id']}:{jobs[i].get('last_detail', 'pending')}"
          for i in sorted(pending))
      print(f"[{now()}] poll {poll}: waiting for {len(pending)} runs: {summary}", flush=True)
      time.sleep(POLL_SECONDS)
  print(f"[{now()}] all corrected Crafter runs reached at least {MIN_VALID_LOGS} valid logs", flush=True)


if __name__ == "__main__":
  main()
