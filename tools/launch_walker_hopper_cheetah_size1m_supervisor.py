#!/usr/bin/env python3
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PYTHON = "/home/jiale/.conda/envs/dreamer/bin/python"
WSC_REPO = Path("/home/jiale/MBRL/ContinualMBRL-wsc")
BASELINE_REPO = Path("/home/jiale/MBRL/ContinualMBRL-baseline")
TASK = "walker_run|hopper_hop|cheetah_run"
PROJECT = (
    "continual_dreamer_soft_reset_"
    "walker_run|hopper_hop|cheetah_run_vision_size1m_1m_x5")
SEEDS = [1000, 2000, 3000]
TOTAL_STEPS = 15_000_000
TASK_INTERVAL = 1_000_000
TASK_REPEAT = 5
MIN_RAM_GIB = 30.0
GPU_MIN_FREE_MB = 2500
POLL_SECONDS = 30
RESTART_BACKOFF_SECONDS = 180
RUN_ID_SUFFIX = "whc-vision-size1m-1m-x5-v1"

STATE_DIR = WSC_REPO / "logdir" / PROJECT / "_supervisor"
STATE_PATH = STATE_DIR / "state.json"
EVENT_LOG = STATE_DIR / "events.jsonl"
SUMMARY_LOG = STATE_DIR / "supervisor.log"


def now():
  return datetime.now().strftime("%Y-%m-%d %H:%M:%S HKT")


def append(path, text):
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("a") as f:
    f.write(text)


def event(kind, **fields):
  row = {"time": now(), "kind": kind, **fields}
  append(EVENT_LOG, json.dumps(row, sort_keys=True) + "\n")
  append(SUMMARY_LOG, f"{row['time']} {kind} {fields}\n")
  print(f"{row['time']} {kind} {fields}", flush=True)


def load_dotenv(repo):
  env = {}
  path = repo / ".env.wandb.local"
  if not path.exists():
    return env
  for line in path.read_text(errors="replace").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
      continue
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip().strip("'").strip('"')
    if key:
      env[key] = value
  return env


def initial_jobs():
  specs = [
      {
          "group": "no_wsc",
          "repo": str(BASELINE_REPO),
          "kill_rank": 3,
          "mechanism": "disabled",
          "data_aug": "disabled",
          "extra": [
              "--run.reset_frequency", "0",
              "--run.reset_mechanism", "disabled",
              "--agent.wsc.mechanism", "disabled",
              "--agent.wsc.target", "all",
          ],
      },
      {
          "group": "wsc_skip_last_layer_constantinit_all",
          "repo": str(WSC_REPO),
          "kill_rank": 2,
          "mechanism": "wsc_skip_last_layer_constantinit_all",
          "data_aug": "disabled",
          "extra": [
              "--run.reset_frequency", "0",
              "--run.reset_mechanism", "wsc_skip_last_layer_constantinit_all",
              "--agent.wsc.mechanism", "wsc_skip_last_layer_constantinit_all",
              "--agent.wsc.target", "all",
              "--agent.wsc.target_norm", "1.0",
              "--agent.wsc.scale_factor", "0.999",
          ],
      },
      {
          "group": "l2_init_2e-4",
          "repo": str(BASELINE_REPO),
          "kill_rank": 1,
          "mechanism": "l2_init",
          "data_aug": "batch_align",
          "extra": [
              "--run.reset_frequency", "0",
              "--run.reset_mechanism", "l2_init",
              "--agent.wsc.mechanism", "l2_init",
              "--agent.wsc.target", "all",
              "--agent.wsc.weight_decay", "2e-4",
              "--agent.wsc.l2_init_weight", "2e-4",
              "--agent.wsc.cbp_eta", "0.99",
              "--agent.wsc.cbp_maturity", "5000",
              "--agent.wsc.cbp_replacement_rate", "1e-4",
          ],
      },
      {
          "group": "data_augmentation_batch_align",
          "repo": str(BASELINE_REPO),
          "kill_rank": 0,
          "mechanism": "disabled",
          "data_aug": "batch_align",
          "extra": [
              "--run.reset_frequency", "0",
              "--run.reset_mechanism", "disabled",
              "--agent.wsc.mechanism", "disabled",
              "--agent.wsc.target", "all",
          ],
      },
  ]
  jobs = []
  for spec in specs:
    for seed in SEEDS:
      repo = Path(spec["repo"])
      logdir = repo / "logdir" / PROJECT / spec["group"] / f"seed_{seed}"
      run_id = (
          f"{repo.name.replace('ContinualMBRL-', '')}-"
          f"walker-hopper-cheetah-{spec['group'].replace('_', '-')}-"
          f"{seed}-{RUN_ID_SUFFIX}")
      jobs.append({
          "id": f"{spec['group']}/seed_{seed}",
          "project": PROJECT,
          "group": spec["group"],
          "seed": seed,
          "repo": str(repo),
          "logdir": str(logdir),
          "run_id": run_id,
          "kill_rank": spec["kill_rank"],
          "mechanism": spec["mechanism"],
          "data_aug": spec["data_aug"],
          "extra": spec["extra"],
          "main_pid": None,
          "proxy_pid": None,
          "status": "pending",
          "attempts": 0,
          "last_start_time": 0.0,
          "last_stop_time": 0.0,
          "last_exit_time": 0.0,
          "last_reason": "",
          "gpu": None,
      })
  return jobs


def load_state():
  if STATE_PATH.exists():
    data = json.loads(STATE_PATH.read_text())
    by_id = {j["id"]: j for j in data.get("jobs", [])}
    jobs = []
    for job in initial_jobs():
      old = by_id.get(job["id"], {})
      merged = {**job, **old}
      jobs.append(merged)
    data["jobs"] = jobs
    return data
  return {
      "project": PROJECT,
      "created": now(),
      "jobs": initial_jobs(),
      "min_ram_gib": MIN_RAM_GIB,
      "gpu_min_free_mb": GPU_MIN_FREE_MB,
      "poll_seconds": POLL_SECONDS,
  }


def save_state(state):
  STATE_DIR.mkdir(parents=True, exist_ok=True)
  tmp = STATE_PATH.with_suffix(".tmp")
  tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
  tmp.replace(STATE_PATH)


def pid_alive(pid):
  return bool(pid) and Path(f"/proc/{pid}").exists()


def proc_cmdline(pid):
  try:
    raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    return [p.decode(errors="replace") for p in raw.split(b"\0") if p]
  except Exception:
    return []


def all_processes():
  out = {}
  for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
      continue
    pid = int(entry.name)
    argv = proc_cmdline(pid)
    if argv:
      out[pid] = argv
  return out


def attach_existing(job, procs):
  logdir = job["logdir"]
  main = None
  proxy = None
  for pid, argv in procs.items():
    joined = "\0".join(argv)
    if logdir not in joined:
      continue
    if any(a.endswith("dreamerv3/main.py") or a == "dreamerv3/main.py" for a in argv):
      main = pid
    elif any(a.endswith("wandb_metrics_proxy.py") for a in argv):
      proxy = pid
  if main and job.get("main_pid") != main:
    job["main_pid"] = main
    job["status"] = "running"
    event("attach_main", job=job["id"], pid=main)
  if proxy and job.get("proxy_pid") != proxy:
    job["proxy_pid"] = proxy
    event("attach_proxy", job=job["id"], pid=proxy)


def descendants(pid):
  if not pid_alive(pid):
    return []
  ppid = {}
  for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
      continue
    try:
      status = (entry / "status").read_text(errors="replace")
      m = re.search(r"^PPid:\s+(\d+)$", status, re.M)
      if m:
        ppid[int(entry.name)] = int(m.group(1))
    except Exception:
      pass
  children = {}
  for child, parent in ppid.items():
    children.setdefault(parent, []).append(child)
  result = []
  stack = list(children.get(pid, []))
  while stack:
    cur = stack.pop()
    result.append(cur)
    stack.extend(children.get(cur, []))
  return result


def stop_pid_tree(pid, reason):
  if not pid_alive(pid):
    return []
  targets = descendants(pid) + [pid]
  targets = [p for p in targets if pid_alive(p)]
  for p in targets:
    try:
      os.kill(p, signal.SIGTERM)
    except ProcessLookupError:
      pass
  deadline = time.time() + 20
  while time.time() < deadline:
    alive = [p for p in targets if pid_alive(p)]
    if not alive:
      return targets
    time.sleep(1)
  for p in targets:
    if pid_alive(p):
      try:
        os.kill(p, signal.SIGKILL)
      except ProcessLookupError:
        pass
  time.sleep(2)
  event("sigkill_leftovers", reason=reason, pids=[p for p in targets if pid_alive(p)])
  return targets


def stop_job(job, reason):
  killed = []
  for key in ("proxy_pid", "main_pid"):
    pid = job.get(key)
    if pid_alive(pid):
      killed.extend(stop_pid_tree(pid, reason))
    job[key] = None
  job["status"] = "paused_for_ram" if reason == "low_ram" else "stopped"
  job["last_stop_time"] = time.time()
  job["last_reason"] = reason
  event("stop_job", job=job["id"], reason=reason, killed=sorted(set(killed)))


def mem_available_gib():
  data = Path("/proc/meminfo").read_text().splitlines()
  values = {}
  for line in data:
    key, rest = line.split(":", 1)
    values[key] = int(rest.split()[0])
  return values.get("MemAvailable", 0) / (1024 ** 2)


def gpu_free_mb():
  try:
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=20,
        check=False,
    ).stdout
  except Exception as e:
    event("nvidia_smi_error", error=str(e))
    return {}
  result = {}
  for line in out.splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) >= 4:
      try:
        result[int(parts[0])] = {
            "free_mb": int(parts[1]),
            "used_mb": int(parts[2]),
            "util": int(parts[3]),
        }
      except ValueError:
        pass
  return result


def metrics_max_step(logdir):
  path = Path(logdir) / "metrics.jsonl"
  if not path.exists():
    return 0
  max_step = 0
  try:
    with path.open() as f:
      for line in f:
        if not line.strip():
          continue
        try:
          row = json.loads(line)
        except json.JSONDecodeError:
          continue
        step = int(row.get("step", 0))
        max_step = max(max_step, step)
  except Exception:
    return max_step
  return max_step


def choose_gpu(gpus, state):
  planned = {idx: 0 for idx in gpus}
  for job in state["jobs"]:
    if job.get("status") == "running" and job.get("gpu") is not None:
      planned[int(job["gpu"])] = planned.get(int(job["gpu"]), 0) + 1
  candidates = []
  for idx, info in gpus.items():
    score = info["free_mb"] - planned.get(idx, 0) * 2500
    if info["free_mb"] >= GPU_MIN_FREE_MB:
      candidates.append((score, info["free_mb"], -planned.get(idx, 0), idx))
  if not candidates:
    return None
  return sorted(candidates, reverse=True)[0][3]


def base_command(job, gpu):
  cmd = [
      PYTHON, "-u", "dreamerv3/main.py",
      "--configs", "dmc_vision", "continual_dmc_vision", "size1m",
      "--task", TASK,
      "--run.steps", str(TOTAL_STEPS),
      "--run.task_interval", str(TASK_INTERVAL),
      "--run.task_repeat", str(TASK_REPEAT),
      "--run.train_ratio", "256",
      "--run.log_every", "10000",
      "--run.report_every", "1000000000",
      "--run.report_batches", "0",
      "--run.save_every", "900",
      "--run.reset_target", "all",
      "--run.revive_epoch", "0",
      "--run.revive_strategy", "fixed",
      "--env.continual_dmc.task_action_space", "6",
      "--agent.redo.redo_enabled", "True",
      "--agent.redo.grad_redo_enabled", "True",
      "--agent.redo.act_log_item", "log+erank+srank",
      "--agent.redo.grad_log_item", "log+erank+srank",
      "--agent.data_augmentation.mode", job["data_aug"],
      "--agent.data_augmentation.pad", "4",
      "--agent.report", "False",
      "--jax.prealloc", "False",
      "--jax.memory_fraction", "0.25",
      "--seed", str(job["seed"]),
      "--logdir", job["logdir"],
      "--egl_device", str(gpu),
  ]
  return cmd + job["extra"]


def launch_proxy(job):
  repo = Path(job["repo"])
  proxy = BASELINE_REPO / "tools" / "wandb_metrics_proxy.py"
  if not proxy.exists():
    proxy = repo / "tools" / "wandb_metrics_proxy.py"
  cmd = [
      PYTHON, "-u", str(proxy),
      "--project", PROJECT,
      "--group", job["group"],
      "--seed", f"seed_{job['seed']}",
      "--run-id", job["run_id"],
      "--logdir", job["logdir"],
      "--interval", "30",
      "--flush-step",
      "--heartbeat-interval", "60",
  ]
  stdout = Path(job["logdir"]) / "wandb_proxy_supervisor.log"
  stdout.parent.mkdir(parents=True, exist_ok=True)
  env = os.environ.copy()
  env.update(load_dotenv(repo))
  env["WANDB_RESUME"] = "allow"
  env.setdefault("WANDB_INIT_TIMEOUT", "1800")
  handle = stdout.open("ab")
  proc = subprocess.Popen(
      cmd,
      cwd=str(repo),
      env=env,
      stdout=handle,
      stderr=subprocess.STDOUT,
      start_new_session=True,
  )
  handle.close()
  job["proxy_pid"] = proc.pid
  event("launch_proxy", job=job["id"], pid=proc.pid)


def launch_job(job, gpu):
  repo = Path(job["repo"])
  logdir = Path(job["logdir"])
  logdir.mkdir(parents=True, exist_ok=True)
  if pid_alive(job.get("proxy_pid")):
    stopped = stop_pid_tree(job.get("proxy_pid"), "relaunch")
    event("stop_existing_proxy_before_launch", job=job["id"], killed=sorted(set(stopped)))
    job["proxy_pid"] = None
  stdout = logdir / "train.log"
  cmd = base_command(job, gpu)
  env = os.environ.copy()
  env.update(load_dotenv(repo))
  env["CUDA_VISIBLE_DEVICES"] = str(gpu)
  env["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
  env["WANDB_RUN_ID"] = job["run_id"]
  env["WANDB_RESUME"] = "allow"
  env.setdefault("WANDB_INIT_TIMEOUT", "1800")
  handle = stdout.open("ab")
  proc = subprocess.Popen(
      cmd,
      cwd=str(repo),
      env=env,
      stdout=handle,
      stderr=subprocess.STDOUT,
      start_new_session=True,
  )
  handle.close()
  job["main_pid"] = proc.pid
  job["gpu"] = gpu
  job["status"] = "running"
  job["attempts"] = int(job.get("attempts", 0)) + 1
  job["last_start_time"] = time.time()
  job["last_reason"] = "launched"
  (logdir / "pid.supervisor").write_text(str(proc.pid) + "\n")
  (logdir / "launch_command.supervisor.json").write_text(json.dumps({
      "time": now(),
      "cwd": str(repo),
      "cmd": cmd,
      "env": {
          "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"],
          "MUJOCO_EGL_DEVICE_ID": env["MUJOCO_EGL_DEVICE_ID"],
          "WANDB_RUN_ID": env["WANDB_RUN_ID"],
          "WANDB_RESUME": env["WANDB_RESUME"],
      },
  }, indent=2))
  event("launch_main", job=job["id"], pid=proc.pid, gpu=gpu, attempt=job["attempts"])
  launch_proxy(job)


def update_job_liveness(job):
  main_alive = pid_alive(job.get("main_pid"))
  proxy_alive = pid_alive(job.get("proxy_pid"))
  max_step = metrics_max_step(job["logdir"])
  job["max_step"] = max_step
  if max_step >= TOTAL_STEPS:
    if job.get("status") != "completed":
      event("job_completed", job=job["id"], max_step=max_step)
    job["status"] = "completed"
    return
  if job.get("status") == "running" and not main_alive:
    job["status"] = "failed"
    job["last_exit_time"] = time.time()
    job["last_reason"] = "main_not_alive"
    event("main_exited", job=job["id"], old_pid=job.get("main_pid"), max_step=max_step)
    job["main_pid"] = None
    if proxy_alive:
      stopped = stop_pid_tree(job.get("proxy_pid"), "main_failed")
      event("stop_orphan_proxy", job=job["id"], killed=sorted(set(stopped)))
      job["proxy_pid"] = None
  if job.get("status") == "running" and main_alive and not proxy_alive:
    event("proxy_exited", job=job["id"], old_pid=job.get("proxy_pid"))
    job["proxy_pid"] = None


def enforce_ram(state):
  mem_gib = mem_available_gib()
  stopped = []
  while mem_gib < MIN_RAM_GIB:
    running = [
        j for j in state["jobs"]
        if j.get("status") == "running" and pid_alive(j.get("main_pid"))]
    if not running:
      event("low_ram_no_jobs_left", mem_gib=round(mem_gib, 2))
      break
    victim = sorted(running, key=lambda j: (j["kill_rank"], j["seed"]))[0]
    stop_job(victim, "low_ram")
    stopped.append(victim["id"])
    time.sleep(5)
    mem_gib = mem_available_gib()
  return mem_gib, stopped


def maybe_launch_jobs(state, mem_gib, gpus):
  if mem_gib < MIN_RAM_GIB:
    return
  for job in state["jobs"]:
    update_job_liveness(job)
  for job in state["jobs"]:
    if job.get("status") == "completed":
      continue
    if job.get("status") == "running":
      if not pid_alive(job.get("proxy_pid")):
        launch_proxy(job)
      continue
    if job.get("status") == "paused_for_ram" and mem_gib < MIN_RAM_GIB + 10:
      continue
    if time.time() - float(job.get("last_start_time", 0.0)) < RESTART_BACKOFF_SECONDS:
      continue
    gpu = choose_gpu(gpus, state)
    if gpu is None:
      event("launch_wait_no_gpu", job=job["id"], min_free_mb=GPU_MIN_FREE_MB)
      return
    launch_job(job, gpu)
    save_state(state)
    time.sleep(5)
    mem_gib = mem_available_gib()
    gpus = gpu_free_mb()
    if mem_gib < MIN_RAM_GIB:
      return


def summarize(state, mem_gib, gpus):
  counts = {}
  for job in state["jobs"]:
    counts[job.get("status", "unknown")] = counts.get(job.get("status", "unknown"), 0) + 1
  gpu_summary = {idx: info["free_mb"] for idx, info in sorted(gpus.items())}
  event("poll", mem_gib=round(mem_gib, 2), counts=counts, gpu_free_mb=gpu_summary)


def main():
  STATE_DIR.mkdir(parents=True, exist_ok=True)
  state = load_state()
  event("supervisor_start", project=PROJECT, state=str(STATE_PATH))
  procs = all_processes()
  for job in state["jobs"]:
    attach_existing(job, procs)
  save_state(state)
  while True:
    procs = all_processes()
    for job in state["jobs"]:
      attach_existing(job, procs)
      update_job_liveness(job)
    mem_gib, stopped = enforce_ram(state)
    gpus = gpu_free_mb()
    maybe_launch_jobs(state, mem_gib, gpus)
    save_state(state)
    summarize(state, mem_available_gib(), gpu_free_mb())
    time.sleep(POLL_SECONDS)


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    event("supervisor_keyboard_interrupt")
    sys.exit(130)
