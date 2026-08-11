#!/usr/bin/env python3
"""Resume and foreground-monitor the non-constant DMC vision runs."""

import json
import os
import pathlib
import re
import shlex
import subprocess
import time
from datetime import datetime


ROOT = pathlib.Path(__file__).resolve().parents[1]
WSC_ROOT = pathlib.Path("/home/jiale/MBRL/ContinualMBRL-wsc")
PYTHON = pathlib.Path("/home/jiale/.conda/envs/dreamer/bin/python")
ENV_FILE = ROOT / ".env.wandb.local"
PROJECT = "continual_dreamer_soft_reset_walker_run|hopper_hop|fish_swim_vision_size12m_1m_x5"
GROUP = "data_augmentation_batch_align"
RUN_VERSION = "dmc-vision-size12m-da-align-1m-x5-v1"
DA_LOG_ROOT = ROOT / "logdir" / PROJECT / GROUP
WSC_VISION_LOG_ROOT = WSC_ROOT / "logdir" / PROJECT
STATE_FILE = ROOT / "logdir" / "scheduler" / "foreground_experiment_monitor.json"
CHECK_SECONDS = 3600
STARTUP_CHECK_SECONDS = 120
STALE_METRICS_SECONDS = 3 * 3600
MIN_FPS = 4.0
MAX_VISION_DA_GPU_MB = 9000
IGNORED_LOGDIR_FRAGMENTS = (
    "vision_size12m_1m_x5/wsc_WSC_skip_last_layer_constant_all",
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


def run(cmd, **kwargs):
  return subprocess.check_output(cmd, text=True, timeout=kwargs.pop("timeout", 30), **kwargs)


def gpu_order():
  try:
    out = run([
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
  except Exception:
    return [0]
  rows = []
  for line in out.splitlines():
    parts = [int(x.strip()) for x in line.split(",")]
    idx, used, total, util = parts
    rows.append((used / max(total, 1), util, idx))
  return [idx for _, _, idx in sorted(rows)]


def gpu_memory_by_pid():
  try:
    out = run([
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ])
  except Exception:
    return {}
  result = {}
  for line in out.splitlines():
    parts = [x.strip() for x in line.split(",")]
    if len(parts) != 2:
      continue
    try:
      result[int(parts[0])] = result.get(int(parts[0]), 0) + int(parts[1])
    except ValueError:
      pass
  return result


def process_alive(pid):
  try:
    stat = run(["ps", "-p", str(pid), "-o", "stat="], timeout=5).strip()
  except Exception:
    return False
  return bool(stat) and not stat.startswith("Z")


def proc_cwd(pid):
  try:
    return pathlib.Path(os.readlink(f"/proc/{pid}/cwd"))
  except Exception:
    return ROOT


def proc_cmd(pid):
  try:
    data = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes()
  except Exception:
    return []
  return [x.decode() for x in data.split(b"\0") if x]


def proc_env(pid):
  env = {}
  try:
    data = pathlib.Path(f"/proc/{pid}/environ").read_bytes()
  except Exception:
    return env
  for item in data.split(b"\0"):
    if b"=" in item:
      key, value = item.split(b"=", 1)
      env[key.decode(errors="ignore")] = value.decode(errors="ignore")
  return env


def option_value(cmd, name):
  for i, item in enumerate(cmd):
    if item == name and i + 1 < len(cmd):
      return cmd[i + 1]
    if item.startswith(name + "="):
      return item.split("=", 1)[1]
  return None


def normalize_logdir(value, cwd):
  if not value:
    return None
  path = pathlib.Path(value)
  if not path.is_absolute():
    path = cwd / path
  try:
    return path.resolve()
  except Exception:
    return path


def ignored_logdir(logdir):
  text = str(logdir)
  return any(fragment in text for fragment in IGNORED_LOGDIR_FRAGMENTS)


def dreamer_processes():
  procs = []
  for procdir in pathlib.Path("/proc").iterdir():
    if not procdir.name.isdigit():
      continue
    try:
      pid = int(procdir.name)
    except ValueError:
      continue
    cmd = proc_cmd(pid)
    if "dreamerv3/main.py" not in " ".join(cmd):
      continue
    cwd = proc_cwd(pid)
    logdir = normalize_logdir(option_value(cmd, "--logdir"), cwd)
    if logdir and ignored_logdir(logdir):
      continue
    procs.append({
        "pid": pid,
        "cmd": cmd,
        "cwd": str(cwd),
        "logdir": str(logdir) if logdir else "",
        "seed": option_value(cmd, "--seed"),
        "egl_device": option_value(cmd, "--egl_device"),
    })
  return sorted(procs, key=lambda x: x["pid"])


def latest_metrics(logdir):
  path = pathlib.Path(logdir) / "metrics.jsonl"
  if not path.exists():
    return None, 0, None
  latest = None
  count = 0
  with path.open() as stream:
    for line in stream:
      line = line.strip()
      if not line:
        continue
      try:
        latest = json.loads(line)
        count += 1
      except json.JSONDecodeError:
        continue
  return latest, count, path.stat().st_mtime


def has_wandb_output(logdir):
  root = pathlib.Path(logdir) / "wandb" / "wandb"
  return any(root.glob("run-*/run-*.wandb")) or any(root.glob("run-*/logs/debug-internal.log"))


def wandb_run_id(logdir):
  root = pathlib.Path(logdir) / "wandb" / "wandb"
  runs = sorted(root.glob("run-*"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
  for path in runs:
    match = re.match(r"run-\d+_\d+-(.+)", path.name)
    if match:
      return match.group(1)
  return None


def da_command(seed, logdir, gpu):
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


def wsc_vision_command(seed, group, mechanism, logdir, gpu):
  cmd = [
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
      "--run.reset_mechanism", mechanism,
      "--run.reset_target", "all",
      "--run.revive_epoch", "0",
      "--run.revive_strategy", "fixed",
      "--env.continual_dmc.task_action_space", "6",
      "--agent.wsc.target", "all",
      "--agent.redo.redo_enabled", "True",
      "--agent.redo.grad_redo_enabled", "True",
      "--agent.redo.act_log_item", "log+erank+srank",
      "--agent.redo.grad_log_item", "log+erank+srank",
      "--jax.prealloc", "False",
      "--jax.memory_fraction", "0.25",
  ]
  if mechanism == "wsc_skip_last_layer_init_all":
    cmd += [
        "--agent.wsc.target_norm", "1.0",
        "--agent.wsc.scale_factor", "0.999",
    ]
  cmd += [
      "--agent.wsc.mechanism", mechanism,
      "--logdir", str(logdir),
      "--seed", str(seed),
      "--egl_device", str(gpu),
  ]
  return cmd


def launch(cmd, cwd, logdir, gpu, wandb_id=None):
  env = load_env()
  env.update({
      "CUDA_VISIBLE_DEVICES": str(gpu),
      "MUJOCO_EGL_DEVICE_ID": str(gpu),
      "WANDB_RESUME": "allow",
  })
  xla_flags = env.get("XLA_FLAGS", "")
  fallback_flag = "--xla_gpu_strict_conv_algorithm_picker=false"
  if fallback_flag not in xla_flags:
    env["XLA_FLAGS"] = (xla_flags + " " + fallback_flag).strip()
  if wandb_id:
    env["WANDB_RUN_ID"] = wandb_id
  pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)
  logfile = pathlib.Path(logdir) / "train.log"
  with logfile.open("ab", buffering=0) as stream:
    stream.write((f"\n[{now()}] Resume: {' '.join(shlex.quote(x) for x in cmd)}\n").encode())
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
        start_new_session=True)
  print(f"[{now()}] launched pid={proc.pid} gpu={gpu} logdir={logdir}", flush=True)
  return proc.pid


def da_jobs():
  gpus = gpu_order()
  jobs = []
  for i, seed in enumerate((1000, 2000, 3000)):
    logdir = DA_LOG_ROOT / f"seed_{seed}"
    jobs.append({
        "seed": seed,
        "logdir": str(logdir),
        "wandb_id": f"baseline-dmc-vision-size12m-da-align-{seed}-{RUN_VERSION}",
        "gpu": gpus[i % len(gpus)],
    })
  return jobs


def protected_vision_jobs():
  gpus = gpu_order()
  jobs = []
  for i, job in enumerate(da_jobs()):
    job.update({
        "id": f"da:{job['seed']}",
        "cwd": str(ROOT),
        "command": da_command(job["seed"], job["logdir"], job["gpu"]),
        "kind": "data_augmentation_batch_align",
    })
    jobs.append(job)
  no_wsc_ids = {1000: "5vxj0d8v", 2000: "3defkxis", 3000: "ncfplgvj"}
  init_ids = {1000: "rls2vu7r", 2000: "q3kac19i", 3000: "o4o8d9sg"}
  specs = (
      ("no_wsc", "disabled", no_wsc_ids),
      ("wsc_WSC_skip_last_layer_init_all", "wsc_skip_last_layer_init_all", init_ids),
  )
  offset = len(jobs)
  for group, mechanism, run_ids in specs:
    for seed in (1000, 2000, 3000):
      gpu = gpus[offset % len(gpus)]
      offset += 1
      logdir = WSC_VISION_LOG_ROOT / group / f"seed_{seed}"
      jobs.append({
          "id": f"{group}:{seed}",
          "kind": group,
          "group": group,
          "seed": seed,
          "gpu": gpu,
          "cwd": str(WSC_ROOT),
          "logdir": str(logdir),
          "wandb_id": run_ids[seed],
          "command": wsc_vision_command(seed, group, mechanism, logdir, gpu),
      })
  return jobs


def procs_for_logdir(logdir, procs):
  target = str(pathlib.Path(logdir).resolve())
  return [proc for proc in procs if proc["logdir"] == target and process_alive(proc["pid"])]


def alive_for_logdir(logdir, procs):
  matches = procs_for_logdir(logdir, procs)
  if matches:
    return matches[0]
  return None


def duplicate_protected_issues(jobs, procs):
  issues = []
  for job in jobs:
    matches = procs_for_logdir(job["logdir"], procs)
    if len(matches) > 1:
      pids = ",".join(str(proc["pid"]) for proc in matches)
      issues.append(f"duplicate_protected {job['id']} pids={pids} logdir={job['logdir']}")
  return issues


def find_process_for_logdir(logdir, procs):
  target = str(pathlib.Path(logdir).resolve())
  for proc in procs:
    if proc["logdir"] == target and process_alive(proc["pid"]):
      return proc
  return None


def resume_protected_vision_runs(state):
  procs = dreamer_processes()
  for job in protected_vision_jobs():
    alive = alive_for_logdir(job["logdir"], procs)
    if alive:
      state.setdefault("protected_pids", {})[job["id"]] = alive["pid"]
      print(f"[{now()}] protected {job['id']} already alive pid={alive['pid']}", flush=True)
      continue
    pid = launch(job["command"], pathlib.Path(job["cwd"]), job["logdir"], job["gpu"], job["wandb_id"])
    state.setdefault("protected_pids", {})[job["id"]] = pid
    state.setdefault("protected_launch_steps", {})[job["id"]] = latest_step(job["logdir"])
    if job["kind"] == GROUP:
      state["da_pids"][str(job["seed"])] = pid
      state["da_launch_steps"][str(job["seed"])] = latest_step(job["logdir"])


def latest_step(logdir):
  metrics, _, _ = latest_metrics(logdir)
  if not metrics:
    return None
  return metrics.get("step")


def validate_da(seed, logdir, launched_step):
  metrics, count, mtime = latest_metrics(logdir)
  if not metrics:
    return False, "no metrics yet"
  missing = []
  for key in (
      "train/data_augmentation/active",
      "train/data_augmentation/batch_align",
      "fps/policy",
      "fps/train",
  ):
    if key not in metrics:
      missing.append(key)
  for prefix in (
      "act_redo/Zombie_Percentage/",
      "act_redo/Saturation_Percentage/",
      "act_redo/Variation_Rank_0.99/",
      "grad_redo/",
      "data_diversity/",
  ):
    if not any(k.startswith(prefix) for k in metrics):
      missing.append(prefix)
  step = metrics.get("step")
  fps = float(metrics.get("fps/policy", 0) or 0)
  active = float(metrics.get("train/data_augmentation/active", 0) or 0)
  align = float(metrics.get("train/data_augmentation/batch_align", 0) or 0)
  if missing:
    return False, f"step={step} missing={missing}"
  if active != 1.0 or align != 1.0:
    return False, f"step={step} da_active={active} da_align={align}"
  if fps < MIN_FPS:
    return False, f"step={step} fps={fps:.2f} < {MIN_FPS}"
  if launched_step is not None and step is not None and step <= launched_step:
    return False, f"waiting for post-resume metrics step>{launched_step}, current={step}"
  if not has_wandb_output(logdir):
    return False, "no local wandb output"
  age = time.time() - mtime
  if age > STALE_METRICS_SECONDS:
    return False, f"records={count} step={step} fps={fps:.2f} stale_metrics_age={age:.0f}s"
  return True, f"records={count} step={step} fps={fps:.2f} metrics_age={age:.0f}s"


def validate_protected(job, state):
  launched_step = state.get("protected_launch_steps", {}).get(job["id"])
  if job["kind"] == GROUP:
    return validate_da(job["seed"], job["logdir"], launched_step)
  metrics, count, mtime = latest_metrics(job["logdir"])
  if not metrics:
    return False, "no metrics yet"
  step = metrics.get("step")
  fps = float(metrics.get("fps/policy", 0) or 0)
  if launched_step is not None and step is not None and step <= launched_step:
    return False, f"waiting for post-resume metrics step>{launched_step}, current={step}"
  if "fps/policy" in metrics and fps < MIN_FPS:
    return False, f"step={step} fps={fps:.2f} < {MIN_FPS}"
  if not has_wandb_output(job["logdir"]):
    return False, "no local wandb output"
  missing = []
  for prefix in (
      "act_redo/Zombie_Percentage/",
      "act_redo/Saturation_Percentage/",
      "act_redo/Variation_Rank_0.99/",
      "grad_redo/",
      "data_diversity/",
  ):
    if not any(k.startswith(prefix) for k in metrics):
      missing.append(prefix)
  if missing:
    return False, f"step={step} missing={missing}"
  age = time.time() - mtime
  if age > STALE_METRICS_SECONDS:
    return False, f"records={count} step={step} fps={fps:.2f} stale_metrics_age={age:.0f}s"
  return True, f"records={count} step={step} fps={fps:.2f} metrics_age={age:.0f}s"


def snapshot_running(state):
  current = {}
  for proc in dreamer_processes():
    if not proc["logdir"]:
      continue
    if ignored_logdir(proc["logdir"]):
      continue
    current[proc["logdir"]] = {
        "pid": proc["pid"],
        "cmd": proc["cmd"],
        "cwd": proc["cwd"],
        "seed": proc["seed"],
        "egl_device": proc["egl_device"],
        "wandb_id": wandb_run_id(proc["logdir"]),
        "first_seen": now(),
    }
  state["tracked"] = current


def check_tracked(state):
  procs = {proc["logdir"]: proc for proc in dreamer_processes() if proc["logdir"]}
  gpu_mem = gpu_memory_by_pid()
  issues = []
  for logdir, tracked in list(state.get("tracked", {}).items()):
    if ignored_logdir(logdir):
      state["tracked"].pop(logdir, None)
      continue
    proc = procs.get(logdir)
    if not proc or not process_alive(proc["pid"]):
      issues.append(f"missing pid={tracked.get('pid')} logdir={logdir}")
      continue
    metrics, count, mtime = latest_metrics(logdir)
    if not metrics:
      issues.append(f"no_metrics pid={proc['pid']} logdir={logdir}")
      continue
    age = time.time() - mtime
    fps = float(metrics.get("fps/policy", metrics.get("fps/train", 0)) or 0)
    if age > STALE_METRICS_SECONDS:
      issues.append(f"stale_metrics pid={proc['pid']} age={age:.0f}s logdir={logdir}")
    if "fps/policy" in metrics and fps < MIN_FPS:
      issues.append(f"low_fps pid={proc['pid']} fps={fps:.2f} logdir={logdir}")
    if not has_wandb_output(logdir):
      issues.append(f"missing_wandb_output pid={proc['pid']} logdir={logdir}")
    if GROUP in logdir and PROJECT in logdir and gpu_mem.get(proc["pid"], 0) > MAX_VISION_DA_GPU_MB:
      issues.append(f"high_cuda pid={proc['pid']} mem={gpu_mem.get(proc['pid'])}MB logdir={logdir}")
  return issues


def write_state(state):
  STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
  tmp = STATE_FILE.with_suffix(".tmp")
  tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
  tmp.replace(STATE_FILE)


def load_state():
  if STATE_FILE.exists():
    try:
      return json.loads(STATE_FILE.read_text())
    except Exception:
      pass
  return {"started_at": now(), "da_pids": {}, "da_launch_steps": {}, "tracked": {}}


def main():
  if not PYTHON.exists():
    raise SystemExit(f"Missing dreamer python: {PYTHON}")
  state = load_state()
  state["tracked"] = {}
  state["protected_launch_steps"] = {}
  state["da_launch_steps"] = {}
  resume_protected_vision_runs(state)
  snapshot_running(state)
  write_state(state)
  print(f"[{now()}] foreground monitor active; hourly checks will continue until interrupted", flush=True)
  while True:
    jobs = protected_vision_jobs()
    procs = dreamer_processes()
    gpu_mem = gpu_memory_by_pid()
    protected_ok = []
    duplicate_issues = duplicate_protected_issues(jobs, procs)
    for issue in duplicate_issues:
      print(f"[{now()}] monitor issue: {issue}", flush=True)
    for job in jobs:
      alive = alive_for_logdir(job["logdir"], procs)
      if not alive:
        print(f"[{now()}] protected {job['id']} not alive; resuming", flush=True)
        pid = launch(job["command"], pathlib.Path(job["cwd"]), job["logdir"], job["gpu"], job["wandb_id"])
        state.setdefault("protected_pids", {})[job["id"]] = pid
        state.setdefault("protected_launch_steps", {})[job["id"]] = latest_step(job["logdir"])
        if job["kind"] == GROUP:
          state["da_pids"][str(job["seed"])] = pid
          state["da_launch_steps"][str(job["seed"])] = latest_step(job["logdir"])
      else:
        state.setdefault("protected_pids", {})[job["id"]] = alive["pid"]
        if job["kind"] == GROUP:
          state["da_pids"][str(job["seed"])] = alive["pid"]
        mem = gpu_mem.get(alive["pid"], 0)
        ok, detail = validate_protected(job, state)
        protected_ok.append(ok)
        print(
            f"[{now()}] protected {job['id']} pid={alive['pid']} "
            f"gpu={alive.get('egl_device')} mem={mem}MB ok={ok} {detail}",
            flush=True)
    snapshot_running(state)
    issues = check_tracked(state)
    if issues:
      print(f"[{now()}] monitor issues ({len(issues)}):", flush=True)
      for issue in issues[:80]:
        print(f"  - {issue}", flush=True)
    else:
      print(f"[{now()}] all tracked Dreamer processes have recent local metrics and wandb output", flush=True)
    state["last_check"] = now()
    state["last_issue_count"] = len(issues)
    write_state(state)
    time.sleep(STARTUP_CHECK_SECONDS if not all(protected_ok) else CHECK_SECONDS)


if __name__ == "__main__":
  main()
