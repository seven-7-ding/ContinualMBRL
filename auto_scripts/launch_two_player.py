#!/usr/bin/env python3
"""Launch/resume the active two-player runs, then monitor five valid logs per run.

Uses only its persisted PIDs. Never signals or modifies unrelated processes.
Run with the dreamer Python; --plan writes a reviewable manifest without launch.
"""
from __future__ import annotations

import argparse
import collections
import fcntl
import functools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / 'logdir/two_player_sweep_20261015'
sys.path.insert(0, str(ROOT))
from tools.inspect_wandb_reference import credentials
from tools import wandb_transport
from embodied.run.two_player import update_counts

PROJECT = 'two_player_dreamer_walker_run|finger_spin|fish_swim_size1m_500k_x10'
TASK = 'walker_run|finger_spin|fish_swim'
STATE = CONTROL / 'manifest.json'
SETTINGS = ['default'] + [f'{mode}_{k}' for k in (32,)
    for mode in ('wm_as_leader', 'agent_as_leader', 'best_response')]
SEEDS = (1000, 2000, 3000)
STEPS_PER_TASK = 500000
CYCLES = 10
TRAIN_RATIO = 256
ENVS = 16
LOG_EVERY = 10000
K32_LOG_EVERY = 2000
SAVE_SECONDS = 900
REDO_FREQUENCY = 100


def save(state):
  STATE.parent.mkdir(parents=True, exist_ok=True)
  tmp = STATE.with_suffix('.tmp')
  tmp.write_text(json.dumps(state, indent=2) + '\n')
  tmp.replace(STATE)


def gpu_stats():
  out = subprocess.check_output(['nvidia-smi',
      '--query-gpu=index,memory.used,memory.total,utilization.gpu',
      '--format=csv,noheader,nounits'], text=True, timeout=20)
  return [dict(zip(('index', 'used', 'total', 'util'), map(int, line.split(','))))
          for line in out.strip().splitlines()]


def available_ram_gib():
  import psutil
  return psutil.virtual_memory().available / 2**30


def alive(job):
  import psutil
  pid = job.get('pid')
  if not pid or not psutil.pid_exists(pid):
    return False
  try:
    proc = psutil.Process(pid)
    return (proc.status() != psutil.STATUS_ZOMBIE and
            abs(proc.create_time() - job['create_time']) < 1)
  except psutil.Error:
    return False


def log_interval(job):
  return K32_LOG_EVERY if job['setting'].endswith('_32') else LOG_EVERY


def command(job):
  optimized = job.get('performance_profile') in ('bounded_replay_v1','bounded_replay_v2')
  return [sys.executable, '-B', '-u', str(ROOT / 'dreamerv3/main.py'),
      '--configs', 'continual_dmc_priori', 'size1m',
      '--task', TASK, '--seed', str(job['seed']), '--logdir', job['logdir'],
      '--run.two_player', job['setting'], '--run.latest_train_metrics', 'True',
      '--run.train_ratio', str(TRAIN_RATIO), '--run.envs', str(ENVS),
      '--run.task_interval', str(STEPS_PER_TASK), '--run.task_repeat', str(CYCLES),
      '--run.steps', str(STEPS_PER_TASK * CYCLES * 3), '--run.log_every', str(log_interval(job)),
      '--run.fused_updates', 'True', '--run.update_batch', '8',
      '--run.prefetch_batches', '8', '--run.report_every', '20000',
      '--run.save_every', str(SAVE_SECONDS), '--run.reset_frequency', '0',
      '--run.reset_mechanism', 'disabled', '--run.revive_epoch', '0',
      '--agent.wsc.mechanism', 'disabled',
      '--agent.opt.wd', '0', '--agent.log_gradients', 'True',
      '--agent.redo.redo_enabled', 'True', '--agent.redo.grad_redo_enabled', 'True',
      '--agent.redo.act_log_item', 'log+erank+srank',
      '--agent.redo.grad_log_item', 'log+erank+srank',
      '--agent.redo.frequency', str(REDO_FREQUENCY), '--agent.report', 'True',
      '--agent.report_gradnorms', 'True', '--logger.outputs', 'jsonl', 'scope', 'wandb',
      '--logger.filter', 'score|fps|loss/|schedule/',
      '--run.usage.psutil', 'True', '--run.usage.nvsmi', 'True',
      '--run.usage.gc', 'True', '--run.usage.malloc', str(not optimized),
      '--jax.prealloc', 'False', '--jax.memory_fraction', '0.18',
      '--jax.precompile', 'False',
      '--jax.profiler', 'False', '--egl_device', str(job['gpu']),
      '--replay.size', '500000', '--replay.chunksize', '1024',
      '--replay.cache_chunks', '0' if optimized else '1024',
      '--replay.cache_bytes', str(2 * 1024**3 if optimized else 0)]


@functools.lru_cache(None)
def work(setting):
  wm, actor = update_counts(setting)
  benchmark = CONTROL / 'benchmark_gpu/result.json'
  cost = 1.0
  if benchmark.exists():
    rates = json.loads(benchmark.read_text())
    if rates.get('wm_updates_per_second', 0) > 0:
      cost = rates['updates_per_second']['fused_8'] / rates['wm_updates_per_second']
  return wm * cost + actor


def plan(existing=None):
  gpus = gpu_stats()
  loads = {g['index']: g['util'] / 10 for g in gpus}
  counts = collections.Counter()
  jobs = []
  prior = {(j['setting'], j['seed']): j for j in (existing or {}).get('jobs', [])}
  for job in prior.values():
    if job['attempts']:
      jobs.append(job)
      loads[job['gpu']] += work(job['setting'])
      counts[job['gpu']] += 1
  specs = [(setting, seed) for setting in SETTINGS for seed in SEEDS
           if not prior.get((setting, seed), {}).get('attempts')]
  specs.sort(key=lambda pair: -work(pair[0]))
  for setting, seed in specs:
    candidates = [g for g in gpus if counts[g['index']] < 3 and
                  g['total'] - g['used'] - 4608 * counts[g['index']] >= 6656]
    if not candidates:
      raise RuntimeError('Insufficient GPU headroom for all runs; no process launched')
    gpu = min(candidates, key=lambda g: (loads[g['index']], g['used']))['index']
    loads[gpu] += work(setting)
    counts[gpu] += 1
    logdir = ROOT / 'logdir' / PROJECT / setting / f'seed_{seed}'
    jobs.append({'setting': setting, 'seed': seed, 'gpu': gpu,
        'logdir': str(logdir), 'wandb_id': f'two-player-{setting}-{seed}-v2-step10k',
        'pid': None, 'attempts': 0, 'valid_logs': 0})
  # Improve the greedy assignment by swapping pending runs. Started jobs stay
  # fixed; swaps preserve per-GPU process counts and memory reservations.
  pending = [j for j in jobs if not j['attempts']]
  for _ in range(100):
    best = None
    improvement = 0
    for index, left in enumerate(pending):
      for right in pending[index + 1:]:
        a, b = left['gpu'], right['gpu']
        if a == b:
          continue
        change = work(right['setting']) - work(left['setting'])
        delta = 2 * change * (loads[a] - loads[b]) + 2 * change ** 2
        if delta < improvement - 1e-6:
          improvement, best = delta, (left, right, change)
    if best is None:
      break
    left, right, change = best
    loads[left['gpu']] += change
    loads[right['gpu']] -= change
    left['gpu'], right['gpu'] = right['gpu'], left['gpu']
  # Resource checks also apply immediately before each actual launch.
  return {**(existing or {}), 'project': PROJECT, 'jobs': jobs,
          'created_at': (existing or {}).get('created_at', time.time()),
          'allocation_workloads': loads,
          'baseline_gpu': gpus, 'baseline_ram_available_gib': available_ram_gib()}


def launch(job, resume='allow'):
  import psutil
  if job.get('user_paused'):
    return False
  stop_plan = CONTROL/'watchdog/scheduled_stop_plan.json'
  if stop_plan.exists():
    intent = json.loads(stop_plan.read_text())
    if (intent.get('enabled') and job['wandb_id'] in intent.get('run_ids',[]) and
        time.time() >= intent['quiesce_at']):
      return False
  g = next(g for g in gpu_stats() if g['index'] == job['gpu'])
  if g['total'] - g['used'] < 6656 or available_ram_gib() < 80:
    return False
  env = wandb_transport.sdk_environment(credentials())
  env.update(WANDB_RUN_ID=job['wandb_id'], WANDB_RESUME=resume, WANDB_MODE='online',
      WANDB_SILENT='true',
      CONTINUALMBRL_REPLAY_PROFILE=job.get('performance_profile','baseline'),
      CUDA_VISIBLE_DEVICES=str(job['gpu']), MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
      OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
      NUMEXPR_NUM_THREADS='1', XLA_PYTHON_CLIENT_PREALLOCATE='false',
      PYTHONDONTWRITEBYTECODE='1')
  env['CONTINUALMBRL_STOP_PLAN'] = str(stop_plan)
  env['CONTINUALMBRL_STOP_PROTOCOL'] = 'checkpoint_stop_v1'
  env['PATH'] = '/usr/local/cuda/bin:' + env.get('PATH', '')
  env['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=false'
  directory = Path(job['logdir'])
  directory.mkdir(parents=True, exist_ok=True)
  ready = directory / 'training_ready.json'
  if ready.exists():
    ready.replace(directory / f"training_ready.attempt_{job['attempts']}.json")
  paths = {'TMPDIR': 'tmp', 'TMP': 'tmp', 'TEMP': 'tmp',
           'XDG_CACHE_HOME': 'cache', 'CUDA_CACHE_PATH': 'cache/cuda',
           'MPLCONFIGDIR': 'cache/matplotlib', 'WANDB_CACHE_DIR': 'cache/wandb',
           'WANDB_CONFIG_DIR': 'config/wandb', 'WANDB_DATA_DIR': 'wandb_staging',
           'JAX_COMPILATION_CACHE_DIR': 'cache/jax'}
  for key, relative in paths.items():
    path = directory / relative
    path.mkdir(parents=True, exist_ok=True)
    env[key] = str(path)
  # Seed private caches from completed immutable entries of earlier runs.
  # Each child then writes exclusively to its own run directory.
  cache = directory / 'cache/jax'
  donors = [CONTROL / 'cache/jax_bootstrap', CONTROL / 'benchmark_gpu/cache/jax',
            *(ROOT / 'logdir' / PROJECT).glob('*/seed_*/cache/jax')]
  for donor in donors:
    if donor == cache:
      continue
    for source in donor.glob('*-cache'):
      target = cache / source.name
      if not target.exists():
        shutil.copy2(source, target)
  cmd = command(job)
  # Restrict new JAX pools to two CPUs each, with shared balanced affinity.
  cpus = sorted(os.sched_getaffinity(0))
  idx = list(SETTINGS).index(job['setting']) * 3 + SEEDS.index(job['seed'])
  stride = max(2, len(cpus) // (len(SETTINGS) * len(SEEDS)))
  pair = [cpus[(stride * idx) % len(cpus)], cpus[(stride * idx + 1) % len(cpus)]]
  job['cpus'] = pair
  cmd = ['taskset', '-c', ','.join(map(str, pair)), *cmd]
  (directory / 'launch.json').write_text(json.dumps({
      'command': cmd, 'gpu': job['gpu'], 'wandb_id': job['wandb_id'],
      'project': PROJECT}, indent=2) + '\n')
  job['attempts'] += 1
  with (directory / f"stdout.{job['attempts']}.log").open('a') as output:
    proc = subprocess.Popen(cmd, cwd=directory, env=env, stdout=output,
        stderr=subprocess.STDOUT, start_new_session=True)
  job['pid'] = proc.pid
  job['create_time'] = psutil.Process(proc.pid).create_time()
  job['launched_at'] = time.time()
  print(f"LAUNCHED {job['setting']} seed={job['seed']} gpu={job['gpu']} pid={proc.pid}", flush=True)
  return True


def valid_rows(job):
  path = Path(job['logdir']) / 'metrics.jsonl'
  rows = []
  if not path.exists():
    return rows
  seen = set()
  for line in path.read_text().splitlines():
    try:
      row = json.loads(line)
    except json.JSONDecodeError:
      continue
    step = row.get('step', 0)
    if step in seen or step <= 0 or step % log_interval(job) != 0:
      continue
    required = ('loss/policy', 'loss/value', 'loss/repval', 'loss/reward',
                'loss/con', 'loss/dyn', 'loss/rep', 'loss/state', 'fps/train',
                'schedule/wm_multiplier', 'schedule/agent_multiplier', 'env_step')
    if not all(k in row and isinstance(row[k], (float, int)) and math.isfinite(row[k]) for k in required):
      continue
    if not all(any(k.startswith(prefix) and isinstance(v, (float, int)) and math.isfinite(v)
                   for k, v in row.items())
               for prefix in ('act_redo/', 'grad_redo/', 'data_diversity/')):
      continue
    if row['env_step'] != step or row['fps/train'] <= 0:
      continue
    rows.append(row)
    seen.add(step)
  return rows


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--plan', action='store_true')
  parser.add_argument('--rebalance-pending', action='store_true',
      help='Reassign only runs that have never started, using measured update costs')
  parser.add_argument('--only', help='setting:seed for a pilot; repeat invocation to launch others')
  parser.add_argument('--monitor-only', action='store_true')
  parser.add_argument('--retry-failed', action='store_true')
  parser.add_argument('--min-logs', type=int, default=5)
  parser.add_argument('--max-attempts', type=int, default=3)
  args = parser.parse_args()
  STATE.parent.mkdir(parents=True, exist_ok=True)
  lock = STATE.with_suffix('.lock').open('w')
  fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
  state = json.loads(STATE.read_text()) if STATE.exists() else plan()
  if args.rebalance_pending:
    state = plan(state)
  save(state)
  if args.plan:
    for job in state['jobs']:
      print(job['setting'], job['seed'], 'gpu', job['gpu'])
    return
  selected = [j for j in state['jobs'] if not j.get('user_paused') and
      (not args.only or args.only == f"{j['setting']}:{j['seed']}")]
  # Exercise the original joint-update path early, then prioritize heavy jobs.
  selected.sort(key=lambda j: (j['setting'], j['seed']) != ('default', 1000))
  while True:
    for job in selected:
      rows = valid_rows(job)
      job['valid_logs'] = len(rows)
      job['last_step'] = rows[-1]['step'] if rows else None
      job['alive'] = alive(job)
    if not args.monitor_only:
      for job in selected:
        if job['attempts'] >= args.max_attempts and not job['alive']:
          print('Launch attempts exhausted; inspect the owned run.', flush=True)
          return 1
        if job['alive'] or (job['attempts'] and not args.retry_failed):
          continue
        if job.get('last_step', 0) and job['last_step'] >= 15000000:
          continue
        # Bound startup pressure while allowing shared compilation-cache reuse.
        starting = [j for j in state['jobs'] if alive(j) and not (Path(j['logdir']) / 'training_ready.json').exists()]
        if len(starting) >= 3:
          break
        if launch(job):
          save(state)
          break
    save(state)
    completed = sum(j['valid_logs'] >= args.min_logs and j['alive'] for j in selected)
    print(time.strftime('%H:%M:%S'), f'healthy={completed}/{len(selected)}',
          ' '.join(f"{j['setting']}:{j['seed']}={j['valid_logs']}@{j.get('last_step')}"
                   + ('' if alive(j) else '(down)') for j in selected), flush=True)
    failed = [j for j in selected if j['attempts'] and not alive(j)]
    if failed and not args.retry_failed:
      print('Owned process exited; inspect its stdout before explicit retry.', flush=True)
      return 1
    if completed == len(selected):
      state['monitor_completed_at'] = time.time()
      save(state)
      print(f'All selected runs have {args.min_logs} valid logs and live processes.', flush=True)
      return 0
    time.sleep(30)


if __name__ == '__main__':
  sys.exit(main())
