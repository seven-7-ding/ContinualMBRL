"""Apply the user's k128 retirement and checkpointed k32 logging change once.

Only manifest-owned processes with matching UID, creation time, main path and
logdir can be signaled. Re-running resumes remote/local deletion from the audit.
"""
import fcntl
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from auto_scripts.launch_two_player import CONTROL, STATE, PROJECT, save, alive
from tools.inspect_wandb_reference import credentials


def write_audit(audit):
  path = CONTROL / 'k128_retirement.json'
  temp = path.with_suffix('.tmp')
  temp.write_text(json.dumps(audit, indent=2) + '\n')
  temp.replace(path)


def main():
  lock = STATE.with_suffix('.lock').open('a')
  fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
  state = json.loads(STATE.read_text())
  audit_path = CONTROL / 'k128_retirement.json'
  if audit_path.exists():
    audit = json.loads(audit_path.read_text())
  else:
    audit = {'started_at': time.time(), 'before': state, 'stopped': [],
             'deleted_remote': [], 'deleted_local': []}
    write_audit(audit)
  retired = [j for j in audit['before']['jobs'] if j['setting'].endswith('_128')]
  restarting = [j for j in audit['before']['jobs'] if j['setting'].endswith('_32')]
  assert len(retired) == 9 and len(restarting) == 9
  protected = json.loads((CONTROL / 'protected_processes_current.json').read_text())
  protected_pids = {p['pid'] for p in protected['current_unrelated_gpu_processes']}
  stopped = {p['pid'] for p in audit['stopped']}
  targets = []
  # Validate the entire set before signaling any process.
  for job in retired + restarting:
    if job['pid'] in stopped:
      continue
    assert alive(job), (job['setting'], job['seed'], 'unexpected missing PID')
    proc = psutil.Process(job['pid'])
    argv = proc.cmdline()
    assert proc.uids().real == os.getuid()
    assert str(ROOT / 'dreamerv3/main.py') in argv
    assert argv[argv.index('--logdir') + 1] == job['logdir']
    assert argv[argv.index('--run.two_player') + 1] == job['setting']
    directory = Path(job['logdir'])
    assert directory == ROOT / 'logdir' / PROJECT / job['setting'] / f"seed_{job['seed']}"
    if job in restarting:
      latest = directory / 'ckpt/latest'
      snapshot = latest.parent / latest.read_text().strip()
      assert (snapshot / 'done').exists()
      assert int(pickle.loads((snapshot / 'step.pkl').read_bytes())) > 0
    family = [proc, *proc.children(recursive=True)]
    assert not protected_pids.intersection(p.pid for p in family)
    assert all(p.uids().real == os.getuid() for p in family)
    targets.append((job, family))
  family = {p.pid: p for _, members in targets for p in members}
  for proc in family.values():
    try:
      proc.terminate()
    except psutil.NoSuchProcess:
      pass
  _, remaining = psutil.wait_procs(list(family.values()), timeout=8)
  for proc in remaining:
    if proc.status() != psutil.STATUS_ZOMBIE:
      proc.kill()
  psutil.wait_procs(remaining, timeout=5)
  for job, members in targets:
    assert not alive(job)
    audit['stopped'].append({'pid': job['pid'], 'create_time': job['create_time'],
        'setting': job['setting'], 'seed': job['seed'],
        'family_pids': [p.pid for p in members]})
    print('STOPPED', job['setting'], job['seed'], job['pid'], flush=True)
  write_audit(audit)
  state['jobs'] = [j for j in state['jobs'] if not j['setting'].endswith('_128')]
  for job in state['jobs']:
    if job['setting'].endswith('_32'):
      job['log_every'] = 2000
  state['retired_k128_at'] = time.time()
  state.pop('monitor_completed_at', None)
  save(state)
  # Deleting these exact W&B IDs is explicitly requested by the user.
  import wandb
  env = credentials()
  os.environ['WANDB_API_KEY'] = env['WANDB_API_KEY']
  api = wandb.Api(timeout=30)
  remote = {run.id: run for run in api.runs(f"{env['WANDB_ENTITY']}/{PROJECT}")}
  for job in retired:
    run_id = job['wandb_id']
    assert run_id == f"two-player-{job['setting']}-{job['seed']}-v2-step10k"
    if run_id not in audit['deleted_remote']:
      if run_id in remote:
        remote[run_id].delete(delete_artifacts=True)
      audit['deleted_remote'].append(run_id)
      write_audit(audit)
      print('DELETED W&B', run_id, flush=True)
    directory = Path(job['logdir'])
    if str(directory) not in audit['deleted_local']:
      if directory.exists():
        shutil.rmtree(directory)
      audit['deleted_local'].append(str(directory))
      write_audit(audit)
      print('DELETED logdir', job['setting'], job['seed'], flush=True)
  for setting in {j['setting'] for j in retired}:
    directory = ROOT / 'logdir' / PROJECT / setting
    if directory.exists():
      directory.rmdir()
  audit['completed_at'] = time.time()
  write_audit(audit)
  print('Retirement complete. Resume the nine k32 jobs using --retry-failed.', flush=True)


if __name__ == '__main__':
  main()
