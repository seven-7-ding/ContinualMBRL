#!/usr/bin/env python3
"""Resume this sweep serially with direct W&B API transport and original IDs."""
import fcntl
import os
from pathlib import Path
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from auto_scripts import experiment_watchdog as w
from auto_scripts import rollout_bounded_replay as recovery
from auto_scripts import scheduled_experiment_stop as stop
from tools import wandb_transport as transport

DIRECTORY = w.CONTROL/'wandb_transport_rollout'
FIELD = 'wandb_transport_rollout'
CANARY = 'two-player-best_response_32-1000-v2-step10k'


def compiled_ready(job):
  proc = w.owned(job)
  if not proc or not stop.protocol_ready(job):
    return False
  env = proc.environ()
  ready = w.read_json(Path(job['logdir'])/'training_ready.json')
  return (env.get('CONTINUALMBRL_WANDB_TRANSPORT') == transport.PROFILE and
      all(transport.API_HOST in env.get(k, '').split(',')
          for k in ('NO_PROXY', 'no_proxy')) and
      type(ready) is int and ready > job[FIELD]['before']['checkpoint']['step'])


def verify_continuation(job, cloud):
  state = job[FIELD]
  if not compiled_ready(job):
    return False
  before = state['before']['checkpoint']
  cp, _ = stop.checkpoint_proof(job)
  local = w.latest_metrics(job['logdir'])
  if (cp['step'] <= before['step'] or
      local.get('step', 0) <= before['step'] or
      not local.get('losses_finite') or not cloud.get('caught_up') or
      not cloud.get('losses_finite') or cloud.get('state') != 'running'):
    return False
  for mode, multiplier in zip(('wm', 'agent'),
                               w.launcher.update_counts(job['setting'])):
    key = f'opt_{mode}/step/value'
    assert cp['updates'][key] == before['updates'][key] + (
        cp['step']-before['step'])*multiplier//4
  proc = w.owned(job)
  env = proc.environ()
  assert env['WANDB_RUN_ID'] == job['wandb_id'] and env['WANDB_RESUME'] == 'must'
  assert env['CUDA_VISIBLE_DEVICES'] == str(job['gpu'])
  assert env['CONTINUALMBRL_REPLAY_PROFILE'] == 'bounded_replay_v2'
  state.update(state='verified', verified_at=w.stamp(), checkpoint=cp,
      local_step=local['step'], remote_step=cloud['step'], pid=proc.pid, gpu=job['gpu'])
  w.write_json(DIRECTORY/(job['wandb_id']+'_verified.json'), state)
  w.event('wandb_direct_continuation_verified', dict(id=job['wandb_id'],
      pid=proc.pid, checkpoint_step=cp['step']), cooldown=0)
  return True


def pending_candidates(jobs):
  canary = next(j for j in jobs if j['wandb_id'] == CANARY)
  status = canary.get(FIELD, {}).get('state')
  if status != 'verified':
    return [canary] if status is None else []
  return [j for j in jobs if not j.get(FIELD)]


def tick():
  with w.launcher.STATE.with_suffix('.lock').open('a') as lock:
    try:
      fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
      return False
    state = w.read_json(w.launcher.STATE)
    plan = w.read_json(stop.PLAN)
    jobs = [j for j in stop.scoped_jobs(plan, state) if not j.get('user_paused')]
    assert len(jobs) == 9 and time.time() < plan['quiesce_at']
    cloud = {r['id']: r for r in w.remote_audit(state['jobs'])['runs']}
    for job in jobs:
      status = job.get(FIELD, {})
      if status.get('state') == 'stopped' and w.owned(job):
        status['state'] = 'started'
      if status.get('state') == 'started':
        verify_continuation(job, cloud[job['wandb_id']])
    w.launcher.save(state)
    if all(j.get(FIELD, {}).get('state') == 'verified' for j in jobs):
      return True
    if any(j.get(FIELD, {}).get('state') in ('started', 'stopped') and
           not compiled_ready(j) for j in jobs):
      return False
    for job in pending_candidates(jobs):
      if not w.owned(job) or w.launcher.available_ram_gib() < w.RESUME:
        continue
      gpu = next(g for g in w.gpu_stats() if g['index'] == job['gpu'])
      if gpu['free_mib'] < 6656 or gpu['temp_c'] >= 85:
        continue
      # Includes CP >= all published local progress and native cloud caught up.
      before = recovery.stop_from_checkpoint(job, cloud[job['wandb_id']])
      if not before:
        continue
      job[FIELD] = dict(state='stopped', before=before)
      w.launcher.save(state)
      if not w.launcher.launch(job, resume='must'):
        raise RuntimeError('Stopped run awaiting ordinary watchdog recovery')
      job[FIELD].update(state='started', started_at=w.stamp(), pid=job['pid'])
      w.launcher.save(state)
      w.event('wandb_direct_protocol_started', dict(id=job['wandb_id'],
          pid=job['pid'], gpu=job['gpu'], resumed_step=before['checkpoint']['step']),
          cooldown=0)
      break
  return False


def main():
  DIRECTORY.mkdir(parents=True, exist_ok=True)
  with (DIRECTORY/'controller.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
      plan = w.read_json(stop.PLAN)
      halted = time.time() >= plan['quiesce_at']
      complete = False
      try:
        if not halted:
          complete = tick()
      except Exception as exc:
        w.event('wandb_direct_rollout_error', dict(error_type=type(exc).__name__),
                cooldown=60)
      jobs = w.read_json(w.launcher.STATE)['jobs']
      w.write_json(DIRECTORY/'status.json', dict(time=w.stamp(), timestamp=time.time(),
          pid=os.getpid(), create_time=psutil.Process().create_time(), complete=complete,
          halted_for_deadline=halted, canary_id=CANARY, runs=[
              dict(id=j['wandb_id'], pid=j.get('pid'), rollout=j.get(FIELD, {}))
              for j in jobs if j['setting'] != 'default']))
      if complete or halted:
        w.event('wandb_direct_rollout_complete' if complete else
                'wandb_direct_rollout_halted_for_deadline', dict(verified=sum(
                    j.get(FIELD, {}).get('state') == 'verified' for j in jobs)), cooldown=0)
        return
      time.sleep(15)


if __name__ == '__main__':
  main()
