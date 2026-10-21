#!/usr/bin/env python3
"""Serial, same-ID checkpoint-preserving rollout of cooperative stopping."""
import fcntl
import os
from pathlib import Path
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from auto_scripts import experiment_watchdog as w
from auto_scripts import rollout_bounded_replay as rollout
from auto_scripts import scheduled_experiment_stop as stop

DIRECTORY = w.CONTROL/'scheduled_stop_rollout'


def compiled_ready(job):
  ready = w.read_json(Path(job['logdir'])/'training_ready.json')
  before = job['stop_protocol_rollout']['before']['checkpoint']['step']
  return stop.protocol_ready(job) and type(ready) is int and ready>before


def verify_continuation(job,cloud):
  state = job['stop_protocol_rollout']
  before = state['before']['checkpoint']
  proc = w.owned(job)
  if not proc or not stop.protocol_ready(job):
    return False
  cp,_ = stop.checkpoint_proof(job)
  if cp['step'] <= before['step']:
    return False
  for mode,multiplier in zip(('wm','agent'),w.launcher.update_counts(job['setting'])):
    key = f'opt_{mode}/step/value'
    assert cp['updates'][key] == before['updates'][key]+(
        cp['step']-before['step'])*multiplier//4
  metrics = w.latest_metrics(job['logdir'])
  if (metrics.get('step',0)<=before['step'] or not metrics.get('losses_finite') or
      not cloud.get('caught_up') or not cloud.get('losses_finite')):
    return False
  env = proc.environ()
  assert env['WANDB_RUN_ID']==job['wandb_id'] and env['WANDB_RESUME']=='must'
  assert env['CONTINUALMBRL_REPLAY_PROFILE']=='bounded_replay_v2'
  assert env['CUDA_VISIBLE_DEVICES']==str(job['gpu'])
  assert env['CONTINUALMBRL_STOP_PLAN']==str(stop.PLAN)
  state.update(state='verified',verified_at=w.stamp(),checkpoint=cp,
      local_step=metrics['step'],remote_step=cloud['step'],pid=proc.pid,gpu=job['gpu'])
  w.write_json(DIRECTORY/(job['wandb_id']+'_verified.json'),state)
  w.event('scheduled_stop_continuation_verified',dict(id=job['wandb_id'],
      pid=proc.pid,checkpoint_step=cp['step']),cooldown=0)
  return True


def tick():
  with w.launcher.STATE.with_suffix('.lock').open('a') as lock:
    try:
      fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
      return False
    state = w.read_json(w.launcher.STATE)
    plan = w.read_json(stop.PLAN)
    jobs = stop.scoped_jobs(plan,state)
    active = [j for j in jobs if not j.get('user_paused')]
    assert len(active)==9
    assert time.time()<plan['quiesce_at']
    remote = {r['id']:r for r in w.remote_audit(jobs)['runs']}
    for job in active:
      status = job.get('stop_protocol_rollout',{})
      if status.get('state')=='stopped' and w.owned(job):
        status['state']='started'
      if status.get('state')=='started':
        verify_continuation(job,remote[job['wandb_id']])
    w.launcher.save(state)
    states = [j.get('stop_protocol_rollout',{}) for j in active]
    if all(s.get('state')=='verified' for s in states):
      return True
    # Only one new compilation at a time. Readiness is emitted by loaded code
    # after the original checkpoint and task phase have been restored.
    if any(j.get('stop_protocol_rollout',{}).get('state') in ('started','stopped') and
           not compiled_ready(j) for j in active):
      return False
    canary = (next(j for j in active if j['wandb_id']==plan['rollout_canary_id'])
        if plan.get('rollout_canary_id') else
        next(j for j in active if j['setting']=='wm_as_leader_32' and j['seed']==3000))
    canary_status = canary.get('stop_protocol_rollout',{}).get('state')
    if canary_status and canary_status!='verified':
      return False
    pending = [j for j in active if not j.get('stop_protocol_rollout')]
    if canary_status is None:
      pending = [canary]
    pending.sort(key=lambda j:(j is not canary,j['setting'],j['seed']))
    for job in pending:
      if not w.owned(job):
        continue  # The ordinary watchdog owns unexpected-process recovery.
      if w.launcher.available_ram_gib()<w.RESUME:
        return False
      gpu = next(g for g in w.gpu_stats() if g['index']==job['gpu'])
      if gpu['free_mib']<6656 or gpu['temp_c']>=85:
        continue
      before = rollout.stop_from_checkpoint(job,remote[job['wandb_id']])
      if not before:
        continue
      job['stop_protocol_rollout'] = dict(state='stopped',before=before)
      w.launcher.save(state)
      if not w.launcher.launch(job,resume='must'):
        raise RuntimeError('Stopped job awaiting ordinary watchdog recovery')
      job['stop_protocol_rollout'].update(state='started',started_at=w.stamp(),
                                         pid=job['pid'])
      w.launcher.save(state)
      w.event('scheduled_stop_protocol_started',dict(id=job['wandb_id'],pid=job['pid'],
          gpu=job['gpu'],resumed_step=before['checkpoint']['step']),cooldown=0)
      break
  return False


def main():
  DIRECTORY.mkdir(parents=True,exist_ok=True)
  with (DIRECTORY/'controller.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    w.write_json(DIRECTORY/'status.json',dict(time=w.stamp(),timestamp=time.time(),
        pid=os.getpid(),create_time=psutil.Process().create_time(),complete=False,runs=[]))
    while True:
      plan = w.read_json(stop.PLAN)
      if time.time()>=plan['quiesce_at']:
        w.write_json(DIRECTORY/'status.json',dict(time=w.stamp(),timestamp=time.time(),
            pid=os.getpid(),create_time=psutil.Process().create_time(),
            complete=False,halted_for_deadline=True))
        w.event('scheduled_stop_rollout_halted_for_deadline',{},cooldown=0)
        return
      try:
        complete = tick()
        jobs = w.read_json(w.launcher.STATE)['jobs']
        rows = [dict(id=j['wandb_id'],pid=j.get('pid'),
            rollout=j.get('stop_protocol_rollout',{})) for j in jobs if not j.get('user_paused')]
        w.write_json(DIRECTORY/'status.json',dict(time=w.stamp(),timestamp=time.time(),
            pid=os.getpid(),create_time=psutil.Process().create_time(),complete=complete,runs=rows))
        if complete:
          w.event('scheduled_stop_rollout_complete',dict(verified=9),cooldown=0)
          return
      except Exception as exc:
        w.event('scheduled_stop_rollout_error',dict(error_type=type(exc).__name__),cooldown=60)
      time.sleep(15)


if __name__=='__main__':
  main()
