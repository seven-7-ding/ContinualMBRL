#!/usr/bin/env python3
"""Rolling checkpoint-preserving optimization of the existing active manifest."""
import argparse
import base64
import contextlib
import fcntl
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time
import urllib.request

import psutil
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from auto_scripts import experiment_watchdog as w

DIRECTORY = w.CONTROL/'bounded_replay_rollout'
CANARIES = [('best_response_32', 1000), ('agent_as_leader_32', 1000)]


def remote_pause(job):
  """Request W&B's supported stop action for an existing run, preserving ID."""
  env = w.credentials()
  def request(query, variables):
    auth = base64.b64encode(('api:'+env['WANDB_API_KEY']).encode()).decode()
    req = urllib.request.Request(
        env.get('WANDB_BASE_URL', 'https://api.wandb.ai').rstrip('/')+'/graphql',
        data=json.dumps(dict(query=query, variables=variables)).encode(),
        headers={'Authorization':'Basic '+auth, 'Content-Type':'application/json'})
    with w.wandb_urlopen(req, timeout=25) as response:
      result = json.load(response)
    if result.get('errors'):
      raise RuntimeError('W&B pause query failed')
    return result['data']
  result = request('''query($entity:String!,$project:String!,$name:String!){
    project(name:$project,entityName:$entity){run(name:$name){id name state}}}''',
    dict(entity=env['WANDB_ENTITY'], project=w.launcher.PROJECT, name=job['wandb_id']))
  run = result['project']['run']
  assert run and run['name'] == job['wandb_id']
  result = request('''mutation($id:ID!){stopRun(input:{id:$id}){success}}''',
                   dict(id=run['id']))
  assert result['stopRun']['success']
  return 'stop_requested'


def stop_from_checkpoint(job, remote):
  """Freeze briefly, then stop only if no published progress would roll back."""
  proc = w.owned(job)
  if not proc:
    return None
  cp = w.checkpoint(job['logdir'], inspect_agent=True)
  local = w.latest_metrics(job['logdir'])
  if not (cp.get('complete') and cp.get('losses_finite') and
          cp['step'] >= local.get('step', 0) and remote.get('caught_up')):
    return None
  proc.suspend()
  stopped = False
  try:
    cp = w.checkpoint(job['logdir'], inspect_agent=True)
    local = w.latest_metrics(job['logdir'])
    if not (cp.get('complete') and cp.get('losses_finite') and
            cp['step'] >= local.get('step', 0) and
            remote.get('step', -1) >= local.get('step', 0)):
      return None
    family = [proc, *proc.children(recursive=True)]
    assert all(p.uids().real == os.getuid() for p in family)
    assert w.owned(job) is not None
    backup = DIRECTORY/'checkpoints'/job['wandb_id']/cp['name']
    if not backup.exists():
      backup.parent.mkdir(parents=True, exist_ok=True)
      shutil.copytree(Path(job['logdir'])/'ckpt'/cp['name'], backup)
    with (backup/'agent.pkl').open('rb') as stream:
      saved = pickle.load(stream)
    assert all(np.isfinite(value).all() for value in saved['params'].values()), (
        job['wandb_id'], 'nonfinite_pre_resume_model_or_optimizer')
    evidence = dict(time=w.stamp(), pid=proc.pid, create_time=proc.create_time(),
        checkpoint=cp, local_step=local.get('step'), remote_step=remote.get('step'),
        children=[dict(pid=p.pid, create_time=p.create_time()) for p in family[1:]],
        logdir=job['logdir'], wandb_id=job['wandb_id'])
    w.write_json(DIRECTORY/f"stop_{job['wandb_id']}_{proc.pid}.json", evidence)
    for member in family:
      with contextlib.suppress(psutil.NoSuchProcess):
        member.terminate()
    proc.resume()  # Deliver SIGTERM to the frozen parent.
    stopped = True
    _, live = psutil.wait_procs(family, timeout=5)
    for member in live:
      if member.status() != psutil.STATUS_ZOMBIE:
        member.kill()
    psutil.wait_procs(live, timeout=2)
    assert not w.owned(job)
    return evidence
  finally:
    if not stopped:
      with contextlib.suppress(psutil.Error):
        proc.resume()


def optimized_process(job):
  proc = w.owned(job)
  if not proc:
    return False
  if (job.get('performance_profile')=='bounded_replay_v2' and
      proc.environ().get('CONTINUALMBRL_REPLAY_PROFILE')!='bounded_replay_v2'):
    return False
  cmd = proc.cmdline()
  return ('--replay.cache_bytes' in cmd and
          cmd[cmd.index('--replay.cache_bytes')+1] == str(2*1024**3) and
          cmd[cmd.index('--run.usage.malloc')+1] == 'False')


def check_new_checkpoint(job):
  before = job['optimization_rollout']['before']['checkpoint']
  cp = w.checkpoint(job['logdir'], inspect_agent=True)
  if (not cp.get('complete') or cp['name'] == before['name'] or
      cp['step'] <= before['step']):
    return None
  assert cp['losses_finite'], (job['wandb_id'], 'nonfinite_checkpoint_losses')
  delta = cp['step'] - before['step']
  for mode, multiplier in zip(('wm', 'agent'), w.launcher.update_counts(job['setting'])):
    key = f'opt_{mode}/step/value'
    expected = before['updates'][key] + delta * .25 * multiplier
    assert cp['updates'][key] == expected, (job['wandb_id'], key,
                                           cp['updates'][key], expected)
  path = Path(job['logdir'])/'ckpt'/cp['name']/'agent.pkl'
  with path.open('rb') as stream:
    saved = pickle.load(stream)
  assert all(np.isfinite(value).all() for value in saved['params'].values()), (
      job['wandb_id'], 'nonfinite_model_or_optimizer_parameter')
  return dict(**cp, all_parameters_finite=True)


def canary_ready(job):
  rollout = job.get('optimization_rollout', {})
  return (rollout.get('state')=='verified' and
      (job.get('performance_profile')!='bounded_replay_v2' or
       bool(rollout.get('checkpoint_verified'))))


def tick():
  remote = w.remote_audit(w.read_json(w.launcher.STATE)['jobs'])
  remotes = {r['id']:r for r in remote['runs']}
  with w.launcher.STATE.with_suffix('.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    state = w.read_json(w.launcher.STATE)
    jobs = state['jobs']
    for job in jobs:
      if not job.get('user_paused'):
        continue
      if w.owned(job):
        before = stop_from_checkpoint(job, remotes[job['wandb_id']])
        if before:
          job['user_pause_checkpoint'] = before['checkpoint']
          job['user_paused_at'] = w.stamp()
          w.launcher.save(state)
          w.event('default_stopped_by_user', dict(id=job['wandb_id'],
              step=before['checkpoint']['step']), cooldown=0)
      if not w.owned(job) and job.get('wandb_paused_state') != 'stop_requested':
        job['wandb_paused_state'] = remote_pause(job)
        w.launcher.save(state)

    targets = [j for j in jobs if j.get('performance_profile') in
               ('bounded_replay_v1','bounded_replay_v2')
               and not j.get('user_paused')]
    for job in targets:
      rollout = job.setdefault('optimization_rollout', {})
      if not optimized_process(job):
        continue
      ready = Path(job['logdir'])/'training_ready.json'
      if ready.exists() and rollout.get('state') not in ('ready', 'verified'):
        rollout.update(state='ready', ready_at=w.stamp(), pid=job['pid'],
                       first_step=int(ready.read_text()))
        w.event('optimization_ready', dict(id=job['wandb_id'],
            step=rollout['first_step'], pid=job['pid']), cooldown=0)
      local = w.latest_metrics(job['logdir'])
      before = rollout.get('before', {})
      cpstep = before.get('checkpoint', {}).get('step', float('inf'))
      if (rollout.get('state') == 'ready' and local.get('step', 0) > cpstep and
          local.get('losses_finite') and remotes[job['wandb_id']].get('caught_up') and
          remotes[job['wandb_id']].get('state') == 'running'):
        rollout.update(state='verified', verified_at=w.stamp(), metrics=local,
                       remote=remotes[job['wandb_id']])
        w.event('optimization_verified', dict(id=job['wandb_id'],
            step=local['step'], fps=local['fps']), cooldown=0)
      if rollout.get('before') and not rollout.get('checkpoint_verified'):
        checked = check_new_checkpoint(job)
        if checked:
          rollout['checkpoint_verified'] = checked
          w.event('optimization_checkpoint_verified', dict(id=job['wandb_id'],
              step=checked['step'], updates=checked['updates']), cooldown=0)
    w.launcher.save(state)
    compiling = any(w.owned(j) and not (Path(j['logdir'])/'training_ready.json').exists()
                    for j in jobs)
    if not compiling and w.launcher.available_ram_gib() >= 150:
      canary_keys = ([('wm_as_leader_32',3000),('best_response_32',1000)]
          if all(j.get('performance_profile')=='bounded_replay_v2' for j in targets)
          else CANARIES)
      canaries = [next(j for j in targets if (j['setting'], j['seed']) == key)
                  for key in canary_keys]
      unfinished = [j for j in canaries if not canary_ready(j)]
      candidates = unfinished[:1] if unfinished else targets
      for job in candidates:
        if optimized_process(job):
          continue
        if not w.safe_gpu(w.gpu_stats(), job['gpu']):
          continue
        rollout = job.setdefault('optimization_rollout', {})
        rollout['state'] = 'preparing'
        w.launcher.save(state)
        before = stop_from_checkpoint(job, remotes[job['wandb_id']])
        if not before:
          continue
        rollout.update(state='stopped', before=before)
        w.launcher.save(state)
        if w.launcher.launch(job, resume='must'):
          rollout.update(state='starting', pid=job['pid'], started_at=w.stamp())
          w.launcher.save(state)
          w.event('optimization_resumed', dict(id=job['wandb_id'],
              pid=job['pid'], checkpoint_step=before['checkpoint']['step'],
              cache_bytes=2*1024**3, wandb_resume='must'), cooldown=0)
        break
    summary = dict(time=w.stamp(), defaults=[dict(seed=j['seed'],
        alive=bool(w.owned(j)), remote_state=j.get('wandb_paused_state')) for j in jobs
        if j['setting']=='default'], optimized=[dict(id=j['wandb_id'],
        **j.get('optimization_rollout', {})) for j in targets])
    w.write_json(DIRECTORY/'status.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='optimized'}), flush=True)
    return all(x.get('state')=='verified' and x.get('checkpoint_verified')
               for x in summary['optimized']) and all(
        not x['alive'] and x['remote_state']=='stop_requested' for x in summary['defaults'])


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--once', action='store_true')
  args = parser.parse_args()
  DIRECTORY.mkdir(parents=True, exist_ok=True)
  with (DIRECTORY/'controller.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
      try:
        complete = tick()
      except Exception as exc:
        w.event('optimization_controller_error', dict(error_type=type(exc).__name__), cooldown=60)
        raise
      if args.once or complete:
        if complete:
          w.event('optimization_rollout_complete',
                  '9 local/W&B continuations and new full optimizer checkpoints verified; 3 defaults paused.',
                  cooldown=0)
        return
      time.sleep(10)


if __name__ == '__main__':
  main()
