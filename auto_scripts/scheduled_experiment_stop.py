#!/usr/bin/env python3
"""Independent timed stop, recovery fence, and verified experiment backup."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from auto_scripts import experiment_watchdog as w
from auto_scripts.rollout_bounded_replay import remote_pause

PLAN = w.CONTROL/'scheduled_stop_plan.json'
STATUS = w.CONTROL/'scheduled_stop_status.json'
PROTOCOL = 'checkpoint_stop_v1'


def write_status(**values):
  previous = w.read_json(STATUS)
  previous.update(values,time=w.stamp(),timestamp=time.time(),pid=os.getpid(),
                  create_time=psutil.Process().create_time())
  w.write_json(STATUS,previous)
  return previous


def scoped_jobs(plan,state):
  jobs = state['jobs']
  assert len(plan['run_ids']) == len(set(plan['run_ids'])) == 12
  assert set(plan['run_ids']) == {j['wandb_id'] for j in jobs}
  assert all(j['setting'] in w.launcher.SETTINGS for j in jobs)
  assert all(Path(j['logdir']).resolve() ==
      ROOT/'logdir'/w.launcher.PROJECT/j['setting']/f"seed_{j['seed']}" for j in jobs)
  return jobs


def pause_manifest(plan):
  with w.launcher.STATE.with_suffix('.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    state = w.read_json(w.launcher.STATE)
    for job in scoped_jobs(plan,state):
      if not job.get('user_paused'):
        job.update(user_paused=True,pause_reason='scheduled_user_stop_'+plan['plan_id'])
      job['scheduled_stop_plan_id'] = plan['plan_id']
    w.launcher.save(state)
  return state


def checkpoint_proof(job):
  for attempt in range(3):
    cp = w.checkpoint(job['logdir'],inspect_agent=True)
    assert cp.get('complete') and cp.get('losses_finite'),(job['wandb_id'],cp)
    try:
      source = Path(job['logdir'])/'ckpt'/cp['name']
      saved = pickle.loads((source/'agent.pkl').read_bytes())
      assert all(np.isfinite(np.asarray(v)).all() for v in saved['params'].values())
      cp['finite_arrays'] = len(saved['params'])
      anchor = job.get('optimization_rollout',{}).get('before',{}).get('checkpoint')
      if anchor and job['setting'] != 'default':
        for mode,multiplier in zip(('wm','agent'),w.launcher.update_counts(job['setting'])):
          key = f'opt_{mode}/step/value'
          assert cp['updates'][key] == anchor['updates'][key]+(
              cp['step']-anchor['step'])*multiplier//4
        cp['optimizer_counts_exact'] = True
      return cp,source
    except FileNotFoundError:
      if attempt==2:
        raise


def snapshot_checkpoints(plan):
  target = Path(plan['backup_dir'])/'pre_stop_checkpoints'
  rows = []
  for job in scoped_jobs(plan,w.read_json(w.launcher.STATE)):
    cp,source = checkpoint_proof(job)
    dest = target/job['wandb_id']/cp['name']
    if not dest.exists():
      dest.parent.mkdir(parents=True,exist_ok=True)
      temporary = dest.with_name(dest.name+'.copying')
      shutil.rmtree(temporary,ignore_errors=True)
      shutil.copytree(source,temporary)
      assert (temporary/'done').exists()
      assert pickle.loads((temporary/'step.pkl').read_bytes()) == cp['step']
      temporary.replace(dest)
    rows.append(dict(id=job['wandb_id'],checkpoint=cp,backup=str(dest)))
  w.write_json(Path(plan['backup_dir'])/'pre_stop_checkpoint_proofs.json',
               dict(time=w.stamp(),runs=rows))
  return rows


def protocol_ready(job):
  p = w.owned(job)
  ready = w.read_json(Path(job['logdir'])/'scheduled_stop_ready.json')
  return bool(p and ready.get('pid')==p.pid and
      ready.get('protocol')==PROTOCOL and ready.get('run_id')==job['wandb_id'] and
      ready.get('plan_path')==str(PLAN) and
      p.environ().get('CONTINUALMBRL_STOP_PROTOCOL')==PROTOCOL)


def freeze_and_stop(plan):
  """Fence all owned parents before any slow backup/network operation."""
  state = pause_manifest(plan)
  jobs = scoped_jobs(plan,state)
  frozen = []
  for job in jobs:
    proc = w.owned(job)
    if not proc:
      continue
    family = [proc,*proc.children(recursive=True)]
    assert all(p.uids().real==os.getuid() for p in family)
    assert w.owned(job)
    proc.suspend()
    for child in family[1:]:
      with contextlib.suppress(psutil.NoSuchProcess):
        child.suspend()
    frozen.append((job,family))
  write_status(phase='training_fenced',fenced_at=time.time(),
               frozen_ids=[j['wandb_id'] for j,_ in frozen])
  # Capture immutable final evidence before removing any processes.
  proofs = []
  for job in jobs:
    fallback = False
    try:
      cp,source = checkpoint_proof(job)
    except Exception as exc:
      baseline = w.read_json(Path(plan['backup_dir'])/'pre_stop_checkpoint_proofs.json')
      previous = next(r for r in baseline['runs'] if r['id']==job['wandb_id'])
      cp,source = previous['checkpoint'],Path(previous['backup'])
      assert (source/'done').exists()
      saved = pickle.loads((source/'agent.pkl').read_bytes())
      assert all(np.isfinite(np.asarray(v)).all() for v in saved['params'].values())
      fallback = True
      dest = Path(plan['backup_dir'])/'fallback_checkpoints'/job['wandb_id']/cp['name']
      if not dest.exists():
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(source,dest)
      w.event('scheduled_stop_checkpoint_fallback',dict(id=job['wandb_id'],
          checkpoint_step=cp['step'],error_type=type(exc).__name__),cooldown=0)
    ack = w.read_json(Path(job['logdir'])/'scheduled_stop_saved.json')
    local_step = w.latest_metrics(job['logdir']).get('step',0)
    scheduled = job['setting']!='default'
    acknowledged = bool(not fallback and ack.get('pid')==job.get('pid') and
        ack.get('plan_id')==plan['plan_id'] and ack.get('step')==cp['step'] and
        ack.get('checkpoint')==cp['name'] and cp['step']>=local_step)
    proofs.append(dict(id=job['wandb_id'],checkpoint=cp,
        fallback_checkpoint=fallback,
        cooperative_save=acknowledged,previously_paused=not scheduled,
        last_local_step=local_step,ack=ack))
  w.write_json(Path(plan['backup_dir'])/'final_checkpoint_proofs.json',
               dict(time=w.stamp(),runs=proofs))
  for job,family in frozen:
    for proc in family:
      with contextlib.suppress(psutil.Error):
        proc.terminate()
        proc.resume()
    _,live = psutil.wait_procs(family,timeout=5)
    for proc in live:
      with contextlib.suppress(psutil.Error):
        if proc.status()!=psutil.STATUS_ZOMBIE:
          proc.kill()
    psutil.wait_procs(live,timeout=2)
  for job in jobs:
    w.cleanup_orphans(job)
    assert not w.owned(job)
  write_status(phase='all_training_stopped',stopped_at=time.time())
  return proofs


def verified_copy(source,dest,heartbeat=None):
  """Independent copy; bounded 8MiB buffer, no hard links or mutable symlinks."""
  dest.parent.mkdir(parents=True,exist_ok=True)
  hasher = hashlib.sha256()
  beat_at = time.monotonic()+5
  with source.open('rb') as src,dest.open('wb') as dst:
    while block:=src.read(8*1024**2):
      hasher.update(block)
      dst.write(block)
      if heartbeat and time.monotonic()>=beat_at:
        heartbeat()
        beat_at=time.monotonic()+5
    dst.flush()
    os.fsync(dst.fileno())
  check = hashlib.sha256()
  with dest.open('rb') as stream:
    while block:=stream.read(8*1024**2):
      check.update(block)
      if heartbeat and time.monotonic()>=beat_at:
        heartbeat()
        beat_at=time.monotonic()+5
  assert check.digest()==hasher.digest()
  shutil.copystat(source,dest)
  return dict(path=str(dest),bytes=dest.stat().st_size,sha256=check.hexdigest())


def backup_artifacts(plan):
  target = Path(plan['backup_dir'])
  catalog = []
  for index,job in enumerate(scoped_jobs(plan,w.read_json(w.launcher.STATE))):
    source = Path(job['logdir'])
    dest = target/'runs'/job['setting']/f"seed_{job['seed']}"
    for directory,subdirs,files in os.walk(source,followlinks=False):
      subdirs[:] = [x for x in subdirs if x not in ('cache','tmp','config') and
                    not (Path(directory)/x).is_symlink()]
      for name in files:
        path = Path(directory)/name
        if path.is_symlink() or not path.is_file():
          continue
        catalog.append(verified_copy(path,dest/path.relative_to(source),
            heartbeat=lambda:write_status(phase='backing_up',backup_file=str(path))))
    write_status(phase='backing_up',backed_up_runs=index+1,
        backup_files=len(catalog),backup_bytes=sum(x['bytes'] for x in catalog))
  # Include monitoring evidence and exact source changes, without credentials/caches.
  for source in (w.launcher.STATE,PLAN,ROOT/'AGENT.md',ROOT/'task_checklist.md',
                 ROOT/'codex-cli-executor.log'):
    catalog.append(verified_copy(source,target/'metadata'/source.name))
  for name in ('events.jsonl','resources.jsonl','foreground.json',
               'foreground_verification_latest.json','wandb.json',
               'scheduled_stop_status.json','scheduled_stop.log',
               'wandb_bridge_status.json','daemon.json'):
    source=w.CONTROL/name
    if source.is_file():
      catalog.append(verified_copy(source,target/'metadata/watchdog'/name))
  changed = subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines()
  added = subprocess.check_output(['git','ls-files','--others','--exclude-standard'],
                                  cwd=ROOT,text=True).splitlines()
  for name in sorted(set(changed+added)):
    source = ROOT/name
    if source.is_file() and source.suffix in ('.py','.yaml','.md'):
      catalog.append(verified_copy(source,target/'source'/name))
  with (target/'metadata/code.patch').open('wb') as stream:
    subprocess.run(['git','diff','--binary'],cwd=ROOT,stdout=stream,check=True)
  head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
  (target/'RESTORE.md').write_text(
      '# Experiment stop backup\n\n'
      'Original local paths and W&B run IDs are in metadata/manifest.json.\n'
      'runs/<setting>/seed_<seed> contains independent checkpoint, replay, '
      'logs, configuration and W&B local records. Final checkpoint and '
      'optimizer proofs are in final_checkpoint_proofs.json.\n'
      'Verify SHA256 hashes in backup_catalog.json before restoring. '
      'Source files and the Git baseline/patch are included. Cache/tmp/private '
      'credential configuration directories and symlinks are excluded.\n'
      'Resume only after an explicit user request: restore the same local '
      'directory, retain its W&B ID, clear the intentional pause and timed '
      'stop plan, then resume with WANDB_RESUME=must in the dreamer environment. '
      'Credentials must remain private and are not copied here.\n')
  recorded = {r['path'] for r in catalog}
  for path in sorted(target.rglob('*')):
    if not path.is_file() or path.is_symlink() or str(path) in recorded or path.name=='backup_catalog.json':
      continue
    digest = hashlib.sha256()
    with path.open('rb') as stream:
      while block:=stream.read(8*1024**2):
        digest.update(block)
    catalog.append(dict(path=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest()))
  report = dict(time=w.stamp(),independent_copies=True,hashes_verified=True,
      excluded_transient_directories=['cache','tmp','config'],
      git_head=head,files=catalog,total_bytes=sum(x['bytes'] for x in catalog))
  w.write_json(target/'backup_catalog.json',report)
  return report


def finalize(plan):
  target = Path(plan['backup_dir'])
  if not w.read_json(STATUS).get('stopped_at'):
    freeze_and_stop(plan)
  if not (target/'backup_catalog.json').exists():
    backup_artifacts(plan)
  jobs = scoped_jobs(plan,w.read_json(w.launcher.STATE))
  remote = {r['id']:r for r in w.remote_audit(jobs)['runs']}
  for job in jobs:
    r = remote[job['wandb_id']]
    assert not w.owned(job)
    if not r.get('stop_requested') and r.get('state') not in ('finished','failed'):
      remote_pause(job)
  verified = w.remote_audit(jobs)
  assert all(r.get('caught_up') and r.get('losses_finite') and
      (r.get('stop_requested') or r.get('state') in ('finished','failed'))
      for r in verified['runs'])
  w.write_json(target/'final_wandb_status.json',verified)
  path=target/'final_wandb_status.json'
  catalog=w.read_json(target/'backup_catalog.json')
  catalog['files']=[r for r in catalog['files'] if r['path']!=str(path)]
  catalog['files'].append(dict(path=str(path),bytes=path.stat().st_size,
      sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
  catalog['total_bytes']=sum(r['bytes'] for r in catalog['files'])
  w.write_json(target/'backup_catalog.json',catalog)
  proofs = w.read_json(target/'final_checkpoint_proofs.json')['runs']
  warnings = [r['id'] for r in proofs if not r['previously_paused'] and
              not r['cooperative_save']]
  write_status(phase='complete' if not warnings else 'stopped_with_save_warning',
      backup_verified=True,completed_at=time.time(),save_warnings=warnings)
  w.event('scheduled_stop_complete',dict(backup_dir=str(target),save_warnings=warnings),
          cooldown=0)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--check',action='store_true')
  args = parser.parse_args()
  plan = w.read_json(PLAN)
  jobs = scoped_jobs(plan,w.read_json(w.launcher.STATE))
  if args.check:
    print(json.dumps(dict(plan=plan,protocol_ready=[j['wandb_id'] for j in jobs
        if not j.get('user_paused') and protocol_ready(j)])))
    return
  Path(plan['backup_dir']).mkdir(parents=True,exist_ok=True,mode=0o700)
  with (w.CONTROL/'scheduled_stop.lock').open('a') as lock:
    try:
      fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
      return  # A calendar fallback may start while the primary helper is alive.
    while True:
      plan = w.read_json(PLAN)
      if not plan.get('enabled'):
        write_status(phase='canceled')
        return
      try:
        now = time.time()
        if now >= plan['quiesce_at']:
          pause_manifest(plan)
        if now >= plan['deadline']:
          finalize(plan)
          return
        previous = w.read_json(STATUS)
        if not previous.get('baseline_saved_at') or (
            now >= plan['deadline']-1800 and not previous.get('late_baseline_saved_at')):
          snapshot_checkpoints(plan)
          write_status(baseline_saved_at=now,
              **({'late_baseline_saved_at':now} if now>=plan['deadline']-1800 else {}))
        jobs = scoped_jobs(plan,w.read_json(w.launcher.STATE))
        missing = [j['wandb_id'] for j in jobs if j['setting']!='default' and
                   not protocol_ready(j) and w.owned(j)]
        if now >= plan['deadline']-3600 and missing:
          w.event('scheduled_stop_protocol_missing',dict(ids=missing),cooldown=300)
        write_status(phase='quiescing' if now>=plan['quiesce_at'] else 'waiting',
            deadline=plan['deadline'],quiesce_at=plan['quiesce_at'],
            protocol_missing=missing,error_type=None)
      except Exception as exc:
        write_status(error_type=type(exc).__name__)
        w.event('scheduled_stop_error',dict(error_type=type(exc).__name__),cooldown=60)
      time.sleep(min(15,max(.1,plan['deadline']-time.time())))


if __name__=='__main__':
  main()
