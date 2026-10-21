#!/usr/bin/env python3
"""Watch the existing manifest; recover only owned runs, preserving checkpoints/IDs.

daemon: 30s resource/process checks, 15min remote audit. wait: foreground event
wait that returns immediately on an alert or a scheduled 30m/1h/2h full check.
No credentials or process environments are persisted in watchdog evidence.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
import base64

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from auto_scripts import launch_two_player as launcher
from tools.inspect_wandb_reference import credentials
from tools.wandb_transport import urlopen as wandb_urlopen

CONTROL = launcher.CONTROL / 'watchdog'
FLOOR = 80.0  # GiB, more conservative than decimal GB.
WARN = 110.0
SHED = 100.0
RESUME = 125.0
TARGET = 15000000
LOSSES = ('policy', 'value', 'repval', 'reward', 'con', 'dyn', 'rep', 'state')


def stamp():
  return dt.datetime.now().astimezone().isoformat(timespec='seconds')


def write_json(path, value):
  path.parent.mkdir(parents=True, exist_ok=True)
  temp = None
  try:
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent,
        prefix=path.name+'.', suffix='.tmp', delete=False) as stream:
      temp = Path(stream.name)
      stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)
  finally:
    if temp is not None:
      temp.unlink(missing_ok=True)


def read_json(path, default=None):
  try:
    return json.loads(path.read_text())
  except (OSError, ValueError):
    return {} if default is None else default


def event(kind, detail, key=None, cooldown=600):
  """Durable alert wakes the foreground wait within two seconds."""
  key = key or kind
  with (CONTROL / 'events.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    notices = read_json(CONTROL / 'notices.json')
    if time.time() - notices.get(key, 0) < cooldown:
      return
    notices[key] = time.time()
    write_json(CONTROL / 'notices.json', notices)
    row = dict(time=stamp(), timestamp=time.time(), kind=kind, detail=detail)
    with (CONTROL / 'events.jsonl').open('a') as stream:
      stream.write(json.dumps(row, allow_nan=False) + '\n')
  print(json.dumps(row, allow_nan=False), flush=True)


def latest_metrics(directory):
  path = Path(directory) / 'metrics.jsonl'
  if not path.exists():
    return {}
  with path.open('rb') as stream:
    stream.seek(max(0, path.stat().st_size - 262144))
    lines = stream.read().splitlines()
  for line in reversed(lines):
    try:
      row = json.loads(line)
      if isinstance(row, dict) and 'step' in row:
        return dict(step=int(row['step']), age_s=round(time.time()-path.stat().st_mtime),
                    fps=row.get('fps/policy'),
                    losses_finite=all(isinstance(row.get('loss/'+k), (float, int))
                        and math.isfinite(row['loss/'+k]) for k in LOSSES))
    except (ValueError, TypeError):
      continue
  return {}


def checkpoint(directory, inspect_agent=False):
  """Read only complete snapshots; tolerate rotation by the live trainer."""
  root = Path(directory) / 'ckpt'
  for _ in range(3):
    try:
      name = (root / 'latest').read_text().strip()
      path = root / name
      if not name or Path(name).name != name or not (path / 'done').exists():
        continue
      step = pickle.loads((path / 'step.pkl').read_bytes())
      if not isinstance(step, int) or step < 0:
        return dict(complete=False, error='invalid_step')
      result = dict(complete=all((path / f).exists() for f in
          ('agent.pkl', 'step.pkl', 'replay.pkl', 'done')), name=name,
          step=step, age_s=round(time.time() - (path / 'done').stat().st_mtime))
      if inspect_agent:
        saved = pickle.loads((path / 'agent.pkl').read_bytes())
        metrics = {}
        for group in saved.get('train_metrics', {}).values():
          metrics.update(zip(group['keys'], group['values']))
        result['losses_finite'] = all('loss/'+k in metrics and
            math.isfinite(float(metrics['loss/'+k])) for k in LOSSES)
        result['updates'] = {k: int(v) for k, v in saved.get('params', {}).items()
            if k in ('opt/step/value', 'opt_wm/step/value', 'opt_agent/step/value')}
      return result
    except FileNotFoundError:
      continue
    except (OSError, ValueError, TypeError, pickle.UnpicklingError, EOFError):
      return dict(complete=False, error='checkpoint_unreadable')
  return dict(complete=False, error='checkpoint_missing_or_rotating')


def owned(job):
  if not launcher.alive(job):
    return None
  try:
    proc = psutil.Process(job['pid'])
    cmd = proc.cmdline()
    if proc.uids().real != os.getuid() or '--logdir' not in cmd:
      return None
    if Path(cmd[cmd.index('--logdir')+1]).resolve() != Path(job['logdir']).resolve():
      return None
    if not any(x.endswith('/dreamerv3/main.py') for x in cmd):
      return None
    return proc
  except (psutil.Error, ValueError, IndexError):
    return None


def matching_processes(job):
  result = []
  for proc in psutil.process_iter(['pid', 'uids', 'cmdline', 'create_time', 'status']):
    try:
      if proc.info['uids'].real != os.getuid() or proc.info['status'] == 'zombie':
        continue
      cmd = proc.info['cmdline'] or []
      if not any(x.endswith('/dreamerv3/main.py') for x in cmd) or '--logdir' not in cmd:
        continue
      if Path(cmd[cmd.index('--logdir')+1]).resolve() == Path(job['logdir']).resolve():
        result.append(proc)
    except (psutil.Error, ValueError, IndexError):
      pass
  return result


def gpu_stats():
  output = subprocess.check_output(['nvidia-smi',
      '--query-gpu=index,memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu',
      '--format=csv,noheader,nounits'], text=True, timeout=15)
  result = []
  for line in output.splitlines():
    row = dict(zip(('index', 'used_mib', 'total_mib', 'free_mib', 'util', 'temp_c'),
                   map(int, line.split(','))))
    result.append(row)
  return result


def remote_audit(jobs):
  """One bounded request for all existing IDs; never creates a W&B run."""
  env = credentials()
  query = '''query($entity: String!, $project: String!) {
    project(name: $project, entityName: $entity) {
      runs(first: 100, order: "-createdAt") {
        edges { node { id name state summaryMetrics heartbeatAt stopped eventsLineCount } }
      }
    }
  }'''
  data = json.dumps(dict(query=query, variables=dict(
      entity=env['WANDB_ENTITY'], project=launcher.PROJECT))).encode()
  auth = base64.b64encode(('api:' + env['WANDB_API_KEY']).encode()).decode()
  request = urllib.request.Request(
      env.get('WANDB_BASE_URL', 'https://api.wandb.ai').rstrip('/') + '/graphql',
      data=data, headers={'Authorization': 'Basic '+auth,
                          'Content-Type': 'application/json'})
  with wandb_urlopen(request, timeout=25) as response:
    result = json.load(response)
  if result.get('errors') or not result.get('data', {}).get('project'):
    raise RuntimeError('W&B project query failed')
  nodes = {edge['node']['name']: edge['node'] for edge in
           result['data']['project']['runs']['edges']}
  rows = []
  for job in jobs:
    runid = job['wandb_id']
    node = nodes.get(runid)
    if not node:
      rows.append(dict(id=runid, exists=False))
      continue
    summary = json.loads(node['summaryMetrics'] or '{}')
    local = latest_metrics(job['logdir']).get('step', 0)
    remote = summary.get('env_step', summary.get('_step', 0))
    rows.append(dict(id=runid, exists=True, state=node['state'], step=remote,
        storage_id=node['id'], heartbeat_at=node.get('heartbeatAt'),
        events_line_count=node.get('eventsLineCount'),
        stop_requested=bool(node.get('stopped')),
        local_step=local, caught_up=remote is not None and remote >= local,
        losses_finite=all(isinstance(summary.get('loss/'+k), (float, int)) and
            math.isfinite(summary['loss/'+k]) for k in LOSSES)))
  report = dict(time=stamp(), timestamp=time.time(), runs=rows)
  write_json(CONTROL / 'wandb.json', report)
  # The bridge samples every minute; a later brief upload delay must not reuse
  # a lag timestamp left by the watchdog's much slower remote audit cadence.
  for row in rows:
    if row.get('caught_up') and row.get('losses_finite'):
      (CONTROL/('remote_lag_'+row['id']+'.json')).unlink(missing_ok=True)
  return report


def stop_for_resources(job, reason):
  proc = owned(job)
  cp = checkpoint(job['logdir'], inspect_agent=True)
  if not proc or not cp.get('complete') or not cp.get('losses_finite'):
    event('cannot_shed_safely', dict(id=job['wandb_id'], reason=reason))
    return False
  children = proc.children(recursive=True)
  # Parent identity and ownership were checked immediately before signaling.
  proc.terminate()
  for child in children:
    with contextlib.suppress(psutil.Error):
      if child.uids().real == os.getuid():
        child.terminate()
  _, live = psutil.wait_procs([proc, *children], timeout=5)
  for child in live:
    with contextlib.suppress(psutil.Error):
      if child.uids().real == os.getuid():
        child.kill()
  job['watchdog_paused'] = reason
  job['watchdog_retry_at'] = time.time() + 300
  event('resource_pause', dict(id=job['wandb_id'], checkpoint=cp['step'],
                             reason=reason), key='pause:'+job['wandb_id'], cooldown=0)
  return True


def cleanup_orphans(job):
  """Only touch descendants recorded while this exact owned parent was alive."""
  survivors = []
  for saved in job.get('watchdog_children', []):
    try:
      child = psutil.Process(saved['pid'])
      if abs(child.create_time()-saved['create_time']) >= 1 or child.uids().real != os.getuid():
        continue
      if child.status() != psutil.STATUS_ZOMBIE:
        child.terminate()
        survivors.append(child)
    except psutil.Error:
      pass
  _, live = psutil.wait_procs(survivors, timeout=3)
  for child in live:
    with contextlib.suppress(psutil.Error):
      child.kill()
  _, live = psutil.wait_procs(live, timeout=2)
  return not live


def safe_gpu(gpus, preferred):
  choices = [g for g in gpus if g['free_mib'] >= 6656 and g['temp_c'] < 85]
  return min(choices, key=lambda g: (g['index'] != preferred, g['temp_c'],
                                    -g['free_mib'])) if choices else None


def check_bridge_health():
  rollout = read_json(CONTROL/'wandb_transport_rollout/status.json')
  if rollout and not rollout.get('complete') and not rollout.get('halted_for_deadline'):
    try:
      proc = psutil.Process(rollout['pid'])
      healthy = (proc.uids().real == os.getuid() and
          abs(proc.create_time()-rollout['create_time']) < 1 and
          proc.status() != psutil.STATUS_ZOMBIE and
          str(ROOT/'auto_scripts/rollout_wandb_transport.py') in proc.cmdline() and
          time.time()-rollout['timestamp'] < 180)
    except (psutil.Error, KeyError):
      healthy = False
    if not healthy:
      event('wandb_direct_rollout_unhealthy', dict(pid=rollout.get('pid')), cooldown=60)
  info = read_json(CONTROL/'wandb_bridge.json')
  if not info:
    return
  try:
    proc = psutil.Process(info['pid'])
    healthy = (abs(proc.create_time()-info['create_time'])<1 and
        proc.uids().real==os.getuid() and proc.is_running() and
        proc.status()!=psutil.STATUS_ZOMBIE and
        str(ROOT/'auto_scripts/wandb_health_bridge.py') in proc.cmdline())
  except (psutil.Error, KeyError):
    healthy = False
  if not healthy:
    event('wandb_bridge_down', dict(pid=info.get('pid')), cooldown=60)
  elif time.time()-read_json(CONTROL/'wandb_bridge_status.json').get('timestamp',0)>180:
    event('wandb_bridge_stale', dict(pid=info['pid']), cooldown=60)


def poll(recover=True):
  with launcher.STATE.with_suffix('.lock').open('a') as lock:
    try:
      fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
      event('manifest_busy', 'Another launcher owns manifest.lock; no mutations.')
      return
    state = read_json(launcher.STATE)
    jobs = state.get('jobs', [])
    if not jobs or any(j['setting'] not in launcher.SETTINGS for j in jobs):
      raise RuntimeError('Unexpected/empty active manifest; no action')
    check_bridge_health()
    plan = read_json(CONTROL/'scheduled_stop_plan.json')
    if plan.get('enabled'):
      schedule = read_json(CONTROL/'scheduled_stop_status.json')
      if schedule.get('phase') not in ('complete','stopped_with_save_warning'):
        try:
          helper = psutil.Process(schedule['pid'])
          healthy = (helper.uids().real==os.getuid() and
              abs(helper.create_time()-schedule['create_time'])<1 and
              str(ROOT/'auto_scripts/scheduled_experiment_stop.py') in helper.cmdline() and
              helper.status()!=psutil.STATUS_ZOMBIE and
              time.time()-schedule.get('timestamp',0)<120)
        except (psutil.Error,KeyError):
          healthy = False
        if not healthy:
          event('scheduled_stop_helper_unhealthy',dict(deadline=plan['deadline']),cooldown=60)
      rollout = read_json(CONTROL/'scheduled_stop_rollout/status.json')
      if rollout and not rollout.get('complete') and not rollout.get('halted_for_deadline'):
        try:
          controller = psutil.Process(rollout['pid'])
          healthy = (controller.uids().real==os.getuid() and
              abs(controller.create_time()-rollout['create_time'])<1 and
              str(ROOT/'auto_scripts/rollout_scheduled_stop.py') in controller.cmdline() and
              controller.status()!=psutil.STATUS_ZOMBIE and
              time.time()-rollout['timestamp']<180)
        except (psutil.Error,KeyError):
          healthy = False
        if not healthy:
          event('scheduled_stop_rollout_unhealthy',dict(pid=rollout.get('pid')),cooldown=60)
      if time.time()>=plan['quiesce_at']:
        for job in jobs:
          if job['wandb_id'] in plan['run_ids'] and not job.get('user_paused'):
            job.update(user_paused=True,pause_reason='scheduled_user_stop_'+plan['plan_id'])
        launcher.save(state)
        # Independent fallback if the dedicated fence has failed to stop a run.
        if recover and time.time()>=plan['deadline']+30:
          for job in jobs:
            if job['wandb_id'] in plan['run_ids'] and owned(job):
              stop_for_resources(job,'scheduled_deadline_fallback')
          launcher.save(state)
    available = launcher.available_ram_gib()
    gpus = gpu_stats()
    if available < WARN:
      event('ram_warning', dict(available_gib=round(available, 2), floor_gib=FLOOR))
    if recover and available < SHED:
      candidates = sorted([j for j in jobs if owned(j)],
          key=lambda j: psutil.Process(j['pid']).memory_info().rss, reverse=True)
      for job in candidates:
        stop_for_resources(job, 'ram_below_100GiB')
        launcher.save(state)
        if launcher.available_ram_gib() >= 120:
          break
    if recover:
      hot = {g['index'] for g in gpus if g['temp_c'] >= 92}
      for job in jobs:
        if job['gpu'] in hot and owned(job):
          stop_for_resources(job, 'gpu_temperature_at_least_92C')
          launcher.save(state)
    previous = read_json(CONTROL / 'status.json')
    old = {r['id']: r for r in previous.get('runs', [])}
    rows = []
    started = False
    for job in jobs:
      runid = job['wandb_id']
      proc = owned(job)
      if not proc:
        matches = matching_processes(job)
        if len(matches) == 1:
          proc = matches[0]
          job.update(pid=proc.pid, create_time=proc.create_time())
          event('adopted_existing', dict(id=runid, pid=proc.pid), cooldown=0)
        elif len(matches) > 1:
          event('duplicate_processes', dict(id=runid, pids=[p.pid for p in matches]))
          continue
      cp = checkpoint(job['logdir'])
      metrics = latest_metrics(job['logdir'])
      row = dict(id=runid, setting=job['setting'], seed=job['seed'],
          pid=job.get('pid'), gpu=job['gpu'], alive=proc is not None,
          checkpoint=cp, metrics=metrics, user_paused=bool(job.get('user_paused')))
      if proc:
        row['rss_gib'] = round(proc.memory_info().rss / 2**30, 3)
        with contextlib.suppress(psutil.Error):
          job['watchdog_children'] = [dict(pid=p.pid, create_time=p.create_time())
              for p in proc.children(recursive=True) if p.uids().real==os.getuid()]
        prior = old.get(runid, {})
        step = cp.get('step')
        changed = step != prior.get('checkpoint', {}).get('step')
        row['last_progress_at'] = time.time() if changed else prior.get('last_progress_at', time.time())
        if time.time() - row['last_progress_at'] > 3600 or cp.get('age_s', 0) > 3600:
          event('checkpoint_stalled', dict(id=runid, checkpoint=cp), key='stalled:'+runid)
        if metrics and not metrics.get('losses_finite'):
          event('nonfinite_metrics', dict(id=runid), key='nonfinite:'+runid)
        if not cp.get('complete'):
          event('checkpoint_incomplete', dict(id=runid), key='checkpoint:'+runid)
      elif job.get('user_paused'):
        row['pause_reason'] = job.get('pause_reason', 'explicit_user_request')
      elif cp.get('complete') and cp.get('step', 0) >= TARGET:
        row['completed'] = True
      else:
        event('process_missing', dict(id=runid, pid=job.get('pid'), checkpoint=cp),
              key='missing:'+runid)
        eligible = recover and not started and time.time() >= job.get('watchdog_retry_at', 0)
        if eligible and launcher.available_ram_gib() >= RESUME:
          checked = checkpoint(job['logdir'], inspect_agent=True)
          card = safe_gpu(gpus, job['gpu'])
          # At most one recovering run compiles at a time.
          compiling = any(owned(j) and not (Path(j['logdir'])/'training_ready.json').exists()
                          for j in jobs)
          if checked.get('complete') and checked.get('losses_finite') and card and not compiling:
            # Leftover workers from an unexpected main-process crash must exit
            # before a replacement can own the same logdir and replay.
            if not cleanup_orphans(job):
              event('orphan_cleanup_pending', dict(id=runid))
              rows.append(row)
              continue
            import shutil
            backup = CONTROL/'recovery_checkpoints'/runid/checked['name']
            if not backup.exists():
              backup.parent.mkdir(parents=True, exist_ok=True)
              shutil.copytree(Path(job['logdir'])/'ckpt'/checked['name'], backup)
            job['gpu'] = card['index']
            job['watchdog_retry_at'] = time.time() + 900
            launcher.save(state)
            if launcher.launch(job, resume='must'):
              job.pop('watchdog_paused', None)
              job['watchdog_recoveries'] = job.get('watchdog_recoveries', 0) + 1
              launcher.save(state)
              event('recovered', dict(id=runid, pid=job['pid'], gpu=job['gpu'],
                    checkpoint_step=checked['step'], wandb_resume='must'), cooldown=0)
              started = True
          else:
            row['recovery_waiting'] = dict(checkpoint_valid=checked.get('losses_finite',False),
                                         gpu_available=bool(card), compiling=compiling)
    
      rows.append(row)
    for card in gpus:
      if card['free_mib'] < 2048 or card['temp_c'] >= 85:
        event('gpu_warning', card, key=f"gpu:{card['index']}")
    available = launcher.available_ram_gib()
    if available < FLOOR:
      event('ram_floor_breached', dict(available_gib=available), cooldown=60)
    launcher.save(state)
    report = dict(time=stamp(), timestamp=time.time(), daemon_pid=os.getpid(),
        ram_available_gib=round(available, 2), ram_floor_gib=FLOOR,
        swap_used_gib=round(psutil.swap_memory().used/2**30, 2),
        gpus=gpus, runs=rows)
    write_json(CONTROL/'status.json', report)
    with (CONTROL/'resources.jsonl').open('a') as stream:
      stream.write(json.dumps({k:v for k,v in report.items() if k!='runs'})+'\n')
    return report


def audit():
  state = read_json(launcher.STATE)
  report = dict(time=stamp(), runs=[])
  for job in state['jobs']:
    cp = checkpoint(job['logdir'], inspect_agent=True)
    report['runs'].append(dict(id=job['wandb_id'], alive=owned(job) is not None,
        user_paused=bool(job.get('user_paused')),
        checkpoint=cp, metrics=latest_metrics(job['logdir'])))
  report['wandb'] = remote_audit(state['jobs'])
  report['ram_available_gib'] = launcher.available_ram_gib()
  report['gpus'] = gpu_stats()
  write_json(CONTROL/'audit_latest.json', report)
  write_json(CONTROL/f'audit_{int(time.time())}.json', report)
  print(json.dumps(report, indent=2), flush=True)
  return report


def daemon():
  with (CONTROL/'daemon.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_json(CONTROL/'daemon.json', dict(pid=os.getpid(),
        create_time=psutil.Process().create_time(), started_at=stamp()))
    event('watchdog_started', dict(pid=os.getpid(), poll_seconds=30), cooldown=0)
    remote_at = 0
    while not (CONTROL/'STOP').exists():
      started = time.monotonic()
      try:
        report = poll()
        if report and time.time() >= remote_at:
          try:
            remote = remote_audit(read_json(launcher.STATE)['jobs'])
            active_ids = {j['wandb_id'] for j in read_json(launcher.STATE)['jobs']
                          if not j.get('user_paused') and owned(j)}
            for run in remote['runs']:
              if (run['id'] in active_ids and run.get('exists') and
                  (run.get('state') != 'running' or run.get('stop_requested'))):
                # Sparse-log runs can be labeled crashed even while their
                # native heartbeat is accepted and history remains caught up.
                # Keep this advisory on the full-audit cadence; genuine upload
                # lag, invalid metrics, process and resource faults still alert.
                stale_label = (run.get('state') == 'crashed' and
                    not run.get('stop_requested') and run.get('caught_up') and
                    run.get('losses_finite'))
                event('wandb_state_not_running', run, key='remote_state:'+run['id'],
                      cooldown=7200 if stale_label else 600)
              if not run.get('exists') or not run.get('caught_up') or not run.get('losses_finite'):
                # Allow one remote audit interval for asynchronous uploads.
                agefile = CONTROL/('remote_lag_'+run['id']+'.json')
                initial = read_json(agefile).get('since',time.time())
                write_json(agefile, dict(since=initial))
                if time.time()-initial >= 900 or not run.get('exists'):
                  event('wandb_lag_or_invalid', run, key='wandb:'+run['id'])
              else:
                (CONTROL/('remote_lag_'+run['id']+'.json')).unlink(missing_ok=True)
          except Exception as exc:
            event('wandb_check_failed', dict(error_type=type(exc).__name__))
          remote_at = time.time()+900
      except Exception as exc:
        event('watchdog_error', dict(error_type=type(exc).__name__), cooldown=60)
      time.sleep(max(1, 30-(time.monotonic()-started)))
    event('watchdog_stopped', 'STOP requested; training left running.', cooldown=0)


def wait_foreground():
  """Persist schedule across tool interruptions; do not auto-ack alerts."""
  path = CONTROL/'foreground.json'
  schedule = read_json(path)
  if not schedule:
    schedule = dict(check_number=0, next_check=time.time()+1800, event_offset=0)
    write_json(path, schedule)
  print(json.dumps(dict(wait_started=stamp(), next_check=dt.datetime.fromtimestamp(
      schedule['next_check']).astimezone().isoformat())), flush=True)
  while True:
    events = CONTROL/'events.jsonl'
    if events.exists() and events.stat().st_size > schedule['event_offset']:
      with events.open('rb') as stream:
        stream.seek(schedule['event_offset'])
        print('ALERT '+stream.read().decode(errors='replace'),flush=True)
      return
    daemon_info = read_json(CONTROL/'daemon.json')
    pid = daemon_info.get('pid')
    try:
      healthy = pid and abs(psutil.Process(pid).create_time()-daemon_info['create_time'])<1
      status = read_json(CONTROL/'status.json')
      healthy = healthy and time.time()-status.get('timestamp',0)<120
    except psutil.Error:
      healthy = False
    if not healthy:
      print('ALERT watchdog dead or heartbeat stale', flush=True)
      return
    if time.time() >= schedule['next_check']:
      print('SCHEDULED_FULL_AUDIT '+stamp(), flush=True)
      return
    time.sleep(2)


def acknowledge(full=False):
  path = CONTROL/'foreground.json'
  schedule = read_json(path)
  if not schedule:
    schedule = dict(check_number=0, next_check=time.time()+1800)
  if full:
    schedule['check_number'] += 1
    schedule['next_check'] = time.time() + (3600 if schedule['check_number']==1 else 7200)
    schedule['last_full_check'] = stamp()
  events = CONTROL/'events.jsonl'
  schedule['event_offset'] = events.stat().st_size if events.exists() else 0
  write_json(path, schedule)
  print(json.dumps(schedule), flush=True)


def main():
  CONTROL.mkdir(parents=True, exist_ok=True)
  os.chmod(CONTROL, 0o700)
  parser = argparse.ArgumentParser()
  parser.add_argument('action', choices=['daemon','audit','wait','ack','ack-full','check'])
  args = parser.parse_args()
  if args.action == 'daemon': daemon()
  elif args.action == 'audit': audit()
  elif args.action == 'wait': wait_foreground()
  elif args.action == 'check': print(json.dumps(poll(recover=False),indent=2))
  else: acknowledge(args.action=='ack-full')


if __name__ == '__main__':
  main()
