#!/usr/bin/env python3
"""Send genuine system health for healthy owned runs with reset event offsets.

Some resumed SDKs write system events below the server's existing event tail.
Empty heartbeats do not refresh the cloud label. Append one measured health row
at the server tail only while the native writer is demonstrably far behind it.
Never write training history, summary, config, or a claimed run state.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from auto_scripts import experiment_watchdog as w


def eligibility(job, remote, previous, now):
  if job.get('user_paused'):
    return 'user_paused', None
  if not remote.get('exists') or remote.get('stop_requested'):
    return 'remote_missing_or_stop_requested', None
  if not remote.get('losses_finite'):
    return 'remote_metrics_nonfinite', None
  if not remote.get('caught_up'):
    return 'remote_metrics_behind', None
  proc = w.owned(job)
  if not proc:
    return 'process_missing', None
  cp = w.checkpoint(job['logdir'], inspect_agent=True)
  metrics = w.latest_metrics(job['logdir'])
  if not cp.get('complete') or not cp.get('losses_finite') or cp.get('age_s', 1801)>1800:
    return 'checkpoint_stale_or_invalid', None
  if cp['step'] >= w.TARGET:
    return 'completed', None
  if not metrics.get('losses_finite'):
    return 'nonfinite_metrics', None
  times = proc.cpu_times()
  cpu = times.user + times.system
  current = dict(pid=proc.pid, create_time=proc.create_time(), cpu_seconds=cpu,
      checkpoint_step=cp['step'], log_step=metrics.get('step', 0), checked_at=now)
  same = (previous.get('pid')==proc.pid and
          previous.get('create_time')==current['create_time'])
  if same and not (cpu-previous.get('cpu_seconds',cpu)>.05 or
      cp['step']>previous.get('checkpoint_step',cp['step']) or
      current['log_step']>previous.get('log_step',current['log_step'])):
    return 'no_observed_progress', current
  if not same and cpu<1:
    return 'initializing', current
  return None, current


def native_event_position(job, now):
  """Read the current process's recent SDK offset, with bounded I/O."""
  root = Path(job['logdir']) / 'wandb' / 'wandb'
  suffix = '-' + job['wandb_id']
  candidates = []
  for directory in root.glob('run-*' + suffix):
    try:
      started = time.mktime(time.strptime(directory.name[4:-len(suffix)],
                                         '%Y%m%d_%H%M%S'))
    except ValueError:
      continue
    if -60 <= started-job['create_time'] <= 300:
      candidates.append((started, directory/'logs/debug-internal.log'))
  for _, path in sorted(candidates, reverse=True):
    try:
      with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-65536))
        lines = stream.read().splitlines()
    except OSError:
      continue
    for line in reversed(lines):
      try:
        row = json.loads(line)
        if 'events_offset' not in row:
          continue
        stamp = dt.datetime.fromisoformat(row['time'].replace('Z','+00:00')).timestamp()
        end = row['events_offset'] + row['events_lines']
        if (-1 <= now-stamp <= 90 and type(end) is int and end >= 0):
          return dict(end=end, timestamp=stamp)
        return None
      except (ValueError, KeyError, TypeError):
        continue
  return None


def system_event_offset(job, remote, now):
  count = remote.get('events_line_count')
  native = native_event_position(job, now)
  # The SDK is still writing its earlier offsets. Keep a large disjoint gap;
  # if it catches up, restarts, or becomes unobservable, do not contend with it.
  if type(count) is not int or not native or count-native['end'] < 1024:
    return None
  return dict(offset=count, native_end=native['end'])


class Client:

  def __init__(self):
    env = w.credentials()
    self.base = env.get('WANDB_BASE_URL','https://api.wandb.ai').rstrip('/')
    self.entity = env['WANDB_ENTITY']
    token = base64.b64encode(('api:'+env['WANDB_API_KEY']).encode()).decode()
    self.headers = {'Authorization':'Basic '+token, 'Content-Type':'application/json'}

  def post(self, path, payload):
    request = urllib.request.Request(self.base+path,
        data=json.dumps(payload).encode(), headers=self.headers)
    # Healthy native requests can take 16s during a server latency spike.
    # Three workers and nine eligible runs bound a full round to about 60s,
    # below the independent watchdog's 180s bridge liveness threshold.
    with w.wandb_urlopen(request, timeout=20) as response:
      body=response.read()
      return json.loads(body) if body else {}

  def heartbeat(self, job):
    path = '/files/'+'/'.join(urllib.parse.quote(value,safe='') for value in
        (self.entity,w.launcher.PROJECT,job['wandb_id']))+'/file_stream'
    # Same idle payload as installed wandb.sdk.internal.file_stream.FileStreamApi.
    self.post(path, dict(complete=False, failed=False, dropped=0, uploaded=[]))

  def system_heartbeat(self, job, progress, offset):
    now = time.time()
    proc = w.owned(job)
    if not proc or proc.pid != progress['pid']:
      raise RuntimeError('Owned process changed before system heartbeat')
    row = {'_wandb':True, '_timestamp':now,
        'system/watchdog/process_cpu_seconds':progress['cpu_seconds'],
        'system/watchdog/process_uptime_seconds':now-proc.create_time(),
        'system/watchdog/process_rss_bytes':proc.memory_info().rss,
        'system/watchdog/checkpoint_step':progress['checkpoint_step'],
        'system/watchdog/log_step':progress['log_step']}
    path = '/files/'+'/'.join(urllib.parse.quote(value,safe='') for value in
        (self.entity,w.launcher.PROJECT,job['wandb_id']))+'/file_stream'
    self.post(path, dict(complete=False, failed=False, files={
        'wandb-events.jsonl':dict(offset=offset, content=[json.dumps(row)])}))


def tick(client):
  jobs = w.read_json(w.launcher.STATE)['jobs']
  remote = {r['id']:r for r in w.remote_audit(jobs)['runs']}
  old = w.read_json(w.CONTROL/'wandb_bridge_status.json').get('runs',{})
  now = time.time()
  def check(job):
    runid = job['wandb_id']
    r = remote[runid]
    reason, progress = eligibility(job,r,old.get(runid,{}),now)
    row = dict(progress or {}, id=runid, reason=reason, remote_state=r.get('state'))
    if reason:
      upload_grace = False
      if reason == 'remote_metrics_behind':
        previous = old.get(runid, {})
        since = (previous.get('upload_lag_since', now)
            if previous.get('upload_lag_pid') == job.get('pid') else now)
        row.update(upload_lag_since=since, upload_lag_pid=job.get('pid'))
        upload_grace = 0 <= now-since < 90
      if reason not in ('user_paused','completed','initializing') and not upload_grace:
        w.event('wandb_heartbeat_skipped',dict(id=runid,reason=reason),key='hb_skip:'+runid)
      return runid,row
    # Re-read explicit pause/PID intent immediately before touching the session.
    current = next(j for j in w.read_json(w.launcher.STATE)['jobs'] if j['wandb_id']==runid)
    if current.get('user_paused') or current['pid']!=job['pid'] or not w.owned(current):
      row['reason']='identity_or_pause_changed'
      return runid,row
    if r.get('state') not in ('running','crashed'):
      row['reason']='unexpected_remote_terminal_state'
      w.event('wandb_state_not_running',r,key='remote_state:'+runid)
      return runid,row
    try:
      if r['state']=='crashed':
        # The backend may derive this label from the last metric timestamp even
        # after accepting native SDK idle heartbeats. Keep the observed label
        # truthful; never repeatedly overwrite it with a claimed running state.
        row['cloud_label_stale']=True
        w.event('wandb_cloud_label_stale',dict(id=runid,
            checkpoint_step=progress['checkpoint_step'],caught_up=r.get('caught_up')),
            key='hb_label:'+runid,cooldown=7200)
      stamp = r.get('heartbeat_at')
      age = now-dt.datetime.fromisoformat(stamp.replace('Z','+00:00')).timestamp() if stamp else 1e9
      if age>=60 or r['state']=='crashed':
        position = system_event_offset(job,r,time.time())
        if position:
          client.system_heartbeat(job,progress,position['offset'])
          row.update(system_event_sent_at=w.stamp(),
                     system_event_offset=position['offset'],
                     native_events_end=position['native_end'])
        else:
          client.heartbeat(job)
          row['heartbeat_sent_at']=w.stamp()
          if r['state']=='crashed':
            w.event('wandb_system_heartbeat_unavailable',dict(id=runid),
                    key='system_hb_unavailable:'+runid)
    except Exception as exc:
      row['error_type']=type(exc).__name__
      w.event('wandb_heartbeat_failed',dict(id=runid,error_type=type(exc).__name__),
              key='hb_failed:'+runid,cooldown=60)
    return runid,row
  with ThreadPoolExecutor(max_workers=3) as pool:
    rows=dict(pool.map(check,jobs))
  report=dict(time=w.stamp(),timestamp=time.time(),pid=os.getpid(),runs=rows)
  w.write_json(w.CONTROL/'wandb_bridge_status.json',report)
  print(json.dumps(dict(time=report['time'],sent=sum(
      'heartbeat_sent_at' in r or 'system_event_sent_at' in r for r in rows.values()),
      skipped=sum(bool(r.get('reason')) for r in rows.values()),
      errors=sum('error_type' in r for r in rows.values()))),flush=True)


def main():
  parser=argparse.ArgumentParser()
  parser.add_argument('--once',action='store_true')
  args=parser.parse_args()
  with (w.CONTROL/'wandb_bridge.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    w.write_json(w.CONTROL/'wandb_bridge.json',dict(pid=os.getpid(),
        create_time=psutil.Process().create_time(),started_at=w.stamp()))
    client=Client()
    while not (w.CONTROL/'STOP').exists():
      begin=time.monotonic()
      try:
        tick(client)
      except Exception as exc:
        w.event('wandb_bridge_error',dict(error_type=type(exc).__name__),cooldown=60)
      if args.once:return
      time.sleep(max(1,60-(time.monotonic()-begin)))


if __name__=='__main__':
  main()
