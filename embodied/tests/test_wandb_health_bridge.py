from types import SimpleNamespace
from unittest import mock
import datetime as dt
import json
import time

import pytest

from auto_scripts import wandb_health_bridge as b


@pytest.mark.parametrize('problem,expected', [
    ('paused','user_paused'), ('missing','process_missing'),
    ('stale','checkpoint_stale_or_invalid'),
    ('nonfinite','checkpoint_stale_or_invalid'),
    ('stopped','remote_missing_or_stop_requested'),
    ('lagged','remote_metrics_behind'),
    ('idle','no_observed_progress'), (None,None)])
def test_heartbeat_requires_owned_progress_and_valid_checkpoint(problem, expected):
  job=dict(logdir='private', user_paused=problem=='paused')
  remote=dict(exists=True,stop_requested=problem=='stopped',losses_finite=True,
              caught_up=problem!='lagged')
  proc=mock.Mock(pid=123)
  proc.create_time.return_value=100
  proc.cpu_times.return_value=SimpleNamespace(user=10.,system=1.)
  cp=dict(complete=True,losses_finite=problem!='nonfinite',step=1000,
          age_s=1900 if problem=='stale' else 100)
  old={} if problem!='idle' else dict(pid=123,create_time=100,cpu_seconds=11.,
                                      checkpoint_step=1000,log_step=1000)
  with mock.patch.object(b.w,'owned',return_value=None if problem=='missing' else proc), \
       mock.patch.object(b.w,'checkpoint',return_value=cp), \
       mock.patch.object(b.w,'latest_metrics',return_value=dict(step=1000,losses_finite=True)):
    reason,progress=b.eligibility(job,remote,old,200)
  assert reason==expected
  if problem is None:
    assert progress['pid']==123 and progress['checkpoint_step']==1000


def test_sdk_heartbeat_has_no_history_config_or_finish_payload():
  client=b.Client.__new__(b.Client)
  client.entity='owner'
  with mock.patch.object(client,'post',return_value={}) as call:
    client.heartbeat({'wandb_id':'same-id'})
  path,payload=call.call_args.args
  assert path.endswith('/same-id/file_stream')
  assert payload==dict(complete=False,failed=False,dropped=0,uploaded=[])
  assert not any(k in payload for k in ('files','config','exitcode'))


def test_stale_cloud_label_is_reported_without_overwriting_run_state(tmp_path):
  job=dict(wandb_id='same-id',pid=123)
  remote=dict(id='same-id',state='crashed',caught_up=True)
  client=mock.Mock(spec=b.Client)
  with mock.patch.object(b.w,'CONTROL',tmp_path), \
       mock.patch.object(b.w,'read_json',return_value={'jobs':[job]}), \
       mock.patch.object(b.w,'remote_audit',return_value={'runs':[remote]}), \
       mock.patch.object(b.w,'owned',return_value=object()), \
       mock.patch.object(b,'system_event_offset',return_value=None), \
       mock.patch.object(b,'eligibility',return_value=(None,dict(checkpoint_step=1000))), \
       mock.patch.object(b.w,'event'), mock.patch.object(b.w,'write_json') as write:
    b.tick(client)
  client.heartbeat.assert_called_once_with(job)
  client.post.assert_not_called()
  row=write.call_args.args[1]['runs']['same-id']
  assert row['remote_state']=='crashed' and row['cloud_label_stale']


@pytest.mark.parametrize('count,native,allowed',[
    (20490,5200,True),(20490,20400,False),(20490,20500,False),
    (None,5200,False),(20490,None,False)])
def test_system_events_require_disjoint_native_offsets(count,native,allowed):
  with mock.patch.object(b,'native_event_position',return_value=(
      None if native is None else dict(end=native,timestamp=100))):
    result=b.system_event_offset({},dict(events_line_count=count),100)
  assert bool(result)==allowed
  if allowed:
    assert result==dict(offset=20490,native_end=5200)


def test_sdk_log_must_match_current_process_and_be_fresh(tmp_path):
  now=time.time()
  dirname='run-'+time.strftime('%Y%m%d_%H%M%S',time.localtime(now))+'-same-id'
  path=tmp_path/'wandb/wandb'/dirname/'logs/debug-internal.log'
  path.parent.mkdir(parents=True)
  path.write_text(json.dumps(dict(time=dt.datetime.fromtimestamp(now).astimezone().isoformat(),
                                 events_offset=5000,events_lines=1))+'\n')
  job=dict(logdir=str(tmp_path),wandb_id='same-id',create_time=now)
  assert b.native_event_position(job,now)['end']==5001
  assert b.native_event_position(job,now+91) is None
  assert b.native_event_position({**job,'create_time':now+601},now) is None


def test_system_heartbeat_appends_only_measured_system_data():
  client=b.Client.__new__(b.Client)
  client.entity='owner'
  proc=mock.Mock(pid=123)
  proc.create_time.return_value=100
  proc.memory_info.return_value=SimpleNamespace(rss=9876)
  progress=dict(pid=123,cpu_seconds=11,checkpoint_step=1000,log_step=900)
  with mock.patch.object(b.w,'owned',return_value=proc), \
       mock.patch.object(b.time,'time',return_value=200), \
       mock.patch.object(client,'post',return_value={}) as post:
    client.system_heartbeat(dict(wandb_id='same-id'),progress,20490)
  path,payload=post.call_args.args
  assert path.endswith('/same-id/file_stream')
  assert set(payload)=={'complete','failed','files'}
  assert payload['complete'] is False and payload['failed'] is False
  assert set(payload['files'])=={'wandb-events.jsonl'}
  event=payload['files']['wandb-events.jsonl']
  assert event['offset']==20490 and len(event['content'])==1
  row=json.loads(event['content'][0])
  assert row['system/watchdog/process_rss_bytes']==9876
  assert row['system/watchdog/process_cpu_seconds']==11
  assert row['system/watchdog/process_uptime_seconds']==100
  assert row['system/watchdog/checkpoint_step']==1000
  assert '_step' not in row and 'env_step' not in row
  assert all(k.startswith('system/watchdog/') or k in ('_wandb','_timestamp') for k in row)


def test_missing_owned_process_cannot_send_system_heartbeat():
  client=b.Client.__new__(b.Client)
  with mock.patch.object(b.w,'owned',return_value=None), \
       mock.patch.object(client,'post') as post:
    with pytest.raises(RuntimeError):
      client.system_heartbeat({},dict(pid=123),1000)
  post.assert_not_called()


def test_tick_uses_system_tail_and_preserves_observed_cloud_label(tmp_path):
  job=dict(wandb_id='same-id',pid=123)
  remote=dict(id='same-id',state='crashed',caught_up=True)
  progress=dict(pid=123,checkpoint_step=1000)
  client=mock.Mock(spec=b.Client)
  with mock.patch.object(b.w,'CONTROL',tmp_path), \
       mock.patch.object(b.w,'read_json',return_value={'jobs':[job]}), \
       mock.patch.object(b.w,'remote_audit',return_value={'runs':[remote]}), \
       mock.patch.object(b.w,'owned',return_value=object()), \
       mock.patch.object(b,'eligibility',return_value=(None,progress)), \
       mock.patch.object(b,'system_event_offset',return_value=dict(offset=20490,native_end=5000)), \
       mock.patch.object(b.w,'event'), mock.patch.object(b.w,'write_json') as write:
    b.tick(client)
  client.system_heartbeat.assert_called_once_with(job,progress,20490)
  client.heartbeat.assert_not_called()
  row=write.call_args.args[1]['runs']['same-id']
  assert row['remote_state']=='crashed'
  assert row['system_event_offset']==20490 and row['native_events_end']==5000
  assert 'system_event_sent_at' in row


@pytest.mark.parametrize('change',[dict(user_paused=True),dict(pid=456)])
def test_pause_or_identity_change_before_send_prevents_all_posts(tmp_path,change):
  job=dict(wandb_id='same-id',pid=123)
  client=mock.Mock(spec=b.Client)
  with mock.patch.object(b.w,'CONTROL',tmp_path), \
       mock.patch.object(b.w,'read_json',side_effect=[{'jobs':[job]}, {},
                                                    {'jobs':[{**job,**change}]}]), \
       mock.patch.object(b.w,'remote_audit',return_value={'runs':[
           dict(id='same-id',state='crashed',caught_up=True)]}), \
       mock.patch.object(b,'eligibility',return_value=(None,dict(pid=123,checkpoint_step=1000))), \
       mock.patch.object(b.w,'owned',return_value=object()), \
       mock.patch.object(b.w,'write_json'):
    b.tick(client)
  client.system_heartbeat.assert_not_called()
  client.heartbeat.assert_not_called()


@pytest.mark.parametrize('previous,now,expected_alert', [
    ({},100,False),
    (dict(upload_lag_since=100,upload_lag_pid=123),189,False),
    (dict(upload_lag_since=100,upload_lag_pid=123),190,True),
    (dict(upload_lag_since=100,upload_lag_pid=456),200,False)])
def test_upload_grace_preserves_lag_and_never_sends_heartbeat(previous,now,expected_alert):
  job=dict(wandb_id='same-id',pid=123)
  client=mock.Mock(spec=b.Client)
  state=[{'jobs':[job]}, {'runs':{'same-id':previous}}]
  with mock.patch.object(b.w,'read_json',side_effect=state), \
       mock.patch.object(b.w,'remote_audit',return_value={'runs':[dict(id='same-id',state='running')]}), \
       mock.patch.object(b,'eligibility',return_value=('remote_metrics_behind',None)), \
       mock.patch.object(b.time,'time',return_value=now), \
       mock.patch.object(b.w,'event') as event, \
       mock.patch.object(b.w,'write_json') as write:
    b.tick(client)
  assert bool(event.call_count)==expected_alert
  row=write.call_args.args[1]['runs']['same-id']
  assert row['reason']=='remote_metrics_behind'
  assert row['upload_lag_pid']==123
  assert row['upload_lag_since']==(previous['upload_lag_since']
      if previous.get('upload_lag_pid')==123 else now)
  client.system_heartbeat.assert_not_called()
  client.heartbeat.assert_not_called()
