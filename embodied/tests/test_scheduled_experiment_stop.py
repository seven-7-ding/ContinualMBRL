"""Deadline and backup tests; never touch real runs, GPUs, or W&B."""
import json
import os
from pathlib import Path
import pickle
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from embodied.run.scheduled_stop import ScheduledStop,PROTOCOL,write_json
from auto_scripts import scheduled_experiment_stop as s
from auto_scripts import rollout_scheduled_stop as r


def plan_for(job=None):
  return dict(enabled=True,plan_id='test',deadline=1000,quiesce_at=940,
              run_ids=['same-id'] if job is None else [job['wandb_id']])


@pytest.mark.parametrize('now,runid,enabled,due',[
    (939,'same-id',True,False),(940,'same-id',True,True),
    (1000,'same-id',True,True),(1000,'other-id',True,False),
    (1000,'same-id',False,False)])
def test_cooperative_deadline_only_affects_explicit_scope(tmp_path,monkeypatch,now,runid,enabled,due):
  path=tmp_path/'plan.json';plan=plan_for();plan['enabled']=enabled;write_json(path,plan)
  monkeypatch.setenv('CONTINUALMBRL_STOP_PLAN',str(path))
  monkeypatch.setenv('WANDB_RUN_ID',runid)
  with mock.patch('embodied.run.scheduled_stop.time.time',return_value=now):
    control=ScheduledStop(tmp_path/'run')
    assert control.due() is due
  ready=json.loads((tmp_path/'run/scheduled_stop_ready.json').read_text())
  assert ready['pid']==os.getpid() and ready['protocol']==PROTOCOL


def test_unscheduled_training_ignores_the_control(monkeypatch,tmp_path):
  monkeypatch.delenv('CONTINUALMBRL_STOP_PLAN',raising=False)
  control=ScheduledStop(tmp_path)
  assert not control.due()
  assert not list(tmp_path.iterdir())


def test_control_reloads_cancellation_and_retains_valid_intent_on_bad_read(tmp_path,monkeypatch):
  path=tmp_path/'plan.json';plan=plan_for();write_json(path,plan)
  monkeypatch.setenv('CONTINUALMBRL_STOP_PLAN',str(path));monkeypatch.setenv('WANDB_RUN_ID','same-id')
  with mock.patch('embodied.run.scheduled_stop.time.time',return_value=950):
    control=ScheduledStop(tmp_path/'run');assert control.due()
  path.write_text('incomplete JSON')
  with mock.patch('embodied.run.scheduled_stop.time.time',return_value=953):
    assert control.due()
  plan['enabled']=False;write_json(path,plan)
  with mock.patch('embodied.run.scheduled_stop.time.time',return_value=956):
    assert not control.due()


def test_cooperative_ack_requires_completed_full_checkpoint(tmp_path,monkeypatch):
  path=tmp_path/'plan.json';write_json(path,plan_for())
  monkeypatch.setenv('CONTINUALMBRL_STOP_PLAN',str(path));monkeypatch.setenv('WANDB_RUN_ID','same-id')
  with mock.patch('embodied.run.scheduled_stop.time.time',return_value=950):
    control=ScheduledStop(tmp_path/'run');assert control.due()
  checkpoint=tmp_path/'run/ckpt/final';checkpoint.mkdir(parents=True)
  cp=SimpleNamespace(latest=lambda:checkpoint)
  with pytest.raises(AssertionError):control.saved(cp,1234)
  assert not (tmp_path/'run/scheduled_stop_saved.json').exists()
  (checkpoint/'done').touch();control.saved(cp,1234)
  ack=json.loads((tmp_path/'run/scheduled_stop_saved.json').read_text())
  assert ack['step']==1234 and ack['checkpoint']=='final' and ack['plan_id']=='test'


def test_final_checkpoint_roundtrip_keeps_model_optimizer_step_and_replay(tmp_path,monkeypatch):
  import elements
  path=tmp_path/'plan.json';write_json(path,plan_for())
  monkeypatch.setenv('CONTINUALMBRL_STOP_PLAN',str(path));monkeypatch.setenv('WANDB_RUN_ID','same-id')
  with mock.patch('embodied.run.scheduled_stop.time.time',return_value=950):
    control=ScheduledStop(tmp_path/'run');assert control.due()
  class Saved:
    def __init__(self,value):self.value=value
    def save(self):return self.value
    def load(self,value):self.value=value
  cp=elements.Checkpoint(tmp_path/'run/ckpt')
  cp.step=Saved(1234)
  cp.agent=Saved(dict(params={'model/kernel':np.array([1.,2.]),
                            'opt_wm/step/value':np.array(986),
                            'opt_agent/step/value':np.array(654)}))
  cp.replay=Saved(dict(sequence=np.arange(65)))
  cp.save();control.saved(cp,1234)
  cp.step.value=-1;cp.agent.value={};cp.replay.value={};cp.load()
  assert cp.step.value==1234
  assert cp.agent.value['params']['opt_wm/step/value']==986
  assert cp.agent.value['params']['opt_agent/step/value']==654
  np.testing.assert_array_equal(cp.replay.value['sequence'],np.arange(65))
  assert json.loads((tmp_path/'run/scheduled_stop_saved.json').read_text())['step']==1234


def test_backup_is_independent_and_hash_verified(tmp_path):
  src=tmp_path/'source';src.write_bytes(bytes(range(256))*8192)
  dest=tmp_path/'backup/copy';proof=s.verified_copy(src,dest)
  assert src.read_bytes()==dest.read_bytes()
  assert src.stat().st_ino!=dest.stat().st_ino
  assert proof['bytes']==src.stat().st_size
  import hashlib
  assert proof['sha256']==hashlib.sha256(src.read_bytes()).hexdigest()
  src.write_bytes(b'modified');assert dest.stat().st_size==proof['bytes']


def test_pause_intent_preserves_existing_default_pause(tmp_path):
  jobs=[dict(wandb_id='a'),dict(wandb_id='b',user_paused=True,pause_reason='user_default')]
  state=dict(jobs=jobs)
  with mock.patch.object(s.w.launcher,'STATE',tmp_path/'manifest.json'), \
       mock.patch.object(s.w,'read_json',return_value=state), \
       mock.patch.object(s,'scoped_jobs',return_value=jobs), \
       mock.patch.object(s.w.launcher,'save') as save:
    s.pause_manifest(dict(plan_id='tomorrow'))
  assert all(j['user_paused'] for j in jobs)
  assert jobs[1]['pause_reason']=='user_default'
  assert jobs[0]['pause_reason']=='scheduled_user_stop_tomorrow'
  save.assert_called_once_with(state)


def test_fence_freezes_all_parents_before_checkpoint_inspection(tmp_path):
  jobs=[dict(wandb_id='a',pid=123,setting='agent_as_leader_32',logdir=str(tmp_path/'a')),
        dict(wandb_id='b',pid=456,setting='wm_as_leader_32',logdir=str(tmp_path/'b'))]
  parents={j['wandb_id']:mock.Mock(pid=j['pid']) for j in jobs}
  calls=[]
  for runid,proc in parents.items():
    proc.children.return_value=[];proc.uids.return_value=SimpleNamespace(real=os.getuid())
    proc.suspend.side_effect=lambda name=runid:calls.append('freeze:'+name)
  def proof(job):
    assert calls[:2]==['freeze:a','freeze:b']
    return dict(step=100,name='snapshot'),tmp_path
  def owned(job):
    return parents[job['wandb_id']] if len(calls)<3 else None
  with mock.patch.object(s,'pause_manifest',return_value={}), \
       mock.patch.object(s,'scoped_jobs',return_value=jobs), \
       mock.patch.object(s.w,'owned',side_effect=lambda j:parents[j['wandb_id']]), \
       mock.patch.object(s,'checkpoint_proof',side_effect=proof), \
       mock.patch.object(s.w,'read_json',return_value={}), \
       mock.patch.object(s.w,'latest_metrics',return_value=dict(step=100)), \
       mock.patch.object(s.w,'write_json'),mock.patch.object(s,'write_status'), \
       mock.patch.object(s.psutil,'wait_procs',return_value=([],[])), \
       mock.patch.object(s.w,'cleanup_orphans',return_value=True):
    # The final ownership query must observe the terminated parents as gone.
    for proc in parents.values():
      proc.terminate.side_effect=lambda:calls.append('terminated')
    with mock.patch.object(s.w,'owned',side_effect=lambda j:(
        parents[j['wandb_id']] if 'terminated' not in calls else None)):
      result=s.freeze_and_stop(dict(plan_id='test',backup_dir=str(tmp_path)))
  assert len(result)==2
  assert all(proc.terminate.called for proc in parents.values())


@pytest.mark.parametrize('ready',[None,100,101])
def test_rollout_waits_for_actual_first_training_update(tmp_path,ready):
  job=dict(logdir=str(tmp_path),stop_protocol_rollout=dict(before=dict(checkpoint=dict(step=100))))
  with mock.patch.object(r.stop,'protocol_ready',return_value=True), \
       mock.patch.object(r.w,'read_json',return_value=ready):
    assert r.compiled_ready(job) is (ready==101)


def test_launcher_deadline_fence_precedes_resource_checks(tmp_path):
  plan=tmp_path/'watchdog/scheduled_stop_plan.json';plan.parent.mkdir(parents=True)
  write_json(plan,plan_for())
  with mock.patch.object(s.w.launcher,'CONTROL',tmp_path), \
       mock.patch.object(s.w.launcher.time,'time',return_value=950), \
       mock.patch.object(s.w.launcher,'gpu_stats') as gpu:
    assert s.w.launcher.launch(dict(wandb_id='same-id'),resume='must') is False
  gpu.assert_not_called()


def test_unavailable_canary_cannot_be_bypassed_by_another_run(tmp_path):
  jobs=[dict(setting=setting,seed=seed,wandb_id=f'{setting}-{seed}',gpu=0)
        for setting in ('best_response_32','agent_as_leader_32','wm_as_leader_32')
        for seed in (1000,2000,3000)]
  canary=jobs[-1]
  state=tmp_path/'manifest.json'
  with mock.patch.object(r.w.launcher,'STATE',state), \
       mock.patch.object(r.w,'read_json',side_effect=lambda path:(
           dict(jobs=jobs) if path==state else dict(quiesce_at=10**20))), \
       mock.patch.object(r.stop,'scoped_jobs',return_value=jobs), \
       mock.patch.object(r.w,'remote_audit',return_value={'runs':[
           dict(id=j['wandb_id']) for j in jobs]}), \
       mock.patch.object(r.w,'owned',return_value=object()), \
       mock.patch.object(r.w.launcher,'save'), \
       mock.patch.object(r.w.launcher,'available_ram_gib',return_value=300), \
       mock.patch.object(r.w,'gpu_stats',return_value=[dict(index=0,free_mib=20000,temp_c=60)]), \
       mock.patch.object(r.rollout,'stop_from_checkpoint',return_value=None) as stop:
    assert r.tick() is False
  stop.assert_called_once_with(canary,dict(id=canary['wandb_id']))


def test_backup_includes_replay_and_logs_but_excludes_caches_and_credentials(tmp_path):
  root=tmp_path/'repo';root.mkdir()
  control=root/'control';control.mkdir()
  source=root/'run';(source/'replay').mkdir(parents=True)
  (source/'replay/chunk.npz').write_bytes(b'original replay')
  (source/'metrics.jsonl').write_text('{"step":1234}\n')
  for folder in ('cache','tmp','config'):
    (source/folder).mkdir();(source/folder/'secret-or-transient').write_bytes(b'exclude')
  (source/'latest-link').symlink_to(source/'metrics.jsonl')
  for name in ('AGENT.md','task_checklist.md','codex-cli-executor.log','new.py'):
    (root/name).write_text('source evidence\n')
  state=root/'manifest.json';state.write_text('{}')
  planfile=control/'plan.json';planfile.write_text('{}')
  job=dict(logdir=str(source),setting='agent_as_leader_32',seed=1000)
  target=root/'backup'
  def git_output(args,**kwargs):
    if args[1:]==['diff','--name-only']:return 'new.py\n'
    if args[1:]==['ls-files','--others','--exclude-standard']:return ''
    if args[1:]==['rev-parse','HEAD']:return 'baseline-hash\n'
    raise AssertionError(args)
  def git_patch(args,**kwargs):
    kwargs['stdout'].write(b'patch evidence\n')
  with mock.patch.object(s,'ROOT',root),mock.patch.object(s,'PLAN',planfile), \
       mock.patch.object(s.w,'CONTROL',control), \
       mock.patch.object(s.w.launcher,'STATE',state), \
       mock.patch.object(s,'scoped_jobs',return_value=[job]), \
       mock.patch.object(s,'write_status'), \
       mock.patch.object(s.subprocess,'check_output',side_effect=git_output), \
       mock.patch.object(s.subprocess,'run',side_effect=git_patch):
    report=s.backup_artifacts(dict(backup_dir=str(target)))
  copied=target/'runs/agent_as_leader_32/seed_1000'
  assert (copied/'replay/chunk.npz').read_bytes()==b'original replay'
  assert (copied/'metrics.jsonl').exists()
  assert not any((copied/x).exists() for x in ('cache','tmp','config','latest-link'))
  import hashlib
  for row in report['files']:
    assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
  (source/'replay/chunk.npz').write_bytes(b'mutated')
  assert (copied/'replay/chunk.npz').read_bytes()==b'original replay'
  assert (target/'RESTORE.md').exists() and (target/'metadata/code.patch').exists()


@pytest.mark.parametrize('now,pause,finish',[(939,False,False),(940,True,False),(1000,True,True)])
def test_helper_does_not_stop_before_deadline(tmp_path,monkeypatch,now,pause,finish):
  plan={**plan_for(), 'backup_dir':str(tmp_path/'backup')}
  monkeypatch.setattr(s.sys,'argv',['scheduled_experiment_stop.py'])
  with mock.patch.object(s.w,'CONTROL',tmp_path), \
       mock.patch.object(s.w,'read_json',side_effect=lambda path:(plan if path==s.PLAN else
           dict(baseline_saved_at=1,late_baseline_saved_at=1))), \
       mock.patch.object(s,'scoped_jobs',return_value=[]), \
       mock.patch.object(s.time,'time',return_value=now), \
       mock.patch.object(s.time,'sleep',side_effect=RuntimeError('end test wait')), \
       mock.patch.object(s,'write_status'), \
       mock.patch.object(s,'pause_manifest') as intent, \
       mock.patch.object(s,'finalize') as final:
    if finish:s.main()
    else:
      with pytest.raises(RuntimeError,match='end test wait'):s.main()
  assert intent.called is pause
  assert final.called is finish
