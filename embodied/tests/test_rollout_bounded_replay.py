from unittest import mock
import io
import json
import pickle
import os
from types import SimpleNamespace

import numpy as np
import pytest

from auto_scripts import rollout_bounded_replay as r


@pytest.mark.parametrize('running_profile',[None,'bounded_replay_v1','bounded_replay_v2'])
def test_compaction_rollout_rejects_previous_loaded_code(running_profile):
  proc=mock.Mock()
  proc.cmdline.return_value=['python','--replay.cache_bytes',str(2*1024**3),
                           '--run.usage.malloc','False']
  proc.environ.return_value={'CONTINUALMBRL_REPLAY_PROFILE':running_profile}
  with mock.patch.object(r.w,'owned',return_value=proc):
    assert r.optimized_process({'performance_profile':'bounded_replay_v2'}) == (
        running_profile=='bounded_replay_v2')


@pytest.mark.parametrize('profile,state,checkpoint,ready',[
    ('bounded_replay_v1','verified',False,True),
    ('bounded_replay_v2','ready',False,False),
    ('bounded_replay_v2','verified',False,False),
    ('bounded_replay_v2','verified',True,True)])
def test_compaction_canary_must_exercise_and_verify_new_checkpoint(
    profile,state,checkpoint,ready):
  job=dict(performance_profile=profile,optimization_rollout=dict(state=state))
  if checkpoint:job['optimization_rollout']['checkpoint_verified']={'all_parameters_finite':True}
  assert r.canary_ready(job)==ready


@pytest.mark.parametrize('bad_key',['model/kernel','opt_agent/moment'])
def test_invalid_full_checkpoint_resumes_frozen_parent_without_terminating(tmp_path,bad_key):
  backup=tmp_path/'checkpoints/existing/snapshot';backup.mkdir(parents=True)
  params={k:np.array([np.nan if k==bad_key else 1.])
          for k in ('model/kernel','opt_agent/moment')}
  (backup/'agent.pkl').write_bytes(pickle.dumps({'params':params}))
  proc=mock.Mock()
  proc.children.return_value=[]
  proc.uids.return_value=SimpleNamespace(real=os.getuid())
  with mock.patch.object(r,'DIRECTORY',tmp_path), \
       mock.patch.object(r.w,'owned',return_value=proc), \
       mock.patch.object(r.w,'checkpoint',return_value=dict(
           complete=True,losses_finite=True,step=120,name='snapshot')), \
       mock.patch.object(r.w,'latest_metrics',return_value=dict(step=100)):
    with pytest.raises(AssertionError):
      r.stop_from_checkpoint(dict(logdir='unused',wandb_id='existing'),
                             dict(caught_up=True,step=100))
  proc.suspend.assert_called_once()
  proc.resume.assert_called_once()
  proc.terminate.assert_not_called()


def test_remote_pause_uses_supported_stop_action_on_existing_storage_id():
  responses = [io.BytesIO(json.dumps({'data': {'project': {'run': {
      'id': 'existing-storage-id', 'name': 'same-run', 'state': 'running'}}}}).encode()),
      io.BytesIO(b'{"data":{"stopRun":{"success":true}}}')]
  with mock.patch.object(r.w, 'credentials', return_value={
      'WANDB_API_KEY': 'test-only', 'WANDB_ENTITY': 'owner'}), \
       mock.patch.object(r.w, 'wandb_urlopen', side_effect=responses) as call:
    assert r.remote_pause({'wandb_id':'same-run'}) == 'stop_requested'
  payload = json.loads(call.call_args_list[1].args[0].data)
  assert 'stopRun' in payload['query']
  assert 'upsertBucket' not in payload['query']
  assert payload['variables'] == {'id':'existing-storage-id'}


def test_stop_rejects_checkpoint_behind_published_history():
  proc = mock.Mock()
  with mock.patch.object(r.w, 'owned', return_value=proc), \
       mock.patch.object(r.w, 'checkpoint', return_value=dict(
           complete=True, losses_finite=True, step=100)), \
       mock.patch.object(r.w, 'latest_metrics', return_value=dict(step=110)):
    assert r.stop_from_checkpoint({'logdir':'unused'}, dict(caught_up=True, step=110)) is None
  proc.suspend.assert_not_called()
  proc.terminate.assert_not_called()


def test_stop_rejects_unuploaded_history():
  proc = mock.Mock()
  with mock.patch.object(r.w, 'owned', return_value=proc), \
       mock.patch.object(r.w, 'checkpoint', return_value=dict(
           complete=True, losses_finite=True, step=120)), \
       mock.patch.object(r.w, 'latest_metrics', return_value=dict(step=110)):
    assert r.stop_from_checkpoint({'logdir':'unused'}, dict(caught_up=False, step=100)) is None
  proc.suspend.assert_not_called()


def test_history_advancing_during_freeze_check_resumes_without_stopping():
  proc = mock.Mock()
  with mock.patch.object(r.w, 'owned', return_value=proc), \
       mock.patch.object(r.w, 'checkpoint', return_value=dict(
           complete=True, losses_finite=True, step=110)), \
       mock.patch.object(r.w, 'latest_metrics', side_effect=[dict(step=100), dict(step=120)]):
    assert r.stop_from_checkpoint({'logdir':'unused'}, dict(caught_up=True, step=100)) is None
  proc.suspend.assert_called_once()
  proc.resume.assert_called_once()
  proc.terminate.assert_not_called()


def test_nonfinite_checkpoint_is_never_used_for_rollout():
  proc = mock.Mock()
  with mock.patch.object(r.w, 'owned', return_value=proc), \
       mock.patch.object(r.w, 'checkpoint', return_value=dict(
           complete=True, losses_finite=False, step=120)), \
       mock.patch.object(r.w, 'latest_metrics', return_value=dict(step=100)):
    assert r.stop_from_checkpoint({'logdir':'unused'}, dict(caught_up=True, step=100)) is None
  proc.suspend.assert_not_called()


@pytest.mark.parametrize('invalid', [None, 'reset_counter', 'nan_moment'])
def test_new_checkpoint_preserves_counters_and_finite_optimizer(tmp_path, invalid):
  folder = tmp_path/'ckpt/new'
  folder.mkdir(parents=True)
  params = {'model/kernel': np.ones((2, 2)),
            'opt_agent/moment': np.array([np.nan if invalid=='nan_moment' else .1])}
  (folder/'agent.pkl').write_bytes(pickle.dumps({'params':params}))
  job = dict(logdir=str(tmp_path), wandb_id='existing', setting='best_response_32',
      optimization_rollout={'before':{'checkpoint':{'name':'old', 'step':16,
      'updates':{'opt_wm/step/value':100, 'opt_agent/step/value':100}}}})
  cp = dict(complete=True, name='new', step=32, losses_finite=True,
      updates={'opt_wm/step/value':228,
               'opt_agent/step/value':0 if invalid=='reset_counter' else 228})
  with mock.patch.object(r.w, 'checkpoint', return_value=cp):
    if invalid:
      with pytest.raises(AssertionError):
        r.check_new_checkpoint(job)
    else:
      assert r.check_new_checkpoint(job)['all_parameters_finite']
