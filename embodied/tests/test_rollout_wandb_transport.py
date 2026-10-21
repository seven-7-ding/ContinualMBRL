from unittest import mock

from auto_scripts import rollout_wandb_transport as r


def test_unavailable_canary_cannot_be_bypassed():
  canary = dict(wandb_id=r.CANARY)
  other = dict(wandb_id='other')
  assert r.pending_candidates([other, canary]) == [canary]
  for state in ('stopped', 'started'):
    canary[r.FIELD] = dict(state=state)
    assert r.pending_candidates([other, canary]) == []
  canary[r.FIELD] = dict(state='verified')
  assert r.pending_candidates([other, canary]) == [other]


def test_ready_requires_both_proxy_exclusions_and_real_compiled_progress():
  job = {r.FIELD: dict(before=dict(checkpoint=dict(step=100)))}
  proc = mock.Mock()
  env = dict(CONTINUALMBRL_WANDB_TRANSPORT='direct_api_v1',
             NO_PROXY='localhost,api.wandb.ai', no_proxy='api.wandb.ai')
  proc.environ.return_value = env
  job['logdir'] = 'private'
  with mock.patch.object(r.w, 'owned', return_value=proc), \
       mock.patch.object(r.stop, 'protocol_ready', return_value=True), \
       mock.patch.object(r.w, 'read_json', return_value=101) as read:
    assert r.compiled_ready(job)
    read.return_value = 100
    assert not r.compiled_ready(job)
    read.return_value = 101
    env['no_proxy'] = 'localhost'
    assert not r.compiled_ready(job)


def test_native_cloud_lag_prevents_continuation_verification():
  job = {r.FIELD: dict(before=dict(checkpoint=dict(step=100))), 'logdir': 'private'}
  with mock.patch.object(r, 'compiled_ready', return_value=True), \
       mock.patch.object(r.stop, 'checkpoint_proof', return_value=(dict(step=200), None)), \
       mock.patch.object(r.w, 'latest_metrics', return_value=dict(step=200, losses_finite=True)), \
       mock.patch.object(r.w, 'event') as event:
    assert not r.verify_continuation(job, dict(caught_up=False, losses_finite=True, state='running'))
  event.assert_not_called()
