import io
import json
from unittest import mock

import pytest

from auto_scripts import experiment_watchdog as w


@pytest.mark.parametrize('local_step,finite,cleared', [
    (1000, True, True), (2000, True, False), (1000, False, False)])
def test_only_verified_catchup_clears_previous_continuous_lag(tmp_path, local_step, finite, cleared):
  runid = 'same-run'
  marker = tmp_path/('remote_lag_'+runid+'.json')
  marker.write_text('{"since":1}')
  summary = dict(env_step=1000, **{'loss/'+k: 1.0 for k in w.LOSSES})
  if not finite:
    summary['loss/policy'] = 'invalid'
  response = dict(data=dict(project=dict(runs=dict(edges=[dict(node=dict(
      id='same-storage-id', name=runid, state='running',
      summaryMetrics=json.dumps(summary), stopped=False))]))))
  with mock.patch.object(w, 'CONTROL', tmp_path), \
       mock.patch.object(w, 'credentials', return_value=dict(WANDB_API_KEY='test-only', WANDB_ENTITY='owner')), \
       mock.patch.object(w, 'latest_metrics', return_value=dict(step=local_step)), \
       mock.patch.object(w, 'wandb_urlopen', return_value=io.BytesIO(json.dumps(response).encode())):
    report = w.remote_audit([dict(wandb_id=runid, logdir='private')])
  assert marker.exists() is not cleared
  assert report['runs'][0]['caught_up'] == (local_step <= 1000)
