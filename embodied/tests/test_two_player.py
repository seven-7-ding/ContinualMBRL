"""Behavioral checks: independent player updates, latest losses, and resume."""
from pathlib import Path
from copy import deepcopy

import elements
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ruamel.yaml import YAML

from dreamerv3.agent import Agent
from embodied.run.two_player import update_counts, update_modes


@pytest.fixture(autouse=True)
def restore_jax_transfer_guard():
  original = jax.config.jax_transfer_guard
  yield
  jax.config.update('jax_transfer_guard', original)


@pytest.mark.parametrize('k', [32, 128])
def test_schedule_counts(k):
  assert update_counts('default') == (1, 1)
  for setting, counts in ((f'wm_as_leader_{k}', (1, k)),
                          (f'agent_as_leader_{k}', (k, 1)),
                          (f'best_response_{k}', (k, k))):
    modes = update_modes(setting)
    assert (modes.count('wm'), modes.count('agent')) == counts
  for setting in ('bad', 'wm_as_leader_0', 'best_response_-1'):
    with pytest.raises(ValueError):
      update_counts(setting)


@pytest.mark.parametrize('group_size,sparse_metrics', [(1, False), (4, False), (4, True)])
def test_player_updates_freeze_opponent_and_resume(tmp_path, group_size, sparse_metrics):
  configs = YAML(typ='safe').load(Path('dreamerv3/configs.yaml').read_text())
  config = elements.Config(configs['defaults']).update(configs['size0.5m'])
  config = config.update({
      'logdir': str(tmp_path), 'batch_size': 2, 'batch_length': 4,
      'replay_context': 0, 'jax.platform': 'cpu', 'jax.precompile': False,
      'jax.profiler': False, 'jax.compute_dtype': 'float32',
      'agent.two_player': True, 'agent.imag_length': 3,
      'agent.opt.warmup': 0, 'agent.redo.frequency': 1,
      'agent.log_gradients': True})
  ac = elements.Config(**config.agent, logdir=config.logdir, seed=config.seed,
      jax=config.jax, batch_size=config.batch_size, batch_length=config.batch_length,
      report_length=config.report_length, replay_context=config.replay_context)
  obs = {'state': elements.Space(np.float32, (3,)),
         'is_first': elements.Space(bool), 'is_last': elements.Space(bool),
         'is_terminal': elements.Space(bool), 'reward': elements.Space(np.float32)}
  act = {'action': elements.Space(np.float32, (2,), -1, 1)}
  agent = Agent(obs, act, ac)
  agent.sparse_intermediate_metrics = sparse_metrics
  original_loss = agent.model.loss

  def probe_loss(*args, **kwargs):
    loss, (carry, entries, outs, metrics) = original_loss(*args, **kwargs)
    if 'imgact' in outs:
      action = outs['imgact']['action'].reshape(-1)
      metrics = dict(metrics, **{
          'debug/action0': action[0], 'debug/action1': action[1],
          'debug/action_sum': action.sum(),
          'debug/adv_mean': metrics.get('advantage_mean', jnp.array(0.0))})
    return loss, (carry, entries, outs, metrics)

  agent.model.loss = probe_loss
  carry = agent.init_train(2)
  rng = np.random.default_rng(0)

  def batch():
    data = agent._zeros(agent.spaces, (2, 4))
    data['state'] = rng.normal(size=(2, 4, 3)).astype(np.float32)
    data['reward'] = rng.normal(size=(2, 4)).astype(np.float32)
    with jax._src.config.explicit_device_get_scope():
      data = jax.device_put(data, agent.train_sharded)
    return dict(data, seed=agent._seeds(int(agent.n_updates), agent.train_mirrored))

  def snapshot():
    return {k: np.array(v) for k, v in jax.device_get(agent.params).items()
            if k.startswith(('enc/', 'dyn/', 'dec/', 'rew/', 'con/',
                             'pol/', 'val/', 'slowval/', 'opt_wm/', 'opt_agent/'))}

  before = snapshot()
  carry, _, wm = agent.train(carry, batch(), train_mode='wm')
  after_wm = snapshot()
  assert 'loss/state' in wm and 'loss/policy' not in wm
  for k in before:
    if k.startswith(('pol/', 'val/', 'slowval/', 'opt_agent/')):
      np.testing.assert_array_equal(before[k], after_wm[k], err_msg=k)
  assert any(not np.array_equal(before[k], after_wm[k]) for k in before if k.startswith('dyn/'))

  # Three independent sampled batches and imagination calls, no cached rollout.
  for _ in range(3):
    carry, _, actor = agent.train(carry, batch(), train_mode='agent')
  after_agent = snapshot()
  assert 'loss/policy' in actor and 'loss/state' not in actor
  assert 'dyn_ent' not in actor and 'dyn_ent' in wm
  assert all(k.rsplit('/', 1)[-1].startswith(('pol_', 'val_'))
             for k in actor if k.startswith('act_redo/'))
  assert all(np.isfinite(actor[k]) for k in ('loss/policy', 'loss/value', 'loss/repval'))
  for k in after_wm:
    if k.startswith(('enc/', 'dyn/', 'dec/', 'rew/', 'con/', 'opt_wm/')):
      np.testing.assert_array_equal(after_wm[k], after_agent[k], err_msg=k)
  assert any(not np.array_equal(after_wm[k], after_agent[k]) for k in after_wm if k.startswith('pol/'))
  assert actor['opt/agent/updates'] > wm['opt/wm/updates']
  agent.sync_policy()
  for key in agent.policy_keys:
    np.testing.assert_array_equal(
        jax.device_get(agent.policy_params[key]), jax.device_get(agent.params[key]))
  saved = agent.save()
  carry, _, _ = agent.train(carry, batch(), train_mode='wm')
  agent.load(saved)
  restored = snapshot()
  for k in after_agent:
    np.testing.assert_array_equal(after_agent[k], restored[k], err_msg=k)

  for mode in ('wm', 'agent'):
    initial = deepcopy(agent.save())
    initial_carry = carry
    batches = [batch() for _ in range(group_size)]
    for index, data in enumerate(batches):
      data['seed'] = agent._seeds(100 + index, agent.train_mirrored)
    history = []
    for data in batches:
      carry, _, sequential_metrics = agent.train(carry, dict(data), train_mode=mode)
      history.append({k: float(v) for k, v in sequential_metrics.items() if k.startswith(('loss/', 'debug/'))})
    sequential = deepcopy(agent.save())
    # CPU device_put may alias NumPy checkpoint arrays; simulate independent
    # deserialization for each branch, as a real disk restore does.
    agent.load(deepcopy(initial))
    agent.debug_train_groups = True
    carry, outs = agent.train_many(initial_carry, batches, mode)
    fused = agent.save()
    import json
    differences = {key: float(np.max(np.abs(sequential['params'][key] - fused['params'][key])))
                   for key in sequential['params']}
    (tmp_path / f'{mode}_fused_differences.json').write_text(json.dumps(
        dict(sorted(differences.items(), key=lambda item: -item[1])), indent=2))
    latest = agent.latest_train_metrics()
    if "debug_metrics" in outs:
      keys = agent._group_metric_keys[mode]
      history = [{k: [v, float(outs["debug_metrics"][i, keys.index(k)])] for k,v in row.items()} for i,row in enumerate(history)]
      (tmp_path / f"{mode}_per_step_losses.json").write_text(json.dumps(history, indent=2))
    (tmp_path / f'{mode}_fused_losses.json').write_text(json.dumps({
        key: [float(value), float(latest[key])] for key, value in sequential_metrics.items()
        if key.startswith('loss/')}))
    for key in sequential['params']:
      np.testing.assert_allclose(
          sequential['params'][key], fused['params'][key], rtol=2e-4, atol=2e-5,
          err_msg=f'{mode}: {key}')
    for key in ('loss/state',) if mode == 'wm' else ('loss/policy', 'loss/value', 'loss/repval'):
      np.testing.assert_allclose(latest[key], sequential_metrics[key], rtol=2e-4, atol=2e-5)
    checkpoint = deepcopy(agent.save())
    agent.load(checkpoint)
    for key, value in latest.items():
      np.testing.assert_equal(agent.latest_train_metrics()[key], value)


def test_default_fused_update_matches_joint_training(tmp_path):
  configs = YAML(typ='safe').load(Path('dreamerv3/configs.yaml').read_text())
  config = elements.Config(configs['defaults']).update(configs['size0.5m'])
  config = config.update({
      'logdir': str(tmp_path), 'batch_size': 2, 'batch_length': 4,
      'replay_context': 0, 'jax.platform': 'cpu', 'jax.precompile': False,
      'jax.profiler': False, 'jax.compute_dtype': 'float32',
      'agent.imag_length': 3, 'agent.redo.frequency': 1})
  ac = elements.Config(**config.agent, logdir=config.logdir, seed=config.seed,
      jax=config.jax, batch_size=2, batch_length=4, report_length=4,
      replay_context=0)
  obs = {'state': elements.Space(np.float32, (3,)),
         'is_first': elements.Space(bool), 'is_last': elements.Space(bool),
         'is_terminal': elements.Space(bool), 'reward': elements.Space(np.float32)}
  agent = Agent(obs, {'action': elements.Space(np.float32, (2,), -1, 1)}, ac)
  carry = agent.init_train(2)
  data = agent._zeros(agent.spaces, (2, 4))
  data['state'] = np.random.default_rng(1).normal(size=(2, 4, 3)).astype(np.float32)
  with jax._src.config.explicit_device_get_scope():
    data = jax.device_put(data, agent.train_sharded)
  data['seed'] = agent._seeds(5, agent.train_mirrored)
  initial = deepcopy(agent.save())
  _, _, metrics = agent.train(carry, dict(data), latest=True)
  sequential = deepcopy(agent.save())
  agent.load(deepcopy(initial))
  agent.train_many(carry, [data], 'all')
  fused = agent.save()
  for key in sequential['params']:
    np.testing.assert_allclose(sequential['params'][key], fused['params'][key],
                               rtol=2e-4, atol=2e-5, err_msg=key)
  latest = agent.latest_train_metrics()
  for key in ('loss/state', 'loss/policy', 'loss/value', 'loss/repval'):
    np.testing.assert_allclose(latest[key], metrics[key], rtol=2e-4, atol=2e-5)


def test_resource_diagnostics_handle_dead_weak_references():
  import gc
  import tracemalloc
  import weakref
  from embodied.run.diagnostics import make_usage

  class Probe:
    pass

  obj = Probe()
  proxy = weakref.proxy(obj)
  del obj
  was_tracing = tracemalloc.is_tracing()
  usage = make_usage({'gc': True, 'malloc': True})
  try:
    gc.collect()
    result = usage.tools['gc'](log=False)
    assert result['gen2/collected/sum'] >= 0
    assert result['gen2/duration/sum'] > 0
    assert result['objcounts']
    assert usage.tools['malloc'](log=False)['full']
    assert type(proxy).__name__ == 'ProxyType'
  finally:
    gc.callbacks.remove(usage.tools['gc']._callback)
    if not was_tracing:
      tracemalloc.stop()
