import jax.numpy as jnp
import ninjax as nj

from embodied.jax import FineGrainedReDo
from embodied.jax import wsc


def test_l2_decay_scales_all_target_params_only():
  params = {
      'enc/mlp0/kernel': jnp.ones((2, 2), jnp.float32),
      'enc/mlp0/bias': jnp.ones((2,), jnp.float32),
      'other/kernel': jnp.ones((2, 2), jnp.float32),
  }
  controller = wsc.MechanismController(
      enabled=True,
      mechanism='l2_decay',
      target='all',
      weight_decay=0.1,
      name='wsc')

  new_params, metrics = controller.step(params, params)

  assert jnp.allclose(new_params['enc/mlp0/kernel'], 0.9)
  assert jnp.allclose(new_params['enc/mlp0/bias'], 0.9)
  assert jnp.allclose(new_params['other/kernel'], 1.0)
  assert metrics['mechanism/l2_decay/factor'] == jnp.asarray(0.9, jnp.float32)


def test_l2_decay_preupdate_scales_before_applying_update():
  pre_params = {
      'enc/mlp0/kernel': jnp.asarray([10.0, 20.0], jnp.float32),
      'other/kernel': jnp.asarray([10.0], jnp.float32),
  }
  updates = {
      'enc/mlp0/kernel': jnp.asarray([1.0, -2.0], jnp.float32),
      'other/kernel': jnp.asarray([3.0], jnp.float32),
  }
  post_params = {
      key: pre_params[key] + updates[key]
      for key in pre_params}
  controller = wsc.MechanismController(
      enabled=True,
      mechanism='l2_decay_preupdate',
      target='all',
      weight_decay=0.1,
      name='wsc')

  new_params, metrics = controller.preupdate_step(pre_params, post_params)

  assert jnp.allclose(
      new_params['enc/mlp0/kernel'], jnp.asarray([10.0, 16.0], jnp.float32))
  assert jnp.allclose(
      new_params['other/kernel'], jnp.asarray([13.0], jnp.float32))
  assert metrics['mechanism/l2_decay_preupdate/factor'] == jnp.asarray(
      0.9, jnp.float32)


def test_l2_init_regularizes_to_initial_target_params():

  class Probe(nj.Module):

    def __call__(self, params):
      controller = wsc.MechanismController(
          enabled=True,
          mechanism='l2_init',
          target='all',
          l2_init_weight=0.5,
          name='wsc')
      return controller.regularization_loss(params)

  params0 = {'enc/kernel': jnp.ones((2,), jnp.float32)}
  params1 = {'enc/kernel': jnp.asarray([3.0, 5.0], jnp.float32)}
  state, loss0 = nj.pure(Probe(name='probe'))({}, params0, create=True)
  _, loss1 = nj.pure(Probe(name='probe'))(state, params1)

  assert loss0 == 0.0
  assert jnp.allclose(loss1, 0.5 * 0.5 * ((3 - 1) ** 2 + (5 - 1) ** 2))


def test_l2_init_reports_module_delta_squares():

  class Probe(nj.Module):

    def __call__(self, params):
      controller = wsc.MechanismController(
          enabled=True,
          mechanism='l2_init',
          target='all',
          l2_init_weight=0.5,
          name='wsc')
      return controller.regularization_metrics(params)

  params0 = {
      'enc/a/kernel': jnp.ones((2,), jnp.float32),
      'enc/b/kernel': jnp.ones((1,), jnp.float32),
      'dyn/c/kernel': jnp.ones((1,), jnp.float32),
  }
  params1 = {
      'enc/a/kernel': jnp.asarray([2.0, 3.0], jnp.float32),
      'enc/b/kernel': jnp.asarray([4.0], jnp.float32),
      'dyn/c/kernel': jnp.asarray([5.0], jnp.float32),
  }
  state, _ = nj.pure(Probe(name='probe'))({}, params0, create=True)
  _, metrics = nj.pure(Probe(name='probe'))(state, params1)

  assert jnp.allclose(
      metrics['mechanism/l2_init/module_delta_sq/enc'],
      (2 - 1) ** 2 + (3 - 1) ** 2 + (4 - 1) ** 2)
  assert jnp.allclose(
      metrics['mechanism/l2_init/module_delta_sq/dyn'],
      (5 - 1) ** 2)


def test_continual_backprop_resets_mature_low_utility_unit():

  class Probe(nj.Module):

    def __call__(self, params, activations):
      nj.context().update(params)
      controller = wsc.MechanismController(
          enabled=True,
          mechanism='continual_backprop',
          target='all',
          cbp_maturity=0,
          cbp_replacement_rate=1.0,
          name='wsc')
      metrics = controller.continual_backprop_step(activations)
      return {k: nj.context()[k] for k in params}, metrics

  params = {
      'enc/a/kernel': jnp.ones((2, 3), jnp.float32),
      'enc/a/bias': jnp.ones((3,), jnp.float32),
      'enc/b/kernel': jnp.ones((3, 2), jnp.float32),
  }
  activations = {
      'enc/a': jnp.asarray([[0.0, 1.0, 2.0]], jnp.float32),
      'enc/b': jnp.asarray([[1.0, 1.0]], jnp.float32),
  }
  _, (new_params, metrics) = nj.pure(Probe(name='probe'))(
      {}, params, activations, seed=jnp.asarray([0, 0], jnp.uint32),
      create=True)

  assert metrics['mechanism/cbp/replaced/enc_a'] == 1.0
  assert metrics['mechanism/cbp/reset_count_since_log/enc_a'] == 1.0
  assert jnp.any(new_params['enc/a/kernel'] != params['enc/a/kernel'])
  assert jnp.any(jnp.all(new_params['enc/b/kernel'] == 0.0, axis=1))


def test_redo_reports_preactivation_and_variation_rank_metrics():

  class Probe(nj.Module):

    def __call__(self, params, activations):
      nj.context().update(params)
      redo = FineGrainedReDo.FGReDo(
          frequency=1, log_item='log', name='act_redo')
      return redo.step(activations)

  params = {
      'agent/rew/linear0/kernel': jnp.ones((2, 3), jnp.float32),
      'agent/rew/norm0/scale': jnp.ones((3,), jnp.float32),
  }
  activations = {
      'agent/rew/linear0': jnp.asarray(
          [[1.0, 2.0, 2.0], [1.0, 4.0, 6.0], [1.0, 6.0, 10.0]],
          jnp.float32),
      'agent/rew/norm0': jnp.asarray(
          [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0]],
          jnp.float32),
  }

  _, metrics = nj.pure(Probe(name='probe'))(
      {}, params, activations, create=True)
  lname = 'agent_rew_linear0'

  assert jnp.allclose(
      metrics[f'act_redo/Zombie_Percentage/{lname}'], 100 / 3)
  assert jnp.allclose(
      metrics[f'act_redo/Saturation_Percentage/{lname}'], 200 / 3)
  assert metrics[f'act_redo/Variation_Rank_0.9/{lname}'] == 2.0
  assert metrics[f'act_redo/Variation_Rank_0.95/{lname}'] == 2.0
  assert metrics[f'act_redo/Variation_Rank_0.99/{lname}'] == 2.0
