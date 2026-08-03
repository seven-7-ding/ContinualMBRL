import jax.numpy as jnp
import ninjax as nj

from embodied.jax import FineGrainedReDo as redo
from embodied.jax import wsc


def test_skip_last_layer_constant_controls_only_rmsnorm_followed_layers():
  params = {
      'enc/mlp0/kernel': jnp.ones((2, 2), jnp.float32),
      'enc/mlp0/bias': jnp.ones((2,), jnp.float32),
      'enc/mlp0norm/scale': jnp.ones((2,), jnp.float32),
      'enc/head/kernel': jnp.ones((2, 2), jnp.float32) * 2,
      'enc/head/bias': jnp.ones((2,), jnp.float32) * 2,
  }
  controller = wsc.WSC(
      enabled=True,
      mechanism='wsc_skip_last_layer_constant',
      target='all',
      target_norm=1.0,
      name='wsc')

  new_params, metrics = controller.step(params, params)

  assert controller.parsed_scale_mode == 'no_scale'
  assert not controller.uses_output_scale
  assert jnp.allclose(wsc.layer_norm(new_params, 'enc/mlp0'), 1.0)
  assert jnp.allclose(
      wsc.layer_norm(new_params, 'enc/head'),
      wsc.layer_norm(params, 'enc/head'))
  assert metrics['wsc/controlled/enc_mlp0'] == 1.0
  assert metrics['wsc/controlled/enc_head'] == 0.0
  assert 'wsc/pre_norm/enc_mlp0' in metrics
  assert 'wsc/pre_norm/enc_head' in metrics
  assert 'wsc/scale/enc_head' not in metrics


def test_regular_constant_still_controls_target_layers_without_rmsnorm():
  params = {
      'enc/head/kernel': jnp.ones((2, 2), jnp.float32) * 2,
      'enc/head/bias': jnp.ones((2,), jnp.float32) * 2,
  }
  controller = wsc.WSC(
      enabled=True,
      mechanism='WSC_grad_scale_constant',
      target='all',
      target_norm=1.0,
      name='wsc')

  new_params, metrics = controller.step(params, params)

  assert jnp.allclose(wsc.layer_norm(new_params, 'enc/head'), 1.0)
  assert 'wsc/pre_norm/enc_head' in metrics


def test_legacy_non_wsc_mechanisms_are_parser_compatible_disabled_modes():
  for mechanism in (
      'l2_init', 'continual_backprop', 'l2_decay_preupdate', 'no_wsc'):
    enabled, scale_mode, norm_mode = wsc.parse_mechanism(mechanism)
    assert not enabled
    assert scale_mode == 'nograd'
    assert norm_mode == 'init'


def test_last_l2_init_adds_penalty_to_only_skipped_layers():
  init_params = {
      'enc/mlp0/kernel': jnp.ones((2, 2), jnp.float32),
      'enc/mlp0/bias': jnp.ones((2,), jnp.float32),
      'enc/mlp0norm/scale': jnp.ones((2,), jnp.float32),
      'enc/head/kernel': jnp.ones((2, 2), jnp.float32),
      'enc/head/bias': jnp.ones((2,), jnp.float32),
  }
  params = {
      **init_params,
      'enc/head/kernel': jnp.ones((2, 2), jnp.float32) * 3,
      'enc/head/bias': jnp.ones((2,), jnp.float32) * 2,
  }
  grads = {key: jnp.zeros_like(value) for key, value in params.items()}
  controller = wsc.WSC(
      enabled=True,
      mechanism='wsc_constant_last_l2_init_2e-5_all',
      target='all',
      target_norm=1.0,
      name='wsc')

  assert controller.parsed_scale_mode == 'no_scale'
  assert controller.parsed_norm_mode == 'constant'
  assert controller.skip_last_layer
  assert controller.last_l2_init
  assert controller.parsed_last_l2_init_weight_decay == 2e-5

  def fn():
    new_params, init_metrics = controller.init_params(init_params)
    new_grads, metrics = controller.add_l2_init_grads(params, grads)
    return new_params, init_metrics, new_grads, metrics

  _state, (new_params, init_metrics, new_grads, metrics) = nj.pure(fn)(
      {}, create=True)

  assert jnp.allclose(wsc.layer_norm(new_params, 'enc/mlp0'), 1.0)
  assert init_metrics['wsc/init_controlled/enc_mlp0'] == 1.0
  assert init_metrics['wsc/init_controlled/enc_head'] == 0.0
  assert jnp.allclose(new_grads['enc/mlp0/kernel'], 0.0)
  assert jnp.allclose(new_grads['enc/head/kernel'], 2e-5 * 2.0)
  assert jnp.allclose(new_grads['enc/head/bias'], 2e-5 * 1.0)
  assert 'wsc/last_l2_init_loss/enc_head' in metrics
  assert 'wsc/last_l2_init_loss/enc_mlp0' not in metrics


def test_redo_activation_diagnostics_from_preactivation_and_silu_output():
  preactivation = jnp.asarray([
      [1.0, 0.0, -1.0],
      [2.0, 0.5, -2.0],
      [3.0, 1.5, -3.0],
  ], jnp.float32)
  activation = jnp.asarray([
      [-3.0, -1.0, -0.1],
      [-1.0, -1.0, -0.1],
      [1.0, 1.0, 0.1],
      [3.0, 1.0, 0.1],
  ], jnp.float32)

  assert jnp.allclose(redo._zombie_percentage(preactivation), 100 / 3)
  assert jnp.allclose(redo._saturation_percentage(preactivation), 200 / 3)
  assert jnp.allclose(redo._variation_rank(activation, 0.9), 2.0)
  assert jnp.allclose(redo._variation_rank(activation, 0.95), 2.0)
  assert jnp.allclose(redo._variation_rank(activation, 0.99), 2.0)
  assert jnp.allclose(
      redo._variation_rank(jnp.ones((4, 3), jnp.float32), 0.99),
      0.0)


def test_redo_rank_singular_values_use_smaller_gram_matrix():
  wide = jnp.asarray([
      [3.0, 0.0, 0.0, 0.0],
      [0.0, 2.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
  ], jnp.float32)
  tall = wide.T

  assert jnp.allclose(
      redo._singular_values_for_rank(wide),
      jnp.linalg.svd(wide, compute_uv=False),
      rtol=1e-5,
      atol=1e-5)
  assert jnp.allclose(
      redo._singular_values_for_rank(tall),
      jnp.linalg.svd(tall, compute_uv=False),
      rtol=1e-5,
      atol=1e-5)
