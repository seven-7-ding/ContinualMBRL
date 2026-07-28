import jax.numpy as jnp

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
