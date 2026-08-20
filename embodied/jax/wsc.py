"""Target-parameter mechanisms used by continual Dreamer runs.

The file keeps the historical module name because configs and training code
already route ``run.reset_mechanism`` through ``embodied.jax.wsc``.
"""

import math

import jax
import jax.numpy as jnp
import ninjax as nj

from . import reset_targets

f32 = jnp.float32
i32 = jnp.int32

CBP_STATE_NAMES = frozenset(('age', 'f', 'u'))


def metric_name(path):
  return path.replace('/', '_')


def module_name(path):
  return path.split('/', 1)[0].replace('/', '_')


def param_path(key):
  if key.startswith(('opt/', 'wsc/')):
    return None
  if any(key.endswith(f'/{name}') for name in CBP_STATE_NAMES):
    return None
  if '/' not in key:
    return None
  return key.rsplit('/', 1)[0]


def mechanism_param_items(params, target):
  target = reset_targets.canonical_target(target)
  items = {}
  for key, value in params.items():
    path = param_path(key)
    if path is None:
      continue
    if not reset_targets.matches_target(path, target):
      continue
    if not jnp.issubdtype(value.dtype, jnp.floating):
      continue
    items[key] = value
  return items


def layer_path(key):
  path = param_path(key)
  if path is None:
    return None
  name = key.rsplit('/', 1)[-1]
  return path if name in ('kernel', 'bias') else None


def following_rmsnorm_path(path):
  parent, _, leaf = path.rpartition('/')
  if leaf.startswith('linear') and leaf[len('linear'):].isdigit():
    return f'{parent}/norm{leaf[len("linear"):]}'
  if leaf.startswith(('mlp', 'cnn', 'obs', 'dynin', 'dynhid', 'prior')):
    suffix = ''
    if leaf.startswith('mlp') and leaf[3:].isdigit():
      suffix = 'norm'
    elif leaf.startswith('cnn') and leaf[3:].isdigit():
      suffix = 'norm'
    elif leaf.startswith('obs') and leaf[3:].isdigit():
      suffix = 'norm'
    elif leaf.startswith('dynin') and leaf[5:].isdigit():
      suffix = 'norm'
    elif leaf.startswith('dynhid') and leaf[6:].isdigit():
      suffix = 'norm'
    elif leaf.startswith('prior') and leaf[5:].isdigit():
      suffix = 'norm'
    if suffix:
      return f'{parent}/{leaf}{suffix}'
  if leaf == 'sp1':
    return f'{parent}/sp1norm'
  if leaf == 'space':
    return f'{parent}/spacenorm'
  if leaf.startswith('conv') and leaf[4:].isdigit():
    return f'{parent}/{leaf}norm'
  return None


def layer_norm(params, path):
  total = jnp.asarray(0, f32)
  for name in ('kernel', 'bias'):
    key = f'{path}/{name}'
    if key in params:
      total = total + jnp.square(f32(params[key])).sum()
  return jnp.sqrt(total)


def parse_mechanism(mechanism, default_norm_mode=None):
  del default_norm_mode
  lower = str(mechanism or 'disabled').lower()
  if lower in ('', 'none', 'false', 'disabled', 'off'):
    return False, 'disabled', 'disabled'
  aliases = {
      'hard': 'hard',
      'reset': 'disabled',
      'shrink_and_perturb': 'sandp',
      'shrink_and_perturb_without_optimizer': 'sandp_wo_opt',
      'disabled': 'disabled',
  }
  lower = aliases.get(lower, lower)
  if lower in (
      'hard', 'l2_decay', 'l2_decay_preupdate', 'l2_init',
      'continual_backprop', 'sandp', 'sandp_wo_opt'):
    return True, lower, lower
  raise ValueError(
      f'Unknown mechanism {mechanism!r}. Supported mechanisms are: '
      'disabled, l2_decay, l2_decay_preupdate, l2_init, '
      'continual_backprop, hard, sandp, sandp_wo_opt.')


def _target_snapshot(params, target):
  return {
      key: f32(value)
      for key, value in mechanism_param_items(params, target).items()}


def _tree_sum_squares(tree):
  if not tree:
    return jnp.asarray(0, f32)
  return jnp.stack([
      jnp.square(f32(value)).sum() for value in tree.values()]).sum()


def _fan_in(kernel):
  if kernel.ndim < 1:
    return 1
  return math.prod(kernel.shape[:-1])


def _sample_like(key, shape, dtype):
  std = math.sqrt(1.0 / max(math.prod(shape[:-1]), 1)) / 0.87962566103423978
  return (jax.random.truncated_normal(key, -2.0, 2.0, shape) * std).astype(dtype)


def _feature_axis_sum_abs(kernel, axis):
  axes = tuple(i for i in range(kernel.ndim) if i != axis)
  return jnp.abs(f32(kernel)).sum(axis=axes)


def _replace_output_axis(value, mask, sample):
  shape = (1,) * (value.ndim - 1) + mask.shape
  return jnp.where(mask.reshape(shape), sample, value)


def _zero_input_axis(value, mask):
  if value.ndim == 2:
    shape = mask.shape + (1,)
  else:
    shape = (1,) * (value.ndim - 2) + mask.shape + (1,)
  return jnp.where(mask.reshape(shape), jnp.zeros_like(value), value)


def _activation_score(activation):
  activation = f32(activation)
  axes = tuple(range(activation.ndim - 1))
  return jnp.abs(activation).mean(axis=axes)


def _cbp_state_init(params, activations):
  state = {}
  for path, act in activations.items():
    key = f'{path}/kernel'
    if key not in params:
      continue
    size = int(act.shape[-1])
    state[f'{path}/age'] = jnp.zeros((size,), i32)
    state[f'{path}/f'] = jnp.zeros((size,), f32)
    state[f'{path}/u'] = jnp.zeros((size,), f32)
  return state


class MechanismController(nj.Module):

  enabled: bool = False
  mechanism: str = 'disabled'
  target: str = 'all'
  weight_decay: float = 2e-5
  l2_init_weight: float = 2e-5
  cbp_eta: float = 0.99
  cbp_maturity: int = 5000
  cbp_replacement_rate: float = 1e-4
  cbp_eps: float = 1e-8

  def __init__(self):
    enabled, mode, _ = parse_mechanism(self.mechanism)
    self._active = bool(self.enabled or enabled)
    self._mode = mode
    self._target = reset_targets.canonical_target(self.target)

  @property
  def active(self):
    return self._active

  @property
  def parsed_scale_mode(self):
    return self._mode

  @property
  def parsed_norm_mode(self):
    return self._mode

  @property
  def parsed_target(self):
    return self._target

  @property
  def uses_output_scale(self):
    return False

  @property
  def needs_activations(self):
    return self.active and self._mode == 'continual_backprop'

  def zero_scale_grads(self, grads):
    return grads

  def init_params(self, params):
    if not self.active:
      return params, {}
    metrics = self._target_metrics(params)
    if self._mode == 'l2_init':
      refs = self.sub(
          'init_params', nj.Tree, lambda p: _target_snapshot(p, self._target),
          params)
      refs.write(refs.read())
      raw = self._l2_init_raw(params, refs.read())
      metrics['mechanism/l2_init/raw_loss'] = raw
      metrics['mechanism/l2_init/weighted_loss'] = raw * f32(self.l2_init_weight)
    return params, metrics

  def regularization_loss(self, params):
    if not self.active or self._mode != 'l2_init':
      return jnp.asarray(0, f32)
    refs = self.sub(
        'init_params', nj.Tree, lambda p: _target_snapshot(p, self._target),
        params)
    raw = self._l2_init_raw(params, refs.read())
    return raw * f32(self.l2_init_weight)

  def regularization_metrics(self, params):
    if not self.active or self._mode != 'l2_init':
      return {}
    refs = self.sub(
        'init_params', nj.Tree, lambda p: _target_snapshot(p, self._target),
        params)
    raw = self._l2_init_raw(params, refs.read())
    metrics = {
        'mechanism/l2_init/raw_loss': raw,
        'mechanism/l2_init/weighted_loss': raw * f32(self.l2_init_weight),
    }
    metrics.update(self._l2_init_module_delta_squares(params, refs.read()))
    return metrics

  def step(self, pre_params, post_params, outputs=None, step=None, lr=None):
    del pre_params, outputs, step, lr
    if not self.active or self._mode != 'l2_decay':
      return post_params, {}
    params = dict(post_params)
    factor = jnp.asarray(1.0 - self.weight_decay, f32)
    count = 0
    for key, value in mechanism_param_items(params, self._target).items():
      params[key] = value * factor.astype(value.dtype)
      count += math.prod(value.shape)
    metrics = self._target_metrics(params)
    metrics.update({
        'mechanism/l2_decay/factor': factor,
        'mechanism/l2_decay/weight_decay': jnp.asarray(self.weight_decay, f32),
        'mechanism/l2_decay/param_count': jnp.asarray(count, f32),
    })
    return params, metrics

  def preupdate_step(self, pre_params, post_params, outputs=None, step=None, lr=None):
    del outputs, step, lr
    if not self.active or self._mode != 'l2_decay_preupdate':
      return post_params, {}
    params = dict(post_params)
    factor = jnp.asarray(1.0 - self.weight_decay, f32)
    count = 0
    for key, pre_value in mechanism_param_items(pre_params, self._target).items():
      update = post_params[key] - pre_value
      params[key] = pre_value * factor.astype(pre_value.dtype) + update
      count += math.prod(pre_value.shape)
    metrics = self._target_metrics(params)
    metrics.update({
        'mechanism/l2_decay_preupdate/factor': factor,
        'mechanism/l2_decay_preupdate/weight_decay': jnp.asarray(
            self.weight_decay, f32),
        'mechanism/l2_decay_preupdate/param_count': jnp.asarray(count, f32),
    })
    return params, metrics

  def continual_backprop_step(self, activations):
    if not self.needs_activations or not activations:
      return {}
    ctx = nj.context()
    paths = [
        path for path in activations
        if f'{path}/kernel' in ctx
        and reset_targets.matches_target(path, self._target)]
    if len(paths) < 2:
      return {'mechanism/cbp/eligible_layers': jnp.asarray(len(paths), f32)}

    state_tree = self.sub(
        'cbp_state', nj.Tree, _cbp_state_init, ctx, activations)
    state = dict(state_tree.read())
    metrics = {'mechanism/cbp/eligible_layers': jnp.asarray(len(paths), f32)}
    eta = jnp.asarray(self.cbp_eta, f32)
    eps = jnp.asarray(self.cbp_eps, f32)

    for current, next_path in zip(paths[:-1], paths[1:]):
      cur_key = f'{current}/kernel'
      next_key = f'{next_path}/kernel'
      if cur_key not in ctx or next_key not in ctx:
        continue
      cur_kernel, next_kernel = ctx[cur_key], ctx[next_key]
      if cur_kernel.ndim not in (2, 4) or next_kernel.ndim not in (2, 4):
        continue
      if cur_kernel.shape[-1] != activations[current].shape[-1]:
        continue
      if next_kernel.ndim == 2 and next_kernel.shape[0] != cur_kernel.shape[-1]:
        continue
      if next_kernel.ndim == 4 and next_kernel.shape[-2] != cur_kernel.shape[-1]:
        continue

      size = int(cur_kernel.shape[-1])
      age_key, f_key, u_key = (
          f'{current}/age', f'{current}/f', f'{current}/u')
      if age_key not in state or state[age_key].shape[0] != size:
        state[age_key] = jnp.zeros((size,), i32)
        state[f_key] = jnp.zeros((size,), f32)
        state[u_key] = jnp.zeros((size,), f32)

      age = state[age_key] + 1
      h = _activation_score(activations[current])
      f_old = state[f_key]
      f_hat = f_old / (1 - jnp.power(eta, age.astype(f32)) + eps)
      f_new = eta * f_old + (1 - eta) * h
      pre_w = _feature_axis_sum_abs(cur_kernel, cur_kernel.ndim - 1) + eps
      post_axis = 0 if next_kernel.ndim == 2 else next_kernel.ndim - 2
      post_w = _feature_axis_sum_abs(next_kernel, post_axis)
      y = jnp.abs(h - f_hat) * post_w / pre_w
      u_new = eta * state[u_key] + (1 - eta) * y
      u_hat = u_new / (1 - jnp.power(eta, age.astype(f32)) + eps)
      eligible = age > self.cbp_maturity
      replace_prob = jnp.minimum(
          1.0, jnp.asarray(size * self.cbp_replacement_rate, f32))
      should_replace = (
          eligible.any() &
          (jax.random.uniform(nj.seed(), ()) < replace_prob))
      masked_utility = jnp.where(eligible, u_hat, jnp.inf)
      replace_index = jnp.argmin(masked_utility)
      mask = jnp.arange(size) == replace_index
      reset_mask = should_replace & mask

      sample = _sample_like(nj.seed(), cur_kernel.shape, cur_kernel.dtype)
      ctx[cur_key] = _replace_output_axis(cur_kernel, reset_mask, sample)
      if f'{current}/bias' in ctx:
        bias = ctx[f'{current}/bias']
        ctx[f'{current}/bias'] = jnp.where(reset_mask, jnp.zeros_like(bias), bias)
      ctx[next_key] = _zero_input_axis(next_kernel, reset_mask)
      state[age_key] = jnp.where(reset_mask, jnp.zeros_like(age), age)
      state[f_key] = jnp.where(reset_mask, jnp.zeros_like(f_new), f_new)
      state[u_key] = jnp.where(reset_mask, jnp.zeros_like(u_new), u_new)

      lname = metric_name(current)
      metrics[f'mechanism/cbp/replaced/{lname}'] = f32(should_replace)
      metrics[f'mechanism/cbp/reset_count_since_log/{lname}'] = (
          f32(reset_mask).sum())
      metrics[f'mechanism/cbp/min_utility/{lname}'] = jnp.min(masked_utility)
      metrics[f'mechanism/cbp/eligible/{lname}'] = f32(eligible).mean()
      metrics[f'mechanism/cbp/mean_age/{lname}'] = f32(age).mean()

    state_tree.write(state)
    return metrics

  def output_metrics(self, outputs):
    del outputs
    return {}

  def _l2_init_raw(self, params, refs):
    total = jnp.asarray(0, f32)
    for key, init_value in refs.items():
      if key not in params:
        continue
      diff = f32(params[key]) - f32(init_value)
      total = total + 0.5 * jnp.square(diff).sum()
    return total

  def _l2_init_module_delta_squares(self, params, refs):
    metrics = {}
    for key, init_value in refs.items():
      if key not in params:
        continue
      path = param_path(key)
      if path is None:
        continue
      name = module_name(path)
      metric = f'mechanism/l2_init/module_delta_sq/{name}'
      diff = f32(params[key]) - f32(init_value)
      metrics[metric] = metrics.get(metric, jnp.asarray(0, f32)) + (
          jnp.square(diff).sum())
    return metrics

  def _target_metrics(self, params):
    items = mechanism_param_items(params, self._target)
    count = sum(math.prod(value.shape) for value in items.values())
    norm = jnp.sqrt(_tree_sum_squares(items))
    return {
        'mechanism/active': jnp.asarray(float(self.active), f32),
        'mechanism/target_param_count': jnp.asarray(count, f32),
        'mechanism/target_param_l2': norm,
    }


def layer_grad_metrics(grads, updates):
  metrics = {}
  for key, grad in grads.items():
    path = layer_path(key)
    if path is None:
      continue
    lname = metric_name(path)
    pname = key.rsplit('/', 1)[-1]
    metrics[f'raw_grad_mean/{lname}/{pname}'] = jnp.abs(f32(grad)).mean()
    if key in updates:
      metrics[f'update_mean/{lname}/{pname}'] = jnp.abs(f32(updates[key])).mean()
  return metrics


# Backward-compatible name for existing imports and checkpoint scopes.
WSC = MechanismController
