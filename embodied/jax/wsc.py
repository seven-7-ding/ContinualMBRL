import math

import jax.numpy as jnp
import ninjax as nj

from . import reset_targets

f32 = jnp.float32

LAYER_PARAM_NAMES = frozenset(('kernel', 'bias'))
SCALE_PARAM_NAME = 'wsc_scale'


def metric_name(path):
  return path.replace('/', '_')


def layer_path(key):
  parts = key.split('/')
  if len(parts) < 2 or parts[-1] not in LAYER_PARAM_NAMES:
    return None
  return '/'.join(parts[:-1])


def scale_key(path):
  return f'{path}/{SCALE_PARAM_NAME}'


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


def has_following_rmsnorm(path, params):
  norm = following_rmsnorm_path(path)
  return bool(norm and f'{norm}/scale' in params)


def layer_norm(params, path):
  total = jnp.asarray(0, f32)
  for name in LAYER_PARAM_NAMES:
    key = f'{path}/{name}'
    if key in params:
      total = total + jnp.square(f32(params[key])).sum()
  return jnp.sqrt(total)


def layer_param_count(params, path):
  total = 0
  for name in LAYER_PARAM_NAMES:
    key = f'{path}/{name}'
    if key in params:
      total += math.prod(params[key].shape)
  return total


def layer_groups(params, target, require_following_rmsnorm=False):
  groups = {}
  for key in params:
    path = layer_path(key)
    if path is None:
      continue
    if not reset_targets.matches_target(path, target):
      continue
    if require_following_rmsnorm and not has_following_rmsnorm(path, params):
      continue
    groups.setdefault(path, []).append(key)
  return groups


def initial_layer_norms(params, target, eps=1e-8, require_following_rmsnorm=False):
  norms = {}
  for path in layer_groups(params, target, require_following_rmsnorm):
    norm = layer_norm(params, path)
    norms[path] = jnp.where(
        norm <= jnp.asarray(eps, f32), jnp.asarray(1.0, f32), norm)
  return norms


def sanitize_init_target_norms(norms, eps=1e-8):
  eps = jnp.asarray(eps, f32)
  return {
      path: jnp.where(
          f32(norm) <= eps, jnp.asarray(1.0, f32), f32(norm))
      for path, norm in norms.items()}


def output_l2_mean(value):
  value = f32(value)
  value = value.reshape((-1, value.shape[-1]))
  return jnp.linalg.norm(value, axis=-1).mean()


def parse_mechanism(mechanism, default_norm_mode='init'):
  mechanism = str(mechanism or 'disabled')
  lower = mechanism.lower()
  if lower in ('', 'none', 'false', 'disabled', 'off'):
    return False, 'nograd', default_norm_mode
  if not lower.startswith('wsc'):
    raise ValueError(f'Unknown WSC mechanism: {mechanism}')
  if is_skip_last_layer_mechanism(lower) or 'no_scale' in lower or 'noscale' in lower:
    scale_mode = 'no_scale'
  else:
    scale_mode = 'grad' if 'grad_scale' in lower and 'nograd' not in lower else 'nograd'
  norm_mode = default_norm_mode
  if any(x in lower for x in ('lr', 'learning_rate')):
    norm_mode = 'lr'
  elif any(x in lower for x in ('factor', 'scale_factor', 'fixed_c')):
    norm_mode = 'factor'
  elif any(x in lower for x in ('constant', 'target_norm', 'fixed_norm')):
    norm_mode = 'constant'
  elif 'init' in lower:
    norm_mode = 'init'
  return True, scale_mode, norm_mode


def is_skip_last_layer_mechanism(mechanism):
  return 'skip_last_layer' in str(mechanism or '').lower()


class WSC(nj.Module):

  enabled: bool = False
  mechanism: str = 'disabled'
  target: str = 'all'
  norm_mode: str = 'init'
  target_norm: float = 1.0
  scale_factor: float = 0.999
  eps: float = 1e-8
  factor_min: float = 0.01
  factor_max: float = 100.0
  scale_min: float = 1e-4
  scale_max: float = 1e4
  nograd_start_step: int = 10000
  scale_adjust_min: float = 0.1
  scale_adjust_max: float = 10.0

  def __init__(self):
    enabled, scale_mode, norm_mode = parse_mechanism(
        self.mechanism, self.norm_mode)
    self._active = bool(self.enabled or enabled)
    self._scale_mode = scale_mode
    self._norm_mode = norm_mode
    self._target = reset_targets.canonical_target(self.target)
    self._skip_last_layer = is_skip_last_layer_mechanism(self.mechanism)

  @property
  def active(self):
    return self._active

  @property
  def parsed_scale_mode(self):
    return self._scale_mode

  @property
  def parsed_norm_mode(self):
    return self._norm_mode

  @property
  def parsed_target(self):
    return self._target

  @property
  def skip_last_layer(self):
    return self._skip_last_layer

  @property
  def nograd_scale(self):
    return self.active and self.parsed_scale_mode == 'nograd'

  @property
  def uses_output_scale(self):
    return self.active and self.parsed_scale_mode in ('grad', 'nograd')

  def _all_layer_groups(self, params):
    return layer_groups(params, self.parsed_target)

  def _controlled_layer_groups(self, params):
    return layer_groups(
        params, self.parsed_target,
        require_following_rmsnorm=self.skip_last_layer)

  def zero_scale_grads(self, grads):
    if not self.active or self.parsed_scale_mode not in ('nograd', 'no_scale'):
      return grads
    return {
        key: jnp.zeros_like(value) if key.endswith('/' + SCALE_PARAM_NAME)
        else value
        for key, value in grads.items()}

  def init_params(self, params):
    if not self.active:
      return params, {}
    if self.parsed_norm_mode not in ('constant', 'init'):
      return params, {}
    all_groups = self._all_layer_groups(params)
    if not all_groups:
      return params, {}
    groups = self._controlled_layer_groups(params)
    new_params = dict(params)
    metrics = {}
    if self.parsed_norm_mode == 'init':
      target_tree = self.sub(
          'target_norms', nj.Tree,
          lambda params: initial_layer_norms(
              params, self.parsed_target, self.eps, self.skip_last_layer),
          params)
      target_norms = sanitize_init_target_norms(target_tree.read(), self.eps)
      target_tree.write(target_norms)
    else:
      target_norms = {}
    for path, keys in groups.items():
      norm = layer_norm(new_params, path)
      if self.parsed_norm_mode == 'constant':
        target = jnp.asarray(self.target_norm, f32)
      else:
        target = f32(target_norms.get(path, norm))
      factor = target / jnp.maximum(norm, jnp.asarray(self.eps, f32))
      factor = jnp.where(jnp.isfinite(factor), factor, jnp.asarray(1.0, f32))
      factor = jnp.clip(
          factor, jnp.asarray(self.factor_min, f32),
          jnp.asarray(self.factor_max, f32))
      for key in keys:
        new_params[key] = new_params[key] * factor.astype(new_params[key].dtype)
      lname = metric_name(path)
      metrics[f'wsc/init_factor/{lname}'] = f32(factor)
      metrics[f'wsc/init_pre_norm/{lname}'] = f32(norm)
      metrics[f'wsc/init_target_norm/{lname}'] = f32(target)
      if self.skip_last_layer:
        metrics[f'wsc/init_controlled/{lname}'] = jnp.asarray(1.0, f32)
    if self.skip_last_layer:
      for path in all_groups:
        if path in groups:
          continue
        lname = metric_name(path)
        metrics[f'wsc/init_factor/{lname}'] = jnp.asarray(1.0, f32)
        metrics[f'wsc/init_pre_norm/{lname}'] = f32(layer_norm(new_params, path))
        metrics[f'wsc/init_target_norm/{lname}'] = jnp.asarray(jnp.nan, f32)
        metrics[f'wsc/init_controlled/{lname}'] = jnp.asarray(0.0, f32)
    return new_params, metrics

  def step(self, pre_params, post_params, outputs=None, step=None, lr=None):
    if not self.active:
      return post_params, {}
    outputs = outputs or {}
    all_groups = self._all_layer_groups(post_params)
    if not all_groups:
      return post_params, {}
    groups = self._controlled_layer_groups(post_params)
    wsc_started = jnp.asarray(True)
    if self.nograd_scale and step is not None:
      wsc_started = jnp.asarray(step >= self.nograd_start_step)

    target_tree = None
    if self.parsed_norm_mode == 'init':
      target_tree = self.sub(
          'target_norms', nj.Tree,
          lambda params: initial_layer_norms(
              params, self.parsed_target, self.eps, self.skip_last_layer),
          pre_params)
      target_norms = sanitize_init_target_norms(target_tree.read(), self.eps)
      target_tree.write(target_norms)
    else:
      target_norms = {}

    params = dict(post_params)
    metrics = {}
    for path, keys in groups.items():
      norm = layer_norm(params, path)
      if self.parsed_norm_mode == 'factor':
        lr_value = jnp.asarray(0.0 if lr is None else lr, f32)
        lr_value = jnp.where(
            jnp.isfinite(lr_value), lr_value, jnp.asarray(0.0, f32))
        factor = 1.0 / (1.0 + lr_value)
        target = jnp.asarray(jnp.nan, f32)
      elif self.parsed_norm_mode == 'lr':
        lr_value = jnp.asarray(0.0 if lr is None else lr, f32)
        lr_value = jnp.where(
            jnp.isfinite(lr_value), lr_value, jnp.asarray(0.0, f32))
        count = jnp.asarray(layer_param_count(params, path), f32)
        factor = 1.0 / (1.0 + lr_value * jnp.sqrt(jnp.maximum(count, 1.0)))
        target = jnp.asarray(jnp.nan, f32)
      elif self.parsed_norm_mode == 'constant':
        target = jnp.asarray(self.target_norm, f32)
        factor = target / jnp.maximum(norm, jnp.asarray(self.eps, f32))
      elif self.parsed_norm_mode == 'init':
        target = f32(target_norms.get(
            path, jnp.maximum(norm, jnp.asarray(1.0, f32))))
        factor = target / jnp.maximum(norm, jnp.asarray(self.eps, f32))
      else:
        raise ValueError(f'Unknown WSC norm mode: {self.parsed_norm_mode}')
      factor = jnp.where(jnp.isfinite(factor), factor, jnp.asarray(1.0, f32))
      factor = jnp.clip(
          factor, jnp.asarray(self.factor_min, f32),
          jnp.asarray(self.factor_max, f32))
      factor = jnp.where(wsc_started, factor, jnp.asarray(1.0, f32))

      for key in keys:
        params[key] = params[key] * factor.astype(params[key].dtype)

      follows_norm = has_following_rmsnorm(path, params)
      skey = scale_key(path)
      if skey in params:
        if self.parsed_scale_mode == 'no_scale':
          params[skey] = jnp.ones_like(params[skey])
        elif self.parsed_scale_mode == 'nograd':
          scale_adjust = 1 / factor
          scale_adjust = jnp.where(
              jnp.isfinite(scale_adjust), scale_adjust, jnp.asarray(1.0, f32))
          scale_adjust = jnp.clip(
              scale_adjust, jnp.asarray(self.scale_adjust_min, f32),
              jnp.asarray(self.scale_adjust_max, f32))
          scale_adjust = jnp.where(
              wsc_started, scale_adjust, jnp.asarray(1.0, f32))
          params[skey] = (
              params[skey] * scale_adjust.astype(params[skey].dtype))
          params[skey] = jnp.clip(
              params[skey], jnp.asarray(self.scale_min, params[skey].dtype),
              jnp.asarray(self.scale_max, params[skey].dtype))
        lname = metric_name(path)
        metrics[f'wsc/scale/{lname}'] = f32(params[skey])
        if path in outputs:
          metrics[f'wsc/output_l2_mean/{lname}'] = output_l2_mean(outputs[path])
      elif self.parsed_scale_mode == 'no_scale':
        lname = metric_name(path)
        metrics[f'wsc/scale/{lname}'] = jnp.asarray(1.0, f32)
      elif not follows_norm:
        lname = metric_name(path)
        metrics[f'wsc/missing_scale/{lname}'] = jnp.asarray(1.0, f32)

      lname = metric_name(path)
      metrics[f'wsc/factor/{lname}'] = f32(factor)
      metrics[f'wsc/pre_norm/{lname}'] = f32(norm)
      metrics[f'wsc/target_norm/{lname}'] = f32(target)
      if self.skip_last_layer:
        metrics[f'wsc/controlled/{lname}'] = jnp.asarray(1.0, f32)
      if self.parsed_norm_mode in ('factor', 'lr'):
        metrics[f'wsc/lr/{lname}'] = f32(lr_value)
      if self.parsed_norm_mode == 'lr':
        metrics[f'wsc/lr_param_count/{lname}'] = f32(count)

    if self.skip_last_layer:
      for path in all_groups:
        if path in groups:
          continue
        lname = metric_name(path)
        metrics[f'wsc/factor/{lname}'] = jnp.asarray(1.0, f32)
        metrics[f'wsc/pre_norm/{lname}'] = f32(layer_norm(params, path))
        metrics[f'wsc/target_norm/{lname}'] = jnp.asarray(jnp.nan, f32)
        metrics[f'wsc/controlled/{lname}'] = jnp.asarray(0.0, f32)

    if self.nograd_scale:
      metrics['wsc/nograd_started'] = f32(wsc_started)

    return params, metrics

  def output_metrics(self, outputs):
    if not self.active:
      return {}
    return {
        f'wsc/output_l2_mean/{metric_name(path)}': output_l2_mean(value)
        for path, value in outputs.items()}


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
