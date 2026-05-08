"""DreamerEnvLearner: Dreamer-style actor-critic using the real env as world model.

Architecture (mirrors DreamerV3 policy/value heads):
  - Hidden layers : 3 x units      (configurable via model_size)
  - Activation    : SiLU
  - Normalisation : RMSNorm (per hidden layer, after Dense)
  - Optimizer     : DreamerV3-aligned (AGC -> RMS -> momentum -> warmup-LR)

Training algorithm (mirrors dreamerv3/agent.py imag_loss):
  - Value head    : TwoHot distributional (255 symexp-spaced bins +-20),
                    identical to DreamerV3 value head.
  - Slow target   : EMA of value parameters (fraction=0.02 by default).
  - Imagination   : real env steps from replay-buffer physics states
                    (replaces DreamerV3's RSSM world-model rollout).
  - Actor loss    : advantage-weighted policy gradient with entropy bonus
                    (actent coefficient, no learnable temperature).
  - Returns       : lambda-return with per-step discounting and continuation mask.
  - Normalizers   : running percentile normalizers for returns / values /
                    advantages (mirrors dreamerv3/agent.py retnorm/valnorm/advnorm).
"""

import copy
import os
import re as _re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Sequence, Tuple

import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
import functools
import distrax

from jaxrl2.agents.agent import Agent
from jaxrl2.networks.constants import default_init
from jaxrl2.networks.mlp import _flatten_dict
from jaxrl2.types import Params, PRNGKey
from jaxrl2.utils.redo import SACReDo, SACGradientReDo

f32 = jnp.float32


# ---------------------------------------------------------------------------
# DreamerV3-aligned optimizer
# ---------------------------------------------------------------------------

def _clip_by_agc(clip: float = 0.3, pmin: float = 1e-3) -> optax.GradientTransformation:
    def init_fn(params):
        return ()
    def update_fn(updates, state, params=None):
        def fn(param, update):
            unorm = jnp.linalg.norm(update.flatten(), 2)
            pnorm = jnp.linalg.norm(param.flatten(), 2)
            upper = clip * jnp.maximum(pmin, pnorm)
            return update * (1 / jnp.maximum(1.0, unorm / upper))
        updates = jax.tree_util.tree_map(fn, params, updates) if clip else updates
        return updates, ()
    return optax.GradientTransformation(init_fn, update_fn)


def _scale_by_rms(beta: float = 0.999, eps: float = 1e-20) -> optax.GradientTransformation:
    def init_fn(params):
        nu = jax.tree_util.tree_map(lambda t: jnp.zeros_like(t, jnp.float32), params)
        step = jnp.zeros((), jnp.int32)
        return (step, nu)
    def update_fn(updates, state, params=None):
        step, nu = state
        step = optax.safe_int32_increment(step)
        nu = jax.tree_util.tree_map(lambda v, u: beta * v + (1 - beta) * (u * u), nu, updates)
        nu_hat = optax.bias_correction(nu, beta, step)
        updates = jax.tree_util.tree_map(lambda u, v: u / (jnp.sqrt(v) + eps), updates, nu_hat)
        return updates, (step, nu)
    return optax.GradientTransformation(init_fn, update_fn)


def _scale_by_momentum(beta: float = 0.9, nesterov: bool = False) -> optax.GradientTransformation:
    def init_fn(params):
        mu = jax.tree_util.tree_map(lambda t: jnp.zeros_like(t), params)
        step = jnp.zeros((), jnp.int32)
        return (step, mu)
    def update_fn(updates, state, params=None):
        step, mu = state
        step = optax.safe_int32_increment(step)
        mu = optax.update_moment(updates, mu, beta, 1)
        if nesterov:
            mu2 = optax.update_moment(updates, mu, beta, 1)
            mu_hat = optax.bias_correction(mu2, beta, step)
        else:
            mu_hat = optax.bias_correction(mu, beta, step)
        return mu_hat, (step, mu)
    return optax.GradientTransformation(init_fn, update_fn)


def _make_dreamer_opt(
    lr: float,
    agc: float = 0.3,
    eps: float = 1e-20,
    beta1: float = 0.9,
    beta2: float = 0.999,
    momentum: bool = True,
    nesterov: bool = False,
    wd: float = 0.0,
    wdregex: str = r'/kernel$',
    schedule: str = 'const',
    warmup: int = 1000,
    anneal: int = 0,
) -> optax.GradientTransformation:
    chain = []
    chain.append(_clip_by_agc(agc))
    chain.append(_scale_by_rms(beta2, eps))
    if momentum:
        chain.append(_scale_by_momentum(beta1, nesterov))
    if wd:
        pattern = _re.compile(wdregex)
        def _wd_mask(params):
            try:
                return jax.tree_util.tree_map_with_path(
                    lambda path, _: bool(pattern.search(
                        '/' + '/'.join(getattr(p, 'key', str(p)) for p in path))),
                    params)
            except AttributeError:
                return jax.tree_util.tree_map(lambda _: True, params)
        chain.append(optax.add_decayed_weights(wd, _wd_mask))
    assert anneal > 0 or schedule == 'const'
    if schedule == 'const':
        sched = optax.constant_schedule(lr)
    elif schedule == 'linear':
        sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
    elif schedule == 'cosine':
        sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
    else:
        raise NotImplementedError(f"Unknown schedule: {schedule!r}")
    if warmup:
        ramp = optax.linear_schedule(0.0, lr, warmup)
        sched = optax.join_schedules([ramp, sched], [warmup])
    chain.append(optax.scale_by_learning_rate(sched))
    return optax.chain(*chain)


# ---------------------------------------------------------------------------
# Size presets (mirrors dreamerv3/configs.yaml)
# ---------------------------------------------------------------------------

SAC_SIZES = {
    'size1m':   (64,   64,   64),
    'size12m':  (256,  256,  256),
    'size25m':  (384,  384,  384),
    'size50m':  (512,  512,  512),
    'size100m': (768,  768,  768),
    'size200m': (1024, 1024, 1024),
    'size400m': (1536, 1536, 1536),
}


# ---------------------------------------------------------------------------
# SiLU+RMSNorm MLP backbone (mirrors DreamerV3 MLP block)
# ---------------------------------------------------------------------------

class _SiLUMLP(nn.Module):
    """MLP with DreamerV3 layer order: Dense -> RMSNorm -> SiLU per hidden layer."""
    hidden_dims: Sequence[int]
    activate_final: bool = False

    @nn.compact
    def __call__(self, x, training: bool = False):
        x = _flatten_dict(x)
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init(), name=f'layer_{i}')(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = nn.RMSNorm(name=f'norm_{i}')(x)
                x = nn.silu(x)
                if self.is_mutable_collection('intermediates'):
                    self.sow('intermediates', f'layer_{i}_act', x)
        return x


# ---------------------------------------------------------------------------
# Actor: NormalTanhPolicySiLU (obs -> tanh-squashed Gaussian)
# ---------------------------------------------------------------------------

class NormalTanhPolicySiLU(nn.Module):
    """Stochastic actor: Gaussian policy with tanh squashing (SiLU+RMSNorm MLP)."""
    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    low: Optional[jnp.ndarray] = None
    high: Optional[jnp.ndarray] = None

    @nn.compact
    def __call__(self, observations, training: bool = False):
        outputs = _SiLUMLP(self.hidden_dims, activate_final=True)(observations, training=training)
        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)
        log_stds = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds))
        layers = []
        if self.low is not None and self.high is not None:
            low, high = self.low, self.high
            def rescale(x):
                return (x + 1) / 2 * (high - low) + low
            def fldj(x):
                h = jnp.broadcast_to(high, x.shape)
                l = jnp.broadcast_to(low, x.shape)
                return jnp.sum(jnp.log(0.5 * (h - l)), -1)
            layers.append(distrax.Lambda(rescale, forward_log_det_jacobian=fldj,
                                         event_ndims_in=1, event_ndims_out=1))
        layers.append(distrax.Block(distrax.Tanh(), 1))
        bijector = distrax.Chain(layers)
        return distrax.Transformed(distribution=distribution, bijector=bijector)


# ---------------------------------------------------------------------------
# Value head: StateValueTwoHotSiLU (obs -> logits[num_bins])
# Mirrors DreamerV3's MLPHead(scalar, outhead=twohot, outscale=0)
# ---------------------------------------------------------------------------

class StateValueTwoHotSiLU(nn.Module):
    """V(s) with TwoHot distributional output (obs -> logits[num_bins])."""
    hidden_dims: Sequence[int]
    num_bins: int = 255

    @nn.compact
    def __call__(self, obs, training: bool = False):
        x = _SiLUMLP(self.hidden_dims, activate_final=True)(obs, training=training)
        logits = nn.Dense(
            self.num_bins,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name='logits',
        )(x)
        return logits


# ---------------------------------------------------------------------------
# TwoHot utilities (mirrors dreamerv3/nets.py TwoHot)
# ---------------------------------------------------------------------------

def _symexp(x):
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def _make_twohot_bins(num_bins: int) -> jnp.ndarray:
    if num_bins % 2 == 1:
        half = jnp.linspace(-20.0, 0.0, (num_bins - 1) // 2 + 1, dtype=jnp.float32)
        half = _symexp(half)
        bins = jnp.concatenate([half, -half[:-1][::-1]], axis=0)
    else:
        half = jnp.linspace(-20.0, 0.0, num_bins // 2, dtype=jnp.float32)
        half = _symexp(half)
        bins = jnp.concatenate([half, -half[::-1]], axis=0)
    return bins


def _twohot_pred(logits, bins):
    probs = jax.nn.softmax(logits, axis=-1)
    n = bins.shape[0]
    if n % 2 == 1:
        m = (n - 1) // 2
        p1, p2, p3 = probs[..., :m], probs[..., m:m+1], probs[..., m+1:]
        b1, b2, b3 = bins[:m], bins[m:m+1], bins[m+1:]
        return (p2 * b2).sum(-1) + ((p1 * b1)[..., ::-1] + (p3 * b3)).sum(-1)
    else:
        h = n // 2
        p1, p2 = probs[..., :h], probs[..., h:]
        b1, b2 = bins[:h], bins[h:]
        return ((p1 * b1)[..., ::-1] + (p2 * b2)).sum(-1)


def _twohot_loss(logits, target, bins):
    target = jax.lax.stop_gradient(target.astype(jnp.float32))
    nb = bins.shape[0]
    below = (bins <= target[..., None]).sum(-1).astype(jnp.int32) - 1
    above = nb - (bins > target[..., None]).sum(-1).astype(jnp.int32)
    below = jnp.clip(below, 0, nb - 1)
    above = jnp.clip(above, 0, nb - 1)
    equal = (below == above)
    dist_to_below = jnp.where(equal, 1.0, jnp.abs(bins[below] - target))
    dist_to_above = jnp.where(equal, 1.0, jnp.abs(bins[above] - target))
    total = dist_to_below + dist_to_above
    weight_below = dist_to_above / total
    weight_above = dist_to_below / total
    target_twohot = (
        jax.nn.one_hot(below, nb) * weight_below[..., None] +
        jax.nn.one_hot(above, nb) * weight_above[..., None])
    log_pred = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    return -(target_twohot * log_pred).sum(-1)


# ---------------------------------------------------------------------------
# Lambda-return (identical to dreamerv3/agent.py lambda_return)
# ---------------------------------------------------------------------------

def lambda_return(last, term, rew, val, boot, disc, lam):
    rets = [boot[:, -1]]
    live = (1 - f32(term))[:, 1:] * disc
    cont = (1 - f32(last))[:, 1:] * lam
    interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
    for t in reversed(range(live.shape[1])):
        rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
    return jnp.stack(list(reversed(rets))[:-1], 1)


# ---------------------------------------------------------------------------
# Running percentile normalizer (mirrors DreamerV3 Normalize)
# ---------------------------------------------------------------------------

class _Normalize:
    def __init__(self, momentum=0.99, perclo=5.0, perchi=95.0, eps=1.0):
        self._momentum = momentum
        self._perclo   = perclo
        self._perchi   = perchi
        self._eps      = eps
        self._offset   = 0.0
        self._scale    = 1.0

    def stats(self):
        return self._offset, self._scale

    def update(self, x):
        x = np.asarray(x).astype(np.float32).flatten()
        if len(x) == 0:
            return
        lo = float(np.percentile(x, self._perclo))
        hi = float(np.percentile(x, self._perchi))
        m = self._momentum
        self._offset = m * self._offset + (1 - m) * (lo + hi) / 2.0
        self._scale  = m * self._scale  + (1 - m) * max(hi - lo, self._eps)


# ---------------------------------------------------------------------------
# JIT-compiled actor+value update step (mirrors dreamerv3/agent.py imag_loss)
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=(
    'num_bins', 'actent', 'slowreg', 'horizon', 'contdisc'))
def _update_jit(
    rng,
    actor: TrainState,
    value: TrainState,
    slow_value_params,
    obs_traj,    # [B, H+1, obs_dim]
    act_traj,    # [B, H+1, act_dim]
    rew_traj,    # [B, H+1]
    last_traj,   # [B, H+1] bool
    term_traj,   # [B, H+1] bool
    ret_offset: float, ret_scale: float,
    val_offset: float, val_scale: float,
    adv_offset: float, adv_scale: float,
    num_bins: int,
    actent: float,
    slowreg: float,
    horizon: float,
    contdisc: bool,
    lam: float,
):
    bins = _make_twohot_bins(num_bins)
    disc = 1.0 if contdisc else 1.0 - 1.0 / horizon

    slow_logits = value.apply_fn(
        {'params': jax.lax.stop_gradient(slow_value_params)}, obs_traj)
    slow_val = _twohot_pred(slow_logits, bins) * val_scale + val_offset

    con = f32(~term_traj)
    weight = jnp.cumprod(disc * con, 1) / disc
    ret = lambda_return(last_traj, term_traj, rew_traj, slow_val, slow_val, disc, lam)

    adv = (ret - slow_val[:, :-1]) / ret_scale
    adv_normed = (adv - adv_offset) / adv_scale

    def actor_loss_fn(actor_params):
        dist = actor.apply_fn({'params': actor_params}, obs_traj[:, :-1])
        # Clip stored tanh-squashed actions to (-1+ε, 1-ε) to prevent
        # arctanh(±1) = ±∞, which causes log_prob = ∞ - ∞ = NaN and
        # subsequently NaN gradients / NaN policy weights.
        actions_taken = jax.lax.stop_gradient(
            jnp.clip(act_traj[:, :-1], -1.0 + 1e-6, 1.0 - 1e-6))
        log_probs = dist.log_prob(actions_taken)
        ents = dist.distribution.entropy()
        policy_loss = (jax.lax.stop_gradient(weight[:, :-1]) *
                       -(log_probs * jax.lax.stop_gradient(adv_normed) + actent * ents)).mean()
        return policy_loss, {
            'loss/policy': policy_loss,
            'train/ent/action': ents.mean(),
        }

    actor_grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=actor_grads)

    tar_normed = (ret - val_offset) / val_scale
    tar_padded = jnp.concatenate([tar_normed, 0.0 * tar_normed[:, -1:]], 1)
    slow_logits_sg = jax.lax.stop_gradient(slow_logits)

    def value_loss_fn(value_params):
        v_logits = value.apply_fn({'params': value_params}, obs_traj)
        ce_loss  = _twohot_loss(v_logits, tar_padded, bins)
        slow_ce  = _twohot_loss(v_logits,
            jax.lax.stop_gradient(_twohot_pred(slow_logits_sg, bins)), bins)
        val_loss = (jax.lax.stop_gradient(weight) * (ce_loss + slowreg * slow_ce))[:, :-1].mean()
        cur_val  = _twohot_pred(v_logits, bins) * val_scale + val_offset
        return val_loss, {
            'loss/value': val_loss,
            'train/value': cur_val[:, :-1].mean(),
        }

    value_grads, value_info = jax.grad(value_loss_fn, has_aux=True)(value.params)
    new_value = value.apply_gradients(grads=value_grads)

    ret_normed = (ret - ret_offset) / ret_scale
    metrics = {
        **actor_info, **value_info,
        'train/imag_return':     ret_normed.mean(),
        'train/imag_return_std': ret_normed.std(),
        'train/advantage':       adv_normed.mean(),
        'train/weight':          weight[:, :-1].mean(),
        'train/imag_reward':     rew_traj[:, 1:].mean(),
        'train/con':             con[:, 1:].mean(),
    }
    return rng, new_actor, new_value, metrics


# ---------------------------------------------------------------------------
# JIT helpers for inference
# ---------------------------------------------------------------------------

@jax.jit
def _sample_actions_jit(key, actor: TrainState, observations):
    dist = actor.apply_fn({"params": actor.params}, observations)
    return dist.sample(seed=key)


@jax.jit
def _eval_actions_jit(actor: TrainState, observations):
    dist = actor.apply_fn({"params": actor.params}, observations)
    return dist.bijector.forward(dist.distribution.mean())


# ---------------------------------------------------------------------------
# Picklable helpers for parallel imagination rollouts
# (module-level so ThreadPoolExecutor workers can call them without closure)
# ---------------------------------------------------------------------------

def _init_imag_env(args):
    """Restore physics state of one imagination env (called in thread pool)."""
    env, state = args
    env.set_physics_state(state)


def _step_imag_env(args):
    """Step one imagination env (called in thread pool).
    Returns (obs, reward, done, info) or None on physics error.
    """
    env, action = args
    try:
        return env.step(action)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# DreamerEnvLearner
# ---------------------------------------------------------------------------

class DreamerEnvLearner(Agent):
    """Dreamer-style actor-critic using the real environment as world model.

    Key differences vs SAC:
      * Learns V(s) (not Q(s,a)), no double-critic ensemble.
      * Policy trained with advantage-weighted PG + entropy bonus (actent),
        no learnable temperature.
      * Slow (EMA) value target; return/value/advantage normalizers.
      * Imagination rollouts executed by real env step() calls, where env
        physics state is reset to replay-buffer states before each rollout.
    """

    def __init__(
        self,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        value_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (64, 64, 64),
        model_size: Optional[str] = None,
        num_bins: int = 255,
        discount: float = 0.997,
        lam: float = 0.95,
        contdisc: bool = True,
        horizon: float = 333.0,
        actent: float = 3e-4,
        slowreg: float = 1.0,
        slow_fraction: float = 0.02,
        imag_length: int = 5,
        vd_mode: str = 'disabled',
        imag_workers: int = 0,
        opt: Optional[Dict] = None,
        redo: Optional[Dict] = None,
    ):
        action_dim = action_space.shape[-1]

        if model_size is not None:
            if model_size not in SAC_SIZES:
                raise ValueError(f"Unknown model_size '{model_size}'. Valid: {list(SAC_SIZES)}")
            hidden_dims = SAC_SIZES[model_size]

        self.discount      = discount
        self.lam           = lam
        self.contdisc      = contdisc
        self.horizon       = horizon
        self.actent        = actent
        self.slowreg       = slowreg
        self.slow_fraction = slow_fraction
        self.imag_length   = imag_length
        self.num_bins      = num_bins
        self._vd_mode      = vd_mode

        observations = observation_space.sample()
        actions      = action_space.sample()

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, value_key = jax.random.split(rng, 3)

        if np.all(action_space.low == -1) and np.all(action_space.high == 1):
            low = high = None
        else:
            low  = action_space.low
            high = action_space.high

        opt = opt or {}
        opt_kwargs = dict(
            agc      = opt.get('agc',      0.3),
            eps      = opt.get('eps',      1e-20),
            beta1    = opt.get('beta1',    0.9),
            beta2    = opt.get('beta2',    0.999),
            momentum = opt.get('momentum', True),
            nesterov = opt.get('nesterov', False),
            wd       = opt.get('wd',       0.0),
            wdregex  = opt.get('wdregex',  r'/kernel$'),
            schedule = opt.get('schedule', 'const'),
            warmup   = opt.get('warmup',   1000),
            anneal   = opt.get('anneal',   0),
        )
        self._opt_kwargs = opt_kwargs

        actor_def = NormalTanhPolicySiLU(hidden_dims, action_dim, low=low, high=high)
        actor_params = actor_def.init(actor_key, observations)['params']
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=_make_dreamer_opt(actor_lr, **opt_kwargs),
        )

        value_def = StateValueTwoHotSiLU(hidden_dims, num_bins=num_bins)
        value_params = value_def.init(value_key, observations)['params']
        value = TrainState.create(
            apply_fn=value_def.apply,
            params=value_params,
            tx=_make_dreamer_opt(value_lr, **opt_kwargs),
        )
        slow_value_params = copy.deepcopy(value_params)

        self._actor = actor
        self._value = value
        self._slow_value_params = slow_value_params
        self._rng   = rng

        self._actor_def  = actor_def
        self._value_def  = value_def
        self._actor_lr   = actor_lr
        self._value_lr   = value_lr
        self._observations_sample = observations
        self._hidden_dims = hidden_dims

        self._retnorm = _Normalize(momentum=0.99, perclo=5.0, perchi=95.0, eps=1.0)
        self._valnorm = _Normalize(momentum=0.99, perclo=5.0, perchi=95.0, eps=1.0)
        self._advnorm = _Normalize(momentum=0.99, perclo=5.0, perchi=95.0, eps=1.0)

        redo = redo or {}
        redo_kw = dict(
            tau=redo.get('tau', 0.05),
            mode=redo.get('mode', 'threshold'),
            frequency=redo.get('frequency', 1000),
            log_item=redo.get('log_item', 'disabled'),
            rank_threshold=redo.get('rank_threshold', 0.99),
            reset_start=redo.get('reset_start', 0),
            reset_end=redo.get('reset_end', 0),
        )
        skip = redo.get('skip_last_layer', False)
        self._actor_redo = SACReDo(name='actor', **redo_kw, skip_last_layer=skip) \
            if redo.get('redo_enabled', False) else None
        self._value_redo = SACReDo(name='value', **redo_kw, skip_last_layer=skip) \
            if redo.get('redo_enabled', False) else None
        grad_kw = {k: v for k, v in redo_kw.items() if k != 'rank_threshold'}
        self._actor_grad_redo = SACGradientReDo(name='actor', **grad_kw) \
            if redo.get('grad_redo_enabled', False) else None
        self._value_grad_redo = SACGradientReDo(name='value', **grad_kw) \
            if redo.get('grad_redo_enabled', False) else None

        # Persistent thread pool for parallel imagination rollouts.
        # MuJoCo releases the GIL during physics simulation, so threads give
        # genuine parallelism across independent env instances.
        _n = imag_workers if imag_workers > 0 else min(os.cpu_count() or 32, 64)
        self._imag_pool = ThreadPoolExecutor(max_workers=_n)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_agent(self) -> None:
        self._rng, actor_key, value_key = jax.random.split(self._rng, 3)
        actor_params = self._actor_def.init(actor_key, self._observations_sample)['params']
        self._actor = self._actor.replace(
            params=actor_params,
            opt_state=_make_dreamer_opt(self._actor_lr, **self._opt_kwargs).init(actor_params),
            step=0,
        )
        value_params = self._value_def.init(value_key, self._observations_sample)['params']
        self._value = self._value.replace(
            params=value_params,
            opt_state=_make_dreamer_opt(self._value_lr, **self._opt_kwargs).init(value_params),
            step=0,
        )
        self._slow_value_params = copy.deepcopy(value_params)
        for obj in (self._actor_redo, self._value_redo,
                    self._actor_grad_redo, self._value_grad_redo):
            if obj is not None:
                obj._step = 0

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        return np.asarray(_eval_actions_jit(self._actor, observations))

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        rng, key = jax.random.split(self._rng)
        self._rng = rng
        return np.asarray(_sample_actions_jit(key, self._actor, observations))

    # ------------------------------------------------------------------
    # Imagination rollout via real envs
    # ------------------------------------------------------------------

    def _rollout_envs(self, imag_envs, batch_obs, batch_physics):
        """Roll out H steps from given physics states using real env step().

        Args:
            imag_envs     : list of ContinualDMCEnv; len >= B.
            batch_obs     : [B, obs_dim] starting observations (float32).
            batch_physics : [B, state_dim] starting physics states (float64).

        Returns:
            obs_traj  : [B, H+1, obs_dim]
            act_traj  : [B, H+1, act_dim]
            rew_traj  : [B, H+1]  (rew[:,0] = 0 placeholder)
            last_traj : [B, H+1] bool
            term_traj : [B, H+1] bool
        """
        B = min(len(imag_envs), batch_obs.shape[0])
        H = self.imag_length

        # Restore physics states in parallel (MuJoCo releases GIL).
        list(self._imag_pool.map(
            _init_imag_env,
            [(imag_envs[i], batch_physics[i]) for i in range(B)]))

        obs_traj  = [batch_obs[:B].astype(np.float32)]
        act_traj  = []
        rew_traj  = [np.zeros(B, dtype=np.float32)]
        last_traj = [np.zeros(B, dtype=bool)]
        term_traj = [np.zeros(B, dtype=bool)]

        for _ in range(H):
            cur_obs = obs_traj[-1]
            self._rng, key = jax.random.split(self._rng)
            actions = np.asarray(_sample_actions_jit(key, self._actor, cur_obs))

            next_obs = cur_obs.copy()
            rew  = np.zeros(B, dtype=np.float32)
            last = last_traj[-1].copy()
            term = np.zeros(B, dtype=bool)

            # Step active envs in parallel.
            active = [i for i in range(B) if not last_traj[-1][i]]
            step_results = list(self._imag_pool.map(
                _step_imag_env,
                [(imag_envs[i], actions[i]) for i in active]))

            for idx, i in enumerate(active):
                result = step_results[idx]
                if result is None:
                    # Physics error: treat as episode end.
                    last[i] = True
                    continue
                n_obs, r, done, info = result
                n_obs_arr = np.asarray(n_obs, dtype=np.float32)
                if not np.all(np.isfinite(n_obs_arr)):
                    # NaN from physics instability: end trajectory.
                    last[i] = True
                    continue
                next_obs[i] = n_obs_arr
                rew[i]  = float(r)
                last[i] = bool(done)
                term[i] = bool(done and not info.get('TimeLimit.truncated', False))

            act_traj.append(actions)
            rew_traj.append(rew)
            last_traj.append(last)
            term_traj.append(term)
            obs_traj.append(next_obs)

        self._rng, key = jax.random.split(self._rng)
        final_acts = np.asarray(_sample_actions_jit(key, self._actor, obs_traj[-1]))
        act_traj.append(final_acts)

        return (
            np.stack(obs_traj,  axis=1),   # [B, H+1, obs_dim]
            np.stack(act_traj,  axis=1),   # [B, H+1, act_dim]
            np.stack(rew_traj,  axis=1),   # [B, H+1]
            np.stack(last_traj, axis=1),   # [B, H+1]
            np.stack(term_traj, axis=1),   # [B, H+1]
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def update(self, batch: FrozenDict, imag_envs: list) -> Dict[str, float]:
        """One gradient update using imagination rollouts from batch physics states.

        The replay buffer batch must contain 'observations' and 'physics_states'.
        imag_envs is a list of ContinualDMCEnv instances used for imagination rollouts.
        """
        obs0   = np.asarray(batch['observations'])
        phys   = np.asarray(batch['physics_states'])
        B_imag = min(len(imag_envs), obs0.shape[0])

        obs_traj, act_traj, rew_traj, last_traj, term_traj = self._rollout_envs(
            imag_envs, obs0[:B_imag], phys[:B_imag])

        # Update running normalizers (outside JIT, mirrors DreamerV3 update=True).
        bins_jnp = _make_twohot_bins(self.num_bins)
        disc = 1.0 if self.contdisc else 1.0 - 1.0 / self.horizon
        val_offset_old, val_scale_old = self._valnorm.stats()

        slow_logits_np = np.asarray(
            self._value.apply_fn({'params': self._slow_value_params}, obs_traj))
        slow_val_np = (np.asarray(_twohot_pred(jnp.array(slow_logits_np), bins_jnp))
                       * val_scale_old + val_offset_old)

        def _np_lambda(last, term, rew, val, disc, lam):
            H = rew.shape[1] - 1
            rets = [val[:, -1].copy()]
            live = (1 - term[:, 1:].astype(np.float32)) * disc
            cont = (1 - last[:, 1:].astype(np.float32)) * lam
            interm = rew[:, 1:] + (1 - cont) * live * val[:, 1:]
            for t in reversed(range(H)):
                rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
            return np.stack(list(reversed(rets))[:-1], 1)

        ret_np = _np_lambda(last_traj, term_traj, rew_traj, slow_val_np, disc, self.lam)

        self._retnorm.update(ret_np)
        self._valnorm.update(ret_np)
        adv_np = (ret_np - slow_val_np[:, :-1]) / max(self._retnorm.stats()[1], 1e-8)
        self._advnorm.update(adv_np)

        ret_offset, ret_scale = self._retnorm.stats()
        val_offset, val_scale = self._valnorm.stats()
        adv_offset, adv_scale = self._advnorm.stats()

        self._rng, new_actor, new_value, info = _update_jit(
            self._rng,
            self._actor, self._value, self._slow_value_params,
            obs_traj, act_traj, rew_traj, last_traj, term_traj,
            ret_offset, ret_scale,
            val_offset, val_scale,
            adv_offset, adv_scale,
            self.num_bins, self.actent, self.slowreg, self.horizon, self.contdisc, self.lam,
        )
        self._actor = new_actor
        self._value = new_value

        # EMA slow value update (mirrors DreamerV3 slowval.update()).
        f = self.slow_fraction
        self._slow_value_params = jax.tree_util.tree_map(
            lambda slow, fast: (1.0 - f) * slow + f * fast,
            self._slow_value_params, self._value.params)

        # ReDo.
        redo_due = any([
            self._actor_redo is not None and self._actor_redo.should_run(),
            self._value_redo is not None and self._value_redo.should_run(),
            self._actor_grad_redo is not None and self._actor_grad_redo.should_run(),
            self._value_grad_redo is not None and self._value_grad_redo.should_run(),
        ])
        if redo_due:
            B, T1, D = obs_traj.shape
            obs_flat = obs_traj.reshape(B * T1, D)
            info.update(self._apply_redo(obs_flat))
        else:
            for obj in (self._actor_redo, self._value_redo,
                        self._actor_grad_redo, self._value_grad_redo):
                if obj is not None:
                    obj._step += 1

        return {k: float(v) for k, v in info.items()}

    # ------------------------------------------------------------------
    # ReDo helpers
    # ------------------------------------------------------------------

    def _collect_actor_acts(self, obs):
        _, state = self._actor.apply_fn(
            {'params': self._actor.params}, obs, mutable=['intermediates'])
        return _flatten_intermediates(state.get('intermediates', {}))

    def _collect_value_acts(self, obs):
        _, state = self._value.apply_fn(
            {'params': self._value.params}, obs, mutable=['intermediates'])
        return _flatten_intermediates(state.get('intermediates', {}))

    def _apply_redo(self, obs_flat):
        info = {}
        if self._actor_redo is not None and self._actor_redo.should_run():
            self._rng, key = jax.random.split(self._rng)
            acts = self._collect_actor_acts(obs_flat)
            new_p, mets = self._actor_redo.step(self._actor.params, acts, key)
            self._actor = self._actor.replace(params=new_p)
            info.update(mets)
        elif self._actor_redo is not None:
            self._actor_redo._step += 1

        if self._value_redo is not None and self._value_redo.should_run():
            self._rng, key = jax.random.split(self._rng)
            acts = self._collect_value_acts(obs_flat)
            new_p, mets = self._value_redo.step(self._value.params, acts, key)
            self._value = self._value.replace(params=new_p)
            info.update(mets)
        elif self._value_redo is not None:
            self._value_redo._step += 1

        return info


def _flatten_intermediates(intermediates: dict) -> Dict:
    flat = {}
    def _walk(node, prefix):
        if not (isinstance(node, dict) or hasattr(node, 'items')):
            v = node[0] if (isinstance(node, tuple) and len(node) == 1) else node
            if hasattr(v, 'ndim') and v.ndim >= 2:
                v = v[0]
            flat[prefix] = v
            return
        for k, v in node.items():
            _walk(v, f'{prefix}/{k}' if prefix else k)
    _walk(intermediates, '')
    return flat
