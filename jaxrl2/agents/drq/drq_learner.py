"""Implementations of algorithms for continuous control."""

import copy
import functools
import os
import tempfile
from typing import Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from jaxrl2.agents.agent import Agent
from jaxrl2.agents.drq.augmentations import batched_random_shift
from jaxrl2.agents.sac.temperature import Temperature
from jaxrl2.agents.sac.temperature_updater import update_temperature
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.agents.sac_dreamer_dist_learner import (
    SAC_SIZES,
    FlaxWSC,
    NormalTanhPolicySiLU,
    StateActionEnsembleSiLU,
    _apply_gradients_with_opt_metrics,
    _eval_actions_tanh_mean_jit,
    _flatten_param_paths,
    _has_following_rmsnorm,
    _make_sac_opt,
    _metric_name,
    _sample_actions_jit,
)
from jaxrl2.networks.encoders import D4PGEncoder, ResNetV2Encoder
from jaxrl2.networks.pixel_multiplexer import PixelMultiplexer
from jaxrl2.types import Params, PRNGKey
from jaxrl2.utils.target_update import soft_target_update

f32 = jnp.float32
_REDO_TAUS = (0.05, 0.1, 0.2, 0.4)


# Helps to minimize CPU to GPU transfer.
def _unpack(batch):
    # Assuming that if next_observation is missing, it's combined with observation:
    if "pixels" in batch["next_observations"]:
        return batch
    obs_pixels = batch["observations"]["pixels"][..., :-1]
    next_obs_pixels = batch["observations"]["pixels"][..., 1:]

    obs = batch["observations"].copy(add_or_replace={"pixels": obs_pixels})
    next_obs = batch["next_observations"].copy(
        add_or_replace={"pixels": next_obs_pixels}
    )

    batch = batch.copy(
        add_or_replace={"observations": obs, "next_observations": next_obs}
    )

    return batch


def _share_encoder(source, target):
    # Use critic conv layers in actor:
    if hasattr(target.params, "copy") and not isinstance(target.params, dict):
        new_params = target.params.copy(
            add_or_replace={"encoder": source.params["encoder"]}
        )
    else:
        new_params = dict(target.params)
        new_params["encoder"] = source.params["encoder"]
    return target.replace(params=new_params)


def _filter_actor_l2_refs(params):
    """Actor encoder is shared from critic and stop-grad; skip it for L2-init."""
    params = dict(params)
    if "encoder" in params:
        params = dict(params)
        params.pop("encoder")
    return params


def _tree_l2_delta_raw(params, refs):
    total = jnp.asarray(0.0, f32)
    flat = _flatten_param_paths(params)
    for key, init_value in _flatten_param_paths(refs).items():
        value = flat.get(key)
        if value is None:
            continue
        diff = f32(value) - f32(init_value)
        total = total + 0.5 * jnp.square(diff).sum()
    return total


def _l2_init_metrics(params, refs, module):
    raw = _tree_l2_delta_raw(params, refs)
    metrics = {
        f"mechanism/l2_init/module_delta_sq/{module}": 2.0 * raw,
    }
    return raw, metrics


@functools.partial(jax.jit, static_argnames=(
    "backup_entropy", "critic_reduction", "augmentation_pad",
    "augmentation_enabled", "l2_init_enabled"))
def _update_jit(
    rng: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    target_critic_params: Params,
    temp: TrainState,
    actor_init_params: Params,
    critic_init_params: Params,
    batch: FrozenDict,
    discount: float,
    tau: float,
    target_entropy: float,
    backup_entropy: bool,
    critic_reduction: str,
    augmentation_pad: int,
    augmentation_enabled: bool,
    l2_init_enabled: bool,
    l2_init_weight: float,
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str, float]]:
    batch = _unpack(batch)
    actor = _share_encoder(source=critic, target=actor)

    rng, key = jax.random.split(rng)
    if augmentation_enabled:
        aug_pixels = batched_random_shift(
            key, batch["observations"]["pixels"], pad=augmentation_pad)
    else:
        aug_pixels = batch["observations"]["pixels"]
    observations = batch["observations"].copy(add_or_replace={"pixels": aug_pixels})
    batch = batch.copy(add_or_replace={"observations": observations})

    rng, key = jax.random.split(rng)
    if augmentation_enabled:
        aug_next_pixels = batched_random_shift(
            key, batch["next_observations"]["pixels"], pad=augmentation_pad)
    else:
        aug_next_pixels = batch["next_observations"]["pixels"]
    next_observations = batch["next_observations"].copy(
        add_or_replace={"pixels": aug_next_pixels}
    )
    batch = batch.copy(add_or_replace={"next_observations": next_observations})

    rng, key, noise_key1, noise_key2 = jax.random.split(rng, 4)
    target_critic = critic.replace(params=target_critic_params)
    dist = actor.apply_fn({"params": actor.params}, batch["next_observations"])
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)
    next_qs = target_critic.apply_fn(
        {"params": target_critic.params},
        batch["next_observations"],
        next_actions,
        rngs={"noise": noise_key1},
    )
    if critic_reduction == "min":
        next_q = next_qs.min(axis=0)
    elif critic_reduction == "mean":
        next_q = next_qs.mean(axis=0)
    else:
        raise NotImplementedError(critic_reduction)
    target_q = batch["rewards"] + discount * batch["masks"] * next_q
    if backup_entropy:
        target_q -= (
            discount
            * batch["masks"]
            * temp.apply_fn({"params": temp.params})
            * next_log_probs
        )

    def critic_loss_fn(critic_params: Params):
        qs = critic.apply_fn(
            {"params": critic_params},
            batch["observations"],
            batch["actions"],
            rngs={"noise": noise_key2},
        )
        critic_loss = ((qs - target_q) ** 2).mean()
        raw_l2, l2_metrics = _l2_init_metrics(
            critic_params, critic_init_params, "critic")
        weighted_l2 = jnp.where(
            l2_init_enabled, raw_l2 * f32(l2_init_weight), jnp.asarray(0.0, f32))
        loss = critic_loss + weighted_l2
        return loss, {
            "critic_loss": critic_loss,
            "q": qs.mean(),
            "target_actor_entropy": -next_log_probs.mean(),
            "critic_l2_init_raw": jnp.where(
                l2_init_enabled, raw_l2, jnp.asarray(0.0, f32)),
            "critic_l2_init_weighted": weighted_l2,
            **(l2_metrics if l2_init_enabled else {}),
        }

    critic_grads, critic_info = jax.grad(
        critic_loss_fn, has_aux=True)(critic.params)
    new_critic, critic_opt_info = _apply_gradients_with_opt_metrics(
        critic, critic_grads, "critic")
    new_target_critic_params = soft_target_update(
        new_critic.params, target_critic_params, tau
    )

    rng, key, noise_key = jax.random.split(rng, 3)

    def actor_loss_fn(actor_params: Params):
        dist = actor.apply_fn({"params": actor_params}, batch["observations"])
        actions, log_probs = dist.sample_and_log_prob(seed=key)
        qs = new_critic.apply_fn(
            {"params": new_critic.params},
            batch["observations"],
            actions,
            rngs={"noise": noise_key},
        )
        q = qs.mean(axis=0)
        actor_loss = (
            log_probs * temp.apply_fn({"params": temp.params}) - q).mean()
        actor_l2_refs = _filter_actor_l2_refs(actor_init_params)
        raw_l2, l2_metrics = _l2_init_metrics(
            _filter_actor_l2_refs(actor_params), actor_l2_refs, "actor")
        weighted_l2 = jnp.where(
            l2_init_enabled, raw_l2 * f32(l2_init_weight), jnp.asarray(0.0, f32))
        loss = actor_loss + weighted_l2
        return loss, {
            "actor_loss": actor_loss,
            "entropy": -log_probs.mean(),
            "actor_l2_init_raw": jnp.where(
                l2_init_enabled, raw_l2, jnp.asarray(0.0, f32)),
            "actor_l2_init_weighted": weighted_l2,
            **(l2_metrics if l2_init_enabled else {}),
        }

    actor_grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor, actor_opt_info = _apply_gradients_with_opt_metrics(
        actor, actor_grads, "actor")
    new_temp, alpha_info = update_temperature(
        temp, actor_info["entropy"], target_entropy
    )

    l2_raw = (
        critic_info.pop("critic_l2_init_raw") +
        actor_info.pop("actor_l2_init_raw"))
    l2_weighted = (
        critic_info.pop("critic_l2_init_weighted") +
        actor_info.pop("actor_l2_init_weighted"))
    mechanism_info = {}
    if l2_init_enabled:
        mechanism_info = {
            "mechanism/l2_init/raw_loss": l2_raw,
            "mechanism/l2_init/weighted_loss": l2_weighted,
            "mechanism/active": jnp.asarray(1.0, f32),
        }

    return (
        rng,
        new_actor,
        new_critic,
        new_target_critic_params,
        new_temp,
        {
            **critic_info,
            **actor_info,
            **alpha_info,
            **critic_opt_info,
            **actor_opt_info,
            **mechanism_info,
            "data_augmentation/drqv2_random_shift_active": jnp.asarray(
                1.0 if augmentation_enabled else 0.0, f32),
            "data_augmentation/drqv2_random_shift_pad": jnp.asarray(
                augmentation_pad, f32),
        },
    )


@functools.partial(jax.jit, static_argnames=(
    "backup_entropy", "critic_reduction", "augmentation_pad",
    "augmentation_enabled", "l2_init_enabled"))
def _grad_trees_jit(
    rng: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    target_critic_params: Params,
    temp: TrainState,
    actor_init_params: Params,
    critic_init_params: Params,
    batch: FrozenDict,
    discount: float,
    target_entropy: float,
    backup_entropy: bool,
    critic_reduction: str,
    augmentation_pad: int,
    augmentation_enabled: bool,
    l2_init_enabled: bool,
    l2_init_weight: float,
) -> Tuple[Params, Params]:
    batch = _unpack(batch)
    actor = _share_encoder(source=critic, target=actor)

    rng, key = jax.random.split(rng)
    if augmentation_enabled:
        aug_pixels = batched_random_shift(
            key, batch["observations"]["pixels"], pad=augmentation_pad)
    else:
        aug_pixels = batch["observations"]["pixels"]
    observations = batch["observations"].copy(add_or_replace={"pixels": aug_pixels})
    batch = batch.copy(add_or_replace={"observations": observations})

    rng, key = jax.random.split(rng)
    if augmentation_enabled:
        aug_next_pixels = batched_random_shift(
            key, batch["next_observations"]["pixels"], pad=augmentation_pad)
    else:
        aug_next_pixels = batch["next_observations"]["pixels"]
    next_observations = batch["next_observations"].copy(
        add_or_replace={"pixels": aug_next_pixels}
    )
    batch = batch.copy(add_or_replace={"next_observations": next_observations})

    rng, key, noise_key1, noise_key2 = jax.random.split(rng, 4)
    target_critic = critic.replace(params=target_critic_params)
    dist = actor.apply_fn({"params": actor.params}, batch["next_observations"])
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)
    next_qs = target_critic.apply_fn(
        {"params": target_critic.params},
        batch["next_observations"],
        next_actions,
        rngs={"noise": noise_key1},
    )
    if critic_reduction == "min":
        next_q = next_qs.min(axis=0)
    elif critic_reduction == "mean":
        next_q = next_qs.mean(axis=0)
    else:
        raise NotImplementedError(critic_reduction)
    target_q = batch["rewards"] + discount * batch["masks"] * next_q
    if backup_entropy:
        target_q -= (
            discount
            * batch["masks"]
            * temp.apply_fn({"params": temp.params})
            * next_log_probs
        )

    def critic_loss_fn(critic_params: Params):
        qs = critic.apply_fn(
            {"params": critic_params},
            batch["observations"],
            batch["actions"],
            rngs={"noise": noise_key2},
        )
        critic_loss = ((qs - target_q) ** 2).mean()
        raw_l2, _ = _l2_init_metrics(
            critic_params, critic_init_params, "critic")
        weighted_l2 = jnp.where(
            l2_init_enabled, raw_l2 * f32(l2_init_weight), jnp.asarray(0.0, f32))
        return critic_loss + weighted_l2

    critic_grads = jax.grad(critic_loss_fn)(critic.params)
    rng, key, noise_key = jax.random.split(rng, 3)

    def actor_loss_fn(actor_params: Params):
        dist = actor.apply_fn({"params": actor_params}, batch["observations"])
        actions, log_probs = dist.sample_and_log_prob(seed=key)
        qs = critic.apply_fn(
            {"params": critic.params},
            batch["observations"],
            actions,
            rngs={"noise": noise_key},
        )
        q = qs.mean(axis=0)
        actor_loss = (
            log_probs * temp.apply_fn({"params": temp.params}) - q).mean()
        actor_l2_refs = _filter_actor_l2_refs(actor_init_params)
        raw_l2, _ = _l2_init_metrics(
            _filter_actor_l2_refs(actor_params), actor_l2_refs, "actor")
        weighted_l2 = jnp.where(
            l2_init_enabled, raw_l2 * f32(l2_init_weight), jnp.asarray(0.0, f32))
        return actor_loss + weighted_l2

    actor_grads = jax.grad(actor_loss_fn)(actor.params)
    return critic_grads, actor_grads


class DrQLearner(Agent):
    def __init__(
        self,
        seed: int,
        observations: Union[jnp.ndarray, DatasetDict],
        actions: jnp.ndarray,
        actor_lr: float = 4e-5,
        critic_lr: float = 4e-5,
        temp_lr: float = 4e-5,
        hidden_dims: Sequence[int] = (64, 64, 64),
        model_size: Optional[str] = "size1m",
        cnn_features: Sequence[int] = (32, 32, 32, 32),
        cnn_filters: Sequence[int] = (3, 3, 3, 3),
        cnn_strides: Sequence[int] = (2, 1, 1, 1),
        cnn_padding: str = "VALID",
        latent_dim: int = 50,
        discount: float = 0.99,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        backup_entropy: bool = True,
        critic_reduction: str = "min",
        init_temperature: float = 1.0,
        encoder: str = "d4pg",
        opt: Optional[Dict] = None,
        wsc: Optional[Dict] = None,
        l2_init: Optional[Dict] = None,
        redo: Optional[Dict] = None,
        augmentation_enabled: bool = False,
        augmentation_pad: int = 4,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = actions.shape[-1]
        if model_size is not None:
            if model_size not in SAC_SIZES:
                raise ValueError(
                    f"Unknown model_size '{model_size}'. Valid: {list(SAC_SIZES)}")
            hidden_dims = SAC_SIZES[model_size]

        if target_entropy is None:
            self.target_entropy = -action_dim / 2
        else:
            self.target_entropy = target_entropy

        self.backup_entropy = backup_entropy
        self.critic_reduction = critic_reduction

        self.tau = tau
        self.discount = discount
        self.augmentation_enabled = bool(augmentation_enabled)
        self.augmentation_pad = int(augmentation_pad)

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
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
            optimizer = opt.get('optimizer', 'adam'),
        )
        self._opt_kwargs = opt_kwargs
        wsc = wsc or {}
        self._wsc = FlaxWSC(
            mechanism=wsc.get('mechanism', 'disabled'),
            target=wsc.get('target', 'all'),
            eps=wsc.get('eps', 1e-8),
            factor_min=wsc.get('factor_min', 0.01),
            factor_max=wsc.get('factor_max', 100.0),
        )
        self._pending_wsc_metrics = {}

        l2_init = l2_init or {}
        self._l2_init_enabled = bool(l2_init.get("enabled", False))
        self._l2_init_weight = float(l2_init.get("weight", 2e-5))
        redo = redo or {}
        self._grad_redo_enabled = bool(redo.get("grad_redo_enabled", True))
        self._grad_redo_frequency = int(redo.get("grad_redo_frequency", 1000))
        self._grad_redo_reset_start = int(redo.get("grad_redo_reset_start", 0))
        self._grad_redo_reset_end = int(redo.get("grad_redo_reset_end", 0))
        self._grad_redo_skip_last_layer = bool(
            redo.get("grad_redo_skip_last_layer", False))

        if encoder == "d4pg":
            encoder_def = D4PGEncoder(
                cnn_features, cnn_filters, cnn_strides, cnn_padding
            )
        elif encoder == "resnet":
            encoder_def = ResNetV2Encoder((2, 2, 2, 2))

        policy_def = NormalTanhPolicySiLU(hidden_dims, action_dim)
        actor_def = PixelMultiplexer(
            encoder=encoder_def,
            network=policy_def,
            latent_dim=latent_dim,
            stop_gradient=True,
        )
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor_params, mets = self._wsc.apply(actor_params, "actor")
        self._pending_wsc_metrics.update(mets)
        actor_init_params = copy.deepcopy(actor_params)
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=_make_sac_opt(actor_lr, **opt_kwargs),
        )

        critic_def = StateActionEnsembleSiLU(hidden_dims, num_qs=2)
        critic_def = PixelMultiplexer(
            encoder=encoder_def, network=critic_def, latent_dim=latent_dim
        )
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic_params, mets = self._wsc.apply(critic_params, "critic")
        self._pending_wsc_metrics.update(mets)
        critic_init_params = copy.deepcopy(critic_params)
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=_make_sac_opt(critic_lr, **opt_kwargs),
        )
        target_critic_params = copy.deepcopy(critic_params)

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=_make_sac_opt(temp_lr, **opt_kwargs),
        )

        self._actor = actor
        self._critic = critic
        self._target_critic_params = target_critic_params
        self._actor_init_params = actor_init_params
        self._critic_init_params = critic_init_params
        self._temp = temp
        self._rng = rng

        self._actor_def = actor_def
        self._critic_def = critic_def
        self._temp_def = temp_def
        self._actor_lr = actor_lr
        self._critic_lr = critic_lr
        self._temp_lr = temp_lr
        self._init_temperature = init_temperature
        self._observations_sample = observations
        self._actions_sample = actions

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        batched, squeeze = _batch_pixel_observation(observations)
        actions = _eval_actions_tanh_mean_jit(self._actor, batched)
        actions = np.asarray(actions)
        return actions[0] if squeeze else actions

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        rng, key = jax.random.split(self._rng)
        self._rng = rng
        batched, squeeze = _batch_pixel_observation(observations)
        actions = _sample_actions_jit(key, self._actor, batched)
        actions = np.asarray(actions)
        return actions[0] if squeeze else actions

    def reset_agent(self) -> None:
        self._rng, actor_key, critic_key, temp_key = jax.random.split(self._rng, 4)
        actor_params = self._actor_def.init(
            actor_key, self._observations_sample)["params"]
        actor_params, _ = self._wsc.apply(actor_params, "actor")
        self._actor_init_params = copy.deepcopy(actor_params)
        self._actor = self._actor.replace(
            params=actor_params,
            opt_state=_make_sac_opt(
                self._actor_lr, **self._opt_kwargs).init(actor_params),
            step=0,
        )
        critic_params = self._critic_def.init(
            critic_key, self._observations_sample, self._actions_sample)["params"]
        critic_params, _ = self._wsc.apply(critic_params, "critic")
        self._critic_init_params = copy.deepcopy(critic_params)
        self._critic = self._critic.replace(
            params=critic_params,
            opt_state=_make_sac_opt(
                self._critic_lr, **self._opt_kwargs).init(critic_params),
            step=0,
        )
        self._target_critic_params = copy.deepcopy(critic_params)
        temp_params = self._temp_def.init(temp_key)["params"]
        self._temp = self._temp.replace(
            params=temp_params,
            opt_state=_make_sac_opt(
                self._temp_lr, **self._opt_kwargs).init(temp_params),
            step=0,
        )

    def _collect_actor_acts(self, obs: np.ndarray) -> Dict:
        _, state = self._actor.apply_fn(
            {'params': self._actor.params},
            obs,
            mutable=['intermediates'],
        )
        return _flatten_intermediates(state.get('intermediates', {}))

    def _collect_critic_acts(self, obs: np.ndarray, actions: np.ndarray) -> Dict:
        _, state = self._critic.apply_fn(
            {'params': self._critic.params},
            obs,
            actions,
            mutable=['intermediates'],
            rngs={'noise': jax.random.PRNGKey(0)},
        )
        return _flatten_intermediates(state.get('intermediates', {}))

    def collect_diagnostics(self, obs: np.ndarray, actions: Optional[np.ndarray] = None) -> Dict:
        info = {}
        batch = None
        if actions is None and isinstance(obs, (dict, FrozenDict)):
            if "observations" in obs and "actions" in obs:
                batch = obs
                actions = obs["actions"]
                obs = obs["observations"]
        actor_acts = self._collect_actor_acts(obs)
        critic_acts = self._collect_critic_acts(obs, actions)
        info.update(_host_linear_wb_fnorm_metrics(self._actor.params, 'actor'))
        info.update(_host_linear_wb_fnorm_metrics(self._critic.params, 'critic'))
        info.update(_host_activation_redo_metrics(actor_acts, 'actor'))
        info.update(_host_activation_redo_metrics(critic_acts, 'critic'))
        info.update(_host_norm_output_metrics(actor_acts, 'actor'))
        info.update(_host_norm_output_metrics(critic_acts, 'critic'))
        if self._grad_redo_enabled and batch is not None:
            critic_grads, actor_grads = _grad_trees_jit(
                self._rng,
                self._actor,
                self._critic,
                self._target_critic_params,
                self._temp,
                self._actor_init_params,
                self._critic_init_params,
                batch,
                self.discount,
                self.target_entropy,
                self.backup_entropy,
                self.critic_reduction,
                self.augmentation_pad,
                self.augmentation_enabled,
                self._l2_init_enabled,
                self._l2_init_weight,
            )
            info.update(_host_grad_redo_metrics(
                self._critic.params, critic_grads, 'critic',
                self._grad_redo_skip_last_layer))
            info.update(_host_grad_redo_metrics(
                self._actor.params, actor_grads, 'actor',
                self._grad_redo_skip_last_layer))
        return info

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        (
            new_rng,
            new_actor,
            new_critic,
            new_target_critic_params,
            new_temp,
            info,
        ) = _update_jit(
            self._rng,
            self._actor,
            self._critic,
            self._target_critic_params,
            self._temp,
            self._actor_init_params,
            self._critic_init_params,
            batch,
            self.discount,
            self.tau,
            self.target_entropy,
            self.backup_entropy,
            self.critic_reduction,
            self.augmentation_pad,
            self.augmentation_enabled,
            self._l2_init_enabled,
            self._l2_init_weight,
        )

        self._rng = new_rng
        self._actor = new_actor
        self._critic = new_critic
        self._target_critic_params = new_target_critic_params
        self._temp = new_temp

        wsc_info = {}
        actor_params, mets = self._wsc.apply(self._actor.params, 'actor')
        if mets:
            self._actor = self._actor.replace(params=actor_params)
            wsc_info.update(mets)
        critic_params, mets = self._wsc.apply(self._critic.params, 'critic')
        if mets:
            self._critic = self._critic.replace(params=critic_params)
            wsc_info.update(mets)
            target_params, _ = self._wsc.apply(
                self._target_critic_params, 'critic_target')
            self._target_critic_params = target_params
        if self._pending_wsc_metrics:
            wsc_info.update(self._pending_wsc_metrics)
            self._pending_wsc_metrics = {}
        info.update(wsc_info)

        return info

    def checkpoint_state(self) -> Dict:
        return {
            "rng": self._rng,
            "actor": self._actor,
            "critic": self._critic,
            "target_critic_params": self._target_critic_params,
            "actor_init_params": self._actor_init_params,
            "critic_init_params": self._critic_init_params,
            "temp": self._temp,
        }

    def restore_checkpoint_state(self, state: Dict) -> None:
        self._rng = state["rng"]
        self._actor = state["actor"]
        self._critic = state["critic"]
        self._target_critic_params = state["target_critic_params"]
        self._actor_init_params = state["actor_init_params"]
        self._critic_init_params = state["critic_init_params"]
        self._temp = state["temp"]
        self._pending_wsc_metrics = {}

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = serialization.to_bytes(self.checkpoint_state())
        fd, tmp_path = tempfile.mkstemp(
            prefix=".agent-", suffix=".msgpack",
            dir=os.path.dirname(path) or ".")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def restore_checkpoint(self, path: str) -> None:
        with open(path, "rb") as handle:
            payload = handle.read()
        state = serialization.from_bytes(self.checkpoint_state(), payload)
        self.restore_checkpoint_state(state)


def _flatten_intermediates(intermediates):
    flat = {}

    def _walk(node, prefix):
        if not (isinstance(node, dict) or hasattr(node, 'items')):
            flat[prefix] = node[0] if isinstance(node, tuple) else node
            return
        for key, value in node.items():
            _walk(value, f'{prefix}/{key}' if prefix else key)

    _walk(intermediates, '')
    return flat


def _host_linear_wb_fnorm_metrics(params, module: str) -> Dict[str, float]:
    metrics = {}
    flat = _flatten_param_paths(params)
    for key, kernel in flat.items():
        if not key.endswith('/kernel') or getattr(kernel, 'ndim', 0) < 2:
            continue
        path = key[:-len('/kernel')]
        total = np.square(np.asarray(jax.device_get(kernel), dtype=np.float32)).sum()
        bias = flat.get(f'{path}/bias')
        if bias is not None:
            total += np.square(np.asarray(jax.device_get(bias), dtype=np.float32)).sum()
        lname = f'{module}_{_metric_name(path)}'
        metrics[f'act_redo/Linear_WB_FNorm/{lname}'] = float(np.sqrt(total))
    return metrics


def _effective_rank_np(sv: np.ndarray) -> float:
    probs = np.abs(sv) / (np.sum(np.abs(sv)) + 1e-8)
    positive = probs > 0.0
    entropy = -np.sum(probs[positive] * np.log(probs[positive]))
    return float(np.exp(entropy))


def _stable_rank_np(sv: np.ndarray, threshold: float = 0.99) -> float:
    probs = np.abs(sv) / (np.sum(np.abs(sv)) + 1e-8)
    return float(np.sum(np.cumsum(probs) < threshold) + 1)


def _activation_singular_values_np(act_2d: np.ndarray) -> np.ndarray:
    if act_2d.shape[0] < act_2d.shape[1]:
        return np.linalg.svd(act_2d, compute_uv=False)
    gram = np.matmul(act_2d.T, act_2d)
    eigvals = np.linalg.eigvalsh(gram)
    return np.sqrt(np.maximum(eigvals, 0.0))


def _host_activation_redo_metrics(activations: Dict, module: str) -> Dict[str, float]:
    metrics = {}
    for path, act in activations.items():
        if not path.endswith("_act"):
            continue
        act = np.asarray(jax.device_get(act), dtype=np.float32)
        if act.ndim < 2:
            continue
        act_2d = act.reshape((-1, act.shape[-1]))
        score = np.abs(act_2d).mean(axis=0)
        norm_score = score / (score.mean() + 1e-9)
        lname = f"{module}_{_metric_name(path[:-len('_act')])}"
        for tau in _REDO_TAUS:
            metrics[f"act_redo/Dormant_{tau}/{lname}"] = float(
                (norm_score <= tau).mean() * 100.0)
        metrics[f"act_redo/Act_Mean/{lname}"] = float(score.mean())
        sv = _activation_singular_values_np(act_2d)
        metrics[f"act_redo/erank/{lname}"] = _effective_rank_np(sv)
        metrics[f"act_redo/srank/{lname}"] = _stable_rank_np(sv)
    return metrics


def _host_grad_redo_metrics(
    params: Params,
    grads: Params,
    module: str,
    skip_last_layer: bool,
) -> Dict[str, float]:
    metrics = {}
    flat_grads = _flatten_param_paths(grads)
    flat_params = _flatten_param_paths(params)
    for key, grad in flat_grads.items():
        if not key.endswith("/kernel") or getattr(grad, "ndim", 0) < 2:
            continue
        if flat_params.get(key) is None:
            continue
        grad_np = np.asarray(jax.device_get(grad), dtype=np.float32)
        path = key[:-len("/kernel")]
        if skip_last_layer and not _has_following_rmsnorm(path, params):
            continue
        base_lname = _metric_name(path)
        is_vmapped_dense = (
            grad_np.ndim == 3 and ("Vmap" in path or "Ensemble" in path)
        )
        members = grad_np.shape[0] if is_vmapped_dense else 1
        for idx in range(members):
            g = grad_np[idx] if is_vmapped_dense else grad_np
            score = np.abs(g).mean(axis=tuple(range(g.ndim - 1)))
            norm_score = score / (score.mean() + 1e-9)
            lname = (
                f"{module}_{idx}_{base_lname}"
                if is_vmapped_dense else f"{module}_{base_lname}"
            )
            for tau in _REDO_TAUS:
                metrics[f"grad_redo/GradDormant_{tau}/{lname}"] = float(
                    (norm_score <= tau).mean() * 100.0)
            metrics[f"grad_redo/Grad_Mean/{lname}"] = float(score.mean())
    return metrics


def _host_norm_output_metrics(activations: Dict, module: str) -> Dict[str, float]:
    metrics = {}
    for path, act in activations.items():
        leaf = path.split('/')[-1]
        if not leaf.endswith('_out'):
            continue
        act = np.asarray(jax.device_get(act), dtype=np.float32)
        if act.ndim < 2:
            continue
        act_2d = act.reshape((-1, act.shape[-1]))
        lname = f'{module}_{_metric_name(path)}'
        mean_l2 = float(np.linalg.norm(act_2d, axis=-1).mean())
        if (
            leaf.startswith('norm_') or
            (leaf.endswith('_norm_out') and
             (leaf.startswith('Dense_') or leaf.startswith('Conv_')))
        ):
            metrics[f'act_redo/RMSNorm_Out_L2_Mean/{lname}'] = mean_l2
        elif leaf.startswith('LayerNorm_'):
            metrics[f'act_redo/LayerNorm_Out_L2_Mean/{lname}'] = mean_l2
    return metrics


def _batch_pixel_observation(observation):
    if not isinstance(observation, (dict, FrozenDict)):
        return observation, False
    pixels = observation.get("pixels")
    if pixels is None or getattr(pixels, "ndim", 0) != 4:
        return observation, False
    if isinstance(observation, FrozenDict):
        batched = observation.copy(add_or_replace={"pixels": pixels[None]})
    else:
        batched = dict(observation)
        batched["pixels"] = pixels[None]
    return batched, True
