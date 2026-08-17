from typing import Dict, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from jaxrl2.networks.constants import default_init


class PixelMultiplexer(nn.Module):
    encoder: nn.Module
    network: nn.Module
    latent_dim: int
    stop_gradient: bool = False

    @nn.compact
    def __call__(
        self,
        observations: Union[FrozenDict, Dict],
        actions: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        observations = FrozenDict(observations)
        assert (
            len(observations.keys()) <= 2
        ), "Can include only pixels and states fields."

        pixels = observations["pixels"]
        x = self.encoder(pixels)

        if self.stop_gradient:
            # We do not update conv layers with policy gradients.
            x = jax.lax.stop_gradient(x)

        x = nn.Dense(
            self.latent_dim, kernel_init=default_init(), name="Dense_0")(x)
        self.sow("intermediates", "Dense_0_act", x)
        x = nn.RMSNorm(name="Dense_0_norm")(x)
        self.sow("intermediates", "Dense_0_norm_out", x)
        x = nn.tanh(x)

        if "states" in observations:
            y = nn.Dense(
                self.latent_dim, kernel_init=default_init(), name="Dense_1")(
                observations["states"]
            )
            self.sow("intermediates", "Dense_1_act", y)
            y = nn.RMSNorm(name="Dense_1_norm")(y)
            self.sow("intermediates", "Dense_1_norm_out", y)
            y = nn.tanh(y)

            x = jnp.concatenate([x, y], axis=-1)

        if actions is None:
            return self.network(x, training=training)
        else:
            return self.network(x, actions, training=training)
