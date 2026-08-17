from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from jaxrl2.networks.constants import default_init


class D4PGEncoder(nn.Module):
    features: Sequence[int] = (32, 32, 32, 32)
    filters: Sequence[int] = (2, 1, 1, 1)
    strides: Sequence[int] = (2, 1, 1, 1)
    padding: str = "VALID"

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        assert len(self.features) == len(self.strides)

        x = observations.astype(jnp.float32) / 255.0
        x = jnp.reshape(x, (*x.shape[:-2], -1))

        for idx, (features, filter_, stride) in enumerate(
            zip(self.features, self.filters, self.strides)
        ):
            x = nn.Conv(
                features,
                kernel_size=(filter_, filter_),
                strides=(stride, stride),
                kernel_init=default_init(),
                padding=self.padding,
                name=f"Conv_{idx}",
            )(x)
            x = nn.RMSNorm(name=f"Conv_{idx}_norm")(x)
            self.sow("intermediates", f"Conv_{idx}_norm_out", x)
            x = nn.silu(x)
            self.sow("intermediates", f"conv_{idx}_act", x)

        return x.reshape((*x.shape[:-3], -1))
