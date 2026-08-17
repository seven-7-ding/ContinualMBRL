"""Pixel augmentations used by DrQ/DrQ-v2 agents."""

import jax
import jax.numpy as jnp


def random_shift(key, img, pad=4):
    """DrQ-v2 random shift for one NHW... image stack.

    This is the JAX/NHWC equivalent of facebookresearch/drqv2's
    RandomShiftsAug(pad=4): edge/replicate-pad height and width, then sample a
    uniformly random spatial shift in [0, 2 * pad].
    """
    crop_from = jax.random.randint(key, (2,), 0, 2 * pad + 1)
    crop_from = jnp.concatenate([
        crop_from,
        jnp.zeros((img.ndim - 2,), dtype=jnp.int32),
    ])
    pad_width = ((pad, pad), (pad, pad)) + ((0, 0),) * (img.ndim - 2)
    padded_img = jnp.pad(img, pad_width, mode="edge")
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


def batched_random_shift(key, imgs, pad=4):
    keys = jax.random.split(key, imgs.shape[0])
    return jax.vmap(random_shift, (0, 0, None))(keys, imgs, pad)


# Backward-compatible aliases for older DrQ code.
random_crop = random_shift
batched_random_crop = batched_random_shift
