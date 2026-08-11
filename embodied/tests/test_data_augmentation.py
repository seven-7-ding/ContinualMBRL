import jax.numpy as jnp
import numpy as np

from dreamerv3.agent import (
    _normalize_data_augmentation_mode,
    _random_shift_images,
)


def test_random_shift_images_uses_replicate_padding_crop():
  image = jnp.arange(4, dtype=jnp.uint8).reshape((1, 1, 2, 2, 1))

  top_left = _random_shift_images(
      image, jnp.array([[[0, 0]]], jnp.int32), pad=1)
  center = _random_shift_images(
      image, jnp.array([[[1, 1]]], jnp.int32), pad=1)
  bottom_right = _random_shift_images(
      image, jnp.array([[[2, 2]]], jnp.int32), pad=1)

  np.testing.assert_array_equal(
      np.asarray(top_left[0, 0, ..., 0]), np.array([[0, 0], [0, 0]]))
  np.testing.assert_array_equal(
      np.asarray(center[0, 0, ..., 0]), np.array([[0, 1], [2, 3]]))
  np.testing.assert_array_equal(
      np.asarray(bottom_right[0, 0, ..., 0]), np.array([[3, 3], [3, 3]]))


def test_data_augmentation_mode_aliases():
  assert _normalize_data_augmentation_mode('disabled') == 'disabled'
  assert _normalize_data_augmentation_mode(
      'data_augmentation_batch_align') == 'batch_align'
  assert _normalize_data_augmentation_mode(
      'data_augmentation_batch_aug') == 'batch_aug'
