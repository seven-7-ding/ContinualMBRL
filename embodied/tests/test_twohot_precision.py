import jax.numpy as jnp
import numpy as np

from embodied.jax.outs import TwoHot


def test_twohot_preserves_tiny_logit_differences_with_large_bins():
  logits = np.array([1e-8, 0, -1e-8], np.float32)
  bins = np.array([-np.expm1(20), 0, np.expm1(20)], np.float32)
  probs = np.exp(logits.astype(np.float64))
  expected = (probs * bins).sum() / probs.sum()
  actual = np.asarray(TwoHot(jnp.array(logits), jnp.array(bins)).pred())
  np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
  assert float(actual) < -3


def test_twohot_matches_general_categorical_expectation():
  for bins in ([-2, -0.3, 0, 1, 4], [-2, -0.3, 1, 4]):
    bins = np.array(bins, np.float32)
    logits = np.linspace(-1, 2, len(bins), dtype=np.float32)
    probs = np.exp(logits.astype(np.float64))
    expected = (probs * bins).sum() / probs.sum()
    actual = np.asarray(TwoHot(jnp.array(logits), jnp.array(bins)).pred())
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
