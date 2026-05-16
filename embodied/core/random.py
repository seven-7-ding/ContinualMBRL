import numpy as np


class RandomAgent:

  def __init__(self, obs_space, act_space, seed=0):
    self.obs_space = obs_space
    self.act_space = act_space
    self.rng = np.random.default_rng(seed)

  def init_policy(self, batch_size):
    return ()

  def init_train(self, batch_size):
    return ()

  def init_report(self, batch_size):
    return ()

  def policy(self, carry, obs, mode='train'):
    batch_size = len(obs['is_first'])
    act = {
        k: np.stack([self._sample(v) for _ in range(batch_size)])
        for k, v in self.act_space.items() if k != 'reset'}
    return carry, act, {}

  def _sample(self, space):
    low, high = space.low, space.high
    if np.issubdtype(space.dtype, np.floating):
      low = np.maximum(np.ones(space.shape) * np.finfo(space.dtype).min, low)
      high = np.minimum(np.ones(space.shape) * np.finfo(space.dtype).max, high)
    if space.discrete:
      if space.dtype == bool:
        return self.rng.integers(0, 2, space.shape).astype(space.dtype)
      return self.rng.integers(low, high, space.shape).astype(space.dtype)
    return self.rng.uniform(low, high, space.shape).astype(space.dtype)

  def train(self, carry, data):
    return carry, {}, {}

  def report(self, carry, data):
    return carry, {}

  def stream(self, st):
    return st

  def save(self):
    return None

  def load(self, data=None):
    pass
