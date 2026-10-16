"""Commit each environment-step log immediately, including media."""
import collections

import elements
import numpy as np


class StepAlignedWandBOutput(elements.logger.WandBOutput):

  def __call__(self, summaries):
    bystep = collections.defaultdict(dict)
    for step, name, value in summaries:
      if not self._pattern.search(name):
        continue
      if isinstance(value, str):
        converted = value
      elif value.ndim == 0:
        converted = float(value)
      elif value.ndim == 1:
        converted = self._wandb.Histogram(value)
      else:
        if value.ndim == 2:
          value = value[..., None]
        assert value.ndim in (3, 4) and value.shape[-1] in (1, 3, 4), value.shape
        if value.dtype != np.uint8:
          value = (255 * np.clip(value, 0, 1)).astype(np.uint8)
        if value.ndim == 3:
          converted = self._wandb.Image(value.transpose(2, 0, 1))
        else:
          converted = self._wandb.Video(value.transpose(0, 3, 1, 2))
      bystep[step][name] = converted
    for step, metrics in bystep.items():
      # With an explicit step, W&B defaults to commit=False. Otherwise the
      # latest 10k boundary stays pending until the next training log.
      self._wandb.log(metrics, step=step, commit=True)
