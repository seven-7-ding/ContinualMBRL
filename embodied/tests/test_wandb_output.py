import re
import types

import numpy as np

from embodied.core.wandb_output import StepAlignedWandBOutput


def test_commits_exact_steps_and_preserves_all_media():
  output = StepAlignedWandBOutput.__new__(StepAlignedWandBOutput)
  output._pattern = re.compile('.*')
  calls = []
  output._wandb = types.SimpleNamespace(
      log=lambda metrics, **kwargs: calls.append((metrics, kwargs)),
      Histogram=lambda value: ('histogram', value),
      Image=lambda value: ('image', value),
      Video=lambda value: ('video', value))
  output(((10000, 'env_step', np.array(10000)),
          (10000, 'loss/policy', np.array(2.5)),
          (10000, 'text', 'diagnostics'),
          (10000, 'hist', np.zeros(4)),
          (10000, 'gray', np.zeros((4, 5))),
          (10000, 'rgb', np.ones((4, 5, 3))),
          (10000, 'video', np.zeros((2, 4, 5, 3), np.uint8)),
          (20000, 'env_step', np.array(20000))))
  assert [options for _, options in calls] == [
      {'step': 10000, 'commit': True}, {'step': 20000, 'commit': True}]
  metrics = calls[0][0]
  assert metrics['env_step'] == 10000 and metrics['loss/policy'] == 2.5
  assert metrics['text'] == 'diagnostics'
  assert metrics['hist'][0] == 'histogram'
  assert metrics['gray'][1].shape == (1, 4, 5)
  assert metrics['rgb'][1].shape == (3, 4, 5)
  assert metrics['rgb'][1].dtype == np.uint8
  assert np.all(metrics['rgb'][1] == 255)
  assert metrics['video'][1].shape == (2, 3, 4, 5)
