"""Bounded CPU replay benchmark using production state/context tensor shapes."""
import json
import sys
import time
import tracemalloc
from copy import deepcopy
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from embodied.core.replay import Replay


def main():
  replay = Replay(length=65, capacity=4096, chunksize=1024, seed=1000)
  rng = np.random.default_rng(1000)
  for index in range(4096):
    replay.add(dict(state=rng.normal(size=32).astype('float32'),
        reward=np.float32(index), action=np.zeros(6, np.float32),
        is_first=np.bool_(index % 100 == 0), is_last=np.bool_(False),
        is_terminal=np.bool_(False),
        **{'dyn/deter': np.zeros(512, np.float32),
           'dyn/stoch': np.zeros((32, 4), np.float32)}))
  state = deepcopy(replay.sampler.rng.bit_generator.state)
  sequential = [replay.sample(16) for _ in range(8)]
  replay.sampler.rng.bit_generator.state = state
  combined = replay.sample(16 * 8)
  for key in combined:
    np.testing.assert_array_equal(combined[key], np.concatenate([x[key] for x in sequential]))
  payload = sum(x.nbytes for x in sequential[0].values())
  result = dict(batch_bytes=payload, group8_bytes=payload * 8,
                draws_and_data_equal=True, measurements=[])
  del sequential, combined
  for traced in (False, True, False):
    tracemalloc.start() if traced else tracemalloc.stop()
    for group in (1, 8):
      begin = time.perf_counter()
      for _ in range(32):
        for _ in range(8 // group):
          data = replay.sample(16 * group)
          # Same finite input check used by the production agent stream.
          for value in data.values():
            if np.issubdtype(value.dtype, np.floating):
              assert not np.isnan(value).any()
      row = dict(tracemalloc=traced, sample_group=group,
                 milliseconds_per_batch=(time.perf_counter()-begin)/256*1000,
                 rss_gib=psutil.Process().memory_info().rss/2**30)
      result['measurements'].append(row)
      print(json.dumps(row), flush=True)
  tracemalloc.stop()
  folder = ROOT/'logdir/two_player_sweep_20261015/benchmark_pipeline_20261019'
  folder.mkdir(exist_ok=True)
  (folder/'replay_diagnostics.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
  main()
