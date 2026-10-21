"""End-to-end updates on private replay copies and a bounded prefetch queue."""
import json
import resource
import shutil
import time
import tracemalloc

import jax
import numpy as np
import psutil

import elements
from embodied.core.replay import Replay
from dreamerv3.main import make_stream


def benchmark(agent, config, directory, data, carry):
  # Compile without tracing so timings measure execution, not compiler work.
  for mode in ('wm', 'agent'):
    carry, _ = agent.train_many(carry, data[:8], mode)
    jax.block_until_ready(agent.params)
  initial = agent.save()
  result = {'minibatch': config.batch_size, 'length': config.batch_length,
            'fused_updates': 8, 'prefetch_batches': 8, 'measurements': []}
  snapshot = directory/'best_response_32'/'source_snapshot'
  assert snapshot.is_dir(), 'Run benchmark_replay_disk.py first'
  elements.UUID.reset(debug=True)
  for count, budget in ((1024, 0), (0, 2 * 1024**3)):
    tracemalloc.stop()
    private = directory/f'training_cache{count}_bytes{budget}'
    assert not private.exists(), private
    shutil.copytree(snapshot, private)
    replay = Replay(65, capacity=500000, directory=str(private),
                    chunksize=1024, seed=1000, cache_chunks=count,
                    cache_bytes=budget)
    replay.load()
    if budget:
      for uuid in list(replay.chunks):
        replay._ensure_chunk_data(uuid)
        replay._evict_chunks()
    agent.load(initial)
    carry = agent.init_train(config.batch_size)
    stream = iter(agent.stream(make_stream(config, replay, 'train')))
    tracemalloc.start()
    for mode in ('wm', 'agent'):
      def group(carry):
        batches = [next(stream) for _ in range(8)]
        carry, outs = agent.train_many(carry, batches, mode)
        for index in range(8):
          replay.update({key: value[index] for key, value in outs['replay'].items()})
        return carry
      carry = group(carry)
      begin = time.perf_counter()
      for _ in range(8):
        carry = group(carry)
      jax.block_until_ready(agent.params)
      seconds = time.perf_counter()-begin
      row = dict(mode=mode, cache_chunks=count, cache_bytes=budget,
                 updates_per_second=64/seconds,
                 cached_gib=sum(c.nbytes for c in list(replay.chunks.values()))/2**30,
                 rss_gib=psutil.Process().memory_info().rss/2**30,
                 process_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
                 jax_memory=agent.train_devices[0].memory_stats())
      assert all(np.isfinite(v) for k, v in agent.latest_train_metrics().items()
                 if k.startswith('loss/'))
      result['measurements'].append(row)
      (directory/'replay_training.json').write_text(json.dumps(result, indent=2)+'\n')
      print('REPLAY_TRAINING', row, flush=True)
    tracemalloc.stop()
  return result
