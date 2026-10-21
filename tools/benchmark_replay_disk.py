"""Profile fragmented replay on private copies; never write to live replay."""
import argparse
import json
import shutil
import sys
import time
import tracemalloc
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import elements
from embodied.core.replay import Replay


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--setting', default='best_response_32')
  args = parser.parse_args()
  control = ROOT/'logdir/two_player_sweep_20261015'
  jobs = json.loads((control/'manifest.json').read_text())['jobs']
  job = next(x for x in jobs if x['setting'] == args.setting and x['seed'] == 1000)
  # Match the existing runs' debug UUID serialization (decimal filenames).
  elements.UUID.reset(debug=True)
  directory = control/'benchmark_pipeline_20261019'/args.setting
  directory.mkdir(parents=True, exist_ok=True)
  snapshot = directory/'source_snapshot'
  if not snapshot.exists():
    snapshot.mkdir()
    for source in (Path(job['logdir'])/'replay').glob('*.npz'):
      shutil.copy2(source, snapshot/source.name)
  result = {'setting': args.setting, 'files': len(list(snapshot.glob('*.npz'))),
            'measurements': []}
  for cache, traced in ((1024, False), (1024, True), (16384, False), (16384, True)):
    # No hard links: replay updates can overwrite a chunk file.
    private = directory/f'cache{cache}_trace{int(traced)}'
    if private.exists():
      shutil.rmtree(private)
    shutil.copytree(snapshot, private)
    replay = Replay(length=65, capacity=1000000, directory=str(private),
                    chunksize=1024, seed=1000, cache_chunks=cache)
    replay.load()
    counters = dict(loads=0, writes=0, load_seconds=0., write_seconds=0.)
    original_ensure = replay._ensure_chunk_data
    original_save = replay._save_chunk_now
    def ensure(uuid):
      missing = replay.chunks[uuid].data is None
      begin = time.perf_counter()
      chunk = original_ensure(uuid)
      if missing:
        counters['loads'] += 1
        counters['load_seconds'] += time.perf_counter()-begin
      return chunk
    def save(chunk):
      dirty = not chunk.saved
      begin = time.perf_counter()
      original_save(chunk)
      if dirty:
        counters['writes'] += 1
        counters['write_seconds'] += time.perf_counter()-begin
    replay._ensure_chunk_data = ensure
    replay._save_chunk_now = save
    # Uniform samples eventually populate the larger cache. Measure warm state
    # without paying compressed-save overhead during the prewarming pass.
    if cache > len(replay.chunks):
      for uuid in list(replay.chunks):
        replay._ensure_chunk_data(uuid)
    counters.update(loads=0, writes=0, load_seconds=0., write_seconds=0.)
    tracemalloc.start() if traced else tracemalloc.stop()
    begin = time.perf_counter()
    for _ in range(32):
      data = replay.sample(16)
      # Production writes these recurrent states back after EVERY update.
      replay.update({key: data[key][:, 1:].copy() for key in
                     ('stepid', 'dyn/deter', 'dyn/stoch')})
    seconds = time.perf_counter()-begin
    row = dict(cache_chunks=cache, tracemalloc=traced,
               milliseconds_per_update=1000*seconds/32,
               cached_gib=sum(c.nbytes for c in replay.chunks.values())/2**30,
               rss_gib=psutil.Process().memory_info().rss/2**30,
               **counters)
    result['measurements'].append(row)
    print(json.dumps(row), flush=True)
    tracemalloc.stop()
    replay.workers.shutdown(wait=True)
    del replay
    (directory/'disk_diagnostics.json').write_text(json.dumps(result, indent=2)+'\n')
  print('RESULT', directory/'disk_diagnostics.json', flush=True)


if __name__ == '__main__':
  main()
