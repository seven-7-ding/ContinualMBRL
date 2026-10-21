"""Measure checkpoint padding on private 16-stream replay; no CUDA/live writes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import elements
from embodied.core.replay import Replay


class LegacyReplay(Replay):
  def _compact_chunk(self, chunk):
    pass

  def _setseq(self, chunkid, index, values):
    return self._setseq_locked(chunkid,index,values)


def main(mode):
  elements.UUID.reset(debug=True)
  resultdir=ROOT/'logdir/two_player_sweep_20261015/benchmark_compaction_20261019'
  resultdir.mkdir(parents=True,exist_ok=True)
  digest=hashlib.sha256()
  rng=np.random.default_rng(1000)
  with tempfile.TemporaryDirectory(prefix=mode+'_',dir=resultdir) as private:
    cls=Replay if mode=='compact' else LegacyReplay
    replay=cls(65,capacity=500000,directory=private,chunksize=1024,
               cache_bytes=2*1024**3,seed=1000)
    checkpoint_seconds=[]
    def sample_update(hash_data=False):
      data=replay.sample(16)
      if hash_data:
        for key in sorted(data):
          digest.update(key.encode());digest.update(data[key].tobytes())
      replay.update(dict(stepid=data['stepid'][:,1:].copy(),
          **{'dyn/deter':data['dyn/deter'][:,1:]+np.float32(.001),
             'dyn/stoch':data['dyn/stoch'][:,1:].copy()}))
      return data
    for cycle in range(8):
      for step in range(128):
        for worker in range(16):
          replay.add(dict(state=rng.normal(size=32).astype(np.float32),
              action=np.full(6,worker,np.float32),reward=np.float32(step+cycle*128),
              is_first=np.bool_(step==0 and cycle==0),is_last=np.bool_(False),
              is_terminal=np.bool_(False),
              **{'dyn/deter':rng.normal(size=512).astype(np.float32),
                 'dyn/stoch':rng.normal(size=(32,4)).astype(np.float32)}),worker)
      for _ in range(8):sample_update(hash_data=True)
      begin=time.perf_counter();replay.save()
      checkpoint_seconds.append(time.perf_counter()-begin)
    begin=time.perf_counter()
    for _ in range(128):data=sample_update()
    milliseconds=1000*(time.perf_counter()-begin)/128
    for key in sorted(data):digest.update(data[key].tobytes())
    assert all(np.isfinite(v).all() for v in data.values())
    replay.save()
    result=dict(mode=mode,streams=16,chunksize=1024,valid_steps=8*128*16,
        checkpoint_cycles=8,sample_batch=16,sample_length=65,cache_limit_gib=2,
        cached_gib=sum(c.nbytes for c in replay.chunks.values())/2**30,
        rss_gib=psutil.Process().memory_info().rss/2**30,
        mean_checkpoint_ms=1000*sum(checkpoint_seconds)/len(checkpoint_seconds),
        sample_writeback_ms=milliseconds,sequence_and_state_digest=digest.hexdigest())
    replay.workers.shutdown(wait=True)
  (resultdir/(mode+'.json')).write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps(result),flush=True)


if __name__=='__main__':
  parser=argparse.ArgumentParser()
  parser.add_argument('--mode',choices=['legacy','compact'],required=True)
  main(parser.parse_args().mode)
