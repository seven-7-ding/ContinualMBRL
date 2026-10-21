from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from copy import deepcopy
import threading
import weakref

import elements
import numpy as np
import pytest

from embodied.core.replay import Replay


@pytest.fixture(autouse=True)
def uuid_mode():
  elements.UUID.reset(debug=False)
  yield
  elements.UUID.reset(debug=False)


def make_saved(directory):
  replay = Replay(3, capacity=64, directory=directory, chunksize=4,
                  save_wait=True, seed=17)
  for step in range(40):
    replay.add(dict(step=np.int32(step), state=np.full(8, step, np.float32)))
  replay.save()
  replay.workers.shutdown(wait=True)
  return replay


@pytest.mark.parametrize('count_limit', [0, 2])
def test_byte_budget_keeps_sequences_and_sampling_order(tmp_path, count_limit):
  reference = make_saved(tmp_path)
  replay = Replay(3, capacity=64, directory=tmp_path, chunksize=4, seed=17,
                  cache_chunks=count_limit, cache_bytes=512)
  replay.load()
  # Restore is lazy even when byte budget is the only enabled limit.
  assert all(chunk.data is None for chunk in replay.chunks.values())
  for _ in range(20):
    expected, actual = reference.sample(4), replay.sample(4)
    for key in expected:
      np.testing.assert_array_equal(actual[key], expected[key])
    assert sum(c.nbytes for c in replay.chunks.values()) <= 512
    if count_limit:
      assert sum(c.data is not None for c in replay.chunks.values()) <= count_limit
  replay.workers.shutdown(wait=True)


def test_eviction_preserves_dirty_recurrent_states_and_resume(tmp_path):
  make_saved(tmp_path)
  replay = Replay(3, capacity=64, directory=tmp_path, chunksize=4, seed=17,
                  cache_bytes=256)
  replay.load()
  sample = replay.sample(1)
  expected = sample['state'] + 1000
  replay.update(dict(stepid=sample['stepid'], state=expected))
  # Force dirty chunks out without dropping the replay items or sampler state.
  for _ in range(30):
    replay.sample(4)
  replay.save()
  replay.workers.shutdown(wait=True)
  restored = Replay(3, capacity=64, directory=tmp_path, chunksize=4,
                    cache_bytes=256)
  restored.load()
  first = sample['stepid'][0, 0].tobytes()
  actual = restored._getseq(elements.UUID(first[:-4]), int.from_bytes(first[-4:], 'big'))
  np.testing.assert_array_equal(actual['state'], expected[0])
  restored.workers.shutdown(wait=True)


def test_live_insert_remove_clear_respects_budget(tmp_path):
  replay = Replay(3, capacity=8, directory=tmp_path, chunksize=4, cache_bytes=512)
  for step in range(60):
    replay.add(dict(step=np.int32(step), state=np.full(8, step, np.float32)))
    if len(replay):
      replay.sample(2)
    # Active append chunks remain resident even if larger than the cache budget.
    active = {uuid for uuid, _ in replay.current.values()}
    active_bytes = sum(replay.chunks[x].nbytes for x in active)
    assert sum(c.nbytes for c in replay.chunks.values()) <= max(512, active_bytes)
  replay.clear()
  assert replay.loaded_bytes == 0
  assert not replay.loaded
  replay.workers.shutdown(wait=True)


def test_concurrent_sample_writeback_has_no_cache_accounting_drift(tmp_path):
  make_saved(tmp_path)
  replay = Replay(3, capacity=64, directory=tmp_path, chunksize=4, cache_bytes=768)
  replay.load()
  def update(_):
    for _ in range(15):
      sample = replay.sample(2)
      np.testing.assert_array_equal(np.diff(sample['step'], axis=1), 1)
      replay.update(dict(stepid=sample['stepid'], state=sample['state']))
  with ThreadPoolExecutor(3) as pool:
    list(pool.map(update, range(3)))
  replay._evict_chunks()
  assert replay.loaded_bytes == sum(c.nbytes for c in replay.chunks.values())
  assert replay.loaded_bytes <= 768
  replay.workers.shutdown(wait=True)


def test_partial_stream_checkpoint_releases_padding_and_preserves_data(tmp_path):
  replay = Replay(3, capacity=4096, directory=tmp_path, chunksize=1024,
                  cache_bytes=64*1024**2, seed=17)
  for step in range(12):
    for worker in range(16):
      replay.add(dict(state=np.full(32, step, np.float32),
          reward=np.float32(step), action=np.zeros(6, np.float32),
          is_first=np.bool_(False), is_last=np.bool_(False),
          is_terminal=np.bool_(False),
          **{'dyn/deter':np.full(512, worker+step, np.float32),
             'dyn/stoch':np.zeros((32,4), np.float32)}), worker)
  chunk = next(iter(replay.chunks.values()))
  old_buffer = weakref.ref(chunk.data['dyn/deter'])
  allocated = sum(c.nbytes for c in replay.chunks.values())
  state = deepcopy(replay.sampler.rng.bit_generator.state)
  before = replay.sample(16)
  replay.save()
  assert old_buffer() is None  # A prefix view would retain the full allocation.
  assert sum(c.nbytes for c in replay.chunks.values()) < allocated/16
  assert replay.loaded_bytes == sum(c.nbytes for c in replay.chunks.values())
  replay.sampler.rng.bit_generator.state = state
  after = replay.sample(16)
  for key in before:
    np.testing.assert_array_equal(after[key], before[key])
  expected = np.full_like(after['dyn/deter'], 777)
  replay.update(dict(stepid=after['stepid'], **{'dyn/deter':expected}))
  replay.save()
  replay.workers.shutdown(wait=True)
  restored = Replay(3, capacity=4096, directory=tmp_path, cache_bytes=64*1024**2)
  restored.load()
  first = after['stepid'][0,0].tobytes()
  seq = restored._getseq(elements.UUID(first[:-4]), int.from_bytes(first[-4:],'big'))
  np.testing.assert_array_equal(seq['dyn/deter'], expected[0])
  restored.workers.shutdown(wait=True)


def test_compaction_and_save_do_not_lose_concurrent_state_writeback(tmp_path):
  replay = Replay(3, capacity=64, directory=tmp_path, chunksize=16,
                  cache_bytes=65536)
  for step in range(6):
    replay.add(dict(step=np.int32(step),state=np.full(8,step,np.float32)))
  sample = replay.sample(1)
  copied, release, updating = threading.Event(), threading.Event(), threading.Event()
  class PausedCopy(np.ndarray):
    def copy(self, order='C'):
      result = np.array(self, copy=True, order=order, subok=False)
      copied.set()
      assert release.wait(5)
      return result
  chunk = next(iter(replay.chunks.values()))
  chunk.data['state'] = chunk.data['state'].view(PausedCopy)
  expected = np.full_like(sample['state'],999)
  def writeback():
    updating.set()
    replay.update(dict(stepid=sample['stepid'],state=expected))
  with ThreadPoolExecutor(2) as pool:
    save = pool.submit(replay.save)
    assert copied.wait(5)
    write = pool.submit(writeback)
    try:
      assert updating.wait(5)
      with pytest.raises(FutureTimeout):
        write.result(timeout=.05)
    finally:
      release.set()
    save.result(timeout=5)
    write.result(timeout=5)
  replay.save()
  replay.workers.shutdown(wait=True)
  restored = Replay(3, capacity=64, directory=tmp_path, cache_bytes=65536)
  restored.load()
  first = sample['stepid'][0,0].tobytes()
  seq = restored._getseq(elements.UUID(first[:-4]),int.from_bytes(first[-4:],'big'))
  np.testing.assert_array_equal(seq['state'],expected[0])
  restored.workers.shutdown(wait=True)
