import threading
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial as bind

import elements
import numpy as np

from . import chunk as chunklib
from . import limiters
from . import selectors

class Replay:

  def __init__(
      self, length, capacity=None, directory=None, chunksize=1024,
      online=False, selector=None, save_wait=False, name='unnamed', seed=0,
      cache_chunks=0):

    self.length = length
    self.capacity = capacity
    self.chunksize = chunksize
    self.name = name

    self.sampler = selector or selectors.Uniform(seed)

    self.chunks = {}
    self.refs = {}
    self.refs_lock = threading.RLock()
    self.cache_chunks = int(cache_chunks or 0)
    self.loaded = OrderedDict()

    self.items = {}
    self.fifo = deque()
    self.itemid = 0

    self.current = {}
    self.streams = defaultdict(deque)
    self.rwlock = elements.RWLock()

    self.online = online
    if online:
      self.lengths = defaultdict(int)
      self.queue = deque()

    if directory:
      self.directory = elements.Path(directory)
      self.directory.mkdir()
      self.workers = ThreadPoolExecutor(16, 'replay_saver')
      self.saved = set()
    else:
      self.directory = None
    self.save_wait = save_wait

    self.metrics = {'samples': 0, 'inserts': 0, 'updates': 0}

  def __len__(self):
    return len(self.items)

  def stats(self):
    ratio = lambda x, y: x / y if y else np.nan
    m = self.metrics
    chunk_nbytes = sum(x.nbytes for x in list(self.chunks.values()))
    stats = {
        'items': len(self.items),
        'chunks': len(self.chunks),
        'loaded_chunks': sum(x.data is not None for x in self.chunks.values()),
        'streams': len(self.streams),
        'ram_gb': chunk_nbytes / (1024 ** 3),
        'inserts': m['inserts'],
        'samples': m['samples'],
        'updates': m['updates'],
        'replay_ratio': ratio(self.length * m['samples'], m['inserts']),
    }
    for key in self.metrics:
      self.metrics[key] = 0
    return stats

  def clear(self, disk=False, archive_prefix='cleared'):
    """Clear all transitions from the replay buffer."""
    with self.rwlock.writing:
      if disk and self.directory:
        files = list(self.directory.glob('*.npz'))
        if files:
          archive = self.directory / f'{archive_prefix}_{elements.timestamp(millis=True)}'
          archive.mkdir()
          for filename in files:
            filename.move(archive / filename.name)

      # Clear all data structures
      self.chunks.clear()
      self.loaded.clear()
      with self.refs_lock:
        self.refs.clear()
      self.items.clear()
      self.fifo.clear()
      self.itemid = 0
      self.current.clear()
      self.streams.clear()
      
      # Clear sampler (calls its internal clear method if available)
      if hasattr(self.sampler, 'clear'):
        self.sampler.clear()
      
      # Clear online mode structures if applicable
      if self.online:
        self.lengths.clear()
        self.queue.clear()
      
      # Optionally reset metrics
      for key in self.metrics:
        self.metrics[key] = 0

  @elements.timer.section('replay_add')
  def add(self, step, worker=0):
    step = {k: v for k, v in step.items() if not k.startswith('log/')}
    with self.rwlock.reading:
      step = {k: np.asarray(v) for k, v in step.items()}

      if worker not in self.current:
        chunk = chunklib.Chunk(self.chunksize)
        with self.refs_lock:
          self.refs[chunk.uuid] = 1
        self.chunks[chunk.uuid] = chunk
        self.current[worker] = (chunk.uuid, 0)

      chunkid, index = self.current[worker]
      step['stepid'] = np.frombuffer(
          bytes(chunkid) + index.to_bytes(4, 'big'), np.uint8)
      stream = self.streams[worker]
      chunk = self.chunks[chunkid]
      assert chunk.length == index, (chunk.length, index)
      chunk.append(step)
      self._touch_chunk(chunk)
      assert chunk.length == index + 1, (chunk.length, index + 1)
      stream.append((chunkid, index))
      with self.refs_lock:
        self.refs[chunkid] += 1

      index += 1
      if index < chunk.size:
        self.current[worker] = (chunkid, index)
      else:
        self._complete(chunk, worker)
      assert len(self.streams) == len(self.current)

      if len(stream) >= self.length:
        # Increment is not thread safe thus inaccurate but faster than locking.
        self.metrics['inserts'] += 1
        chunkid, index = stream.popleft()
        self._insert(chunkid, index)

        if self.online and self.lengths[worker] % self.length == 0:
          self.queue.append((chunkid, index))

      if self.online:
        self.lengths[worker] += 1
      self._evict_chunks()

  @elements.timer.section('replay_sample')
  def sample(self, batch, mode='train'):
    message = f'Replay buffer {self.name} is empty'
    limiters.wait(lambda: len(self.sampler), message)
    seqs, is_online = zip(*[self._sample(mode) for _ in range(batch)])
    data = self._assemble_batch(seqs, 0, self.length)
    data = self._annotate_batch(data, is_online, True)
    self._evict_chunks()
    return data

  @elements.timer.section('replay_update')
  def update(self, data):
    stepid = data.pop('stepid')
    priority = data.pop('priority', None)
    assert stepid.ndim == 3, stepid.shape
    self.metrics['updates'] += int(np.prod(stepid.shape[:-1]))
    if priority is not None:
      assert priority.ndim == 2, priority.shape
      self.sampler.prioritize(
          stepid.reshape((-1, stepid.shape[-1])),
          priority.flatten())
    if data:
      for i, stepid in enumerate(stepid):
        stepid = stepid[0].tobytes()
        chunkid = elements.UUID(stepid[:-4])
        index = int.from_bytes(stepid[-4:], 'big')
        values = {k: v[i] for k, v in data.items()}
        try:
          self._setseq(chunkid, index, values)
        except KeyError:
          pass
    self._evict_chunks()

  def _sample(self, mode):
    assert mode in ('train', 'report', 'eval'), mode
    if mode == 'train':
      # Increment is not thread safe thus inaccurate but faster than locking.
      self.metrics['samples'] += 1
    while True:
      try:
        if self.online and self.queue and mode == 'train':
          chunkid, index = self.queue.popleft()
          is_online = True
        else:
          with elements.timer.section('sample'):
            itemid = self.sampler()
          chunkid, index = self.items[itemid]
          is_online = False
        seq = self._getseq(chunkid, index, concat=False)
        return seq, is_online
      except KeyError:
        continue

  def _insert(self, chunkid, index):
    while self.capacity and len(self.items) >= self.capacity:
      self._remove()
    itemid = self.itemid
    self.itemid += 1
    self.items[itemid] = (chunkid, index)
    stepids = (
        self._stepids_for_seq(chunkid, index)
        if self._sampler_needs_stepids(self.sampler) else None)
    self.sampler[itemid] = stepids
    self.fifo.append(itemid)

  def _remove(self):
    itemid = self.fifo.popleft()
    del self.sampler[itemid]
    chunkid, index = self.items.pop(itemid)
    with self.refs_lock:
      self.refs[chunkid] -= 1
      if self.refs[chunkid] < 1:
        del self.refs[chunkid]
        chunk = self.chunks.pop(chunkid)
        if chunk.succ in self.refs:
          self.refs[chunk.succ] -= 1

  def _getseq(self, chunkid, index, keys=None, concat=True):
    chunk = self._ensure_chunk_data(chunkid)
    available = chunk.length - index
    if available >= self.length:
      with elements.timer.section('get_slice'):
        seq = chunk.slice(index, self.length)
        if not concat:
          seq = {k: [v] for k, v in seq.items()}
        return seq
    else:
      with elements.timer.section('get_compose'):
        parts = [chunk.slice(index, available)]
        remaining = self.length - available
        while remaining > 0:
          chunk = self._ensure_chunk_data(chunk.succ)
          used = min(remaining, chunk.length)
          parts.append(chunk.slice(0, used))
          remaining -= used
        seq = {k: [p[k] for p in parts] for k in keys or parts[0].keys()}
        if concat:
          seq = {k: np.concatenate(v, 0) for k, v in seq.items()}
        return seq

  def _setseq(self, chunkid, index, values):
    length = len(next(iter(values.values())))
    chunk = self._ensure_chunk_data(chunkid)
    available = chunk.length - index
    if index < 0 or available < 0:
      raise KeyError(chunkid)
    if available >= length:
      with elements.timer.section('set_slice'):
        result = chunk.update(index, length, values)
        self._mark_dirty(chunk)
        return result
    else:
      with elements.timer.section('set_compose'):
        part = {k: v[:available] for k, v in values.items()}
        values = {k: v[available:] for k, v in values.items()}
        chunk.update(index, available, part)
        self._mark_dirty(chunk)
        remaining = length - available
        while remaining > 0:
          chunk = self._ensure_chunk_data(chunk.succ)
          used = min(remaining, chunk.length)
          if used <= 0:
            raise KeyError(chunk.uuid)
          part = {k: v[:used] for k, v in values.items()}
          values = {k: v[used:] for k, v in values.items()}
          chunk.update(0, used, part)
          self._mark_dirty(chunk)
          remaining -= used

  def _stepids_for_seq(self, chunkid, index):
    stepids = []
    remaining = self.length
    chunk = self.chunks[chunkid]
    while remaining > 0:
      available = chunk.length - index
      used = min(remaining, available)
      for offset in range(index, index + used):
        stepids.append(np.frombuffer(
            bytes(chunk.uuid) + offset.to_bytes(4, 'big'), np.uint8))
      remaining -= used
      if remaining > 0:
        chunk = self.chunks[chunk.succ]
        index = 0
    return np.stack(stepids, 0)

  def _sampler_needs_stepids(self, sampler):
    if isinstance(sampler, selectors.Prioritized):
      return True
    if isinstance(sampler, selectors.Mixture):
      return any(self._sampler_needs_stepids(x) for x in sampler.selectors)
    return False

  def _ensure_chunk_data(self, chunkid):
    chunk = self.chunks[chunkid]
    if chunk.data is None:
      if not self.directory:
        raise KeyError(chunkid)
      loaded = chunklib.Chunk.load(self.directory / chunk.filename, error='none')
      if loaded is None:
        raise KeyError(chunkid)
      chunk.data = loaded.data
      chunk.saved = True
    self._touch_chunk(chunk)
    return chunk

  def _touch_chunk(self, chunk):
    if chunk.data is None:
      return
    self.loaded.pop(chunk.uuid, None)
    self.loaded[chunk.uuid] = None

  def _mark_dirty(self, chunk):
    if not self.directory:
      return
    chunk.saved = False
    self.saved.discard(chunk.uuid)
    self._touch_chunk(chunk)

  def _save_chunk_now(self, chunk):
    if not self.directory or chunk.data is None:
      return
    if chunk.saved:
      self.saved.add(chunk.uuid)
      return
    chunk.save(self.directory)
    self.saved.add(chunk.uuid)

  def _evict_chunks(self):
    if not self.cache_chunks or not self.directory:
      return
    current = {chunkid for chunkid, _ in self.current.values()}
    while len(self.loaded) > self.cache_chunks:
      chunkid = next(iter(self.loaded))
      if chunkid in current:
        self.loaded.move_to_end(chunkid)
        if all(uuid in current for uuid in self.loaded):
          break
        continue
      chunk = self.chunks.get(chunkid)
      self.loaded.pop(chunkid, None)
      if chunk is None or chunk.data is None:
        continue
      self._save_chunk_now(chunk)
      if chunk.uuid in self.saved:
        chunk.data = None

  # def dataset(self, batch, length=None, consec=None, prefix=0, report=False):
  #   length = length or self.length
  #   consec = consec or (self.length - prefix) // length
  #   assert consec <= (self.length - prefix) // length, (
  #       self.length, length, consec, prefix)
  #   limiters.wait(lambda: len(self.sampler), 'Replay buffer is empty')
  #   # For performance, each batch should be consecutive in memory, rather than
  #   # a non-consecutive view into a longer batch. For example, this allows
  #   # near-instant serialization when sending over the network.
  #   while True:
  #     seqs, is_online = zip(*[self._sample(report) for _ in range(batch)])
  #     for i in range(consec):
  #       offset = i * length
  #       data = self._assemble_batch(seqs, offset, offset + length + prefix)
  #       data = self._annotate_batch(data, is_online, is_first=(i == 0))
  #       data['consec'] = np.full(data['is_first'].shape, i, np.int32)
  #       yield data

  @elements.timer.section('assemble_batch')
  def _assemble_batch(self, seqs, start, stop):
    shape = (len(seqs), stop - start)
    data = {
        key: np.empty((*shape, *parts[0].shape[1:]), parts[0].dtype)
        for key, parts in seqs[0].items()}
    for n, seq in enumerate(seqs):
      st, dt = 0, 0  # Source and destination time index.
      for p in range(len(seq['stepid'])):
        partlen = len(seq['stepid'][p])
        if start < st + partlen:
          part_start = max(0, start - st)
          part_stop = min(stop - st, partlen)
          num = part_stop - part_start
          for k in data.keys():
            data[k][n, dt: dt + num] = seq[k][p][part_start: part_stop]
          dt += num
        st += partlen
        if st >= stop:
          break
    return data

  @elements.timer.section('annotate_batch')
  def _annotate_batch(self, data, is_online, is_first):
    data = data.copy()
    # if self.online:
    #   broadcasted = [[x] for x in is_online]
    #   data['is_online'] = np.full(data['is_first'].shape, broadcasted, bool)
    if 'is_first' in data:
      if is_first:
        data['is_first'] = data['is_first'].copy()
        data['is_first'][:, 0] = True
      if 'is_last' in data:
        # Make sure that abandoned episodes have is_last set.
        next_is_first = np.roll(data['is_first'], shift=-1, axis=1)
        next_is_first[:, -1] = False
        data['is_last'] = data['is_last'] | next_is_first
    return data

  @elements.timer.section('replay_save')
  def save(self):
    if self.directory:
      with self.rwlock.writing:
        for worker, (chunkid, _) in self.current.items():
          chunk = self.chunks[chunkid]
          if chunk.length > 0:
            self._complete(chunk, worker)
        promises = []
        for chunk in self.chunks.values():
          if chunk.length <= 0 or chunk.data is None:
            continue
          if chunk.saved:
            self.saved.add(chunk.uuid)
            continue
          promises.append(self.workers.submit(chunk.save, self.directory))
        if self.save_wait or self.cache_chunks:
          [promise.result() for promise in promises]
          self.saved.update(chunk.uuid for chunk in self.chunks.values()
                            if chunk.saved)
        self._evict_chunks()
    return None

  @elements.timer.section('replay_load')
  def load(self, data=None, directory=None, amount=None):

    directory = directory or self.directory
    amount = amount or self.capacity or np.inf
    if not directory:
      return
    revsorted = lambda x: list(reversed(sorted(list(x))))
    directory = elements.Path(directory)
    names_loaded = revsorted(x.filename for x in list(self.chunks.values()))
    names_ondisk = revsorted(x.name for x in directory.glob('*.npz'))
    names_ondisk = [x for x in names_ondisk if x not in names_loaded]
    if not names_ondisk:
      return

    numitems = self._numitems(names_loaded + names_ondisk)
    uuids = [elements.UUID(x.split('-')[1]) for x in names_ondisk]
    total = 0
    numchunks = 0
    for uuid in uuids:
      numchunks += 1
      total += numitems[uuid]
      if total >= amount:
        break

    filenames = [directory / x for x in names_ondisk[:numchunks]]
    if self.cache_chunks:
      load = bind(chunklib.Chunk.metadata, validate=True, error='none')
      with ThreadPoolExecutor(16, 'replay_metadata_loader') as pool:
        chunks = [x for x in pool.map(load, filenames) if x]
    else:
      load = bind(chunklib.Chunk.load, error='none')
      with ThreadPoolExecutor(16, 'replay_loader') as pool:
        chunks = [x for x in pool.map(load, filenames) if x]

    # We need to recompute the number of items per chunk now because some
    # chunks may be corrupted and thus not available.
    # numitems = self._numitems(chunks + list(self.chunks.values()))
    numitems = self._numitems(chunks)

    with self.rwlock.writing:
      self.saved.update([chunk.uuid for chunk in chunks])
      with self.refs_lock:
        for chunk in chunks:
          self.chunks[chunk.uuid] = chunk
          self.refs[chunk.uuid] = 0
          self._touch_chunk(chunk)
        for chunk in reversed(chunks):
          amount = numitems[chunk.uuid]
          self.refs[chunk.uuid] += amount
          if chunk.succ in self.refs:
            self.refs[chunk.succ] += 1
          for index in range(amount):
            self._insert(chunk.uuid, index)

  @elements.timer.section('complete_chunk')
  def _complete(self, chunk, worker):
    succ = chunklib.Chunk(self.chunksize)
    with self.refs_lock:
      self.refs[chunk.uuid] -= 1
      self.refs[succ.uuid] = 2
    self.chunks[succ.uuid] = succ
    self.current[worker] = (succ.uuid, 0)
    chunk.succ = succ.uuid
    self._touch_chunk(succ)
    return succ

  def _numitems(self, chunks):
    chunks = [x.filename if hasattr(x, 'filename') else x for x in chunks]
    if not chunks:
      return 0
    chunks = list(reversed(sorted([elements.Path(x).stem for x in chunks])))
    times, uuids, succs, lengths = zip(*[x.split('-') for x in chunks])
    uuids = [elements.UUID(x) for x in uuids]
    succs = [elements.UUID(x) for x in succs]
    lengths = {k: int(v) for k, v in zip(uuids, lengths)}
    future = {}
    for uuid, succ in zip(uuids, succs):
      future[uuid] = lengths[uuid] + future.get(succ, 0)
    numitems = {}
    for uuid, succ in zip(uuids, succs):
      numitems[uuid] = lengths[uuid] + 1 - self.length + future.get(succ, 0)
    numitems = {k: np.clip(v, 0, lengths[k]) for k, v in numitems.items()}
    return numitems

  def _notempty(self, reason=False):
    if reason:
      return (True, 'ok') if len(self.sampler) else (False, 'empty buffer')
    else:
      return bool(len(self.sampler))
