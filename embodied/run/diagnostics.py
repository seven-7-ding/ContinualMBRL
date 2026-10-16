"""Compatibility adapters for optional elements 3.22 resource statistics."""

import collections
import gc
import sys
import time
import types

import elements
from elements.usage import GcStats, MallocStats


class GCStats(GcStats):

  def _summary(self):
    # isinstance()/inspect.isframe() dereference weak proxies and can raise
    # ReferenceError during a live JAX process. Inspect types without deref.
    lines = ['GC Most Common Types']
    for generation in range(3):
      counts = collections.Counter(type(obj).__name__
          for obj in gc.get_objects(generation) if type(obj) is not types.FrameType)
      previous = self.counts[generation]
      deltas = {name: count - previous.get(name, 0) for name, count in counts.items()}
      self.counts[generation] = counts
      lines.append(f'Generation {generation}')
      for name in sorted(deltas, key=lambda key: -abs(deltas[key]))[:10]:
        lines.append(f'{name}: {deltas[name]:+d} ({counts[name]})')
    return '\n'.join(lines)

  def _callback(self, phase, info):
    if sys.is_finalizing():
      return
    now = time.perf_counter_ns()
    if phase == 'start':
      self.start = now
    elif phase == 'stop' and self.start:
      gen = info['generation']
      self.stats.add(f'gen{gen}/calls', 1, agg='sum')
      values = dict(collected=info['collected'],
                    uncollectable=info['uncollectable'],
                    duration=(now - self.start) / 1e9)
      for key, value in values.items():
        for reducer in ('avg', 'max', 'sum'):
          self.stats.add(f'gen{gen}/{key}/{reducer}', value, agg=reducer)
      self.start = None


def make_usage(options):
  options = dict(options)
  malloc = options.pop('malloc', False)
  gc = options.pop('gc', False)
  usage = elements.Usage(**options)
  if gc:
    usage.tools['gc'] = GCStats()
  if malloc:
    usage.tools['malloc'] = MallocStats(root_module='embodied')
  return usage
