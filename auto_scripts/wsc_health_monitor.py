#!/usr/bin/env python3

import json
import math
import re
import time
from pathlib import Path


ROOTS = [
    Path('logdir/continual_dreamer_soft_reset_size1m'),
    Path('logdir/continual_dreamer_soft_reset_dog_stand|dog_walk|dog_trot_size1m'),
]
ERROR_RE = re.compile(
    r'Traceback|Exception|RuntimeError|ValueError|FileNotFoundError|'
    r'NaN|nan detected')


def latest_metrics(path):
  last = None
  try:
    for line in path.read_text(errors='ignore').splitlines():
      if not line.strip():
        continue
      try:
        item = json.loads(line)
      except Exception:
        continue
      if 'fps/policy' in item or 'fps/train' in item:
        last = item
  except Exception as exc:
    return None, f'metrics_read_error {exc}'
  return last, None


def scan_once():
  rows = []
  nofps = []
  bad = []
  errs = []
  for root in ROOTS:
    if not root.exists():
      continue
    for train in sorted(root.glob('**/train.log')):
      text_path = str(train)
      if '.failed.' in text_path or 'reset_archive' in text_path:
        continue
      metrics = train.with_name('metrics.jsonl')
      if not metrics.exists():
        nofps.append(str(train.parent))
      else:
        last, err = latest_metrics(metrics)
        if err:
          errs.append((str(metrics), err))
        if last:
          policy = float(last.get('fps/policy', math.nan))
          train_fps = float(last.get('fps/train', math.nan))
          step = int(last.get('step') or 0)
          row = (policy, train_fps, step, str(train.parent))
          rows.append(row)
          if (
              (math.isfinite(policy) and policy < 6.0) or
              (math.isfinite(train_fps) and train_fps < 6.0)):
            bad.append(row)
        else:
          nofps.append(str(train.parent))
      try:
        tail = train.read_text(errors='ignore')[-12000:]
      except Exception as exc:
        errs.append((str(train), f'train_read_error {exc}'))
      else:
        if ERROR_RE.search(tail):
          errs.append((str(train), 'error_pattern_in_tail'))
  return rows, nofps, bad, errs


def print_scan():
  rows, nofps, bad, errs = scan_once()
  print(f'--- health {time.strftime("%F %T")} ---', flush=True)
  print(
      'runs_with_fps', len(rows),
      'no_fps', len(nofps),
      'bad', len(bad),
      'errs', len(errs),
      flush=True)
  def sort_key(row):
    return math.inf if not math.isfinite(row[0]) else row[0]
  for policy, train_fps, step, run in sorted(rows, key=sort_key)[:30]:
    print(f'{step}\tpolicy={policy:.2f}\ttrain={train_fps:.2f}\t{run}', flush=True)
  if nofps:
    print('NO_FPS', flush=True)
    for item in nofps[:80]:
      print(item, flush=True)
  if bad:
    print('BAD', flush=True)
    for item in bad[:80]:
      print(item, flush=True)
  if errs:
    print('ERRS', flush=True)
    for item in errs[:80]:
      print(item, flush=True)


def main():
  while True:
    print_scan()
    time.sleep(300)


if __name__ == '__main__':
  main()
