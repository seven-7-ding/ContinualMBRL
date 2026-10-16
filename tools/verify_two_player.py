"""Audit local logs, live PIDs, checkpoints, resources and W&B upload."""
import json
import argparse
import math
import os
from pathlib import Path
import sys
import time
import pickle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from auto_scripts.launch_two_player import STATE, PROJECT, ROOT as REPO_ROOT, TRAIN_RATIO, alive, valid_rows, gpu_stats, available_ram_gib, log_interval
from embodied.run.two_player import update_counts
from tools.inspect_wandb_reference import credentials


def main():
  import psutil
  import wandb
  env = credentials()
  os.environ['WANDB_API_KEY'] = env['WANDB_API_KEY']
  api = wandb.Api(timeout=30)
  state = json.loads(STATE.read_text())
  parser = argparse.ArgumentParser()
  parser.add_argument('--only')
  args = parser.parse_args()
  report = {'time': time.time(), 'project': PROJECT, 'runs': []}
  keys = ['env_step', 'loss/policy', 'loss/value', 'loss/reward', 'fps/train']
  for job in state['jobs']:
    if args.only and args.only != f"{job['setting']}:{job['seed']}":
      continue
    rows = valid_rows(job)
    item = {k: job[k] for k in ('setting', 'seed', 'gpu', 'pid', 'wandb_id')}
    item.update(alive=alive(job), local_valid_logs=len(rows),
                last_step=rows[-1]['step'] if rows else None,
                log_interval=log_interval(job))
    wm, actor = update_counts(job['setting'])
    counts = [('opt/updates', 1)] if job['setting'] == 'default' else [
        ('opt/wm/updates', wm), ('opt/agent/updates', actor)]
    item['update_frequency_verified'] = len(rows) >= 2 and all(
        key in before and key in after and
        after[key] - before[key] == (after['step'] - before['step']) * TRAIN_RATIO / 1024 * multiplier
        for before, after in zip(rows, rows[1:]) for key, multiplier in counts)
    if item['alive']:
      item['ram_gib'] = psutil.Process(job['pid']).memory_info().rss / 2**30
    ckpt = Path(job['logdir']) / 'ckpt/latest'
    item['checkpoint'] = ckpt.read_text().strip() if ckpt.exists() else None
    if item['checkpoint']:
      snapshot = ckpt.parent / item['checkpoint']
      item['checkpoint_complete'] = (snapshot / 'done').exists()
      item['checkpoint_step'] = int(pickle.loads((snapshot / 'step.pkl').read_bytes()))
      if item['checkpoint_step'] > 0:
        try:
          # Trusted checkpoint produced by this run. Inspect actual saved
          # optimizer clocks and last losses even before its first 10k log.
          saved = pickle.loads((snapshot / 'agent.pkl').read_bytes())
          metrics = {}
          for group in saved.get('train_metrics', {}).values():
            metrics.update(zip(group['keys'], group['values']))
          losses = ('policy', 'value', 'repval', 'reward', 'con', 'dyn', 'rep', 'state')
          item['checkpoint_loss_finite'] = all(
              f'loss/{k}' in metrics and math.isfinite(float(metrics[f'loss/{k}']))
              for k in losses)
          item['checkpoint_updates'] = {k: int(v) for k, v in saved['params'].items()
              if k in ('opt/step/value', 'opt_wm/step/value', 'opt_agent/step/value')}
          del saved
        except (FileNotFoundError, KeyError, ValueError) as exc:
          item['checkpoint_inspection_error'] = type(exc).__name__
    try:
      run = api.run(f"{env['WANDB_ENTITY']}/{PROJECT}/{job['wandb_id']}")
      remote = 0
      # Environment steps are sparse W&B indices. Sample actual rows instead
      # of scanning thousands of empty step-index ranges.
      remote_steps = set()
      for row in run.history(keys=keys, samples=1000, pandas=False):
        if (all(isinstance(row.get(k), (int, float)) and math.isfinite(row[k]) for k in keys)
            and row['env_step'] > 0 and row['env_step'] % log_interval(job) == 0
            and row['fps/train'] > 0):
          remote_steps.add(row['env_step'])
          remote = len(remote_steps)
        if remote >= 5:
          break
      item['remote_valid_logs'] = remote
      item['remote_metric_count'] = len(run.summary.keys())
      item['wandb_url'] = run.url
      item['wandb_state'] = run.state
    except Exception as exc:
      item['remote_error_type'] = type(exc).__name__
    print(item['setting'], item['seed'], 'alive', item['alive'],
          'local', len(rows), 'remote', item.get('remote_valid_logs', 0),
          'step', item['last_step'], 'checkpoint', item.get('checkpoint_step'),
          'saved_losses_finite', item.get('checkpoint_loss_finite'), flush=True)
    report['runs'].append(item)
  report['gpus'] = gpu_stats()
  report['ram_available_gib'] = available_ram_gib()
  originals = [800243,800291,800317,800268,800339,828953,800246,800294,800318,
      800342,800270,800249,800321,800297,800345,800273,800252,800324,800303,
      800277,800348,800255,800327,800279,800302,800351,800282,800306,800260,
      800354,800335,800284,800356,800309,800261,800264,800288,800312,800358,800336]
  report['original_pids_alive'] = [p for p in originals if psutil.pid_exists(p)]
  report['original_pids_missing'] = [p for p in originals if not psutil.pid_exists(p)]
  protected = REPO_ROOT / 'logdir/two_player_sweep_20261015/protected_processes_current.json'
  if protected.exists():
    snapshot = json.loads(protected.read_text())
    report['current_protected_processes_alive'] = [p['pid'] for p in snapshot['current_unrelated_gpu_processes']
        if psutil.pid_exists(p['pid']) and abs(psutil.Process(p['pid']).create_time() - p['create_time']) < 1]
  report['all_verified'] = all(j['alive'] and j['local_valid_logs'] >= 5 and
      j.get('remote_valid_logs', 0) >= 5 and j.get('checkpoint_complete') and
      j.get('checkpoint_step', 0) > 0 and j.get('checkpoint_loss_finite') and
      j['update_frequency_verified'] for j in report['runs'])
  path = STATE.with_name('two_player_verification.json' if not args.only else 'pilot_verification.json')
  path.write_text(json.dumps(report, indent=2) + '\n')
  print('all_verified', report['all_verified'], 'report', path, flush=True)
  return 0 if report['all_verified'] else 1


if __name__ == '__main__':
  sys.exit(main())
