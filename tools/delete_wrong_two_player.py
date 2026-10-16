"""Delete only the superseded, manifest-owned v1 runs requested by the user."""
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.inspect_wandb_reference import credentials


def main():
  import wandb
  control = ROOT / 'logdir/two_player_sweep_20261015'
  state = json.loads((control / 'wrong_manifest.json').read_text())
  env = credentials()
  os.environ['WANDB_API_KEY'] = env['WANDB_API_KEY']
  project = state['project']
  assert project.startswith('two_player_dreamer_')
  allowed = {j['wandb_id'] for j in state['jobs']}
  assert all(run_id.startswith('two-player-') and run_id.endswith('-v1') for run_id in allowed)
  api = wandb.Api(timeout=30)
  deleted = []
  for run in api.runs(f"{env['WANDB_ENTITY']}/{project}"):
    if run.id in allowed:
      run.delete(delete_artifacts=True)
      deleted.append(run.id)
      print('Deleted superseded W&B run', run.id, flush=True)
  bootstrap = control / 'cache/jax_bootstrap'
  bootstrap.mkdir(parents=True, exist_ok=True)
  for job in state['jobs']:
    directory = Path(job['logdir'])
    assert directory.is_relative_to(ROOT / 'logdir' / project)
    for source in (directory / 'cache/jax').glob('*-cache'):
      target = bootstrap / source.name
      if not target.exists():
        shutil.copy2(source, target)
  # The legacy cache alias belonged only to the superseded pilot.
  alias = ROOT / 'logdir/two_player_jax_cache'
  if alias.is_symlink():
    alias.unlink()
  for job in state['jobs']:
    directory = Path(job['logdir'])
    if directory.exists():
      shutil.rmtree(directory)
  (control / 'deletion_audit.json').write_text(json.dumps({
      'project': project, 'deleted_ids': deleted, 'expected_ids': sorted(allowed),
      'local_run_directories_deleted': True}, indent=2) + '\n')
  print('Removed wrong local runs; retained only reusable compiler entries and deletion audit.', flush=True)


if __name__ == '__main__':
  main()
