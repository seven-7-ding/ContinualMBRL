"""Read reference metric names without emitting credentials or API payloads."""

import json
import os
from pathlib import Path
import urllib.request
import base64

ROOT = Path(__file__).resolve().parents[1]
PROJECT = 'continual_dreamer_soft_reset_walker_run|finger_spin|fish_swim_size1m_200k_x10'


def credentials():
  env = os.environ.copy()
  for filename in ('.env', '.env.local', '.env.wandb.local'):
    path = ROOT / filename
    if path.exists():
      for line in path.read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
          key, value = line.split('=', 1)
          env[key.strip()] = value.strip().strip('"').strip("'")
  key = env.get('WANDB_API_KEY') or env.get('wandb_api')
  if not key:
    raise RuntimeError('Missing W&B credential')
  env['WANDB_API_KEY'] = key
  env.setdefault('WANDB_ENTITY', 'stevendiffdiff-')
  return env


def main():
  env = credentials()
  query = '''query($entity: String!, $project: String!) {
    project(name: $project, entityName: $entity) {
      runs(first: 3, order: "-createdAt") {
        edges { node { name config summaryMetrics } }
      }
    }
  }'''
  data = json.dumps({'query': query, 'variables': {
      'entity': env['WANDB_ENTITY'], 'project': PROJECT}}).encode()
  host = env.get('WANDB_BASE_URL', 'https://api.wandb.ai')
  auth = base64.b64encode(('api:' + env['WANDB_API_KEY']).encode()).decode()
  req = urllib.request.Request(host.rstrip('/') + '/graphql', data=data,
      headers={'Authorization': 'Basic ' + auth, 'Content-Type': 'application/json'})
  try:
    with urllib.request.urlopen(req, timeout=30) as response:
      result = json.load(response)
    project = result.get('data', {}).get('project')
    if not project:
      print('Reference project unavailable; GraphQL error count:', len(result.get('errors', [])))
      return
    output = []
    for edge in project['runs']['edges']:
      node = edge['node']
      summary = json.loads(node['summaryMetrics'])
      output.append({'id': node['name'], 'metric_keys': sorted(summary),
                     'config': json.loads(node['config'])})
    dest = ROOT / 'logdir/two_player_sweep_20261015/reference.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(output, indent=2))
    print('Reference runs:', len(output), 'metric counts:', [len(x['metric_keys']) for x in output])
    print('Saved reference configuration and metric names:', dest)
  except Exception as exc:
    # Never include HTTP response bodies, headers, or credential-bearing URLs.
    print('Reference request failed:', type(exc).__name__, getattr(exc, 'code', ''))


if __name__ == '__main__':
  main()
