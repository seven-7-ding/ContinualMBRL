import json
import os
import sys
from pathlib import Path

from dm_control import suite
from dm_control.locomotion.examples import basic_rodent_2020

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from embodied.envs.general_dmc_priori import GeneralDMCPriori, TASK_OBS_DIMS


PROBE_OBS_DIM = 512
PROBE_ACT_DIM = 128
OUTPUT_PATH = Path(__file__).with_name('dmc_priori_dims.json')


def suite_task_to_env_task(domain, task):
  if domain == 'ball_in_cup':
    return f'cup_{task}'
  return f'{domain}_{task}'


def candidate_tasks():
  tasks = {suite_task_to_env_task(domain, task) for domain, task in suite.ALL_TASKS}
  tasks.update(
      name for name in dir(basic_rodent_2020)
      if name.startswith('rodent_') and callable(getattr(basic_rodent_2020, name)))
  tasks.update(TASK_OBS_DIMS.keys())
  tasks.add('dog_walk')
  return sorted(tasks)


def probe_task(task):
  try:
    env = GeneralDMCPriori(
        task, repeat=1, task_action_space=[PROBE_ACT_DIM], obs_dim=PROBE_OBS_DIM)
    return {
        'status': 'ok',
        'task': task,
        'real_obs_dim': int(env._real_obs_dim),
        'real_act_dim': int(env._real_act_space['action'].shape[0]),
        'padded_obs_dim': int(env._obs_dim),
        'general_act_dim': int(env._general_act_size[0]),
        'known_obs_dim_hint': TASK_OBS_DIMS.get(task),
    }
  except Exception as exc:
    return {
        'status': 'error',
        'task': task,
        'error_type': type(exc).__name__,
        'error': str(exc),
        'known_obs_dim_hint': TASK_OBS_DIMS.get(task),
    }


def main():
  os.environ.setdefault('MUJOCO_GL', 'egl')
  results = [probe_task(task) for task in candidate_tasks()]
  payload = {
      'probe_obs_dim': PROBE_OBS_DIM,
      'probe_act_dim': PROBE_ACT_DIM,
      'num_tasks': len(results),
      'tasks': results,
  }
  OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=False) + '\n')
  ok = sum(item['status'] == 'ok' for item in results)
  print(f'Wrote {OUTPUT_PATH} with {ok}/{len(results)} successful probes.')


if __name__ == '__main__':
  main()
