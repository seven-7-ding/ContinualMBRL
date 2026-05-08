#!/usr/bin/env python
"""Continual DreamerEnvLearner training: Dreamer-style actor-critic with real env as world model.

Key differences from train_continual_dreamer_dist.py:
  - Uses DreamerEnvLearner (V(s) not Q(s,a), advantage-weighted PG).
  - Replay buffer stores physics states for imagination rollouts.
  - Imagination uses a pool of real envs: their state is reset to replay-sampled
    physics states, then H env.step() calls produce the trajectory.
  - No temperature parameter; actent controls entropy regularization.

Example
-------
    python examples/train_continual_dreamer_env.py \\
        --tasks finger_spin,walker_walk,cheetah_run,reacher_easy \\
        --obs_dim 32 --act_dim 6 \\
        --task_steps 200000 \\
        --num_envs 16 \\
        --num_imag_envs 64 \\
        --save_dir ./logdir/continual_dreamer_env/test/seed_42
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import tqdm
import wandb
import numpy as np
import gym
import gym.vector
from absl import app, flags
from ml_collections import config_flags

from jaxrl2.agents import DreamerEnvLearner
from jaxrl2.data import ReplayBuffer
from jaxrl2.evaluation import evaluate
from jaxrl2.wrappers import wrap_gym
from jaxrl2.envs import ContinualDMCEnv

FLAGS = flags.FLAGS

flags.DEFINE_string("tasks",
                    "finger_spin,walker_walk,cheetah_run,reacher_easy",
                    "Comma-separated list of 'domain_task' strings.")
flags.DEFINE_integer("obs_dim",   32,  "Fixed (padded) observation dimension.")
flags.DEFINE_integer("act_dim",    6,  "Fixed (padded) action dimension.")
flags.DEFINE_integer("task_steps", 200_000,
                     "Environment steps (individual transitions) per task.")
flags.DEFINE_integer("task_repeats", 10,
                     "How many times to cycle through the full task list.")
flags.DEFINE_integer("start_training", 10_000,
                     "Transitions before training begins.")
flags.DEFINE_integer("batch_size",  1024, "Mini-batch size for replay sampling.")
flags.DEFINE_integer("utd",           1,  "Updates per environment step.")
flags.DEFINE_integer("num_envs",     16,  "Parallel env workers for data collection.")
flags.DEFINE_integer("num_imag_envs", 64, "Number of envs for imagination rollouts.")
flags.DEFINE_integer("eval_episodes", 5,  "Episodes per eval.")
flags.DEFINE_integer("eval_interval", 20_000, "Steps between evals.")
flags.DEFINE_integer("log_interval",  10_000, "Steps between log flushes.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("save_dir",
                    "./logdir/continual_dreamer_env/default/seed_none",
                    "Directory for logs.")
flags.DEFINE_string("vd_mode", "disabled",
                    "reset_all to reset agent on task switch; otherwise ignored.")
flags.DEFINE_boolean("tqdm",  False,  "Show tqdm progress bar.")
flags.DEFINE_boolean("wandb", True,   "Log to wandb.")
config_flags.DEFINE_config_file(
    "config",
    "configs/continual_dreamer_env.py",
    "Training hyperparameter config.",
    lock_config=False,
)


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------

def _make_single_env(task_list, obs_dim, act_dim, seed):
    env = ContinualDMCEnv(task_list, obs_dim=obs_dim, act_dim=act_dim, seed=seed)
    env = wrap_gym(env, rescale_actions=False)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    return env


def make_vec_env(task_list, obs_dim, act_dim, seed, num_envs):
    def _env_fn(worker_seed):
        def _make():
            return _make_single_env(task_list, obs_dim, act_dim, worker_seed)
        return _make
    env_fns = [_env_fn(seed + i) for i in range(num_envs)]
    return gym.vector.SyncVectorEnv(env_fns)


def make_eval_env(task_name, obs_dim, act_dim, seed):
    env = ContinualDMCEnv([task_name], obs_dim=obs_dim, act_dim=act_dim, seed=seed)
    env = wrap_gym(env, rescale_actions=False)
    return env


def make_imag_envs(task_name, obs_dim, act_dim, seed, num_imag_envs):
    """Create a pool of bare ContinualDMCEnv instances for imagination rollouts.

    These envs are not wrapped (no gym.wrappers) so that set_physics_state /
    step() work directly without wrapper interference.
    """
    return [
        ContinualDMCEnv([task_name], obs_dim=obs_dim, act_dim=act_dim, seed=seed + i)
        for i in range(num_imag_envs)
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(_):
    kwargs = dict(FLAGS.config)
    kwargs.pop("jax_mem_fraction", None)
    redo_cfg = dict(kwargs.pop("redo", {}))
    redo_cfg["frequency"] = redo_cfg["frequency"] * FLAGS.utd
    opt_cfg = dict(kwargs.pop("opt", {}))
    kwargs.setdefault("model_size", None)

    num_envs  = FLAGS.num_envs
    task_list = [t.strip() for t in FLAGS.tasks.split(',')]
    task_schedule = task_list * FLAGS.task_repeats
    total_steps = FLAGS.task_steps * len(task_schedule)

    os.makedirs(FLAGS.save_dir, exist_ok=True)
    project, group, run = FLAGS.save_dir.split('/')[-3:]
    if FLAGS.wandb:
        wandb.init(project=project, group=group, name=run, dir=FLAGS.save_dir)
        wandb.config.update(FLAGS)

    vec_env = make_vec_env(task_list, FLAGS.obs_dim, FLAGS.act_dim,
                           FLAGS.seed, num_envs)
    eval_env = make_eval_env(task_schedule[0], FLAGS.obs_dim, FLAGS.act_dim,
                              FLAGS.seed + 42)
    imag_envs = make_imag_envs(task_schedule[0], FLAGS.obs_dim, FLAGS.act_dim,
                                FLAGS.seed + 100, FLAGS.num_imag_envs)

    single_obs_space = vec_env.single_observation_space
    single_act_space = vec_env.single_action_space

    # Probe physics state dimension for the first task.
    _probe = ContinualDMCEnv([task_schedule[0]], obs_dim=FLAGS.obs_dim,
                              act_dim=FLAGS.act_dim, seed=FLAGS.seed)
    _probe.reset()
    physics_dim = _probe.get_physics_state_dim()
    _probe.close()
    print(f'[train] physics_state_dim={physics_dim}')

    agent = DreamerEnvLearner(
        FLAGS.seed,
        single_obs_space,
        single_act_space,
        redo=redo_cfg,
        vd_mode=FLAGS.vd_mode,
        opt=opt_cfg,
        **kwargs,
    )

    replay_buffer = ReplayBuffer(
        single_obs_space, single_act_space, 5_000_000,
        physics_state_dim=physics_dim)
    replay_buffer.seed(FLAGS.seed)

    task_idx  = 0
    for w in vec_env.envs:
        w.unwrapped._load_task(task_schedule[0])

    obs = vec_env.reset()

    global_step = 0
    task_local  = 0

    # Physics states are tracked per-env worker: we store the physics state
    # BEFORE each env.step() call so that the replay buffer entry for
    # (obs, act, ...) has the matching state to restore the env for imagination.
    cur_physics = np.zeros((num_envs, physics_dim), dtype=np.float64)
    for i, w in enumerate(vec_env.envs):
        cur_physics[i] = w.unwrapped.get_physics_state()

    pbar = tqdm.tqdm(total=total_steps, smoothing=0.1, disable=not FLAGS.tqdm)

    while global_step < total_steps:
        # ---- task switch ----
        if task_local >= FLAGS.task_steps:
            task_idx += 1
            if task_idx >= len(task_schedule):
                break
            cur_task   = task_schedule[task_idx]
            repeat_idx = task_idx // len(task_list)
            print(f'\n[Step {global_step}] Cycle {repeat_idx + 1}/{FLAGS.task_repeats} '
                  f'-> task: {cur_task}')
            for w in vec_env.envs:
                w.unwrapped._load_task(cur_task)
                w.unwrapped.task_step = 0
            obs = vec_env.reset()
            for i, w in enumerate(vec_env.envs):
                cur_physics[i] = w.unwrapped.get_physics_state()

            # Re-probe physics dim in case the new task has different state size.
            _probe = ContinualDMCEnv([cur_task], obs_dim=FLAGS.obs_dim,
                                     act_dim=FLAGS.act_dim, seed=FLAGS.seed)
            _probe.reset()
            new_phys_dim = _probe.get_physics_state_dim()
            _probe.close()
            if new_phys_dim != physics_dim:
                physics_dim = new_phys_dim
                cur_physics = np.zeros((num_envs, physics_dim), dtype=np.float64)
                for i, w in enumerate(vec_env.envs):
                    cur_physics[i] = w.unwrapped.get_physics_state()
                print(f'[train] physics_state_dim updated to {physics_dim}')

            replay_buffer = ReplayBuffer(
                single_obs_space, single_act_space, FLAGS.task_steps,
                physics_state_dim=physics_dim)
            replay_buffer.seed(FLAGS.seed + task_idx)

            eval_env = make_eval_env(cur_task, FLAGS.obs_dim, FLAGS.act_dim, FLAGS.seed + 42)

            # Switch imagination envs to the new task.
            for env_i in imag_envs:
                env_i._load_task(cur_task)

            task_local = 0
            if 'reset_all' in FLAGS.vd_mode:
                agent.reset_agent()
                print('  -> agent fully reset (vd_mode=reset_all)')

        # ---- collect one vectorised step ----
        if task_local < FLAGS.start_training:
            actions = np.array([single_act_space.sample() for _ in range(num_envs)])
        else:
            actions = agent.sample_actions(obs)

        next_obs, rewards, dones, infos = vec_env.step(actions)

        if isinstance(infos, list):
            per_env_infos = infos
        else:
            per_env_infos = [{} for _ in range(num_envs)]
            for _key, _values in infos.items():
                if _key.startswith('_'):
                    continue
                _mask = infos.get(f'_{_key}', None)
                for _j in range(num_envs):
                    if _mask is None or _mask[_j]:
                        per_env_infos[_j][_key] = _values[_j]

        for i in range(num_envs):
            truncated = per_env_infos[i].get('TimeLimit.truncated', False)
            mask = 0.0 if (dones[i] and not truncated) else 1.0
            if dones[i]:
                terminal_obs = per_env_infos[i].get('terminal_observation', next_obs[i])
            else:
                terminal_obs = next_obs[i]

            replay_buffer.insert(dict(
                observations=obs[i],
                actions=actions[i],
                rewards=float(rewards[i]),
                masks=mask,
                dones=bool(dones[i]),
                next_observations=terminal_obs,
                physics_states=cur_physics[i].astype(np.float64),
            ))

            # Update physics snapshot for next step.
            if dones[i]:
                # After auto-reset the env is already in the new episode state.
                cur_physics[i] = vec_env.envs[i].unwrapped.get_physics_state()
            else:
                cur_physics[i] = vec_env.envs[i].unwrapped.get_physics_state()

        obs = next_obs
        global_step += num_envs
        task_local  += num_envs
        pbar.update(num_envs)

        # ---- training ----
        if task_local >= FLAGS.start_training:
            update_info = {}
            for _ in range(FLAGS.utd * num_envs):
                if len(replay_buffer) >= FLAGS.batch_size:
                    batch     = replay_buffer.sample(FLAGS.batch_size)
                    step_info = agent.update(batch, imag_envs)
                    update_info.update(step_info)

            if FLAGS.wandb and (global_step % FLAGS.log_interval < num_envs):
                log_dict = {}
                for k, v in update_info.items():
                    if k.startswith('loss/') or k.startswith('train/'):
                        log_dict[k] = v
                    elif 'redo' in k:
                        log_dict[f'act_redo/{k}'] = v
                log_dict['train/global_step'] = global_step
                wandb.log(log_dict, step=global_step)

        # ---- evaluation ----
        if global_step % FLAGS.eval_interval < num_envs:
            eval_info = evaluate(agent, eval_env, FLAGS.eval_episodes)
            if FLAGS.wandb:
                wandb.log({'eval/return': eval_info['return']}, step=global_step)
            print(f'[Step {global_step}] eval return: {eval_info["return"]:.2f}')

    pbar.close()
    print('Training complete.')


if __name__ == '__main__':
    app.run(main)
