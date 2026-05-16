"""State-based (proprio) continual DMC environment for DreamerV3.

Wraps a dm_control task to expose:
  - A fixed-size flat ``state`` observation (zero-padded to ``obs_dim``).
  - A generalised action space (zero-padded to ``task_action_space``).

This mirrors the padded-obs design used in the VDRL ContinualDMCEnv so that
the network architecture stays fixed across tasks with different intrinsic
observation / action dimensionalities.
"""

import functools
import os

import elements
import embodied
import numpy as np
from dm_control import manipulation
from dm_control import suite
from dm_control.locomotion.examples import basic_rodent_2020

from . import from_dm


# Known proprio obs dims for common dm_control tasks (flattened).
# Used only for a fast pre-check; the code will auto-detect at runtime too.
TASK_OBS_DIMS = {
    'acrobot_swingup':          6,
    'acrobot_swingup_sparse':   6,
    'ball_in_cup_catch':        8,
    'cartpole_balance':         5,
    'cartpole_balance_sparse':  5,
    'cartpole_swingup':         5,
    'cartpole_swingup_sparse':  5,
    'cartpole_two_poles':       8,
    'cheetah_run':             17,
    'finger_spin':              9,
    'finger_turn_easy':         9,
    'finger_turn_hard':         9,
    'fish_swim':               24,
    'fish_upright':            24,
    'hopper_hop':              15,
    'hopper_stand':            15,
    'humanoid_run':            67,
    'humanoid_stand':          67,
    'humanoid_walk':           67,
    'pendulum_swingup':         3,
    'point_mass_easy':          4,
    'point_mass_hard':          4,
    'quadruped_escape':        78,
    'quadruped_fetch':         78,
    'quadruped_run':           78,
    'quadruped_walk':          78,
    'reacher_easy':             6,
    'reacher_hard':             6,
    'swimmer_swimmer6':        24,
    'swimmer_swimmer15':       24,
    'walker_run':              24,
    'walker_stand':            24,
    'walker_walk':             24,
}


class GeneralDMCPriori(embodied.Env):
    """Continual dm_control env using fixed-size padded state observations.

    Parameters
    ----------
    env : str
        Task string of the form ``"domain_task"``, e.g. ``"cheetah_run"``.
    repeat : int
        Number of physics steps per agent step (action repeat).
    task_action_space : list[int]
        Shape of the generalised action space, e.g. ``[6]``.  Actions
        from the network are sliced to the true task dimensionality before
        being passed to the env.
    obs_dim : int
        Size of the fixed flat state observation.  The true proprio vector
        is zero-padded (or truncated with a warning) to this size.
    camera : int
        Camera index used for rendering (only for ``log/image``).
    size : tuple[int, int]
        Image render size (used only for the auxiliary ``log/image`` key).
    """

    DEFAULT_CAMERAS = dict(
        quadruped=2,
        rodent=4,
    )

    def __init__(
        self,
        env,
        repeat=1,
        task_action_space=None,
        obs_dim=32,
        camera=-1,
        size=(64, 64),
    ):
        if task_action_space is None:
            task_action_space = [6]
        if 'MUJOCO_GL' not in os.environ:
            os.environ['MUJOCO_GL'] = 'egl'

        if isinstance(env, str):
            domain, task = env.split('_', 1)
            if camera == -1:
                camera = self.DEFAULT_CAMERAS.get(domain, 0)
            if domain == 'cup':
                domain = 'ball_in_cup'
            if domain == 'manip':
                dm_env = manipulation.load(task + '_vision')
            elif domain == 'rodent':
                dm_env = getattr(basic_rodent_2020, task)()
            else:
                dm_env = suite.load(domain, task)
        else:
            dm_env = env
            domain, task = 'unknown', 'unknown'

        self._dmenv = dm_env
        self._env = from_dm.FromDM(self._dmenv)
        self._env = embodied.wrappers.ActionRepeat(self._env, repeat)
        self._general_act_size = tuple(task_action_space)
        self._real_act_space = self._env.act_space
        self._obs_dim = obs_dim
        self._camera = camera
        self._size = size

        # Compute real proprio dim by probing the env.
        ts = self._dmenv.reset()
        real_obs = np.concatenate(
            [np.asarray(v).flatten() for v in ts.observation.values()]
        )
        self._real_obs_dim = real_obs.shape[0]

        if self._real_obs_dim > self._obs_dim:
            import warnings
            warnings.warn(
                f"[GeneralDMCPriori] Task {domain}_{task} has real obs dim "
                f"{self._real_obs_dim} > obs_dim={self._obs_dim}. "
                "Truncating observations – consider increasing obs_dim."
            )

        print(
            f"[GeneralDMCPriori] Task: {domain}_{task}, "
            f"real_obs_dim={self._real_obs_dim}, padded_obs_dim={self._obs_dim}, "
            f"real_act_dim={self._real_act_space['action'].shape[0]}, "
            f"general_act_dim={self._general_act_size}"
        )

    # ------------------------------------------------------------------
    # Spaces
    # ------------------------------------------------------------------

    @functools.cached_property
    def obs_space(self):
        basic = {
            'is_first':   elements.Space(bool),
            'is_last':    elements.Space(bool),
            'is_terminal':elements.Space(bool),
            'reward':     elements.Space(np.float32),
        }
        basic['state'] = elements.Space(np.float32, (self._obs_dim,))
        # Keep a small rendered image for logging / visualisation only.
        basic['log/image'] = elements.Space(np.uint8, self._size + (3,))
        return basic

    @functools.cached_property
    def act_space(self):
        act_low = self._real_act_space['action'].low[0]
        act_high = self._real_act_space['action'].high[0]
        general_act_space = self._real_act_space.copy()
        general_act_space['action'] = elements.Space(
            np.float64, self._general_act_size, low=act_low, high=act_high
        )
        return general_act_space

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, general_action):
        action = general_action.copy()
        # Slice generalised action to real task dimensionality.
        real_act_dim = self._real_act_space['action'].shape[0]
        action['action'] = action['action'][:real_act_dim]

        for key, space in self.act_space.items():
            if not space.discrete:
                assert np.isfinite(action[key]).all(), (key, action[key])

        raw_obs = self._env.step(action)

        # Build padded state vector from all proprio keys.
        proprio_parts = []
        skip_keys = {'is_first', 'is_last', 'is_terminal', 'reward', 'reset'}
        for key in sorted(raw_obs.keys()):
            if key in skip_keys:
                continue
            v = np.asarray(raw_obs[key]).flatten()
            proprio_parts.append(v)

        if proprio_parts:
            proprio_flat = np.concatenate(proprio_parts).astype(np.float32)
        else:
            proprio_flat = np.zeros(self._obs_dim, dtype=np.float32)

        # Pad or truncate to obs_dim.
        padded = np.zeros(self._obs_dim, dtype=np.float32)
        copy_len = min(proprio_flat.shape[0], self._obs_dim)
        padded[:copy_len] = proprio_flat[:copy_len]

        obs = {
            'is_first':    raw_obs['is_first'],
            'is_last':     raw_obs['is_last'],
            'is_terminal': raw_obs['is_terminal'],
            'reward':      raw_obs['reward'],
            'state':       padded,
            'log/image':   self._dmenv.physics.render(*self._size, camera_id=self._camera),
        }

        for key, space in self.obs_space.items():
            if np.issubdtype(space.dtype, np.floating):
                assert np.isfinite(obs[key]).all(), (key, obs[key])

        return obs
