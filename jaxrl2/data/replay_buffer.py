from typing import Optional, Union

import gym
import gym.spaces
import numpy as np

from jaxrl2.data.dataset import Dataset, DatasetDict


def _init_replay_dict(
    obs_space: gym.Space, capacity: int
) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _insert_recursively(
    dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int
):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        assert dataset_dict.keys() == data_dict.keys()
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()


def _nbytes_recursively(dataset_dict: DatasetDict) -> int:
    if isinstance(dataset_dict, np.ndarray):
        return int(dataset_dict.nbytes)
    if isinstance(dataset_dict, dict):
        return sum(_nbytes_recursively(value) for value in dataset_dict.values())
    raise TypeError()


class ReplayBuffer(Dataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
        physics_state_dim: Optional[int] = None,
    ):
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space, capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=bool),
        )
        if physics_state_dim is not None:
            dataset_dict['physics_states'] = np.zeros(
                (capacity, physics_state_dim), dtype=np.float64)

        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0
        self._physics_state_dim = physics_state_dim

    def __len__(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    def allocated_nbytes(self) -> int:
        return _nbytes_recursively(self.dataset_dict)

    def used_nbytes(self) -> int:
        if self._capacity <= 0:
            return 0
        return int(self.allocated_nbytes() * (self._size / self._capacity))

    def ram_usage(self) -> dict:
        allocated = self.allocated_nbytes()
        used = self.used_nbytes()
        return {
            "allocated_bytes": allocated,
            "used_bytes": used,
            "allocated_mb": allocated / (1024 ** 2),
            "used_mb": used / (1024 ** 2),
            "allocated_gb": allocated / (1024 ** 3),
            "used_gb": used / (1024 ** 3),
            "size": self._size,
            "capacity": self._capacity,
        }

    def insert(self, data_dict: DatasetDict):
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)
