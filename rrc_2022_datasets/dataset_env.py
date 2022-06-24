from copy import deepcopy
import os
from typing import Union, Tuple, Dict, Optional, List
import urllib.request

import gym
import gym.spaces
import h5py
import numpy as np
from tqdm import tqdm

from .sim_env import SimTriFingerCubeEnv


def download_dataset(url, name):
    data_dir = os.path.expanduser("~/.rrc_2022_datasets")
    os.makedirs(data_dir, exist_ok=True)
    local_path = os.path.join(data_dir, name + ".hdf5")
    if not os.path.exists(local_path):
        print(f'Downloading dataset "{url}" to "{local_path}".')
        urllib.request.urlretrieve(url, local_path)
        if not os.path.exists(local_path):
            raise IOError(f"Failed to download dataset from {url}.")
    return local_path


class TriFingerDatasetEnv(gym.Env):
    """TriFinger environment which can load an offline RL dataset from a file.

    Similar to D4RL's OfflineEnv but with slightly different data loading and
    options for customization of observation space."""

    def __init__(
        self,
        name,
        dataset_url,
        ref_max_score,
        ref_min_score,
        trifinger_kwargs,
        real_robot=False,
        visualization=None,
        obs_to_keep=None,
        flatten_obs=True,
        scale_obs=False,
        set_terminals=False,
        **kwargs,
    ):
        """Args:
        name (str): Name of the dataset.
        dataset_url (str): URL pointing to the dataset.
        ref_max_score (float): Maximum score (for score normalization)
        ref_min_score (float): Minimum score (for score normalization)
        trifinger_kwargs (dict): Keyword arguments for underlying
            SimTriFingerCubeEnv environment.
        real_robot (bool): Whether the data was collected on real
            robots.
        visualization (bool): Enables rendering for simulated
            environment.
        obs_to_keep (dict): Dictionary with the same structure as
            the observation of SimTriFingerCubeEnv. The boolean
            value of each item indicates whether it should be
            included in the observation. If None, the
            SimTriFingerCubeEnv is used.
        flatten_obs (bool): Whether to flatten the observation. Can
            be combined with obs_to_keep.
        scale_obs (bool): Whether to scale all components of the
            observation to interval [-1, 1]. Only implemented
            for flattend observations.
        """
        super().__init__(**kwargs)
        t_kwargs = deepcopy(trifinger_kwargs)
        if visualization is not None:
            t_kwargs["visualization"] = visualization
        # underlying simulated TriFinger environment
        self.sim_env = SimTriFingerCubeEnv(**t_kwargs)
        self._orig_obs_space = self.sim_env.observation_space

        self.name = name
        self.dataset_url = dataset_url
        self.ref_max_score = ref_max_score
        self.ref_min_score = ref_min_score
        self.real_robot = real_robot
        self.obs_to_keep = obs_to_keep
        self.flatten_obs = flatten_obs
        self.scale_obs = scale_obs
        self.set_terminals = set_terminals

        if scale_obs and not flatten_obs:
            raise NotImplementedError(
                "Scaling of observations only "
                "implemented for flattened observations, i.e., for "
                "flatten_obs=True."
            )

        # action space
        self.action_space = self.sim_env.action_space

        # observation space
        if self.obs_to_keep is not None:
            # construct filtered observation space
            self._filtered_obs_space = self._filter_dict(
                keys_to_keep=self.obs_to_keep, d=self.sim_env.observation_space
            )
        else:
            self._filtered_obs_space = self.sim_env.observation_space
        if self.flatten_obs:
            # flat obs space
            self.observation_space = gym.spaces.flatten_space(self._filtered_obs_space)
            if self.scale_obs:
                self._obs_unscaled_low = self.observation_space.low
                self._obs_unscaled_high = self.observation_space.high
                # scale observations to [-1, 1]
                self.observation_space = gym.spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=self.observation_space.shape,
                    dtype=self.observation_space.dtype,
                )
        else:
            self.observation_space = self._filtered_obs_space

    def _filter_dict(self, keys_to_keep, d):
        """Keep only a subset of keys in dict.

        Applied recursively.
        Args:
            keys_to_keep (dict): (Nested) dictionary with values being
                either a dict or a bolean indicating whether to keep
                an item.
            d (dict or gym.spaces.Dict): Dicitionary or Dict space that
                is to be filtered."""

        filtered_dict = {}
        for k, v in keys_to_keep.items():
            if isinstance(v, dict):
                subspace = self._filter_dict(v, d[k])
                filtered_dict[k] = subspace
            elif isinstance(v, bool) and v:
                filtered_dict[k] = d[k]
            elif not isinstance(v, bool):
                raise TypeError(
                    "Expected boolean to indicate whether item "
                    "in observation space is to be kept."
                )
        if isinstance(d, gym.spaces.Dict):
            filtered_dict = gym.spaces.Dict(spaces=filtered_dict)
        return filtered_dict

    def _scale_obs(self, obs: np.ndarray) -> np.ndarray:
        """Scale observation components to [-1, 1]."""

        interval = self._obs_unscaled_high.high - self._obs_unscaled_low.low
        a = (obs - self._obs_unscaled_low.low) / interval
        return a * 2.0 - 1.0

    def _process_obs(self, obs: np.ndarray) -> np.ndarray:
        """Process obs according to params."""

        if self.obs_to_keep is not None:
            # filter obs
            if self.obs_to_keep is not None:
                obs = self._filter_dict(self.obs_to_keep, obs)
        if self.flatten_obs and isinstance(obs, dict):
            # flatten obs
            obs = gym.spaces.flatten(self._filtered_obs_space, obs)
        if self.scale_obs:
            # scale obs
            obs = self._scale_obs(obs)
        return obs

    def get_dataset(self, h5path=None, clip=True):
        if h5path is None:
            h5path = download_dataset(self.dataset_url, self.name)

        data_dict = {}
        with h5py.File(h5path, "r") as dataset_file:
            for k in tqdm(dataset_file.keys(), desc="Loading datafile"):
                data_dict[k] = dataset_file[k][:]

        n_transitions = data_dict["observations"].shape[0]

        # clip to make sure that there are no outliers in the data
        if clip:
            orig_flat_obs_space = gym.spaces.flatten_space(self._orig_obs_space)
            data_dict["observations"] = data_dict["observations"].clip(
                min=orig_flat_obs_space.low,
                max=orig_flat_obs_space.high,
                dtype=orig_flat_obs_space.dtype,
            )

        if not (self.flatten_obs and self.obs_to_keep is None):
            # unflatten observations, i.e., turn them into dicts again
            unflattened_obs = []
            obs = data_dict["observations"]
            for i in range(obs.shape[0]):
                unflattened_obs.append(
                    gym.spaces.unflatten(self.sim_env.observation_space, obs[i, ...])
                )
            data_dict["observations"] = unflattened_obs

        # timeouts, terminals and info
        episode_ends = data_dict["episode_ends"]
        data_dict["timeouts"] = np.zeros(n_transitions, dtype=bool)
        if not self.set_terminals:
            data_dict["timeouts"][episode_ends] = True
        data_dict["terminals"] = np.zeros(n_transitions, dtype=bool)
        if self.set_terminals:
            data_dict["terminals"][episode_ends] = True
        data_dict["infos"] = [{} for _ in range(n_transitions)]

        # process obs (filtering, flattening, scaling)
        for i in range(n_transitions):
            data_dict["observations"][i] = self._process_obs(
                data_dict["observations"][i]
            )
        # turn observations into array if obs are flattened
        if self.flatten_obs:
            data_dict["observations"] = np.array(
                data_dict["observations"], dtype=self.observation_space.dtype
            )

        return data_dict

    def get_dataset_chunk(self, chunk_id, h5path=None):
        raise NotImplementedError()

    def compute_reward(
        self, achieved_goal: dict, desired_goal: dict, info: dict
    ) -> float:
        """Compute the reward for the given achieved and desired goal.
        Args:
            achieved_goal: Current pose of the object.
            desired_goal: Goal pose of the object.
            info: An info dictionary containing a field "time_index" which
                contains the time index of the achieved_goal.
        Returns:
            The reward that corresponds to the provided achieved goal w.r.t. to
            the desired goal.
        """
        return self.sim_env.compute_reward(achieved_goal, desired_goal, info)

    def step(
        self, action: np.ndarray, **kwargs
    ) -> Tuple[Union[Dict, np.ndarray], float, bool, Dict]:
        if self.real_robot:
            raise NotImplementedError(
                "The step method is not available for real-robot data."
            )
        obs, rew, done, info = self.sim_env.step(action, **kwargs)
        # process obs
        processed_obs = self._process_obs(obs)
        return processed_obs, rew, done, info

    def reset(
        self, return_info: bool = False
    ) -> Union[Union[Dict, np.ndarray], Tuple[Union[Dict, np.ndarray], Dict]]:
        if self.real_robot:
            raise NotImplementedError(
                "The reset method is not available for real-robot data."
            )
        rvals = self.sim_env.reset(return_info)
        if return_info:
            obs, info = rvals
        else:
            obs = rvals
        # process obs
        processed_obs = self._process_obs(obs)
        if return_info:
            return processed_obs, info
        else:
            return processed_obs

    def seed(self, seed: Optional[int] = None) -> List[int]:
        return self.sim_env.seed(seed)

    def render(self, mode: str = "human"):
        if self.real_robot:
            raise NotImplementedError(
                "The render method is not available for real-robot data."
            )
        self.sim_env.render(mode)

    def reset_fingers(self, reset_wait_time: int = 3000, return_info: bool = False):
        if self.real_robot:
            raise NotImplementedError(
                "The reset_fingers method is not available for real-robot data."
            )
        rvals = self.sim_env.reset_fingers(reset_wait_time, return_info)
        if return_info:
            obs, info = rvals
        else:
            obs = rvals
        # process obs
        processed_obs = self._process_obs(obs)
        if return_info:
            return processed_obs, info
        else:
            return processed_obs
