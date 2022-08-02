__version__ = "1.0.1"

from gym.envs.registration import register

from .dataset_env import TriFingerDatasetEnv
from .evaluation import Evaluation
from .policy_base import PolicyBase


dataset_params = [
    # simulation
    # =========
    # push-expert
    {
        "name": "trifinger-cube-push-sim-expert-v0",
        "dataset_url": (
            "https://owncloud.tuebingen.mpg.de/index.php/s/XKcRys7JLPTyaqx/download"
        ),
        "ref_min_score": 0.0,
        "ref_max_score": 1.0 * 15000 / 20,
        "real_robot": False,
        "trifinger_kwargs": {
            "episode_length": 750,
            "difficulty": 1,
            "keypoint_obs": True,
            "obs_action_delay": 10,
        },
    },
    # lift-expert
    {
        "name": "trifinger-cube-lift-sim-expert-v0",
        "dataset_url": (
            "https://nextcloud.tuebingen.mpg.de/index.php/s/fW2sqpBwnBzsjXY/download"
        ),
        "ref_min_score": 0.0,
        "ref_max_score": 1.0 * 30000 / 20,
        "real_robot": False,
        "trifinger_kwargs": {
            "episode_length": 1500,
            "difficulty": 4,
            "keypoint_obs": True,
            "obs_action_delay": 2,
        },
    },
    # real robot
    # ==========
    # real-robot stage/pushing expert
    {
        "name": "trifinger-cube-push-real-expert-v0",
        "dataset_url": "http://robots.real-robot-challenge.com/public/rrc2022/datasets/trifinger-cube-push-real-expert-v0.hdf5",
        "ref_min_score": 0.0,
        "ref_max_score": 1.0 * 15000 / 20,
        "real_robot": True,
        "trifinger_kwargs": {
            "episode_length": 750,
            "difficulty": 1,
            "keypoint_obs": True,
            "obs_action_delay": 10,
        },
    },
    # real-robot stage/pushing mixed
    {
        "name": "trifinger-cube-push-real-mixed-v0",
        "dataset_url": "http://robots.real-robot-challenge.com/public/rrc2022/datasets/trifinger-cube-push-real-mixed-v0.hdf5",
        "ref_min_score": 0.0,
        "ref_max_score": 1.0 * 15000 / 20,
        "real_robot": True,
        "trifinger_kwargs": {
            "episode_length": 750,
            "difficulty": 1,
            "keypoint_obs": True,
            "obs_action_delay": 10,
        },
    },
    # real-robot stage/lifting expert
    {
        "name": "trifinger-cube-lift-real-expert-v0",
        "dataset_url": "http://robots.real-robot-challenge.com/public/rrc2022/datasets/trifinger-cube-lift-real-expert-v0.hdf5",
        "ref_min_score": 0.0,
        "ref_max_score": 1.0 * 30000 / 20,
        "real_robot": True,
        "trifinger_kwargs": {
            "episode_length": 1500,
            "difficulty": 4,
            "keypoint_obs": True,
            "obs_action_delay": 2,
        },
    },
    # real-robot stage/lifting mixed
    {
        "name": "trifinger-cube-lift-real-mixed-v0",
        "dataset_url": "http://robots.real-robot-challenge.com/public/rrc2022/datasets/trifinger-cube-lift-real-mixed-v0.hdf5",
        "ref_min_score": 0.0,
        "ref_max_score": 1.0 * 30000 / 20,
        "real_robot": True,
        "trifinger_kwargs": {
            "episode_length": 1500,
            "difficulty": 4,
            "keypoint_obs": True,
            "obs_action_delay": 2,
        },
    },
]


def get_env(**kwargs):
    return TriFingerDatasetEnv(**kwargs)


for params in dataset_params:
    register(id=params["name"], entry_point="rrc_2022_datasets:get_env", kwargs=params)


__all__ = ("TriFingerDatasetEnv", "Evaluation", "PolicyBase", "get_env")
