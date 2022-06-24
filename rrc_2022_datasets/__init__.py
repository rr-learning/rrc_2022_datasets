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
        "dataset_url": "https://owncloud.tuebingen.mpg.de/index.php/s/XKcRys7JLPTyaqx/download",
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
        "dataset_url": "https://nextcloud.tuebingen.mpg.de/index.php/s/gjtikkDnQjmRAKg/download",
        "ref_min_score": 0.0,
        "ref_max_score": 1.0 * 30000 / 20,
        "real_robot": False,
        "trifinger_kwargs": {
            "episode_length": 30000,
            "difficulty": 4,
            "keypoint_obs": True,
            "obs_action_delay": 2,
        },
    },
    # real robot
    # ==========
]


def get_env(**kwargs):
    return TriFingerDatasetEnv(**kwargs)


for params in dataset_params:
    register(id=params["name"], entry_point="rrc_2022_datasets:get_env", kwargs=params)
