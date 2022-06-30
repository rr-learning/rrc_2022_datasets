# Real Robot Challenge III (2022) Datasets

This repository provides offline reinforcement learning datasets collected on the TriFinger platform (simulated or real). It follows the interface suggested by [D4RL](https://github.com/rail-berkeley/d4rl). 

The datasets are used for the [Real Robot Challenge 2022](https://real-robot-challenge.com).

## Installation

To install the package run with python 3.8 in the root directory of the repository (we recommend doing this in a virtual environment):

```bash
pip install --upgrade pip  # make sure recent version of pip is installed
pip install .
```

## Usage

### Loading the dataset

The datasets are accessible via gym environments which are automatically registered when importing the package. They are automatically downloaded when requested and stored in `~/.rrc_2022_datasets` as HDF5 files.

The datasets are named following the pattern `trifinger-cube-task-source-quality-v0` where `task` is either `push` or `lift`, `source` is either `sim` or `real` and `quality` can be either `mixed` or `expert`.

By default the observations are loaded as flat arrays. For the simulated datasets the environment can be stepped and visualized. Example usage (also see `demo/load_dataset.py`):

```python
import gym

import rrc_2022_datasets

env = gym.make(
    "trifinger-cube-push-sim-expert-v0",
    disable_env_checker=True,
    visualization=True,  # enable visualization
)

dataset = env.get_dataset()

print("First observation: ", dataset["observations"][0])
print("First action: ", dataset["actions"][0])
print("First reward: ", dataset["rewards"][0])

obs = env.reset()
done = False
while not done:
    obs, rew, done, info = env.step(env.action_space.sample())
```

Alternatively, the observations can be obtained as nested dictionaries. This simplifies working with the data. As some parts of the observations might be more useful than others, it is also possible to filter the observations when requesting dictionaries (see `demo/load_filtered_dicts.py`):
```
    # Nested dictionary defines which observations to keep.
    # Everything that is not included or has value False
    # will be dropped.
    obs_to_keep = {
        "robot_observation": {
            "position": True,
            "velocity": True,
            "fingertip_force": False,
        },
        "object_observation": {"keypoints": True},
    }
    env = gym.make(
        args.env_name,
        disable_env_checker=True,
        # filter observations,
        obs_to_keep=obs_to_keep,
    )
```
To transform the observation back to a flat array after filtering, simply set the keyword argument `flatten_obs` to true. Note that the step and reset functions will transform observations in the same manner as the `get_dataset` method to ensure compatibility. A downside of working with observations in the form of dictionaries is that they cause a considerable memory overhead during dataset loading.

### Evaluating a policy in simulation

This package contains an executable module `rrc_2022_datasets.evaluate_pre_stage`, which
can be used to evaluate a policy in simulation.  As arguments it expects the task
("push" or "lift") and a Python class that implements the policy, following the
`PolicyBase` interface:

    python3 -m rrc_2022_datasets.evaluate_pre_stage push my_package.MyPolicy

For more options see `--help`.

This is also used for the evaluation of submissions in the pre-stage of the challenge
(hence the name).
