# Real Robot Challenge III (2022) Datasets

This repository provides offline reinforcement learning datasets collected on the TriFinger platform (simulated or real). It follows the interface suggested by [D4RL](https://github.com/rail-berkeley/d4rl). 

## Installation

To install the package run with python 3.8 in the root directory of the repository (we recommend doing this in a virtual environment):

```bash
pip install -e .
```

## Usage

The datasets are accessible via gym environments which are automatically registered when importing the package. They are automatically downloaded and stored in `~/.rrc_2022_datasets` as HDF5 files.

Example usage (also see `demo/load_dataset.py`):

```python
import gym

import trifinger_datasets

env = gym.make("trifinger-cube-push-sim-expert-v0")

dataset = env.get_dataset()

print("First observation: ", dataset["observations"][0])
print("First action: ", dataset["actions"][0])
print("First reward: ", dataset["rewards"][0])

obs = env.reset()
done = False
while not done:
    obs, rew, done, info = env.step(env.action_space.sample())
```
