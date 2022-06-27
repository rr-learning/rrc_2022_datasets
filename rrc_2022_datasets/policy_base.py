import typing
from abc import ABC, abstractmethod

import gym
import numpy as np

ObservationType = typing.Union[np.ndarray, typing.Dict[str, typing.Any]]


class PolicyBase(ABC):
    """Base class defining interface for policies."""

    def __init__(
        self, action_space: gym.Space, observation_space: gym.Space, episode_length: int
    ):
        """
        Args:
            action_space:  Action space of the environment.
            observation_space:  Observation space of the environment.
            episode_length:  Number of steps in one episode.
        """
        pass

    def reset(self) -> None:
        """Will be called at the beginning of each episode."""
        pass

    @abstractmethod
    def get_action(self, observation: ObservationType) -> np.ndarray:
        """Returns action that is executed on the robot.

        Args:
            Observation of the current time step.

        Returns:
            Action that is sent to the robot.
        """
        pass
