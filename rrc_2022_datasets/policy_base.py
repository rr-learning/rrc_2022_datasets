import typing
from abc import ABC, abstractmethod

import gym
import numpy as np


class PolicyBase(ABC):
    """Base class defining interface for policies."""

    def __init__(self, action_space: gym.Space):
        """
        Args:
            action_space:  Action space of the environment.
        """
        pass

    def reset(self) -> None:
        """Will be called at the beginning of each episode."""
        pass

    @abstractmethod
    def get_action(self, observation: typing.Dict[str, str]) -> np.ndarray:
        """Returns action that is executed on the robot.

        Args:
            Observation of the current time step.

        Returns:
            Action that is sent to the robot.
        """
        pass
