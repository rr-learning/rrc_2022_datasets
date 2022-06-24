import typing
from abc import ABC, abstractmethod

import numpy as np


class PolicyBase(ABC):
    """Base class defining interface for policies."""

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
