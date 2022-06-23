from abc import ABC, abstractmethod


class PolicyBase(ABC):
    """Base class defining interface for policies."""

    @abstractmethod
    def get_action(self, obs):
        pass
