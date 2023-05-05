"""A class that does not store any train_auto_glm_data. This is the default memory provider."""
from __future__ import annotations

from typing import Any

from autogpt.memory.base import MemoryProviderSingleton


class NoMemory(MemoryProviderSingleton):
    """
    A class that does not store any train_auto_glm_data. This is the default memory provider.
    """

    def __init__(self, cfg):
        """
        Initializes the NoMemory provider.

        Args:
            cfg: The config object.

        Returns: None
        """
        pass

    def add(self, data: str) -> str:
        """
        Adds a train_auto_glm_data point to the memory. No action is taken in NoMemory.

        Args:
            data: The train_auto_glm_data to add.

        Returns: An empty string.
        """
        return ""

    def get(self, data: str) -> list[Any] | None:
        """
        Gets the train_auto_glm_data from the memory that is most relevant to the given train_auto_glm_data.
        NoMemory always returns None.

        Args:
            data: The train_auto_glm_data to compare to.

        Returns: None
        """
        return None

    def clear(self) -> str:
        """
        Clears the memory. No action is taken in NoMemory.

        Returns: An empty string.
        """
        return ""

    def get_relevant(self, data: str, num_relevant: int = 5) -> list[Any] | None:
        """
        Returns all the train_auto_glm_data in the memory that is relevant to the given train_auto_glm_data.
        NoMemory always returns None.

        Args:
            data: The train_auto_glm_data to compare to.
            num_relevant: The number of relevant train_auto_glm_data to return.

        Returns: None
        """
        return None

    def get_stats(self):
        """
        Returns: An empty dictionary as there are no stats in NoMemory.
        """
        return {}
