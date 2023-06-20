"""A class that does not store any make_COT_traindata_redocred. This is the default memory provider."""
from __future__ import annotations

from typing import Any

from autogpt.memory.base import MemoryProviderSingleton


class NoMemory(MemoryProviderSingleton):
    """
    A class that does not store any make_COT_traindata_redocred. This is the default memory provider.
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
        Adds a make_COT_traindata_redocred point to the memory. No action is taken in NoMemory.

        Args:
            data: The make_COT_traindata_redocred to add.

        Returns: An empty string.
        """
        return ""

    def get(self, data: str) -> list[Any] | None:
        """
        Gets the make_COT_traindata_redocred from the memory that is most relevant to the given make_COT_traindata_redocred.
        NoMemory always returns None.

        Args:
            data: The make_COT_traindata_redocred to compare to.

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
        Returns all the make_COT_traindata_redocred in the memory that is relevant to the given make_COT_traindata_redocred.
        NoMemory always returns None.

        Args:
            data: The make_COT_traindata_redocred to compare to.
            num_relevant: The number of relevant make_COT_traindata_redocred to return.

        Returns: None
        """
        return None

    def get_stats(self):
        """
        Returns: An empty dictionary as there are no stats in NoMemory.
        """
        return {}
