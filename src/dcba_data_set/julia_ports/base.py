"""Abstract base class for graph generator configurations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class BaseGraphConfig(ABC):
    """Abstract base configuration for graph generators."""

    n: int

    @classmethod
    @abstractmethod
    def from_yaml(cls, config: dict[str, Any]) -> "BaseGraphConfig":
        """Create an instance from a YAML-like dictionary."""
        ...

    @abstractmethod
    def to_yaml(self) -> dict[str, Any]:
        """Convert configuration into a serialisable format."""
        ...
