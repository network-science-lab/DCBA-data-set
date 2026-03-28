"""Abstract base class for graph generator configurations."""

from typing import Any

from pydantic import BaseModel


class BaseGraphConfig(BaseModel):
    """Abstract base configuration for graph generators."""

    n: int

    @classmethod
    def from_yaml(cls, config: dict[str, Any]) -> "BaseGraphConfig":
        """Create an instance from a YAML-like dictionary."""
        raise NotImplementedError

    def to_yaml(self) -> dict[str, Any]:
        """Convert configuration into a serialisable format."""
        raise NotImplementedError
