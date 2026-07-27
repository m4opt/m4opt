"""This module contains builtin settings for supported missions."""

from ._core import Mission
from ._registry import get, names

__all__ = (
    "Mission",
    "get",
    "names",
)
