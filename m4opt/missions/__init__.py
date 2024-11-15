"""This module contains builtin settings for supported missions."""

from ._core import Mission
from ._uvex import uvex
from ._ultrasat import ultrasat

__all__ = ("Mission", "uvex", "ultrasat")
