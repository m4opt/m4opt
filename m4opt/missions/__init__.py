"""This module contains builtin settings for supported missions."""

from ._core import Mission
from ._ultrasat import ultrasat
from ._uvex import uvex

__all__ = ("Mission", "uvex", "ultrasat")
