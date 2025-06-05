"""This module contains builtin settings for supported missions."""

from ._core import Mission
from ._lsst import lsst
from ._ultrasat import ultrasat
from ._uvex import uvex
from ._ztf import ztf

__all__ = ("Mission", "lsst", "ultrasat", "uvex", "ztf")
