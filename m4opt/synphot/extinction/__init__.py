"""Extinction models."""

from ._atmosphere import AtmosphericExtinction
from ._dust import DustExtinction

__all__ = (
    "AtmosphericExtinction",
    "DustExtinction",
)
