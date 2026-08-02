"""Command line interface."""

# Import these modules only to register the subcommands.
from . import (
    animate,  # noqa: F401
    prime,  # noqa: F401
    schedule,  # noqa: F401
)
from .core import app

__all__ = ("app",)
