"""Mixed integer linear programs (MILP) with pluggable solver backends.

Supported backends:
- ``gurobi``: Uses `gurobipy <https://pypi.org/project/gurobipy/>`_.
- ``cplex``: Uses `cplex <https://pypi.org/project/cplex/>`_ and
  `docplex <https://pypi.org/project/docplex/>`_.

The backend is auto-detected based on which packages are installed (CPLEX is
preferred if both are available). You can override the choice with
:func:`set_backend` or the ``M4OPT_SOLVER`` environment variable.
"""

import os

from ._base import (
    ProgressData,
    ProgressDataRecorder,
    SolveDetails,
    VariableArray,
)

__all__ = (
    "Model",
    "ProgressData",
    "ProgressDataRecorder",
    "SolveDetails",
    "SolveSolution",
    "VariableArray",
    "set_backend",
)

_backend = None


def set_backend(name: str):
    """Set the solver backend.

    Parameters
    ----------
    name
        One of ``'gurobi'`` or ``'cplex'``.
    """
    global _backend
    if name not in ("gurobi", "cplex"):
        raise ValueError(f"Unknown backend: {name!r}. Choose 'gurobi' or 'cplex'.")
    _backend = name


def _get_backend():
    """Determine which backend to use via explicit setting, env var, or auto-detect."""
    if _backend is not None:
        return _backend
    env = os.environ.get("M4OPT_SOLVER")
    if env:
        return env
    try:
        import cplex  # noqa: F401

        return "cplex"
    except ImportError:
        pass
    try:
        import gurobipy  # noqa: F401

        return "gurobi"
    except ImportError:
        pass
    raise ImportError(
        "No MILP solver found. Install gurobipy or cplex.\n"
        "  pip install gurobipy   # for Gurobi\n"
        "  pip install cplex docplex   # for CPLEX"
    )


def Model(**kwargs):
    """Create a MILP model using the active backend.

    Parameters
    ----------
    **kwargs
        Keyword arguments forwarded to the backend Model constructor
        (``timelimit``, ``jobs``, ``memory``, ``lowercutoff``, ``verbose``).

    Returns
    -------
    model
        A backend-specific model instance with a unified API.
    """
    backend = _get_backend()
    if backend == "gurobi":
        from ._gurobi import GurobiModel

        return GurobiModel(**kwargs)
    elif backend == "cplex":
        from ._cplex import CplexModel

        return CplexModel(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")


def SolveSolution(*args, **kwargs):
    """Create a SolveSolution using the active backend.

    Normally you do not need to construct this directly; it is returned by
    :meth:`Model.solve`.
    """
    backend = _get_backend()
    if backend == "gurobi":
        from ._gurobi import GurobiSolveSolution

        return GurobiSolveSolution(*args, **kwargs)
    elif backend == "cplex":
        from ._cplex import CplexSolveSolution

        return CplexSolveSolution(*args, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")
