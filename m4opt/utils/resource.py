"""Functions for monitoring resource usage."""

from functools import cache
from resource import RUSAGE_SELF, getrusage

import numpy as np

__all__ = ("get_maxrss_bytes",)


def _get_maxrss_raw():
    return getrusage(RUSAGE_SELF).ru_maxrss


def _get_bytes_per_maxrss_unit_func():
    """Measure the units in bytes of maxrss.

    The units are different on macOS/BSD and Linux. See
    https://github.com/python/cpython/issues/64667.
    """
    maxrss1 = _get_maxrss_raw()
    dummy = np.ones(1_000_000)
    maxrss2 = _get_maxrss_raw()
    return 1 << (
        10 * int(np.rint(np.log(dummy.nbytes / (maxrss2 - maxrss1)) / np.log(1024)))
    )


@cache
def _get_bytes_per_maxrss_unit():
    import multiprocessing

    with multiprocessing.Pool(1) as pool:
        return pool.apply(_get_bytes_per_maxrss_unit_func)


def get_maxrss_bytes():
    """Get the memory usage of the current process and all of its children."""
    return _get_maxrss_raw() * _get_bytes_per_maxrss_unit()
