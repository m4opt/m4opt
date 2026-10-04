"""Convert sampled field-of-regard constraints to observing intervals."""

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from ..utils.numpy import clump_nonzero_inclusive
from ._core import Constraint

__all__ = ("visibility_windows",)


def visibility_windows(
    constraint: Constraint,
    observer_location: EarthLocation,
    target_coord: SkyCoord,
    obstime: Time,
    *,
    min_duration: u.Quantity = 0 * u.s,
    time_margin: u.Quantity = 0 * u.s,
) -> list[Time]:
    """
    Find observing windows from constraints evaluated on a time grid.

    Parameters
    ----------
    constraint
        Field-of-regard constraint, including any logical combinations.
    observer_location
        Observer location, either scalar or sampled on the time grid.
    target_coord
        A scalar target or a one-dimensional array of targets.
    obstime
        One-dimensional, strictly increasing sample times. The first and last
        samples bound the planning horizon; spacing may be nonuniform.
    min_duration
        Minimum window duration after applying the time margin. Include
        acquisition overhead here if it must also fit inside a window.
    time_margin
        Duration removed from each end of every sampled window.

    Returns
    -------
    list of astropy.time.Time
        One array of shape ``(n_windows, 2)`` per target, containing closed
        start/end pairs in the input time scale. A scalar target still returns
        a one-element list. Targets with no usable windows have shape ``(0, 2)``.

    Notes
    -----
    Each window runs from the first to the last passing sample in a contiguous
    run. Boundaries are not extrapolated into neighboring failing samples.
    An isolated passing sample gives a zero-duration window when both duration
    options are zero. An empty time grid returns empty windows. Minimum-duration
    comparisons allow one nanosecond for floating-point time arithmetic.

    The constraint is only tested at the supplied samples. Short visibility
    intervals or interruptions between samples can be missed. Choose a grid
    suited to the constraint's variation and independently check scheduled
    exposures at finer spacing. The time margin shrinks windows but does not
    establish visibility between samples or refine transition times.
    """
    if obstime.ndim != 1:
        raise ValueError("obstime must be one-dimensional")
    if target_coord.ndim > 1:
        raise ValueError("target_coord must be scalar or one-dimensional")
    for name, value in (("min_duration", min_duration), ("time_margin", time_margin)):
        seconds = value.to_value(u.s)
        if np.ndim(seconds) or not np.isfinite(seconds) or seconds < 0:
            raise ValueError(f"{name} must be a finite, nonnegative scalar duration")
    # Subtraction preserves the precision of Astropy's two-part time values.
    offsets = (obstime - obstime[:1]).to_value(u.s) if len(obstime) else np.array([])
    if np.ma.is_masked(offsets) or not np.all(np.isfinite(offsets)):
        raise ValueError("obstime must contain finite, unmasked times")
    if np.any(np.diff(offsets) <= 0):
        raise ValueError("obstime must be strictly increasing")
    targets = target_coord.reshape(-1)
    if not len(obstime) or not len(targets):
        return [obstime[:0].reshape(0, 2) for _ in targets]
    passed = constraint(observer_location, targets[:, np.newaxis], obstime)
    if np.ma.is_masked(passed):
        raise ValueError("constraint must return unmasked visibility values")
    passed = np.broadcast_to(passed, (len(targets), len(obstime)))
    windows = []
    for indices in clump_nonzero_inclusive(passed):
        starts = obstime[indices[:, 0]] + time_margin
        ends = obstime[indices[:, 1]] - time_margin
        duration = (ends - starts).to_value(u.s)
        minimum = min_duration.to_value(u.s)
        keep = (duration >= 0) & (
            (duration >= minimum) | np.isclose(duration, minimum, rtol=0, atol=1e-9)
        )
        # Indexing the original Time array also preserves its format and scale.
        pairs = obstime[indices[keep]].copy()
        pairs[:, 0] = starts[keep]
        pairs[:, 1] = ends[keep]
        windows.append(pairs)
    return windows
