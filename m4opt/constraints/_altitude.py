from typing import Optional

from astropy import units as u
from astropy.coordinates import AltAz

from ..utils.typing_extensions import override
from ._core import Constraint


class AltitudeConstraint(Constraint):
    """
    Constrain the altitude of the target.

    Parameters
    ----------
    min : `~astropy.units.Quantity`, optional
        Minimum altitude of the target (inclusive). Default is `-90 deg`.
    max : `~astropy.units.Quantity`, optional
        Maximum altitude of the target (inclusive). Default is `90 deg`.
    boolean_constraint : bool, optional
        If True, the constraint returns a boolean array.
        If False, it returns a float array scaled between 0 and 1.

    Examples
    --------
    >>> from astropy.coordinates import EarthLocation, SkyCoord
    >>> from astropy.time import Time
    >>> from astropy import units as u
    >>> from m4opt.constraints import AltitudeConstraint
    >>> time = Time("2017-08-17T00:41:04Z")
    >>> target = SkyCoord.from_name("NGC 4993")
    >>> location = EarthLocation.of_site("Rubin Observatory")
    >>> constraint = AltitudeConstraint(min=20*u.deg, max=85*u.deg)
    >>> constraint(location, target, time)
    np.True_
    """

    def __init__(
        self,
        min: Optional[u.Quantity] = -90 * u.deg,
        max: Optional[u.Quantity] = 90 * u.deg,
        boolean_constraint: bool = True,
    ):
        self.min = min
        self.max = max
        self.boolean_constraint = boolean_constraint

    @override
    def __call__(self, observer_location, target_coord, obstime):
        """
        Compute the altitude constraint.

        Parameters
        ----------
        observer_location : `~astropy.coordinates.EarthLocation`
            The observing location.
        target_coord : `~astropy.coordinates.SkyCoord`
            The celestial coordinates of the target.
        obstime : `~astropy.time.Time`
            The observation time.

        Returns
        -------
        `numpy.ndarray`
            Boolean mask (if `boolean_constraint=True`) or scaled values (if `False`).
        """
        altaz = target_coord.transform_to(
            AltAz(obstime=obstime, location=observer_location)
        )
        alt = altaz.alt

        if self.boolean_constraint:
            return (self.min <= alt) & (alt <= self.max)
        return AltitudeConstraint.max_best_rescale(
            alt, self.min, self.max, greater_than_max=0
        )

    @staticmethod
    def max_best_rescale(vals, min_val, max_val, greater_than_max=1):
        """
        Rescales an input array ``vals`` to be a score (between 0 and 1),
        where ``max_val`` goes to 1, and ``min_val`` goes to 0.

        Parameters
        ----------
        vals : array-like
            The values that need to be rescaled.
        min_val : float
            Worst acceptable value (rescales to 0).
        max_val : float
            Best value cared about (rescales to 1).
        greater_than_max : 0 or 1, optional
            What is returned for ``vals`` above ``max_val``. Defaults to 1.

        Returns
        -------
        numpy.ndarray
            Array of floats between 0 and 1 inclusive.

        Examples
        --------
        >>> import numpy as np
        >>> AltitudeConstraint.max_best_rescale(np.array([20, 30, 40, 45, 55, 70]), 35, 60)
        array([0. , 0. , 0.2, 0.4, 0.8, 1. ])
        """
        rescaled = (vals - min_val) / (max_val - min_val)
        rescaled[vals < min_val] = 0
        rescaled[vals > max_val] = greater_than_max
        return rescaled
