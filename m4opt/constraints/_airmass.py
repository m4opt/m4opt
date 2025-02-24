from typing import Optional

from astropy.coordinates import AltAz

from ..utils.typing_extensions import override
from ._core import Constraint


class AirmassConstraint(Constraint):
    """
    Constrain the airmass of a target.

    In the current implementation, the airmass is approximated by the secant of
    the zenith angle.

    .. note::
        The ``max`` and ``min`` arguments appear in the order (max, min)
        in this initializer to support the common case for users who care
        about the upper limit on the airmass (``max``) and not the lower
        limit.

    Parameters
    ----------
    max : float or `None`, optional
        Maximum airmass of the target. `None` indicates no limit.
    min : float or `None`, optional
        Minimum airmass of the target. Default is `1` (the zenith).
    boolean_constraint : bool, optional
        If True, the constraint returns a boolean array.
        If False, it returns a float array scaled between 0 and 1.

    Examples
    --------
    To create a constraint that requires the airmass to be "better than 2",
    i.e., at a higher altitude than airmass=2::

        >>> from astropy.coordinates import EarthLocation, SkyCoord
        >>> from astropy.time import Time
        >>> from astropy import units as u
        >>> from m4opt.constraints import AirmassConstraint
        >>> time = Time("2017-08-17T00:41:04Z")
        >>> target = SkyCoord.from_name("NGC 4993")
        >>> location = EarthLocation.of_site("Rubin Observatory")
        >>> constraint = AirmassConstraint(min=1, max=3)
        >>> constraint(location, target, time)
        np.True_
    """

    def __init__(
        self,
        max: Optional[float] = None,
        min: Optional[float] = 1,
        boolean_constraint: bool = True,
    ):
        self.min = min
        self.max = max
        self.boolean_constraint = boolean_constraint

    @override
    def __call__(self, observer_location, target_coord, obstime):
        """
        Compute the airmass constraint.

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
        secz = altaz.secz.value

        if self.boolean_constraint:
            if self.min is None and self.max is not None:
                return secz <= self.max
            elif self.max is None and self.min is not None:
                return self.min <= secz
            elif self.min is not None and self.max is not None:
                return (self.min <= secz) & (secz <= self.max)
            else:
                raise ValueError("No max and/or min specified in AirmassConstraint.")

        if self.max is None:
            raise ValueError("Cannot have a float AirmassConstraint if max is None.")

        min_val = self.min if self.min is not None else 1
        return AirmassConstraint.min_best_rescale(
            secz, min_val, self.max, less_than_min=0
        )

    @staticmethod
    def min_best_rescale(vals, min_val, max_val, less_than_min=1):
        """
        Rescales an input array ``vals`` to be a score (between 0 and 1),
        where ``min_val`` goes to 1, and ``max_val`` goes to 0.

        Parameters
        ----------
        vals : array-like
            The values that need to be rescaled.
        min_val : float
            Best acceptable value (rescales to 1).
        max_val : float
            Worst value cared about (rescales to 0).
        less_than_min : 0 or 1, optional
            What is returned for ``vals`` below ``min_val``. Defaults to 1.

        Returns
        -------
        numpy.ndarray
            Array of floats between 0 and 1 inclusive.

        Examples
        --------
        >>> import numpy as np
        >>> AirmassConstraint.min_best_rescale(np.array([1, 1.5, 2, 3, 4]), 1, 3)
        array([1. , 0.5, 0. , 0. , 0. ])
        """
        rescaled = (max_val - vals) / (max_val - min_val)
        rescaled[vals < min_val] = less_than_min
        rescaled[vals > max_val] = 0
        return rescaled
