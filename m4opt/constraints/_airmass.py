from typing import Optional

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle

from ..utils.typing_extensions import override
from .altitude import AltitudeConstraint


class AirmassConstraint(AltitudeConstraint):
    """
    Constrains the airmass of a target by converting airmass limits to altitude limits.

    The airmass is approximated as the secant of the zenith angle, which allows
    for an equivalent formulation in terms of altitude constraints. This class
    extends `AltitudeConstraint` and internally converts the given airmass
    limits into corresponding altitude limits.

    Parameters
    ----------
    min_airmass : float or `None`, optional
        Minimum airmass of the target. Default is `1` (the zenith).
        `None` indicates no lower limit.
    max_airmass : float or `None`, optional
        Maximum airmass of the target. `None` indicates no upper limit.

    Examples
    --------
    To create a constraint that ensures the airmass is below 3, i.e., at a
    higher altitude than that corresponding to airmass=3::

        >>> from astropy.coordinates import EarthLocation, SkyCoord
        >>> from astropy.time import Time
        >>> from astropy import units as u
        >>> from m4opt.constraints import AirmassConstraint
        >>> time = Time("2017-08-17T00:41:04Z")
        >>> target = SkyCoord.from_name("NGC 4993")
        >>> location = EarthLocation.of_site("Rubin Observatory")
        >>> constraint = AirmassConstraint(min_airmass=1, max_airmass=3)
        >>> constraint(location, target, time)
        np.True_

    Notes
    -----
    - The conversion from airmass to altitude follows the relation:
        `altitude = arcsin(1 / airmass)`, which assumes a standard atmosphere.
    """

    def __init__(
        self,
        min_airmass: Optional[float] = 1,
        max_airmass: Optional[float] = None,
    ):
        min_alt = (
            Angle(90 * u.deg)
            if min_airmass is None
            else Angle(np.degrees(np.arcsin(1 / min_airmass)) * u.deg)
        )
        max_alt = (
            Angle(np.nan * u.deg)
            if max_airmass is None
            else Angle(np.degrees(np.arcsin(1 / max_airmass)) * u.deg)
        )

        super().__init__(min=min_alt, max=max_alt)

    @override
    def __call__(self, observer_location, target_coord, obstime):
        """
        Compute the airmass constraint by leveraging altitude constraints.
        """
        return super().__call__(observer_location, target_coord, obstime)
