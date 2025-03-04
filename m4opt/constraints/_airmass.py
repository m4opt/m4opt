import numpy as np
from astropy import units as u

from ._altitude import AltitudeConstraint


class AirmassConstraint(AltitudeConstraint):
    """
    Constrains the airmass of a target by converting airmass limits to altitude limits.

    The airmass is approximated as the secant of the zenith angle, which allows
    for an equivalent formulation in terms of altitude constraints. This class
    extends `AltitudeConstraint` and internally converts the given airmass
    limits into corresponding altitude limits.

    Parameters
    ----------
    max_airmass : float
        Maximum airmass of the target (corresponds to minimum altitude).
    min_airmass : float,
        Minimum airmass of the target (corresponds to maximum altitude).
        Default is `1` (the zenith).

    Notes
    -----
    - The conversion from airmass to altitude follows:
        `altitude = arcsin(1 / airmass)`, assuming a standard atmosphere.

    Examples
    --------
    >>> from astropy.coordinates import EarthLocation, SkyCoord
    >>> from astropy.time import Time
    >>> from astropy import units as u
    >>> from m4opt.constraints import AirmassConstraint
    >>> time = Time("2017-08-17T00:41:04Z")
    >>> target = SkyCoord.from_name("NGC 4993")
    >>> location = EarthLocation.of_site("Rubin Observatory")
    >>> constraint = AirmassConstraint(max_airmass=3, min_airmass=1)
    >>> constraint(location, target, time)
    np.True_
    """

    def __init__(
        self,
        max_airmass: float,
        min_airmass: float = 1.0,
    ):
        min_airmass, max_airmass = sorted([min_airmass, max_airmass])
        min_alt = np.arcsin(1 / max_airmass) * u.rad
        max_alt = np.arcsin(1 / min_airmass) * u.rad

        super().__init__(min=min_alt, max=max_alt)
