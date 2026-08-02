import numpy as np
from astropy import units as u

from ._positional import AltitudeConstraint


def airmass_to_altitude(
    airmass: float | u.Quantity[u.physical.dimensionless],
) -> u.Quantity[u.physical.angle]:
    """Convert airmass to altitude using the cosecant formula.

    Examples
    --------
    You can pass either floating-point values or dimensionless values with units.

    >>> from astropy import units as u
    >>> from m4opt.constraints._airmass import airmass_to_altitude
    >>> airmass_to_altitude(2.5)
    <Quantity 0.41151685 rad>
    >>> airmass_to_altitude(2.5 * u.dimensionless_unscaled)
    <Quantity 0.41151685 rad>
    """
    return np.arcsin(u.dimensionless_unscaled / airmass)


class AirmassConstraint(AltitudeConstraint):
    """
    Constrains the airmass of a target by converting airmass limits to altitude limits.

    The airmass is approximated as the secant of the zenith angle, which allows
    for an equivalent formulation in terms of altitude constraints. This class
    extends `AltitudeConstraint` and internally converts the given airmass
    limits into corresponding altitude limits.

    Parameters
    ----------
    max
        Maximum airmass of the target (corresponds to minimum altitude).
    min
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
    >>> constraint = AirmassConstraint(3)
    >>> constraint(location, target, time)
    np.True_
    """

    def __init__(
        self,
        max: float | u.Quantity[u.physical.dimensionless],
        min: float | u.Quantity[u.physical.dimensionless] = 1.0,
    ):
        super().__init__(*airmass_to_altitude([max, min]))
