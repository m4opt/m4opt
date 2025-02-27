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
    ):
        self.min = min
        self.max = max

    @override
    def __call__(self, observer_location, target_coord, obstime):
        alt = target_coord.transform_to(
            AltAz(obstime=obstime, location=observer_location)
        ).alt
        return (self.min <= alt) & (alt <= self.max)
