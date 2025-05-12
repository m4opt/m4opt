from dataclasses import dataclass
from typing import Literal

from aep8 import flux
from astropy import units as u

from ._core import Constraint


@dataclass
class RadiationBeltConstraint(Constraint):
    """Constrain the flux of charged particles in the Earth's radiation belts.

    Notes
    -----
    This constraint evaluates the
    :doc:`NASA AE8/AP8 model from IRBEM <irbem:api/radiation_models>`.

    Examples
    --------
    >>> from astropy.coordinates import EarthLocation, SkyCoord
    >>> from astropy.time import Time
    >>> from astropy import units as u
    >>> from m4opt.constraints import RadiationBeltConstraint
    >>> time = Time("2017-08-17T00:41:04Z")
    >>> target = SkyCoord.from_name("NGC 4993")
    >>> location = EarthLocation.from_geodetic(35 * u.deg, -20 * u.deg, 300 * u.km)
    >>> constraint = RadiationBeltConstraint(
    ...     flux=20e3 * u.cm**-2 * u.s**-1, energy=1 * u.MeV,
    ...     particle="e", solar="max")
    >>> constraint(location, target, time)
    np.True_
    """

    flux: u.Quantity[u.physical.particle_flux]
    """The maximum flux in units compatible with cm^-2 s^-1."""

    energy: u.Quantity[u.physical.energy]
    """The particle energy in units compatible with MeV."""

    particle: Literal["e", "p"]
    """The particle species, ``"e"`` for electrons or ``"p"`` for protons."""

    solar: Literal["max", "min"]
    """The phase in the solar cycle, ``"max"`` for solar maximum or ``"min"`` for
    solar minimum."""

    def __call__(self, observer_location, target_coord, obstime):
        return (
            flux(
                observer_location,
                obstime,
                self.energy,
                kind="integral",
                solar=self.solar,
                particle=self.particle,
            )
            <= self.flux
        )
