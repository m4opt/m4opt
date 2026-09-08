from typing import Literal

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz
from astropy.modeling import Model, Parameter
from astropy.table import Table
from astropy.utils.data import download_file
from frozendict import frozendict
from synphot.spectrum import BaseSpectrum, Empirical1D, SpectralElement

from .._extrinsic import state

# Airmass beyond which the cosecant formula stops being worth trusting. A
# target this low is extinguished past any usable depth.
_MAX_AIRMASS = 10.0

type ExtinctionModelName = Literal[
    "ctio", "kpno", "lapalma", "mko", "mtham", "paranal", "apo"
]


def read_extinction_model(
    name: ExtinctionModelName,
):
    url = f"https://raw.githubusercontent.com/astropy/specreduce-data/refs/heads/main/specreduce_data/reference_data/extinction/{name}extinct.dat"
    filename = download_file(url, cache=True)
    table = Table.read(filename, format="ascii")
    return SpectralElement(
        Empirical1D,
        points=(table["col1"] * u.angstrom).to(BaseSpectrum._internal_wave_unit),
        lookup_table=table["col2"],
    )


def AtmosphericExtinction(site: ExtinctionModelName, airmass: float | None = None):
    """Atmospheric extinction.

    Parameters
    ----------
    site:
        The name of the site from which to use atmospheric extinction data.
        This can be any site
        :doc:`supported by the specreduce package <specreduce:extinction>`.
    airmass:
        Airmass.

    Examples
    --------

    The model gives the fraction of light transmitted by the atmosphere. You
    can set the airmass explicitly:

    >>> from astropy import units as u
    >>> from m4opt.synphot.extinction import AtmosphericExtinction
    >>> extinction = AtmosphericExtinction("kpno", airmass=1.0)  # doctest: +REMOTE_DATA
    >>> extinction(5000 * u.angstrom)  # doctest: +REMOTE_DATA
    <Quantity 0.84722741>

    Or you can leave it unspecified, to evaluate later for a given sky location
    using :meth:`m4opt.synphot.observing`:

    >>> from astropy.coordinates import EarthLocation, SkyCoord
    >>> from astropy.time import Time
    >>> from m4opt.synphot import observing
    >>> extinction = AtmosphericExtinction("kpno")  # doctest: +REMOTE_DATA
    >>> with observing(EarthLocation.of_site("Kitt Peak"), SkyCoord.from_name("NGC 4993"), Time("2017-08-17")):  # doctest: +REMOTE_DATA
    ...     extinction(5000 * u.angstrom)
    <Quantity 0.73398752>
    """
    if airmass is None:
        return SpectralElement(
            type(
                AtmosphericExtinctionForSkyCoord.__name__,
                (AtmosphericExtinctionForSkyCoord,),
                {"_extinction_curve": read_extinction_model(site)},
            )()
        )
    else:
        return SpectralElement(
            type(
                AtmosphericExtinctionForAirmass.__name__,
                (AtmosphericExtinctionForAirmass,),
                {"_extinction_curve": read_extinction_model(site)},
            )(airmass)
        )


class AtmosphericExtinctionBase(Model):
    n_inputs = 1
    n_outputs = 1
    input_units = frozendict(x=BaseSpectrum._internal_wave_unit)
    return_units = frozendict(y=u.dimensionless_unscaled)
    input_units_equivalencies = frozendict(x=u.spectral())

    # synphot.BaseSpectrum passes unitless quantities to the underlying model.
    _input_units_allow_dimensionless = True

    def evaluate(self, x, airmass):
        # synphot passes the wavelength through without units.
        if getattr(x, "unit", None) in (None, u.dimensionless_unscaled):
            x = x * self.input_units["x"]

        # The tabulated curve is in magnitudes of extinction per airmass, so
        # it is converted to a transmission here. It is stripped of its
        # dimensionless unit first: multiplying a dimensionless quantity by a
        # magnitude unit converts the value rather than labelling it.
        extinction = airmass * np.asarray(self._extinction_curve(x))
        return np.power(10.0, -0.4 * extinction)


class AtmosphericExtinctionForAirmass(AtmosphericExtinctionBase):
    airmass = Parameter()


class AtmosphericExtinctionForSkyCoord(AtmosphericExtinctionBase):
    @property
    def airmass(self):
        """Airmass of the line of sight in the current observing context."""
        s = state.get()
        frame = AltAz(location=s.observer_location, obstime=s.obstime)
        alt = s.target_coord.transform_to(frame).alt
        # The cosecant formula, matching
        # :class:`~m4opt.constraints.AirmassConstraint`. It runs away at the
        # horizon, so it is capped below; a target that low is opaque anyway,
        # and without the cap one below the horizon would come out amplified
        # rather than extinguished.
        sin_alt = np.sin(alt).to_value(u.dimensionless_unscaled)
        return 1 / np.clip(sin_alt, 1 / _MAX_AIRMASS, None)

    def at_airmass(self, airmass):
        """The same site's extinction, at an airmass given explicitly.

        Used to tabulate the extinction over a grid of airmasses, in the same
        way that dust extinction is tabulated over reddening.
        """
        return SpectralElement(
            type(
                AtmosphericExtinctionForAirmass.__name__,
                (AtmosphericExtinctionForAirmass,),
                {"_extinction_curve": self._extinction_curve},
            )(airmass)
        )

    def evaluate(self, x):
        return super().evaluate(x, self.airmass)
