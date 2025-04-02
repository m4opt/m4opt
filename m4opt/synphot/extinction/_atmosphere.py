from typing import Literal, TypeAlias

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz
from astropy.modeling import Model, Parameter
from astropy.table import Table
from astropy.utils.data import download_file
from synphot.spectrum import BaseSpectrum, Empirical1D, SpectralElement

from .._extrinsic import state

ExtinctionModelName: TypeAlias = Literal[
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

    You can create an extinction model with an explicitly set value of E(B-V):

    >>> from astropy import units as u
    >>> from m4opt.synphot.extinction import AtmosphericExtinction
    >>> extinction = AtmosphericExtinction("kpno", airmass=1.0)
    >>> extinction(10 * u.micron)
    <Quantity 0.024>

    Or you can leave it unspecified, to evaluate later for a given sky location
    using :meth:`m4opt.synphot.observing`:

    >>> from astropy.coordinates import EarthLocation, SkyCoord
    >>> from astropy.time import Time
    >>> from m4opt.synphot import observing
    >>> extinction = AtmosphericExtinction("kpno")
    >>> with observing(EarthLocation.of_site("Kitt Peak"), SkyCoord.from_name("NGC 4993"), Time("2017-08-17")):
    ...     extinction(10 * u.micron)
    <Quantity 0.04477044>
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
    input_units = {"x": BaseSpectrum._internal_wave_unit}
    return_units = {"y": u.dimensionless_unscaled}
    input_units_equivalencies = {"x": u.spectral()}

    # argh, why!! synphot.BaseSpectrum passes unitless quantities to the underlying model.
    _input_units_allow_dimensionless = True

    def evaluate(self, x, airmass):
        # Handle dimensionless units
        if (
            not hasattr(x, "unit")
            or u.unit is None
            or u.unit is u.dimensionless_unscaled
        ):
            x = x * self.input_units["x"]

        return (
            airmass * self._extinction_curve(x) * u.mag(u.dimensionless_unscaled)
        ).to_value(u.dimensionless_unscaled)


class AtmosphericExtinctionForAirmass(AtmosphericExtinctionBase):
    airmass = Parameter()


class AtmosphericExtinctionForSkyCoord(AtmosphericExtinctionBase):
    def evaluate(self, x):
        s = state.get()
        frame = AltAz(location=s.observer_location, obstime=s.obstime)
        alt = s.target_coord.transform_to(frame).alt
        airmass = 1 / np.sin(alt).to_value(u.dimensionless_unscaled)
        return super().evaluate(x, airmass)
