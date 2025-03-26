import warnings
from functools import cache

import numpy as np
from astropy import units as u
from astropy.modeling import Model, Parameter
from astropy.utils.data import download_file
from dust_extinction.parameter_averages import G23

with warnings.catch_warnings():
    # Suppress configuration file warnings from the dustmaps package.
    # We are not using the configuration file.
    warnings.simplefilter("ignore", UserWarning)
    from dustmaps.planck import PlanckGNILCQuery
from synphot.spectrum import BaseSpectrum, SpectralElement

from .._extrinsic import state

reddening_law = G23()


@cache
def dust_map():
    sources = [
        "https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/component-maps/foregrounds/COM_CompMap_Dust-GNILC-Model-Opacity_2048_R2.01.fits",
        "https://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=COM_CompMap_Dust-GNILC-Model-Opacity_2048_R2.01.fits",
    ]
    return PlanckGNILCQuery(download_file(sources[-1], cache=True, sources=sources))


def DustExtinction(Ebv: float | None = None):
    """Milky Way dust extinction.

    Parameters
    ----------
    Ebv:
        Reddening color excess, :math:`E(B-V)`.

    Notes
    -----
    We use :class:`dust_extinction.parameter_averages.G23` because of its broad
    wavelength coverage from ultraviolet to infrared.

    Examples
    --------

    You can create an extinction model with an explicitly set value of E(B-V):

    >>> from astropy import units as u
    >>> from m4opt.synphot.extinction import DustExtinction
    >>> extinction = DustExtinction(Ebv=1.0)
    >>> extinction(10 * u.micron)
    <Quantity 0.7903132>

    Or you can leave it unspecified, to evaluate later for a given sky location
    using :meth:`m4opt.synphot.observing`:

    >>> from astropy.coordinates import EarthLocation, SkyCoord
    >>> from astropy.time import Time
    >>> from m4opt.synphot import observing
    >>> extinction = DustExtinction()
    >>> with observing(EarthLocation.of_site("Las Campanas Observatory"), SkyCoord.from_name("NGC 4993"), Time("2017-08-17")):
    ...     extinction(10 * u.micron)
    <Quantity 0.97517859>
    """

    if Ebv is None:
        return SpectralElement(DustExtinctionForSkyCoord())
    else:
        return SpectralElement(DustExtinctionForEbv(Ebv=Ebv))


class DustExtinctionBase(Model):
    n_inputs = 1
    n_outputs = 1
    input_units = {"x": BaseSpectrum._internal_wave_unit}
    return_units = {"y": u.dimensionless_unscaled}
    input_units_equivalencies = {"x": u.spectral()}

    # argh, why!! synphot.BaseSpectrum passes unitless quantities to the underlying model.
    _input_units_allow_dimensionless = True

    @classmethod
    def evaluate(cls, x, Ebv):
        # Handle dimensionless units
        if (
            not hasattr(x, "unit")
            or u.unit is None
            or u.unit is u.dimensionless_unscaled
        ):
            x = x * cls.input_units["x"]

        # Convert to internal units used by dust_extinction package:
        # wavenumber in units of inverse microns.
        x = x.to(1 / u.micron, equivalencies=u.spectral())

        lo, hi = reddening_law.x_range
        good = (x.value >= lo) & (x.value <= hi)
        x[~good] = np.nan
        return reddening_law.extinguish(x, Ebv=Ebv)


class DustExtinctionForEbv(DustExtinctionBase):
    Ebv = Parameter(description="E(B-V)")


class DustExtinctionForSkyCoord(DustExtinctionBase):
    @property
    def Ebv_max(self):
        """Maximum value of E(B-V) for the assumed dust model."""
        return dust_map()._pix_val.max()

    @classmethod
    def evaluate(cls, x):
        return super().evaluate(x, dust_map().query(state.get().target_coord))
