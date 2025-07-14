from functools import cache
from importlib import resources

import astropy.units as u
import numpy as np
from synphot import Empirical1D, SourceSpectrum

from . import data

kpno_sky_tables = {
    "low": "10JunZen.txt",
    "medium": "10FebZen.txt",
    "high": "10Phx.txt",
    "veryhigh": "10Tuc.txt",
    "skycalc": "skycalcbrightness.txt",
}


@cache
def read_kpno_sky_data(key):
    try:
        filename = kpno_sky_tables[key]
    except KeyError:
        raise ValueError("option must be one of {0}".format(kpno_sky_tables.keys()))

    with resources.files(data).joinpath(filename).open("rb") as f:
        x, y = np.loadtxt(f).T

    # According to per private communication with P. Massey, sky brightness is
    # given in units AB magnitudes/arcsec^2
    x *= u.Angstrom
    y *= u.ABmag

    return SourceSpectrum(Empirical1D, points=x, lookup_table=y)


class SkyBackground:
    """
    Sky Brightness background: sky glow due to scattered and diffuse light

    Currently, only data from ESO's SkyCalc (Noll et al. (2012) [2]_ and Jones
    et al. (2013) [3]_) and the Kitt Peak sky brightness observations from
    Neugent and Massey (2010) [1]_ are supported. There are four methods,
    sampling a selection of sky brightness conditions:

    - :meth:`low`: At zenith in June
    - :meth:`medium` : At zenith in February
    - :meth:`high`: At a zenith angle of 60 degrees towards Phoenix
    - :meth:`veryhigh`: At a zenith angle of 60 degrees towards Tucson
    - :meth:`skycalc`: ESO's SkyCalc model for Cerro Paranal

    Notes
    -----
    Spectra are given only for B and V bands (~3750 - ~6875 Angstrom) and do
    not include effect of lunar phase.

    References
    ----------
    .. [1] Neugent, K. and Massey, P., 2010, "The Spectrum of the Night Sky
           Over Kitt Peak: Changes Over Two Decades". PASP, 122, 1246-1253
           doi:10.1086/656425
    .. [2] Noll, S., Kausch, W., Barden, M., Jones, A. C., Szyszka, C., Kimeswenger,
           S., & Vinther, J., 2012, "An atmospheric radiation model for Cerro
           Paranal". A&A, 543, A92. doi:10.1051/0004-6361/201219040
    .. [3] Jones, A., Noll, S., Kausch, W., Szyszka, C., & Kimeswenger, S., 2013,
           "An advanced scattered moonlight model for Cerro Paranal". A&A, 560,
           A91. doi:10.1051/0004-6361/201322433

    Examples
    --------
    >>> from astropy import units as u
    >>> from m4opt.synphot.background import SkyBackground

    >>> background = SkyBackground.low()
    >>> background(5890 * u.angstrom, flux_unit=u.ABmag)
    <Magnitude 20.66945113 mag(AB)>

    >>> background = SkyBackground.veryhigh()
    >>> background(5890 * u.angstrom, flux_unit=u.ABmag)
    <Magnitude 19.09920111 mag(AB)>

    .. plot::
        :caption: Sky background spectra

        from astropy import units as u
        from astropy.visualization import quantity_support
        from matplotlib import pyplot as plt
        import numpy as np

        from m4opt.synphot.background import SkyBackground

        quantity_support()

        wave = np.linspace(3750, 6868) * u.angstrom
        ax = plt.axes()
        for key in ['skycalc', 'veryhigh', 'high', 'medium', 'low']:
            surf = getattr(SkyBackground, key)()(wave, flux_unit=u.ABmag)
            ax.plot(wave, surf, label=f'SkyBackground.{key}')
        ax.legend()
        ax.invert_yaxis()
    """

    @staticmethod
    def low():
        return read_kpno_sky_data("low")

    @staticmethod
    def medium():
        return read_kpno_sky_data("medium")

    @staticmethod
    def high():
        return read_kpno_sky_data("high")

    @staticmethod
    def veryhigh():
        return read_kpno_sky_data("veryhigh")

    @staticmethod
    def skycalc():
        return read_kpno_sky_data("skycalc")
