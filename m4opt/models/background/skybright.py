try:
    from functools import cache
except ImportError:  # FIXME: drop once we require Python >= 3.9
    from functools import lru_cache as cache

from importlib import resources
import numpy as np
from astropy.table import QTable
import astropy.units as u
from astropy.modeling.models import Tabular1D


from . import data
from .core import Background


kpno_sky_tables = {'low': '10JunZen.txt', 'medium': '10FebZen.txt',
                   'high': '10Phx.txt', 'veryhigh': '10Tuc.txt'}


@cache
def read_kpno_sky_data(option="medium"):
    """
    Data taken from Neugent & Massey(2010)[1]_; as per private communication
    with P. Massey, sky brightness is given in units AB magnitudes/arcsec^2

    Options include:
    - "low"    : As measured at Zenith in June; lowest level of sky brightness
    - "medium" : At Zenith in February
    - "high"   : As measured at a zenith distance of 60 degrees towards Phoenix
    - "veryhigh" : Measured at a zenith distance of 60 degrees towards Tucson

    Notes
    -----

    Spectra is given only for B and V bands (~3750 - ~6875 Angstrom). This does
    not yet include effect of lunar phase.

    References
    ----------
    .. [1] Neugent, K. and Massey, P., 2010, "The Spectrum of the Night Sky
           Over Kitt Peak: Changes Over Two Decades". PASP, 122, 1246-1253
           doi:10.1086/656425
    """

    key = option.lower()
    try:
        filename = kpno_sky_tables[key]
    except KeyError:
        raise ValueError("option must be one of {0}".format(
                             kpno_sky_tables.keys()))

    with resources.path(data, filename) as path:
        table = QTable.read(path, format='ascii',
                            names=('wavelength', 'surface_brightness'))

    x = table['wavelength'] * u.Angstrom
    y = table['surface_brightness'] * u.mag(u.AB/u.arcsec**2)

    # convert to desired units
    x = x.to(Background.input_units['x'], equivalencies=u.spectral())
    y = y.to(Background.return_units['y'], equivalencies=u.spectral_density(x))

    return np.flipud(x), np.flipud(y)


class SkyBackground:
    """
    Sky Brightness background: sky glow due to scattered and diffuse light

    Currently, only the Kitt Peak sky brightness observations from Neugent and
    Massey (2010)[1]_ is supported. See `read_kpno_sky_data()` for discussion.

    References
    ----------
    .. [1] Neugent, K. and Massey, P., 2010, "The Spectrum of the Night Sky
           Over Kitt Peak: Changes Over Two Decades". PASP, 122, 1246-1253
           doi:10.1086/656425

    Examples
    --------
    There are four options for getting a SkyBackground model: 'low', 'medium',
    'high', and 'veryhigh':

    >>> from astropy import units as u
    >>> from m4opt.models.background import SkyBackground

    >>> background = SkyBackground.low()
    >>> background(5890 * u.angstrom).to(u.mag(u.AB / u.arcsec**2))
    <Magnitude 20.66945113 mag(AB / arcsec2)>

    >>> background = SkyBackground.veryhigh()
    >>> background(5890 * u.angstrom).to(u.mag(u.AB / u.arcsec**2))
    <Magnitude 19.09920111 mag(AB / arcsec2)>
    """

    @staticmethod
    def low():
        result = Tabular1D(*read_kpno_sky_data("low"))
        result.input_units_equivalencies = Background.input_units_equivalencies
        return result

    @staticmethod
    def medium():
        result = Tabular1D(*read_kpno_sky_data("medium"))
        result.input_units_equivalencies = Background.input_units_equivalencies
        return result

    @staticmethod
    def high():
        result = Tabular1D(*read_kpno_sky_data("high"))
        result.input_units_equivalencies = Background.input_units_equivalencies
        return result

    @staticmethod
    def veryhigh():
        result = Tabular1D(*read_kpno_sky_data("veryhigh"))
        result.input_units_equivalencies = Background.input_units_equivalencies
        return result
