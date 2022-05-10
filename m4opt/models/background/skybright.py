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
    Data taken from Neugent & Massey(2010)[1]_; sky brightness is originally in
    units magnitudes/arcsec^2. According to the paper, the magnitudes were
    calculated such that Vega is 0.03 mag in both B and V bands. Thus, this
    is assumed to be Johnson magnitudes (based on Vega; though it is not
    explicitly stated). The output data is returned with flux units
    (i.e. ergs / cm^2 / s / Hz / sr and spectral equivalences).

    Spectra only for B and V bands (~3750 - ~6875 Angstrom).

    Options include:
    - "low"    : As measured at Zenith in June; lowest level of sky brightness
    - "medium" : At Zenith in February
    - "high"   : As measured at a zenith distance of 60 degrees towards Phoenix
    - "veryhigh" : Measured at a zenith distance of 60 degrees towards Tucson

    Notes
    -----
    Unfortunately, NM10 is unclear as to what their measured Vega fluxes
    and the resulting zero point values are. Table A2 of Bessel+98[2]_ implies
    zero_B = -0.602 and zero_V = 0.00 such that mag_B = -2.5*log10(F_L)
    - 21.1 - zero_B, (and similar for the V band) with F_L given in
    units ergs / cm^2 / sec / Angstrom.

    However, NM10[1]_ also cite Turnrose (1974)[3_] and state that "his λ4540
    flux is equivalent to 22.30 mag/arcsec^2". From Turnrose's Table 2, the
    λ4540 flux is 6.38 * 10^-18 ergs / cm^2 / sec / Angstrom / arcsec^2; this
    implies zero_B = -0.42 to match the stated 22.3 mag / arcsec^2. Turnrose
    also states, for the V band, that F_L = 3.64 * 10^-9 ergs / cm^2 / sec
    / Angstrom for magnitude 0; this corresponds to zero_V = 0.00.

    Thus we use zero_V = 0.00 and zero_B = -0.42.

    Since the B and V bands overlap, we define the appropriate zero_pt for
    an chosen overlapping range using a weighted average to soften the
    transition from one zero point to the next. We follow NM10 and use
    Bessel (1990)[4]_ to define the bands: this leads to a midpoint of 4965
    Angstroms where the B and V system response is roughly equal. We will use
    a 200 Angstrom transition period, such that

    zero_pt(4865 < λ < 5065) = zero_B + (λ - 4865) * (zero_V - zero_B) / (200)

    References
    ----------
    .. [1] Neugent, K. and Massey, P., 2010, "The Spectrum of the Night Sky
           Over Kitt Peak: Changes Over Two Decades". PASP, 122, 1246-1253
           doi:10.1086/656425
    .. [2] Bessel, M.S., Castelli, F., and B. Plez, 1998, "Model atmospheres
           broad-band colors, bolometric corrections and temperature
           calibrations for O-M stars*", AA, 333, 231-250
    .. [3] Turnrose, B., 1974, "Absolute Spectral Energy Distribution of
           the Night Sky at Palomar and Mount Wilson Observatories", PASP,
           86, 512, doi:10.1086/129642
    .. [4] Bessel, M.S., 1990, "UBVRI Passbands", PASP, 102, 1181-1199,
           doi:10.1086/132749
    """

    if option.lower() not in kpno_sky_tables.keys():
        raise AttributeError("option must be one of {0}".format(
                             kpno_sky_tables.keys()))

    with resources.path(data, kpno_sky_tables[option]) as path:
        table = QTable.read(path, format='ascii',
                            names=('wavelength', 'surface_brightness'))

    x = (table['wavelength']*u.Angstrom).to(
        Background.input_units['x'], equivalencies=u.spectral())

    x = table['wavelength']
    y = table['surface_brightness']

    # Converting from Vega mag/arcsec^2
    # to flux ergs / s / cm^2 / Angstrom / arcsec^2
    # From Bessel+1998, Bessel 1990;
    # defining overlapping point as 4965 Angstrom; see Notes above
    zero_B = -0.42
    zero_V = 0.
    y[x > 5065] = 10**(-0.4*(y[x > 5065] + 21.1 + zero_V))
    y[x <= 4865] = 10**(-0.4*(y[x <= 4865] + 21.1 + zero_B))

    zero_trans = zero_B + (x[(x > 4865) & (x <= 5065)] - 4865) * (
                           zero_V - zero_B) / (200)
    y[(x > 4865) & (x <= 5065)] = 10**(-0.4*(y[(x > 4865) & (x <= 5065)]
                                       + 21.1 + zero_trans))

    # convert to desired units
    x = (x*u.Angstrom).to(
        Background.input_units['x'], equivalencies=u.spectral())

    y = (y * u.erg / u.cm**2 / u.s / u.Angstrom / u.arcsec**2).to(
        Background.return_units['y'], equivalencies=u.spectral_density(x))

    return np.flipud(x), np.flipud(y)


class SkyBrightness:
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
    There are four options for getting a SkyBrightness model: 'low', 'medium',
    'high', and 'veryhigh':

    >>> from astropy import units as u
    >>> from m4opt.models.background import SkyBrightness

    >>> background = SkyBrightness.low()
    >>> background(5890 * u.angstrom).to(u.mag(u.AB / u.arcsec**2))
    <Magnitude 20.51092355 mag(AB / arcsec2)>

    >>> background = SkyBrightness.veryhigh()
    >>> background(5890 * u.angstrom).to(u.mag(u.AB / u.arcsec**2))
    <Magnitude 18.94067232 mag(AB / arcsec2)>
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
