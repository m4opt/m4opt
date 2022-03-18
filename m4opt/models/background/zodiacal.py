try:
    from functools import cache
except ImportError:  # FIXME: drop once we require Python >= 3.9
    from functools import lru_cache as cache
from importlib import resources

from astropy.coordinates import GeocentricTrueEcliptic, get_sun, SkyCoord
from astropy.modeling import custom_model
from astropy.modeling.models import Const1D, Tabular1D
from astropy.table import QTable
from astropy import units as u
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from . import data
from .core import Background
from ..state import position, time

mag_to_scale = u.mag(1).to_physical
mag_low = 23.3
mag_mid = 22.7
mag_high = 22.1


@cache
def read_stis_zodi_high():
    with resources.path(data, 'stis_zodi_high.ecsv') as path:
        table = QTable.read(path)
    x = table['wavelength'].to(
        Background.input_units['x'], equivalencies=u.spectral())
    y = table['surface_brightness'].to(
        Background.return_units['y'], equivalencies=u.spectral_density(x))
    return np.flipud(x), np.flipud(y)  # reversed, for frequency -> wavelength


@cache
def read_leinert_angular_interp():
    # Zodiacal light angular dependence from Table 16 of
    # Leinert et al. (2017), https://doi.org/10.1051/aas:1998105.
    with resources.path(data, 'leinert_zodi.txt') as p:
        table = np.loadtxt(p)
    lat = table[0, 1:]
    lon = table[1:, 0]
    s10 = table[1:, 1:]

    # The table only extends up to a latitude of 75°. According to the paper,
    # "Towards the ecliptic pole, the brightness as given above is 60 ± 3 S10."
    lat = np.append(lat, 90)
    s10 = np.append(s10, np.tile(60.0, (len(lon), 1)), axis=1)

    # The table is in units of S10: the number of 10th magnitude stars per
    # square degree. Convert to magnitude per square arcsecond.
    sb = 10 - 2.5 * np.log10(s10 / 60**4)
    return RegularGridInterpolator([lon, lat], sb)


def get_scale(position, time):
    interp = read_leinert_angular_interp()
    frame = GeocentricTrueEcliptic(equinox=time)
    obj = SkyCoord(position).transform_to(frame)
    sun = get_sun(time).transform_to(frame)

    # Wrap angles and look up in table
    lat = np.abs(obj.lat.deg)
    lon = np.abs((obj.lon - sun.lon).wrap_at(180 * u.deg).deg)
    mag = interp(np.stack((lon, lat), axis=-1))

    # When interp2d encounters infinities, it returns nan. Fix that up.
    mag = np.where(np.isnan(mag), -np.inf, mag)

    # Fix up shape
    if obj.isscalar:
        mag = mag.item()

    mag -= mag_high
    return mag_to_scale(mag)


@custom_model
def ZodiacalScale(x):
    return get_scale(position.get(), time.get()) * u.dimensionless_unscaled


class ZodiacalBackground:
    """
    Zodiacal light sky background: sunlight scattered by interplanetary dust.

    This is the zodiacal light model that is described in the HST STIS
    Instrument Handbook [1]_. The "high" zodiacal light spectrum is taken from
    `Table 6.4`_ and the "average" and "low" spectra are scaled from it so that
    they have visual surface brightness of 22.1, 22.7, and 23.3 magnitudes per
    square arcsecond.

    The dependence on sky position is taken from Table 16 of [2]_, which is a
    higher-resolution version of `Table 6.2`_ from the HST STIS Instrument
    Handbook.

    .. _`Table 6.2`: https://hst-docs.stsci.edu/stisihb/chapter-6-exposure-time-calculations/6-5-detector-and-sky-backgrounds#id-6.5DetectorandSkyBackgrounds-Table6.2
    .. _`Table 6.4`: https://hst-docs.stsci.edu/stisihb/chapter-6-exposure-time-calculations/6-6-tabular-sky-backgrounds#id-6.6TabularSkyBackgrounds-Table6.4

    References
    ----------
    .. [1] Prichard, L., Welty, D. and Jones, A., et al. 2022 "STIS Instrument
           Handbook," Version 21.0, (Baltimore: STScI)
    .. [2] Leinert, Ch., Bowyer, S., and Haikala, L. K., et al. 1998 "The 1997
           reference of diffuse night sky brightness", Astron. Astrophys.
           Suppl. Ser. 127, 1-99. https://doi.org/10.1051/aas:1998105

    Examples
    --------

    You can specify the zodiacal light background in several different ways.
    You can get a typical background for "low", "average", or "high"
    conditions.

    >>> from astropy import units as u
    >>> from m4opt.models.background import ZodiacalBackground
    >>> background = ZodiacalBackground.low()
    >>> background(3000 * u.angstrom).to(u.mag(u.AB / u.arcsec**2))
    <Magnitude 26.16417045 mag(AB / arcsec2)>
    >>> background = ZodiacalBackground.mid()
    >>> background(3000 * u.angstrom).to(u.mag(u.AB / u.arcsec**2))
    <Magnitude 25.56417045 mag(AB / arcsec2)>
    >>> background = ZodiacalBackground.high()
    >>> background(3000 * u.angstrom).to(u.mag(u.AB / u.arcsec**2))
    <Magnitude 24.96417045 mag(AB / arcsec2)>

    You can get the background for a target at a particular sky position,
    observed at a particular time.

    >>> from astropy.coordinates import SkyCoord
    >>> from astropy.time import Time
    >>> skycoord = SkyCoord.from_name('NGC 4993')
    >>> obstime = Time('2017-08-17T12:41:04.4')
    >>> background = ZodiacalBackground.at(position=skycoord, time=obstime)
    >>> background(3000 * u.angstrom).to(u.mag(u.AB / u.arcsec**2))
    <Magnitude 24.75100719 mag(AB / arcsec2)>

    Lastly, you can leave the position and time free, to be specified at a
    later point.

    >>> background = ZodiacalBackground()
    >>> background(3000 * u.angstrom).to(u.mag(u.AB / u.arcsec**2))
    Traceback (most recent call last):
      ...
    ValueError: Unknown position. Please evaluate the model by providing the \
    target position and observing time in a `with:` statement, like this:
        from m4opt.models import position, time
        with position.set(skycoord), time.set(obstime):
            ...  # evaluate model here

    >>> from m4opt.models import position, time
    >>> with position.set(skycoord), time.set(obstime):
    ...     background(3000 * u.angstrom).to(u.mag(u.AB / u.arcsec**2))
    <Magnitude 24.75100719 mag(AB / arcsec2)>
     """  # noqa: E501

    def __new__(cls):
        return cls.high() * ZodiacalScale()

    @classmethod
    def at(cls, position, time):
        """Get the model for a fixed sky position and time.

        Parameters
        ----------
        position : :class:`astropy.coordinates.SkyCoord`
            The coordinates of the object under observation. If the coordinates
            do not specify a distance, then the object is assumed to be a fixed
            star at infinite distance for the purpose of calculating its
            helioecliptic position.
        time : :class:`astropy.time.Time`
            The time of the observation.
        """
        return cls.high() * Const1D(get_scale(position, time))

    @classmethod
    def low(cls):
        """Zodiacal background for typical "low" background conditions.

        Following the conventions in the HST STIS manual, this is
        1.2 mag / arcsec2 fainter than the "high" model at all frequencies.
        """
        return cls.high() * Const1D(mag_to_scale(mag_low - mag_high))

    @classmethod
    def mid(cls):
        """Zodiacal background for "average" background conditions.

        Following the conventions in the HST STIS manual, this is
        0.6 mag / arcsec2 fainter than the "high" model at all frequencies.
        """
        return cls.high() * Const1D(mag_to_scale(mag_mid - mag_high))

    @staticmethod
    def high():
        """Zodiacal background for typical "high" background conditions."""
        result = Tabular1D(*read_stis_zodi_high())
        result.input_units_equivalencies = Background.input_units_equivalencies
        return result
