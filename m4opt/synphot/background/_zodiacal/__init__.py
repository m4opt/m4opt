from importlib import resources

import numpy as np
from astropy import units as u
from astropy.coordinates import GeocentricTrueEcliptic, SkyCoord, get_sun
from astropy.table import QTable
from scipy.interpolate import RegularGridInterpolator
from synphot import Empirical1D, SourceSpectrum, SpectralElement

from ....utils.typing_extensions import override
from ..._extrinsic import ExtrinsicScaleFactor
from .._core import BACKGROUND_SOLID_ANGLE
from . import data

mag_to_scale = u.mag(1).to_physical
mag_low = 23.3
mag_mid = 22.7
mag_high = 22.1


class ZodiacalBackgroundScaleFactor(ExtrinsicScaleFactor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Zodiacal light angular dependence from Table 16 of
        # Leinert et al. (2017), https://doi.org/10.1051/aas:1998105.
        with resources.files(data).joinpath("leinert_zodi.txt").open("rb") as f:
            table = np.loadtxt(f)
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
        self._interp = RegularGridInterpolator([lon, lat], sb)

    @override
    def at(self, observer_location, target_coord, obstime):
        frame = GeocentricTrueEcliptic(equinox=obstime)
        obj = SkyCoord(target_coord).transform_to(frame)
        sun = get_sun(obstime).transform_to(frame)

        # Wrap angles and look up in table
        lat = np.abs(obj.lat.deg)
        lon = np.abs((obj.lon - sun.lon).wrap_at(180 * u.deg).deg)
        mag = self._interp(np.stack((lon, lat), axis=-1))

        # When interp2d encounters infinities, it returns nan. Fix that up.
        mag = np.where(np.isnan(mag), -np.inf, mag)

        # Fix up shape
        if obj.isscalar:
            mag = mag.item()

        mag -= mag_high
        return mag_to_scale(mag)


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

    Warnings
    --------
    This model should only be used for observers near Earth --- in Earth orbit,
    as Hubble is, or on the Earth, or even on the Moon or in cislunar space. It
    should NOT be used for observers in orbits around other planets, or in
    distant solar orbits, or at Earth-Sun Lagrange points.

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
    >>> from m4opt.synphot.background import ZodiacalBackground
    >>> background = ZodiacalBackground.low()
    >>> background(3000 * u.angstrom, flux_unit=u.ABmag)
    <Magnitude 26.16417045 mag(AB)>
    >>> background = ZodiacalBackground.mid()
    >>> background(3000 * u.angstrom, flux_unit=u.ABmag)
    <Magnitude 25.56417045 mag(AB)>
    >>> background = ZodiacalBackground.high()
    >>> background(3000 * u.angstrom, flux_unit=u.ABmag)
    <Magnitude 24.96417045 mag(AB)>

    You can get the background for a target at a particular sky position,
    observed at a particular time.

    >>> background = ZodiacalBackground()
    >>> background(3000 * u.angstrom, flux_unit=u.ABmag)
    Traceback (most recent call last):
      ...
    ValueError: Unknown target. Please evaluate the model by providing the \
    position and observing time in a `with:` statement, like this:
        from m4opt.synphot import observing
        with observing(observer_location=loc, target_coord=coord, obstime=time):

    >>> from astropy.coordinates import EarthLocation, SkyCoord
    >>> from astropy.time import Time
    >>> loc = EarthLocation.from_geocentric(0 * u.m, 0 * u.m, 0 * u.m)
    >>> coord = SkyCoord.from_name('NGC 4993')
    >>> time = Time('2017-08-17T12:41:04.4')
    >>> from m4opt.synphot import observing
    >>> with observing(observer_location=loc, target_coord=coord, obstime=time):
    ...     background(3000 * u.angstrom, flux_unit=u.ABmag)
    <Magnitude 24.75101362 mag(AB)>

    .. plot::
        :caption: Mid, low, and high zodiacal light spectra

        from matplotlib import pyplot as plt
        import numpy as np
        from astropy import units as u
        from astropy.visualization import quantity_support

        from m4opt.synphot.background import ZodiacalBackground

        quantity_support()

        wave = np.linspace(1000, 11000) * u.angstrom
        ax = plt.axes()
        for key in ['high', 'mid', 'low']:
            surf = getattr(ZodiacalBackground, key)()(wave, flux_unit=u.ABmag)
            ax.plot(wave, surf, label=f'ZodiacalBackground.{key}()')
        ax.invert_yaxis()
        ax.legend()

    .. plot::
        :caption: Zodiacal light at 10000 Å observed at 2023-06-30T00:00:00

        from astropy.coordinates import EarthLocation, get_body, ICRS
        from astropy.time import Time
        from astropy import units as u
        from astropy_healpix import HEALPix
        from matplotlib import pyplot as plt
        from matplotlib.colors import LogNorm
        import numpy as np
        import ligo.skymap.plot

        from m4opt.synphot.background import ZodiacalBackground
        from m4opt.synphot import observing

        wave = 10000 * u.angstrom
        hpx = HEALPix(nside=512, frame=ICRS())
        loc = EarthLocation.from_geocentric(0 * u.m, 0 * u.m, 0 * u.m)
        coord = hpx.healpix_to_skycoord(np.arange(hpx.npix))
        obstime = Time('2023-06-30T00:00:00')
        zodi = ZodiacalBackground()
        with observing(observer_location=loc, target_coord=coord, obstime=obstime):
            surf = zodi(wave, flux_unit=u.ABmag)

        fig = plt.figure()
        ax = fig.add_subplot(projection='astro hours mollweide')
        im = ax.imshow_hpx(surf.value, cmap='viridis')
        fig.colorbar(im, extend='both', orientation='horizontal').set_label(surf.unit)

        sun = get_body('sun', obstime)
        transform = ax.get_transform('world')
        ax.plot(
            sun.ra, sun.dec, 'or', ms=1,
            transform=transform)
        ax.plot(
            sun.ra, sun.dec, 'or', mfc='none',
            transform=transform)
        ax.text(sun.ra, sun.dec, '  Sun', color='red', transform=transform)

        ax.grid()

     """  # noqa: E501

    def __new__(cls):
        return cls.high() * SpectralElement(ZodiacalBackgroundScaleFactor())

    @classmethod
    def low(cls):
        """Zodiacal background for typical "low" background conditions.

        Following the conventions in the HST STIS manual, this is
        1.2 mag / arcsec2 fainter than the "high" model at all frequencies.
        """
        return cls.high() * mag_to_scale(mag_low - mag_high)

    @classmethod
    def mid(cls):
        """Zodiacal background for "average" background conditions.

        Following the conventions in the HST STIS manual, this is
        0.6 mag / arcsec2 fainter than the "high" model at all frequencies.
        """
        return cls.high() * mag_to_scale(mag_mid - mag_high)

    @staticmethod
    def high():
        """Zodiacal background for typical "high" background conditions."""
        with resources.files(data).joinpath("stis_zodi_high.ecsv").open("rb") as f:
            table = QTable.read(f, format="ascii.ecsv")
        return SourceSpectrum(
            Empirical1D,
            points=table["wavelength"],
            lookup_table=table["surface_brightness"] * BACKGROUND_SOLID_ANGLE,
        )
