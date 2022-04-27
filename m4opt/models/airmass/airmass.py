# try:
#    from functools import cache
# except ImportError:  # FIXME: drop once we require Python >= 3.9
#    from functools import lru_cache as cache

# from importlib import resources

# from . import data

import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
# from astropy.time import Time


def simple_airmass(z):
    """
    Simple airmass calculation assuming plane-parallel atmosphere.
    Not recommended for zenith angles (z) > 75 degreees.

    Input:
    z : zenith angle (radians)
    """
    return 1./np.cos(z)


def KastenYoung_airmass(z):
    """
    From Kasten, F.; Young, A. T. (1989).
    "Revised optical air mass tables and approximation formula".
    Applied Optics. 28 (22): 4735–4738.
    doi:10.1364/AO.28.004735. PMID 20555942.

    Original formula given in altitude, here is rewritten
    in terms of zenith angle (z)

    Input:
    z : zenith angle (radians)

    """
    return 1./(np.cos(z) + 0.50572 * (96.07995 - np.rad2deg(z))**(-1.6364))


airmass_models = {'simple': simple_airmass, 'kastenyoung': KastenYoung_airmass}


class Airmass:
    """
    Airmass:

    Object to calculate airmass at a given observatory location,
    for a provided target SkyCoord.

    Airmass

    Examples
    --------
    The Airmass object requires a ground location at initialization. You can
    pass in your own location via `astropy.coordinates.EarthLocation`:

    >>> from m4opt.models.airmass import Airmass
    >>> import astropy.units as u
    >>> from astropy.coordinates import SkyCoord, EarthLocation
    >>> from astropy.time import Time

    >>> place_airm = Airmass(EarthLocation(lat=41.3*u.deg, lon=-74*u.deg,
                             height=390*u.m))

    or choose a pre-defined location, e.g. Kitt Peak:

    >>> kpno_airm = Airmass.kpno()

    To evaluate an airmass for an observing target, we pass in the SkyCoord
    and the Time for the target, using `at()`:

    >>> time = Time('2012-7-13 03:00:00')
    >>> m33 = SkyCoord.from_name('m33')
    >>> place_airm.at(m33, time)
    442.5785661289659

    The default airmass model is a plane-parallel atmosphere, which defines
    the airmass as the secant of the zenith angle. We can choose a different
    model, if we wish to use more realistic values near the horizon:

    >>> place_airm.set_model("kastenyoung")
    >>> place_airm.at(m33,time)
    36.05249806866009

    """

    def __init__(self, earth_location):
        if not isinstance(earth_location, EarthLocation):
            raise TypeError("Input earth_location must be of type \
                             astropy.coordinates.earth.EarthLocation")

        self.earth_loc = earth_location
        self.model = simple_airmass
        return

    def set_model(self, model_name):
        if model_name.lower() not in airmass_models.keys():
            raise AttributeError("model must be one of {0}".format(
                                 airmass_models.keys()))
        self.model = airmass_models[model_name.lower()]

    def at(self, target_coord, obs_time):
        """
        returns airmass at target sky location at obs_time
        """
        if not isinstance(target_coord, SkyCoord):
            raise TypeError("argument target_coord must be of type \
                            astropy.coordinates.sky_coordinate.SkyCoord")

        frame = AltAz(obstime=obs_time, location=self.earth_loc)
        tf_target = target_coord.transform_to(frame)
        return self.model(tf_target.zen.radian)

    @classmethod
    def kpno(cls):
        """
        defines Airmass Object using location of Kitt Peak National Observatory

        Last updated via EarthLocation.of_site("Kitt Peak") on 2022/04/27
        """
        return cls(EarthLocation(-1994502.60430614, -5037538.54232911,
                                 3358104.99690298, unit='m'))

# class Extinction:
    """
    Extinction:

    Model to calculate extinction due to airmass.
    """
