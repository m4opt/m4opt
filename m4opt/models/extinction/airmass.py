from ..core import state
from .core import BaseExtinction

try:
    from functools import cache
except ImportError:  # FIXME: drop once we require Python >= 3.9
    from functools import lru_cache as cache

import numpy as np
import astropy.units as u
from astropy.coordinates import EarthLocation, AltAz
from astropy.modeling import custom_model
from astropy.modeling.models import Const1D, Tabular1D
from astropy.table import QTable
from astropy.utils.data import download_file


path_to_extin = ('https://raw.githubusercontent.com/astropy/'
                 'specreduce-data/main/specreduce_data/'
                 'reference_data/extinction/')

__all__ = ('Airmass', 'AtmosphericExtinction',
           'PlaneParallelAirmass', 'KastenYoungAirmass')


@cache
def read_extinction_table(placename):
    """
    extinction tables usually have
    x in angstroms
    y in mag per airmass
    """
    placename = placename.lower()
    try:
        tablename = AtmosphericExtinction.available_tables[placename]
    except KeyError:
        raise ValueError(f"Invalid placename {placename}. "
                         "Must be one of "
                         f"{AtmosphericExtinction.available_tables.keys()}")

    table = QTable.read(download_file(path_to_extin+tablename, cache=True),
                        format='ascii', names=('wavelength', 'extinction'))

    x = (table['wavelength']*u.Angstrom).to(
        BaseExtinction.input_units['x'], equivalencies=u.spectral())
    y = table['extinction']
    return np.flipud(x), np.flipud(y)


class BaseAirmass:
    """
    Object to calculate airmass at a given observatory location,
    for a provided target SkyCoord.

    The BaseAirmass class provides logic for initializing from observatory
    position, but derived classes must implement the airmass model calculation
    via defining the 'calc_airmass()' function.
    """

    def __init__(self, earth_location):
        if not isinstance(earth_location, EarthLocation):
            raise TypeError("Input earth_location must be of type"
                            "astropy.coordinates.earth.EarthLocation")

        self.earth_loc = earth_location

    @staticmethod
    def calc_airmass(zen):
        """
        Airmass Model must take in the zenith angle for airmass calculations.

        Input:
        zen : zenith angle `astropy.units.Quantity`
        """
        pass

    def at(self, target_coord, obs_time):
        """
        returns airmass at target sky location at obs_time
        """
        if not hasattr(target_coord, 'transform_to'):
            raise TypeError("argument target_coord must be an astropy "
                            "coordinates object")

        frame = AltAz(obstime=obs_time, location=self.earth_loc)
        tf_target = target_coord.transform_to(frame)
        return self.calc_airmass(tf_target.zen)


class PlaneParallelAirmass(BaseAirmass):
    """

    Plane-Parallel atmosphere, airmass = 1/cos(zenith).

    Notes
    -----
    Not recommended for zenith angles > 75 degrees. See `KastenYoungAirmass`
    for alternative model implementation.

    Examples
    --------
    This object requires a ground observer location at initialization,
    passed in via `astropy.coordinates.EarthLocation`:

    >>> from m4opt.models.extinction import PlaneParallelAirmass
    >>> import astropy.units as u
    >>> from astropy.coordinates import SkyCoord, EarthLocation
    >>> from astropy.time import Time

    >>> place_airm = PlaneParallelAirmass(EarthLocation(lat=41.3*u.deg,\
                                          lon=-74*u.deg,\
                                          height=390*u.m))

    or choose a pre-defined location, e.g. Kitt Peak:

    >>> kpno_airm = PlaneParallelAirmass(EarthLocation.of_site('Kitt Peak'))

    To evaluate an airmass for an observing target, we pass in the SkyCoord
    and the Time for the target, using `at()`:

    >>> time = Time('2012-7-13 07:00:00')
    >>> target = SkyCoord.from_name('m33')
    >>> place_airm.at(target, time)
    <Quantity 1.53952747>
    """

    @staticmethod
    def calc_airmass(zen):
        """
        Simple airmass calculation assuming plane-parallel atmosphere.

        Input:
        zen : zenith angle `astropy.units.Quantity`
        """
        return 1./np.cos(zen)


class KastenYoungAirmass(BaseAirmass):
    """
    Kasten+Young89 interpolation fit to airmass table data[1]_ .

    References
    ----------
    .. [1] Kasten, F.; Young, A. T. (1989). "Revised optical air mass
           tables and approximation formula". Applied Optics. 28 (22):
           4735-4738. doi:10.1364/AO.28.004735. PMID 20555942.

    Examples
    --------
    This object requires a ground observer location at initialization,
    passed in via `astropy.coordinates.EarthLocation`:

    >>> from m4opt.models.extinction import KastenYoungAirmass
    >>> import astropy.units as u
    >>> from astropy.coordinates import SkyCoord, EarthLocation
    >>> from astropy.time import Time

    >>> place_airm = KastenYoungAirmass(EarthLocation(lat=41.3*u.deg,\
                                          lon=-74*u.deg,\
                                          height=390*u.m))

    or choose a pre-defined location, e.g. Kitt Peak:

    >>> kpno_airm = KastenYoungAirmass(EarthLocation.of_site('Kitt Peak'))

    To evaluate an airmass for an observing target, we pass in the SkyCoord
    and the Time for the target, using `at()`:

    >>> target = SkyCoord.from_name('m33')
    >>> time = Time('2012-7-13 03:00:00')
    >>> place_airm.at(target,time)
    <Quantity 36.05249807>
    """

    @staticmethod
    def calc_airmass(zen):
        """
        From Kasten, F.; Young, A. T. (1989).
        "Revised optical air mass tables and approximation formula".
        Applied Optics. 28 (22): 4735-4738.
        doi:10.1364/AO.28.004735. PMID 20555942.

        Original formula given in altitude, here is rewritten
        in terms of zenith angle

        Input:
        zen : zenith angle `astropy.units.Quantity`
        """

        return 1./(np.cos(zen) + 0.50572 * (96.07995 -
                   zen.to(u.deg).value)**(-1.6364))


# default Airmass class
Airmass = PlaneParallelAirmass


def KnownAirmassState(airmass):
    """Returns airmass state with given airmass
    """

    @custom_model
    def AState(x):
        observing_state = state.get()
        return airmass.at(observing_state.target_coord,
                          observing_state.obstime) * u.dimensionless_unscaled

    return AState()


def UnknownAirmassState(airmass_class=None):

    if airmass_class is None:
        airmass_class = Airmass

    @custom_model
    def UAState(x):
        """Returns airmass state with unknown observatory location, which
        will need to be set as part of the ObservingState
        """
        observing_state = state.get()
        airmass = airmass_class(observing_state.observatory_loc)
        return airmass.at(observing_state.target_coord,
                          observing_state.obstime) * u.dimensionless_unscaled

    return UAState()


class AtmosphericExtinction:
    """
    Attentuation of spectrum due to atmospheric effects.

    Atmospheric effects are calculated via an airmass extinction table,
    and so require an appropriate Airmass instance (or a defined observatory
    location) for the extinction calculation.

    Notes
    -----

    Default extinction table is Kitt Peak.

    Examples
    --------

    >>> from m4opt.models.extinction import AtmosphericExtinction, Airmass
    >>> import astropy.units as u
    >>> from astropy.coordinates import SkyCoord, EarthLocation
    >>> from astropy.time import Time
    >>> time = Time('2012-7-13 07:00:00')
    >>> target = SkyCoord.from_name('m33')

    AtmosphericExtinction models calculate the extinction of a target spectrum
    due to the atmosphere. In addition to target location and observing time,
    this requires an observatory location in order to calculate the airmass via
    the local zenith angle of the target. We can do this in several different
    ways. First, we can simply pass all of the required information using
    `at()`:

    >>> place = EarthLocation(lat=41.3*u.deg, lon=-74*u.deg, height=390*u.m)
    >>> airmass = Airmass(place)
    >>> extn = AtmosphericExtinction.at(airmass, target, time)
    >>> extn(3200*u.Angstrom)
    <Quantity 0.23643961>

    Alternatively, we can define the object for a given observer airmass:

    >>> extn = AtmosphericExtinction(airmass)

    However, this requires the use of `state` to fill in the
    target information:

    >>> from m4opt.models import state
    >>> with state.set_observing(target_coord=target, obstime=time):
    ...     print(extn(3200*u.Angstrom))
    0.23643960524293295

    If we do not have a known observatory location, we can initialize a blank
    state with `generic_observer()`:

    >>> extn = AtmosphericExtinction.generic_observer()
    >>> with state.set_observing(target_coord=target, obstime=time, \
                                 observatory_loc=place):
    ...     print(extn(3200*u.Angstrom))
    0.23643960524293295

    You can also define the airmass model for this blank state:

    >>> extn = AtmosphericExtinction.generic_observer(airmass_class=\
                                                      KastenYoungAirmass)
    >>> with state.set_observing(target_coord=target, obstime=time, \
                                 observatory_loc=place):
    ...     print(extn(3200*u.Angstrom))
    0.2369337657824151

    All AtmosphericExtinction models require an extinction table. The
    models above have used the default table; however, other tables
    can be chosen at initialization by passing the appropriate parameter:

    >>> AtmosphericExtinction.available_tables
    {'apo': 'apoextinct.dat',\
 'kpno': 'kpnoextinct.dat',\
 'ctio': 'ctioextinct.dat',\
 'lapalm': 'lapalmextinct.dat',\
 'mko': 'mkoextinct.dat',\
 'lick': 'mthamextinct.dat',\
 'paranal': 'paranalextinct.dat'}

    >>> extn = AtmosphericExtinction.at(airmass, target, time, \
        table_name='kpno')
    >>> extn(3200*u.Angstrom)
    <Quantity 0.23643961>

    >>> extn = AtmosphericExtinction.generic_observer(table_name='apo')
    >>> with state.set_observing(target_coord=target, obstime=time, \
                                 observatory_loc=place):
    ...     print(extn(3200*u.Angstrom))
    0.2651060248681333

    Finally, we can pass in the interpolation method for the extinction table
    if finer control is needed. The default is 'linear', but other options can
    be selected (see `astropy.modeling.tabular.Tabular1D` for details).

    >>> extn = AtmosphericExtinction.generic_observer(table_name='apo', \
                                                      table_method='nearest')
    >>> with state.set_observing(target_coord=target, obstime=time, \
                                 observatory_loc=place):
    ...     print(extn(3200*u.Angstrom))
    0.2727180718446889
    """

    def __new__(cls, airmass, table_name=None, table_method='linear'):
        airmass_state = KnownAirmassState(airmass)
        return Const1D(10.)**(
                              Const1D(-0.4)*airmass_state
                              * cls.table(table_name, table_method)
                              )

    @classmethod
    def generic_observer(cls, table_name=None, table_method="linear",
                         airmass_class=Airmass):
        return Const1D(10.)**(
                              Const1D(-0.4)
                              * UnknownAirmassState(airmass_class)
                              * cls.table(table_name, table_method)
                             )

    @classmethod
    def at(cls, airmass, target_coord, obstime, table_name=None,
           table_method='linear'):
        return Const1D(10.)**(
                              Const1D(-0.4*airmass.at(target_coord, obstime))
                              * cls.table(table_name, table_method)
                              )

    @classmethod
    def table(cls, table_name, method='linear'):
        if table_name is None:
            return cls.default_table(method)
        else:
            return cls.extinction_table(table_name, method)

    @classmethod
    def default_table(cls, method="linear"):
        """Default Airmass Extinction Table"""
        return cls.extinction_table("kpno", method)

    @classmethod
    def extinction_table(cls, table_name, method="linear"):

        result = Tabular1D(*read_extinction_table(table_name),
                           method=method)
        result.input_units_equivalencies = (BaseExtinction.
                                            input_units_equivalencies)
        return result

    available_tables = {"apo": 'apoextinct.dat', 'kpno': 'kpnoextinct.dat',
                        'ctio': 'ctioextinct.dat',
                        'lapalm': 'lapalmextinct.dat', 'mko': 'mkoextinct.dat',
                        'lick': 'mthamextinct.dat',
                        'paranal': 'paranalextinct.dat'}
