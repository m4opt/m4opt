from ..core import state
from .core import BaseExtinction

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.modeling import custom_model
from astropy.modeling.models import Const1D, Tabular1D
from astropy.table import QTable
from astropy.utils.data import download_file

path_to_extin = ('https://raw.githubusercontent.com/astropy/' +
                 'specreduce-data/main/specreduce_data/' +
                 'reference_data/extinction/')


def read_extinction_table(placename):
    """
    extinction tables usually have
    x in angstroms
    y in mag per airmass

    """
    try:
        tablename = AtmoExtinction.available_tables[placename]
    except KeyError:
        raise ValueError("Invalid placename {0}. Must be one of {1}".format(
                        placename, AtmoExtinction.available_tables.keys()))

    table = QTable.read(download_file(path_to_extin+tablename, cache=True),
                        format='ascii', names=('wavelength', 'extinction'))

    x = (table['wavelength']*u.Angstrom).to(
        BaseExtinction.input_units['x'], equivalencies=u.spectral())
    y = table['extinction']
    return np.flipud(x), np.flipud(y)


def simple_airmass(z):
    """
    Simple airmass calculation assuming plane-parallel atmosphere.
    Not recommended for zenith angles (z) > 75 degreees.

    Input:
    z : zenith angle (`astropy.units.Quantity`)
    """
    return 1./np.cos(z)


def KastenYoung_airmass(z):
    """
    From Kasten, F.; Young, A. T. (1989).
    "Revised optical air mass tables and approximation formula".
    Applied Optics. 28 (22): 4735-4738.
    doi:10.1364/AO.28.004735. PMID 20555942.

    Original formula given in altitude, here is rewritten
    in terms of zenith angle (z)

    Input:
    z : zenith angle (`astropy.units.Quantity`)

    """
    return 1./(np.cos(z) + 0.50572 * (96.07995 - z.to(u.deg).value)**(-1.6364))


airmass_models = {'simple': simple_airmass, 'kastenyoung': KastenYoung_airmass}


class Airmass:
    """
    Airmass:

    Object to calculate airmass at a given observatory location,
    for a provided target SkyCoord.

    Airmass class currently supports two models:
    1. 'simple' : Plane-Parallel atmosphere, airmass = 1/cos(zenith)
    2. 'kastenyoung' : Kasten+Young89 interpolation fit to data[1]_

    References
    ----------
    .. [1] Kasten, F.; Young, A. T. (1989). "Revised optical air mass
           tables and approximation formula". Applied Optics. 28 (22):
           4735-4738. doi:10.1364/AO.28.004735. PMID 20555942.

    Examples
    --------
    The Airmass object requires a ground observer location at initialization,
    passed in via `astropy.coordinates.EarthLocation`:

    >>> from m4opt.models.extinction import Airmass
    >>> import astropy.units as u
    >>> from astropy.coordinates import SkyCoord, EarthLocation
    >>> from astropy.time import Time

    >>> place_airm = Airmass(EarthLocation(lat=41.3*u.deg, lon=-74*u.deg,\
                             height=390*u.m))

    or choose a pre-defined location, e.g. Kitt Peak:

    >>> kpno_airm = Airmass.kpno()

    To evaluate an airmass for an observing target, we pass in the SkyCoord
    and the Time for the target, using `at()`:

    >>> time = Time('2012-7-13 07:00:00')
    >>> target = SkyCoord.from_name('m33')
    >>> place_airm.at(target, time)
    <Quantity 1.53952747>

    The default airmass model is a plane-parallel atmosphere, which defines
    the airmass as the secant of the zenith angle. We can choose a different
    model if we wish to use more realistic values, especially near
    the horizon:

    >>> time = Time('2012-7-13 03:00:00')
    >>> place_airm.at(target, time)
    <Quantity 442.57856613>
    >>> place_airm.set_model("kastenyoung")
    >>> place_airm.at(target,time)
    <Quantity 36.05249807>

    """

    def __init__(self, earth_location, airmass_model=None):
        if not isinstance(earth_location, EarthLocation):
            raise TypeError("Input earth_location must be of type \
                             astropy.coordinates.earth.EarthLocation")

        self.earth_loc = earth_location
        self.set_model(airmass_model)
        return

    def set_model(self, model_name):
        if model_name is None:
            self.model = simple_airmass
        elif model_name.lower() not in airmass_models.keys():
            raise AttributeError("model must be one of {0}".format(
                                 airmass_models.keys()))
        else:
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
        return self.model(tf_target.zen)

    @classmethod
    def kpno(cls):
        """
        defines Airmass Object using location of Kitt Peak National Observatory

        Last updated via EarthLocation.of_site("Kitt Peak") on 2022/04/27
        """
        return cls(EarthLocation(-1994502.60430614, -5037538.54232911,
                                 3358104.99690298, unit='m'))


def airmass_func(observatory_loc):
    """
    Returns function to calculate airmass from target properties
    """
    airmass = Airmass(observatory_loc)

    def airmass_calc(coord, obstime):
        return airmass.at(coord, obstime)

    return airmass_calc


def KnownAirmassState(airmass):
    """Returns airmass state with given airmass
    """

    @custom_model
    def AState(x):
        observing_state = state.get()
        return airmass.at(observing_state.target_coord,
                          observing_state.obstime) * u.dimensionless_unscaled

    return AState()


@custom_model
def UnknownAirmassState(x):
    """Returns airmass state with unknown observatory location, which
    will need to be set as part of the ObservingState
    """
    observing_state = state.get()
    airmass = Airmass(observing_state.observatory_loc)
    return airmass.at(observing_state.target_coord,
                      observing_state.obstime) * u.dimensionless_unscaled


class AtmoExtinction:
    """
    AtmoExtinction: Attentuation of spectrum due to atmospheric effects.

    Atmospheric effects are calculated via an airmass extinction table,
    and so require the observatory location to be defined for airmass
    calculation.

    NOTE: Default extinction table is Kitt Peak.

    Examples
    --------

    >>> from m4opt.models.extinction import AtmoExtinction, Airmass
    >>> import astropy.units as u
    >>> from astropy.coordinates import SkyCoord, EarthLocation
    >>> from astropy.time import Time

    >>> time = Time('2012-7-13 07:00:00')
    >>> target = SkyCoord.from_name('m33')

    AtmoExtinction models require an observer location in order to calculate
    the airmass via the local zenith angle of the target. We can do this
    several different ways. First, we can simply pass all of the
    required information:

    >>> place = EarthLocation(lat=41.3*u.deg, lon=-74*u.deg, height=390*u.m)
    >>> airmass = Airmass(place)
    >>> extn = AtmoExtinction.at(airmass, target, time)
    >>> extn(3200*u.Angstrom)
    <Quantity 0.23643961>

    Alternatively, we can define the object for a given observer location:
    >>> extn = AtmoExtinction.from_observer(place, airmass_model='simple')

    However, this requires the use of `state` to fill in the
    target information:

    >>> from m4opt.models import state
    >>> with state.set_observing(target_coord=target, obstime=time):
    ...     print(extn(3200*u.Angstrom))
    0.23643960524293295

    If we already have an airmass, we can initialize with it instead:
    >>> extn = AtmoExtinction.from_airmass(airmass)
    >>> with state.set_observing(target_coord=target, obstime=time):
    ...     print(extn(3200*u.Angstrom))
    0.23643960524293295

    Or we can initialize a blank state, and fill in the blanks later:
    >>> extn = AtmoExtinction()
    >>> with state.set_observing(target_coord=target, obstime=time, \
                                 observatory_loc=place):
    ...     print(extn(3200*u.Angstrom))
    0.23643960524293295

    Finally, AtmoExtinction models require an extinction table. All of the
    models above have used the default table. However, other tables
    can be chosen at initialization by passing the appropriate parameter:
    >>> AtmoExtinction.available_tables
    {'apo': 'apoextinct.dat',
    'kpno': 'kpnoextinct.dat',
    'ctio': 'ctioextinct.dat',
    'lapalm': 'lapalmextinct.dat',
    'mko': 'mkoextinct.dat',
    'lick': 'mthamextinct.dat',
    'paranal': 'paranalextinct.dat'}

    >>> extn = AtmoExtinction.at(airmass, target, time, table_name='kpno')
    >>> extn(3200*u.Angstrom)
    <Quantity 0.23643961>

    >>> extn = AtmoExtinction(table_name='apo')
    >>> with state.set_observing(target_coord=target, obstime=time, \
                                 observatory_loc=place):
    ...     print(extn(3200*u.Angstrom))
    0.2651060248681333

    """

    def __new__(cls, table_name=None):
        return Const1D(10.)**(
                              Const1D(-0.4) * UnknownAirmassState()
                              * cls.table(table_name)
                             )

    @classmethod
    def from_observer(cls, earth_loc, table_name=None, **kwargs):
        airmass = Airmass(earth_loc, **kwargs)
        airmass_state = KnownAirmassState(airmass)
        return Const1D(10.)**(
                              Const1D(-0.4)*airmass_state
                              * cls.table(table_name)
                              )

    @classmethod
    def from_airmass(cls, airmass, table_name=None):
        airmass_state = KnownAirmassState(airmass)
        return Const1D(10.)**(
                              Const1D(-0.4)*airmass_state
                              * cls.table(table_name)
                              )

    @classmethod
    def at(cls, airmass, target_coord, obstime, table_name=None):
        return Const1D(10.)**(
                              Const1D(-0.4*airmass.at(target_coord, obstime))
                              * cls.table(table_name)
                              )

    @classmethod
    def table(cls, table_name):
        if table_name is None:
            return cls.default_table()
        elif table_name.lower() not in cls.available_tables:
            raise AttributeError("table name must be one of {0}".format(
                                  cls.available_tables))
        else:
            return cls.extinction_table(table_name)

    @classmethod
    def default_table(cls):
        """Default Airmass Extinction Table"""
        return cls.extinction_table("kpno")

    @staticmethod
    def extinction_table(table_name):
        result = Tabular1D(*read_extinction_table(table_name))
        result.input_units_equivalencies = (BaseExtinction.
                                            input_units_equivalencies)
        return result

    available_tables = {"apo": 'apoextinct.dat', 'kpno': 'kpnoextinct.dat',
                        'ctio': 'ctioextinct.dat',
                        'lapalm': 'lapalmextinct.dat', 'mko': 'mkoextinct.dat',
                        'lick': 'mthamextinct.dat',
                        'paranal': 'paranalextinct.dat'}
