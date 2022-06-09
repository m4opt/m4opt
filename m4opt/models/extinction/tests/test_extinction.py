from ..airmass import Airmass, AtmosphericExtinction
from ..airmass import simple_airmass, KastenYoung_airmass

import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
import numpy as np

AIRMASS_KP = Airmass(EarthLocation(-1994502.60430614, -5037538.54232911,
                     3358104.99690298, unit='m'))
TARGET = SkyCoord.from_name('m33')
TIME = Time('2012-7-13 07:00:00')


def test_simple():
    assert np.isclose(simple_airmass(0*u.deg).value, 1.)
    assert np.isclose(simple_airmass(0.785398*u.rad).value, 1.4142133312942642)
    assert np.isclose(simple_airmass(89*u.deg).value, 57.2986884985499)


def test_kasyoung():
    assert np.isclose(KastenYoung_airmass(0.*u.rad).value, 0.9997119918558381)
    assert np.isclose(KastenYoung_airmass(45.*u.deg).value, 1.4125952520262743)
    assert np.isclose(KastenYoung_airmass(89*u.deg).value, 26.31055506838526)


def test_Atmo_KPNO_linear():
    atmo = AtmosphericExtinction.at(AIRMASS_KP, TARGET, TIME,
                                    table_name='kpno', table_method='linear')
    assert np.isclose(atmo(3250*u.Angstrom).value, 0.006363833100109559)
    assert np.isclose(atmo(3275*u.Angstrom).value, 0.008351825903814643)


# default is kpno
def test_Atmo_default_linear():
    atmo = AtmosphericExtinction.at(AIRMASS_KP, TARGET, TIME,
                                    table_method='linear')
    assert np.isclose(atmo(3250*u.Angstrom).value, 0.006363833100109559)
    assert np.isclose(atmo(3275*u.Angstrom).value, 0.008351825903814643)


def test_Atmo_default_nearest():
    atmo = AtmosphericExtinction.at(AIRMASS_KP, TARGET, TIME,
                                    table_name='kpno', table_method='nearest')
    assert np.isclose(atmo(3250*u.Angstrom).value, 0.006363833100109559)
    assert np.isclose(atmo(3275*u.Angstrom).value, 0.010915792117502507)


"""
@pytest.mark.remote_data
def test_read_extinction():
    freq, ext = read_extinction_table("parnal")
    assert freq[20].value == 422242898591549.25
    assert ext[20] == 0.072
"""
