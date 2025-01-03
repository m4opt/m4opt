import pytest
from astropy import units as u
from hypothesis import given, settings
from hypothesis import strategies as st

from ...tests.hypothesis import earth_locations, obstimes, skycoords
from .._extinction import DustExtinction, dust_map, reddening_law
from .._extrinsic import observing


@settings(deadline=None)
@given(
    earth_locations,
    skycoords,
    obstimes,
    st.floats(0.0912, 32.0).map(lambda _: _ * u.micron),
)
def test_extinction(observer_location, target_coord, obstime, wavelength):
    Ebv = dust_map().query(target_coord)
    expected = reddening_law.extinguish(wavelength, Ebv=Ebv)

    with observing(
        observer_location=observer_location, target_coord=target_coord, obstime=obstime
    ):
        result = DustExtinction()(wavelength)
    assert result.unit == u.dimensionless_unscaled
    assert expected == pytest.approx(result.value)

    result = DustExtinction(Ebv)(wavelength)
    assert result.unit == u.dimensionless_unscaled
    assert expected == pytest.approx(result.value)


def test_broadcast_dust_extinction():
    DustExtinction(Ebv=[0.1, 0.2])([1000] * u.angstrom)
    DustExtinction(Ebv=[[0.1], [0.2]])([1000] * u.angstrom)
    DustExtinction(Ebv=[[0.1], [0.2]])(1000 * u.angstrom)
    DustExtinction(Ebv=0.1)([1000, 2000] * u.angstrom)
