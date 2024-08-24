import pytest
from astropy import units as u
from hypothesis import given, settings
from hypothesis import strategies as st

from ...tests.hypothesis import skycoords
from .._core import state
from .._extinction import Extinction, axav, dust_map


@settings(deadline=None)
@given(skycoords, st.floats(0.0912, 32.0).map(lambda _: _ * u.micron))
def test_extinction(target_coord, wavelength):
    Ebv = dust_map().query(target_coord)
    expected = axav.extinguish(wavelength, Ebv=Ebv)

    with state.set_observing(target_coord=target_coord):
        result = Extinction()(wavelength)
    assert expected == pytest.approx(result)

    result = Extinction(Ebv)(wavelength)
    assert expected == pytest.approx(result)

    result = Extinction.at(target_coord)(wavelength)
    assert expected == pytest.approx(result)
