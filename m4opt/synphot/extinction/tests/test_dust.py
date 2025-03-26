import numpy as np
import pytest
import synphot
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from hypothesis import given, settings
from hypothesis import strategies as st

from ....missions import ultrasat
from ....tests.hypothesis import earth_locations, obstimes, skycoords
from ..._extrinsic import observing
from .._dust import DustExtinction, dust_map, reddening_law


@settings(deadline=None)
@given(
    earth_locations,
    skycoords,
    obstimes,
    st.floats(0.0912, 32.0).map(lambda _: _ * u.micron),
)
def test_dust_extinction(observer_location, target_coord, obstime, wavelength):
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


@pytest.mark.parametrize(
    "Ebv,wavelength,expected",
    [
        ([0.1, 0.2], [1000], [0.235205, 0.05532139]),
        ([[0.1], [0.2]], [1000], [[0.235205], [0.055321]]),
        ([[0.1], [0.2]], 1000, [[0.235205], [0.055321]]),
        (0.1, [1000, 2000], [0.235205, 0.439729]),
    ],
)
def test_broadcast_dust_extinction(Ebv, wavelength, expected):
    result = DustExtinction(Ebv=Ebv)(wavelength * u.angstrom).to_value(
        u.dimensionless_unscaled
    )
    np.testing.assert_array_almost_equal(result, expected)


def test_broadcast_dust_extinction_skycoord():
    with observing(
        observer_location=EarthLocation(0 * u.m, 0 * u.m, 0 * u.m),
        obstime=Time("2024-01-01"),
        target_coord=SkyCoord([0, 0] * u.deg, [0, 90] * u.deg),
    ):
        result = ultrasat.detector.get_limmag(
            10,
            300 * u.s,
            synphot.SourceSpectrum(synphot.ConstFlux1D, amplitude=0 * u.ABmag)
            * synphot.SpectralElement(DustExtinction()),
        ).to_value(u.mag)
    np.testing.assert_almost_equal(result, [20.4014027, 19.1963705])
