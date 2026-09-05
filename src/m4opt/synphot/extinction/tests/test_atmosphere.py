import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from ..._extrinsic import observing
from .._atmosphere import _MAX_AIRMASS, AtmosphericExtinction, read_extinction_model

SITE = "kpno"
WAVELENGTH = 4800 * u.angstrom
OBSTIME = Time("2025-03-19T07:00:00")


@pytest.fixture
def location():
    return EarthLocation.of_site("Palomar")


@pytest.mark.remote_data
@pytest.mark.parametrize("airmass", [1.0, 1.5, 2.0, 2.5])
def test_transmission_follows_bouguer(airmass):
    """Transmission is the tabulated extinction in magnitudes times the airmass."""
    coefficient = float(read_extinction_model(SITE)(WAVELENGTH))
    transmission = float(AtmosphericExtinction(SITE, airmass=airmass)(WAVELENGTH))
    np.testing.assert_allclose(transmission, 10 ** (-0.4 * coefficient * airmass))


@pytest.mark.remote_data
def test_transmission_decreases_with_airmass():
    """More atmosphere means less light."""
    airmass = np.asarray([1.0, 1.5, 2.0, 2.5, 3.0])
    transmission = [
        float(AtmosphericExtinction(SITE, airmass=x)(WAVELENGTH)) for x in airmass
    ]
    assert np.all(np.diff(transmission) < 0)
    assert np.all(np.asarray(transmission) <= 1)


@pytest.mark.remote_data
def test_transmission_is_greater_in_the_red():
    """Extinction is dominated by Rayleigh scattering, so the blue suffers more."""
    extinction = AtmosphericExtinction(SITE, airmass=1.5)
    blue, red = (float(extinction(w * u.angstrom)) for w in (4000, 8000))
    assert blue < red


@pytest.mark.remote_data
def test_matches_the_airmass_of_the_line_of_sight(location):
    """Evaluating in an observing context agrees with the explicit airmass."""
    altitude = 40 * u.deg
    coord = SkyCoord(
        alt=altitude, az=0 * u.deg, frame=AltAz(location=location, obstime=OBSTIME)
    ).icrs
    airmass = 1 / np.sin(altitude).to_value(u.dimensionless_unscaled)

    with observing(location, coord, OBSTIME):
        from_context = float(AtmosphericExtinction(SITE)(WAVELENGTH))
    explicit = float(AtmosphericExtinction(SITE, airmass=airmass)(WAVELENGTH))
    np.testing.assert_allclose(from_context, explicit, rtol=1e-3)


@pytest.mark.remote_data
def test_below_the_horizon_is_opaque(location):
    """A target below the horizon is extinguished, not amplified.

    The cosecant formula runs away at the horizon and changes sign below it, so
    the airmass is capped.
    """
    extinction = AtmosphericExtinction(SITE)
    transmission = []
    for altitude in [10, 1, 0, -10, -45] * u.deg:
        coord = SkyCoord(
            alt=altitude, az=0 * u.deg, frame=AltAz(location=location, obstime=OBSTIME)
        ).icrs
        with observing(location, coord, OBSTIME):
            transmission.append(float(extinction(WAVELENGTH)))
    transmission = np.asarray(transmission)

    assert np.all(transmission >= 0)
    assert np.all(transmission <= 1)
    coefficient = float(read_extinction_model(SITE)(WAVELENGTH))
    np.testing.assert_allclose(
        transmission[-1], 10 ** (-0.4 * coefficient * _MAX_AIRMASS)
    )
