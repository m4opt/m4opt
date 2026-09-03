"""Tests for the earthshine background model."""

import numpy as np
import pytest
from astropy import units as u
from astropy.constants import R_earth
from astropy.coordinates import (
    GCRS,
    EarthLocation,
    SkyCoord,
    SphericalRepresentation,
    get_sun,
)
from astropy.time import Time

from .....constraints._earth_limb import _get_angle_from_earth_limb
from .... import observing
from .. import (
    _HST_ALTITUDE,
    _HST_ANGULAR_RADIUS_DEG,
    _REFERENCE_STRAY_LIGHT,
    EarthshineBackground,
    EarthshineBackgroundScaleFactor,
    _stray_light,
)

# On 2025-01-01, the Sun is at RA~281 deg, Dec~-23 deg. The limb angle tests
# divide out the illumination factor so that the two are exercised separately.
_TEST_OBSTIME = Time("2025-01-01T00:00:00Z")
_SUN_RA_DEG = 281.0


def test_earthshine_high_positive():
    """EarthshineBackground.high() returns positive flux across its wavelength range."""
    spec = EarthshineBackground.high()
    wave = np.arange(1500, 10001) * u.AA
    assert np.all(spec(wave).value > 0)


def test_earthshine_high_regression():
    """Flux at specific wavelengths matches frozen values from the ECSV data."""
    spec = EarthshineBackground.high()
    np.testing.assert_almost_equal(
        spec(2600 * u.AA).value, 7.355851128493407e-11, decimal=16
    )
    np.testing.assert_almost_equal(
        spec(5000 * u.AA).value, 6.619863286318664e-07, decimal=12
    )


def test_earthshine_high_uv_fainter_than_visible():
    """Earthshine is reflected sunlight, so UV should be fainter than visible."""
    spec = EarthshineBackground.high()
    uv = spec(2600 * u.AA).value
    vis = spec(5000 * u.AA).value
    assert uv < vis


def test_earthshine_in_context():
    """Works within observing() context."""
    loc = EarthLocation.from_geodetic(
        lon=15 * u.deg, lat=0 * u.deg, height=35786 * u.km
    )
    coord = SkyCoord(0 * u.deg, 0 * u.deg)
    obstime = Time("2025-05-18T02:48:00Z")

    bg = EarthshineBackground()
    with observing(observer_location=loc, target_coord=coord, obstime=obstime):
        val = bg(2600 * u.AA)
    # FIXME: add a more constraining test (e.g., regression value check)
    assert val.value > 0


def test_earthshine_requires_context():
    """EarthshineBackground() raises ValueError without observing() context."""
    bg = EarthshineBackground()
    with pytest.raises(ValueError, match="Unknown target"):
        bg(5000 * u.AA)


@pytest.mark.parametrize(
    "limb_angle_deg,expected_scale", [(24, 2.0), (38, 1.0), (50, 0.5)]
)
def test_stray_light_reproduces_calibration_points(limb_angle_deg, expected_scale):
    """The integral reproduces the STIS levels for the geometry they describe.

    Those are measurements from HST looking past a fully sunlit Earth, so they
    fix the exponent of the point source transmittance; anything else is a
    prediction. Unit vectors are used directly so that the check does not
    depend on placing an observer in a particular frame.
    """
    distance = ((R_earth + _HST_ALTITUDE) / R_earth).to_value(u.dimensionless_unscaled)
    observer = np.array([0.0, 0.0, 1.0])
    separation = np.radians(_HST_ANGULAR_RADIUS_DEG + limb_angle_deg)
    target = np.array([np.sin(separation), 0.0, -np.cos(separation)])

    scale = _stray_light(distance, observer, target, observer) / _REFERENCE_STRAY_LIGHT
    np.testing.assert_allclose(scale, expected_scale, rtol=0.25)


def test_earthshine_scale_factor_below_limb():
    """Scale factor is zero for targets below the Earth's limb."""
    sf = EarthshineBackgroundScaleFactor()

    # Observer at 2*R_earth, limb_alt = 60 deg.
    # Target at alt = -90 deg => limb_angle = -90 + 60 = -30 deg (below limb).
    loc = EarthLocation.from_geocentric(0 * u.m, 0 * u.m, 2 * R_earth)
    coord = SkyCoord(_SUN_RA_DEG * u.deg, -90 * u.deg)

    scale = sf.at(loc, coord, _TEST_OBSTIME)
    assert scale == 0.0


def test_stray_light_falls_off_away_from_the_limb():
    """Earthshine decreases as the line of sight moves away from the Earth."""
    distance = 6.6  # geostationary, in Earth radii
    observer = np.array([0.0, 0.0, 1.0])
    angular_radius = np.degrees(np.arcsin(1 / distance))

    separation = np.radians(angular_radius + np.array([5.0, 20.0, 45.0, 80.0]))
    target = np.stack(
        [np.sin(separation), np.zeros_like(separation), -np.cos(separation)], axis=-1
    )
    scale = _stray_light(distance, observer, target, observer)
    assert np.all(np.diff(scale) < 0)


def test_earthshine_illumination_sunlit_vs_dark():
    """Earthshine is stronger when looking toward the sunlit side of Earth."""
    sf = EarthshineBackgroundScaleFactor()

    loc = EarthLocation.from_geocentric(0 * u.m, 0 * u.m, 2 * R_earth)

    # Use the same limb angle (38 deg => alt = -22 deg) but different RAs:
    # one near the Sun (sunlit limb) and one opposite (dark limb).
    sun = get_sun(_TEST_OBSTIME)
    alt_deg = -22  # limb angle = 38 deg

    coord_sunlit = SkyCoord(sun.ra, alt_deg * u.deg)
    coord_dark = SkyCoord(sun.ra + 180 * u.deg, alt_deg * u.deg)

    scale_sunlit = sf.at(loc, coord_sunlit, _TEST_OBSTIME)
    scale_dark = sf.at(loc, coord_dark, _TEST_OBSTIME)

    assert scale_sunlit > 0
    assert scale_dark >= 0
    assert scale_sunlit > 2 * scale_dark


def test_earthshine_depends_on_observer():
    """Two observers on opposite sides of the Earth see different earthshine.

    They look past opposite parts of the Earth, one of which is better lit.
    """
    sf = EarthshineBackgroundScaleFactor()
    coord = SkyCoord(_SUN_RA_DEG * u.deg, 0 * u.deg)
    near = EarthLocation.from_geocentric(2 * R_earth, 0 * u.m, 0 * u.m)
    far = EarthLocation.from_geocentric(-2 * R_earth, 0 * u.m, 0 * u.m)
    assert sf.at(near, coord, _TEST_OBSTIME) != sf.at(far, coord, _TEST_OBSTIME)


def observer_at(distance):
    """An observer at a given geocentric distance, over the equator."""
    return EarthLocation(
        *SphericalRepresentation(0 * u.deg, 0 * u.deg, distance).to_cartesian().xyz
    )


def nadir_at(observer_location, obstime):
    """The direction from the observer back toward the Earth."""
    frame = GCRS(obstime=obstime)
    cartesian = observer_location.get_gcrs(obstime).cartesian
    spherical = SkyCoord(
        *(-(cartesian / cartesian.norm())).xyz.value,
        representation_type="cartesian",
        frame=frame,
    ).spherical
    return SkyCoord(ra=spherical.lon, dec=spherical.lat, frame=frame)


@pytest.mark.parametrize("distance", [2, 5, 6.6, 20, 100] * u.Rearth)
def test_earthshine_occultation_shrinks_with_distance(distance):
    """The occulted region is the Earth's disk, which shrinks with distance.

    Restricted to distances where an :class:`~astropy.coordinates.EarthLocation`
    still describes the observer well; see the warning in
    :mod:`m4opt.synphot.background`.
    """
    sf = EarthshineBackgroundScaleFactor()
    observer_location = observer_at(distance)
    nadir = nadir_at(observer_location, _TEST_OBSTIME)
    angular_radius = np.arcsin(u.Rearth / distance).to(u.deg)

    # Spanning the limb at every distance, but avoiding the nadir itself, where
    # the geodetic vertical and the geocentric radius differ by a few arcsec.
    separation = np.concatenate(
        [angular_radius * [0.5, 0.9, 1.1, 2.0], [10, 45, 90] * u.deg]
    )
    coord = nadir.directional_offset_by(0 * u.deg, separation)
    limb_angle = _get_angle_from_earth_limb(observer_location, coord, _TEST_OBSTIME)

    # The limb angle is the separation from the Earth's centre less its angular
    # radius, so the occulted region shrinks in proportion to the Earth's size.
    assert u.allclose(limb_angle, separation - angular_radius, atol=2 * u.arcsec)
    assert np.all(sf.at(observer_location, coord[limb_angle < 0], _TEST_OBSTIME) == 0)


@pytest.mark.parametrize("distance", [2, 5, 6.6, 20, 100, 1000] * u.Rearth)
def test_earthshine_finite_at_all_distances(distance):
    """The scale factor stays finite however small the Earth appears.

    The weights that average the illumination around the limb must not
    underflow for a distant observer, where the whole limb subtends a tiny
    angle.
    """
    sf = EarthshineBackgroundScaleFactor()
    observer_location = observer_at(distance)
    nadir = nadir_at(observer_location, _TEST_OBSTIME)
    offsets = np.geomspace(0.01, 90, 40) * u.deg
    coord = nadir.directional_offset_by(0 * u.deg, offsets)

    scale = sf.at(observer_location, coord, _TEST_OBSTIME)
    assert np.all(np.isfinite(scale))
    assert np.all(scale >= 0)


def test_earthshine_broadcasts_over_observers():
    """Arrays of observers and of targets broadcast against each other."""
    sf = EarthshineBackgroundScaleFactor()
    distance = [2, 5, 20] * u.Rearth
    observer_location = EarthLocation(
        *SphericalRepresentation([0, 45, 90] * u.deg, [0, 30, -30] * u.deg, distance)
        .to_cartesian()
        .xyz
    )
    coord = SkyCoord([0, 90, 180] * u.deg, [0, 30, -60] * u.deg)

    scale = sf.at(observer_location, coord, _TEST_OBSTIME)
    assert np.shape(scale) == (3,)
    assert np.all(np.isfinite(scale))

    # Each element must agree with evaluating that observer on its own.
    for i in range(3):
        one = EarthLocation(*observer_location[i].geocentric)
        np.testing.assert_allclose(
            sf.at(one, coord[i], _TEST_OBSTIME), scale[i], rtol=1e-12
        )
