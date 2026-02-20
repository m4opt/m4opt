import numpy as np
import synphot
from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time

from ... import rubin
from ....synphot._extrinsic import observing


def test_rubin_detector_exists():
    """Test that the Rubin mission has a detector configured."""
    assert rubin.detector is not None
    assert set(rubin.detector.bandpasses.keys()) == {"u", "g", "r", "i", "z", "y"}


def test_rubin_detector_dark_noise():
    """Test that dark current is set to 0.2 e-/s/pixel."""
    assert rubin.detector.dark_noise.to_value(u.Hz) == 0.2


def test_rubin_limmag_at_zenith():
    """Test 5-sigma limiting magnitudes for a 30s exposure at zenith."""
    obstime = Time("2025-03-19T07:00:00")
    loc = rubin.observer_location(obstime)
    frame = AltAz(location=loc, obstime=obstime)
    coord = SkyCoord(alt=90 * u.deg, az=0 * u.deg, frame=frame)

    with observing(loc, coord, obstime):
        limmags = [
            rubin.detector.get_limmag(
                5,
                30 * u.s,
                synphot.SourceSpectrum(synphot.ConstFlux1D, amplitude=0 * u.ABmag),
                band,
            ).to_value(u.mag)
            for band in "ugrizy"
        ]

    np.testing.assert_almost_equal(
        limmags,
        [24.6962075, 25.4230120, 25.0115625, 24.8190254, 24.5274659, 24.0041817],
    )


def test_rubin_limmag_increases_with_exptime():
    """Test that longer exposures yield deeper limiting magnitudes."""
    obstime = Time("2025-03-19T07:00:00")
    loc = rubin.observer_location(obstime)
    frame = AltAz(location=loc, obstime=obstime)
    coord = SkyCoord(alt=90 * u.deg, az=0 * u.deg, frame=frame)

    with observing(loc, coord, obstime):
        lm_30s = rubin.detector.get_limmag(
            5,
            30 * u.s,
            synphot.SourceSpectrum(synphot.ConstFlux1D, amplitude=0 * u.ABmag),
            "r",
        ).to_value(u.mag)
        lm_300s = rubin.detector.get_limmag(
            5,
            300 * u.s,
            synphot.SourceSpectrum(synphot.ConstFlux1D, amplitude=0 * u.ABmag),
            "r",
        ).to_value(u.mag)

    np.testing.assert_almost_equal(lm_30s, 25.0115625)
    np.testing.assert_almost_equal(lm_300s, 26.3266559)
