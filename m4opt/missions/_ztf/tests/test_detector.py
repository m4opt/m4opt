import numpy as np
import synphot
from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time

from ... import ztf
from ....synphot._extrinsic import observing


def test_ztf_detector_exists():
    """Test that the ZTF mission has a detector configured."""
    assert ztf.detector is not None
    assert set(ztf.detector.bandpasses.keys()) == {"g", "r", "i"}


def test_ztf_detector_dark_noise():
    """Test that dark current is set to 3 e-/pixel/hour."""
    assert ztf.detector.dark_noise.to_value(1 / u.hr) == 3.0


def test_ztf_limmag_at_zenith():
    """Smoke test: compute 5-sigma limiting magnitudes for a 30s exposure at zenith."""
    obstime = Time("2025-03-19T07:00:00")
    loc = ztf.observer_location(obstime)
    frame = AltAz(location=loc, obstime=obstime)
    coord = SkyCoord(alt=90 * u.deg, az=0 * u.deg, frame=frame)

    with observing(loc, coord, obstime):
        limmags = {}
        for band in "gri":
            limmags[band] = ztf.detector.get_limmag(
                5,
                30 * u.s,
                synphot.SourceSpectrum(synphot.ConstFlux1D, amplitude=0 * u.ABmag),
                band,
            ).to_value(u.mag)

    # Limiting magnitudes should be finite and in a reasonable range
    for band, lm in limmags.items():
        assert np.isfinite(lm), f"{band}-band limiting magnitude is not finite"
        assert 18 < lm < 24, f"{band}-band limiting magnitude {lm:.1f} out of range"

    # Bluer bands should generally be deeper (g > r > i)
    assert limmags["g"] > limmags["r"]
    assert limmags["r"] > limmags["i"]


def test_ztf_limmag_increases_with_exptime():
    """Test that longer exposures yield deeper limiting magnitudes."""
    obstime = Time("2025-03-19T07:00:00")
    loc = ztf.observer_location(obstime)
    frame = AltAz(location=loc, obstime=obstime)
    coord = SkyCoord(alt=90 * u.deg, az=0 * u.deg, frame=frame)

    with observing(loc, coord, obstime):
        lm_30s = ztf.detector.get_limmag(
            5,
            30 * u.s,
            synphot.SourceSpectrum(synphot.ConstFlux1D, amplitude=0 * u.ABmag),
            "r",
        ).to_value(u.mag)
        lm_300s = ztf.detector.get_limmag(
            5,
            300 * u.s,
            synphot.SourceSpectrum(synphot.ConstFlux1D, amplitude=0 * u.ABmag),
            "r",
        ).to_value(u.mag)

    assert lm_300s > lm_30s
