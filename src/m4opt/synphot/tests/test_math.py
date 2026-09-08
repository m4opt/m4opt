"""Tests for the count rate integral."""

import numpy as np
import pytest
import synphot
from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time

from ...missions import ztf
from .._extrinsic import observing
from .._math import countrate
from ..extinction import AtmosphericExtinction, DustExtinction

OBSTIME = Time("2025-03-19T07:00:00")
FLAT = synphot.SourceSpectrum(synphot.ConstFlux1D, amplitude=0 * u.ABmag)

# Enough targets to take the interpolated path, in batches small enough to
# take the direct one.
TARGETS = 600
BATCH = 50


def sky_positions(n, seed=1):
    """Random positions above the horizon, as seen from the observatory."""
    rng = np.random.default_rng(seed)
    location = ztf.observer_location(OBSTIME)
    coord = SkyCoord(
        alt=rng.uniform(25, 89, n) * u.deg,
        az=rng.uniform(0, 360, n) * u.deg,
        frame=AltAz(location=location, obstime=OBSTIME),
    ).icrs
    return location, coord


@pytest.mark.remote_data
@pytest.mark.parametrize(
    "spectrum",
    [
        FLAT * AtmosphericExtinction("kpno"),
        FLAT * DustExtinction(),
        FLAT * DustExtinction() * AtmosphericExtinction("kpno"),
    ],
    ids=["atmosphere", "dust", "both"],
)
def test_extinction_interpolation_matches_direct_integration(spectrum):
    """Extinction interpolated over a grid matches integrating every target.

    Above a threshold number of targets the spectrum is integrated over a grid
    of reddening and of airmass and interpolated onto the targets, rather than
    integrated once for each of them. Evaluating the same targets in batches
    small enough to stay under that threshold takes the direct path instead, so
    the two must agree.
    """
    bandpass = ztf.detector.bandpasses["g"]
    location, coord = sky_positions(TARGETS)

    with observing(location, coord, OBSTIME):
        interpolated = np.asarray(countrate(spectrum, bandpass).value)

    direct = []
    for start in range(0, TARGETS, BATCH):
        with observing(location, coord[start : start + BATCH], OBSTIME):
            direct.append(
                np.atleast_1d(np.asarray(countrate(spectrum, bandpass).value))
            )
    direct = np.concatenate(direct)

    np.testing.assert_allclose(interpolated, direct, rtol=1e-3)
