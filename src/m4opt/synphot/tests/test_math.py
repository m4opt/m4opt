"""Tests for the count rate integral."""

import numpy as np
import synphot
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from .._extrinsic import observing
from .._math import countrate
from ..extinction import DustExtinction

OBSTIME = Time("2024-01-01")

# Enough targets to take the interpolated path, in batches small enough to
# take the direct one.
TARGETS = 600
BATCH = 50


def test_dust_interpolation_matches_direct_integration():
    """Dust extinction interpolated over a grid matches integrating every target.

    Above a threshold number of targets the spectrum is integrated over a grid
    of reddenings and interpolated onto the targets, rather than integrated
    once for each of them. Evaluating the same targets in batches small enough
    to stay under that threshold takes the direct path instead, so the two must
    agree.
    """
    location = EarthLocation.of_site("Palomar")
    spectrum = (
        synphot.SourceSpectrum(synphot.BlackBody1D, temperature=1000 * u.Kelvin)
        * DustExtinction()
    )
    bandpass = synphot.SpectralElement.from_filter("johnson_r")

    rng = np.random.default_rng(3)
    coord = SkyCoord(
        rng.uniform(0, 360, TARGETS) * u.deg,
        np.degrees(np.arcsin(rng.uniform(-1, 1, TARGETS))) * u.deg,
    )

    with observing(location, coord, OBSTIME):
        interpolated = np.asarray(countrate(spectrum, bandpass).value)

    direct = np.concatenate(
        [
            np.atleast_1d(
                np.asarray(
                    _batch_countrate(
                        location, coord[start : start + BATCH], spectrum, bandpass
                    )
                )
            )
            for start in range(0, TARGETS, BATCH)
        ]
    )

    np.testing.assert_allclose(interpolated, direct, rtol=1e-6)


def _batch_countrate(location, coord, spectrum, bandpass):
    with observing(location, coord, OBSTIME):
        return countrate(spectrum, bandpass).value
