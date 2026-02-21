"""Test detector exposure time calculators."""

import numpy as np
import pytest
import synphot
from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time

from ...synphot import observing
from .. import Mission, rubin, ztf


@pytest.mark.parametrize(
    "mission,bands,desired",
    [
        [
            rubin,
            "ugrizy",
            [
                [24.6962075, 26.4438678],
                [25.4230120, 26.7672659],
                [25.0115625, 26.3266559],
                [24.8190254, 26.1477613],
                [24.5274659, 25.8741048],
                [24.0041817, 25.4119549],
            ],
        ],
        [
            ztf,
            "gri",
            [
                [21.1216691, 22.5947771],
                [20.8017654, 22.1993963],
                [20.4476654, 21.8932569],
            ],
        ],
    ],
)
def test_ground_based_limmags(mission: Mission, bands: str, desired: list[list[float]]):
    """Test 5-sigma limiting magnitudes at zenith."""
    detector = mission.detector
    obstime = Time("2025-03-19T07:00:00")
    loc = mission.observer_location(obstime)
    frame = AltAz(location=loc, obstime=obstime)
    coord = SkyCoord(alt=90 * u.deg, az=0 * u.deg, frame=frame)
    spectrum = synphot.SourceSpectrum(synphot.ConstFlux1D, amplitude=0 * u.ABmag)
    exptime = [30, 300] * u.s
    snr = 5

    assert detector is not None
    assert set(detector.bandpasses.keys()) == set(bands), (
        "include tests for all of the detector's supported bandpasses"
    )

    with observing(loc, coord, obstime):
        actual = [
            detector.get_limmag(snr, exptime, spectrum, band).to_value(u.mag)
            for band in bands
        ]

    np.testing.assert_almost_equal(actual, desired)
