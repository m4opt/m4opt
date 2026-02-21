import numpy as np
import synphot
from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time

from ....synphot._extrinsic import observing
from ... import rubin


def test_rubin_limmag():
    """Test 5-sigma limiting magnitudes at zenith."""
    obstime = Time("2025-03-19T07:00:00")
    loc = rubin.observer_location(obstime)
    frame = AltAz(location=loc, obstime=obstime)
    coord = SkyCoord(alt=90 * u.deg, az=0 * u.deg, frame=frame)

    with observing(loc, coord, obstime):
        limmags = [
            rubin.detector.get_limmag(
                5,
                [30, 300] * u.s,
                synphot.SourceSpectrum(synphot.ConstFlux1D, amplitude=0 * u.ABmag),
                band,
            ).to_value(u.mag)
            for band in "ugrizy"
        ]

    np.testing.assert_almost_equal(
        limmags,
        [
            [24.6962075, 26.4438678],
            [25.4230120, 26.7672659],
            [25.0115625, 26.3266559],
            [24.8190254, 26.1477613],
            [24.5274659, 25.8741048],
            [24.0041817, 25.4119549],
        ],
    )
