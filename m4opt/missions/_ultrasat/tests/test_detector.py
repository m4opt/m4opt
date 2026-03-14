import numpy as np
import synphot
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from ....synphot._extrinsic import observing
from ... import ultrasat


def test_ultrasat_limmag():
    """Test 5-sigma limiting magnitudes for ULTRASAT NUV band at GEO."""
    obstime = Time("2025-05-18T02:48:00Z")
    loc = ultrasat.observer_location(obstime)
    coord = SkyCoord(ra=180 * u.deg, dec=0 * u.deg)

    with observing(loc, coord, obstime):
        limmag = ultrasat.detector.get_limmag(
            5,
            [300, 900] * u.s,
            synphot.SourceSpectrum(synphot.ConstFlux1D, amplitude=0 * u.ABmag),
            "NUV",
        ).to_value(u.mag)

    np.testing.assert_almost_equal(
        limmag,
        [21.59579087, 22.27842069],
    )
