"""
This test suite verifies that the model module can plug into the synphot machinery.
"""

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time

from m4opt.missions._uvex import uvex
from m4opt.models import VanVelzenTDESED
from m4opt.skygrid import _geodesic
from m4opt.synphot import observing
from m4opt.synphot.extinction import DustExtinction


@pytest.mark.xfail(
    reason=(
        "m4opt.synphot._math.countrate's cubic-interpolation shortcut for "
        "dust extinction (skygrid sizes >= 512) assumes the source spectrum "
        "carries no batch axis of its own; np.vectorize(otypes=[float]) "
        "chokes when the per-Ebv countrate comes back array-valued instead "
        "of scalar, as happens for a per-event batched SED. Needs a fix in "
        "_math.py to interpolate along a trailing Ebv axis instead of "
        "assuming a scalar result."
    ),
    strict=True,
)
def test_synthetic_photometry():

    # Generate a skygrid of points so that we can place a synthetic event
    # at each of them.
    skygrid = _geodesic.for_subdivision(21, 4, "icosahedron")
    n_events = len(skygrid)

    # Choose a photometry model. For this instance, we'll just use
    # a simple Van Vezlen + 2021 TDE model that has pre-set priors.
    SED_MODEL = VanVelzenTDESED()

    # Sample a set of parameters, one realization per event, from a fixed
    # seed for reproducibility. Each parameter gets a trailing axis so that
    # its per-event batch axis broadcasts against the (unbatched) wavelength
    # axis the spectrum is called with, rather than colliding with it.
    PARAMETERS = {
        name: value[:, np.newaxis]
        for name, value in SED_MODEL.sample_parameters(n_events, rng=0).items()
    }

    # Observe every event 10 days after explosion, at a single observing epoch.
    TIME_SINCE_EXPLOSION = 10 * u.day
    OBSTIME = Time("2025-01-01T00:00:00", scale="utc")

    # Create one batched SourceSpectrum, fixed at TIME_SINCE_EXPLOSION, ready
    # to feed into the detector -- still fully vectorized over all n_events
    # parameter realizations via ordinary NumPy broadcasting. Multiply in
    # Milky Way dust extinction, looked up per sky position from the
    # `observing()` state below (see DustExtinctionForSkyCoord).
    spectra = SED_MODEL.generate_spectrum(TIME_SINCE_EXPLOSION, **PARAMETERS)
    spectra = spectra * DustExtinction()

    with observing(
        uvex.observer_location(OBSTIME),
        skygrid,
        OBSTIME,
    ):
        SNR = uvex.detector.get_snr(900 * u.s, spectra, "FUV")

    assert np.all(SNR >= 0), "All events should have positive SNR in the FUV band."
