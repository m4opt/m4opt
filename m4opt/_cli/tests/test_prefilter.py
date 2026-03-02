import numpy as np
from astropy import units as u
from astropy.coordinates import ICRS, SkyCoord
from astropy.table import Table
from astropy_healpix import HEALPix

from ..schedule import prefilter_fields


def test_prefilter_keeps_overlapping_fields():
    """Fields overlapping pixels in the credible region are kept."""
    hpx = HEALPix(nside=32, frame=ICRS(), order="nested")
    npix = hpx.npix
    # Create a sky map with a cluster of high-probability pixels near (0, 0)
    prob = np.full(npix, 1e-30)
    # Set ~100 pixels near the pole to high probability
    center = SkyCoord(0 * u.deg, 0 * u.deg)
    all_coords = hpx.healpix_to_skycoord(np.arange(npix))
    nearby = center.separation(all_coords) < 10 * u.deg
    prob[nearby] = 1.0
    prob /= prob.sum()
    skymap_flat = Table({"PROB": prob})

    # One field near (0, 0) and one far away
    target_coords = SkyCoord([0, 180] * u.deg, [0, 0] * u.deg)
    fov_radius = 5 * u.deg

    mask = prefilter_fields(hpx, skymap_flat, target_coords, fov_radius, level=0.99)
    assert mask[0]
    assert not mask[1]


def test_prefilter_level_1():
    """Level=1.0 keeps all fields that overlap any pixel."""
    hpx = HEALPix(nside=32, frame=ICRS(), order="nested")
    prob = np.ones(hpx.npix) / hpx.npix
    skymap_flat = Table({"PROB": prob})

    target_coords = SkyCoord([0, 90, 180] * u.deg, [0, 0, 0] * u.deg)
    fov_radius = 5 * u.deg

    mask = prefilter_fields(hpx, skymap_flat, target_coords, fov_radius, level=1.0)
    assert mask.all()
