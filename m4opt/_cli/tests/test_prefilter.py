import numpy as np
from astropy import units as u
from astropy.coordinates import ICRS, SkyCoord
from astropy.table import Table
from astropy_healpix import HEALPix

from ..schedule import prefilter_fields


def test_prefilter_keeps_overlapping_fields():
    """Fields overlapping pixels in the credible region are kept."""
    hpx = HEALPix(nside=32, frame=ICRS(), order="nested")
    # Create a sky map with one high-probability pixel at (0, 0)
    pixel_index = hpx.skycoord_to_healpix(SkyCoord(0 * u.deg, 0 * u.deg))
    prob = np.full(hpx.npix, 1e-30)
    prob[pixel_index] = 1.0
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
