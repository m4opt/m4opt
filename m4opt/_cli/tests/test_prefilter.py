import numpy as np
from astropy import units as u
from astropy.coordinates import ICRS, SkyCoord
from astropy.table import Table
from astropy_healpix import HEALPix

from ..schedule import prefilter_fields


def test_prefilter_keeps_overlapping_fields():
    """Fields overlapping nonzero pixels are kept."""
    hpx = HEALPix(nside=32, frame=ICRS(), order="nested")
    # Create a sky map with one nonzero pixel at (0, 0)
    pixel_index = hpx.skycoord_to_healpix(SkyCoord(0 * u.deg, 0 * u.deg))
    prob = np.zeros(hpx.npix)
    prob[pixel_index] = 1.0
    skymap_flat = Table({"PROB": prob})

    # One field near (0, 0) and one far away
    target_coords = SkyCoord([0, 180] * u.deg, [0, 0] * u.deg)
    fov_radius = 5 * u.deg

    mask = prefilter_fields(hpx, skymap_flat, target_coords, fov_radius)
    assert mask[0]
    assert not mask[1]


def test_prefilter_all_zero():
    """All-zero sky map returns all-False mask."""
    hpx = HEALPix(nside=32, frame=ICRS(), order="nested")
    skymap_flat = Table({"PROB": np.zeros(hpx.npix)})
    target_coords = SkyCoord([0, 90] * u.deg, [0, 0] * u.deg)
    fov_radius = 5 * u.deg

    mask = prefilter_fields(hpx, skymap_flat, target_coords, fov_radius)
    assert not mask.any()
