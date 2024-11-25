import numpy as np
from astropy import units as u
from astropy.coordinates import ICRS
from astropy_healpix import HEALPix
from ligo.skymap.bayestar.filter import ceil_pow_2


def healpix(area: u.Quantity[u.physical.solid_angle]):
    """Generate a grid in HEALPix coordinates.

    Parameters
    ----------
    area
        The average area per tile in any Astropy solid angle units:
        for example, :samp:`10 * astropy.units.deg**2` or
        :samp:`0.1 * astropy.units.steradian`.

    Returns
    -------
    :
        The coordinates of the tiles.

    """
    nside = np.sqrt(u.spat / (12 * area)).to_value(u.dimensionless_unscaled)
    nside = int(max(ceil_pow_2(nside), 1))
    hpx = HEALPix(nside, frame=ICRS())
    return hpx.healpix_to_skycoord(np.arange(hpx.npix))
