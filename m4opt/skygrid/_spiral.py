import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

GOLDEN_ANGLE = np.pi * (3 - np.sqrt(5)) * u.rad


def golden_angle_spiral(area: u.Quantity[u.physical.solid_angle]):
    """Generate a tile grid from a spiral employing the
    `golden angle <https://mathworld.wolfram.com/GoldenAngle.html>`_.

    This is a spiral-based spherical packing scheme that was used by GRANDMA
    during LIGO/Virgo O3 :footcite:`2020MNRAS.497.5518A`.

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

    References
    ----------
    .. footbibliography::

    """
    n = int(np.ceil(1 / area.to_value(u.spat)))
    ra = GOLDEN_ANGLE * np.arange(n)
    dec = np.arcsin(np.linspace(1 - 1.0 / n, 1.0 / n - 1, n)) * u.rad
    return SkyCoord(ra, dec)
