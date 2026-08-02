import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord


def sinusoidal(area: u.Quantity[u.physical.solid_angle]):
    """Generate a uniform grid on a
    `sinusoidal projection <https://mathworld.wolfram.com/SinusoidalProjection.html>`_.

    This is similar to what was used for GRANDMA follow-up in LIGO/Virgo
    Observing Run 3 (O3), but is more efficient at tiling the poles
    :footcite:`2019ApJ...881L..16A`.

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
    # Diameter of the field of view
    width = np.sqrt(area.to_value(u.sr))

    # Declinations of equal-declination strips
    n_decs = int(np.ceil(np.pi / width)) + 1
    decs = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_decs)
    d_dec = decs[1] - decs[0]

    # Number of RA steps in each equal-declination strip
    decs_edge = decs - 0.5 * d_dec * np.where(decs >= 0, 1, -1)
    n_ras = np.ceil(2 * np.pi * np.cos(decs_edge) / width).astype(int)
    n_ras[0] = n_ras[-1] = 1

    ras = np.concatenate([np.linspace(0, 2 * np.pi, n, endpoint=False) for n in n_ras])
    decs = np.concatenate([np.repeat(dec, n) for n, dec in zip(n_ras, decs)])
    return SkyCoord(ras * u.rad, decs * u.rad)
