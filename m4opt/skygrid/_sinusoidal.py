import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord


def sinusoidal(area):
    """Generate a uniform grid on a sinusoidal equal area projection.

    This is similar to what was used for GRANDMA follow-up in LIGO/Virgo
    Observing Run 3 (O3), but is more efficient at tiling the poles.
    See :doi:`10.3847/2041-8213/ab3399`.

    Parameters
    ----------
    area : :class:`astropy.units.Quantity`
        The average area per tile in any Astropy solid angle units:
        for example, :samp:`10 * astropy.units.deg**2` or
        :samp:`0.1 * astropy.units.steradian`.

    Returns
    -------
    coords : :class:`astropy.coordinates.SkyCoord`
        The coordinates of the tiles.

    References
    ----------
    https://en.wikipedia.org/wiki/Sinusoidal_projection

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
