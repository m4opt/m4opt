"""Earthshine (stray light) background model.

This module models the earthshine background: sunlight reflected off Earth that
scatters into the telescope. The "High Earthshine" spectrum is taken from
`Table 6.4`_ of the HST STIS Instrument Handbook, measured at 38 degrees from
Earth's limb.

This is relevant for satellites in Earth orbit (LEO, GEO, etc.) where
earthshine is a significant source of stray light.

.. _`Table 6.4`: https://hst-docs.stsci.edu/stisihb/chapter-6-exposure-time-calculations/6-6-tabular-sky-backgrounds

References
----------
.. [1] Prichard, L., Welty, D. and Jones, A., et al. 2022 "STIS Instrument
       Handbook," Version 21.0, (Baltimore: STScI)
"""

from importlib import resources

from astropy.table import QTable
from synphot import Empirical1D, SourceSpectrum

from .._core import BACKGROUND_SOLID_ANGLE
from . import data


class EarthshineBackground:
    """Earthshine sky background: sunlight reflected off Earth.

    This is the worst-case "High Earthshine" spectrum from the HST STIS
    Instrument Handbook [1]_, `Table 6.4`_. It is a constant spectrum with no
    dependence on observer or target geometry.

    .. _`Table 6.4`: https://hst-docs.stsci.edu/stisihb/chapter-6-exposure-time-calculations/6-6-tabular-sky-backgrounds

    References
    ----------
    .. [1] Prichard, L., Welty, D. and Jones, A., et al. 2022 "STIS Instrument
           Handbook," Version 21.0, (Baltimore: STScI)

    Examples
    --------

    >>> from astropy import units as u
    >>> from m4opt.synphot.background import EarthshineBackground
    >>> background = EarthshineBackground()
    >>> float(background(5000 * u.angstrom).value) > 0
    True

    """  # noqa: E501

    def __new__(cls):
        with (
            resources.files(data).joinpath("stis_earthshine_high.ecsv").open("rb") as f
        ):
            table = QTable.read(f, format="ascii.ecsv")
        return SourceSpectrum(
            Empirical1D,
            points=table["wavelength"],
            lookup_table=table["surface_brightness"] * BACKGROUND_SOLID_ANGLE,
        )
