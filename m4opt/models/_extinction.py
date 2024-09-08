from functools import cache
from urllib.error import URLError

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.modeling import Model, custom_model
from astropy.modeling.models import Const1D
from astropy.utils.data import download_file
from dust_extinction.parameter_averages import G23
from dustmaps.planck import PlanckGNILCQuery

from ._core import state

axav = G23(Rv=3.1)
axav.input_units_equivalencies = {"x": u.spectral()}


def download_file_from_mirrors(mirror, *more_mirrors, **kwargs):
    try:
        return download_file(mirror, **kwargs)
    except URLError:
        if not more_mirrors:
            raise
        return download_file_from_mirrors(*more_mirrors, **kwargs)


@cache
def dust_map():
    return PlanckGNILCQuery(
        download_file_from_mirrors(
            "https://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=COM_CompMap_Dust-GNILC-Model-Opacity_2048_R2.01.fits",
            "https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/component-maps/foregrounds/COM_CompMap_Dust-GNILC-Model-Opacity_2048_R2.01.fits",
            cache=True,
        )
    )


@custom_model
def EbvScale(x):
    return dust_map().query(state.get().target_coord)


class Extinction:
    """Milky Way dust extinction.

    Notes
    -----
    We use :class:`dust_extinction.parameter_averages.G23` because of its broad
    wavelength coverage from ultraviolet to infrared.
    """

    def __new__(cls, Ebv=None):
        if Ebv is None:
            Ebv = EbvScale()
        if isinstance(Ebv, Model):
            factor = Const1D(-0.4 * axav.Rv) * Ebv
        else:
            factor = Const1D(-0.4 * axav.Rv * Ebv)
        return Const1D(10) ** (factor * axav)

    @classmethod
    def at(cls, target_coord: SkyCoord):
        return cls(Ebv=dust_map().query(target_coord))
