from .core import Background
from .zodiacal import ZodiacalBackground
from .skybright import SkyBackground

__all__ = ('ZodiacalBackground', 'SkyBackground',)

__doc__ = f"""
Sky background models: models of the surface brightness of the sky.

Sky background models are 1D models that take spectral frequency or wavelength
as input, and return the surface brightness in units of
{Background.return_units['y']} as output.
"""
