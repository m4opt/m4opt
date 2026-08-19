"""
Spectral modeling infrastructure for generating synthetic photometry.
"""

__all__ = [
    "core",
    "lightcurves",
    "spectra",
    "supernovae",
    "tdes",
]

# Imports from the core submodule.
from . import core
from .core import *

__all__.extend(core.__all__)

# Imports from the spectra submodule.
from . import spectra
from .spectra import *

__all__.extend(spectra.__all__)

# Imports from the lightcurve submodule.
from . import lightcurves
from .lightcurves import *

__all__.extend(lightcurves.__all__)

# Imports from the supernovae submodule.
from . import supernovae
from .supernovae import *

__all__.extend(supernovae.__all__)

# Imports from the tdes submodule.
from . import tdes
from .tdes import *

__all__.extend(tdes.__all__)
