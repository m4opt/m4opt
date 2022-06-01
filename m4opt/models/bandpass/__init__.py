from .bandpass import Bandpass, tynt_filters

__all__ = ('Bandpass', 'tynt_filters',)

__doc__ = """
Bandpass models: models of instrument bandpasses.

Bandpass models are 1D models that take spectral frequency or wavelength
as input, and return a dimensionless attentuation factor as output.
"""
