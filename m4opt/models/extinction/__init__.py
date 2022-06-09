"""
Extinction models: models for light attentuation due to Earth's atmosphere.
Includes Airmass Object for airmass calculation.

Extinction models are defined at the observatory location and with an
extinction table. They take the sky coordinate of the target as input,
and return an attentuation factor (dimensionless) as output.

Airmass models are defined at the observatory location and return the airmass
for a given target in the sky.
"""

from .airmass import Airmass, AtmosphericExtinction

__all__ = ('Airmass', 'AtmosphericExtinction',)
