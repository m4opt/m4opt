from .airmass import Airmass

__all__ = ('Airmass',)

__doc__ = f"""
Airmass models: models for light attentuation due to Earth's atmosphere.

Airmass models are defined at the observatory location using extinction tables,
and take the sky coordinate of the target as input, and return an attentuation
factor (dimensionless) as output.
"""