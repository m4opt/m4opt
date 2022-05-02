from astropy.modeling import Model
from astropy import units as u


class BaseExtinction(Model):
    """"Base class for extinction Model.

    This model provides a dimensionless parameter for the
    attentuation of light due to extinction
    as a function of frequency."""

    n_inputs = 1
    n_outputs = 1
    input_units = {'x': u.Hz}
    return_units = {'y': u.dimensionless_unscaled}
    input_units_equivalencies = {'x': u.spectral()}
