from astropy import units as u
from astropy.modeling import Model


class Background(Model):
    """ "Base class for sky background model spectra.

    A sky background model is a 1D spectral model that provides surface
    brightness as a function of frequency."""

    n_inputs = 1
    n_outputs = 1
    input_units = {"x": u.Hz}
    return_units = {"y": u.erg / u.Hz / u.cm**2 / u.s / u.sr}
    input_units_equivalencies = {"x": u.spectral()}
