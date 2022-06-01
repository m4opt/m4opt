from astropy.modeling import Model
from astropy import units as u


class Background(Model):
    """"Base class for bandpass model spectra.

    A bandpass model is a 1D spectral model that provides the bandpass
    attentuation as a function of frequency."""

    n_inputs = 1
    n_outputs = 1
    input_units = {'x': u.Hz}
    return_units = {'y': u.erg / u.Hz / u.cm ** 2 / u.s / u.sr}
    input_units_equivalencies = {'x': u.spectral()}
