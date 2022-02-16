"""
Bandpass Object: holds bandpass information

"""

#dependencies
from astropy.modeling import Model

# TODO: Should we merge this with PhotonSource? it has essentially the same features
# except it uses Spectral Element instead of SourceSpectrum

class Bandpass(Model):
    """
    simple Bandpass Object

    Parameters
    ----------
    name : string
        Name of object. Used for accessing internal parameters in compound background model.
    bandpass : ``synphot.SpectralElement`` 
    """

    # definitions required for astropy.Model
    n_inputs = 1 # wavelength
    n_outputs = 1 # scaling factor

    def __init__(self, name="bandpass", bandpass = None):
        self.bandpass = bandpass
        super().__init__()

        #goes after __init__() because reasons(?)
        self.name = name

    def valid(self):
        if self.bandpass is None:
            return False
        else:
            return True

    def set_bandpass(self, bandpass):
        self.bandpass = bandpass

    def evaluate(self, wavelength):
        if self.valid():
            return self.bandpass(wavelength)
        else:
            return RuntimeError("bandpass is not defined")