"""
Instrument Object:
contains information about instrument

"""

from astropy import units as u

# default instrument readouts:
dorado = {
    "area" : 100.0 * u.cm**2,
    "plate_scale" : 25.0 * u.arcsec * u.pix**-1,
    "npix" : np.pi * 0.89**2,
    "aperture_correction" : 0.7,
    "dark_noise" : 4. * 30. * 0.124 * u.hour**-1,
    "read_noise" : 5.0
}

class Instrument:
    def __init__(self, area=None, plate_scale=None, npix=None, aperture_correction=None,
                    dark_noise = None, read_noise=None):
        
        self.area = area
        self.plate_scale = plate_scale
        self.npix = npix
        self.aperture_correction = aperture_correction
        self.dark_noise = dark_noise
        self.read_noise = read_noise        

    @classmethod
    def dorado(cls):
        return cls(**dorado)
