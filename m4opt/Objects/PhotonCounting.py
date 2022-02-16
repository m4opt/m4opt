"""
PhotonCounting: functions for getting count rate based on Instrument 
and Spectrum (background and/or target)

"""

from astropy.modeling import CompoundModel
from synphot.units import PHOTLAM

# TODO: further generalize functions; currently based on dorado-sensitivity (for Dorado instrument)

# Hack needed to deal with CompoundModel background objects
# since I am not able to extend them with the integrate() function
def integrate_bandpass(background, bandpass, **kwargs):
    if isinstance(background, CompoundModel):
        keys = [bk for bk in background.submodel_names]
        return sum([(background[name].scale_factor*background[name].spectrum*bandpass.bandpass).integrate(**kwargs) for name in keys])
    else: #could check isinstance(background, Background) in future
        return (background.scale_factor*background.spectrum*bandpass.bandpass).integrate(**kwargs)

def get_count_rate(bkg, bandpass, instrument):
    flux = integrate_bandpass(bkg, bandpass, flux_unit=PHOTLAM)
    return flux * (instrument.area/ u.ph)

