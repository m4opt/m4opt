"""
PhotonCounting: functions for getting count rate based on Instrument
and Spectrum (background and/or target)

"""

from astropy.modeling import CompoundModel
from synphot.units import PHOTLAM
import astropy.units as u
from PhotonSource import set_background_scales

# TODO: further generalize functions; currently based on dorado-sensitivity
# (for Dorado instrument)


# Hack needed to deal with CompoundModel background objects
# since I am not able to extend them with the integrate() function
def integrate_bandpass(background, bandpass, **kwargs):
    if isinstance(background, CompoundModel):
        return sum([(background[name].scale_factor*background[name].spectrum *
                    bandpass.bandpass).integrate(**kwargs)
                    for name in background.submodel_names])
    else:  # TODO: could check isinstance(background, Background) in future
        return (background.scale_factor *
                background.spectrum * bandpass.bandpass).integrate(**kwargs)


def get_count_rate(source, bandpass, instrument):
    flux = integrate_bandpass(source, bandpass, flux_unit=PHOTLAM)
    return flux * (instrument.area / u.ph)


def get_background_count_rate(background, bandpass, instrument, **kwargs):
    # set scale factors
    set_background_scales(background, **kwargs)

    return get_count_rate(background, bandpass, instrument)
