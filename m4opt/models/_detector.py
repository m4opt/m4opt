from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy.modeling import Model
from astropy.stats import signal_to_noise_oir_ccd
from synphot import SourceSpectrum, SpectralElement

from ._math import countrate
from .background._core import BACKGROUND_SOLID_ANGLE


def exptime_oir_ccd(snr, source_eps, sky_eps, dark_eps, rd, npix, gain=1.0):
    """Inverse of :meth:`astropy.stats.signal_to_noise_oir_ccd`."""
    c1 = np.square(source_eps * gain)
    c2 = (source_eps * gain + npix * (sky_eps * gain + dark_eps)) * snr
    c3 = npix * np.square(rd)
    return 0.5 * snr * (c2 + np.sqrt(4 * c1 * c3 + np.square(c2))) / c1


@dataclass
class Detector:
    """Sensitivity calculator: compute SNR for exposure time or vice-versa."""

    npix: float
    """Effective number of pixels in the photometry aperture."""

    plate_scale: u.physical.solid_angle
    """Solid angle per pixel."""

    dark_noise: u.Quantity[u.physical.frequency]
    """Dark noise count rate."""

    read_noise: float
    """Number of noise counts due to readout."""

    area: u.Quantity[u.physical.area]
    """Effective collecting area."""

    bandpasses: dict[Hashable, Model]
    """Filter bandpasses: dictionary of 1D models mapping wavelength to dimensionless transmission."""

    background: SourceSpectrum
    """Background: 1D model mapping wavelength to surface brightness."""

    aperture_correction: float = 1.0
    """Fraction of the signal from a point source falls within the aperture."""

    extinction: SpectralElement | None = None
    """Extinction: 1D model mapping wavelength to dimensionless attenuation of source."""

    def _get_count_rates(self, source_spectrum: Model, bandpass: Hashable | None):
        if bandpass is not None:
            bp = self.bandpasses[bandpass]
        elif len(self.bandpasses) == 1:
            (bp,) = self.bandpasses.values()
        else:
            raise ValueError(
                f"This instrument has more than one bandpass. Please specify one of them: {self.bandpasses.keys()}"
            )
        if self.extinction is not None:
            source_spectrum = source_spectrum * self.extinction
        src_count_rate = (
            self.aperture_correction * self.area * countrate(source_spectrum, bp)
        )
        bkg_count_rate = (
            self.area
            * self.plate_scale
            / BACKGROUND_SOLID_ANGLE
            * countrate(self.background, bp)
        )
        return src_count_rate, bkg_count_rate

    def get_snr(
        self,
        exptime: u.Quantity[u.physical.time],
        source_spectrum: Model,
        bandpass: Hashable | None = None,
    ):
        src, bkg = self._get_count_rates(source_spectrum, bandpass)
        return signal_to_noise_oir_ccd(
            exptime,
            src,
            bkg,
            self.dark_noise,
            self.read_noise,
            self.npix,
        ).to_value(u.dimensionless_unscaled)

    def get_exptime(
        self, snr: float, source_spectrum: Model, bandpass: Hashable | None = None
    ):
        src, bkg = self._get_count_rates(source_spectrum, bandpass)
        return exptime_oir_ccd(
            snr,
            src,
            bkg,
            self.dark_noise,
            self.read_noise,
            self.npix,
        )
