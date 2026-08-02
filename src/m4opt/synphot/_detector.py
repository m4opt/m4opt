from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy.modeling import Model
from astropy.stats import signal_to_noise_oir_ccd
from synphot import SourceSpectrum

from ._math import countrate
from .background._core import BACKGROUND_SOLID_ANGLE


def exptime_oir_ccd(snr, source_eps, sky_eps, dark_eps, rd, npix, gain):
    """Inverse of :meth:`astropy.stats.signal_to_noise_oir_ccd`."""
    c1 = np.square(source_eps * gain)
    c2 = (source_eps * gain + npix * (sky_eps * gain + dark_eps)) * snr
    c3 = npix * np.square(rd)
    return 0.5 * snr * (c2 + np.sqrt(4 * c1 * c3 + np.square(c2))) / c1


def amplitude_oir_ccd(snr, t, source_eps, sky_eps, dark_eps, rd, npix, gain):
    """Inverse of :meth:`astropy.stats.signal_to_noise_oir_ccd`."""
    c1 = t * source_eps * gain
    c2 = npix * (t * (sky_eps * gain + dark_eps) + np.square(rd))
    return 0.5 * snr * (snr + np.sqrt(4 * c2 + np.square(snr))) / c1


@dataclass(repr=False)
class Detector:
    """Sensitivity calculator: compute SNR for exposure time or vice-versa."""

    plate_scale: u.physical.solid_angle
    """Solid angle per pixel."""

    area: u.Quantity[u.physical.area]
    """Effective collecting area."""

    bandpasses: dict[Hashable, Model]
    """Dictionary of 1D models mapping wavelength to dimensionless transmission."""

    background: SourceSpectrum
    """Background 1D model mapping wavelength to surface brightness."""

    dark_noise: u.Quantity[u.physical.frequency] = 0 * u.Hz
    """Dark noise count rate."""

    read_noise: float = 0.0
    """Number of noise counts due to readout."""

    npix: float = 4 * np.pi
    """Effective number of pixels in the photometry aperture.

    The default value of 4π is appropriate for a PSF that is critically sampled
    by the pixels :footcite:`2005MNRAS.361..861M`.

    References
    ----------
    .. footbibliography::
    """

    aperture_correction: float = 1.0
    """Fraction of the signal from a point source in the photometry aperture.

    The default value of 1 is appropriate for PSF photometry."""

    gain: float = 1.0
    """Detector gain."""

    def _get_count_rates(self, source_spectrum: Model, bandpass: Hashable | None):
        if bandpass is not None:
            bp = self.bandpasses[bandpass]
        elif len(self.bandpasses) == 1:
            (bp,) = self.bandpasses.values()
        else:
            raise ValueError(
                f"This instrument has more than one bandpass. Please specify one of them: {list(self.bandpasses.keys())}"
            )
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
            exptime, src, bkg, self.dark_noise, self.read_noise, self.npix, self.gain
        ).to_value(u.dimensionless_unscaled)

    def get_exptime(
        self, snr: float, source_spectrum: Model, bandpass: Hashable | None = None
    ):
        src, bkg = self._get_count_rates(source_spectrum, bandpass)
        return exptime_oir_ccd(
            snr, src, bkg, self.dark_noise, self.read_noise, self.npix, self.gain
        )

    def get_limmag(
        self,
        snr: float,
        exptime: u.Quantity[u.physical.time],
        source_spectrum: Model,
        bandpass: Hashable | None = None,
    ):
        """Get the limiting magnitude for a given SNR and exposure time.

        Note that the limiting magnitude is relative to the source spectrum,
        so you should pass a source spectrum that has an apparent magnitude of
        0.
        """
        src, bkg = self._get_count_rates(source_spectrum, bandpass)
        a = amplitude_oir_ccd(
            snr,
            exptime,
            src,
            bkg,
            self.dark_noise,
            self.read_noise,
            self.npix,
            self.gain,
        )
        return (a * u.dimensionless_unscaled).to(u.mag(u.dimensionless_unscaled))
