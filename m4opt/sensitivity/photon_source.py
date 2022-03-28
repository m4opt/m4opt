"""

Photon Source: Objects to hold source spectrum and calculate scale factors
for photon rate evaluation

"""

# dependencies
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.modeling import Model
from synphot.spectrum import SourceSpectrum
from astropy.table import QTable
import astropy.units as u
from synphot import Empirical1D, ConstFlux1D, GaussianFlux1D, PowerLawFlux1D
from synphot.units import PHOTLAM

# all backgrounds here based on dorado-sensitivity/backgrounds


# TODO: should we change default behavior of GalacticBackground
# so that it returns one object insteadgit of a CompoundModel?


# Photon Source (target or Background)
class PhotonSource(Model):
    """
    simple Photon Source Object

    Parameters
    ----------
    name : string
        Name of object.
        Used for accessing internal parameters in compound background model.
    spectrum : ``synphot.SourceSpectrum``
        Background Spectrum used for calculating source counts
    scale_factor : float
        Scaling for spectrum flux
    """

    # definitions required for astropy.Model
    n_inputs = 1  # wavelength
    n_outputs = 1  # flux

    def __init__(self, name="target", spectrum=None, scale_factor=1.):
        self.spectrum = spectrum
        self.scale_factor = scale_factor
        super().__init__()

        # goes after __init__() because reasons(?)
        self.name = name

    def valid(self):
        if self.spectrum is None:
            return False
        else:
            return True

    def set_spectrum(self, spectrum):
        self.spectrum = spectrum

    def set_scale_factor(self, scale_factor):
        self.scale_factor = scale_factor

    def evaluate(self, wavelength):
        if self.valid():
            return self.spectrum(wavelength)*self.scale_factor
        else:
            return RuntimeError("spectrum is not defined")


# Airglow Background
class AirglowBackground(PhotonSource):
    """
    Airglow Background Object

    Parameters
    ----------
    name : string
        Name of object.
        Used for accessing internal parameters in compound background model.
    spectrum : ``synphot.SourceSpectrum``
        Background Spectrum used for calculating source counts
        If not defined (i.e. 'None'), default airglow spectrum will be used.
    scale_factor : float
        Scale factor for airglow light; expressed as a ratio between
        desired value and set spectrum.
        See ``set_airglow_scale()`` for provided default function,
        or ``set_scale_factor()`` to set directly from user-provided value.

    Notes
    -----
    ``default()`` initializes to the daytime airglow spectrum.
    Here, ``set_airglow_scale()`` will provide correct scale factor based
    on whether it is daytime or not.
    Otherwise, if user-defined spectrum is provided, ``set_scale_factor()``
    should instead be used to directly set the appropriate scale factor.

    Examples
    --------
    TBD

    """
    # definitions required for astropy.Model
    n_inputs = 1  # wavelength
    n_outputs = 1  # flux

    def __init__(self, name="airglow", spectrum=None, scale_fac=1.):

        super().__init__(name, spectrum, scale_fac)

    @classmethod
    def default(cls):
        """Airglow spectrum in daytime, normalized to 1 square arcsecond."""
        default_airglow = SourceSpectrum(
            GaussianFlux1D,
            mean=2471 * u.angstrom,
            fwhm=0.023 * u.angstrom,
            total_flux=1.5e-15 * u.erg * u.s**-1 * u.cm**-2
            )
        return cls(spectrum=default_airglow)

    @classmethod
    def from_file(cls, path, name):
        table = QTable.read(path)

        spectrum = SourceSpectrum(
            Empirical1D,
            points=table['wavelength'],
            lookup_table=table['surface_brightness'] * u.arcsec**2
            )

        return cls(name=name, spectrum=spectrum)

    @classmethod
    def from_amplitude(cls, amp, name):
        spectrum = SourceSpectrum(
            ConstFlux1D,
            amplitude=amp * PHOTLAM * u.steradian**-1 * u.arcsec**2
            )
        return cls(name=name, spectrum=spectrum)

    def set_airglow_scale(self, night):
        """
        Reference
        ----------
        https://hst-docs.stsci.edu/stisihb/chapter-6-exposure-time-calculations/6-5-detector-and-sky-backgrounds
        """
        self.scale_factor = np.where(night, 1e-2, 1)


# Galactic light background
class GalacticBackground(PhotonSource):
    """
    Galactic Background Object

    Parameters
    ----------
    name : string
        Name of object.
        Used for accessing internal parameters in compound background model.
    spectrum : ``synphot.SourceSpectrum``
        Background Spectrum used for calculating source counts
        If not defined (i.e. 'None'), default galactic spectrum will be used.
    scale_factor : float
        Scale factor for galactic light; expressed as a ratio between
        desired value and set spectrum.
        See ``get_default_galactic_scale()`` for default scale function,
        or ``set_scale_factor()`` to setting directly from user-provided value.

    Notes
    -----
    ``default()`` estimates the Galactic diffuse emission based
    on the cosecant fits from Murthy (2014).
    In this case, there are two combined spectra and two scale factors,
    so the class returned by ``default()`` is actually a ``CompoundModel``,
    not a ``GalacticBackground``. However, ``get_default_galactic_scale()``
    will still provide correct default scale factors based on
    galactic coordinate.

    However, if user-defined spectrum is provided, ``set_scale_factor()``
    should instead be used to directly set the appropriate scale factor.

    Examples
    --------
    TBD

    """
    # definitions required for astropy.Model
    n_inputs = 1  # wavelength
    n_outputs = 1  # flux

    def __init__(self, name="galactic",
                 spectrum=None, default=0, scale_fac=1.):

        self.default = default  # used only in default case
        super().__init__(name, spectrum, scale_fac)

    @classmethod
    def default(cls):
        galactic1 = SourceSpectrum(
            ConstFlux1D,
            amplitude=1 * PHOTLAM * u.steradian**-1 * u.arcsec**2
            )

        galactic2 = SourceSpectrum(
            PowerLawFlux1D,
            x_0=1528*u.angstrom,
            alpha=-1,
            amplitude=1 * PHOTLAM * u.steradian**-1 * u.arcsec**2
            )

        # technically returns CompoundModel
        return (cls(name='galactic1', spectrum=galactic1, default=1) +
                cls(name='galactic2', spectrum=galactic2, default=2))

    @classmethod
    def from_file(cls, path, name):
        table = QTable.read(path)

        spectrum = SourceSpectrum(
            Empirical1D,
            points=table['wavelength'],
            lookup_table=table['surface_brightness'] * u.arcsec**2
            )

        return cls(name=name, spectrum=spectrum)

    @classmethod
    def from_amplitude(cls, amp, name):
        spectrum = SourceSpectrum(
            ConstFlux1D,
            amplitude=amp * PHOTLAM * u.steradian**-1 * u.arcsec**2
            )
        return cls(name=name, spectrum=spectrum)

    def get_default_galactic_scales(self, coord):
        """Get the Galactic diffuse emission, normalized to 1 square arcsecond.
        Estimate the Galactic diffuse emission based on the cosecant fits from
        Murthy (2014).
        Parameters
        ----------
        coord : astropy.coordinates.SkyCoord
            The coordinates of the object under observation.
        Returns
        -------
        synphot.SourceSpectrum
            The Galactic diffuse emission spectrum, normalized to 1 square
            arcsecond.
        References
        ----------
        https://doi.org/10.3847/1538-4357/aabcb9
        """

        # NOTE: only works with default GalacticBackground,
        # which returns compound model
        assert self.default == 1 or self.default == 2, (
            "Function only works with objects instantiated ",
            "from GalacticBackground.default()")

        b = SkyCoord(coord).galactic.b
        csc = 1 / np.sin(b)
        pos = (csc > 0)

        # Constants from Murthy (2014) Table 4.
        # Note that slopes for the Southern hemisphere have been negated
        # to cancel the minus sign in the Galactic latitude.
        fuv_a = np.where(pos, 93.4, -205.5)
        fuv_b = np.where(pos, 133.2, -401.8)

        fuv = fuv_a + fuv_b * csc

        if self.default == 1:
            return fuv
        else:
            nuv_a = np.where(pos, 257.5, 66.7)
            nuv_b = np.where(pos, 185.1, -356.3)
            nuv = nuv_a + nuv_b * csc

            # GALEX filter effective wavelengths in angstroms from
            # http://www.galex.caltech.edu/researcher/techdoc-ch1.html#3
            fuv_wave = 1528
            nuv_wave = 2271

            return (nuv - fuv) / (nuv_wave - fuv_wave)
