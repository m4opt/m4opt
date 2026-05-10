"""Cerenkov particle-induced background radiation model.

This module models the background due to Cerenkov radiation emitted by charged
particles (primarily electrons) from Earth's radiation belts passing through
telescope optics. It uses the NASA AE8/AP8 trapped particle model
:footcite:`2016PASP..128c5005K` to estimate the flux of charged particles.

The implementation follows the conservative noise modeling approach from the
ULTRASAT design :footcite:`2024ApJ...964...74S`.

This is a Python adaptation of the MATLAB ``Cerenkov`` function from the
`MAATv2 AstroPack <https://github.com/EranOfek/AstroPack>`_.

.. footbibliography::
"""

from typing import Literal, override

import numpy as np
from aep8 import flux as aep8_flux
from astropy import units as u
from astropy.constants import alpha, c, m_e, m_p
from astropy.coordinates import EarthLocation
from astropy.time import Time
from scipy.interpolate import CubicSpline
from synphot import Empirical1D, SourceSpectrum, SpectralElement

from ..._extrinsic import ExtrinsicScaleFactor
from .._core import BACKGROUND_SOLID_ANGLE
from ._electron_loss import get_electron_energy_loss
from ._refraction_index import REFRACTION_INDEX

# Reference location: geostationary orbit at lon=15°, lat=0°, height=35786 km
_REFERENCE_LOCATION = EarthLocation.from_geodetic(
    lon=15 * u.deg, lat=0 * u.deg, height=35786 * u.km
)
_REFERENCE_OBSTIME = Time("2025-05-18T02:48:00Z")
_REFERENCE_ENERGY = 1.0 * u.MeV

# Material optical parameters: (scalar refractive index, density)
_MATERIAL_PROPERTIES = {
    "sio2": (1.5, 2.2 * u.g / u.cm**3),
    "sio2_suprasil_2a": (1.5, 2.2 * u.g / u.cm**3),
    "sapphire": (1.75, 4.0 * u.g / u.cm**3),
}


class CerenkovScaleFactor(ExtrinsicScaleFactor):
    """Scale factor for Cerenkov background based on radiation belt flux.

    The scale factor is the ratio of the AE8 integral flux at the observer's
    location to the flux at a reference geostationary orbit location (at 1 MeV).

    Parameters
    ----------
    particle : {'e', 'p'}
        Particle type.
    solar : {'max', 'min'}
        Solar activity condition.
    """

    def __init__(self, particle="e", solar="max", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._particle = particle
        self._solar = solar
        self._reference_flux = aep8_flux(
            _REFERENCE_LOCATION,
            _REFERENCE_OBSTIME,
            _REFERENCE_ENERGY,
            kind="integral",
            particle=particle,
            solar=solar,
        )

    @override
    def at(self, observer_location, target_coord, obstime):
        current_flux = aep8_flux(
            observer_location,
            obstime,
            _REFERENCE_ENERGY,
            kind="integral",
            particle=self._particle,
            solar=self._solar,
        )
        ref = self._reference_flux
        # Guard against zero reference flux
        if ref == 0:
            return 0.0
        return current_flux / ref


class CerenkovBackground:
    """Cerenkov particle-induced background radiation.

    This model estimates the background due to Cerenkov radiation from charged
    particles (electrons or protons) in Earth's radiation belts interacting
    with telescope optics.

    The spectral shape is computed at a reference geostationary orbit location,
    and then scaled by a :class:`CerenkovScaleFactor` that adjusts the
    amplitude based on the actual radiation belt flux at the observer's
    location via the AE8/AP8 model.

    Parameters
    ----------
    factor : float
        Geometric suppression factor accounting for baffle suppression and
        shielding (default: 21 for ULTRASAT).
    particle : {'e', 'p'}
        Particle type (default: ``'e'``).
    solar : {'max', 'min'}
        Solar activity condition (default: ``'max'``).

    Examples
    --------

    Create a Cerenkov background model for ULTRASAT:

    >>> from m4opt.synphot.background import CerenkovBackground
    >>> background = CerenkovBackground(factor=21)

    .. plot::
        :caption: Cerenkov background reference spectrum at GEO

        from matplotlib import pyplot as plt
        import numpy as np
        from astropy import units as u
        from astropy.visualization import quantity_support

        from m4opt.synphot.background._cerenkov import CerenkovBackground

        quantity_support()

        wave = np.linspace(1862, 12000) * u.angstrom
        spec = CerenkovBackground.reference(factor=21)
        ax = plt.axes()
        ax.plot(wave, spec(wave))
        ax.set_title('Cerenkov Background (GEO reference)')

    .. plot::
        :caption: Electron energy loss (dE/dX) vs. kinetic energy

        import matplotlib.pyplot as plt
        from m4opt.synphot.background._cerenkov._electron_loss import get_electron_energy_loss

        Ek, dEdX_SiO2 = get_electron_energy_loss(material="sio2")
        _, dEdX_Al2O3 = get_electron_energy_loss(material="sapphire")

        fig, ax = plt.subplots()
        ax.loglog(Ek.value, dEdX_SiO2.value, label=r"Silica (SiO$_2$)")
        ax.loglog(Ek.value, dEdX_Al2O3.value, label=r"Sapphire (Al$_2$O$_3$)")
        ax.set_xlabel("Electron kinetic energy [MeV]")
        ax.set_ylabel(r"Energy loss dE/dX [MeV/(g cm$^{-2}$)]")
        ax.legend()
        ax.grid(True, which='both', ls='--')

    .. plot::
        :caption: Refractive index of Suprasil 2A vs. wavelength

        import matplotlib.pyplot as plt
        from m4opt.synphot.background._cerenkov._refraction_index import REFRACTION_INDEX

        L, n, _ = REFRACTION_INDEX["sio2_suprasil_2a"]()

        fig, ax = plt.subplots()
        ax.plot(L.value, n)
        ax.set_xlabel(r"Wavelength [$\\AA$]")
        ax.set_ylabel("Refractive index")
        ax.grid(True)

    References
    ----------
    .. footbibliography::
    """

    def __new__(
        cls,
        factor: float = 21,
        particle: Literal["e", "p"] = "e",
        solar: Literal["max", "min"] = "max",
    ):
        return cls.reference(
            factor=factor, particle=particle, solar=solar
        ) * SpectralElement(CerenkovScaleFactor(particle=particle, solar=solar))

    @staticmethod
    def reference(
        factor: float = 21,
        particle: Literal["e", "p"] = "e",
        solar: Literal["max", "min"] = "max",
        material: str = "sio2_suprasil_2a",
        energy: tuple[u.Quantity, u.Quantity] = (0.05 * u.MeV, 8.5 * u.MeV),
        nbins: int = 1000,
    ) -> SourceSpectrum:
        """Cerenkov background spectrum at the reference GEO location.

        Parameters
        ----------
        factor : float
            Geometric suppression factor (default: 21).
        particle : {'e', 'p'}
            Particle type (default: ``'e'``).
        solar : {'max', 'min'}
            Solar activity condition (default: ``'max'``).
        material : str
            Optical material (default: ``'sio2_suprasil_2a'``).
        energy : tuple
            (min, max) energy range.
        nbins : int
            Number of energy bins.

        Returns
        -------
        `~synphot.SourceSpectrum`
            Reference Cerenkov emission spectrum.
        """
        # Build radiation belt flux table.
        # TODO: open issue at https://github.com/m4opt/aep8 to vectorize
        # aep8.flux over energy so this loop is unnecessary.
        emin, emax = energy
        ee = np.geomspace(emin, emax, num=nbins)
        Fe = u.Quantity(
            [
                aep8_flux(
                    _REFERENCE_LOCATION,
                    _REFERENCE_OBSTIME,
                    e,
                    kind="integral",
                    solar=solar,
                    particle=particle,
                )
                for e in ee
            ]
        )

        # Zero flux: return zero spectrum
        if np.all(Fe.value == 0):
            Lam, _, _ = REFRACTION_INDEX[material]()
            zero_flux = np.zeros_like(Lam.value) * u.photon / (u.cm**2 * u.s * u.AA)
            return SourceSpectrum(Empirical1D, points=Lam, lookup_table=zero_flux)

        n_val, rho = _MATERIAL_PROPERTIES[material]

        # Midpoint energies
        em = 0.5 * (ee[:-1] + ee[1:])

        # Particle mass
        mass = m_e if particle == "e" else m_p
        mass_mev = (mass * c**2).to(u.MeV)

        # Lorentz gamma and beta at midpoints
        gamma = 1 + em / mass_mev
        beta = np.sqrt(1 - 1.0 / gamma**2)

        # Interpolate flux at midpoints
        cs_fm = CubicSpline(ee.value, Fe.value, bc_type="natural", extrapolate=True)  # type: ignore[attr-defined]
        Fm = u.Quantity(cs_fm(em.value), Fe.unit)  # type: ignore[attr-defined]

        # Cerenkov emission condition: n * beta > 1
        fC_energy = np.maximum(0, 1 - 1.0 / n_val**2 / beta**2)

        # Electron stopping power
        Ek, dEdX = get_electron_energy_loss(material)

        # Inverse energy loss
        cs_dEdX = CubicSpline(
            Ek.value, 1.0 / dEdX.value, bc_type="natural", extrapolate=True
        )
        gEE = u.Quantity(cs_dEdX(em.value), 1 / dEdX.unit)  # type: ignore[attr-defined]

        # Cerenkov emission integrand at midpoints
        intg = gEE * Fm * fC_energy
        cs_intg = CubicSpline(em.value, intg.value, bc_type="natural", extrapolate=True)  # type: ignore[attr-defined]

        # Wavelength-dependent refractive index
        Lam, n, _ = REFRACTION_INDEX[material]()

        # Emission factor for all (wavelength, energy) pairs
        fC_wavelength_energy = np.maximum(
            0, 1 - 1.0 / n[:, np.newaxis] ** 2 / beta[np.newaxis, :] ** 2
        )
        intg = gEE * Fm * fC_wavelength_energy

        # Normalization at 1 MeV
        cs_val = cs_intg(1.0) * intg.unit

        # Cerenkov emission: 2*pi*alpha / (rho * lambda^2)
        Lam_cm = Lam.to(u.cm)
        Lnorm = 2 * np.pi * alpha / rho / Lam_cm**2 * cs_val * u.photon
        Lnorm = Lnorm.to(u.photon / (u.MeV * u.s * u.cm**2 * u.micron))

        # Integrate over energy for each wavelength
        int_val = np.sum(intg * np.diff(ee)[np.newaxis, :], axis=1) / cs_val

        # Total Cerenkov emission per micron per area per time
        L1mu = int_val * Lnorm

        # Intensity per wavelength (divide by 2*pi*n^2)
        IC1mu = L1mu / (2 * np.pi * n**2) / u.sr

        # Convert to per-arcsec^2 per Angstrom
        intensity_angstrom = IC1mu.to(u.photon / u.cm**2 / u.s / u.sr / u.Angstrom)
        intensity_arcsec2 = intensity_angstrom.to(
            u.photon / u.cm**2 / u.s / BACKGROUND_SOLID_ANGLE / u.Angstrom
        )
        intensity_photlam = intensity_arcsec2 * BACKGROUND_SOLID_ANGLE

        return SourceSpectrum(
            Empirical1D, points=Lam, lookup_table=intensity_photlam / factor
        )
