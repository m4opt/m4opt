from dataclasses import dataclass
from typing import Literal

import numpy as np
from aep8 import flux
from astropy import units as u
from astropy.constants import alpha, c, m_e, m_p
from astropy.table import Table
from scipy.interpolate import CubicSpline
from synphot import Empirical1D, SourceSpectrum

# from synphot import Empirical1D, SourceSpectrum
from .._core import BACKGROUND_SOLID_ANGLE
from ._electron_loss import get_electron_energy_loss
from ._refraction_index import get_refraction_index


def radiation_belt_flux_table(
    observer_location,
    obstime,
    energy: tuple[u.Quantity, u.Quantity] = (0.05 * u.MeV, 8.5 * u.MeV),
    nbins: int = 1000,
    particle: Literal["e", "p"] = "e",
    solar: Literal["max", "min"] = "max",
) -> Table:
    """
    Returns a table of integral flux values from Earth's radiation belts.
    """
    emin, emax = energy
    energy_bins = np.geomspace(emin, emax, num=nbins)
    flux_integral = [
        flux(
            observer_location,
            obstime,
            e,
            kind="integral",
            solar=solar,
            particle=particle,
        )
        for e in energy_bins
    ]
    return Table([energy_bins, u.Quantity(flux_integral)], names=["energy", "flux"])


def cerenkov_emission_core(
    energy_grid: u.Quantity,
    flux_grid: u.Quantity,
    material: str = "SiO2_suprasil_2a",
    particle: Literal["e", "p"] = "e",
) -> SourceSpectrum:
    """Calculate the Cerenkov radiation intensity for the given conditions."""

    ee = energy_grid
    Fe = flux_grid

    # --- Material optical parameters: index of refraction and density
    # Dict: material name -> (refractive index at ~1 micron, density)
    material_properties = {
        "sio2": (1.5, 2.2 * u.g / u.cm**3),
        "SiO2_suprasil_2a": (1.5, 2.2 * u.g / u.cm**3),
        "sapphire": (1.75, 4.0 * u.g / u.cm**3),
    }
    try:
        n_val, rho = material_properties[material]
    except KeyError:
        raise ValueError(f"Unknown material option: '{material}'")

    # Compute midpoint energies between grid bins
    em = 0.5 * (ee[:-1] + ee[1:])

    # Get particle mass (rest energy) in MeV
    mass = m_e if particle == "e" else m_p
    mass_mev = (mass * c**2).to(u.MeV)

    # Calculate Lorentz gamma [1 + E/(m*c^2)] and beta [v/c] at midpoints
    gamma = 1 + em / mass_mev
    beta = np.sqrt(1 - 1.0 / gamma**2)

    # Interpolate flux at midpoints
    cs_fm = CubicSpline(ee.value, Fe.value, bc_type="natural", extrapolate=True)
    Fm = u.Quantity(cs_fm(em.value), Fe.unit)

    # --- Cerenkov emission condition: n * beta > 1
    ## fC: emission factor, zero if below threshold
    fC_energy = np.maximum(0, 1 - 1.0 / n_val**2 / beta**2)

    # --- Retrieve electron stopping power data for the material
    # Ek: energies (MeV), dEdX: stopping power (MeV/(g cm^-2))
    Ek, dEdX = get_electron_energy_loss(material)

    # Inverse energy loss function (1 / dEdX)
    cs_dEdX = CubicSpline(
        Ek.value, 1.0 / dEdX.value, bc_type="natural", extrapolate=True
    )
    gEE = u.Quantity(cs_dEdX(em.value), 1 / dEdX.unit)

    # Cerenkov emission integrand at midpoints: flux × path length × emission factor
    intg = gEE * Fm * fC_energy
    cs_intg = CubicSpline(em.value, intg.value, bc_type="natural", extrapolate=True)

    # Wavelength-dependent refractive index and emission
    # Lam: wavelength grid (Amstrong), n: refractive index at Lam
    Lam, n, _ = get_refraction_index(material)

    # Calculate emission factor for all (wavelength, energy) pairs
    fC_wavelength_energy = np.maximum(
        0, 1 - 1.0 / n[:, np.newaxis] ** 2 / beta[np.newaxis, :] ** 2
    )
    intg = gEE * Fm * fC_wavelength_energy

    # Evaluate normalization using integrand spline at 1 MeV
    cs_val = cs_intg(1.0) * intg.unit

    # Fine-structure constant :  alpha = e^2 / (4 * np.pi * epsilon_0 * hbar * c)
    # 2 * pi * alpha  factor from fine structure constant (alpha), rho: density, Lam_cm: wavelength in cm
    Lam_cm = Lam.to(u.cm)
    Lnorm = 2 * np.pi * alpha / rho / Lam_cm**2 * cs_val * u.photon
    Lnorm = Lnorm.to(u.photon / (u.MeV * u.s * u.cm**2 * u.micron))

    # --- Integrate Cherenkov emission over energy for each wavelength
    # np.diff(ee): width of each energy bin (MeV)
    int_val = np.sum(intg * np.diff(ee)[np.newaxis, :], axis=1) / cs_val

    # Total Cherenkov emissionper micron per area per time (photon / (s micron cm^2)).
    L1mu = int_val * Lnorm

    # Cherenkov intensity per wavelength (divide by 2*pi*n^2, normalization)
    IC1mu = L1mu / (2 * np.pi * n**2) / u.sr

    # Convert intensity to arcseconds squared
    intensity_angstrom = IC1mu.to(u.photon / u.cm**2 / u.s / u.sr / u.Angstrom)

    # 'ph/A' : [ phothon / s / cm^2 / Angstrom / arcsec^2]
    intensity_arcsec2 = intensity_angstrom.to(
        u.photon / u.cm**2 / u.s / BACKGROUND_SOLID_ANGLE / u.Angstrom
    )
    intensity_photlam = intensity_arcsec2 * BACKGROUND_SOLID_ANGLE

    wavelength = Lam

    # 'cgs/Hz': [erg/s/cm^2/Hz]
    # FIXME :  in case we need to use another unit
    # freq = (c / (wavelength)).to(u.Hz)
    # intensity_erg = (intensity_photlam) / u.photon * h.to(u.erg * u.s) * freq

    return SourceSpectrum(
        Empirical1D, points=wavelength, lookup_table=intensity_photlam
    )


def cerenkov_emission(
    observer_location,
    obstime,
    material: str = "SiO2_suprasil_2a",
    particle: Literal["e", "p"] = "e",
    solar: Literal["max", "min"] = "max",
    energy: tuple[u.Quantity, u.Quantity] = (0.05 * u.MeV, 8.5 * u.MeV),
    nbins: int = 1000,
) -> SourceSpectrum:
    """Calculate the Cerenkov radiation intensity from flux AE8/production."""

    flux_data = radiation_belt_flux_table(
        observer_location, obstime, energy, nbins, particle, solar
    )
    energy_grid = flux_data["energy"]
    flux_grid = flux_data["flux"]
    return cerenkov_emission_core(
        energy_grid, flux_grid, material=material, particle=particle, nbins=nbins
    )


@dataclass
class CerenkovBackground:
    """
    Model for Cerenkov particle-induced background using the AEP8 flux model.

    This model estimates the background induced by Cerenkov radiation from charged particles (electrons or protons)
    interacting with telescope optics. It uses integral flux data from the NASA AE8/AP8 radiation belt model
    to calculate the intensity of radiation emitted by typical optical materials such as silica or suprasil,
    used in the ULTRASAT telescope lenses.

    Parameters
    ----------
    particle : {'e', 'p'}, optional
        Particle type ('e' for electrons, 'p' for protons), by default 'e'.
    solar : {'max', 'min'}, optional
        Solar activity condition, by default 'max'.
    material : str, optional : { "sio2", "SiO2_suprasil_2a"}
        Material type for optics, by default 'SiO2_suprasil_2a'.

    References
    ----------
    This module is a Python adaptation of the MATLAB `Cerenkov` function from the MAATv2 package:
    https://www.mathworks.com/matlabcentral/fileexchange/128984-astropack-maatv2

    It incorporates:
    - The AE8 trapped electron flux model from radiation belt :doc:`NASA AE8/AP8, IRBEM <irbem:api/radiation_models>`,
    constraining the flux of charged particles in Earth's radiation belts.
    - Stopping power and refractive index models :footcite:`2018PASP..130g5002S, 2019PASP..131e4504O`.
    - The conservative noise modeling approach from ULTRASAT :footcite:`2024ApJ...964...74S`.
    - The energy grid used for interpolation was read from Figure 6 :footcite:`2016PASP..128c5005K`.

    .. footbibliography::

    Examples
    --------
    Cenrenkov background spectrum for a given observer location and time.

    >>> from astropy.coordinates import EarthLocation
    >>> from astropy.time import Time
    >>> from astropy import units as u
    >>> from m4opt.synphot.background._cerenkov import CerenkovBackground
    >>> observer_location = EarthLocation.from_geodetic(lon=15 * u.deg, lat=0 * u.deg, height=35786 * u.km)
    >>> obstime = Time("2025-05-18T02:48:00Z")
    >>> background = CerenkovBackground(particle='e', solar='max', material='SiO2_suprasil_2a')
    >>> spectrum = background.cerenkov_emission(observer_location, obstime)
    >>> wavelength = spectrum.waveset
    >>> spectrum(wavelength[0])
    <Quantity 5.5374254e-07 PHOTLAM>

    .. plot::
        :include-source: False
        :caption: Cerenkov background spectrum for GEO at 2025-05-18T02:48:00Z

        from astropy.coordinates import EarthLocation, ICRS
        from astropy_healpix import HEALPix
        from astropy.time import Time
        from astropy import units as u
        import numpy as np
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = "Times New Roman"

        from m4opt.synphot.background._cerenkov import CerenkovBackground
        from m4opt.synphot import observing

        observer_location = EarthLocation.from_geodetic(
            lon=15 * u.deg, lat=0 * u.deg, height=35786 * u.km
        )
        obstime = Time("2025-05-18T02:48:00Z")
        hpx = HEALPix(nside=512, frame=ICRS())
        coord = hpx.healpix_to_skycoord(np.arange(hpx.npix))

        with observing(observer_location=observer_location, target_coord=coord, obstime=obstime):
            cerenkov_model = CerenkovBackground(particle='e', solar='max', material='SiO2_suprasil_2a')
            spectrum = cerenkov_model.cerenkov_emission(observer_location, obstime)
            wavelength = spectrum.waveset
            intensity = spectrum(wavelength)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(wavelength, intensity)
        ax.set_xlabel(rf"Wavelength [{wavelength.unit}]")
        ax.set_ylabel(rf"Intensity [{intensity.unit} (erg / s cm$^{{2}}$ Hz)]")
        ax.set_title(r"Cerenkov Background Spectrum at GEO")
        ax.grid()
        fig.tight_layout()

    The integral flux of electrons in the Earth's radiation belts
    at a given observer location and time, using the AE8 model.

    .. plot::
        :include-source: False
        :caption: AE8 radiation belt electron integral flux .

        from astropy.coordinates import EarthLocation
        from astropy.time import Time
        from astropy import units as u
        import matplotlib.pyplot as plt
        from m4opt.synphot.background._cerenkov import radiation_belt_flux_table

        observer_location = EarthLocation.from_geodetic(lon=15 * u.deg, lat=0 * u.deg, height=35786 * u.km)
        obstime = Time("2025-05-18T02:48:00Z")
        tbl = radiation_belt_flux_table(observer_location, obstime)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(tbl["energy"], tbl["flux"])
        ax.set_xlabel("Energy [MeV]")
        ax.set_ylabel(r"Integral flux [cm$^{-2}$ s$^{-1}$]")
        ax.set_title("AE8 Radiation Belt Electron Flux")
        fig.tight_layout()


    The energy loss per unit mass thickness (dE/dX) of electrons
    in selected materials as a function of kinetic energy.

    .. plot::
        :include-source: False
        :caption: Electron energy loss (dE/dX) vs. kinetic energy for various materials (log-log).

        import matplotlib.pyplot as plt
        from m4opt.synphot.background._cerenkov import get_electron_energy_loss

        lw = 1
        label_fs = 12
        tick_fs = 10

        Ek, dEdX_O = get_electron_energy_loss(material="oxygen")
        _, dEdX_Si = get_electron_energy_loss(material="silicon")
        _, dEdX_Al = get_electron_energy_loss(material="aluminum")
        _, dEdX_SiO2 = get_electron_energy_loss(material="sio2")
        _, dEdX_Al2O3 = get_electron_energy_loss(material="sapphire")

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog(Ek.value, dEdX_O.value, label="Oxygen (O)", linewidth=lw)
        ax.loglog(Ek.value, dEdX_Si.value, label="Silicon (Si)", linewidth=lw)
        ax.loglog(Ek.value, dEdX_Al.value, label="Aluminum (Al)", linewidth=lw)
        ax.loglog(Ek.value, dEdX_SiO2.value, label="Silica (SiO$_2$)", linewidth=lw)
        ax.loglog(Ek.value, dEdX_Al2O3.value, label="Sapphire (Al$_2$O$_3$)", linewidth=lw)

        ax.set_xlabel("Electron kinetic energy [MeV]", fontsize=label_fs)
        ax.set_ylabel(r"Energy loss dE/dX [MeV/(g cm$^{-2}$)]", fontsize=label_fs)
        ax.set_title(r"Electron Energy Loss per Unit Mass Thickness", fontsize=label_fs)
        ax.legend(fontsize=label_fs)
        ax.grid(True, which='both', ls='--')
        ax.tick_params(labelsize=tick_fs)
        fig.tight_layout()


    .. plot::
        :include-source: False
        :caption:  Inverse energy loss for silica (log-log scale).

        import matplotlib.pyplot as plt
        from m4opt.synphot.background._cerenkov import get_electron_energy_loss

        Ek, dEdX_SiO2 = get_electron_energy_loss(material="sio2")

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog(Ek.value, 1 / dEdX_SiO2.value, linewidth=2, color='tab:green')
        ax.set_xlabel(r"Electron kinetic energy [MeV]")
        ax.set_ylabel(r"Inverse energy loss $(dE/dX)^{-1}$ [(g cm$^{-2}$)/MeV]")
        ax.set_title(r"Inverse Energy Loss for Silica (SiO$_2$)")
        ax.grid(True, which='both', ls='--')
        ax.axis([1e-2, 1e2, 1e-2, 1])
        ax.tick_params(labelsize=10)
        fig.tight_layout()


    """

    particle: Literal["e", "p"] = "e"
    solar: Literal["max", "min"] = "max"
    material: str = "SiO2_suprasil_2a"
    energy: tuple[u.Quantity, u.Quantity] = (0.05 * u.MeV, 8.5 * u.MeV)
    nbins: int = 1000

    def cerenkov_emission(self, observer_location, obstime):
        """
        Compute the Cerenkov emission spectrum for the given observer and time.

        Parameters
        ----------
        observer_location : EarthLocation
            Observer location, e.g., in geodetic coordinates.
        obstime : Time
            Observation time.

        Returns
        -------
        SourceSpectrum
            The computed Cerenkov background spectrum as a synphot SourceSpectrum.
        """
        return cerenkov_emission(
            observer_location=observer_location,
            obstime=obstime,
            material=self.material,
            particle=self.particle,
            solar=self.solar,
            energy=self.energy,
            nbins=self.nbins,
        )
