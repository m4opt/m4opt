from dataclasses import dataclass
from typing import Literal

import numpy as np
from aep8 import flux
from astropy import units as u
from astropy.constants import c, h, m_e, m_p
from astropy.table import Table
from scipy.interpolate import CubicSpline
from synphot import Empirical1D, SourceSpectrum

from .._core import BACKGROUND_SOLID_ANGLE
from ._electron_loss import get_electron_energy_loss
from ._refraction_index import get_refraction_index


def radiation_belt_flux_table(
    observer_location,
    obstime,
    energy: tuple[u.Quantity, u.Quantity] = (0.05 * u.MeV, 8.5 * u.MeV),
    nbins: int = 20,
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


@dataclass
class CerenkovEmission:
    material: str = "SiO2_suprasil_2a"
    particle: Literal["e", "p"] = "e"
    solar: Literal["max", "min"] = "max"
    energy: tuple[u.Quantity, u.Quantity] = (0.05 * u.MeV, 8.5 * u.MeV)
    nbins: int = 20

    def emission(self, observer_location, obstime) -> tuple[u.Quantity, u.Quantity]:
        """
        Calculate the Cerenkov radiation intensity for the given conditions.
        """

        # --- Retrieve Radiation belt electron flux data (AE8 model)
        flux_data = radiation_belt_flux_table(
            observer_location,
            obstime,
            energy=self.energy,
            nbins=self.nbins,
            particle=self.particle,
            solar=self.solar,
        )

        # --- Material optical parameters: index of refraction and density
        material_properties = {
            "sio2": (1.5, 2.2 * u.g / u.cm**3),
            "SiO2_suprasil_2a": (1.5, 2.2 * u.g / u.cm**3),
            "sapphire": (1.75, 4.0 * u.g / u.cm**3),  # at 1 mu, n(0.25 mu)=1.85
        }
        if self.material not in material_properties:
            raise ValueError(f"Unknown material option: '{self.material}'")

        n_val, rho = material_properties[self.material]

        # Energy grid for interpolation (MeV)
        # The read off figure  6 of Kruk  et al. (2016), https://iopscience.iop.org/article/10.1088/1538-3873/128/961/035005.
        ee = np.logspace(np.log10(0.04), np.log10(8.0), 1000) * u.MeV
        cs_flux = CubicSpline(
            flux_data["energy"].value,
            flux_data["flux"].value,
            bc_type="natural",
            extrapolate=True,
        )
        Fe = u.Quantity(cs_flux(ee.value), flux_data["flux"].unit)

        # Compute midpoint energies between bins
        em = 0.5 * (ee[:-1] + ee[1:])

        # Get rest mass of particle and convert to MeV
        mass = m_e if self.particle == "e" else m_p
        mass_mev = (mass * c**2).to("MeV")

        # Compute Lorentz gamma and beta (v/c)
        gamma = 1 + em / mass_mev
        beta = np.sqrt(1 - 1.0 / gamma**2)

        # Interpolate flux at midpoints
        cs_fm = CubicSpline(ee.value, Fe.value, bc_type="natural", extrapolate=True)
        Fm = u.Quantity(cs_fm(em.value), Fe.unit)

        # Cerenkov emission condition: n * beta > 1
        fC = np.maximum(0, 1 - 1.0 / n_val**2 / beta**2)

        # Get inverse energy loss (1/dE/dX) from stopping power
        Ek, dEdX = get_electron_energy_loss(self.material)
        cs_dEdX = CubicSpline(
            Ek.value, 1.0 / dEdX.value, bc_type="natural", extrapolate=True
        )
        gEE = u.Quantity(cs_dEdX(em.value), 1 / dEdX.unit)

        # Cerenkov emission integrand: flux × path length × emission factor
        intg = gEE * Fm * fC
        cs_intg = CubicSpline(em.value, intg.value, bc_type="natural", extrapolate=True)

        # Wavelength-dependent refractive index and emission
        Lam, n, _ = get_refraction_index(self.material)
        Nn = len(n)
        IC1mu = np.empty(Nn, dtype=object)

        for i in range(Nn):
            gamma_i = 1 + em / mass_mev
            beta_i = np.sqrt(1 - 1.0 / gamma_i**2)
            fCi = np.maximum(0, 1 - 1.0 / n[i] ** 2 / beta_i**2)
            intg_i = gEE * Fm * fCi
            cs_val = cs_intg(1.0) * intg.unit
            Lnorm = 2 * np.pi / 137 / rho / Lam[i].to(u.cm) ** 2 * cs_val
            Lnorm = Lnorm.to(1 / (u.MeV * u.s * u.cm**2 * u.um))
            int_val = np.sum(intg_i * np.diff(ee)) / cs_val
            L1mu = int_val * Lnorm
            IC1mu[i] = L1mu.value / (2 * np.pi * n[i] ** 2)

        # Convert intensity to arcseconds squared
        # FIXME: does u.count = u.photon ?
        intensity_unit = u.photon / u.cm**2 / u.s / u.sr / u.micron
        intensity_micron = IC1mu * intensity_unit
        intensity_angstrom = intensity_micron.to(
            u.photon / u.cm**2 / u.s / u.sr / u.Angstrom
        )

        # 'ph/A' : [u.ph/u.s/u.cm**2/u.AA]
        intensity_arcsec = intensity_angstrom * (BACKGROUND_SOLID_ANGLE).to(u.sr)

        wavelength = Lam
        freq = (c / (wavelength)).to(u.Hz)

        # 'cgs/Hz': [erg/s/cm^2/Hz]
        intensity_erg = (intensity_arcsec / u.photon) * h.to(u.erg * u.s) * freq
        return SourceSpectrum(
            Empirical1D, points=wavelength, lookup_table=intensity_erg
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


    The integral flux of electrons in the Earth's radiation belts
    at a given observer location and time, using the AE8 model.

    .. plot::
        :caption: AE8 radiation belt electron integral flux .

        from astropy.coordinates import EarthLocation
        from astropy.time import Time
        from astropy import units as u
        import matplotlib.pyplot as plt
        from m4opt.synphot.background._cerenkov import radiation_belt_flux_table

        observer_location = EarthLocation.from_geodetic(lon=15 * u.deg, lat=0 * u.deg, height=35786 * u.km)
        obstime = Time("2025-05-18T02:48:00Z")
        tbl = radiation_belt_flux_table(observer_location, obstime)
        plt.figure(figsize=(7,5))
        plt.plot(tbl["energy"], tbl["flux"])
        plt.xlabel("Energy [MeV]")
        plt.ylabel("Integral flux [cm$^{-2}$ s$^{-1}$]")
        plt.title("AE8 Radiation Belt Electron Flux")
        plt.tight_layout()
        plt.show()

    The energy loss per unit mass thickness (dE/dX) of electrons
    in selected materials as a function of kinetic energy.

    .. plot::
        :caption: Electron energy loss per unit mass thickness (dE/dX) versus kinetic energy for several astrophysical
        and detector materials (log-log scale).

        import matplotlib.pyplot as plt
        from m4opt.synphot.background._cerenkov import get_electron_energy_loss

        lw = 1
        label_fs = 12
        tick_fs = 10

        # Compute energy loss for each material
        Ek, dEdX_O = get_electron_energy_loss(material="oxygen")
        _, dEdX_Si = get_electron_energy_loss(material="silicon")
        _, dEdX_Al = get_electron_energy_loss(material="aluminum")
        _, dEdX_SiO2 = get_electron_energy_loss(material="sio2")
        _, dEdX_Al2O3 = get_electron_energy_loss(material="sapphire")

        plt.figure(figsize=(7, 5))
        plt.loglog(Ek.value, dEdX_O.value, label=r"Oxygen (O)", linewidth=lw)
        plt.loglog(Ek.value, dEdX_Si.value, label=r"Silicon (Si)", linewidth=lw)
        plt.loglog(Ek.value, dEdX_Al.value, label=r"Aluminum (Al)", linewidth=lw)
        plt.loglog(Ek.value, dEdX_SiO2.value, label=r"Silica (SiO$_2$)", linewidth=lw)
        plt.loglog(Ek.value, dEdX_Al2O3.value, label=r"Sapphire (Al$_2$O$_3$)", linewidth=lw)

        plt.xlabel("Electron kinetic energy [MeV]", fontsize=label_fs)
        plt.ylabel(r"Energy loss dE/dX [MeV/(g cm$^{-2}$)]", fontsize=label_fs)
        plt.title("Electron Energy Loss per Unit Mass Thickness", fontsize=label_fs)
        plt.legend(fontsize=label_fs)
        plt.grid(True, which='both', ls='--')
        plt.tick_params(labelsize=tick_fs)
        plt.tight_layout()
        plt.show()


    .. plot::
        :caption:  Inverse energy loss for silica (log-log scale)

        import matplotlib.pyplot as plt
        from m4opt.synphot.background._cerenkov import get_electron_energy_loss

        Ek, dEdX_SiO2 = get_electron_energy_loss(material="sio2")

        plt.figure(figsize=(7, 5))
        plt.loglog(Ek.value, 1 / dEdX_SiO2.value, linewidth=2, color='tab:green')
        plt.xlabel("Electron kinetic energy [MeV]")
        plt.ylabel(r"Inverse energy loss $(dE/dX)^{-1}$ [(g cm$^{-2}$)/MeV]")
        plt.title("Inverse Energy Loss for Silica (SiO$_2$)")
        plt.grid(True, which='both', ls='--')
        plt.axis([1e-2, 1e2, 1e-2, 1])
        plt.tick_params(labelsize=tick_fs)
        plt.tight_layout()
        plt.show()


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
    <Quantity 5.62717185e-07 PHOTLAM>

    .. plot::
        :caption: Cerenkov background spectrum for GEO at 2025-05-18T02:48:00Z (log-log scale)

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

        plt.loglog(wavelength, intensity)
        plt.xlabel(rf"Wavelength [{wavelength.unit}]")
        plt.ylabel(rf"Intensity [{intensity.unit} (erg / s cm$^{2}$ Hz)]")
        plt.title(r"Cerenkov Background Spectrum at GEO")
        plt.tight_layout()
        plt.grid()
        plt.show()
    """

    particle: Literal["e", "p"] = "e"
    solar: Literal["max", "min"] = "max"
    material: str = "SiO2_suprasil_2a"

    def cerenkov_emission(self, observer_location, obstime):
        """
        Computes the wavelength-dependent Cerenkov emission intensity.
        """
        emission_model = CerenkovEmission(
            material=self.material,
            particle=self.particle,
            solar=self.solar,
            energy=self.energy,
            nbins=self.nbins,
        )
        return emission_model.emission(observer_location, obstime)
