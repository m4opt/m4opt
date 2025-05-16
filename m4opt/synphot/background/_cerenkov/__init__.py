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


@dataclass
class RadiationBelt:
    energy: tuple[u.Quantity, u.Quantity] = (0.05 * u.MeV, 8.5 * u.MeV)
    nbins: int = 20
    particle: Literal["e", "p"] = "e"
    solar: Literal["max", "min"] = "max"

    def flux_table(self, observer_location, obstime) -> Table:
        emin, emax = self.energy
        energy_bins = np.geomspace(emin, emax, num=self.nbins)
        flux_integral = [
            flux(
                observer_location,
                obstime,
                e,
                kind="integral",
                solar=self.solar,
                particle=self.particle,
            )
            for e in energy_bins
        ]
        return Table([energy_bins, u.Quantity(flux_integral)], names=["energy", "flux"])


@dataclass
class CerenkovEmission:
    material: str = "si02_suprasil_2a"
    particle: Literal["e", "p"] = "e"
    solar: Literal["max", "min"] = "max"
    energy: tuple[u.Quantity, u.Quantity] = (0.05 * u.MeV, 8.5 * u.MeV)

    def emission(self, observer_location, obstime) -> tuple[u.Quantity, u.Quantity]:
        """
        Calculate the Cerenkov radiation intensity for the given conditions.
        """
        # --- Material optical parameters: index of refraction and density
        material_properties = {
            "silica": (1.5, 2.2 * u.g / u.cm**3),
            "sio2": (1.5, 2.2 * u.g / u.cm**3),
            "si02_suprasil_2a": (1.5, 2.2 * u.g / u.cm**3),
            "sapphire": (1.75, 4.0 * u.g / u.cm**3),  # at 1 mu, n(0.25 mu)=1.85
        }
        if self.material not in material_properties:
            raise ValueError(f"Unknown material option: '{self.material}'")

        n_val, rho = material_properties[self.material]

        # Retrieve electron flux data (AE8 model)
        rb = RadiationBelt(energy=self.energy, particle=self.particle, solar=self.solar)
        flux_data = rb.flux_table(observer_location, obstime)

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
            IC1mu[i] = L1mu / (2 * np.pi * n[i] ** 2)

        IC1mu = u.Quantity(
            [q.value if hasattr(q, "value") else q for q in IC1mu], unit=L1mu.unit
        )

        # Convert intensity to arcseconds squared
        # FIXME: does u.count = u.photon ?
        intensity_micron = IC1mu.value * (u.photon / u.cm**2 / u.s / u.sr / u.micron)
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
    material : str, optional
        Material type for optics, by default 'si02_suprasil_2a'.

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
    >>> from astropy.coordinates import EarthLocation
    >>> from astropy.time import Time
    >>> from astropy import units as u
    >>> from m4opt.synphot.background._cerenkov import CerenkovBackground
    >>> observer_location = EarthLocation.from_geodetic(lon=15 * u.deg, lat=0 * u.deg, height=35786 * u.km)
    >>> obstime = Time("2025-05-18T02:48:00Z")
    >>> cerenkov_model = CerenkovBackground(particle='e', solar='max', material='si02_suprasil_2a')
    >>> spectrum = cerenkov_model.cerenkov_emission(observer_location, obstime)
    >>> wavelength = spectrum.waveset
    >>> spectrum(wavelength[0])
    <Quantity 5.62717185e-07 PHOTLAM>

    .. plot::
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
            cerenkov_model = CerenkovBackground(particle='e', solar='max', material='si02_suprasil_2a')
            spectrum = cerenkov_model.cerenkov_emission(observer_location, obstime)
            wavelength = spectrum.waveset
            intensity = spectrum(wavelength)

        plt.plot(wavelength, intensity)
        plt.xlabel(rf"Wavelength [{wavelength.unit}]")
        plt.ylabel(rf"Intensity [{intensity.unit} (erg / s cm$^{2}$ Hz)]")
        plt.title(r"Cerenkov Background Spectrum at GEO")
        plt.tight_layout()
        plt.grid()
    """

    particle: Literal["e", "p"] = "e"
    solar: Literal["max", "min"] = "max"
    material: str = "si02_suprasil_2a"

    def radiation_belt(self, observer_location, obstime) -> Table:
        """
        Returns a table of integral flux values from Earth's radiation belts.
        """
        rb = RadiationBelt(particle=self.particle, solar=self.solar)
        return rb.flux_table(observer_location, obstime)

    def cerenkov_emission(
        self, observer_location, obstime
    ) -> tuple[u.Quantity, u.Quantity]:
        """
        Computes the wavelength-dependent Cerenkov emission intensity.
        """
        emission_model = CerenkovEmission(
            material=self.material, particle=self.particle, solar=self.solar
        )
        return emission_model.emission(observer_location, obstime)
