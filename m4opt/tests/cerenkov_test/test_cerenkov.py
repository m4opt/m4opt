import xml.etree.ElementTree as ET
from importlib import resources
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import units as u
from astropy.constants import alpha, c, m_e, m_p
from astropy.table import Table
from scipy.interpolate import CubicSpline

from m4opt.synphot.background._cerenkov._electron_loss import get_electron_energy_loss
from m4opt.synphot.background._cerenkov._refraction_index import get_refraction_index
from m4opt.synphot.background._core import BACKGROUND_SOLID_ANGLE

from . import data


def geostat_electrons_spec_flux():
    """
    Return electron flux spectra in geostationary orbit (AE9 empirical model).
    Columns: Energy (MeV), DailyMin/Max Mean/95/50/75 Flux.
    All fluxes: counts(>E)/cm^2/s.

    data from https://github.com/EranOfek/AstroPack/blob/main/matlab/astro/+ultrasat/geostat_electrons_spec_flux.m
    """
    Fmat1 = np.array(
        [
            [0.04, 63619019.6, 61598061.5, 219133005, 209374930],
            [0.07, 36872356.9, 36604332.6, 121419597, 119071840],
            [0.1, 23629478.6, 24117441.4, 75420703.3, 76234260.4],
            [0.25, 6777807.92, 7653097.8, 21683725.7, 24401822.4],
            [0.5, 1505849.67, 1987502.64, 5285939.47, 6961335.26],
            [0.75, 532336.857, 785319.25, 2060170.96, 3014375.25],
            [1, 256518.175, 401961.156, 1043794.38, 1618766.73],
            [1.5, 77102.0547, 130440.563, 324862.269, 544886.193],
            [2, 22695.0465, 40323.4414, 97545.6771, 172236.871],
            [2.5, 7408.36548, 13171.9014, 32144.5391, 56945.9078],
            [3, 2902.45382, 4869.66977, 12657.8754, 21224.3154],
            [3.5, 1298.75613, 2032.75609, 5661.91617, 8881.94978],
            [4, 648.157196, 957.567543, 2833.55844, 4193.47025],
            [4.5, 355.238406, 513.454146, 1572.8338, 2274.06138],
            [5, 219.826615, 318.476592, 988.712553, 1433.74367],
            [5.5, 150.173834, 218.868829, 683.61939, 998.874262],
            [6, 107.712056, 157.225558, 493.910694, 723.398852],
            [6.5, 77.8067567, 113.637181, 358.767662, 525.475644],
            [7, 57.4414259, 84.0157197, 265.858268, 389.596614],
            [8.5, 21.5238149, 31.6371704, 100.141456, 147.135716],
        ]
    )
    Fmat2 = np.array(
        [
            [0.04, 61598061.5, 35728619.8, 82167128.4, 209374930],
            [0.07, 36604332.6, 22752921.9, 49675314.8, 119071840],
            [0.1, 24117441.4, 15615086.5, 33024741.1, 76234260.4],
            [0.25, 7653097.8, 4890918.78, 10414799.6, 24401822.4],
            [0.5, 1987502.64, 1091815, 2599014.31, 6961335.26],
            [0.75, 785319.25, 354441.093, 970066.244, 3014375.25],
            [1, 401961.156, 158626.819, 475113.598, 1618766.73],
            [1.5, 130440.563, 45347.7461, 146957.507, 544886.193],
            [2, 40323.4414, 12783.5097, 43765.0585, 172236.871],
            [2.5, 13171.9014, 3942.62118, 13947.1046, 56945.9078],
            [3, 4869.66977, 1395.76484, 5055.76672, 21224.3154],
            [3.5, 2032.75609, 572.229433, 2089.12917, 8881.94978],
            [4, 957.567543, 263.889122, 970.814129, 4193.47025],
            [4.5, 513.454146, 130.556961, 499.560374, 2274.06138],
            [5, 318.476592, 71.2148678, 291.441096, 1433.74367],
            [5.5, 218.868829, 42.8504716, 188.206425, 998.874262],
            [6, 157.225558, 27.661211, 128.388022, 723.398852],
            [6.5, 113.637181, 18.2519175, 88.693106, 525.475644],
            [7, 84.0157197, 12.5727802, 63.283367, 389.596614],
            [8.5, 31.6371704, 4.11078406, 22.1665976, 147.135716],
        ]
    )
    Fmat3 = np.array(
        [
            [0.04, 63619019.6, 36076316.5, 84378698, 219133005],
            [0.07, 36872356.9, 22505752.7, 49850150.8, 121419597],
            [0.1, 23629478.6, 15098752.9, 32292885.2, 75420703.3],
            [0.25, 6777807.92, 4312831.41, 9226564.31, 21683725.7],
            [0.5, 1505849.67, 824050.883, 1968346.63, 5285939.47],
            [0.75, 532336.857, 235221.892, 652987.839, 2060170.96],
            [1, 256518.175, 97922.906, 299592.627, 1043794.38],
            [1.5, 77102.0547, 25912.6128, 85714.0753, 324862.269],
            [2, 22695.0465, 6993.83398, 24345.3269, 97545.6771],
            [2.5, 7408.36548, 2178.20925, 7785.44585, 32144.5391],
            [3, 2902.45382, 829.45825, 3009.48586, 12657.8754],
            [3.5, 1298.75613, 370.461383, 1342.67285, 5661.91617],
            [4, 648.157196, 181.007089, 661.824066, 2833.55844],
            [4.5, 355.238406, 91.3309337, 348.523534, 1572.8338],
            [5, 219.826615, 50.3048578, 204.295304, 988.712553],
            [5.5, 150.173834, 30.8234226, 132.670368, 683.61939],
            [6, 107.712056, 20.3254414, 91.3718367, 493.910694],
            [6.5, 77.8067567, 13.5646922, 63.4974925, 358.767662],
            [7, 57.4414259, 9.37080518, 45.3675733, 265.858268],
            [8.5, 21.5238149, 3.0546999, 15.8521726, 100.141456],
        ]
    )

    arr = np.column_stack(
        [
            Fmat1,
            Fmat2[:, 2:4],
            Fmat3[:, 2:4],
        ]
    )
    colnames = [
        "Energy",
        "DailyMin_MeanFlux",
        "DailyMax_MeanFlux",
        "DailyMin_95Flux",
        "DailyMax_95Flux",
        "DailyMax_50Flux",
        "DailyMax_75Flux",
        "DailyMin_50Flux",
        "DailyMin_75Flux",
    ]
    return Table(arr, names=colnames)


def cerenkov_emission_mat(xml_path, fields=None):
    r"""
    Reads a MATLAB struct-style XML file and returns the results as an Astropy Table.

    The XML file contains the Cerenkov emission calculated using the daily flux (DailyMax)
    at the 95th percentile, for the material: 'si02_suprasil_2a'.

    Args:
        xml_path (str): Path to the XML file.
        fields (list or None): List of fields/tags to extract. If None, detects all automatically.

    Returns:
        astropy.table.Table : Table containing the results as columns.

        - 'Lam' : np.ndarray
            Wavelength array [Å] (Angström, equivalent to Ang or \(\AA\)).
        - 'Int' : np.ndarray
            Cerenkov intensity [count/cm^2/s/sr/μm].
        - 'Int_Units' : str
            Units of intensity, 'count/cm^2/s/sr/μm'.
        - 'IntAA' : np.ndarray
            Intensity per arcsecond squared [count/cm^2/s/arcsec^2/Ang].
        - 'IntAA_Units' : str
            Units for 'IntAA', 'count/cm^2/s/arcsec^2/Ang'.
        - 'IntFA' : np.ndarray
            Energy flux in erg/cm^2/s/arcsec^2/Ang.
        - 'IntFA_Units' : str
            Units for 'IntFA', 'erg/cm^2/s/arcsec^2/Ang'.
        - 'n' : np.ndarray
            Refractive index values for each wavelength.
        - 'Lum' : np.ndarray
            Luminosity per wavelength [count/cm^2/s/μm].
        - 'Int_arcsec_Units' : str
            Optional label for display: 'counts cm$^{-2}$ s$^{-1}$ arcsec$^{-2}$ \AA$^{-1}$'.
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Auto-detect fields
    if fields is None:
        fields = list({child.tag for child in root if not child.tag.endswith("_Units")})

    # Extract values for each field
    data = {}
    for tag in fields:
        try:
            data[tag] = [float(e.text) for e in root.findall(tag)]
        except Exception:
            data[tag] = [e.text for e in root.findall(tag)]

    # Truncate to minimum length (in case of unequal arrays)
    min_len = min(len(lst) for lst in data.values())
    data = {k: v[:min_len] for k, v in data.items()}

    # Create and return Astropy Table
    return Table(data)


def cerenkov_emission_py(
    material: str = "SiO2_suprasil_2a",
    particle: Literal["e", "p"] = "e",
    nbins: int = 1000,
) -> tuple[u.Quantity, u.Quantity, u.Quantity]:
    r"""
    Compute the Cerenkov emission spectrum using the AE9 DailyMax_95Flux electron model (Python version).

    This function is the Python equivalent of the MATLAB 'Cerenkov.m' script,
    and always uses the electron spectrum from geostat_electrons_spec_flux()['DailyMax_95Flux'].

    Parameters
    ----------
    material : str
        Material name (e.g. "SiO2_suprasil_2a").

    Returns
    -------
    wavelength : Quantity
        Wavelength grid.
    intensity_arcsec2 : Quantity
        Intensity per arcsec².
    intensity_photlam : Quantity
        Intensity per solid angle.

    """

    # --- Get AE9 electron flux (DailyMax_95Flux) ---
    flux_data = geostat_electrons_spec_flux()
    energy = flux_data["Energy"] * u.MeV  # Energy grid
    flux = flux_data["DailyMax_95Flux"] * (
        1 / (u.cm**2 * u.s)
    )  # Flux: counts(>E)/cm^2/s

    # --- Material optical parameters
    material_properties = {
        "sio2": (1.5, 2.2 * u.g / u.cm**3),
        "SiO2_suprasil_2a": (1.5, 2.2 * u.g / u.cm**3),
        "sapphire": (1.75, 4.0 * u.g / u.cm**3),
    }
    if material not in material_properties:
        raise ValueError(f"Unknown material option: '{material}'")

    n_val, rho = material_properties[material]

    # Grid based based on figure  6 of Kruk  et al. (2016), https://iopscience.iop.org/article/10.1088
    ee = np.logspace(np.log10(0.04), np.log10(8.0), nbins) * u.MeV

    # --- Interpolate the electron flux onto the energy grid
    cs_flux = CubicSpline(
        energy.value,
        flux.value,
        bc_type="natural",
        extrapolate=True,
    )
    Fe = u.Quantity(cs_flux(ee.value), flux.unit)

    # Compute midpoint energies between grid bins
    em = 0.5 * (ee[:-1] + ee[1:])

    # Get particle mass (rest energy) in MeV
    mass = m_e if particle == "e" else m_p
    mass_mev = (mass * c**2).to("MeV")

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
    # "photon/cm^2/s/sr/μm",
    intensity_micron = L1mu / (2 * np.pi * n**2) / u.sr

    # Convert intensity to arcseconds squared
    intensity_angstrom = intensity_micron.to(
        u.photon / u.cm**2 / u.s / u.sr / u.Angstrom
    )

    # 'ph/A' : [ phothon / s / cm^2 / Angstrom / arcsec^2]
    intensity_arcsec2 = intensity_angstrom.to(
        u.photon / u.cm**2 / u.s / BACKGROUND_SOLID_ANGLE / u.Angstrom
    )
    intensity_photlam = intensity_arcsec2 * BACKGROUND_SOLID_ANGLE

    wavelength = Lam
    return wavelength, intensity_arcsec2, intensity_photlam


def test_cerenkov_numerical():
    r"""
    Validate consistency between Python and MATLAB Cerenkov emission calculations.

    This test checks that the wavelength and intensity arrays computed in Python
    agree with those exported from MATLAB (see Notes for format). Relative differences
    must remain below 4% across the spectrum.

    Checks that:
    - The wavelength arrays match closely.
    - The relative difference between intensity_arcsec2 (Python) and IntAA (MATLAB)
      is less than 4% everywhere.

    Raises
    ------
    AssertionError if results differ beyond the allowed tolerance.

    Notes
    -----
    The test requires the MATLAB Cerenkov output to be exported as an XML file
    (e.g. "../Cerenkov_output.xml") containing at least the fields 'Lam' and 'IntAA'.

    """

    # Matlab output
    xml_path = resources.files(data) / "Cerenkov_output.xml"
    table = cerenkov_emission_mat(xml_path, fields=["Lam", "IntAA"])
    Lam = table["Lam"]
    IntAA = table["IntAA"]

    # Python results
    wavelength, intensity_arcsec2, intensity_photlam = cerenkov_emission_py(
        material="SiO2_suprasil_2a", particle="e", nbins=1000
    )

    # Enforce the check for python output
    # Check that the photlam intensity is compatible with the one wait for  synphot Spectrum
    # https://github.com/EranOfek/AstroPack/blob/main/matlab/astro/%2Bultrasat/Cerenkov.m#L217
    assert intensity_arcsec2.unit.is_equivalent(
        u.photon / u.cm**2 / u.s / u.arcsec**2 / u.Angstrom
    )

    # check the length
    assert len(wavelength) == len(intensity_arcsec2) == len(intensity_photlam)

    # Check that every value  is >= 0
    assert not (intensity_arcsec2.value < 0).any()
    assert not (intensity_photlam.value < 0).any()

    # check if there are NAN
    assert not np.any(np.isnan(intensity_arcsec2.value))
    assert not np.any(np.isnan(intensity_photlam.value))

    # Check value accuracy with MATLAB reference
    assert np.allclose(wavelength.value, Lam, rtol=1e-10), "Wavelength mismatch"
    rel_diff = np.abs(IntAA - intensity_arcsec2.value) / (np.abs(IntAA) + 1e-30)
    assert np.all(rel_diff < 4e-2), (
        f"Python vs MATLAB: maximum relative difference {np.max(rel_diff):.2e} exceeded"
    )


@pytest.mark.mpl_image_compare(tolerance=1)
def test_cerenkov_image():
    """Validate consistency between Python and MATLAB Cerenkov emission plot."""

    # Matlab output
    xml_path = resources.files(data) / "Cerenkov_output.xml"
    table = cerenkov_emission_mat(xml_path, fields=["Lam", "IntAA"])
    Lam = table["Lam"]
    IntAA = table["IntAA"]

    # Python results
    wavelength, intensity_arcsec2, _ = cerenkov_emission_py(
        material="SiO2_suprasil_2a", particle="e", nbins=1000
    )

    fig, ax = plt.subplots()
    ax.plot(Lam, IntAA, label="MATLAB", color="blue")
    ax.plot(
        wavelength.value, intensity_arcsec2.value, "--", label="Python", color="red"
    )
    ax.set_xlabel(r"Wavelength [$\AA$]")
    ax.set_ylabel(r"Intensity [arcsec$^{-2}$]")
    ax.legend()
    return fig
