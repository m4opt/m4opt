import xml.etree.ElementTree as ET
from importlib import resources
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table
from scipy.interpolate import CubicSpline
from synphot import Empirical1D, SourceSpectrum

from ..._core import BACKGROUND_SOLID_ANGLE
from .. import cerenkov_emission_core
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

    Parameters
    ----------
    xml_path : str
        Path to the XML file.
    fields : list of str or None, optional
        List of fields/tags to extract. If None, detects all fields automatically.

    Returns
    -------
    astropy.table.Table
        Table containing the extracted results as columns:

        - `Lam` : numpy.ndarray
          Wavelength array :math:`[\text{\AA}]` (Angström).
        - `Int` : numpy.ndarray
          Cerenkov intensity :math:`[\mathrm{count}/\mathrm{cm}^2/\mathrm{s}/\mathrm{sr}/\mu\mathrm{m}]`.
        - `Int_Units` : str
          Units of intensity: :math:`\mathrm{count}/\mathrm{cm}^2/\mathrm{s}/\mathrm{sr}/\mu\mathrm{m}`.
        - `IntAA` : numpy.ndarray
          Intensity per arcsecond squared :math:`[\mathrm{count}/\mathrm{cm}^2/\mathrm{s}/\mathrm{arcsec}^2/\text{\AA}]`.
        - `IntAA_Units` : str
          Units for `IntAA`: :math:`\mathrm{count}/\mathrm{cm}^2/\mathrm{s}/\mathrm{arcsec}^2/\text{\AA}`.
        - `IntFA` : numpy.ndarray
          Energy flux :math:`[\mathrm{erg}/\mathrm{cm}^2/\mathrm{s}/\mathrm{arcsec}^2/\text{\AA}]`.
        - `IntFA_Units` : str
          Units for `IntFA`: :math:`\mathrm{erg}/\mathrm{cm}^2/\mathrm{s}/\mathrm{arcsec}^2/\text{\AA}`.
        - `n` : numpy.ndarray
          Refractive index values as a function of wavelength.
        - `Lum` : numpy.ndarray
          Luminosity per wavelength :math:`[\mathrm{count}/\mathrm{cm}^2/\mathrm{s}/\mu\mathrm{m}]`.
        - `Int_arcsec_Units` : str
          Units for arcsecond-based intensity values.

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
    table = Table(data)

    Lam = table["Lam"]
    intensity_arcsec2 = table["IntAA"] * (
        u.photon / u.cm**2 / u.s / BACKGROUND_SOLID_ANGLE / u.Angstrom
    )
    intensity_photlam = intensity_arcsec2 * BACKGROUND_SOLID_ANGLE

    wavelength = Lam
    return SourceSpectrum(
        Empirical1D, points=wavelength, lookup_table=intensity_photlam
    )


def cerenkov_emission_py(
    material: str = "SiO2_suprasil_2a",
    particle: Literal["e", "p"] = "e",
    nbins: int = 1000,
) -> tuple[u.Quantity, u.Quantity, u.Quantity]:
    r"""
    Compute the Cerenkov emission spectrum using the AE9 DailyMax_95Flux electron model (Python version).

    This function is the Python equivalent of the MATLAB 'Cerenkov.m' script,
    and always uses the electron spectrum from geostat_electrons_spec_flux()['DailyMax_95Flux'].
    """

    # --- Get AE9 electron flux (DailyMax_95Flux) ---
    flux_data = geostat_electrons_spec_flux()
    energy = flux_data["Energy"] * u.MeV
    flux = flux_data["DailyMax_95Flux"] * (1 / (u.cm**2 * u.s))

    # Grid based based on figure  6 of Kruk  et al. (2016), https://iopscience.iop.org/article/10.1088
    ee = np.logspace(np.log10(0.04), np.log10(8.0), nbins) * u.MeV
    energy_grid = ee

    # --- Interpolate the electron flux onto the energy grid
    cs_flux = CubicSpline(
        energy.value,
        flux.value,
        bc_type="natural",
        extrapolate=True,
    )
    flux_grid = u.Quantity(cs_flux(ee.value), flux.unit)

    return cerenkov_emission_core(
        energy_grid, flux_grid, material=material, particle=particle
    )


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

    # Matlab output (loads AE9/MATLAB data)

    xml_path = resources.files(data) / "Cerenkov_output.xml"
    spectrum_mat = cerenkov_emission_mat(xml_path, fields=["Lam", "IntAA"])

    wavelength_mat = spectrum_mat.waveset
    intensity_mat = spectrum_mat(wavelength_mat)

    # Python results
    spectrum_py = cerenkov_emission_py(
        material="SiO2_suprasil_2a", particle="e", nbins=1000
    )
    wavelength_py = spectrum_py.waveset
    intensity_py = spectrum_py(wavelength_py)

    # Enforce the check for python output
    # Check that the photlam intensity is compatible with the one wait for  synphot Spectrum
    # https://github.com/EranOfek/AstroPack/blob/main/matlab/astro/%2Bultrasat/Cerenkov.m#L217

    # Check value accuracy with MATLAB reference
    np.testing.assert_allclose(
        wavelength_mat.value,
        wavelength_py.value,
        rtol=1e-10,
        err_msg="Wavelength mismatch",
    )
    np.testing.assert_allclose(
        intensity_py.value,
        intensity_mat.value,
        rtol=4e-2,
        err_msg="Intensity mismatch (Python vs MATLAB)",
    )


@pytest.mark.mpl_image_compare(tolerance=1)
def test_cerenkov_image():
    """Validate consistency between Python and MATLAB Cerenkov emission plot."""

    xml_path = resources.files(data) / "Cerenkov_output.xml"
    spectrum_mat = cerenkov_emission_mat(xml_path, fields=["Lam", "IntAA"])

    # The Matlab results
    wavelength_mat = spectrum_mat.waveset
    intensity_mat = spectrum_mat(wavelength_mat)

    # The Python results
    spectrum_py = cerenkov_emission_py(
        material="SiO2_suprasil_2a", particle="e", nbins=1000
    )
    wavelength_py = spectrum_py.waveset
    intensity_py = spectrum_py(wavelength_py)

    fig, ax = plt.subplots()
    ax.plot(wavelength_mat.value, intensity_mat.value, label=r"MATLAB", color="blue")
    ax.plot(wavelength_py.value, intensity_py.value, "--", label=r"Python", color="red")
    ax.set_xlabel(rf"Wavelength [{wavelength_py.unit}]")
    ax.set_ylabel(rf"Intensity [{intensity_py.unit}]")
    ax.legend()
    return fig
