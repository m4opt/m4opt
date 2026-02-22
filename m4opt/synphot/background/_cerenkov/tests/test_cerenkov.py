"""Tests for the Cerenkov background radiation model."""

import xml.etree.ElementTree as ET
from importlib import resources

import numpy as np
from astropy import units as u
from astropy.constants import alpha, c, e, hbar, m_e, m_p
from astropy.coordinates import SkyCoord
from astropy.table import Table
from scipy.interpolate import CubicSpline
from synphot import Empirical1D, SourceSpectrum

from ..._core import BACKGROUND_SOLID_ANGLE
from .. import (
    _MATERIAL_PROPERTIES,
    _REFERENCE_LOCATION,
    _REFERENCE_OBSTIME,
    CerenkovBackground,
    CerenkovScaleFactor,
)
from .._electron_loss import get_electron_energy_loss
from .._refraction_index import get_refraction_index
from . import data


def test_refraction_index():
    """Fused silica refractive index matches expected values."""
    L, n, _ = get_refraction_index("sio2_suprasil_2a")
    # At sodium D line (~5893 AA), n should be ~1.458
    idx = np.argmin(np.abs(L.value - 5893))
    assert abs(n[idx] - 1.458) < 0.002
    # UV refractive index should be higher than visible
    uv_idx = np.argmin(np.abs(L.value - 2500))
    vis_idx = np.argmin(np.abs(L.value - 5500))
    assert n[uv_idx] > n[vis_idx]


def test_electron_energy_loss_positive():
    """Energy loss is positive for all energies and materials."""
    for material in ["sio2", "sio2_suprasil_2a", "sapphire"]:
        Ek, dEdX = get_electron_energy_loss(material)
        assert np.all(dEdX.value > 0), f"dEdX not positive for {material}"


# ---------------------------------------------------------------------------
# Bethe-Bloch derivation (used to verify precomputed electron energy loss data)
# ---------------------------------------------------------------------------

# For each tuple: (Atomic number Z, Mass number A, Atom count in molecular unit)
_MATERIALS = {
    "sio2": {"elements": [(14, 28, 1), (8, 16, 2)]},
    "sio2_suprasil_2a": {"elements": [(14, 28, 1), (8, 16, 2)]},
    "sapphire": {"elements": [(13, 27, 2), (8, 16, 3)]},
}


def _calc_dEdX(Z, A, Ek, g, b):
    """Calculate energy loss for a single element via Bethe-Bloch."""
    e_cgs = e.esu
    me_cgs = m_e.cgs
    mp_cgs = m_p.cgs
    c_cgs = c.cgs
    hbar_cgs = hbar.cgs

    Iav = (1.3 * 10 * Z * 1 * u.eV).to(u.erg)

    dedx_prefactor = (
        2 * np.pi * e_cgs**4 * (Z / A) / (mp_cgs * me_cgs * c_cgs**2) / b**2
    ).to(u.MeV / (u.g * u.cm**-2))

    dEdXI = dedx_prefactor * (
        np.log(((g**2 - 1) * me_cgs * c_cgs**2 / Iav) ** 2 / 2.0 / (1 + g))
        - (2.0 / g - 1.0 / g**2) * np.log(2)
        + 1.0 / g**2
        + (1 - 1.0 / g) ** 2 / 8
    )
    dEdXI = np.maximum(dEdXI, 0)

    dEdXB = (
        4
        * Z**2
        / A
        * e_cgs**6
        / (me_cgs**2 * mp_cgs * c**4 * hbar_cgs)
        * Ek
        / (b * c_cgs)
        * (np.log(183 / Z ** (1.0 / 3)) + 1.0 / 8)
    ).to(u.MeV / (u.g * u.cm**-2))

    return dEdXI + dEdXB


def _compute_electron_energy_loss(material):
    """Derive electron energy loss from Bethe-Bloch formula."""
    elements = _MATERIALS[material]["elements"]
    g = 1 + 10 ** np.arange(-3, 4.01, 0.01)
    b = np.sqrt(1 - 1.0 / g**2)
    Ek = ((g - 1) * m_e.cgs * c.cgs**2).to(u.MeV)
    total_mass = sum(A * count for _, A, count in elements)
    dEdX_total = 0
    for Z, A, count in elements:
        mass_fraction = A * count / total_mass
        dEdX_total += mass_fraction * _calc_dEdX(Z, A, Ek, g, b)
    return Ek, dEdX_total


def test_electron_energy_loss_matches_derivation():
    """Precomputed energy loss data matches Bethe-Bloch derivation."""
    for material in ["sio2", "sapphire"]:
        Ek_tab, dEdX_tab = get_electron_energy_loss(material)
        Ek_calc, dEdX_calc = _compute_electron_energy_loss(material)
        np.testing.assert_allclose(
            Ek_tab.value,
            Ek_calc.value,
            rtol=1e-10,
            err_msg=f"Energy mismatch for {material}",
        )
        np.testing.assert_allclose(
            dEdX_tab.value,
            dEdX_calc.value,
            rtol=1e-10,
            err_msg=f"dEdX mismatch for {material}",
        )


# ---------------------------------------------------------------------------


def test_cerenkov_reference_regression():
    """Reference spectrum matches frozen regression values."""
    spec = CerenkovBackground.reference(factor=21)
    np.testing.assert_almost_equal(
        spec(1862 * u.AA).value, 2.636869237717084e-08, decimal=13
    )
    np.testing.assert_almost_equal(
        spec(2600 * u.AA).value, 1.211405272508389e-08, decimal=13
    )
    np.testing.assert_almost_equal(
        spec(5000 * u.AA).value, 3.014408520526302e-09, decimal=14
    )
    np.testing.assert_almost_equal(
        spec(12000 * u.AA).value, 5.062175552605670e-10, decimal=15
    )


def test_cerenkov_uv_brighter():
    """UV emission is brighter than visible (Cerenkov ~ 1/lambda^2)."""
    spec = CerenkovBackground.reference(factor=21)
    uv = spec(2500 * u.AA).value
    vis = spec(5500 * u.AA).value
    assert uv > vis
    # The ratio should be substantial given the 1/lambda^2 dependence
    assert uv / vis > 3


def test_scale_factor_at_reference():
    """CerenkovScaleFactor returns 1.0 at reference location."""
    sf = CerenkovScaleFactor(particle="e", solar="max")
    val = sf.at(
        _REFERENCE_LOCATION,
        SkyCoord(0 * u.deg, 0 * u.deg),
        _REFERENCE_OBSTIME,
    )
    assert val == 1.0


def test_cerenkov_in_context():
    """Full pipeline works within observing() context at GEO."""
    from m4opt.synphot import observing

    loc = _REFERENCE_LOCATION
    coord = SkyCoord(0 * u.deg, 0 * u.deg)
    obstime = _REFERENCE_OBSTIME

    bg = CerenkovBackground(factor=21)
    with observing(observer_location=loc, target_coord=coord, obstime=obstime):
        val = bg(2600 * u.AA)
    assert val.value > 0


def _geostat_electrons_spec_flux():
    """Return electron flux spectra in geostationary orbit (AE9 empirical model).

    Data from
    https://github.com/EranOfek/AstroPack/blob/main/matlab/astro/+ultrasat/geostat_electrons_spec_flux.m
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


def _cerenkov_emission_from_matlab_xml(xml_path, factor=21):
    """Load MATLAB Cerenkov output from XML and return as SourceSpectrum."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    lam = np.array([float(e.text) for e in root.findall("Lam")])
    intaa = np.array([float(e.text) for e in root.findall("IntAA")])

    intensity_arcsec2 = intaa * (
        u.photon / u.cm**2 / u.s / BACKGROUND_SOLID_ANGLE / u.Angstrom
    )
    intensity_photlam = intensity_arcsec2 * BACKGROUND_SOLID_ANGLE

    return SourceSpectrum(
        Empirical1D,
        points=lam * u.Angstrom,
        lookup_table=intensity_photlam / factor,
    )


def _cerenkov_spectrum_from_flux(
    ee,
    Fe,
    material="sio2_suprasil_2a",
    particle="e",
    factor=21,
):
    """Compute Cerenkov spectrum from an energy grid and flux (test helper)."""
    if np.all(Fe.value == 0):
        Lam, _, _ = get_refraction_index(material)
        zero = np.zeros_like(Lam.value) * u.photon / (u.cm**2 * u.s * u.AA)
        return SourceSpectrum(Empirical1D, points=Lam, lookup_table=zero)

    n_val, rho = _MATERIAL_PROPERTIES[material]
    em = 0.5 * (ee[:-1] + ee[1:])
    mass = m_e if particle == "e" else m_p
    mass_mev = (mass * c**2).to(u.MeV)
    gamma = 1 + em / mass_mev
    beta = np.sqrt(1 - 1.0 / gamma**2)
    cs_fm = CubicSpline(ee.value, Fe.value, bc_type="natural", extrapolate=True)
    Fm = u.Quantity(cs_fm(em.value), Fe.unit)
    fC_energy = np.maximum(0, 1 - 1.0 / n_val**2 / beta**2)
    Ek, dEdX = get_electron_energy_loss(material)
    cs_dEdX = CubicSpline(
        Ek.value, 1.0 / dEdX.value, bc_type="natural", extrapolate=True
    )
    gEE = u.Quantity(cs_dEdX(em.value), 1 / dEdX.unit)
    intg = gEE * Fm * fC_energy
    cs_intg = CubicSpline(em.value, intg.value, bc_type="natural", extrapolate=True)
    Lam, n, _ = get_refraction_index(material)
    fC_wl_e = np.maximum(0, 1 - 1.0 / n[:, np.newaxis] ** 2 / beta[np.newaxis, :] ** 2)
    intg = gEE * Fm * fC_wl_e
    cs_val = cs_intg(1.0) * intg.unit
    Lam_cm = Lam.to(u.cm)
    Lnorm = 2 * np.pi * alpha / rho / Lam_cm**2 * cs_val * u.photon
    Lnorm = Lnorm.to(u.photon / (u.MeV * u.s * u.cm**2 * u.micron))
    int_val = np.sum(intg * np.diff(ee)[np.newaxis, :], axis=1) / cs_val
    L1mu = int_val * Lnorm
    IC1mu = L1mu / (2 * np.pi * n**2) / u.sr
    intensity_angstrom = IC1mu.to(u.photon / u.cm**2 / u.s / u.sr / u.Angstrom)
    intensity_arcsec2 = intensity_angstrom.to(
        u.photon / u.cm**2 / u.s / BACKGROUND_SOLID_ANGLE / u.Angstrom
    )
    intensity_photlam = intensity_arcsec2 * BACKGROUND_SOLID_ANGLE
    return SourceSpectrum(
        Empirical1D, points=Lam, lookup_table=intensity_photlam / factor
    )


def _cerenkov_emission_from_ae9(
    material="sio2_suprasil_2a",
    particle="e",
    factor=21,
    nbins=1000,
):
    """Compute Cerenkov emission using AE9 DailyMax_95Flux electron model."""
    flux_data = _geostat_electrons_spec_flux()
    energy = flux_data["Energy"] * u.MeV
    flux_vals = flux_data["DailyMax_95Flux"] * (1 / (u.cm**2 * u.s))

    ee = np.logspace(np.log10(0.04), np.log10(8.0), nbins) * u.MeV

    cs_flux = CubicSpline(
        energy.value, flux_vals.value, bc_type="natural", extrapolate=True
    )
    flux_grid = u.Quantity(cs_flux(ee.value), flux_vals.unit)

    return _cerenkov_spectrum_from_flux(
        ee, flux_grid, material=material, particle=particle, factor=factor
    )


def test_cerenkov_vs_matlab():
    """Python Cerenkov emission matches MATLAB reference within 4%."""
    xml_path = resources.files(data) / "Cerenkov_output.xml"
    spectrum_mat = _cerenkov_emission_from_matlab_xml(str(xml_path))
    wavelength_mat = spectrum_mat.waveset
    intensity_mat = spectrum_mat(wavelength_mat)

    spectrum_py = _cerenkov_emission_from_ae9(
        material="sio2_suprasil_2a", particle="e", nbins=1000
    )
    wavelength_py = spectrum_py.waveset
    intensity_py = spectrum_py(wavelength_py)

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
