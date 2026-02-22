"""Tests for the Cerenkov background radiation model."""

import xml.etree.ElementTree as ET
from importlib import resources

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from synphot import Empirical1D, SourceSpectrum

from ..._core import BACKGROUND_SOLID_ANGLE
from .. import (
    _REFERENCE_LOCATION,
    _REFERENCE_OBSTIME,
    CerenkovBackground,
    CerenkovScaleFactor,
)
from .._electron_loss import get_electron_energy_loss
from .._refraction_index import REFRACTION_INDEX
from . import data


def test_refraction_index():
    """Fused silica refractive index matches expected values."""
    L, n, _ = REFRACTION_INDEX["sio2_suprasil_2a"]()
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


def test_cerenkov_vs_matlab():
    """Python Cerenkov spectral shape matches MATLAB reference.

    The production code uses AE8 radiation belt fluxes (via ``aep8``), while
    the MATLAB reference was computed with hardcoded AE9 flux data.  The
    absolute amplitude therefore differs by a constant factor (~3x), but the
    spectral *shape* (relative intensity vs. wavelength) should agree because
    the Cerenkov physics is the same.

    We verify that the ratio Python/MATLAB is constant to within 5%.
    """
    # Load MATLAB reference output
    xml_path = resources.files(data) / "Cerenkov_output.xml"
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    lam = np.array([float(e.text) for e in root.findall("Lam")])
    intaa = np.array([float(e.text) for e in root.findall("IntAA")])

    intensity_arcsec2 = intaa * (
        u.photon / u.cm**2 / u.s / BACKGROUND_SOLID_ANGLE / u.Angstrom
    )
    intensity_photlam = intensity_arcsec2 * BACKGROUND_SOLID_ANGLE
    spectrum_mat = SourceSpectrum(
        Empirical1D,
        points=lam * u.Angstrom,
        lookup_table=intensity_photlam / 21,
    )

    # Production code output
    spectrum_py = CerenkovBackground.reference(factor=21)

    # Compare spectral shapes at MATLAB wavelengths
    wavelengths = spectrum_mat.waveset
    mat_vals = spectrum_mat(wavelengths).value
    py_vals = spectrum_py(wavelengths).value

    ratio = py_vals / mat_vals

    # The ratio should be approximately constant (same shape, different amplitude)
    np.testing.assert_allclose(
        ratio,
        np.median(ratio),
        rtol=0.05,
        err_msg="Spectral shape mismatch (Python vs MATLAB)",
    )
