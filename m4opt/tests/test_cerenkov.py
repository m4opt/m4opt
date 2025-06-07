import pytest


@pytest.mark.parametrize("material", ["sio2", "SiO2_suprasil_2a"])
@pytest.mark.parametrize("particle", ["e"])
@pytest.mark.parametrize("solar", ["max", "min"])
def test_cerenkov_emission_variants(material, particle, solar):
    """
    Test that output units match both the original MATLAB implementation (intensity_arcsec2)
    and the expected format for synphot Spectrum (intensity_photlam).
    """
    import numpy as np
    from astropy import units as u
    from astropy.coordinates import EarthLocation
    from astropy.time import Time

    from m4opt.synphot.background import _cerenkov

    observer_location = EarthLocation.from_geodetic(
        lon=15 * u.deg, lat=0 * u.deg, height=35786 * u.km
    )
    obstime = Time("2025-06-07T13:03:00Z")

    wavelength, intensity_arcsec2, intensity_photlam = _cerenkov.cerenkov_emission(
        observer_location, obstime, material=material, particle=particle, solar=solar
    )
    # Check that the unit of intensity_arcsec2 matches the original output from the MATLAB:
    # https://github.com/EranOfek/AstroPack/blob/main/matlab/astro/%2Bultrasat/Cerenkov.m#L217
    assert intensity_arcsec2.unit.is_equivalent(
        u.photon / u.cm**2 / u.s / u.arcsec**2 / u.Angstrom
    ), (
        f"Unit of intensity_arcsec2 is {intensity_arcsec2.unit}, expected photon/cm2/s/arcsec2/Angstrom"
    )

    # Check that the photlam intensity is compatible with the one wait for  synphot Spectrum
    assert intensity_photlam.unit.is_equivalent(
        u.photon / u.cm**2 / u.s / u.Angstrom
    ), (
        f"Unit of intensity_photlam is {intensity_photlam.unit}, expected photon/cm2/s/Angstrom"
    )

    # Check that every value  is >= 0
    assert not (intensity_arcsec2.value < 0).any()
    assert not (intensity_photlam.value < 0).any()

    # check the length
    assert len(wavelength) == len(intensity_arcsec2) == len(intensity_photlam)

    # check if there are NAN
    assert not np.any(np.isnan(intensity_arcsec2.value))
    assert not np.any(np.isnan(intensity_photlam.value))
