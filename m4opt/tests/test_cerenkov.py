import pytest


@pytest.mark.parametrize("material", ["sio2", "SiO2_suprasil_2a"])
@pytest.mark.parametrize("particle", ["e"])
@pytest.mark.parametrize("solar", ["max", "min"])
def test_cerenkov_emission_variants(material, particle, solar):
    """Test for units and NAN"""
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

    assert intensity_arcsec2.unit.is_equivalent(
        u.photon / u.cm**2 / u.s / u.arcsec**2 / u.Angstrom
    ), (
        f"Unit of intensity_arcsec2 is {intensity_arcsec2.unit}, expected photon/cm2/s/arcsec2/Angstrom"
    )

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
