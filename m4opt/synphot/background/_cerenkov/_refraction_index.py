"""
Python conversion of the MATLAB script `refraction_index.m` by Eran Ofek,
available at:
https://github.com/EranOfek/AstroPack

This module computes the refractive index and transmission for optical
materials as a function of wavelength.
"""

from importlib import resources

import astropy.units as u
import numpy as np
from astropy.table import Table

from . import data


def ref_index_fused_silica():
    """Refractive index (Sellmeier equation) for fused silica (SiO2)."""
    L = np.arange(0.21, 6.71, 0.01)
    n = np.sqrt(
        1
        + 0.6961663 * L**2 / (L**2 - 0.0684043**2)
        + 0.4079426 * L**2 / (L**2 - 0.1162414**2)
        + 0.8974794 * L**2 / (L**2 - 9.896161**2)
    )
    L = (L * u.micron).to(u.Angstrom)
    t = None
    return L, n, t


def ref_index_suprasil_2a():
    """Refractive index and transmission for Suprasil 2A (tabulated data)."""
    table_n = Table.read(
        resources.files(data) / "suprasil_2a_refractive_index.csv", format="ascii.csv"
    )
    L = table_n["wavelength"] * u.Angstrom
    n = table_n["refractive_index"]

    table_t = Table.read(
        resources.files(data) / "suprasil_2a_transmission.csv", format="ascii.csv"
    )
    t = table_t["transmission"] / u.cm

    return L, n, t


_MATERIALS = {
    "sio2": ref_index_fused_silica,
    "sio2_suprasil_2a": ref_index_suprasil_2a,
}


def get_refraction_index(material="SiO2_suprasil_2a"):
    """
    Compute refractive index and transmission for a given material.

    Parameters
    ----------
    material : str, optional
        Supported options:
        - 'SiO2' (silica): fused silica (Sellmeier equation)
        - 'SiO2_suprasil_2a': Suprasil 2A (tabulated data)
        Default: 'SiO2_suprasil_2a'.

    Returns
    -------
    L : astropy.units.Quantity
        Wavelengths [Å].
    n : np.ndarray
        Refractive index.
    t : astropy.units.Quantity or None
        Transmission per 1 cm (only for Suprasil 2A), or None.

    Raises
    ------
    ValueError
        If unsupported material is specified.
    """
    material = material.lower()
    try:
        return _MATERIALS[material]()
    except KeyError:
        raise ValueError(f"Unknown material option: {material}")
