"""
Python conversion of the MATLAB script `dEdX_calc.m` by Eran Ofek,
available at:
https://github.com/EranOfek/AstroPack

This module computes the energy loss per unit mass thickness (dE/dX)
of electrons in a material as a function of kinetic energy.
"""

import astropy.units as u
import numpy as np
from astropy.constants import c, e, hbar, m_e, m_p

# For each tuple: (Atomic number Z, Mass number A, Atom count in molecular unit)
_MATERIALS = {
    "sio2": {"elements": [(14, 28, 1), (8, 16, 2)]},  # silica (SiO2): Si + 2 O
    "sio2_suprasil_2a": {
        "elements": [(14, 28, 1), (8, 16, 2)]
    },  # Specific high-purity silica
    "sapphire": {
        "elements": [(13, 27, 2), (8, 16, 3)]
    },  # Aluminum oxide (Al2O3): 2 Al + 3 O

    "aluminum": {"elements": [(13, 27, 1)]},    # Al
    "oxygen": {"elements": [(8, 16, 1)]},       # O
    "silicon": {"elements": [(14, 28, 1)]},     # Si
}


def _calc_dEdX(Z, A, Ek, g, b):
    """
    Parameters
    ----------
    Z : int
        Atomic number of the material (number of protons).
    A : int
        Mass number of the material (total number of protons and neutrons).
    """

    e_cgs = e.esu  # statCoulomb
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


def get_electron_energy_loss(material="sio2_suprasil_2a"):
    """
    Calculate the energy loss per unit mass thickness (dE/dX) of electrons
    as a function of kinetic energy in a given material.

    Parameters
    ----------
    material : str, optional
        The material through which electrons propagate. Supported options are:
        - 'sio2_suprasil_2a' (default): Specific high-purity silica (Suprasil 2A).
        - 'sio2' (silica): Silicon dioxide, sharing identical properties with 'sio2_suprasil_2a' for this calculation.
        - 'sapphire': Aluminum oxide (Al2O3).

    Returns
    -------
    Ek : ndarray
        Electron kinetic energy array [MeV].
    dEdX : ndarray
        Energy loss per unit mass thickness array [MeV/(g cm^-2)].

    Raises
    ------
    ValueError
        If an unsupported material name is provided.

    Notes
    -----
    The function calculates energy loss combining collisional and radiative
    components for specified material compositions. Calculations utilize
    approximate Bethe-Bloch formulas and simplified radiative loss estimates.
    """

    material = material.lower()
    if material not in _MATERIALS:
        raise ValueError(f"Unknown material option: '{material}'")

    elements = _MATERIALS[material]["elements"]

    g = 1 + 10 ** np.arange(-3, 4.01, 0.01)
    b = np.sqrt(1 - 1.0 / g**2)
    Ek = ((g - 1) * m_e.cgs * c.cgs**2).to(u.MeV)

    total_mass = sum(A * count for _, A, count in elements)

    dEdX_total = 0
    for Z, A, count in elements:
        mass_fraction = A * count / total_mass
        dEdX_element = _calc_dEdX(Z, A, Ek, g, b)
        dEdX_total += mass_fraction * dEdX_element

    return Ek, dEdX_total
