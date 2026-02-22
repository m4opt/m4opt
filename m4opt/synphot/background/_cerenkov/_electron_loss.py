"""Electron energy loss per unit mass thickness (dE/dX) in materials.

Python conversion of the MATLAB script ``dEdX_calc.m`` by Eran Ofek,
available at https://github.com/EranOfek/AstroPack.
"""

import astropy.units as u
import numpy as np
from astropy.constants import c, e, hbar, m_e, m_p

# For each tuple: (Atomic number Z, Mass number A, Atom count in molecular unit)
_MATERIALS = {
    "sio2": {"elements": [(14, 28, 1), (8, 16, 2)]},
    "sio2_suprasil_2a": {"elements": [(14, 28, 1), (8, 16, 2)]},
    "sapphire": {"elements": [(13, 27, 2), (8, 16, 3)]},
    "aluminum": {"elements": [(13, 27, 1)]},
    "oxygen": {"elements": [(8, 16, 1)]},
    "silicon": {"elements": [(14, 28, 1)]},
}


def _calc_dEdX(Z, A, Ek, g, b):
    """Calculate energy loss per unit mass thickness for a single element.

    Parameters
    ----------
    Z : int
        Atomic number of the material.
    A : int
        Mass number of the material.
    Ek : `~astropy.units.Quantity`
        Kinetic energy array.
    g : `~numpy.ndarray`
        Lorentz gamma factor.
    b : `~numpy.ndarray`
        Velocity in units of c (beta).
    """
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


def get_electron_energy_loss(material="sio2_suprasil_2a"):
    """Calculate energy loss per unit mass thickness of electrons in a material.

    Parameters
    ----------
    material : str
        Supported options:

        - ``'sio2_suprasil_2a'``: high-purity silica (Suprasil 2A)
        - ``'sio2'``: silicon dioxide
        - ``'sapphire'``: aluminum oxide (Al2O3)
        - ``'aluminum'``, ``'oxygen'``, ``'silicon'``: pure elements

    Returns
    -------
    Ek : `~astropy.units.Quantity`
        Electron kinetic energy array [MeV].
    dEdX : `~astropy.units.Quantity`
        Energy loss per unit mass thickness [MeV/(g cm^-2)].

    Raises
    ------
    ValueError
        If an unsupported material name is provided.
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
