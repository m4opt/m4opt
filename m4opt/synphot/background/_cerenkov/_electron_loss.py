"""Electron energy loss per unit mass thickness (dE/dX) in materials.

The tabulated data were precomputed from the Bethe–Bloch formula (ionization)
and bremsstrahlung energy loss, following the MATLAB ``dEdX_calc.m`` script
by Eran Ofek (https://github.com/EranOfek/AstroPack).  The derivation code
is verified in the test suite.
"""

from importlib import resources

import astropy.units as u
from astropy.table import Table

from . import data

_ELECTRON_LOSS_FILES = {
    "sio2": "electron_energy_loss_sio2.csv",
    "sio2_suprasil_2a": "electron_energy_loss_sio2.csv",
    "sapphire": "electron_energy_loss_sapphire.csv",
}


def get_electron_energy_loss(material="sio2_suprasil_2a"):
    """Load precomputed electron energy loss for a material.

    Parameters
    ----------
    material : str
        Supported options:

        - ``'sio2_suprasil_2a'``: high-purity silica (Suprasil 2A)
        - ``'sio2'``: silicon dioxide
        - ``'sapphire'``: aluminum oxide (Al2O3)

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
    try:
        filename = _ELECTRON_LOSS_FILES[material]
    except KeyError:
        raise ValueError(f"Unknown material: {material!r}")

    table = Table.read(
        resources.files(data) / filename,
        format="ascii.csv",
    )
    Ek = table["energy_mev"] * u.MeV
    dEdX = table["dedx_mev_per_g_cm2"] * u.MeV / (u.g * u.cm**-2)
    return Ek, dEdX
