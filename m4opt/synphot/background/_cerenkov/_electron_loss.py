"""
Python conversion of the MATLAB script `dEdX_calc.m` by Eran Ofek,
available at:
https://github.com/EranOfek/AstroPack

This module computes the energy loss per unit mass thickness (dE/dX)
of electrons in a material as a function of kinetic energy.
"""

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np


def get_electron_energy_loss(material="si02_suprasil_2a", plot=False):
    """
    Calculate the energy loss per unit mass thickness (dE/dX) of electrons
    as a function of kinetic energy in a given material.

    Parameters
    ----------
    material : str, optional
        The material in which electrons are propagating. Supported options are:
        - 'si02_suprasil_2a' or 'silica' or 'sio2' (default): silicon dioxide.
        - 'sapphire' : aluminum oxide.
    plot : bool, optional
        If True, plots the dE/dX curves and the inverse normalized energy loss.

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
    The function combines collisional and radiative energy loss components
    for oxygen, silicon, and aluminum, depending on the material composition.
    The calculations use approximate Bethe-Bloch formulas and simplified
    radiative loss estimates.
    """

    e = 4.8e-10  # statC
    me = 9.1e-28  # g
    mp = 1.7e-24  # g
    c = 3e10  # cm/s
    hbar = 1.05e-27  # erg*s

    g = 1 + 10 ** np.arange(-3, 4.01, 0.01)
    b = np.sqrt(1 - 1.0 / g**2)
    Ek = (g - 1) * me * c**2 / 1.6e-6  # MeV

    def calc_dEdX(Z, A):
        Iav = 1.3 * 10 * Z * 1.6e-12
        dEdXI = (
            2
            * np.pi
            * e**4
            * (Z / A)
            / (mp * me * c**2)
            / 1.6e-6
            / b**2
            * (
                np.log(((g**2 - 1) * me * c**2 / Iav) ** 2 / 2.0 / (1 + g))
                - (2.0 / g - 1.0 / g**2) * np.log(2)
                + 1.0 / g**2
                + (1 - 1.0 / g) ** 2 / 8
            )
        )
        dEdXI = np.maximum(dEdXI, 0)
        dEdXB = (
            4
            * Z**2
            / A
            * e**6
            / me**2
            / mp
            / c**4
            / hbar
            * Ek
            / b
            / c
            * (np.log(183 / Z ** (1.0 / 3)) + 1.0 / 8)
        )
        return dEdXI + dEdXB

    dEdXSi = calc_dEdX(14, 28)  # Silicon
    dEdXO = calc_dEdX(8, 16)  # Oxygen
    dEdXAl = calc_dEdX(13, 27)  # Aluminium

    material = material.lower()
    if material in {"sio2", "silica", "si02_suprasil_2a"}:
        dEdX = 28 / (2 * 16 + 28) * dEdXSi + 2 * 16 / (2 * 16 + 28) * dEdXO
    elif material == "sapphire":
        dEdX = 27 * 2 / (2 * 27 + 3 * 16) * dEdXAl + 3 * 16 / (2 * 27 + 3 * 16) * dEdXO
    else:
        raise ValueError(f"Unknown material option: '{material}'")

    if plot:
        lw = 1.5
        label_fs = 15
        tick_fs = 12

        plt.figure()
        plt.loglog(Ek, dEdXO, label="Oxygen", linewidth=lw)
        plt.loglog(Ek, dEdXSi, label="Silicon", linewidth=lw)
        plt.loglog(Ek, dEdXAl, label="Aluminium", linewidth=lw)
        plt.loglog(Ek, dEdX, label="Combined", linewidth=lw)
        plt.xlabel("Electron Kinetic Energy [MeV]", fontsize=label_fs)
        plt.ylabel("dE/dX [MeV/(g cm$^{-2}$)]", fontsize=label_fs)
        plt.legend()
        plt.grid(True)
        plt.tick_params(labelsize=tick_fs)
        plt.show()

        plt.figure()
        plt.loglog(Ek, 1 / dEdX, linewidth=lw)
        plt.xlabel("Electron Kinetic Energy [MeV]", fontsize=label_fs)
        plt.ylabel(
            r"Inverse Normalized Energy Loss $(dE/dX)^{-1}$ [(g cm$^{-2}$)/MeV]",
            fontsize=label_fs,
        )
        plt.grid(True)
        plt.axis([1e-2, 1e2, 1e-2, 1])
        plt.tick_params(labelsize=tick_fs)
        plt.show()

    return Ek * u.MeV, dEdX * u.MeV / (u.g * u.cm**-2)
