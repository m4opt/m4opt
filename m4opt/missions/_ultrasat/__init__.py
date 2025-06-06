from importlib import resources

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from regions import RectangleSkyRegion
from synphot import Gaussian1D, SpectralElement

from ...constraints import (
    EarthLimbConstraint,
    MoonSeparationConstraint,
    SunSeparationConstraint,
)
from ...dynamics import EigenAxisSlew
from ...observer import TleObserverLocation
from ...synphot import Detector
from ...synphot.background import GalacticBackground, ZodiacalBackground
from .._core import Mission
from . import data


def _read_allsky_skygrid() -> SkyCoord:
    # Load the All-Sky Survey (AllSS) grid.
    table = Table.read(
        resources.files(data) / "AllSS_grid_361.txt",
        format="ascii.csv",
        data_start=0,
        names=["ra", "dec"],
    )
    return SkyCoord(table["ra"], table["dec"], unit=u.deg)


def _read_nonoverlapping_skygrid() -> SkyCoord:
    # Load the non-overlapping Low-Cadence Survey (LCS) grid.
    table = Table.read(
        resources.files(data) / "LCS_nonoverlapping_grid.csv", format="ascii.csv"
    )
    return SkyCoord(table["RA"], table["Dec"], unit=u.deg)


ultrasat = Mission(
    name="ultrasat",
    fov=RectangleSkyRegion(
        center=SkyCoord(0 * u.deg, 0 * u.deg), width=14.28 * u.deg, height=14.28 * u.deg
    ),
    constraints=(
        EarthLimbConstraint(48 * u.deg)
        & SunSeparationConstraint(70 * u.deg)
        & MoonSeparationConstraint(35 * u.deg)
    ),
    detector=Detector(
        npix=4 * np.pi,
        plate_scale=(5.4 * u.arcsec) ** 2,
        # Circular aperture with a diameter of 33 cm
        area=np.pi * np.square(0.5 * 33 * u.cm),
        bandpasses={
            "NUV": SpectralElement(
                Gaussian1D,
                amplitude=0.25,
                mean=2600 * u.angstrom,
                stddev=340 * u.angstrom,
            ),
        },
        # FIXME: Add models for Cerenkov radiation and stray light
        # Zodiacal light, Cerenkov radiation, and Stray light dominate ULTRASAT’s background noise.
        background=GalacticBackground() + ZodiacalBackground(),
        read_noise=6,
        dark_noise=12 / 300 * u.Hz,
        gain=1,
    ),
    # ULTRASAT will be in a geosynchronous orbit similar to GOES-17.
    # This is the TLE downloaded from Celestrak at 2024-11-15T09:15:20Z.
    # https://celestrak.org/NORAD/elements/weather.txt
    observer_location=TleObserverLocation(
        "1 43226U 18022A   24320.05692005 -.00000082  00000+0  00000+0 0  9997",
        "2 43226   0.0007  47.5006 0003498 198.5164  84.4417  1.00271931 24622",
    ),
    # Sky grid optimized for ULTRASAT's wide field of view.
    skygrid={
        "allsky": _read_allsky_skygrid(),
        "non-overlap": _read_nonoverlapping_skygrid(),
    },
    # Slew model tailored for ULTRASAT's operational requirements.
    slew=EigenAxisSlew(
        max_angular_velocity=1 * u.deg / u.s,
        max_angular_acceleration=0.025 * u.deg / u.s**2,
    ),
)
ultrasat.__doc__ = r"""ULTRASAT, the Ultraviolet Transient Astronomy Satellite.

`ULTRASAT <http://www.weizmann.ac.il/ultrasat>`_ is an Israeli ultraviolet 
space telescope currently under development. It is designed to monitor the 
transient sky with a wide-field imager :footcite:`2024ApJ...964...74S`.
Expected to launch in 2027, ULTRASAT aims to provide continuous monitoring of
large areas of the sky to detect and study transient astronomical events in the
ultraviolet spectrum.

The skygrid includes 240 non-overlapping fields (7° radius) covering the entire sky 
for its low-cadence extragalactic survey. Each field is annotated with visibility 
for at least one 180-day or 45-day period per year, and average UV extinction, 
following the baseline survey strategy discussed in the `ULTRASAT Working Groups Reports 
from the September 16–18, 2024 sessions <https://www.weizmann.ac.il/ultrasat/for-scientists/working-groups/working-groups>`_.

References
----------
.. footbibliography::

Examples
--------

.. plot::
    :include-source: False
    :caption: Median signal-to-noise ratio, averaged over target coordinates and observation time.

    from astropy import units as u
    from astropy.coordinates import EarthLocation, ICRS
    from astropy.time import Time
    from astropy_healpix import HEALPix
    from matplotlib import pyplot as plt
    from m4opt.missions import ultrasat
    from m4opt.synphot import observing
    import numpy as np
    from tqdm import tqdm 
    from synphot import SourceSpectrum
    from synphot.models import BlackBody1D

    dwell = u.def_unit("dwell", 300 * u.s)
    exptime = 3 * dwell
    obstime = Time("2024-01-01") + np.linspace(0, 1) * u.year
    hpx = HEALPix(8, frame=ICRS())
    target_coords = hpx.healpix_to_skycoord(np.arange(hpx.npix))
    observer_location = EarthLocation(0 * u.m, 0 * u.m, 0 * u.m)

    g_dwarf_spectrum = SourceSpectrum(BlackBody1D, temperature=5000 * u.K)
    m_dwarf_spectrum = SourceSpectrum(BlackBody1D, temperature=3000 * u.K)
    magnitudes = np.linspace(5, 25, 100) * u.ABmag

    snrs_g = []
    snrs_m = []
    for mag in tqdm(magnitudes, desc="Calculating SNRs", unit="magnitude"):
        with observing(
            observer_location,
            target_coords[np.newaxis, :, np.newaxis],
            obstime[np.newaxis, np.newaxis, :],
        ):
            snr_g = ultrasat.detector.get_snr(
                exptime=exptime,
                source_spectrum=g_dwarf_spectrum.normalize(renorm_val=mag, band=ultrasat.detector.bandpasses['NUV'], vegaspec=None),
                bandpass='NUV',
            )
            median_snr_g = np.median(snr_g)
            snrs_g.append(median_snr_g)

            snr_m = ultrasat.detector.get_snr(
                exptime=exptime,
                source_spectrum=m_dwarf_spectrum.normalize(renorm_val=mag, band=ultrasat.detector.bandpasses['NUV'], vegaspec=None),
                bandpass='NUV',
            )
            median_snr_m = np.median(snr_m)
            snrs_m.append(median_snr_m)

    fig, ax = plt.subplots()
    ax.plot(magnitudes, snrs_g, label='G Dwarf', color='g')
    ax.plot(magnitudes, snrs_m, label='M Dwarf', color='r')
    ax.set_xlim(10, 22)
    ax.set_ylim(1e1, 1e4)
    ax.set_yscale('log')
    ax.set_xlabel("AB Magnitude")
    ax.set_ylabel('SNR')
    ax.grid(True)
    ax.legend()
"""
