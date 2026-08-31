from importlib import resources

import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable
from astropy.utils.data import download_file

from .. import data

UPSTREAM = "https://sncosmo.github.io/data/bandpasses/ultrasat/"

# Sample the band itself at the full resolution of the upstream table, and the
# red tail every tenth sample. Coarsening the tail any further starts to show
# up in integrated background counts.
FINE_SAMPLING_BELOW = 3200 * u.angstrom
COARSE_SAMPLING = 10


def downsample(wavelength, transmission):
    """Reduce the upstream throughput table to the vendored one."""
    split = np.searchsorted(wavelength, FINE_SAMPLING_BELOW)
    keep = np.concatenate(
        [
            np.arange(0, split),
            np.arange(split, len(wavelength) - 1, COARSE_SAMPLING),
            [len(wavelength) - 1],
        ]
    )
    return QTable({"wavelength": wavelength[keep], "transmission": transmission[keep]})


@pytest.mark.remote_data
def test_throughput_matches_upstream():
    """The vendored table is the on-axis column of the upstream ULTRASAT data.

    Regenerate it by writing the result of :func:`downsample` over
    ``data/throughput.ecsv`` in ``ascii.ecsv`` format.
    """
    wavelength = (
        np.loadtxt(download_file(f"{UPSTREAM}Wavelength.dat", cache=True)) * u.angstrom
    )
    # One column per radial distance from the optical axis; the first is on axis.
    transmission = np.loadtxt(
        download_file(f"{UPSTREAM}ULTRASAT_TR.dat", cache=True), delimiter=","
    )[:, 0]

    expected = downsample(wavelength, transmission)
    actual = QTable.read(resources.files(data) / "throughput.ecsv")

    np.testing.assert_array_equal(
        actual["wavelength"].to_value(u.angstrom),
        expected["wavelength"].to_value(u.angstrom),
    )
    np.testing.assert_array_equal(actual["transmission"], expected["transmission"])


def test_throughput_reproduces_published_values():
    """The throughput matches the values quoted in the ULTRASAT reference paper."""
    table = QTable.read(resources.files(data) / "throughput.ecsv")
    wavelength = table["wavelength"].to_value(u.angstrom)
    transmission = table["transmission"]

    def band_average(mask):
        # Weighted by wavelength, since the red tail is sampled more coarsely
        # than the band itself.
        return np.trapezoid(transmission[mask], wavelength[mask]) / np.ptp(
            wavelength[mask]
        )

    in_band = (wavelength >= 2300) & (wavelength <= 2900)
    np.testing.assert_allclose(band_average(in_band), 0.25, atol=0.005)

    out_of_band = wavelength > 3000
    np.testing.assert_allclose(band_average(out_of_band), 2.9e-5, rtol=0.05)
