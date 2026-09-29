from importlib import resources
from unittest.mock import patch

import pytest
import uvex_imager_etc.etc
import uvex_imager_etc.uvex
from astropy import units as u
from astropy.coordinates import EarthLocation
from synphot import ConstFlux1D, SourceSpectrum

from ....synphot import observing
from .. import data, uvex


@pytest.mark.parametrize("bandpass", ["NUV", "FUV"])
@pytest.mark.parametrize("exptime", [300 * u.s, 900 * u.s])
@pytest.mark.parametrize("snr", [5, 10])
def test_etc(bandpass, exptime, snr):
    """Test against the mission-supported uvex-imager-etc."""
    # FIXME: uvex-imager-etc currently has no way to set the CALDB directory.
    # See https://github.com/uvex-mission/uvex-imager-etc/issues/24.
    # For now, monkeypatch it.
    with patch.object(
        uvex_imager_etc.uvex, "response_files_dir", resources.files(data)
    ):
        telescope = uvex_imager_etc.uvex.UVEX()
    etc = uvex_imager_etc.etc.ETC(telescope=telescope)
    expected = etc.get_limiting_mag(
        snr=snr, exptime=exptime, n_frames=1, band=bandpass.lower()
    ).value

    observer_location = EarthLocation.from_geocentric(0 * u.m, 0 * u.m, 0 * u.m)
    source_spectrum = SourceSpectrum(ConstFlux1D, amplitude=0 * u.ABmag)
    with observing(
        target_coord=etc.default_coord,
        obstime=etc.default_obstime,
        observer_location=observer_location,
    ):
        result = uvex.detector.get_limmag(
            snr=snr, exptime=exptime, source_spectrum=source_spectrum, bandpass=bandpass
        ).value

    # FIXME: Decrease test tolerance after we have matched background models.
    assert expected == pytest.approx(result, abs=0.15)
