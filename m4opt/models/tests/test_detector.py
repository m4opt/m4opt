import numpy as np
import pytest
from astropy import units as u
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from synphot import ConstFlux1D, SourceSpectrum, SpectralElement

from .._detector import Detector
from ..background import ZodiacalBackground


@given(
    npix=st.floats(min_value=1e-30, max_value=1e30),
    aperture_correction=st.floats(min_value=0, max_value=1, exclude_min=True),
    plate_scale=st.floats(min_value=0, max_value=1e30, exclude_min=True),
    dark_noise=st.floats(min_value=0, max_value=1e30, exclude_min=True),
    read_noise=st.floats(min_value=0, max_value=1e30, exclude_min=True),
    area=st.floats(min_value=0, max_value=1e30, exclude_min=True),
    exptime=st.floats(min_value=0, max_value=1e30),
)
@settings(suppress_health_check=[HealthCheck.filter_too_much])
@np.errstate(divide="ignore", invalid="ignore", over="ignore")
def test_detector(
    npix, aperture_correction, plate_scale, dark_noise, read_noise, area, exptime
):
    detector = Detector(
        npix=npix,
        aperture_correction=aperture_correction,
        plate_scale=plate_scale * u.steradian,
        dark_noise=dark_noise * u.Hz,
        read_noise=read_noise,
        area=area * u.cm**2,
        bandpasses={"R": SpectralElement.from_filter("johnson_r")},
        background=ZodiacalBackground.high(),
    )
    spec = SourceSpectrum(ConstFlux1D, amplitude=0 * u.ABmag)
    try:
        snr = detector.get_snr(exptime * u.s, spec)
    except OverflowError:
        assume(False)
    assume(np.isfinite(snr))
    exptime_result = detector.get_exptime(snr, spec).to_value(u.s)
    assume(np.isfinite(exptime_result))
    assert exptime_result == pytest.approx(exptime)
