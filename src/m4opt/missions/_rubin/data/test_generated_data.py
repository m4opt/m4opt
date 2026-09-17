import pytest
import yaml
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.utils.data import get_readable_fileobj
from regions import RectangleSkyRegion, Regions


@pytest.mark.remote_data
def test_cam_dot_ds9(generated_file):
    """Generate LSST camera footprint region file."""
    with get_readable_fileobj(
        "https://github.com/lsst/obs_lsst/raw/refs/tags/w.2025.52/policy/lsstCamSim.yaml"
    ) as f:
        yaml_data = yaml.safe_load(f)
    cams = Table(list(yaml_data["CCDs"].values()))
    PLATE_SCALE = 0.2 * u.arcsec
    out = Regions(
        [
            RectangleSkyRegion(
                SkyCoord(*(row["offset"][:2] * PLATE_SCALE / row["pixelSize"])),
                *(row["bbox"][1] * PLATE_SCALE),
            )
            for row in cams
            if row["detectorType"] == 0  # Science only detectors
        ]
    ).serialize("ds9")
    generated_file(out)
