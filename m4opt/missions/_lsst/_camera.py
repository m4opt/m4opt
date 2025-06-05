"""
LSST Detector positions and Camera geometry

Detector positions and LSST Camera geometry are extracted from:

- LSSTCam YAML configuration:
  https://github.com/lsst/obs_lsst/blob/main/policy/lsstCamSim.yaml

- Only science detectors are extracted (wavefront and guide sensors are excluded).
- Offsets and pixel sizes are provided in millimeters (mm).
- Data is parsed from YAML into an Astropy Table and used to construct the LSST Field of View (FOV).
"""

from importlib import resources

import yaml
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from regions import RectangleSkyRegion, Regions

from . import data

# Constants
PLATE_SCALE = 0.2 * u.arcsec  # arcsec/pixel
PIXEL_SIZE = 0.01  # mm/pixel
MM_TO_ARCSEC = PLATE_SCALE / PIXEL_SIZE

# LSST Detector Bounding Box Sizes (Pixels)
LSST_DETECTOR_SIZES = {
    "E2V": [4095, 4003],
    "ITL": [4071, 3999],
}


def read_yaml():
    """Parse LSST detector data from YAML, returning only science detectors."""
    file_path = resources.files(data) / "lsstCamSim.yaml"
    with file_path.open() as file:
        camera_data = yaml.safe_load(file)

    science_detectors = []
    for det_name, det_info in camera_data.get("CCDs", {}).items():
        if det_info.get("detectorType") != 0 or any(
            tag in det_name for tag in ["SG", "SW"]
        ):
            continue

        science_detectors.append(
            {
                "detector_name": det_name,
                "x_offset": det_info.get("offset", [0, 0])[0],
                "y_offset": det_info.get("offset", [0, 0])[1],
                "physical_type": det_info.get("physicalType", "Unknown"),
            }
        )

    return Table(
        rows=science_detectors,
        names=["detector_name", "x_offset", "y_offset", "physical_type"],
    )


def get_bbox_size(physical_type):
    """Retrieve bounding box dimensions (in pixels) for detector physical type."""
    return LSST_DETECTOR_SIZES.get(physical_type)


def make_fov():
    """Generate LSST FOV as rectangular sky regions from detector positions."""
    detectors = read_yaml()
    fov_regions = []

    for det in detectors:
        bbox_size = get_bbox_size(det["physical_type"])
        if bbox_size is None:
            continue

        # Convert offsets to arcsec, then to degrees
        x_deg = (det["x_offset"] * MM_TO_ARCSEC).to(u.deg)
        y_deg = (det["y_offset"] * MM_TO_ARCSEC).to(u.deg)

        width_deg = (bbox_size[0] * PLATE_SCALE).to(u.deg)
        height_deg = (bbox_size[1] * PLATE_SCALE).to(u.deg)

        center_coord = SkyCoord(ra=x_deg, dec=y_deg, frame="icrs")

        rect_region = RectangleSkyRegion(
            center=center_coord, width=width_deg, height=height_deg
        )
        fov_regions.append(rect_region)

    return Regions(fov_regions)
