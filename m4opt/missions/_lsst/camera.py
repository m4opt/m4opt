"""
Detector positions and LSST Camera geometry are extracted from:

    - LSSTCam: https://github.com/lsst/obs_lsst/blob/main/policy/lsstCamSim.yaml
    - Science detectors are extracted, excluding wavefront and guide sensors.
    - Offsets and pixel sizes are provided in millimeters (mm).
    - The data is parsed directly from YAML into an Astropy Table and used to make the FOV.
"""

from importlib import resources

import yaml
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from regions import PolygonSkyRegion, RectangleSkyRegion, Regions

from . import data


class LSSTCameraFOV:
    """LSST Field of View (FOV) based on detector positions."""

    PLATE_SCALE = 0.2 * u.arcsec  # arcsec/pixel
    PIXEL_SIZE = 0.01  # mm/pixel
    MM_TO_ARCSEC = PLATE_SCALE / PIXEL_SIZE

    # LSST Detector Bounding Box Sizes (Pixels)
    LSST_DETECTOR_SIZES = {
        "E2V": [4095, 4003],  # LSST detectors from E2V
        "ITL": [4071, 3999],  # LSST detectors from ITL
    }

    def __init__(self):
        """Initialize LSST camera with detector data loaded from YAML."""
        self.detectors = self.read_yaml()

    def read_yaml(self):
        """
        Read LSST science detectors from YAML (excluding wavefront and guide sensors).
        """
        file_path = resources.files(data) / "lsstCamSim.yaml"
        with open(file_path, "r") as file:
            camera_data = yaml.safe_load(file)

        science_detectors = []
        for det_name, det_info in camera_data.get("CCDs", {}).items():
            detector_type = det_info.get("detectorType")

            if detector_type != 0 or any(sub in det_name for sub in ["SG", "SW"]):
                continue

            detector_offset = det_info.get("offset", [0, 0])
            physical_type = det_info.get("physicalType", "Unknown")

            science_detectors.append(
                {
                    "detector_name": det_name,
                    "x_offset": detector_offset[0],
                    "y_offset": detector_offset[1],
                    "physical_type": physical_type,
                }
            )

        table = Table(
            rows=science_detectors,
            names=["detector_name", "x_offset", "y_offset", "physical_type"],
        )
        return table

    @staticmethod
    def get_bbox_size(physical_type):
        """Return detector bounding box size (width, height) in pixels."""
        return LSSTCameraFOV.LSST_DETECTOR_SIZES.get(physical_type, None)

    @staticmethod
    def convert_rectangle_to_polygon_skyregion(rect):
        """Convert RectangleSkyRegion to PolygonSkyRegion in RA/Dec."""
        ra, dec = rect.center.ra.deg, rect.center.dec.deg
        half_w = rect.width.to_value(u.deg) / 2
        half_h = rect.height.to_value(u.deg) / 2
        corners_ra = [ra - half_w, ra + half_w, ra + half_w, ra - half_w]
        corners_dec = [dec - half_h, dec - half_h, dec + half_h, dec + half_h]
        return PolygonSkyRegion(
            vertices=SkyCoord(ra=corners_ra * u.deg, dec=corners_dec * u.deg)
        )

    def make_fov(self):
        """Compute LSST FOV regions in RA/Dec from detector positions."""
        fov_regions = []

        for det in self.detectors:
            bbox_size = self.get_bbox_size(det["physical_type"])
            if not bbox_size:
                continue

            # Convert offsets from mm to arcsec
            x_arcsec = det["x_offset"] * self.MM_TO_ARCSEC
            y_arcsec = det["y_offset"] * self.MM_TO_ARCSEC

            # Convert bounding box size from pixels to arcsec
            width = bbox_size[0] * self.PLATE_SCALE
            height = bbox_size[1] * self.PLATE_SCALE

            # Center coordinate in degrees
            center_coord = SkyCoord(
                ra=x_arcsec.to(u.deg), dec=y_arcsec.to(u.deg), frame="icrs"
            )

            rect = RectangleSkyRegion(
                center=center_coord, width=width.to(u.deg), height=height.to(u.deg)
            )
            fov_regions.append(self.convert_rectangle_to_polygon_skyregion(rect))

        return Regions(fov_regions)
