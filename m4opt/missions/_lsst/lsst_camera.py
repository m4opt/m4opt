"""
Detector positions and LSST Camera geometry are extracted from:

    - LSSTCam: https://github.com/lsst/obs_lsst/blob/main/policy/lsstCamSim.yaml
    - Science detectors are extracted, excluding wavefront and guide sensors.
    - Offsets and pixel sizes are provided in millimeters (mm).
    - The extracted data is saved in CSV formats for or processing.
"""

from importlib import resources

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from regions import PolygonSkyRegion, RectangleSkyRegion, Regions

from . import data


class LSSTfieldOfView:
    """LSST Field of View (FOV) based on detector positions."""

    # LSST-specific plate scale and pixel size
    PLATE_SCALE = 0.2 * u.arcsec  # arcsec/pixel
    PIXEL_SIZE = 0.01  # mm/pixel
    MM_TO_ARCSEC = PLATE_SCALE / PIXEL_SIZE

    # LSST Detector Bounding Box Sizes (Pixels)
    LSST_DETECTOR_SIZES = {
        "E2V": [4095, 4003],  # LSST detectors from E2V
        "ITL": [4071, 3999],  # LSST detectors from ITL
    }

    def __init__(self):
        """Initialize with LSST detector data from an internal package resource."""
        self.detectors = self.read_csv()

    @staticmethod
    def read_csv():
        """Read the LSST detector file."""
        file_path = resources.files(data) / "lsst_science_detectors.csv"
        with file_path.open("rb") as f:
            return Table.read(f, format="csv")

    @staticmethod
    def get_bbox_size(physical_type):
        """Return the LSST detector bounding box size (width, height in pixels)."""
        return LSSTfieldOfView.LSST_DETECTOR_SIZES.get(physical_type, None)

    @staticmethod
    def convert_rectangle_to_polygon_skyregion(rect):
        """Convert a RectangleSkyRegion to a PolygonSkyRegion in RA/Dec."""
        ra, dec = rect.center.ra.deg, rect.center.dec.deg
        half_w, half_h = rect.width.to_value(u.deg) / 2, rect.height.to_value(u.deg) / 2
        corners_ra = [ra - half_w, ra + half_w, ra + half_w, ra - half_w]
        corners_dec = [dec - half_h, dec - half_h, dec + half_h, dec + half_h]
        return PolygonSkyRegion(
            vertices=SkyCoord(ra=corners_ra * u.deg, dec=corners_dec * u.deg)
        )

    def make_fov(self):
        """LSST FOV regions in RA/Dec from detector positions."""
        fov_regions = []

        for det in self.detectors:
            bbox_size = self.get_bbox_size(det["physical_type"])
            if not bbox_size:
                continue

            # Convert offsets from mm to arcsec
            x_arcsec = det["x_offset"] * self.MM_TO_ARCSEC
            y_arcsec = det["y_offset"] * self.MM_TO_ARCSEC

            # Convert bounding box size from pixels to arcsec
            width, height = bbox_size * self.PLATE_SCALE

            # Convert to celestial coordinates (RA/Dec)
            center_coord = SkyCoord(ra=x_arcsec, dec=y_arcsec, frame="icrs")
            rect = RectangleSkyRegion(center=center_coord, width=width, height=height)
            fov_regions.append(self.convert_rectangle_to_polygon_skyregion(rect))

        return Regions(fov_regions)
