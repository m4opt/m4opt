import astropy.units as u
from astropy.coordinates import SkyCoord
from regions import PolygonSkyRegion, RectangleSkyRegion, Regions


class UltrasatCameraFOV:
    """
    Class representing the field-of-view (FOV) of the ULTRASAT camera.

    Attributes
    ----------
    PIXEL_SIZE : Quantity
        Physical pixel size of the ULTRASAT sensor in mm/pixel (:footcite:`2021SPIE11821E..0UA`).
    N_PIXELS : Quantity
        Number of pixels per detector axis (:footcite:`2021SPIE11821E..0UA`).
    PLATE_SCALE : Quantity
        Plate scale of the ULTRASAT camera in arcsec/pixel (:footcite:`2024ApJ...964...74S`).

    Parameters
    ----------
    center_ra : Quantity, optional
        Right ascension coordinate of the camera center. Default is 0 arcsec.
    center_dec : Quantity, optional
        Declination coordinate of the camera center. Default is 0 arcsec.
    """

    PIXEL_SIZE = 9.5e-3 * (u.mm / u.pixel)
    N_PIXELS = 4738 * u.pixel
    SENSOR_SIZE_MM = PIXEL_SIZE * N_PIXELS  # 45.011 mm
    PLATE_SCALE = 5.4 * (u.arcsec / u.pixel)

    # Total angular size per detector (one tile) in arseconds
    TILE_SIZE_ARCSEC = N_PIXELS * PLATE_SCALE

    def __init__(self, center_ra=0 * u.arcsec, center_dec=0 * u.arcsec):
        self.center = SkyCoord(ra=center_ra, dec=center_dec, frame="icrs")
        self.fov_regions = self.make_fov()

    def make_fov(self):
        # Offset is half of the TILE_SIZE in degrees
        offset_arcsec = (self.TILE_SIZE_ARCSEC / 2).value

        # Four detectors in a 2x2 grid (assuming no gaps)
        offsets = [
            (-offset_arcsec, offset_arcsec),
            (offset_arcsec, offset_arcsec),
            (-offset_arcsec, -offset_arcsec),
            (offset_arcsec, -offset_arcsec),
        ]

        fov_regions = []
        for d_ra, d_dec in offsets:
            # Convert to celestial coordinates (RA/Dec)
            center_coord = SkyCoord(
                ra=self.center.ra + d_ra * u.arcsec,
                dec=self.center.dec + d_dec * u.arcsec,
                frame="icrs",
            )
            rect = RectangleSkyRegion(
                center=center_coord,
                width=self.TILE_SIZE_ARCSEC,
                height=self.TILE_SIZE_ARCSEC,
            )
            fov_regions.append(rect)

        return Regions(self.to_polygons(fov_regions))

    def to_polygons(self, fov_regions):
        """Convert rectangles to polygons."""
        polygons = []
        for rect in fov_regions:
            ra = rect.center.ra.deg
            dec = rect.center.dec.deg
            half_w = rect.width.to(u.deg).value / 2
            half_h = rect.height.to(u.deg).value / 2
            corners_ra = [ra - half_w, ra + half_w, ra + half_w, ra - half_w]
            corners_dec = [dec - half_h, dec - half_h, dec + half_h, dec + half_h]
            polygons.append(
                PolygonSkyRegion(
                    vertices=SkyCoord(ra=corners_ra * u.deg, dec=corners_dec * u.deg)
                )
            )
        return polygons
