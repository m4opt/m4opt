import healpy as hp
import numpy as np
import shapely
from astropy import units as u
from astropy.coordinates import (
    ICRS,
    SkyCoord,
    SkyOffsetFrame,
    UnitSphericalRepresentation,
)
from astropy.wcs import WCS
from astropy_healpix import HEALPix
from numpy import typing as npt
from regions import (
    CircleSkyRegion,
    PixCoord,
    PointSkyRegion,
    PolygonPixelRegion,
    PolygonSkyRegion,
    RectangleSkyRegion,
    Region,
    Regions,
)

query_disc = np.vectorize(
    hp.query_disc, signature="(3),()->()", excluded=[0, "nest"], otypes=[object]
)
"""Vectorized version of :meth:`healpy.query_disc`."""

query_polygon = np.vectorize(
    hp.query_polygon, signature="(n,3)->()", excluded=[0, "nest"], otypes=[object]
)
"""Vectorized version of :meth:`healpy.query_polygon`."""

ArrayOfPointSkyRegion = np.vectorize(PointSkyRegion, signature="()->()")
"""Construct a Numpy array of :class:`regions.CircularSkyRegion` instances."""

ArrayOfCircleSkyRegion = np.vectorize(CircleSkyRegion, signature="(),()->()")
"""Construct a Numpy array of :class:`regions.CircularSkyRegion` instances."""

ArrayOfRectangleSkyRegion = np.vectorize(
    RectangleSkyRegion, signature="(),(),(),()->()"
)
"""Construct a Numpy array of :class:`regions.RectangleSkyRegion` instances."""


def ArrayOfPolygonSkyRegion(vertices):
    """Construct a Numpy array of :class:`regions.PolygonSkyRegion` instances."""
    shape = vertices.shape[:-1]
    regions = np.empty(shape, dtype=object)
    for i in np.ndindex(shape):
        regions[i] = PolygonSkyRegion(vertices[i])
    return regions


def ArrayOfRegions(first, *rest):
    """Construct a Numpy array of :class:`regions.Regions` instances."""
    regions = np.empty_like(first)
    for i in np.ndindex(regions.shape):
        regions[i] = Regions([arg[i] for arg in (first, *rest)])
    return regions


def concat_healpix(shape, *args):
    regions = np.empty(shape, dtype=object)
    for i in np.ndindex(regions.shape):
        regions[i] = np.unique(np.concatenate([arg[i] for arg in args]))
    return regions


def unwrap_scalar(a):
    if isinstance(a, np.ndarray) and a.ndim == 0:
        a = a.item()
    return a


def skycoord_to_offset(coord: SkyCoord, frame: SkyOffsetFrame):
    return SkyCoord(coord.icrs.data, frame=frame).icrs


def skycoord_to_healpy_vec(coord: SkyCoord):
    return np.moveaxis(coord.cartesian.xyz.value, 0, -1)


def circle_to_polygon(region: CircleSkyRegion, n: int) -> PolygonSkyRegion:
    """Convert a circle region to a polygon that approximates it.

    Parameters
    ----------
    region
        The circle to approximate.
    n
        The number of vertices.

    Notes
    -----
    Unlike :meth:`regions.CircleSkyRegion.to_polygon`, this function does not
    require a :class:`~astropy.wcs.WCS` object.
    """
    return PolygonSkyRegion(
        SkyCoord(
            region.radius,
            0 * u.deg,
            frame=region.center.skyoffset_frame(
                np.linspace(0, 360, n, endpoint=False) * u.deg
            ),
        ).icrs
    )


def rectangle_to_polygon(region: RectangleSkyRegion):
    """Convert a rectangle region to a polygon.

    Rotated rectangle regions do not correctly account for spherical geometry,
    but polygon regions do.
    """
    x = 0.5 * region.width
    y = 0.5 * region.height
    return PolygonSkyRegion(
        skycoord_to_offset(
            SkyCoord([-x, x, x, -x], [-y, -y, y, y]),
            region.center.skyoffset_frame(region.angle),
        )
    )


def is_convex(region: PolygonSkyRegion) -> bool:
    """Check if a spherical polygon is convex

    Notes
    -----
    This should agree exactly with the convexity check in the query_polygon
    function of healpy/healpix-cxx."""
    coords = region.vertices.cartesian
    dotprods = coords.cross(np.roll(coords, 1)).dot(np.roll(coords, 2))
    signs = np.sign(dotprods)
    return (np.all(np.abs(dotprods) >= 1e-10) and np.all(signs[1:] == signs[0])).item()


def centered_wcs(region: PolygonSkyRegion) -> WCS:
    """Create a local WCS centered on the polygon."""
    center = (
        region.vertices.transform_to(ICRS())
        .represent_as(UnitSphericalRepresentation)
        .sum()
    )
    return WCS(
        {
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRVAL1": center.lon.deg,
            "CRVAL2": center.lat.deg,
            "CUNIT1": "deg",
            "CUNIT2": "deg",
        }
    )


def footprint_inner(region: Region | Regions, frame: SkyOffsetFrame):
    match region:
        case Regions():
            return ArrayOfRegions(
                *(footprint_inner(subregion, frame) for subregion in region.regions)
            )
        case CircleSkyRegion():
            return ArrayOfCircleSkyRegion(
                skycoord_to_offset(region.center, frame), region.radius
            )
        case PointSkyRegion():
            return ArrayOfPointSkyRegion(skycoord_to_offset(region.center, frame))
        case PolygonSkyRegion():
            return ArrayOfPolygonSkyRegion(
                skycoord_to_offset(region.vertices, frame[..., np.newaxis])
            )
        case RectangleSkyRegion():
            return footprint_inner(rectangle_to_polygon(region), frame)
        case _:
            raise NotImplementedError(
                f"Footprint transformations are not implemented for {region.__class__.__name__}"
            )


def footprint(
    region: Region | Regions,
    target_coord: SkyCoord,
    rotation: u.Quantity[u.physical.angle] | None = None,
):
    """
    Transform a region to the desired target coordinate and optional rotation.

    The region is expected to represent the field of view of an instrument at
    a standard orientation of R.A.=0, Dec.=0, P.A.=0. This function rotates the
    region as if the instrument is pointed at the given target coordinate and
    optional rotation.

    The target coordinate and rotation may be arrays; in that case the return
    value is a Numpy array of regions.

    Parameters
    ----------
    region:
        The shape of the field of view in the standard orientation.
    target_coord:
        The position for the center of the field of view.
    rotation:
        The rotation of the field of view about its center.

    Examples
    --------

    First, some imports:

    >>> from regions import CircleSkyRegion, EllipseSkyRegion, PointSkyRegion, PolygonSkyRegion, RectangleSkyRegion, Regions
    >>> from astropy.coordinates import SkyCoord
    >>> from astropy import units as u
    >>> from m4opt.fov import footprint
    >>> import numpy as np

    We support pointlike FOVs:

    >>> region = PointSkyRegion(SkyCoord(0 * u.deg, 0 * u.deg))
    >>> target_coord = SkyCoord(5 * u.deg, -5 * u.deg)
    >>> footprint(region, target_coord)
    <PointSkyRegion(center=<SkyCoord (ICRS): (ra, dec, distance) in (deg, deg, )
        (5., -5., 1.)>)>

    and circular FOVs:

    >>> region = CircleSkyRegion(SkyCoord(0 * u.deg, 0 * u.deg), 3 * u.deg)
    >>> target_coord = SkyCoord(5 * u.deg, -5 * u.deg)
    >>> footprint(region, target_coord)
    <CircleSkyRegion(center=<SkyCoord (ICRS): (ra, dec, distance) in (deg, deg, )
        (5., -5., 1.)>, radius=3.0 deg)>

    and rectangular FOVs:

    >>> region2 = RectangleSkyRegion(SkyCoord(0 * u.deg, 0 * u.deg), 6 * u.deg, 8 * u.deg)
    >>> target_coord = SkyCoord(5 * u.deg, -5 * u.deg)
    >>> footprint(region2, target_coord)
    <PolygonSkyRegion(vertices=<SkyCoord (ICRS): (ra, dec, distance) in (deg, deg, )
        [(1.97003363, -8.99308801, 1.), (8.02996637, -8.99308801, 1.),
         (7.99313556, -0.99317201, 1.), (2.00686444, -0.99317201, 1.)]>)>

    We can compute the footprints for an array of target coordinates:

    >>> ras, decs = np.meshgrid([0, 1], [2, 3]) * u.deg
    >>> target_coords = SkyCoord(ras, decs)
    >>> footprint(region, target_coords)
    array([[<CircleSkyRegion(center=<SkyCoord (ICRS): (ra, dec, distance) in (deg, deg, )
                (0., 2., 1.)>, radius=3.0 deg)>                                          ,
            <CircleSkyRegion(center=<SkyCoord (ICRS): (ra, dec, distance) in (deg, deg, )
                (1., 2., 1.)>, radius=3.0 deg)>                                          ],
           [<CircleSkyRegion(center=<SkyCoord (ICRS): (ra, dec, distance) in (deg, deg, )
                (0., 3., 1.)>, radius=3.0 deg)>                                          ,
            <CircleSkyRegion(center=<SkyCoord (ICRS): (ra, dec, distance) in (deg, deg, )
                (1., 3., 1.)>, radius=3.0 deg)>                                          ]],
          dtype=object)

    We support polygon regions:

    >>> region = PolygonSkyRegion(SkyCoord([-2, 2, 0] * u.deg, [0, 0, 2] * u.deg))
    >>> footprint(region, target_coord)
    <PolygonSkyRegion(vertices=<SkyCoord (ICRS): (ra, dec, distance) in (deg, deg, )
        [(2.99236656, -4.99694639, 1.), (7.00763344, -4.99694639, 1.),
         (5.        , -3.        , 1.)]>)>

    And arrays of target coordinates:

    >>> footprint(region, target_coords)
    array([[<PolygonSkyRegion(vertices=<SkyCoord (ICRS): (ra, dec, distance) in (deg, deg, )
                [(357.9987819, 1.99878116, 1.), (  2.0012181, 1.99878116, 1.),
                 (  0.       , 4.        , 1.)]>)>                                          ,
            <PolygonSkyRegion(vertices=<SkyCoord (ICRS): (ra, dec, distance) in (deg, deg, )
                [(358.9987819, 1.99878116, 1.), (  3.0012181, 1.99878116, 1.),
                 (  1.       , 4.        , 1.)]>)>                                          ],
           [<PolygonSkyRegion(vertices=<SkyCoord (ICRS): (ra, dec, distance) in (deg, deg, )
                [(357.99725754, 2.99817081, 1.), (  2.00274246, 2.99817081, 1.),
                 (  0.        , 5.        , 1.)]>)>                                         ,
            <PolygonSkyRegion(vertices=<SkyCoord (ICRS): (ra, dec, distance) in (deg, deg, )
                [(358.99725754, 2.99817081, 1.), (  3.00274246, 2.99817081, 1.),
                 (  1.        , 5.        , 1.)]>)>                                         ]],
          dtype=object)

    Compound regions are also fine:

    >>> regions = Regions([
    ...     CircleSkyRegion(SkyCoord(0 * u.deg, 0 * u.deg), 3 * u.deg),
    ...     PolygonSkyRegion(SkyCoord([-2, 2, 0] * u.deg, [0, 0, 2] * u.deg))])
    >>> footprint(regions, target_coord)
    <Regions([<CircleSkyRegion(center=<SkyCoord (ICRS): (ra, dec, distance) in (deg, deg, )
        (5., -5., 1.)>, radius=3.0 deg)>, <PolygonSkyRegion(vertices=<SkyCoord (ICRS): (ra, dec, distance) in (deg, deg, )
        [(2.99236656, -4.99694639, 1.), (7.00763344, -4.99694639, 1.),
         (5.        , -3.        , 1.)]>)>])>

    Not all region types are supported:

    >>> region = EllipseSkyRegion(SkyCoord(0 * u.deg, 0 * u.deg), 5 * u.deg, 2 * u.deg)
    >>> footprint(region, target_coord)
    Traceback (most recent call last):
      ...
    NotImplementedError: Footprint transformations are not implemented for EllipseSkyRegion
    """
    return unwrap_scalar(
        footprint_inner(region, target_coord.skyoffset_frame(rotation))
    )


def footprint_healpix_convex_polygon_inner(
    hpx: HEALPix, region: PolygonSkyRegion, frame: SkyOffsetFrame
):
    return query_polygon(
        hpx.nside,
        skycoord_to_healpy_vec(
            skycoord_to_offset(region.vertices, frame[..., np.newaxis])
        ),
        nest=hpx.order == "nested",
    )


def footprint_healpix_inner(
    hpx: HEALPix, region: Region | Regions, frame: SkyOffsetFrame
):
    match region:
        case Regions():
            return concat_healpix(
                frame.shape,
                *(
                    footprint_healpix_inner(hpx, subregion, frame)
                    for subregion in region.regions
                ),
            )
        case CircleSkyRegion():
            return query_disc(
                hpx.nside,
                skycoord_to_healpy_vec(skycoord_to_offset(region.center, frame)),
                region.radius.to_value(u.rad),
                nest=hpx.order == "nested",
            )
        case PointSkyRegion():
            return hpx.skycoord_to_healpix(skycoord_to_offset(region.center, frame))[
                ..., np.newaxis
            ]
        case PolygonSkyRegion():
            if is_convex(region):
                return footprint_healpix_convex_polygon_inner(hpx, region, frame)
            else:
                wcs = centered_wcs(region)
                pixel_region = region.to_pixel(wcs)
                triangles: list[shapely.Polygon] = (
                    shapely.constrained_delaunay_triangles(
                        shapely.polygons(np.transpose(pixel_region.vertices.xy))
                    )
                ).geoms
                return concat_healpix(
                    frame.shape,
                    *(
                        footprint_healpix_convex_polygon_inner(
                            hpx,
                            PolygonPixelRegion(
                                PixCoord(*np.transpose(triangle.exterior.coords[:-1]))
                            ).to_sky(wcs),
                            frame,
                        )
                        for triangle in triangles
                    ),
                )
        case RectangleSkyRegion():
            return footprint_healpix_inner(hpx, rectangle_to_polygon(region), frame)
        case _:
            raise NotImplementedError(
                f"Footprint transformations are not implemented for {region.__class__.__name__}"
            )


def footprint_healpix(
    hpx: HEALPix,
    region: Region | Regions,
    target_coord: SkyCoord | None = None,
    rotation: u.Quantity[u.physical.angle] | None = None,
):
    """
    Calculate the HEALPix pixels inside an observing footprint.

    The region is expected to represent the field of view of an instrument at
    a standard orientation of R.A.=0, Dec.=0, P.A.=0. This function rotates the
    region as if the instrument is pointed at the given target coordinate and
    optional rotation.

    The target coordinate and rotation may be arrays; in that case the return
    value is a Numpy array of arrays of uneven length.

    Parameters
    ----------
    hpx:
        The HEALPix object specifying the ordering and resolution.
    region:
        The shape of the field of view in the standard orientation.
    target_coord:
        The position for the center of the field of view.
    rotation:
        The rotation of the field of view about its center.

    Examples
    --------

    First, some imports:

    >>> from astropy.coordinates import ICRS, SkyCoord
    >>> from astropy import units as u
    >>> from astropy_healpix import HEALPix
    >>> from m4opt.fov import footprint_healpix
    >>> import numpy as np
    >>> from regions import CircleSkyRegion, EllipseSkyRegion, PointSkyRegion, PolygonSkyRegion, RectangleSkyRegion, Regions

    We support pointlike FOVs:

    >>> hpx = HEALPix(nside=32, frame=ICRS())
    >>> region = PointSkyRegion(SkyCoord(0 * u.deg, 0 * u.deg))
    >>> target_coord = SkyCoord(5 * u.deg, -5 * u.deg)
    >>> footprint_healpix(hpx, region, target_coord)
    array([6593])

    We support circular FOVs:

    >>> hpx = HEALPix(nside=32, frame=ICRS())
    >>> region = CircleSkyRegion(SkyCoord(0 * u.deg, 0 * u.deg), 3 * u.deg)
    >>> target_coord = SkyCoord(5 * u.deg, -5 * u.deg)
    >>> footprint_healpix(hpx, region, target_coord)
    array([6337, 6465, 6466, 6593, 6594, 6721, 6722, 6849, 6850])

    and rectangular FOVs:

    >>> region2 = RectangleSkyRegion(SkyCoord(0 * u.deg, 0 * u.deg), 6 * u.deg, 8 * u.deg)
    >>> target_coord = SkyCoord(5 * u.deg, -5 * u.deg)
    >>> footprint_healpix(hpx, region2, target_coord)
    array([6209, 6210, 6337, 6338, 6465, 6466, 6593, 6594, 6721, 6722, 6849,
           6850, 6977, 6978])

    We can compute the footprints for an array of target coordinates:

    >>> ras, decs = np.meshgrid([0, 1], [2, 3]) * u.deg
    >>> target_coords = SkyCoord(ras, decs)
    >>> footprint_healpix(hpx, region, target_coords)
    array([[array([5696, 5824, 5951, 5952, 5953, 6079, 6080, 6207]),
            array([5568, 5696, 5697, 5824, 5951, 5952, 5953, 6080])],
           [array([5440, 5568, 5695, 5696, 5697, 5823, 5824, 5951, 5952]),
            array([5568, 5695, 5696, 5697, 5824, 5951, 5952, 5953])]],
          dtype=object)

    We support polygon regions:

    >>> region = PolygonSkyRegion(SkyCoord([-2, 2, 0] * u.deg, [0, 0, 2] * u.deg))
    >>> footprint_healpix(hpx, region, target_coord)
    array([6593])

    And arrays of target coordinates:

    >>> footprint_healpix(hpx, region, target_coords)
    array([[array([5696, 5824, 5951]), array([5824])],
           [array([5696]), array([5696])]], dtype=object)

    And even non-convex polygon regions:

    >>> region = PolygonSkyRegion(SkyCoord([-2, 0, 2, 2, -2] * u.deg, [2, 0, 2, -2, -2] * u.deg))
    >>> footprint_healpix(hpx, region, target_coord)
    array([6593, 6722])

    Compound regions are also fine:

    >>> regions = Regions([
    ...     CircleSkyRegion(SkyCoord(0 * u.deg, 0 * u.deg), 3 * u.deg),
    ...     PolygonSkyRegion(SkyCoord([-2, 2, 0] * u.deg, [0, 0, 2] * u.deg))])
    >>> footprint_healpix(hpx, regions, target_coord)
    array([6337, 6465, 6466, 6593, 6594, 6721, 6722, 6849, 6850])

    Not all region types are supported:

    >>> region = EllipseSkyRegion(SkyCoord(0 * u.deg, 0 * u.deg), 5 * u.deg, 2 * u.deg)
    >>> footprint_healpix(hpx, region, target_coord)
    Traceback (most recent call last):
      ...
    NotImplementedError: Footprint transformations are not implemented for EllipseSkyRegion
    """
    if target_coord is None:
        target_coord = SkyCoord(0 * u.deg, 0 * u.deg)
    return unwrap_scalar(
        footprint_healpix_inner(hpx, region, target_coord.skyoffset_frame(rotation))
    )


def contains(region: Region | Regions, target_coord: SkyCoord) -> npt.NDArray[np.bool_]:
    """
    Test if a region contains a given sky coordinate.

    This is similar to :meth:`regions.SkyRegion.contains`, but does not require
    you to specify a :class:`~astropy.wcs.WCS`.

    Parameters
    ----------
    region
        A sky region.
    target_coord
        The coordinates to test.

    Returns
    -------
    :
        A flag indicating whether each targert coordinate is contained within
        the region.

    Notes
    -----
    Edges of polygons and rectangles are assumed to be great circle arcs.

    When using a :class:`regions.PolygonSkyRegion`, this method is only valid
    for polygons that fit in a single hemisphere, because it relies on
    transforming the polygon to a gnomonic projection, which is a projection
    of half of the sphere in which great circles are straight lines.

    Examples
    --------
    >>> from astropy.coordinates import SkyCoord
    >>> from astropy import units as u
    >>> from regions import CircleSkyRegion, PolygonSkyRegion, RectangleSkyRegion
    >>> from m4opt.fov import contains
    >>> region = CircleSkyRegion(SkyCoord(0 * u.deg, 0 * u.deg), 5 * u.deg)
    >>> contains(region, SkyCoord(0 * u.deg, 0 * u.deg))
    np.True_
    >>> region = PolygonSkyRegion(SkyCoord([359, 1, 1, 359] * u.deg, [-2, -2, 2, 2] * u.deg))
    >>> contains(region, SkyCoord(0 * u.deg, 0 * u.deg))
    True
    >>> region = RectangleSkyRegion(SkyCoord(0 * u.deg, 0 * u.deg), 2 * u.deg, 4 * u.deg)
    >>> contains(region, SkyCoord(0 * u.deg, 0 * u.deg))
    True
    """
    match region:
        case Regions():
            return np.logical_or.reduce(
                [contains(r, target_coord) for r in region.regions]
            )
        case CircleSkyRegion():
            return region.center.separation(target_coord) <= region.radius
        case RectangleSkyRegion():
            return contains(rectangle_to_polygon(region), target_coord)
        case PolygonSkyRegion():
            result = region.contains(target_coord, centered_wcs(region))
            if target_coord.isscalar:
                return result.item()
            else:
                return result
        case _:
            raise NotImplementedError
