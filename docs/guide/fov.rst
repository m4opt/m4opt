****************************
Fields of View (`m4opt.fov`)
****************************

This module provides functions to compute the footprint of the field of view of
a detector on the sky when oriented toward arbitrary coordinates. We provide two
functions: :meth:`~m4opt.fov.footprint` which transforms the field of view to
any sky coordinate (and optional positional angle), and
:meth:`~m4opt.fov.footprint_healpix` which computes the HEALPix pixels
contained within the field of view.

Supported region types
----------------------

You supply the field of view of the detector using
:doc:`Astropy regions <regions:index>`. The following region types are
supported: :class:`~regions.CircleSkyRegion`,
:class:`~regions.PolygonSkyRegion`, :class:`~regions.RectangleSkyRegion`, and
any :class:`~regions.Regions` object consisting regions of the aforementioned
types.

.. note:: All of these region types are treated as true spherical geometry.
    :class:`~regions.CircleSkyRegion` is treated as a
    `spherical cap <https://mathworld.wolfram.com/SphericalCap.html>`_.
    :class:`~regions.PolygonSkyRegion` and :class:`~regions.RectangleSkyRegion`
    are treated as `spherical polygons <https://mathworld.wolfram.com/SphericalPolygon.html>`_
    whose edges are great circles.

    Some other astronomy software (for example,
    `DS9 <https://sites.google.com/cfa.harvard.edu/saoimageds9>`_) does not
    distinguish between planar and spherical geometry, and the Astropy regions
    package itself has some ambiguity here as well (see
    `astropy/regions#276 <https://github.com/astropy/regions/issues/276>`_).

.. plot::
    :caption: Gallery of supported region types
    :include-source: False

    import numpy as np
    import regions
    from astropy import units as u
    from astropy.coordinates import ICRS, SkyCoord
    from astropy_healpix import HEALPix
    from ligo.skymap import plot
    from matplotlib import pyplot as plt

    from m4opt.fov import footprint, footprint_healpix

    hpx = HEALPix(nside=64, frame=ICRS())

    regions = {
        "circle": regions.CircleSkyRegion(
            center=SkyCoord(0 * u.deg, 0 * u.deg), radius=5 * u.deg
        ),
        "polygon": regions.PolygonSkyRegion(
            SkyCoord(
                10 * np.asarray([-0.5, 0.5, 0]) * u.deg,
                10 * np.asarray([-np.sqrt(3) / 4, -np.sqrt(3) / 4, np.sqrt(3) / 4]) * u.deg,
            )
        ),
        "rectangle": regions.RectangleSkyRegion(
            center=SkyCoord(0 * u.deg, 0 * u.deg), width=8 * u.deg, height=12 * u.deg
        ),
        "rotated rectangle": regions.RectangleSkyRegion(
            center=SkyCoord(0 * u.deg, 0 * u.deg),
            width=8 * u.deg,
            height=12 * u.deg,
            angle=15 * u.deg,
        ),
    }

    fig, axs = plt.subplots(
        len(regions),
        3,
        subplot_kw=dict(
            projection="astro degrees zoom", center="0deg 0deg", radius="32deg"
        ),
        figsize=(5, 7),
        tight_layout=True,
    )
    for axrow, (key, region) in zip(axs, regions.items()):
        ax = axrow[0]
        ax.add_artist(region.to_pixel(ax.wcs).as_artist())
        ax.set_ylabel(key)

        ax = axrow[1]
        target_coords = SkyCoord(*(np.meshgrid([-15, 0, 15], [-15, 0, 15]) * u.deg))
        for footprint_region in footprint(region, target_coords).ravel():
            ax.add_artist(footprint_region.to_pixel(ax.wcs).as_artist())

        ax = axrow[2]
        pixels = np.unique(
            np.concatenate(footprint_healpix(hpx, region, target_coords).ravel())
        )
        for verts in hpx.boundaries_skycoord(pixels, 1):
            ax.add_patch(
                plt.Polygon(
                    np.column_stack((verts.ra.deg, verts.dec.deg)),
                    transform=ax.get_transform("world"),
                )
            )

    axs[0, 0].set_title("Region")
    axs[0, 1].set_title("Footprints")
    axs[0, 2].set_title("HEALPix")

    for ax in axs.ravel():
        ax.grid()

    for ax in axs[1:, :].ravel():
        ax.coords["ra"].set_axislabel("", visible=False)

    for ax in axs[:-1, :].ravel():
        ax.coords["ra"].set_ticklabel_visible(False)

    for ax in axs[:, 1:].ravel():
        ax.coords["dec"].set_ticklabel_visible(False)
        ax.coords["dec"].set_axislabel("", visible=False)

Regions from files
------------------

You can also read field of view shapes from common region files using
:meth:`regions.Regions.read`. For example, here is a
`DS9 region file <https://ds9.si.edu/doc/ref/region.html>`_ describing the
field of view of the `Wide-Field Instrument <https://roman.gsfc.nasa.gov/science/WFI_technical.html>`_
on the `Nancy Grace Roman Space Telescope <https://roman.gsfc.nasa.gov>`_:

.. plot::
    :include-source: False
    :show-source-link: False
    :nofigs:

    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import pysiaf
    from regions import PolygonSkyRegion, Regions
    import re

    attmat = pysiaf.utils.rotations.attitude_matrix(0, 0, 0, 0, 0)
    roman = pysiaf.Siaf('roman')
    regions = Regions()
    for aper_name, aper in roman.apertures.items():
        if re.match('^WFI\d\d_FULL$', aper_name):
            aper.set_attitude_matrix(attmat)
            regions.append(PolygonSkyRegion(SkyCoord(*aper.corners('sky'), unit=u.deg)))
    regions.write('roman_wfi.ds9', overwrite=True)

.. literalinclude:: roman_wfi.ds9
    :caption: roman_wfi.ds9
    :language: text

The following example code reads the region file and plots a grid of footprints:

.. plot::

    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import ligo.skymap.plot
    from matplotlib import pyplot as plt
    from m4opt.fov import footprint
    import numpy as np
    from regions import Regions

    # Read shapes from a DS9 region file
    roman_wfi = Regions.read('roman_wfi.ds9')

    # Create a grid of target coordinates
    ra, dec = np.meshgrid(
        np.linspace(-1.5, 0.5, 4) * u.deg,
        np.linspace(-1, 1, 3) * u.deg
    )
    target_coords = SkyCoord(ra, dec).ravel()

    # Transform shapes to target coordinates
    footprints = footprint(roman_wfi, target_coords, -30 * u.deg)

    # Plot footprints
    ax = plt.axes(
        projection='astro zoom',
        center=SkyCoord(0 * u.deg, 0 * u.deg),
        radius=2 * u.deg)
    for regions in footprints:
        for region in regions:
            ax.add_patch(region.to_pixel(ax.wcs).as_artist())
    ax.grid()


.. automodapi:: m4opt.fov
