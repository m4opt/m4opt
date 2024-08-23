"""
Calculate footprints of an instrument at arbitrary orientations.

This module provides functions to compute the footprint of the field of view of
a detector on the sky when oriented toward arbitary coordinates. We provide two
functions: :meth:`footprint` which transforms the field of view to any sky
coordinate (and optional positional angle), and :meth:`footprint_healpix` which
computes the HEALPix pixels contained within the field of view.

You supply the field of view of the detector using
:doc:`Astropy regions <regions:index>`. The following region types are
supported: :class:`~regions.CircleSkyRegion`,
:class:`~regions.PolygonSkyRegion`, :class:`~regions.RectangleSkyRegion`, and
any :class:`~regions.Regions` object consisting regions of the aforementioned
types.

.. plot::
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
            np.concatenate(footprint_healpix(hpx.nside, region, target_coords).ravel())
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

"""

from .core import footprint, footprint_healpix

__all__ = ("footprint", "footprint_healpix")
