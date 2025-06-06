***********************************
Mission Profiles (`m4opt.missions`)
***********************************

.. plot::
    :caption: Gallery of FOVs of supported missions
    :include-source: False

    import ligo.skymap.plot  # noqa: F401
    import numpy as np
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS
    from matplotlib import pyplot as plt
    from matplotlib.transforms import Affine2D
    from regions import Regions

    import m4opt.fov
    import m4opt.missions
    from m4opt.utils.optimization import pack_boxes

    missions = [
        obj
        for obj in (getattr(m4opt.missions, key) for key in m4opt.missions.__all__)
        if isinstance(obj, m4opt.missions.Mission)
    ]

    wcs = WCS(
        dict(
            CTYPE1="RA---CAR",
            CTYPE2="DEC--CAR",
            CRPIX1=1,
            CRPIX2=1,
        )
    )

    origin = SkyCoord(0 * u.deg, 0 * u.deg)
    fovs = []
    for mission in missions:
        regions = m4opt.fov.footprint(mission.fov, origin)
        if isinstance(regions, Regions):
            regions = regions.regions
        else:
            regions = [regions]
        fovs.append([region.to_pixel(wcs) for region in regions])

    widths = 2.1 * np.asarray(
        [max(np.abs(region.vertices.xy).max() for region in regions) for regions in fovs]
    )
    widths = np.column_stack((widths, widths))
    xy, dims = pack_boxes(widths)

    if dims[0] < dims[1]:
        xy = np.fliplr(xy)
        dims = np.flipud(dims)

    ax = plt.axes(aspect=1, frameon=False)
    ax.set_xlim(0, dims[0])
    ax.set_ylim(0, dims[1])
    ax.set_xticks([])
    ax.set_yticks([])

    for mission, center, text_center, regions in zip(
        missions,
        xy + 0.5 * widths,
        xy + np.column_stack((0.5 * widths[:, 0], np.zeros(len(widths)))),
        fovs,
    ):
        transform = Affine2D().translate(*(center)) + ax.transData
        ax.text(*text_center, mission.name, ha="center", va="top")
        for region in regions:
            ax.add_patch(
                region.as_artist(
                    facecolor="lightgray", edgecolor="black", fill=True, transform=transform
                )
            )

.. automodapi:: m4opt.missions
    :include-all-objects:
