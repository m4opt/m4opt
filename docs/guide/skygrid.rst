***************************************
Sky Grid Tessellation (`m4opt.skygrid`)
***************************************

Methods for tessellating the sky into survey tiles.

The functions in this module provide a variety of different methods of
selecting points on the unit sphere with approximately uniform density per unit
area. All of thee functions take one required argument, ``area``, which is the
average area per tile. Some (like :meth:`~m4opt.skygrid.geodesic`) take
additional optional keyword arguments.

Note that in the case of :meth:`~m4opt.skygrid.geodesic` and
:meth:`~m4opt.skygrid.healpix`, the number of tiles that may be returned is
constrained to certain values. For these methods, the number of tiles will be
the smallest possible number that is greater than or equal to 4π/area.

Examples
--------
>>> from astropy import units as u
>>> from m4opt import skygrid
>>> points = skygrid.sinusoidal(100 * u.deg**2)

Gallery
-------
.. plot::
    :include-source: False

    import numpy as np
    from astropy import units as u
    import ligo.skymap.plot
    from matplotlib import pyplot as plt
    from m4opt import skygrid

    areas = np.asarray([1000, 500, 100, 50]) * u.deg**2
    methods = [
        skygrid.geodesic,
        skygrid.golden_angle_spiral,
        skygrid.healpix,
        skygrid.sinusoidal,
    ]

    fig, axs = plt.subplots(
        len(methods),
        len(areas),
        figsize=(8, 6),
        tight_layout=True,
        gridspec_kw=dict(wspace=0.1, hspace=0.1),
        subplot_kw=dict(projection="astro globe", center="0d 25d"),
    )

    for method, ax in zip(methods, axs[:, 0]):
        ax.text(
            -0.2,
            0.5,
            method.__name__,
            rotation=90,
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    for area, ax in zip(areas, axs[0, :]):
        ax.text(
            0.5,
            1.2,
            area.to_string(format="latex"),
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    for method, axrow in zip(methods, axs):
        for area, ax in zip(areas, axrow):
            for coord in ax.coords:
                coord.set_ticklabel_visible(False)
                coord.set_ticks_visible(False)
            ax.plot_coord(method(area), ".")
            ax.grid()

.. automodapi:: m4opt.skygrid
