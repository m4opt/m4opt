from math import gcd
from typing import Literal, Tuple

import numpy as np
from anti_lib_progs.geodesic import Vec, get_poly, grid_to_points, make_grid
from astropy import units as u
from astropy.coordinates import (
    CartesianRepresentation,
    SkyCoord,
    UnitSphericalRepresentation,
)


def triangulation_number(b, c):
    return b * b + b * c + c * c


def solve_number_of_vertices(n, base, class_):
    base_count = {"icosahedron": 10, "octahedron": 4, "tetrahedron": 2}[base]

    if class_ == "I":
        b = int(np.ceil(np.sqrt((n - 2) / base_count)))
        c = 0
        t = b * b
    elif class_ == "II":
        b = c = int(np.ceil(np.sqrt((n - 2) / (base_count * 3))))
        t = 3 * b * b
    elif class_ == "III":
        # FIXME: This is a brute-force search.
        # This could be solved easily by Gurobi as a non-convex MIQCQP problem,
        # or by a custom dynamic programming solution.
        b_max = int(np.ceil(np.sqrt((n - 2) / base_count)))
        t, c, b = min(
            (triangulation_number(b, c), c, b)
            for b in range(b_max + 1)
            for c in range(b_max + 1)
            if triangulation_number(b, c) * base_count + 2 >= n
        )
    else:
        raise ValueError("Unknown breakdown class")

    return base_count * t + 2, b, c


def for_subdivision(
    b: int,
    c: int,
    base: Literal["icosahedron", "octahedron", "tetrahedron"] | str,
):
    # Adapted from
    # https://github.com/antiprism/antiprism_python/blob/master/anti_lib_progs/geodesic.py
    verts: list[Vec] = []
    edges: dict[Tuple[int, int], int] = {}
    faces: list[Tuple[int, int]] = []
    get_poly(base[0], verts, edges, faces)

    reps = gcd(b, c)
    b //= reps
    c //= reps
    t = reps * triangulation_number(b, c)

    grid = make_grid(t, b, c)

    points = verts
    for face in faces:
        points.extend(
            grid_to_points(grid, t, False, [verts[face[i]] for i in range(3)], face)
        )

    coords = SkyCoord(
        *zip(*(point.v for point in points)),
        representation_type=CartesianRepresentation,
    )
    coords.representation_type = UnitSphericalRepresentation
    return coords


def geodesic(
    area: u.Quantity[u.physical.solid_angle],
    base: Literal["icosahedron", "octahedron", "tetrahedron"] | str = "icosahedron",
    class_: Literal["I", "II", "III"] | str = "I",
):
    """Generate a geodesic polyhedron with the fewest vertices >= `n`.

    Parameters
    ----------
    area
        The average area per tile in any Astropy solid angle units:
        for example, :samp:`10 * astropy.units.deg**2` or
        :samp:`0.1 * astropy.units.steradian`.
    base
        The base polyhedron of the tessellation.
    class_
        The class of the geodesic polyhedron, which constrains the allowed
        values of the number of points. Class III permits the most freedom.

    Returns
    -------
    :
        The coordinates of the vertices of the geodesic polyhedron.

    References
    ----------
    https://en.wikipedia.org/wiki/Geodesic_polyhedron

    Examples
    --------

    .. plot::
        :context: reset

        from astropy import units as u
        from matplotlib import pyplot as plt
        import ligo.skymap.plot
        import numpy as np

        from m4opt import skygrid

        n_vertices_target = 1024
        vertices = skygrid.geodesic(4 * np.pi * u.sr / n_vertices_target)
        n_vertices = len(vertices)

        ax = plt.axes(projection='astro globe', center='0d 25d')
        plt.suptitle('Class I')
        ax.set_title(f'{n_vertices} vertices (goal was {n_vertices_target})')
        ax.plot_coord(vertices, '.')
        ax.grid()

    .. plot::
        :context: close-figs

        vertices = skygrid.geodesic(4 * np.pi * u.sr / n_vertices_target,
                                    class_='II')
        n_vertices = len(vertices)

        ax = plt.axes(projection='astro globe', center='0d 25d')
        plt.suptitle('Class II')
        ax.set_title(f'{n_vertices} vertices (goal was {n_vertices_target})')
        ax.plot_coord(vertices, '.')
        ax.grid()

    .. plot::
        :context: close-figs

        vertices = skygrid.geodesic(4 * np.pi * u.sr / n_vertices_target,
                                        class_='III')
        n_vertices = len(vertices)

        ax = plt.axes(projection='astro globe', center='0d 25d')
        plt.suptitle('Class III')
        ax.set_title(f'{n_vertices} vertices (goal was {n_vertices_target})')
        ax.plot_coord(vertices, '.')
        ax.grid()

    """
    n = int(np.ceil(1 / area.to_value(u.spat)))
    n, b, c = solve_number_of_vertices(n, base, class_)
    points = for_subdivision(b, c, base=base)
    assert len(points) == n
    return points
