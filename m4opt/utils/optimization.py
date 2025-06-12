"""Miscellaneous optimization utilities."""

import networkx as nx
import numpy as np
import pymetis
from scipy.sparse import csr_array

from ..milp import Model

__all__ = ("pack_boxes", "partition_graph", "solve_tsp")


def pack_boxes(wh: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Pack non-overlapping hypercubes into the smallest possible hypercube.

    Parameters
    ----------
    wh
        A Numpy array of shape `(n, m)` containing the dimensions of `n`
        hypercubes in `m` dimensions.
    kwargs
        Additional arguments passed to :class:`m4opt.milp.Model`.

    Returns
    -------
    :
        The anchor points of the rectangles, and the total dimensions.

    Examples
    --------
    .. plot::

        import numpy as np
        from matplotlib import pyplot as plt

        from m4opt.utils.optimization import pack_boxes

        rng = np.random.RandomState(seed=42)
        for n in range(1, 5):
            wh = rng.uniform(size=(n, 2))
            xy, (total_width, total_height) = pack_boxes(wh)
            fig, ax = plt.subplots(subplot_kw=dict(aspect=1))
            ax.set_xlim(0, total_width)
            ax.set_ylim(0, total_height)
            for xy_, wh_ in zip(xy, wh):
                ax.add_patch(plt.Rectangle(xy_, *wh_, edgecolor="black"))
    """
    n = len(wh)
    if n == 0:
        return np.zeros_like(wh), np.zeros(wh.shape[1])
    with Model(**kwargs) as m:
        xy = m.continuous_vars(wh.shape, lb=0.5 * wh)
        if n > 1:
            i, j = ij = np.triu_indices(n, 1)
            avoid = m.binary_vars(np.shape(ij)[::-1])
            m.add_constraints_(m.abs(xy[i] - xy[j]) >= 0.5 * (wh[i] + wh[j]) * avoid)
            m.add_constraints_(m.sum_vars_all_different(row) >= 1 for row in avoid)
        m.minimize(m.sum([m.max(*row).item() for row in (xy + 0.5 * wh).T]))
        xy_result = m.solve().get_values(xy)
    return xy_result - 0.5 * wh, np.max(xy_result + 0.5 * wh, axis=0)


def partition_graph(
    graph: np.ndarray | csr_array | nx.Graph,
    n: int,
    recursive: bool | None = None,
    **kwargs,
) -> np.ndarray:
    """Partition a graph into contiguous subgraphs.

    Partition a graph into subgraphs using
    `METIS <https://github.com/KarypisLab/METIS>`_.
    :footcite:`metis1,metis2,metis3`

    Parameters
    ----------
    graph
        A graph in the form of a :class:`networkx.Graph` object, an adjacency
        matrix, or an edge weight matrix.
    n
        The desired number of partitions. The returned number of partitions may
        be smaller.
    recursive
        Whether to use recursive or K-way partitioning.
    kwargs
        Additional arguments passed to :class:`pymetis.Options`.

    Returns
    -------
    :
        Partition assignments for all nodes.

    References
    ----------
    .. footbibliography::

    Notes
    -----
    If the graph has edge weights, then the weights must be integer-valued.

    Example
    -------
    .. plot::

        from matplotlib import pyplot as plt
        from m4opt.utils.optimization import partition_graph
        import networkx as nx

        graph = nx.triangular_lattice_graph(10, 20)
        part = partition_graph(graph, 5, seed=42)
        ax = plt.axes(aspect=1)
        nx.draw(
            graph,
            ax=ax,
            pos=nx.get_node_attributes(graph, "pos"),
            node_size=50,
            node_color=part,
            cmap="prism",
        )
    """
    if isinstance(graph, nx.Graph):
        sparse = nx.to_scipy_sparse_array(graph)
    else:
        sparse = csr_array(graph)

    _, result = pymetis.part_graph(
        nparts=n,
        adjacency=None,
        adjncy=sparse.indices,
        xadj=sparse.indptr,
        eweights=sparse.data.astype(np.intp),
        options=pymetis.Options(**kwargs),
        recursive=recursive,
    )
    return np.asarray(result)


def solve_tsp(distances: np.ndarray, **kwargs) -> tuple[np.ndarray, float]:
    """Solve the Traveling Salesman problem.

    Parameters
    ----------
    distances
        A square matrix of size (2, 2) or greater representing the distances
        between each pair of nodes.
    kwargs
        Additional arguments passed to :class:`m4opt.milp.Model`.

    Returns
    -------
    sequence
        The indices that place the nodes in the order to visit, of length 1
        greater than the rank of the distance matrix. The first and last value
        must be equal.
    length
        The total path length of the tour.

    Notes
    -----
    This uses the `Miller-Tucker-Zemlin formulation
    <https://en.wikipedia.org/wiki/Travelling_salesman_problem#Miller–Tucker–Zemlin_formulation>`_.

    Examples
    --------

    .. plot::

        from scipy.spatial.distance import pdist, squareform
        import numpy as np
        from matplotlib import pyplot as plt
        from m4opt.utils.optimization import solve_tsp

        # Construct a random cloud of points.
        points = np.random.default_rng(1234).random((30, 2))

        # Calculate the Euclidean distances between each pair of points.
        dist = squareform(pdist(points))

        # Find the shortest path, and the path length.
        indices, path_length = solve_tsp(dist)

        # Plot the points and the path.
        ax = plt.axes(aspect=1, xlim=(0, 1), ylim=(0, 1))
        ax.plot(*points[indices].T, "o-")
    """
    n = len(distances)
    assert n >= 2
    with Model(**kwargs) as m:
        x = m.binary_vars((n, n))
        y = m.integer_vars(n - 1, lb=1, ub=n - 1)
        m.add_constraints_(
            [
                m.sum_vars_all_different([x[i, j] for i in range(n) if i != j]) == 1
                for j in range(n)
            ]
        )
        m.add_constraints_(
            [
                m.sum_vars_all_different([x[i, j] for j in range(n) if i != j]) == 1
                for i in range(n)
            ]
        )
        m.add_constraints_(
            [
                y[i] - y[j] + 1 <= (n - 1) * (1 - x[i + 1, j + 1])
                for i in range(n - 1)
                for j in range(n - 1)
                if i != j
            ]
        )
        m.minimize(
            m.sum(
                [
                    distances[i, j] * x[i, j]
                    for i in range(n)
                    for j in range(n)
                    if i != j
                ]
            )
        )
        solution = m.solve()

    sequence = np.rint(solution.get_values(y)).astype(np.intp)
    result = np.empty(n + 1, dtype=np.intp)
    result[sequence] = np.arange(1, n)
    result[0] = result[-1] = 0
    return result, solution.get_objective_value()
