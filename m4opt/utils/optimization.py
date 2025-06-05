"""Miscellaneous optimization utilities."""

import networkx as nx
import numpy as np
import pymetis
from scipy.sparse import csr_array

from ..milp import Model

__all__ = ("partition_graph", "solve_tsp")


def partition_graph(
    graph: np.ndarray | csr_array | nx.Graph, n: int, seed: int | None = None
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
    seed
        Optional random seed.

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

    # Options:
    # - contig=True: find contiguous clusters (doesn't seem to do anything; see
    #   https://github.com/inducer/pymetis/issues/60)
    # - no2hop=True: don't permit 2-hop connections in a partition
    #   (doesn't seem to do anything)
    options = pymetis.Options(contig=True, no2hop=True)
    if seed is not None:
        options.seed = seed

    _, result = pymetis.part_graph(
        nparts=n,
        adjacency=None,
        adjncy=sparse.indices,
        xadj=sparse.indptr,
        eweights=sparse.data.astype(np.intp),
        options=options,
        # Note: I get much nicer-looking output when I set recursive=True.
        # Without this option, the partitions have very ragged edges and there
        # are some non-contiguous partitions too.
        recursive=True,
    )
    return np.asarray(result)


def solve_tsp(distances: np.ndarray) -> tuple[np.ndarray, float]:
    """Solve the Traveling Salesman problem.

    Parameters
    ----------
    distances
        A square matrix of size (2, 2) or greater representing the distances
        between each pair of nodes.

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
    with Model() as m:
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
