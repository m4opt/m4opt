"""Miscellaneous optimization utilities."""

from operator import itemgetter

import networkx as nx
import numpy as np
import pymetis

from ..milp import Model

__all__ = (
    "pack_boxes",
    "partition_graph",
    "partition_graph_milp",
    "partition_graph_milp2",
    "partition_graph_milp_recursive",
    "partition_graph_color",
    "solve_tsp",
)


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
    n, m = wh.shape
    if n == 0 or m == 0:
        return np.zeros_like(wh), np.zeros(wh.shape[1])
    with Model(**kwargs) as model:
        xy = model.continuous_vars(wh.shape, lb=0.5 * wh)
        if n > 1:
            i, j = np.triu_indices(n, 1)
            avoid = model.binary_vars((len(i), m))
            model.add_constraints_(
                model.abs(xy[i] - xy[j]) >= 0.5 * (wh[i] + wh[j]) * avoid
            )
            model.add_constraints_(
                model.sum_vars_all_different(col) >= 1 for col in avoid
            )
        model.minimize(model.sum([model.max(*row).item() for row in (xy + 0.5 * wh).T]))
        xy_result = model.solve().get_values(xy)
    return xy_result - 0.5 * wh, np.max(xy_result + 0.5 * wh, axis=0)


def partition_graph(
    graph: nx.Graph,
    n: int,
    recursive: bool | None = None,
    node_weight: str | None = None,
    edge_weight: str = "weight",
    **kwargs,
) -> np.ndarray:
    """Partition a graph into contiguous subgraphs.

    Partition a graph into subgraphs using
    `METIS <https://github.com/KarypisLab/METIS>`_.
    :footcite:`metis1,metis2,metis3`

    Parameters
    ----------
    graph
        A graph in the form of a :class:`networkx.Graph` object.
    n
        The desired number of partitions. The returned number of partitions may
        be smaller.
    recursive
        Whether to use recursive or K-way partitioning.
    node_weight
        Optional key for node weights.
    edge_weight
        Optional key for edge weights.
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
        :caption: Basic example of graph partitioning.

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

    .. plot::
        :caption: Graph partitioning with node weights (larger weights toward center).

        from matplotlib import pyplot as plt
        from m4opt.utils.optimization import partition_graph
        import networkx as nx
        import numpy as np

        graph = nx.triangular_lattice_graph(30, 50)
        center = np.mean(list(nx.get_node_attributes(graph, "pos").values()), axis=0)
        for node, data in graph.nodes(data=True):
            data["distance"] = np.ceil(np.sqrt(np.sum(np.square(node - center))) ** 3).astype(
                np.intp
            )

        part = partition_graph(graph, 50, seed=42, node_weight="distance")
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
    sparse = nx.to_scipy_sparse_array(graph, weight=edge_weight)
    _, result = pymetis.part_graph(
        nparts=n,
        adjacency=None,
        adjncy=sparse.indices,
        xadj=sparse.indptr,
        vweights=None
        if node_weight is None
        else [data for _, data in graph.nodes(data=node_weight)],
        eweights=sparse.data.astype(np.intp),
        options=pymetis.Options(**kwargs),
        recursive=recursive,
    )
    return np.asarray(result)


def partition_graph_milp(
    graph: nx.Graph,
    n: int,
    **kwargs,
) -> np.ndarray:
    """Partition a graph into contiguous subgraphs.

    Partition a graph into subgraphs using the MILP flow formulation from
    Section 4 of :footcite:`2019arXiv191105723M`.

    Parameters
    ----------
    graph
        A graph in the form of a :class:`networkx.Graph` object, an adjacency
        matrix, or an edge weight matrix.
    n
        The desired number of partitions. The returned number of partitions may
        be smaller.
    kwargs
        Additional arguments passed to :class:`m4opt.milp.Model`.

    Returns
    -------
    :
        Partition assignments for all nodes.

    References
    ----------
    .. footbibliography::

    Example
    -------
    .. plot::

        from matplotlib import pyplot as plt
        from m4opt.utils.optimization import partition_graph_milp
        import networkx as nx

        graph = nx.triangular_lattice_graph(10, 20)
        part = partition_graph_milp(graph, 5)
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
    digraph = graph.to_directed()
    sentinel = object()
    flow_source_nodes = [(sentinel, i) for i in range(n)]
    digraph.add_edges_from(
        (flow_source_node, node)
        for node in graph.nodes
        for flow_source_node in flow_source_nodes
    )

    with Model(**kwargs) as m:
        flows = m.integer_vars(
            digraph.number_of_edges(), lb=0, ub=graph.number_of_nodes()
        )
        has_flows = m.binary_vars(digraph.number_of_edges())
        for (_, _, data), flow, has_flow in zip(digraph.edges.data(), flows, has_flows):
            data["flow"] = flow
            data["has_flow"] = has_flow

        m.maximize(
            m.sum_vars_all_different(
                data["flow"]
                for _, _, data in digraph.out_edges(flow_source_nodes[0], data=True)
            )
        )

        # Eq. (7)
        m.add_constraints_(
            m.sum_vars_all_different(
                data["flow"]
                for _, _, data in digraph.out_edges(flow_source_nodes[i], data=True)
            )
            <= m.sum_vars_all_different(
                data["flow"]
                for _, _, data in digraph.out_edges(flow_source_nodes[i + 1], data=True)
            )
            for i in range(n - 1)
        )

        # Eq. (8)
        m.add_constraints_(
            m.sum_vars_all_different(
                data["flow"] for _, _, data in digraph.in_edges(node, data=True)
            )
            - m.sum_vars_all_different(
                data["flow"] for _, _, data in digraph.out_edges(node, data=True)
            )
            == 1
            for node in graph.nodes
        )

        # Eq. (9)
        M = graph.number_of_edges()
        m.add_constraints_(flow <= M * on for flow, on in zip(flows, has_flows))

        # Eq. (10)
        m.add_constraints_(
            m.sum_vars_all_different(
                data["has_flow"] for _, _, data in digraph.out_edges(node, data=True)
            )
            <= 1
            for node in flow_source_nodes
        )

        # Eq. (11)
        m.add_constraints_(
            m.sum_vars_all_different(
                data["has_flow"] for _, _, data in digraph.in_edges(node, data=True)
            )
            <= 1
            for node in graph.nodes
        )

        solution = m.solve()

    edges, flows = list(
        zip(*(((v1, v2), data["flow"]) for v1, v2, data in digraph.edges(data=True)))
    )
    flows = np.rint(solution.get_values(flows)).astype(bool)
    digraph.remove_edges_from(edge for edge, flow in zip(edges, flows) if not flow)
    digraph.remove_nodes_from(flow_source_nodes)
    graph = nx.convert_node_labels_to_integers(digraph.to_undirected(as_view=True))
    result = np.empty(graph.number_of_nodes(), dtype=np.intp)
    for component, nodes in enumerate(nx.connected_components(graph)):
        result[list(nodes)] = component
    return result


def partition_graph_milp2(
    graph: nx.Graph,
    n: int,
    **kwargs,
) -> np.ndarray:
    """Partition a graph into contiguous subgraphs.

    Partition a graph into subgraphs using the MILP flow formulation from
    Section 4 of :footcite:`2019arXiv191105723M`, but with the Eqs (10-11)
    replaced with SOS1 constraints.

    Parameters
    ----------
    graph
        A graph in the form of a :class:`networkx.Graph` object, an adjacency
        matrix, or an edge weight matrix.
    n
        The desired number of partitions. The returned number of partitions may
        be smaller.
    kwargs
        Additional arguments passed to :class:`m4opt.milp.Model`.

    Returns
    -------
    :
        Partition assignments for all nodes.

    References
    ----------
    .. footbibliography::

    Example
    -------
    .. plot::

        from matplotlib import pyplot as plt
        from m4opt.utils.optimization import partition_graph_milp2
        import networkx as nx

        graph = nx.triangular_lattice_graph(10, 20)
        part = partition_graph_milp2(graph, 5)
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
    digraph = graph.to_directed()
    sentinel = object()
    flow_source_nodes = [(sentinel, i) for i in range(n)]
    digraph.add_edges_from(
        (flow_source_node, node)
        for node in graph.nodes
        for flow_source_node in flow_source_nodes
    )

    with Model(**kwargs) as m:
        flows = m.integer_vars(
            digraph.number_of_edges(), lb=0, ub=graph.number_of_nodes()
        )
        for (_, _, data), flow in zip(digraph.edges.data(), flows):
            data["flow"] = flow

        m.maximize(
            m.sum_vars_all_different(
                data["flow"]
                for _, _, data in digraph.out_edges(flow_source_nodes[0], data=True)
            )
        )

        # Eq. (7)
        m.add_constraints_(
            m.sum_vars_all_different(
                data["flow"]
                for _, _, data in digraph.out_edges(flow_source_nodes[i], data=True)
            )
            <= m.sum_vars_all_different(
                data["flow"]
                for _, _, data in digraph.out_edges(flow_source_nodes[i + 1], data=True)
            )
            for i in range(n - 1)
        )

        # Eq. (8)
        m.add_constraints_(
            m.sum_vars_all_different(
                data["flow"] for _, _, data in digraph.in_edges(node, data=True)
            )
            - m.sum_vars_all_different(
                data["flow"] for _, _, data in digraph.out_edges(node, data=True)
            )
            == 1
            for node in graph.nodes
        )

        # Eq. (10)
        for node in flow_source_nodes:
            m.add_sos1(
                [data["flow"] for _, _, data in digraph.out_edges(node, data=True)]
            )

        # Eq. (11)
        for node in graph.nodes:
            m.add_sos1(
                [data["flow"] for _, _, data in digraph.in_edges(node, data=True)]
            )

        solution = m.solve()

    edges, flows = list(
        zip(*(((v1, v2), data["flow"]) for v1, v2, data in digraph.edges(data=True)))
    )
    flows = np.rint(solution.get_values(flows)).astype(bool)
    digraph.remove_edges_from(edge for edge, flow in zip(edges, flows) if not flow)
    digraph.remove_nodes_from(flow_source_nodes)
    graph = nx.convert_node_labels_to_integers(digraph.to_undirected(as_view=True))
    result = np.empty(graph.number_of_nodes(), dtype=np.intp)
    for component, nodes in enumerate(nx.connected_components(graph)):
        result[list(nodes)] = component
    return result


def _ensure_connected_bisection(graph, group0, group1):
    """Repair a bisection so that each group is connected.

    If either group has disconnected components, reassign the smaller
    disconnected components to the other group (if they are adjacent to it).
    """
    for _ in range(50):  # iterate until stable
        changed = False
        for groups in [(group0, group1), (group1, group0)]:
            this, other = groups
            sub = graph.subgraph(this)
            components = list(nx.connected_components(sub))
            if len(components) <= 1:
                continue
            # Keep the largest component, try to move others
            components.sort(key=len, reverse=True)
            for comp in components[1:]:
                # Check if this component is adjacent to the other group
                adjacent = any(
                    neighbor in other
                    for node in comp
                    for neighbor in graph.neighbors(node)
                )
                if adjacent:
                    this -= comp
                    other |= comp
                    changed = True
        if not changed:
            break
    return group0, group1


def _bisect_graph_heuristic(graph):
    """Bisect a graph using Kernighan-Lin with connectivity repair.

    Fast heuristic for large subgraphs where MILP is too slow.
    """
    group0, group1 = nx.community.kernighan_lin_bisection(graph)
    group0, group1 = set(group0), set(group1)
    return _ensure_connected_bisection(graph, group0, group1)


def _bisect_graph_milp(graph, **kwargs):
    """Bisect a graph into 2 balanced connected parts using the MILP flow
    formulation with SOS1 constraints.

    Parameters
    ----------
    graph
        A :class:`networkx.Graph` to bisect.
    kwargs
        Additional arguments passed to :class:`m4opt.milp.Model`.

    Returns
    -------
    group0, group1
        Two sets of node labels.
    """
    digraph = graph.to_directed()
    sentinel = object()
    sources = [(sentinel, 0), (sentinel, 1)]
    digraph.add_edges_from((src, node) for node in graph.nodes for src in sources)

    with Model(**kwargs) as m:
        flows = m.integer_vars(
            digraph.number_of_edges(), lb=0, ub=graph.number_of_nodes()
        )
        for (_, _, data), flow in zip(digraph.edges.data(), flows):
            data["flow"] = flow

        m.maximize(
            m.sum_vars_all_different(
                data["flow"] for _, _, data in digraph.out_edges(sources[0], data=True)
            )
        )

        # Eq. (7): ordering constraint (single constraint for k=2)
        m.add_constraints_(
            m.sum_vars_all_different(
                data["flow"] for _, _, data in digraph.out_edges(sources[i], data=True)
            )
            <= m.sum_vars_all_different(
                data["flow"]
                for _, _, data in digraph.out_edges(sources[i + 1], data=True)
            )
            for i in range(1)
        )

        # Eq. (8): flow conservation
        m.add_constraints_(
            m.sum_vars_all_different(
                data["flow"] for _, _, data in digraph.in_edges(node, data=True)
            )
            - m.sum_vars_all_different(
                data["flow"] for _, _, data in digraph.out_edges(node, data=True)
            )
            == 1
            for node in graph.nodes
        )

        # SOS1 constraints (replacing Eqs 10-11)
        for node in sources:
            m.add_sos1(
                [data["flow"] for _, _, data in digraph.out_edges(node, data=True)]
            )
        for node in graph.nodes:
            m.add_sos1(
                [data["flow"] for _, _, data in digraph.in_edges(node, data=True)]
            )

        solution = m.solve()

    edges, flows = list(
        zip(*(((v1, v2), data["flow"]) for v1, v2, data in digraph.edges(data=True)))
    )
    flows = np.rint(solution.get_values(flows)).astype(bool)
    digraph.remove_edges_from(edge for edge, flow in zip(edges, flows) if not flow)
    digraph.remove_nodes_from(sources)

    components = list(nx.connected_components(digraph.to_undirected(as_view=True)))

    if len(components) >= 2:
        # Sort by size; return the largest and merge the rest
        components.sort(key=len, reverse=True)
        group0 = components[0]
        group1 = set()
        for c in components[1:]:
            group1 |= c
        return group0, group1
    else:
        # Bisection failed; split arbitrarily
        return _bisect_graph_heuristic(graph)


def _bisect_graph(graph, milp_max_nodes=500, milp_timelimit=30, **kwargs):
    """Bisect a graph into 2 balanced connected parts.

    For small subgraphs (<= milp_max_nodes), uses the MILP flow formulation.
    For larger subgraphs, uses the Kernighan-Lin heuristic with connectivity
    repair.

    Parameters
    ----------
    graph
        A :class:`networkx.Graph` to bisect.
    milp_max_nodes
        Maximum number of nodes for which to use the MILP formulation.
        Larger graphs use the Kernighan-Lin heuristic.
    milp_timelimit
        Time limit in seconds for each MILP bisection.
    kwargs
        Additional arguments passed to :class:`m4opt.milp.Model`.

    Returns
    -------
    group0, group1
        Two sets of node labels.
    """
    import astropy.units as u

    if graph.number_of_nodes() > milp_max_nodes:
        return _bisect_graph_heuristic(graph)

    try:
        return _bisect_graph_milp(graph, timelimit=milp_timelimit * u.s, **kwargs)
    except Exception:
        return _bisect_graph_heuristic(graph)


def partition_graph_milp_recursive(
    graph: nx.Graph,
    n: int,
    **kwargs,
) -> np.ndarray:
    """Partition a graph into contiguous subgraphs using recursive bisection.

    At each level, the graph is bisected into two balanced connected parts
    using the MILP flow formulation from Section 4 of
    :footcite:`2019arXiv191105723M` with SOS1 constraints and k=2.
    The number of target partitions is split proportionally to the sizes of
    the two halves. This reduces the problem from a single k=240 MILP
    (~1.3M variables) to ~480 bisection MILPs of decreasing size.

    Parameters
    ----------
    graph
        A graph in the form of a :class:`networkx.Graph` object.
    n
        The desired number of partitions.
    kwargs
        Additional arguments passed to :class:`m4opt.milp.Model`.

    Returns
    -------
    :
        Partition assignments for all nodes.

    References
    ----------
    .. footbibliography::

    Example
    -------
    .. plot::

        from matplotlib import pyplot as plt
        from m4opt.utils.optimization import partition_graph_milp_recursive
        import networkx as nx

        graph = nx.triangular_lattice_graph(10, 20)
        part = partition_graph_milp_recursive(graph, 5)
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
    graph = nx.convert_node_labels_to_integers(graph)
    num_nodes = graph.number_of_nodes()
    result = np.empty(num_nodes, dtype=np.intp)
    next_label = [0]

    def _recurse(nodes, k):
        if k <= 1 or len(nodes) <= 1:
            label = next_label[0]
            next_label[0] += 1
            for node in nodes:
                result[node] = label
            return

        if len(nodes) <= k:
            # More partitions requested than nodes; each node is its own
            for node in nodes:
                result[node] = next_label[0]
                next_label[0] += 1
            return

        subgraph = graph.subgraph(nodes)
        group0, group1 = _bisect_graph(subgraph, **kwargs)

        # Split k proportionally to group sizes
        total = len(group0) + len(group1)
        k0 = max(1, min(k - 1, round(k * len(group0) / total)))
        k1 = k - k0

        _recurse(list(group0), k0)
        _recurse(list(group1), k1)

    _recurse(list(range(num_nodes)), n)
    return result


def partition_graph_color(
    graph: nx.Graph, partition: np.ndarray, **kwargs
) -> np.ndarray:
    """Find a coloring for a partition of a graph.

    Parameters
    ----------
    graph
        A graph in the form of a :class:`networkx.Graph` object.
    partition
        Partition assignments of the nodes in the graphs as returned by
        :meth:`partition_graph`.
    **kwargs
        Any additional arguments to pass to
        :obj:`networkx.algorithms.coloring.greedy_color`.

    Returns
    -------
    :
        An integer-valued array of color assignments for each partition.
        The color for node `i` in the original graph is `color[partition[i]]`.

    Example
    -------
    .. plot::

        from matplotlib import pyplot as plt
        from m4opt.utils.optimization import partition_graph, partition_graph_color
        import networkx as nx

        graph = nx.triangular_lattice_graph(20, 40)
        part = partition_graph(graph, 20, seed=42)
        color = partition_graph_color(
            graph, part, strategy="connected_sequential", interchange=True)
        ax = plt.axes(aspect=1)
        nx.draw(
            graph,
            ax=ax,
            pos=nx.get_node_attributes(graph, "pos"),
            node_size=50,
            node_color=color[part],
            cmap="cool",
        )
    """
    if isinstance(graph, nx.Graph):
        adjacency = nx.to_numpy_array(graph)
    else:
        adjacency = graph

    n_partitions = partition.max() + 1
    partition_adjacency = np.zeros((n_partitions, n_partitions), dtype=bool)
    for i in range(n_partitions):
        for j in range(n_partitions):
            if i != j:
                partition_adjacency[i, j] = np.any(
                    adjacency[np.ix_(partition == i, partition == j)]
                )
    partition_graph = nx.from_numpy_array(partition_adjacency)

    return np.asarray(
        list(
            map(
                itemgetter(1),
                sorted(
                    nx.algorithms.coloring.greedy_color(
                        partition_graph, **kwargs
                    ).items()
                ),
            )
        )
    )


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
