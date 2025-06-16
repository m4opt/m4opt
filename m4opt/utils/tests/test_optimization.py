import networkx as nx
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ..optimization import pack_boxes, partition_graph, partition_graph_color, solve_tsp


@settings(deadline=None)
@given(n=st.integers(0, 4), m=st.integers(0, 3))
def test_pack_boxes_random(n, m):
    """Test packing random boxes in multiple dimensions."""
    wh = np.random.uniform(size=(n, m))
    eps = np.finfo(wh.dtype).eps
    lower, _ = pack_boxes(wh)
    upper = lower + wh
    i, j = np.triu_indices(n, 1)
    assert m == 0 or np.all(
        np.any((upper[j] <= lower[i] + eps) | (upper[i] <= lower[j] + eps), axis=1)
    )


@settings(deadline=None)
@given(n=st.integers(0, 2))
def test_pack_boxes_perfect_square(n):
    """Test packing a perfect square number of unit boxes in 2 dimensions."""
    wh = np.ones((n**2, 2))
    eps = np.finfo(wh.dtype).eps
    lower, total_wh = pack_boxes(wh)
    upper = lower + wh
    i, j = np.triu_indices(n, 1)
    assert np.all(
        np.any((upper[j] <= lower[i] + eps) | (upper[i] <= lower[j] + eps), axis=1)
    )
    np.testing.assert_array_equal(total_wh, [n, n])


def test_partition_graph():
    n = 5
    kwargs = dict(n=n, seed=42)
    graph = nx.triangular_lattice_graph(10, 20)
    sparse_adjacency = nx.to_scipy_sparse_array(graph)
    dense_adjacency = nx.to_numpy_array(graph)

    # Test that all three graph encodings return the same partition.
    part1 = partition_graph(graph, **kwargs)
    part2 = partition_graph(sparse_adjacency, **kwargs)
    part3 = partition_graph(dense_adjacency, **kwargs)
    np.testing.assert_array_equal(part1, part2)
    np.testing.assert_array_equal(part2, part3)

    # Test that if seed argument is missing, we still get a valid result.
    assert len(np.unique(partition_graph(graph, n))) <= n


def test_partition_graph_color():
    n = 5
    kwargs = dict(n=n, seed=42)
    graph = nx.convert_node_labels_to_integers(nx.triangular_lattice_graph(10, 20))
    adjacency = nx.to_numpy_array(graph)

    part = partition_graph(graph, **kwargs)

    # Test that all three graph encodings return the same coloring.
    colors1 = partition_graph_color(graph, part)
    colors2 = partition_graph_color(adjacency, part)
    np.testing.assert_array_equal(colors1, colors2)

    for node1, node2 in graph.edges:
        part1 = part[node1]
        part2 = part[node2]
        if part1 != part2:
            assert colors1[part1] != colors2[part2]


def solve_tsp_approx(distances):
    """Find an approximate solution to the Traveling Salesman Problem."""
    n = len(distances)
    i, j = np.triu_indices(n, 1)
    graph = nx.Graph()
    graph.add_weighted_edges_from(zip(i, j, distances[i, j]))
    return nx.approximation.christofides(graph)


def tour_total_length(distances, sequence):
    """Find the total path length of a tour."""
    return distances[sequence[:-1], sequence[1:]].sum()


def random_distance_matrix(n: int):
    """Construct a random, symmetric distance matrix."""
    result = np.random.uniform(size=(n, n))
    result[np.diag_indices_from(result)] = 0
    return result


@settings(deadline=None)
@given(st.integers(min_value=2, max_value=30))
def test_solve_tsp(n):
    dist = random_distance_matrix(n)
    result, result_length = solve_tsp(dist)
    approx = solve_tsp_approx(dist)

    # Check that both solutions visit each node exactly once.
    for sequence in (result, approx):
        assert sequence[0] == sequence[-1], (
            "The tour must start and end at the same node"
        )
        np.testing.assert_array_equal(
            np.sort(sequence[:-1]),
            np.arange(n),
            "The tour must visit all nodes exactly once",
        )

    assert result_length == pytest.approx(tour_total_length(dist, result)), (
        "Objective value must equal tour length"
    )
    assert result_length <= tour_total_length(dist, approx), (
        "The exact solution must have a shorter path than the approximate solution"
    )
