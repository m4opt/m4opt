import networkx as nx
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ..optimization import partition_graph, solve_tsp


def test_partition_graph():
    n = 5
    kwargs = dict(n=n, seed=42)
    graph = nx.triangular_lattice_graph(10, 20)

    # Test that all three graph encodings return the same partition.
    part1 = partition_graph(graph, **kwargs)
    part2 = partition_graph(nx.to_scipy_sparse_array(graph), **kwargs)
    part3 = partition_graph(nx.to_numpy_array(graph), **kwargs)
    np.testing.assert_array_equal(part1, part2)
    np.testing.assert_array_equal(part2, part3)

    # Test that if seed argument is missing, we still get a valid result.
    assert len(np.unique(partition_graph(graph, n))) <= n


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
