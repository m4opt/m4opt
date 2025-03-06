"""Miscellaneous optimization utilities."""

import numpy as np

from ..milp import Model

__all__ = ("solve_tsp",)


def solve_tsp(distances: np.ndarray) -> np.ndarray:
    """Solve the Traveling Salesman problem.

    Parameters
    ----------
    distances
        A square matrix of size (2, 2) or greater representing the distances
        between each pair of nodes.

    Returns
    -------
    indices
        The indices that place the nodes in the order to visit, of length 1
        greater than the rank of the distance matrix. The first and last value
        must be equal.

    Notes
    -----
    This uses the `Miller-Tucker-Zemlin formulation
    <https://en.wikipedia.org/wiki/Travelling_salesman_problem#Miller–Tucker–Zemlin_formulation>`_.
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
    return result
