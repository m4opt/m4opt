from itertools import combinations
from math import comb

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as xp

from ..numpy import count_intersect1d, count_intersect1d_combinations

sets1d = xp.arrays(
    dtype=np.intp,
    shape=xp.array_shapes(min_dims=1, max_dims=1, min_side=0, max_side=100),
    unique=True,
).map(np.sort)


def count_intersect1d_slow(a, b):
    return np.intersect1d(a, b).size


@given(a=sets1d, b=sets1d)
def test_count_intersect1d(a, b):
    assert count_intersect1d(a, b) == count_intersect1d_slow(a, b)


def setup_set1d():
    return np.sort(np.random.choice(1_000_000, 100_000, replace=False)).astype(
        np.intp, copy=False
    )


def setup():
    return (setup_set1d(), setup_set1d()), {}


@pytest.mark.parametrize("implementation", (count_intersect1d, count_intersect1d_slow))
def test_benchmark_count_intersect1d(implementation, benchmark):
    benchmark.pedantic(implementation, setup=setup, rounds=100, warmup_rounds=1)


def count_intersect1d_combinations_slow(a):
    return np.asarray([count_intersect1d(*args) for args in combinations(a, 2)])


@given(arrays=st.lists(sets1d, min_size=2))
def test_count_intersect1d_combinations(arrays):
    np.testing.assert_array_equal(
        count_intersect1d_combinations(arrays),
        count_intersect1d_combinations_slow(arrays),
    )


@pytest.mark.parametrize("n", [0, 1])
def test_count_intersect1d_combinations_too_few_args(n):
    with pytest.raises(
        ValueError,
        match=rf"count_intersect1d_combinations\(\) expects a sequence of at least 2 arrays \({n} given\)",
    ):
        count_intersect1d_combinations([[]] * n)


def test_count_intersect1d_combinations_very_large():
    """Test that we don't run out of stack memory for large numbers of inputs."""
    n = 2000
    np.testing.assert_array_equal(
        count_intersect1d_combinations([[]] * n), np.zeros(comb(n, 2), dtype=np.intp)
    )


def setup_combinations():
    return ([setup_set1d() for _ in range(20)],), {}


@pytest.mark.parametrize(
    "implementation",
    (count_intersect1d_combinations, count_intersect1d_combinations_slow),
)
def test_benchmark_count_intersect1d_combinations(implementation, benchmark):
    benchmark.pedantic(
        implementation, setup=setup_combinations, rounds=10, warmup_rounds=1
    )
