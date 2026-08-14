import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra import numpy as xp

from ..numpy import count_intersect1d

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
def test_benchmark(implementation, benchmark):
    benchmark.pedantic(implementation, setup=setup, rounds=100, warmup_rounds=1)
