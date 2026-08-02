import numpy as np
from hypothesis import given
from hypothesis.extra import numpy as xp

from .._numpy import count_intersect1d

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
