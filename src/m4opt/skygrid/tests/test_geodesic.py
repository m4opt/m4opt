import pytest
from astropy import units as u
from hypothesis import given
from hypothesis.strategies import integers

from ...skygrid._geodesic import BASE_COUNT, for_subdivision, geodesic, num_vertices


@given(b=integers(min_value=1, max_value=10), c=integers(min_value=1, max_value=10))
@pytest.mark.parametrize("base", BASE_COUNT.keys())
def test_num_vertices(b, c, base):
    assert len(for_subdivision(b, c, base)) == num_vertices(b, c, base)


def test_invalid_geodesic_polyhedron():
    with pytest.raises(ValueError):
        # No such thing as a class IV geodesic polyhedron
        geodesic(10 * u.deg**2, class_="IV")
