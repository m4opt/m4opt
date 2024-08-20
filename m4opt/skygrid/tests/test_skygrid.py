from functools import partial

import pytest
from astropy import units as u

from ... import skygrid

geodesic_methods = [
    partial(skygrid.geodesic, base=base, class_=class_)
    for base in ["icosahedron", "octahedron", "tetrahedron"]
    for class_ in ["I", "II", "III"]
]


@pytest.mark.parametrize(
    "method",
    [
        *geodesic_methods,
        skygrid.golden_angle_spiral,
        skygrid.healpix,
        skygrid.sinusoidal,
    ],
)
@pytest.mark.parametrize("area", [10, 100, 1000] * u.deg**2)
def test_skygrid(method, area):
    coords = method(area)
    assert len(coords) >= (u.spat / area)


def test_invalid_geodesic_polyhedron():
    with pytest.raises(ValueError):
        # No such thing as a class IV geodesic polyhedron
        skygrid.geodesic(10 * u.deg**2, class_="IV")
