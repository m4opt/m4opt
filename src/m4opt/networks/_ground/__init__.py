from .._core import Network
from ...missions import ztf, rubin

ground = Network(
    name="ground",
    telescopes=[ztf, rubin],
)
ground.__doc__ = r"""
Example case of the `Network` class with 2 ground missions: ZTF (Zwicky Transient Facility) and Rubin (Vera C Rubin Observatory. The Legacy Survey of Space and Time (LSST)).
"""