import numpy as np
import pytest
from astropy import units as u

from .. import ultrasat, uvex, ztf


@pytest.mark.parametrize("mission", [ztf, uvex, ultrasat])
def test_skygrid_spans_the_sky(mission):
    """Reference pointings cover the full range of right ascension."""
    grids = (
        mission.skygrid.values()
        if isinstance(mission.skygrid, dict)
        else [mission.skygrid]
    )
    for grid in grids:
        ra = grid.ra.wrap_at(360 * u.deg).deg
        assert ra.min() < 10, "right ascension must reach the start of the sky"
        assert ra.max() > 350, "right ascension must reach the end of the sky"
        assert len(np.unique(np.round(ra))) > 100, "pointings must be spread in R.A."
