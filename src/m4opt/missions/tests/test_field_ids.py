from importlib import resources

import numpy as np
import pytest
from astropy.table import Table

from .. import ultrasat, uvex, ztf
from .._ztf import data


def test_the_ztf_grid_is_numbered_by_field_id():
    """ZTF names its fields, so its grid is indexed by those names."""
    assert len(ztf.skygrid) == 1898
    assert np.sum(~ztf.skygrid.ra.mask) == 1778


@pytest.mark.parametrize("field_id", [0, 882, 1000])
def test_a_field_ztf_does_not_use_is_masked(field_id):
    """The numbering has one gap, over 882 to 1000, and does not use zero."""
    assert ztf.skygrid[field_id].ra.mask


@pytest.mark.parametrize("field_id", [1, 881, 1001, 1897])
def test_a_field_ztf_uses_is_not_masked(field_id):
    assert not ztf.skygrid[field_id].ra.mask


def test_the_ztf_grid_matches_the_field_list():
    """Each coordinate sits at the row its own identifier names."""
    table = Table.read(
        resources.files(data) / "ZTF_Fields.txt",
        format="ascii.no_header",
        comment="%",
    )
    grid = ztf.skygrid[np.asarray(table["col1"])]
    np.testing.assert_allclose(grid.ra.unmasked.deg, np.asarray(table["col2"]))
    np.testing.assert_allclose(grid.dec.unmasked.deg, np.asarray(table["col3"]))


@pytest.mark.parametrize("mission", [uvex, ultrasat])
def test_a_generated_grid_has_no_gaps(mission):
    """A mission that generates its grid numbers the rows consecutively."""
    grids = mission.skygrid
    for grid in grids.values() if isinstance(grids, dict) else [grids]:
        assert not hasattr(grid.ra, "mask")
