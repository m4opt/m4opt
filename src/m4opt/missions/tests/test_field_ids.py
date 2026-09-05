import numpy as np
import pytest

from .. import ultrasat, uvex, ztf


def test_ztf_field_ids_match_the_sky_grid():
    """ZTF names each of its fields, one identifier per reference pointing."""
    assert len(ztf.field_ids) == len(ztf.skygrid)


def test_ztf_field_ids_are_not_row_numbers():
    """The identifiers have gaps, so they cannot be inferred from position."""
    ids = np.asarray(ztf.field_ids)
    assert ids.min() == 1
    assert ids.max() > len(ids)
    assert not np.array_equal(ids, np.arange(1, len(ids) + 1))


@pytest.mark.parametrize("mission", [uvex, ultrasat])
def test_missions_without_named_fields(mission):
    """A mission that generates its grid does not name the fields."""
    assert mission.field_ids is None
