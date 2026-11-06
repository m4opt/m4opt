import io
import json
import os
import time
from urllib.error import HTTPError

import pytest

from .. import _tle
from .._tle import TleObserverLocation

FIELDS = {
    "OBJECT_NAME": "FGRST (GLAST)",
    "OBJECT_ID": "2008-029A",
    "EPOCH": "2026-11-06T08:26:43.148256",
    "MEAN_MOTION": 15.24593003,
    "ECCENTRICITY": 0.00078321,
    "INCLINATION": 25.5829,
    "RA_OF_ASC_NODE": 296.8732,
    "ARG_OF_PERICENTER": 201.8333,
    "MEAN_ANOMALY": 158.1873,
    "EPHEMERIS_TYPE": 0,
    "CLASSIFICATION_TYPE": "U",
    "NORAD_CAT_ID": 33053,
    "ELEMENT_SET_NO": 999,
    "REV_AT_EPOCH": 1740,
    "BSTAR": 0.00011686289,
    "MEAN_MOTION_DOT": 3.306e-5,
    "MEAN_MOTION_DDOT": 0,
}


@pytest.fixture
def celestrak(monkeypatch, tmp_path):
    """Replace Celestrak with a fake server and use an empty cache."""
    requests = []
    responses = []

    def urlopen(url, timeout):
        requests.append(url)
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return io.BytesIO(json.dumps([response]).encode())

    monkeypatch.setattr(_tle, "urlopen", urlopen)
    monkeypatch.setattr(_tle, "get_cache_dir_path", lambda: tmp_path)
    return requests, responses, tmp_path / "m4opt" / "celestrak" / "33053.json"


def http_error(code):
    return HTTPError("https://celestrak.org", code, "error", {}, None)


def test_from_id_downloads_omm(celestrak):
    requests, responses, path = celestrak
    responses.append(FIELDS)
    TleObserverLocation.from_id(33053)
    assert requests == [
        "https://celestrak.org/NORAD/elements/gp.php?CATNR=33053&FORMAT=json"
    ]
    assert json.loads(path.read_text()) == [FIELDS]


def test_from_id_reuses_fresh_cache(celestrak):
    requests, responses, _ = celestrak
    responses.append(FIELDS)
    TleObserverLocation.from_id(33053)
    TleObserverLocation.from_id(33053)
    assert len(requests) == 1


def test_from_id_refreshes_stale_cache(celestrak):
    requests, responses, path = celestrak
    responses.extend([FIELDS, FIELDS])
    TleObserverLocation.from_id(33053)
    stale = time.time() - _tle._CELESTRAK_UPDATE_INTERVAL - 1
    os.utime(path, (stale, stale))
    TleObserverLocation.from_id(33053)
    assert len(requests) == 2


def test_from_id_falls_back_to_cache_on_error(celestrak):
    requests, responses, path = celestrak
    responses.extend([FIELDS, http_error(403)])
    TleObserverLocation.from_id(33053)
    stale = time.time() - _tle._CELESTRAK_UPDATE_INTERVAL - 1
    os.utime(path, (stale, stale))
    with pytest.warns(UserWarning, match="Using cached data"):
        TleObserverLocation.from_id(33053)

    # The failure resets the clock, so there is no immediate retry.
    TleObserverLocation.from_id(33053)
    assert len(requests) == 2


def test_from_id_raises_without_cache(celestrak):
    _, responses, path = celestrak
    responses.append(http_error(404))
    with pytest.raises(RuntimeError, match="NORAD ID 33053"):
        TleObserverLocation.from_id(33053)
    assert not path.exists()
