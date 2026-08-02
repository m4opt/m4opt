from runpy import run_module

import pytest

from .. import console


def test_console(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="Failed"):
        run_module(console.__name__, run_name="__main__")
