import pytest

from ..console import demo, quiet


def test_console(capsys, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="Failed"):
        demo()
    assert "Task I" in capsys.readouterr().out

    with pytest.raises(RuntimeError, match="Failed"), quiet():
        demo()
    assert "Task I" not in capsys.readouterr().out
