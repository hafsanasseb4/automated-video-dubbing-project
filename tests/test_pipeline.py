import pytest
from app.main import Segment, main

def test_segment_duration_is_safe_for_empty_windows():
    assert Segment(3, 3, "hello").duration == 0.05
    assert Segment(1, 4, "hello").duration == 3

def test_cli_help_is_available(capsys):
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
    assert "English dubbed" in capsys.readouterr().out
