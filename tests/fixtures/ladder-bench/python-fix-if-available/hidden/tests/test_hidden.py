"""Hidden judgement for the python bench task."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from duration import seconds_to_label  # noqa: E402


def test_minutes_and_seconds():
    assert seconds_to_label(90) == "1m 30s"
    assert seconds_to_label(120) == "2m 0s"
    assert seconds_to_label(3661) == "61m 1s"


def test_sub_minute_unchanged():
    assert seconds_to_label(0) == "0s"
    assert seconds_to_label(45) == "45s"
    assert seconds_to_label(-1) == "0s"
