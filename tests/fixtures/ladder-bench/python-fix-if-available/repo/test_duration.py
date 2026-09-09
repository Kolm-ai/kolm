"""Tests for the sub-minute behaviour of seconds_to_label."""

from duration import seconds_to_label


def test_sub_minute():
    assert seconds_to_label(0) == "0s"
    assert seconds_to_label(45) == "45s"


def test_negative_is_zero():
    assert seconds_to_label(-1) == "0s"
