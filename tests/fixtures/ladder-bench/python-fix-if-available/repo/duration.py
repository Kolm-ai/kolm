"""Duration helpers for the report renderer."""


def seconds_to_label(seconds):
    """Return a short label for a whole number of seconds."""
    if seconds is None or seconds < 0:
        return "0s"
    if seconds < 60:
        return "%ds" % seconds
    minutes = seconds // 60
    return "%dm" % minutes
