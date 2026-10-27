"""Metric suite. Every metric is ``f(candidate, reference, ...) -> float`` with a
declared better-direction in ``suite.METRICS`` so gates and reports never guess.
"""

from .suite import METRICS, GROUPS, evaluate_sequence, evaluate_still  # noqa: F401
from .generic import generic_metrics  # noqa: F401
