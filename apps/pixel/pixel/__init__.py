"""kolm pixel: verifiable image-quality claims and the data pipeline behind them.

Three things live here and nothing else:

1. A physical floor (``camera.BestCaseCamera``) and ceiling (``display.Display``)
   so a claim like "better than any lens" has a measurable meaning.
2. A metric suite and gate (``metrics``, ``gate``) that turn a candidate image or
   video plus a procedural ground truth into a PASS/FAIL report.
3. A data pipeline (``data``, ``degrade``, ``label``, ``train``) that scales by
   construction: sources are stored once, every degraded training pair is a pure
   function of (source, seed), and every label is derived, never hand-made.
"""

__version__ = "0.1.0"
