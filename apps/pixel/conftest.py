# Keeps the repo root on sys.path so `import harness` works under pytest without installing.

# The free-space floors (150-300 GB by default) are for the user's data drive, not a test runner: a CI box or a
# laptop with 20 GB free must still run the collector and store tests. Tests that exercise a floor set it with
# monkeypatch, which overrides these defaults.
import os

for _env in ("HARNESS_COLLECT_RESERVE_GB", "HARNESS_REALSTORE_RESERVE_GB", "HARNESS_DECODED_RESERVE_GB", "HARNESS_SIDD_RESERVE_GB"):
    os.environ.setdefault(_env, "0")
