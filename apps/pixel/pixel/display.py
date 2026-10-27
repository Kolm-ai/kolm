"""The ceiling: what a viewer can resolve on a given screen at a given distance."""

from __future__ import annotations

from dataclasses import dataclass, asdict

from .physics import pixels_per_degree

# Measured acuity thresholds for high-contrast patterns viewed directly
# (Ashraf et al. 2025, Nature Communications): mean ~94 ppd, best observers ~120.
PPD_THRESHOLD_MEAN = 94.0
PPD_THRESHOLD_BEST = 120.0


@dataclass(frozen=True)
class Display:
    diagonal_in: float = 31.5
    width_px: int = 7680
    height_px: int = 4320
    distance_m: float = 0.6

    @property
    def ppd(self) -> float:
        return pixels_per_degree(self.diagonal_in, self.width_px, self.height_px, self.distance_m)

    @property
    def resolution_headroom(self) -> float:
        """ppd relative to the best-observer threshold. >1 means pixels beyond acuity."""
        return self.ppd / PPD_THRESHOLD_BEST

    def master_size(self, factor: float = 2.0) -> tuple[int, int]:
        """Working resolution for a master that survives cropping and resampling."""
        return int(round(self.width_px * factor)), int(round(self.height_px * factor))

    def summary(self) -> dict:
        d = asdict(self)
        d.update(
            ppd=round(self.ppd, 1),
            ppd_threshold_mean=PPD_THRESHOLD_MEAN,
            ppd_threshold_best=PPD_THRESHOLD_BEST,
            resolution_headroom=round(self.resolution_headroom, 2),
            master_size=self.master_size(),
        )
        return d
